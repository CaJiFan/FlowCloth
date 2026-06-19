"""
Offline evaluation script for cloth state estimation models.

Define EVAL_RUNS below: each entry specifies an experiment directory and one
or more checkpoint steps to evaluate.  The script auto-detects the config
(and therefore DDPM vs Flow-Matching) from the saved config.yml inside each
experiment directory, so no manual scheduler flag is needed.

For each (exp_dir, step) pair the script:
  1. Loads experiments/<exp_dir>/checkpoints/checkpoint-<step>/model
  2. Runs full inference over the entire validation set (no augmentation).
  3. Computes: MSE, Chamfer-L1, F@1cm, F@2cm, corner error,
     Hausdorff, perimeter error, coverage area, latency (ms/sample).
  4. Logs a wandb Table + per-metric scalars to --wandb_project_eval.
  5. Prints and saves a LaTeX table (eval_results_table.tex).

Usage:
  # from the UniClothDiff/ directory:
  python scripts/eval_offline.py
  python scripts/eval_offline.py --no_wandb
  python scripts/eval_offline.py --batch_size 16 --num_workers 4
"""

import os
import sys
import time
import argparse
import importlib
import numpy as np
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from scipy.spatial import cKDTree, ConvexHull

# Make sure the repo root is on the path when called from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import wandb
from uniclothdiff.registry import build_model, build_dataset, build_scheduler
from uniclothdiff.pipelines.cloth_state_est_pipeline import ClothStateEstPipeline
from uniclothdiff.pipelines.cloth_state_est_fm_pipeline import ClothStateEstFMPipeline

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation runs — edit this table to add / remove experiments.
#
# Each entry:
#   exp_dir   : path to the experiment folder (contains config.yml and
#               checkpoints/).  Scheduler type (DDPM vs FM) and dataset mode
#               (full vs edge) are auto-detected from the saved config.yml.
#   steps     : checkpoint steps to evaluate.  Use "latest" for the most
#               recent checkpoint.
#   inf_steps : list of inference step counts to sweep for this run.
#               For FM try [1, 4, 10, 50]; for DDPM try [20, 100].
#               Omit (or set to None) to fall back to the CLI defaults
#               (--num_inference_steps_ddpm / --num_inference_steps_fm).
#   label     : base display name used in LaTeX tables and wandb.  The
#               actual row label gets "@ <inf_steps> steps" appended.
# ─────────────────────────────────────────────────────────────────────────────
EVAL_RUNS = [
    {
        "exp_dir":   "experiments/vr_diff_full",
        "steps":     [250000],
        "inf_steps": [20, 100],
        "label":     "Full / DDPM",
    },
    {
        "exp_dir":   "experiments/vr_fm_full",
        "steps":     [250000],
        "inf_steps": [1, 4, 10, 50],
        "label":     "Full / FM",
    },
    {
        "exp_dir":   "experiments/vr_diff_edge",
        "steps":     [250000],
        "inf_steps": [20, 100],
        "label":     "Edge / DDPM",
    },
    {
        "exp_dir":   "experiments/vr_fm_edge",
        "steps":     [250000],
        "inf_steps": [1, 4, 10, 50],
        "label":     "Edge / FM",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers for resolving EVAL_RUNS into concrete (config, checkpoint_path) pairs
# ─────────────────────────────────────────────────────────────────────────────

def resolve_checkpoint(exp_dir: str, step) -> str:
    """Returns the checkpoint directory path for a given step or 'latest'."""
    ckpts_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.isdir(ckpts_dir):
        raise FileNotFoundError(f"checkpoints/ not found in {exp_dir}")

    available = sorted(
        [d for d in os.listdir(ckpts_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1])
    )
    if not available:
        raise FileNotFoundError(f"No checkpoints found in {ckpts_dir}")

    if step == "latest":
        chosen = available[-1]
    else:
        # find checkpoint whose step is closest to the requested step
        avail_steps = [int(d.split("-")[1]) for d in available]
        idx = min(range(len(avail_steps)), key=lambda i: abs(avail_steps[i] - int(step)))
        chosen = available[idx]
        if avail_steps[idx] != int(step):
            print(f"  [warn] requested step {step} not found; "
                  f"using closest: {chosen}")

    return os.path.join(ckpts_dir, chosen)


def _corner_indices(dataset_type: str, is_edge: bool) -> list:
    """Return corner indices for the 4 physical corners of the cloth grid."""
    if "TRTM" in dataset_type:
        # 21×21 grid: corners at [0, 20, 420, 440]; edge contour corners [0, 20, 59, 79]
        return [0, 20, 59, 79] if is_edge else [0, 20, 420, 440]
    else:
        # 20×20 grid (VR-Folding): [0, 19, 380, 399]; edge [0, 19, 56, 75]
        return [0, 19, 56, 75] if is_edge else [0, 19, 380, 399]


def expand_eval_runs(eval_runs: list, args=None) -> list:
    """
    Expand EVAL_RUNS into a flat list of evaluation specs, one per
    (exp_dir, checkpoint_step, inf_steps) triple.  Reads the saved config.yml
    from the experiment directory to auto-detect scheduler type / dataset mode.

    args is passed only to resolve the CLI-default inf_steps fallback; it may
    be None when called just to list available short names.
    """
    specs = []
    for run in eval_runs:
        exp_dir    = run["exp_dir"]
        base_label = run["label"]

        config_path = os.path.join(exp_dir, "config.yml")
        if not os.path.isfile(config_path):
            print(f"  [SKIP] {exp_dir} — config.yml not found")
            continue

        config  = OmegaConf.load(config_path)
        use_fm  = "FlowMatching" in config.diffusion_cfg.type
        is_edge = config.dataset_cfg.get("train_mode", "full") == "edge_only"

        # Resolve the inf_steps list for this run
        inf_steps_list = run.get("inf_steps") or None
        if inf_steps_list is None:
            if args is not None:
                default = args.num_inference_steps_fm if use_fm else args.num_inference_steps_ddpm
            else:
                default = 50 if use_fm else 100
            inf_steps_list = [default]

        for ckpt_step in run["steps"]:
            try:
                ckpt_path = resolve_checkpoint(exp_dir, ckpt_step)
            except FileNotFoundError as e:
                print(f"  [SKIP] {base_label} @ step {ckpt_step} — {e}")
                continue

            actual_step = int(os.path.basename(ckpt_path).split("-")[1])

            for inf_steps in inf_steps_list:
                label = f"{base_label} @ {actual_step//1000}k, {inf_steps} steps"
                short = (
                    f"{'Edge' if is_edge else 'Full'}-"
                    f"{'FM' if use_fm else 'DDPM'}-"
                    f"{actual_step//1000}k-i{inf_steps}"
                )
                specs.append({
                    "label":          label,
                    "short":          short,
                    "config":         config,
                    "checkpoint_dir": ckpt_path,
                    "use_fm":         use_fm,
                    "is_edge":        is_edge,
                    "inf_steps":      inf_steps,
                    # Corner indices are grid-size dependent
                    "corner_indices": _corner_indices(
                        config.dataset_cfg.get("type", "ClothTrackingDataset"),
                        is_edge
                    ),
                })
                print(f"  Resolved: {label}  →  {ckpt_path}")

    return specs

# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def mse(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(np.mean((pred - gt) ** 2))


def chamfer_l1(pred: np.ndarray, gt: np.ndarray) -> float:
    total = 0.0
    for p, g in zip(pred, gt):
        d_p2g = cKDTree(g).query(p)[0]
        d_g2p = cKDTree(p).query(g)[0]
        total += (d_p2g.mean() + d_g2p.mean()) / 2.0
    return total / len(pred)


def f_score_at(pred: np.ndarray, gt: np.ndarray, threshold: float) -> float:
    total = 0.0
    for p, g in zip(pred, gt):
        d_p2g = cKDTree(g).query(p)[0]
        d_g2p = cKDTree(p).query(g)[0]
        precision = (d_p2g < threshold).mean()
        recall    = (d_g2p < threshold).mean()
        denom = precision + recall
        total += (2 * precision * recall / denom) if denom > 0 else 0.0
    return total / len(pred)

def hausdorff(pred: np.ndarray, gt: np.ndarray) -> float:
    total = 0.0
    for p, g in zip(pred, gt):
        h1 = cKDTree(g).query(p)[0].max()
        h2 = cKDTree(p).query(g)[0].max()
        total += max(h1, h2)
    return total / len(pred)


def corner_error(pred: np.ndarray, gt: np.ndarray, indices: list) -> float:
    return float(np.linalg.norm(pred[:, indices] - gt[:, indices], axis=-1).mean())


def perimeter_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean absolute difference in contour perimeter (sum of edge lengths)."""
    def perimeter(pts):
        return float(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=-1).sum())
    errs = [abs(perimeter(p) - perimeter(g)) for p, g in zip(pred, gt)]
    return float(np.mean(errs))


def coverage_area_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean abs difference in projected   XZ-plane convex hull area."""
    def area_2d(pts):
        xy = pts[:, [0, 2]]
        try:
            return ConvexHull(xy).volume  # 'volume' = area in 2-D
        except Exception:
            return 0.0
    errs = [abs(area_2d(p) - area_2d(g)) for p, g in zip(pred, gt)]
    return float(np.mean(errs))



# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline(config, checkpoint_dir: str, device: torch.device):
    model_cls = getattr(
        importlib.import_module("uniclothdiff.models"), config.model_cfg.type
    )
    model = model_cls.from_pretrained(checkpoint_dir, subfolder="model")
    model.to(device, dtype=torch.float32).eval()

    diff_dict = OmegaConf.to_container(config.diffusion_cfg)
    scheduler = build_scheduler(diff_dict)

    use_fm = "FlowMatching" in config.diffusion_cfg.type
    PipelineCls = ClothStateEstFMPipeline if use_fm else ClothStateEstPipeline
    pipeline = PipelineCls(model=model, scheduler=scheduler)
    pipeline.to(device, dtype=torch.float32)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline, use_fm


@torch.no_grad()
def run_inference_batch(pipeline, use_fm: bool, pcd, q_temp, num_steps: int,
                        device: torch.device):
    """Returns (pred [B,N,3], elapsed_ms float, gpu_mem_mb float).

    Uses CUDA events for accurate GPU timing when available; falls back to
    perf_counter on CPU.  Resets peak memory stats so gpu_mem_mb reflects
    only this call.
    """
    B, N, _ = q_temp.shape
    pcd    = pcd.to(device, dtype=torch.float32)
    q_temp = q_temp.to(device, dtype=torch.float32)

    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t_start = torch.cuda.Event(enable_timing=True)
        t_end   = torch.cuda.Event(enable_timing=True)
        t_start.record()
    else:
        t0 = time.perf_counter()

    out = pipeline(
        encoder_hidden_states=pcd,
        q_temp=q_temp,
        shape=(B, N, 3),
        num_inference_steps=num_steps,
        do_classifier_free_guidance=False,
        call_v2=True,
    )

    if use_cuda:
        t_end.record()
        torch.cuda.synchronize(device)
        elapsed_ms  = t_start.elapsed_time(t_end)          # ms, GPU-accurate
        gpu_mem_mb  = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        gpu_mem_mb = 0.0

    pred = out.frames if isinstance(out.frames, np.ndarray) else out.frames.cpu().numpy()
    return pred, elapsed_ms, gpu_mem_mb


# ─────────────────────────────────────────────────────────────────────────────
# Full evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(spec: dict, pipeline, val_loader, args, device: torch.device) -> dict:
    """
    spec     : one entry from expand_eval_runs()
    pipeline : pre-loaded ClothStateEst(FM)Pipeline, shared across inf_steps sweeps
               on the same checkpoint
    val_loader : pre-built DataLoader, shared across all checkpoints of the same
                 experiment directory
    """
    print(f"\n{'='*60}")
    print(f"  Evaluating: {spec['label']}")
    print(f"{'='*60}")

    config         = spec["config"]
    checkpoint_dir = spec["checkpoint_dir"]
    use_fm         = spec["use_fm"]
    corner_indices = spec["corner_indices"]

    # ── pipeline ─────────────────────────────────────────────────────────────
    # (pipeline is passed in from main() — already loaded and cached)
    num_steps = spec["inf_steps"]
    print(f"  Checkpoint  : {os.path.basename(checkpoint_dir)}")
    print(f"  Inference   : {'Flow Matching' if use_fm else 'DDPM'}  ({num_steps} steps)")

    # ── warmup (1 batch, no timing) ─────────────────────────────────────────
    warmup_batch = next(iter(val_loader))
    with torch.no_grad():
        run_inference_batch(
            pipeline, use_fm,
            warmup_batch["pcd"], warmup_batch["q_temp"],
            num_steps, device,
        )
    print(f"  Warmup done.")

    # ── accumulate metrics ──────────────────────────────────────────────────
    all_mse        = []
    all_chamfer    = []
    all_f1cm       = []
    all_f2cm       = []
    all_corner     = []
    all_hausdorff  = []
    all_perimeter  = []
    all_coverage   = []
    all_latency_ms = []   # per-sample latency (ms)
    all_gpu_mem_mb = []   # peak GPU mem per batch (MB)

    for batch in tqdm(val_loader, desc="  Batching", leave=False):
        pcd    = batch["pcd"]
        q_temp = batch["q_temp"]
        q_gt   = batch["q_gt"].numpy()

        pred, elapsed_ms, gpu_mem_mb = run_inference_batch(
            pipeline, use_fm, pcd, q_temp, num_steps, device
        )
        B = pred.shape[0]
        all_latency_ms.append(elapsed_ms / B)   # per sample
        all_gpu_mem_mb.append(gpu_mem_mb)

        all_mse.append(mse(pred, q_gt))
        all_chamfer.append(chamfer_l1(pred, q_gt))
        all_f1cm.append(f_score_at(pred, q_gt, threshold=0.01))
        all_f2cm.append(f_score_at(pred, q_gt, threshold=0.02))
        all_corner.append(corner_error(pred, q_gt, corner_indices))
        all_hausdorff.append(hausdorff(pred, q_gt))
        all_perimeter.append(perimeter_error(pred, q_gt))
        all_coverage.append(coverage_area_error(pred, q_gt))

    lat_arr = np.array(all_latency_ms)   # per-sample latencies
    lat_mean        = float(lat_arr.mean())
    lat_p50         = float(np.percentile(lat_arr, 50))
    lat_p95         = float(np.percentile(lat_arr, 95))
    fps             = 1000.0 / lat_mean if lat_mean > 0 else 0.0
    lat_per_step_ms = lat_mean / num_steps
    gpu_mem_peak_mb = float(max(all_gpu_mem_mb)) if all_gpu_mem_mb else 0.0

    results = {
        "label":              spec["label"],
        "short":              spec["short"],
        "mse":                float(np.mean(all_mse)),
        "chamfer_l1":         float(np.mean(all_chamfer)),
        "f1_1cm":             float(np.mean(all_f1cm)),
        "f1_2cm":             float(np.mean(all_f2cm)),
        "corner_err":         float(np.mean(all_corner)),
        "hausdorff":          float(np.mean(all_hausdorff)),
        "perimeter_err":      float(np.mean(all_perimeter)),
        "coverage_err":       float(np.mean(all_coverage)),
        # ── inference timing ──────────────────────────────────────────────
        "latency_ms":         lat_mean,
        "latency_ms_p50":     lat_p50,
        "latency_ms_p95":     lat_p95,
        "fps":                fps,
        "latency_per_step_ms":lat_per_step_ms,
        "gpu_mem_mb":         gpu_mem_peak_mb,
        # ── metadata ──────────────────────────────────────────────────────
        "checkpoint":         os.path.basename(checkpoint_dir),
        "num_val_samples":    len(val_loader.dataset),
        "inference_steps":    num_steps,
        "is_edge":            spec["is_edge"],
        "use_fm":             spec["use_fm"],
    }

    # pretty-print
    quality_keys = ["mse", "chamfer_l1", "f1_1cm", "f1_2cm",
                    "corner_err", "hausdorff", "perimeter_err", "coverage_err"]
    timing_keys  = ["latency_ms", "latency_ms_p50", "latency_ms_p95",
                    "fps", "latency_per_step_ms", "gpu_mem_mb"]
    print("  Quality metrics:")
    for k in quality_keys:
        print(f"    {k:<22s}: {results[k]:.4f}")
    print("  Timing metrics:")
    for k in timing_keys:
        unit = " MB" if k == "gpu_mem_mb" else (" fps" if k == "fps" else " ms")
        print(f"    {k:<22s}: {results[k]:.2f}{unit}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table printer
# ─────────────────────────────────────────────────────────────────────────────

LATEX_TEMPLATE = r"""
% ─────────────────────────────────────────────────────────────────────────────
% Auto-generated by eval_offline.py
% ─────────────────────────────────────────────────────────────────────────────
\begin{{table*}}[t]
  \centering
  \caption{{Cloth state estimation on VR-Folding validation set.
            Best results per column in \textbf{{bold}}.
            Metrics evaluated on {n_samples} validation samples.}}
  \label{{tab:state_est}}
  \setlength{{\tabcolsep}}{{5pt}}
  \begin{{tabular}}{{l c c c c c c c c}}
    \toprule
    \textbf{{Method}} &
    \textbf{{MSE\,↓}} &
    \textbf{{CD-L1\,↓}} &
    \textbf{{F@1cm\,↑}} &
    \textbf{{F@2cm\,↑}} &
    \textbf{{Corner\,↓}} &
    \textbf{{Hausdorff\,↓}} &
    \textbf{{Perim.\,↓}} &
    \textbf{{Lat.\,(ms)\,↓}} \\
    \midrule
{rows}    \bottomrule
  \end{{tabular}}
\end{{table*}}
"""

COL_KEYS   = ["mse", "chamfer_l1", "f1_1cm", "f1_2cm",
              "corner_err", "hausdorff", "perimeter_err", "latency_ms"]
COL_BETTER = ["min",  "min",        "max",    "max",
              "min",   "min",        "min",     "min"]


def fmt(val: float, key: str, best_val: float, better: str) -> str:
    is_best = abs(val - best_val) < 1e-9
    # FPS and latency look better with 1 decimal place
    s = f"{val:.1f}" if key in ("fps", "latency_ms", "latency_ms_p50", "latency_ms_p95") else f"{val:.4f}"
    return f"\\textbf{{{s}}}" if is_best else s


def build_latex(all_results: list[dict]) -> str:
    # find best per column
    bests = {}
    for key, better in zip(COL_KEYS, COL_BETTER):
        vals = [r[key] for r in all_results]
        bests[key] = min(vals) if better == "min" else max(vals)

    rows = ""
    for i, r in enumerate(all_results):
        cells = " & ".join(
            fmt(r[k], k, bests[k], b) for k, b in zip(COL_KEYS, COL_BETTER)
        )
        sep = r"    \midrule" + "\n" if i == 1 else ""   # visual break edge vs full
        rows += f"{sep}    {r['label']} & {cells} \\\\\n"

    n_samples = all_results[0]["num_val_samples"]
    return LATEX_TEMPLATE.format(rows=rows, n_samples=n_samples)


# ─────────────────────────────────────────────────────────────────────────────
# Efficiency table (matches manuscript tab:efficiency)
# ─────────────────────────────────────────────────────────────────────────────

# Maps (is_edge, use_fm) → LaTeX method macro
_METHOD_MACRO = {
    (False, False): r"\fullddpm",
    (True,  False): r"\edgeddpm",
    (False, True):  r"\fullfm",
    (True,  True):  r"\edgefm",
}
_NODE_MACRO = {
    False: r"$\Nfull$ (full)",
    True:  r"$\Nedge$ (edge)",
}
# Canonical ordering for the efficiency table rows (method first, then steps asc)
_EFF_METHOD_ORDER = [(False, False), (True, False), (False, True), (True, True)]
_EFF_BOLD_KEY     = (True, True)    # edge + FM: bold in manuscript


def build_efficiency_latex(all_results: list[dict]) -> str:
    """
    Produces a table matching the manuscript's tab:efficiency format.
    Each unique (is_edge, use_fm, inf_steps) combination gets its own row.
    Timing metrics are averaged over checkpoint steps that share the same
    (is_edge, use_fm, inf_steps) — latency is architecture+steps determined.
    """
    from collections import defaultdict
    # key: (is_edge, use_fm, inf_steps)
    groups: dict = defaultdict(list)
    for r in all_results:
        key = (r["is_edge"], r["use_fm"], r["inference_steps"])
        groups[key].append(r)

    def _avg(rs, field):
        return float(np.mean([r[field] for r in rs]))

    # Sort rows: by method order first, then inf_steps ascending
    sorted_keys = sorted(
        groups.keys(),
        key=lambda k: (_EFF_METHOD_ORDER.index(k[:2]) if k[:2] in _EFF_METHOD_ORDER else 99, k[2])
    )

    rows = ""
    # TRTM — non-diffusion baseline, hardcoded placeholder
    rows += "        TRTM              & $\\Nfull$ (full)         & 1 (direct)  & ---    & ---    \\\\\n"

    prev_method = None
    for (is_edge, use_fm, inf_steps) in sorted_keys:
        rs      = groups[(is_edge, use_fm, inf_steps)]
        macro   = _METHOD_MACRO.get((is_edge, use_fm), r"\unknown")
        nodes   = _NODE_MACRO[is_edge]
        steps_s = f"{inf_steps} (Euler)" if use_fm else str(inf_steps)
        lat_ms  = _avg(rs, "latency_ms")
        fps_val = _avg(rs, "fps")

        # Insert a \midrule between different methods for readability
        method_key = (is_edge, use_fm)
        if prev_method is not None and method_key != prev_method:
            rows += "        \\midrule\n"
        prev_method = method_key

        is_bold = (method_key == _EFF_BOLD_KEY)
        if is_bold:
            row = (
                f"        \\textbf{{{macro}}}  & "
                f"$\\mathbf{{\\Nedge}}$ \\textbf{{(edge)}} & "
                f"\\textbf{{{steps_s}}} & "
                f"\\textbf{{{lat_ms:.1f}}} & "
                f"\\textbf{{{fps_val:.1f}}} \\\\\n"
            )
        else:
            row = (
                f"        {macro:<17s} & {nodes:<24s} & {steps_s:<11s} "
                f"& {lat_ms:.1f}    & {fps_val:.1f}    \\\\\n"
            )
        rows += row

    return (
        "% Auto-generated by eval_offline.py\n"
        "\\begin{table}[h]\n"
        "    \\centering\n"
        "    \\caption{%\n"
        "        Inference efficiency on a single NVIDIA L40S GPU.\n"
        "        The ``Steps'' column refers to denoising steps (DDPM) or Euler integration steps (FM).\n"
        "    }\n"
        "    \\label{tab:efficiency}\n"
        "    \\begin{tabular}{llccc}\n"
        "        \\toprule\n"
        "        \\textbf{Method} & \\textbf{Nodes} & \\textbf{Steps} & \\textbf{Time (ms) $\\downarrow$} & \\textbf{FPS $\\uparrow$} \\\\\n"
        "        \\midrule\n"
        + rows
        + "        \\bottomrule\n    \\end{tabular}\n\\end{table}\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Wandb logging
# ─────────────────────────────────────────────────────────────────────────────

def log_to_wandb(all_results: list[dict], args):
    run = wandb.init(
        project=args.wandb_project_eval,
        name="offline_eval",
        config=vars(args),
        tags=["eval", "state_estimation"],
    )
    # summary table as a wandb Table
    cols = ["Method", "MSE", "CD-L1", "F@1cm", "F@2cm",
            "Corner-Err", "Hausdorff", "Perimeter-Err",
            "Latency-ms", "Lat-p50-ms", "Lat-p95-ms",
            "FPS", "Lat-per-step-ms", "GPU-mem-MB",
            "Checkpoint", "InfSteps"]
    table = wandb.Table(columns=cols)
    for r in all_results:
        table.add_data(
            r["label"], r["mse"], r["chamfer_l1"],
            r["f1_1cm"], r["f1_2cm"], r["corner_err"],
            r["hausdorff"], r["perimeter_err"],
            r["latency_ms"], r["latency_ms_p50"], r["latency_ms_p95"],
            r["fps"], r["latency_per_step_ms"], r["gpu_mem_mb"],
            r["checkpoint"], r["inference_steps"],
        )
    run.log({"eval/results_table": table})

    # also log individual scalars so they're searchable in wandb
    for r in all_results:
        prefix = f"eval/{r['short']}"
        run.log({
            f"{prefix}/mse":                r["mse"],
            f"{prefix}/chamfer_l1":         r["chamfer_l1"],
            f"{prefix}/f1_1cm":             r["f1_1cm"],
            f"{prefix}/f1_2cm":             r["f1_2cm"],
            f"{prefix}/corner_err":         r["corner_err"],
            f"{prefix}/hausdorff":          r["hausdorff"],
            f"{prefix}/perimeter_err":      r["perimeter_err"],
            f"{prefix}/latency_ms":         r["latency_ms"],
            f"{prefix}/latency_ms_p50":     r["latency_ms_p50"],
            f"{prefix}/latency_ms_p95":     r["latency_ms_p95"],
            f"{prefix}/fps":                r["fps"],
            f"{prefix}/latency_per_step_ms":r["latency_per_step_ms"],
            f"{prefix}/gpu_mem_mb":         r["gpu_mem_mb"],
        })
    run.finish()
    print(f"\n  Logged to wandb project: {args.wandb_project_eval}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Offline evaluation for cloth state estimation")
    p.add_argument("--device",           default="cuda")
    p.add_argument("--batch_size",       type=int, default=32)
    p.add_argument("--num_workers",      type=int, default=8)
    p.add_argument("--num_inference_steps_ddpm", type=int, default=100,
                   help="Fallback DDPM steps when inf_steps is not set in EVAL_RUNS")
    p.add_argument("--num_inference_steps_fm",   type=int, default=50,
                   help="Fallback FM Euler steps when inf_steps is not set in EVAL_RUNS")
    p.add_argument("--wandb_entity",         default=None,
                   help="wandb entity/username (for logging eval results)")
    p.add_argument("--wandb_project_eval",   default="ClothDiffusion_Eval",
                   help="wandb project to log evaluation results")
    p.add_argument("--no_wandb",  action="store_true",
                   help="Skip wandb logging (still prints LaTeX)")
    p.add_argument("--runs", nargs="*", default=None,
                   help="Subset of short labels to evaluate, e.g. Edge-FM-250k Full-DDPM-350k")
    p.add_argument("--out", default="eval_results_table.tex",
                   help="Path to write quality metrics LaTeX table")
    p.add_argument("--out_efficiency", default="eval_efficiency_table.tex",
                   help="Path to write inference efficiency LaTeX table")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Resolving evaluation runs...")
    specs = expand_eval_runs(EVAL_RUNS, args)

    if args.runs:
        specs = [s for s in specs if s["short"] in args.runs]
        if not specs:
            avail = [s["short"] for s in expand_eval_runs(EVAL_RUNS, args)]
            print(f"No runs matched {args.runs}.\nAvailable: {avail}")
            return

    if not specs:
        print("No evaluation specs resolved (check EVAL_RUNS and that exp dirs exist).")
        return

    # Build one val DataLoader per unique exp_dir (checkpoints and inf_step sweeps
    # within the same experiment all share the same dataset).
    loader_cache:   dict[str, torch.utils.data.DataLoader] = {}
    # Build one pipeline per unique checkpoint_dir (inf_step sweeps on the same
    # checkpoint reuse the loaded weights — only num_inference_steps changes).
    pipeline_cache: dict[str, object] = {}

    all_results = []
    for spec in specs:
        ckpt_dir = spec["checkpoint_dir"]

        # — DataLoader (keyed on exp_dir) ——————————————————————————————
        exp_dir = None
        for run in EVAL_RUNS:
            if ckpt_dir.startswith(run["exp_dir"]):
                exp_dir = run["exp_dir"]
                break

        if exp_dir not in loader_cache:
            config  = spec["config"]
            val_cfg = OmegaConf.to_container(config.dataset_cfg, resolve=True)
            val_cfg["mode"] = "val"
            val_cfg["do_point_cloud_augmentation"] = False
            val_dataset = build_dataset(val_cfg)
            loader_cache[exp_dir] = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            print(f"  [dataset] Built val DataLoader for {exp_dir}  "
                  f"({len(val_dataset)} samples)")

        # — Pipeline/model (keyed on checkpoint_dir) ——————————————————————
        if ckpt_dir not in pipeline_cache:
            print(f"  [model] Loading checkpoint {os.path.basename(ckpt_dir)} ...")
            pipeline_cache[ckpt_dir], _ = load_pipeline(
                spec["config"], ckpt_dir, device
            )

        val_loader = loader_cache[exp_dir]
        pipeline   = pipeline_cache[ckpt_dir]
        try:
            results = evaluate(spec, pipeline, val_loader, args, device)
            all_results.append(results)
        except Exception as e:
            print(f"\n  [ERROR] {spec['label']}: {e}")
            import traceback; traceback.print_exc()

    if not all_results:
        print("\nNo results collected.")
        return

    latex = build_latex(all_results)
    print("\n" + "─" * 60)
    print("LaTeX table:")
    print("─" * 60)
    print(latex)

    with open(args.out, "w") as f:
        f.write(latex)
    print(f"\n  Saved to: {args.out}")

    eff_latex = build_efficiency_latex(all_results)
    print("\n" + "─" * 60)
    print("Efficiency table:")
    print("─" * 60)
    print(eff_latex)
    with open(args.out_efficiency, "w") as f:
        f.write(eff_latex)
    print(f"  Saved to: {args.out_efficiency}")

    if not args.no_wandb:
        log_to_wandb(all_results, args)


if __name__ == "__main__":
    main()
