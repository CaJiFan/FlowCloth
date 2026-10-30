"""
Offline evaluation script for TRTM cloth state estimation models.
Upgraded for IEEE R-AL submission (Aggregates multiple seeds with Mean ± Std).

Quality table columns: MSE, CD-L1, F@2cm, Corner (CLE)
Dropped: F@1cm, Hausdorff, Perimeter, Latency (latency lives in efficiency table / Pareto plot only)
"""

import os
import sys
import argparse
import numpy as np
import torch
import math
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── import all shared helpers from the VR-Folding eval script ────────────────
from eval_offline import (
    resolve_checkpoint,
    _corner_indices,
    expand_eval_runs,
    mse, chamfer_l1, f_score_at, hausdorff,
    corner_error, perimeter_error, coverage_area_error,
    load_pipeline, run_inference_batch,
    evaluate,
    log_to_wandb,
)

# ── Trimmed column set for the quality table (Latency lives in efficiency table only) ──
COL_KEYS   = ["mse", "chamfer_l1", "f1_2cm", "corner_err"]
COL_BETTER = ["min",  "min",        "max",    "min"]

# ─────────────────────────────────────────────────────────────────────────────
# TRTM evaluation runs (Seed-Aware)
# ─────────────────────────────────────────────────────────────────────────────
# Define the base patterns. The script will automatically flatten these into 
# individual runs for each seed before evaluation.
BASE_EVAL_RUNS = [
    # --- Full Mesh (441 nodes) Laplacian Ablation ---
    {
        "base_dir":  "experiments/ablation_fm_lapl_w1.0",
        "seeds":     [1],
        "steps":     ["best"],
        "inf_steps": [1, 4, 10],
        "label":     r"\textsc{FlowCloth}",
        "weight_tex": "1.0",
    },
    {
        "base_dir":  "experiments/ablation_fm_lapl_w5.0",
        "seeds":     [1],
        "steps":     ["best"],
        "inf_steps": [1, 4, 10],
        "label":     r"\textsc{FlowCloth}",
        "weight_tex": "5.0",
    },
    {
        "base_dir":  "experiments/ablation_fm_lapl_w10.0",
        "seeds":     [1],
        "steps":     ["best"],
        "inf_steps": [1, 4, 10],
        "label":     r"\textsc{FlowCloth}",
        "weight_tex": "10.0",
    },
    # --- Contour Mesh (80 nodes) Laplacian Ablation ---
    {
        "base_dir":  "experiments/ablation_fm_edge_lapl_w1.0",
        "seeds":     [1],
        "steps":     ["best"],
        "inf_steps": [1, 4, 10],
        "label":     r"\textsc{Contour-FlowCloth}",
        "weight_tex": "1.0",
    },
    {
        "base_dir":  "experiments/ablation_fm_edge_lapl_w5.0",
        "seeds":     [1],
        "steps":     ["best"],
        "inf_steps": [1, 4, 10],
        "label":     r"\textsc{Contour-FlowCloth}",
        "weight_tex": "5.0",
    },
    {
        "base_dir":  "experiments/ablation_fm_edge_lapl_w10.0",
        "seeds":     [1],
        "steps":     ["best"],
        "inf_steps": [1, 4, 10],
        "label":     r"\textsc{Contour-FlowCloth}",
        "weight_tex": "10.0",
    },
]

# --- Observation normalisation / fold synthesis / new loss ablation (--suite obsnorm) ---
# weight_tex holds the variant tag: N = obs. normalisation, F = fold synthesis,
# L(w) = GT-relative t-weighted Laplacian with lambda_x1 = w
_OBSNORM_VARIANTS = [
    ("obsnorm",                 "N"),
    ("obsnorm_folds",           "N+F"),
    ("obsnorm_folds_nl_x1w10",  "N+F+L(10)"),
    ("obsnorm_folds_nl_x1w1",   "N+F+L(1)"),
]
BASE_EVAL_RUNS_OBSNORM = [
    {
        "base_dir":   f"experiments/ablation_fm{'_edge' if edge else ''}_{name}",
        "seeds":      [1],
        "steps":      ["best"],
        "inf_steps":  [1, 4, 10],
        "label":      r"\textsc{Contour-FlowCloth}" if edge else r"\textsc{FlowCloth}",
        "weight_tex": tag,
    }
    for edge in (False, True) for name, tag in _OBSNORM_VARIANTS
]
SUITES = {
    "lapl":    (BASE_EVAL_RUNS,         r"$\lambda_{\mathrm{lapl}}$",
                r"Ablation on discrete Laplacian weight $\lambda_{\mathrm{lapl}}$"),
    "obsnorm": (BASE_EVAL_RUNS_OBSNORM, r"Variant",
                r"Ablation on observation normalisation (N), fold synthesis (F) and GT-relative Laplacian loss (L)"),
}


def flatten_eval_runs(base_runs: list) -> list:
    eval_runs = []
    for run in base_runs:
        if "weight_tex" in run:
            weight_tex = run["weight_tex"]
        elif "_lapl_w" in run["base_dir"]:
            raw_w = run["base_dir"].split("_lapl_w")[-1]
            weight_tex = f"{raw_w}"
        elif "_sw" in run["base_dir"]:
            raw_w = run["base_dir"].split("_sw")[-1]
            if "e" in raw_w:
                base, exp = raw_w.split("e")
                weight_tex = f"${float(base):.1f} \\times 10^{{{int(exp)}}}$"
            else:
                weight_tex = f"${float(raw_w)}$"
        else:
            weight_tex = "-"

        for seed in run["seeds"]:
            exp_dir = f"{run['base_dir']}_s{seed}" 
            new_run = {
                "exp_dir": exp_dir,
                "steps": run["steps"],
                "inf_steps": run["inf_steps"],
                "label": f"{run['label']} (s{seed})", 
                "group_label": run["label"], 
                "weight_tex": weight_tex,
                "seed": seed
            }
            eval_runs.append(new_run)
    return eval_runs


EVAL_RUNS = flatten_eval_runs(BASE_EVAL_RUNS)
ACTIVE_SUITE = "lapl"
# ─────────────────────────────────────────────────────────────────────────────
# LaTeX quality table (Mean ± Std)
# ─────────────────────────────────────────────────────────────────────────────

LATEX_TEMPLATE = r"""
% ─────────────────────────────────────────────────────────────────────────────
% Auto-generated by eval_offline_trtm.py
% ─────────────────────────────────────────────────────────────────────────────
\begin{{table*}}[t]
  \centering
  \small
  \caption{{{caption_title} on TRTM validation set. Results are Mean {caption_seeds}.
            Best results per column in \textbf{{bold}}. Evaluated on {n_samples} samples.}}
  \label{{tab:state_est_trtm}}
  \setlength{{\tabcolsep}}{{4pt}}
  \begin{{adjustbox}}{{center}}
  \begin{{tabular}}{{l c c c c c c | c c}}
    \toprule
    \textbf{{Method}} &
    \textbf{{Inf. steps}} &
    \textbf{{{param_header}}} &
    \textbf{{MSE\,$\downarrow$}} &
    \textbf{{CD-L1\,$\downarrow$}} &
    \textbf{{F@2cm\,$\uparrow$}} &
    \textbf{{Corner\,$\downarrow$}} &
    \textbf{{Time (ms)\,$\downarrow$}} &
    \textbf{{FPS\,$\uparrow$}} \\
     & & & ($\times 10^{{-2}}$) & ($\times 10^{{-2}}$) & ($\times 10^{{-1}}$) & ($\times 10^{{-2}}$) & & \\
    \midrule
{rows}    \bottomrule
  \end{{tabular}}
  \end{{adjustbox}}
\end{{table*}}
"""

def build_latex_aggregated(agg_results: list) -> str:
    # Compute bests separately for Full and Contour (Edge) subsets, matching paper convention
    subgroups = set(r["is_edge"] for r in agg_results)
    bests = {sg: {} for sg in subgroups}
    for sg in subgroups:
        sub_rs = [r for r in agg_results if r["is_edge"] == sg]
        for key, better in zip(COL_KEYS, COL_BETTER):
            vals = [r[key + "_mean"] for r in sub_rs]
            bests[sg][key] = min(vals) if better == "min" else max(vals)
        bests[sg]["latency_ms"] = min(r["latency_ms_mean"] for r in sub_rs)
        bests[sg]["fps"]        = max(r["fps_mean"] for r in sub_rs)

    # Scale factors: MSE, CD-L1, Corner are multiplied ×100 to report as ×10^{-2}
    SCALE = {"mse": 100, "chamfer_l1": 100, "f1_2cm": 1, "corner_err": 100}

    rows = ""
    for i, r in enumerate(agg_results):
        sg = r["is_edge"]
        cells = []
        for k, b in zip(COL_KEYS, COL_BETTER):
            scale    = SCALE.get(k, 1)
            mean_val = r[k + "_mean"] * scale
            std_val  = r[k + "_std"]  * scale
            is_best  = math.isclose(r[k + "_mean"], bests[sg][k], rel_tol=1e-4)

            dec = 3 if k == "f1_2cm" else 2

            if r[k + "_std"] > 0:
                val_str = f"{mean_val:.{dec}f} \\pm {std_val:.{dec}f}"
            else:
                val_str = f"{mean_val:.{dec}f}"

            if is_best:
                cells.append(f"$\\mathbf{{{val_str}}}$")
            else:
                cells.append(f"${val_str}$")

        row_str = " & ".join(cells)

        lat_val = r["latency_ms_mean"]
        fps_val = r["fps_mean"]
        is_best_lat = math.isclose(lat_val, bests[sg]["latency_ms"], rel_tol=1e-3)
        is_best_fps = math.isclose(fps_val, bests[sg]["fps"], rel_tol=1e-3)

        lat_str = f"$\\mathbf{{{lat_val:.1f}}}$" if is_best_lat else f"${lat_val:.1f}$"
        fps_str = f"$\\mathbf{{{fps_val:.1f}}}$" if is_best_fps else f"${fps_val:.1f}$"

        sep = ""
        if i > 0:
            prev = agg_results[i - 1]
            if r['label'] != prev['label'] or r['weight_tex'] != prev['weight_tex']:
                sep = r"    \midrule" + "\n"

        rows += f"{sep}    {r['label']} & {r['inference_steps']} & {r['weight_tex']} & {row_str} & {lat_str} & {fps_str} \\\\\n"

    n_samples = agg_results[0]["num_val_samples"]
    seeds_seen = set(r.get("seed", 1) for r in agg_results)
    caption_seeds = f"across {len(seeds_seen)} seeds" if len(seeds_seen) > 1 else "(single seed)"
    _, param_header, caption_title = SUITES[ACTIVE_SUITE]
    return LATEX_TEMPLATE.format(rows=rows, n_samples=n_samples, caption_seeds=caption_seeds,
                                 param_header=param_header, caption_title=caption_title)


# ─────────────────────────────────────────────────────────────────────────────
# Efficiency table (Aggregated)
# ─────────────────────────────────────────────────────────────────────────────

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
_EFF_METHOD_ORDER = [(False, False), (True, False), (False, True), (True, True)]
_EFF_BOLD_KEY     = (True, True)

def build_efficiency_latex(agg_results: list) -> str:
    # Uses the already aggregated latency and fps means
    sorted_results = sorted(
        agg_results,
        key=lambda r: (
            _EFF_METHOD_ORDER.index((r["is_edge"], r["use_fm"])) if (r["is_edge"], r["use_fm"]) in _EFF_METHOD_ORDER else 99,
            r["inference_steps"]
        )
    )

    rows = ""
    prev_method = None
    for r in sorted_results:
        is_edge = r["is_edge"]
        use_fm = r["use_fm"]
        inf_steps = r["inference_steps"]
        
        macro   = _METHOD_MACRO.get((is_edge, use_fm), r"\unknown")
        nodes   = _NODE_MACRO[is_edge]
        steps_s = f"{inf_steps} (Euler)" if use_fm else str(inf_steps)
        lat_ms  = r["latency_ms_mean"]
        fps_val = r["fps_mean"]

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
        "% Auto-generated by eval_offline_trtm.py\n"
        "\\begin{table}[h]\n"
        "    \\centering\n"
        "    \\caption{%\n"
        "        Inference efficiency on TRTM dataset, single NVIDIA L40S GPU.\n"
        "        The ``Steps'' column refers to denoising steps (DDPM) or Euler integration steps (FM).\n"
        "    }\n"
        "    \\label{tab:efficiency_trtm}\n"
        "    \\begin{tabular}{llccc}\n"
        "        \\toprule\n"
        "        \\textbf{Method} & \\textbf{Nodes} & \\textbf{Steps} "
        "& \\textbf{Time (ms) $\\downarrow$} & \\textbf{FPS $\\uparrow$} \\\\\n"
        "        \\midrule\n"
        + rows
        + "        \\bottomrule\n    \\end{tabular}\n\\end{table}\n"
    )

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Offline evaluation for cloth state estimation — TRTM dataset"
    )
    p.add_argument("--device",           default="cuda")
    p.add_argument("--batch_size",       type=int, default=32)
    p.add_argument("--num_workers",      type=int, default=8)
    p.add_argument("--num_inference_steps_ddpm", type=int, default=100)
    p.add_argument("--num_inference_steps_fm",   type=int, default=50)
    p.add_argument("--wandb_entity",         default=None)
    p.add_argument("--wandb_project_eval",   default="ClothDiffusion_TRTM_Eval")
    p.add_argument("--no_wandb",  action="store_true")
    p.add_argument("--runs", nargs="*", default=None,
                   help="Subset of short labels to evaluate")
    p.add_argument("--out",            default="eval_results_trtm_table.tex")
    p.add_argument("--out_efficiency", default="eval_efficiency_trtm_table.tex")
    p.add_argument("--suite", choices=list(SUITES), default="lapl",
                   help="Which set of runs to evaluate (BASE_EVAL_RUNS or BASE_EVAL_RUNS_OBSNORM)")
    p.add_argument("--eval_norm_mode", choices=["config", "gt", "obs"], default="config",
                   help="Override dataset norm_mode at eval. 'obs' reproduces deployment preprocessing for any model")
    p.add_argument("--raw_units", action="store_true",
                   help="Undo normalisation before computing metrics, so models with different norm_mode are comparable")
    p.add_argument("--val_fold_prob", type=float, default=0.0,
                   help="Replace this fraction of val samples with deterministic procedural folds (fold-robustness eval)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    global EVAL_RUNS, ACTIVE_SUITE
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ACTIVE_SUITE = args.suite
    EVAL_RUNS = flatten_eval_runs(SUITES[args.suite][0])

    print("Resolving TRTM evaluation runs...")
    specs = expand_eval_runs(EVAL_RUNS, args)

    if args.runs:
        specs = [s for s in specs if s["short"] in args.runs]
        if not specs:
            avail = [s["short"] for s in expand_eval_runs(EVAL_RUNS, args)]
            print(f"No runs matched {args.runs}.\nAvailable: {avail}")
            return

    if not specs:
        print("No evaluation specs resolved (check BASE_EVAL_RUNS and that exp dirs exist).")
        return

    loader_cache:   dict = {}
    pipeline_cache: dict = {}

    all_results = []
    for spec in specs:
        from omegaconf import OmegaConf
        ckpt_dir = spec["checkpoint_dir"]

        # 1. Recover the original base run settings from the flattened EVAL_RUNS
        original_run = next(r for r in EVAL_RUNS if ckpt_dir.startswith(r["exp_dir"]))
        group_label = original_run["group_label"]
        seed = original_run["seed"]

        # 2. Build DataLoader (Re-use across seeds to save RAM/Time!)
        if group_label not in loader_cache:
            config  = spec["config"]
            val_cfg = OmegaConf.to_container(config.dataset_cfg, resolve=True)
            val_cfg["mode"] = "val"
            val_cfg["do_point_cloud_augmentation"] = False
            if args.eval_norm_mode != "config":
                val_cfg["norm_mode"] = args.eval_norm_mode
            val_cfg["eval_fold_prob"] = args.val_fold_prob
            val_cfg["return_norm_params"] = args.raw_units
            from uniclothdiff.registry import build_dataset
            val_dataset = build_dataset(val_cfg)
            loader_cache[group_label] = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            print(f"  [dataset] Built val DataLoader for {group_label}  "
                  f"({len(val_dataset)} samples)")

        # 3. Load Pipeline
        if ckpt_dir not in pipeline_cache:
            print(f"  [model] Loading checkpoint {os.path.basename(ckpt_dir)} (Seed {seed}) ...")
            pipeline_cache[ckpt_dir], _ = load_pipeline(
                spec["config"], ckpt_dir, device
            )

        val_loader = loader_cache[group_label]
        pipeline   = pipeline_cache[ckpt_dir]
        
        try:
            results = evaluate(spec, pipeline, val_loader, args, device)
            # Inject grouping identifiers
            results["group_label"] = group_label
            results["weight_tex"] = original_run.get("weight_tex", "-")
            results["seed"] = seed
            all_results.append(results)
        except Exception as e:
            print(f"\n  [ERROR] {spec['label']}: {e}")
            import traceback; traceback.print_exc()

    if not all_results:
        print("\nNo results collected.")
        return

    # ─────────────────────────────────────────────────────────────────────────
    # SEED AGGREGATION LOGIC
    # ─────────────────────────────────────────────────────────────────────────
    print("\nAggregating results across seeds...")
    grouped = defaultdict(list)
    for r in all_results:
        # Include weight_tex in the grouping key!
        key = (r["group_label"], r["weight_tex"], r["inference_steps"])
        grouped[key].append(r)

    agg_results = []
    for (g_label, w_tex, inf_steps), runs in grouped.items():
        agg_r = {
            "label": g_label,
            "weight_tex": w_tex, # Pass it to the aggregated result
            "inference_steps": inf_steps,
            "is_edge": runs[0]["is_edge"],
            "use_fm": runs[0]["use_fm"],
            "num_val_samples": runs[0]["num_val_samples"],
        }
        
        # Always aggregate latency/fps for the efficiency table,
        # regardless of what's in the quality-table COL_KEYS.
        metrics_to_agg = COL_KEYS + ["latency_ms", "fps"]
        for k in metrics_to_agg:
            vals = [r[k] for r in runs]
            agg_r[f"{k}_mean"] = float(np.mean(vals))
            agg_r[f"{k}_std"]  = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            
        agg_results.append(agg_r)

    # ─────────────────────────────────────────────────────────────────────────
    # OUTPUT GENERATION
    # ─────────────────────────────────────────────────────────────────────────
    latex = build_latex_aggregated(agg_results)
    print("\n" + "─" * 60)
    print("LaTeX table (TRTM - Aggregated):")
    print("─" * 60)
    print(latex)
    with open(args.out, "a") as f:
        f.write(latex)
    print(f"\n  Saved to: {args.out}")

    eff_latex = build_efficiency_latex(agg_results)
    print("\n" + "─" * 60)
    print("Efficiency table (TRTM - Aggregated):")
    print("─" * 60)
    print(eff_latex)
    with open(args.out_efficiency, "a") as f:
        f.write(eff_latex)
    print(f"  Saved to: {args.out_efficiency}")

    if not args.no_wandb:
        log_to_wandb(all_results, args) # Logs individual runs to wandb

if __name__ == "__main__":
    main()