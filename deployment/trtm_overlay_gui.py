#!/usr/bin/env python3
"""
trtm_overlay_gui.py
───────────────────
Interactive GUI for TRTM dataset inspection and model prediction overlay.
Allows visualizing training/validation/test samples side-by-side with
FlowCloth / UniClothDiff model predictions, without needing live camera hardware.

Usage:
  python deployment/trtm_overlay_gui.py \
      --ckpt_dir checkpoints/ablation_fm_sw100.3_s3/checkpoints/checkpoint-925000/ \
      --data_dir /path/to/TRTM/template_square \
      --mode full \
      --num_steps 5 \
      --tta_rotations 4
"""

import os
import sys
import time
import glob
import pickle
import random
import argparse
import importlib
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np
import torch

# Ensure repository root is on sys.path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from omegaconf import OmegaConf
from uniclothdiff.registry import build_scheduler
from uniclothdiff.pipelines.cloth_state_est_pipeline import ClothStateEstPipeline
from uniclothdiff.pipelines.cloth_state_est_fm_pipeline import ClothStateEstFMPipeline

# ─────────────────────────────────────────────────────────────────────────────
# Constants & TRTM Standards
# ─────────────────────────────────────────────────────────────────────────────
TARGET_CANVAS_SIZE = 720
IDEAL_CLOTH_SIZE   = 480
FLAT_PIXEL_VAL     = 192
PIXEL_PER_MM       = 2.0
TRTM_FOV_DEG       = 60.0
SCALE_PX_SIMU      = 187.0 * (TARGET_CANVAS_SIZE / 552.0)  # ~243.913 px/m for TARGET_CANVAS_SIZE

# Corner indices for 21x21 (441) and 80-edge grids
# In TRTM world coordinates (+Y up, -Y down):
#   Row 20 (Y = +1) is TOP of table -> Index 420 (TL), Index 440 (TR)
#   Row 0  (Y = -1) is BOTTOM of table -> Index 0 (BL), Index 20 (BR)
CORNER_IDX_FULL = [420, 440, 0, 20]
CORNER_IDX_EDGE = [59, 79, 0, 20]
CORNER_NAMES    = ["TL", "TR", "BL", "BR"]
CORNER_COLORS   = [
    (0, 0, 255),    # TL: Red
    (0, 255, 0),    # TR: Green
    (255, 0, 0),    # BL: Blue
    (255, 0, 255),  # BR: Magenta
]


# ─────────────────────────────────────────────────────────────────────────────
# Data Sample Class
# ─────────────────────────────────────────────────────────────────────────────
class DatasetSample:
    def __init__(
        self,
        sample_id: str,
        depth_path: str,
        split: str = "unknown",
        color_path: Optional[str] = None,
        mesh_gt_path: Optional[str] = None,
    ):
        self.sample_id = sample_id
        self.depth_path = depth_path
        self.color_path = color_path
        self.mesh_gt_path = mesh_gt_path
        self.split = split

        self._depth_img = None
        self._color_img = None
        self._mesh_gt = None

    def get_depth_image(self) -> Optional[np.ndarray]:
        if self._depth_img is None and os.path.exists(self.depth_path):
            img = cv2.imread(self.depth_path, cv2.IMREAD_UNCHANGED)
            if img is not None:
                if img.ndim == 3:
                    img = img[:, :, 0]
                if img.shape[:2] != (TARGET_CANVAS_SIZE, TARGET_CANVAS_SIZE):
                    img = cv2.resize(img, (TARGET_CANVAS_SIZE, TARGET_CANVAS_SIZE), interpolation=cv2.INTER_AREA)
                self._depth_img = img
        return self._depth_img

    def get_color_image(self) -> Optional[np.ndarray]:
        if self._color_img is None and self.color_path and os.path.exists(self.color_path):
            img = cv2.imread(self.color_path, cv2.IMREAD_COLOR)
            if img is not None:
                if img.shape[:2] != (TARGET_CANVAS_SIZE, TARGET_CANVAS_SIZE):
                    img = cv2.resize(img, (TARGET_CANVAS_SIZE, TARGET_CANVAS_SIZE), interpolation=cv2.INTER_AREA)
                self._color_img = img
        return self._color_img

    def get_gt_mesh(self) -> Optional[np.ndarray]:
        if self._mesh_gt is None and self.mesh_gt_path and os.path.exists(self.mesh_gt_path):
            try:
                raw = np.loadtxt(self.mesh_gt_path, dtype=np.float32)
                if raw.ndim == 2:
                    self._mesh_gt = raw[:, :3]  # take first 3 coords (X, Y, Z)
            except Exception as e:
                print(f"[Warning] Could not load GT mesh from {self.mesh_gt_path}: {e}")
        return self._mesh_gt


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Discovery & Loading
# ─────────────────────────────────────────────────────────────────────────────
def discover_trtm_samples(
    data_dir: Optional[str] = None,
    split: str = "auto",
    max_samples: Optional[int] = None,
    random_subset: Optional[int] = None,
    shuffle: bool = False,
) -> List[DatasetSample]:
    """
    Intelligently discovers TRTM samples from data_dir or common search paths.
    Supports:
      - TRTM simu (simu/train, simu/val) with *.simu_depth.png & *.simu_mesh.txt
      - TRTM real (real/test) with *.real_depth.png & *.real_color.png
      - Recorded calibration folders with depth_trtm_*.png, depth_raw_*.png
      - Prediction eval folders with *.real_depth.png & *.pred_mesh.txt
    """
    search_dirs = []
    if data_dir and os.path.exists(data_dir):
        search_dirs.append(os.path.abspath(data_dir))
    else:
        # Default fallbacks
        candidates = [
            "/mnt/Data/cjimenez/TRTM/datasets/template_square",
            os.path.join(_REPO_ROOT, "datasets", "template_square"),
            os.path.expanduser("~/Projects/Tracking/Code/TrackAnyCloth/TRTM/datasets/template_square"),
            os.path.expanduser("~/projects/datasets/TRTM/template_square"),
            os.path.expanduser("~/Downloads/pred_contour_loss_full_input"),
            os.path.expanduser("~/Downloads/dataset_trtm/trtm_orbbec_tuning_20260129_120342"),
            os.path.expanduser("~/Projects/FC-DC/trtm_data_calibrated_20260619_104254"),
        ]
        for c in candidates:
            if os.path.exists(c):
                search_dirs.append(c)

    if not search_dirs:
        print("[Error] No dataset directory found. Please specify --data_dir /path/to/dataset")
        return []

    samples: List[DatasetSample] = []

    for base in search_dirs:
        print(f"[Dataset] Scanning directory: {base}")

        # Check for TRTM simu directory structure (simu/train, train/simu, simu/val, val/simu)
        simu_splits = []
        if split in ["train", "auto"]:
            simu_splits.append(("train", os.path.join(base, "simu", "train")))
            simu_splits.append(("train", os.path.join(base, "train", "simu")))
            simu_splits.append(("train", os.path.join(base, "train")))
        if split in ["val", "auto"]:
            simu_splits.append(("val", os.path.join(base, "simu", "val")))
            simu_splits.append(("val", os.path.join(base, "val", "simu")))
            simu_splits.append(("val", os.path.join(base, "val")))

        for s_name, s_dir in simu_splits:
            if os.path.isdir(s_dir):
                # Look for *.simu_depth.png
                depth_files = sorted(glob.glob(os.path.join(s_dir, "*.simu_depth.png")))
                for df in depth_files:
                    stem = df[:-len(".simu_depth.png")]
                    mf = f"{stem}.simu_mesh.txt"
                    mf_path = mf if os.path.exists(mf) else None
                    samples.append(DatasetSample(
                        sample_id=os.path.basename(stem),
                        depth_path=df,
                        split=f"simu/{s_name}",
                        color_path=None,
                        mesh_gt_path=mf_path,
                    ))

        # Also check if base itself directly contains *.simu_depth.png
        if not samples and os.path.isdir(base):
            simu_depths = sorted(glob.glob(os.path.join(base, "*.simu_depth.png")))
            if simu_depths:
                for df in simu_depths:
                    stem = df[:-len(".simu_depth.png")]
                    mf = f"{stem}.simu_mesh.txt"
                    samples.append(DatasetSample(
                        sample_id=os.path.basename(stem),
                        depth_path=df,
                        split="simu/train",
                        color_path=None,
                        mesh_gt_path=mf if os.path.exists(mf) else None,
                    ))

        # Check for TRTM real test directory (real/test) only if requested
        if not samples and split in ["real_test", "auto", "test"]:
            real_dir = os.path.join(base, "real", "test")
            if os.path.isdir(real_dir):
                depth_files = sorted(glob.glob(os.path.join(real_dir, "*.real_depth.png")))
                for df in depth_files:
                    stem = df[:-len(".real_depth.png")]
                    cf = f"{stem}.real_color.png"
                    samples.append(DatasetSample(
                        sample_id=os.path.basename(stem),
                        depth_path=df,
                        split="real/test",
                        color_path=cf if os.path.exists(cf) else None,
                        mesh_gt_path=None,
                    ))

        # Check for direct flat directory containing real depth images as fallback
        if not samples:
            if split in ["train", "val"]:
                print(f"[Dataset] Note: '{split}' simulation data was not found in {base}.")
                print(f"[Dataset] Falling back to available real depth test captures.")
            real_depths = sorted(glob.glob(os.path.join(base, "*.real_depth.png")))
            if real_depths:
                print(f"[Dataset] Found {len(real_depths)} real depth samples in: {base}")
                for df in real_depths:
                    stem = df[:-len(".real_depth.png")]
                    cf = f"{stem}.real_color.png"
                    mf = None
                    for ext in [".simu_mesh.txt", ".pred_mesh.txt", ".mesh.txt"]:
                        cand = f"{stem}{ext}"
                        if os.path.exists(cand):
                            mf = cand
                            break
                    samples.append(DatasetSample(
                        sample_id=os.path.basename(stem),
                        depth_path=df,
                        split="real_depth",
                        color_path=cf if os.path.exists(cf) else None,
                        mesh_gt_path=mf,
                    ))

            trtm_depths = sorted(glob.glob(os.path.join(base, "depth_trtm_*.png")))
            if trtm_depths:
                print(f"[Dataset] Found {len(trtm_depths)} calibrated depth samples in: {base}")
                for df in trtm_depths:
                    fname = os.path.basename(df)
                    num_str = fname.replace("depth_trtm_", "").replace(".png", "")
                    cf = os.path.join(base, f"color_{num_str}.png")
                    samples.append(DatasetSample(
                        sample_id=num_str,
                        depth_path=df,
                        split="calibrated_trtm",
                        color_path=cf if os.path.exists(cf) else None,
                        mesh_gt_path=None,
                    ))

        if samples:
            break

    if not samples:
        print("[Dataset] No matching samples found. If you have TRTM simulation training data (simu/train), please specify:")
        print("          python3 deployment/trtm_overlay_gui.py --data_dir /path/to/template_square")

    if shuffle:
        random.seed(42)
        random.shuffle(samples)

    if random_subset and len(samples) > random_subset:
        random.seed(42)
        samples = random.sample(samples, random_subset)
    elif max_samples and len(samples) > max_samples:
        samples = samples[:max_samples]

    print(f"[Dataset] Total samples loaded: {len(samples)}")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Model & Template Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_pipeline(config, checkpoint_dir: str, device: torch.device):
    """Load model weights and build the inference pipeline."""
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


def load_template(config, is_edge: bool) -> np.ndarray:
    """Load the template mesh from pickle file."""
    patch_file = config.model_cfg.get("patch_file", "assets/trtm_441_voronoi_template.pkl")
    if not os.path.isabs(patch_file):
        patch_file = os.path.join(_REPO_ROOT, patch_file)

    if not os.path.exists(patch_file):
        fallback = os.path.join(_REPO_ROOT, "assets", "trtm_441_voronoi_template.pkl")
        if os.path.exists(fallback):
            patch_file = fallback
        else:
            raise FileNotFoundError(f"Template patch file not found: {patch_file}")

    with open(patch_file, "rb") as f:
        data = pickle.load(f)

    pts = data["points"].astype(np.float32) if "points" in data else data["mesh_pos"].astype(np.float32)

    if is_edge:
        N = 21
        grid = np.arange(N * N).reshape(N, N)
        contour = np.unique(np.concatenate([
            grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]
        ]))
        if len(pts) == N * N:
            pts = pts[contour]

    pts -= pts.mean(axis=0)
    sc = float(np.max(np.linalg.norm(pts, axis=1)))
    if sc > 1e-6:
        pts /= sc

    return pts


# ─────────────────────────────────────────────────────────────────────────────
# Point Cloud Extraction
# ─────────────────────────────────────────────────────────────────────────────
def depth_to_pcd_trtm(depth_img: np.ndarray, num_points: int = 400, is_simu: bool = True):
    """
    Back-projects a TRTM depth image into a normalised 3D point cloud [num_points, 3].
    - For simulation (*.simu_depth.png): Uses exact orthographic metric projection matching training:
        x = (xs - cx) / scale_px,  y = -(ys - cy) / scale_px,  z = (192 - d) / 228
    - For real-world images: Uses calibrated perspective camera backprojection.
    """
    d = depth_img.astype(np.float32)
    if d.ndim == 3:
        d = d[:, :, 0]
    h, w = d.shape[:2]
    cx = w / 2.0
    cy = h / 2.0

    if is_simu:
        cloth_mask = d < 250
        ys, xs = np.where(cloth_mask)
        scale_px = 187.0 * (w / 552.0)
        intrinsics = (scale_px, scale_px, cx, cy)
        if len(xs) == 0:
            return np.zeros((num_points, 3), dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0, intrinsics

        # Random subsample cloth pixels (matching training pipeline)
        if len(xs) >= num_points:
            sel = np.random.choice(len(xs), num_points, replace=False)
        else:
            sel = np.random.choice(len(xs), num_points, replace=True)
        xs_s, ys_s = xs[sel], ys[sel]
        d_vals = d[ys_s, xs_s]

        x = (xs_s.astype(np.float32) - cx) / scale_px
        y = -(ys_s.astype(np.float32) - cy) / scale_px
        z = (192.0 - d_vals) / 228.0

        pts = np.stack([x, y, z], axis=1).astype(np.float32)
        com = pts.mean(axis=0)
        pts_centered = pts - com
        scale = float(np.max(np.linalg.norm(pts_centered, axis=1)))
        if scale > 1e-6:
            pts_norm = pts_centered / scale
        else:
            pts_norm = pts_centered

        return pts_norm, com, scale, intrinsics
    else:
        fov_rad = np.deg2rad(TRTM_FOV_DEG)
        fx = fy = (w / 2.0) / np.tan(fov_rad / 2.0)

        # Segment cloth: table is around 192, folds go down to ~80-120
        # Background is typically > 240 or < 10
        cloth_mask = (d < 240) & (d > 10)
        if not np.any(cloth_mask):
            cloth_mask = d < 250

        ys, xs = np.where(cloth_mask)
        intrinsics = (fx, fy, cx, cy)
        if len(xs) == 0:
            return np.zeros((num_points, 3), dtype=np.float32), np.zeros(3, dtype=np.float32), 1.0, intrinsics

        if len(xs) >= num_points:
            sel = np.random.choice(len(xs), num_points, replace=False)
        else:
            sel = np.random.choice(len(xs), num_points, replace=True)
        xs_s, ys_s = xs[sel], ys[sel]
        d_vals = d[ys_s, xs_s]

        # Height relative to flat table (inverted shading)
        zs = (FLAT_PIXEL_VAL - d_vals) / PIXEL_PER_MM

        # Synthetic distance in camera frame
        dist = 800.0 - zs
        X = (xs_s.astype(np.float32) - cx) / fx * dist
        Y = -(ys_s.astype(np.float32) - cy) / fy * dist  # +Y is forward/UP (matching table frame & training meshes)
        Z = dist

        pts = np.stack([X, Y, Z], axis=1).astype(np.float32)

        # Normalise: zero-mean, unit max L2-norm
        com = pts.mean(axis=0)
        pts_centered = pts - com
        scale = float(np.max(np.linalg.norm(pts_centered, axis=1)))
        if scale > 1e-6:
            pts_norm = pts_centered / scale
        else:
            pts_norm = pts_centered

        return pts_norm, com, scale, intrinsics


def mesh_to_pcd_simu(mesh_gt: np.ndarray, num_points: int = 400, jitter_sigma: float = 0.002):
    """
    Generates point cloud directly from simulation mesh vertices (same as training pipeline).
    """
    mesh = mesh_gt.copy()
    com = mesh.mean(axis=0)
    mesh_centered = mesh - com
    scale = float(np.max(np.linalg.norm(mesh_centered, axis=1)))
    if scale > 1e-6:
        mesh_norm = mesh_centered / scale
    else:
        mesh_norm = mesh_centered

    idx = np.random.randint(0, len(mesh_norm), size=num_points)
    pcd = mesh_norm[idx].copy()
    if jitter_sigma > 0:
        pcd += (np.random.randn(*pcd.shape) * jitter_sigma).astype(np.float32)
    return pcd, com, scale


# ─────────────────────────────────────────────────────────────────────────────
# Inference with TTA
# ─────────────────────────────────────────────────────────────────────────────
def _yaw_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _chamfer_score(pred: np.ndarray, pcd: np.ndarray) -> float:
    """Symmetric Chamfer distance to prevent picking collapsed / shrunken meshes."""
    diff = pred[:, None, :] - pcd[None, :, :]
    d2 = (diff ** 2).sum(axis=-1)
    return float(d2.min(axis=1).mean() + d2.min(axis=0).mean())


@torch.no_grad()
def run_inference(
    pipeline,
    use_fm: bool,
    pcd_np: np.ndarray,
    q_temp_np: np.ndarray,
    num_steps: int,
    device: torch.device,
    tta_rotations: int = 1,
) -> np.ndarray:
    """Runs single pass or Test-Time Augmentation forward pass."""
    if tta_rotations <= 1:
        pcd = torch.tensor(pcd_np[None], dtype=torch.float32, device=device)
        q_temp = torch.tensor(q_temp_np[None], dtype=torch.float32, device=device)
        N = q_temp.shape[1]
        out = pipeline(
            encoder_hidden_states=pcd,
            q_temp=q_temp,
            shape=(1, N, 3),
            num_inference_steps=num_steps,
            do_classifier_free_guidance=False,
            call_v2=True,
        )
        pred = out.frames if isinstance(out.frames, np.ndarray) else out.frames.cpu().numpy()
        return pred[0]

    angles = [2 * np.pi * k / tta_rotations for k in range(tta_rotations)]
    best_pred, best_score = None, float("inf")

    for theta in angles:
        R = _yaw_matrix(theta)
        Rt = R.T
        pcd_rot = pcd_np @ R
        q_temp_rot = q_temp_np @ R

        pcd_t = torch.tensor(pcd_rot[None], dtype=torch.float32, device=device)
        q_temp_t = torch.tensor(q_temp_rot[None], dtype=torch.float32, device=device)
        N = q_temp_t.shape[1]

        out = pipeline(
            encoder_hidden_states=pcd_t,
            q_temp=q_temp_t,
            shape=(1, N, 3),
            num_inference_steps=num_steps,
            do_classifier_free_guidance=False,
            call_v2=True,
        )
        pred_rot = out.frames if isinstance(out.frames, np.ndarray) else out.frames.cpu().numpy()
        pred_orig = pred_rot[0] @ Rt

        score = _chamfer_score(pred_orig, pcd_np)
        if score < best_score:
            best_score = score
            best_pred = pred_orig

    return best_pred


# ─────────────────────────────────────────────────────────────────────────────
# Rendering & Visual Overlay
# ─────────────────────────────────────────────────────────────────────────────
def project_to_canvas(
    pts_norm: np.ndarray,
    com: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
    intrinsics: Optional[Tuple[float, float, float, float]] = None,
    canvas_size: int = TARGET_CANVAS_SIZE,
    cloth_size: int = IDEAL_CLOTH_SIZE,
    is_ortho: bool = False,
) -> np.ndarray:
    """
    Projects 3D points back to canvas pixels [u, v].
    If com, scale, and intrinsics (fx, fy, cx, cy) are provided:
      - If is_ortho: u = pts_3d[:, 0] * fx + cx, v = cy - pts_3d[:, 1] * fy
      - If perspective: u = (pts_3d[:, 0] * fx / pts_3d[:, 2]) + cx, v = cy - (pts_3d[:, 1] * fy / pts_3d[:, 2])
    Otherwise, falls back to canonical canvas mapping.
    """
    if com is not None and scale is not None and intrinsics is not None and len(intrinsics) == 4:
        fx, fy, cx, cy = intrinsics
        pts_3d = pts_norm * scale + com
        if is_ortho or np.all(pts_3d[:, 2] < 5.0):
            # Orthographic projection (metric units)
            u = pts_3d[:, 0] * fx + cx
            v = cy - pts_3d[:, 1] * fy
        else:
            # Pinhole perspective projection (Z in mm)
            u = (pts_3d[:, 0] * fx / pts_3d[:, 2]) + cx
            v = cy - (pts_3d[:, 1] * fy / pts_3d[:, 2])
        return np.stack([u, v], axis=1).astype(np.int32)
    else:
        cx = canvas_size / 2.0
        cy = canvas_size / 2.0
        radius = cloth_size / 2.0
        u = cx + pts_norm[:, 0] * radius
        v = cy - pts_norm[:, 1] * radius
        return np.stack([u, v], axis=1).astype(np.int32)


def draw_wireframe_grid(img: np.ndarray, uvs: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255), thickness: int = 1):
    """Draws 21x21 quadrilateral grid lines."""
    grid_size = 21
    if len(uvs) != grid_size * grid_size:
        return
    for r in range(grid_size):
        for c in range(grid_size):
            idx = r * grid_size + c
            p1 = tuple(uvs[idx])
            if c < grid_size - 1:
                p2 = tuple(uvs[idx + 1])
                cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
            if r < grid_size - 1:
                p3 = tuple(uvs[idx + grid_size])
                cv2.line(img, p1, p3, color, thickness, cv2.LINE_AA)


def draw_edge_contour(img: np.ndarray, uvs: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255), thickness: int = 2):
    """Draws 80-point closed contour loop."""
    pts = uvs.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def compute_metrics(pred_norm: np.ndarray, gt_mesh: np.ndarray, corner_idx: List[int]) -> Dict[str, float]:
    """Calculates CD-L1, MSE, and Corner Error against GT mesh."""
    gt_norm = gt_mesh.copy()
    gt_norm -= gt_norm.mean(axis=0)
    sc = float(np.max(np.linalg.norm(gt_norm, axis=1)))
    if sc > 1e-6:
        gt_norm /= sc

    mse_val = float(np.mean((pred_norm - gt_norm) ** 2))
    diff = pred_norm[:, None, :] - gt_norm[None, :, :]
    d1 = np.linalg.norm(diff, axis=-1)
    cd_l1 = float(0.5 * (d1.min(axis=1).mean() + d1.min(axis=0).mean()))

    pred_corners = pred_norm[corner_idx]
    gt_corners = gt_norm[corner_idx]
    corner_err = float(np.linalg.norm(pred_corners - gt_corners, axis=1).mean())

    return {
        "mse": mse_val,
        "cd_l1": cd_l1,
        "corner_err": corner_err,
        "gt_norm": gt_norm,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main GUI Application
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="TRTM Dataset Inspection & Model Prediction Overlay GUI")
    parser.add_argument("--ckpt_dir", default=None, help="Path to checkpoint directory (optional)")
    parser.add_argument("--data_dir", default=None, help="Path to TRTM dataset or image folder")
    parser.add_argument("--split", choices=["train", "val", "real_test", "auto"], default="train", help="Dataset split to display (default: train)")
    parser.add_argument("--mode", choices=["full", "edge"], default=None, help="Model mode (default: full or auto-detect)")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of inference steps")
    parser.add_argument("--tta_rotations", type=int, default=1, help="TTA yaw rotations (1 = off, 4 = 0/90/180/270°)")
    parser.add_argument("--random_subset", type=int, default=None, help="Display a random subset of N samples")
    parser.add_argument("--max_samples", type=int, default=None, help="Cap total number of samples loaded")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle sample list")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # 1. Load Model Pipeline (if checkpoint provided)
    pipeline = None
    use_fm = False
    q_temp = None
    is_edge = False
    corner_idx = CORNER_IDX_FULL

    if args.ckpt_dir:
        print(f"[Model] Loading checkpoint: {args.ckpt_dir}")
        clean_ckpt_dir = os.path.normpath(args.ckpt_dir)
        # Check parent experiment directory for config.yml
        exp_dir = os.path.dirname(os.path.dirname(clean_ckpt_dir))
        config_path = os.path.join(exp_dir, "config.yml")
        if not os.path.isfile(config_path):
            config_path = os.path.join(clean_ckpt_dir, "config.yml")

        if not os.path.isfile(config_path):
            print(f"[Warning] config.yml not found in {exp_dir}. Model loading might fail.")
            config = None
        else:
            config = OmegaConf.load(config_path)

        if config:
            # Determine mode (edge vs full)
            if args.mode is not None:
                is_edge = (args.mode == "edge")
            else:
                patch_file = config.model_cfg.get("patch_file", "")
                is_edge = ("edge" in patch_file) or (config.dataset_cfg.get("train_mode", "") == "edge_only")

            corner_idx = CORNER_IDX_EDGE if is_edge else CORNER_IDX_FULL
            pipeline, use_fm = load_pipeline(config, args.ckpt_dir, device)
            q_temp = load_template(config, is_edge)
            print(f"[Model] Loaded successfully! Mode: {'EDGE (80)' if is_edge else 'FULL (441)'}, Use FM: {use_fm}")

            # If data_dir not specified, try config's dataset_cfg.data_dir
            if not args.data_dir and "data_dir" in config.dataset_cfg:
                cfg_data_dir = config.dataset_cfg.data_dir
                if cfg_data_dir:
                    candidates = [
                        cfg_data_dir,
                        "/mnt/Data/cjimenez/TRTM/datasets/template_square",
                        cfg_data_dir.replace("/home/cjimenez/projects/datasets/", "/mnt/Data/cjimenez/"),
                        cfg_data_dir.replace("/home/cjimenez/Projects/datasets/", "/mnt/Data/cjimenez/"),
                        cfg_data_dir.replace("/projects/", "/Projects/"),
                        cfg_data_dir.replace("/Projects/", "/projects/"),
                    ]
                    for c in candidates:
                        if os.path.exists(c):
                            args.data_dir = c
                            break

    # 2. Discover Dataset Samples
    samples = discover_trtm_samples(
        data_dir=args.data_dir,
        split=args.split,
        max_samples=args.max_samples,
        random_subset=args.random_subset,
        shuffle=args.shuffle,
    )

    if not samples:
        print("[Error] No samples found to display. Exiting.")
        return

    # 3. State Management
    current_idx = 0
    pcd_mode = "depth"  # 'depth' (backprojected from depth map) or 'mesh' (sampled from GT mesh)
    show_gt = True
    show_gt_left = False  # Toggle GT mesh overlay directly on Left Panel (depth map)
    show_wireframe = True
    show_color_left = False
    tta_rotations = args.tta_rotations
    save_feedback_timer = 0

    # In-memory prediction cache: (sample_idx, pcd_mode, tta, steps) -> (pred_norm, latency_ms, pcd_input)
    pred_cache: Dict[Tuple[int, str, int, int], Tuple[np.ndarray, float, np.ndarray]] = {}

    # 4. Window Setup
    window_name = "TRTM Dataset & Model Overlay GUI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1440, 780)

    def on_trackbar(pos):
        nonlocal current_idx
        current_idx = max(0, min(pos, len(samples) - 1))

    cv2.createTrackbar("Sample", window_name, 0, max(1, len(samples) - 1), on_trackbar)

    os.makedirs("trtm_eval_captures", exist_ok=True)

    print("\n" + "=" * 60)
    print("READY. Interactive Controls:")
    print("  [Left / Right or A / D] : Previous / Next Sample")
    print("  [M]                     : Toggle GT Mesh Overlay on Left (over Depth Map)")
    print("  [G]                     : Toggle GT Mesh Overlay on Right (over Prediction)")
    print("  [R]                     : Jump to Random Sample")
    print("  [P]                     : Toggle PCD Source (Depth Image vs. GT Mesh)")
    print("  [W]                     : Toggle Wireframe vs. Nodes Only")
    print("  [T]                     : Toggle TTA (1x vs. 4x)")
    print("  [C]                     : Toggle Left Panel (Depth vs. RGB Color)")
    print("  [S]                     : Save Comparison Screenshot")
    print("  [Q or ESC]              : Quit")
    print("=" * 60 + "\n")

    # 5. Event Loop
    while True:
        sample = samples[current_idx]
        cv2.setTrackbarPos("Sample", window_name, current_idx)

        # Load Depth and Color images
        depth_raw = sample.get_depth_image()
        color_raw = sample.get_color_image()
        gt_mesh = sample.get_gt_mesh()

        if depth_raw is None:
            depth_raw = np.full((TARGET_CANVAS_SIZE, TARGET_CANVAS_SIZE), 255, dtype=np.uint8)

        is_simu_sample = ("simu" in sample.split) or (sample.mesh_gt_path is not None and "simu_mesh" in sample.mesh_gt_path)

        # Compute baseline point cloud, center of mass, scale, and intrinsics for current sample
        pcd_input, cur_com, cur_scale, cur_intrinsics = depth_to_pcd_trtm(depth_raw, 400, is_simu=is_simu_sample)

        # ── LEFT PANEL: Dataset Input View ────────────────────────────────
        # Build Left Panel base
        if show_color_left and color_raw is not None:
            left_view = color_raw.copy()
            left_title = "DATASET: COLOR IMAGE"
        else:
            left_view = cv2.cvtColor(depth_raw, cv2.COLOR_GRAY2BGR)
            left_title = "DATASET: TRTM DEPTH MAP"

        # Draw green target guide box (480x480 center)
        cx_c = TARGET_CANVAS_SIZE // 2
        half_box = IDEAL_CLOTH_SIZE // 2
        p1 = (cx_c - half_box, cx_c - half_box)
        p2 = (cx_c + half_box, cx_c + half_box)
        cv2.rectangle(left_view, p1, p2, (0, 255, 0), 2)
        cv2.putText(left_view, "Target 480px", (p1[0], p1[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        # Pixel Stats
        cloth_pts = depth_raw[(depth_raw < 245) & (depth_raw > 15)]
        min_fold = int(cloth_pts.min()) if len(cloth_pts) else int(depth_raw.min())
        max_fold = int(cloth_pts.max()) if len(cloth_pts) else int(depth_raw.max())
        mean_val = float(cloth_pts.mean()) if len(cloth_pts) else float(depth_raw.mean())

        # Header for Left Panel
        cv2.putText(left_view, left_title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(left_view, f"Pixel Range: [{min_fold}, {max_fold}] | Mean: {mean_val:.1f}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(left_view, f"Neutral Table: ~{FLAT_PIXEL_VAL} (Inverted: Dark=Fold)", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

        # Optional GT Mesh overlay on Left Panel (toggled via [M])
        if show_gt_left and gt_mesh is not None:
            if ("simu" in sample.split) or (sample.mesh_gt_path and "simu_mesh" in sample.mesh_gt_path):
                # TRTM simulation mesh is metric world frame: +Y is forward/UP
                cx_c = TARGET_CANVAS_SIZE / 2.0
                cy_c = TARGET_CANVAS_SIZE / 2.0
                u_gt = cx_c + gt_mesh[:, 0] * SCALE_PX_SIMU
                v_gt = cy_c - gt_mesh[:, 1] * SCALE_PX_SIMU
                uvs_gt_left = np.stack([u_gt, v_gt], axis=1).astype(np.int32)
            else:
                gt_norm = gt_mesh.copy()
                gt_norm -= gt_norm.mean(axis=0)
                sc = float(np.max(np.linalg.norm(gt_norm, axis=1)))
                if sc > 1e-6:
                    gt_norm /= sc
                if sample.mesh_gt_path and "pred_mesh" in sample.mesh_gt_path:
                    gt_norm[:, 1] = -gt_norm[:, 1]
                uvs_gt_left = project_to_canvas(gt_norm, cur_com, cur_scale, cur_intrinsics)

            if show_wireframe and not is_edge:
                draw_wireframe_grid(left_view, uvs_gt_left, color=(0, 255, 0), thickness=1)
            elif show_wireframe and is_edge:
                draw_edge_contour(left_view, uvs_gt_left, color=(0, 255, 0), thickness=2)

            for i, c_i in enumerate(corner_idx):
                u_gt, v_gt = uvs_gt_left[c_i]
                c_col = CORNER_COLORS[i % len(CORNER_COLORS)]
                cv2.circle(left_view, (u_gt, v_gt), 6, c_col, -1)
                cv2.circle(left_view, (u_gt, v_gt), 8, (255, 255, 255), 1)
                cv2.putText(left_view, CORNER_NAMES[i], (u_gt + 9, v_gt - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_col, 2)

            lbl = "GT MESH OVERLAY: ON [M]" if ("simu_mesh" in (sample.mesh_gt_path or "")) else "PREV MODEL REF MESH (Not GT): ON [M]"
            cv2.putText(left_view, lbl, (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)
        elif show_gt_left and gt_mesh is None:
            cv2.putText(left_view, "NO MESH AVAILABLE [M]", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # ── RIGHT PANEL: Model Prediction & Overlay ───────────────────────
        # Default right panel background to depth map (matching orbbec_overlay_3d.py)
        # Press [C] to toggle between Depth and Color image
        if show_color_left and color_raw is not None:
            right_view = color_raw.copy()
        else:
            right_view = cv2.cvtColor(depth_raw, cv2.COLOR_GRAY2BGR)

        # Run or Retrieve Inference
        pred_norm = None
        t_inf = 0.0
        pcd_input = None
        cur_com = None
        cur_scale = None
        cur_intrinsics = None

        if pipeline is not None and q_temp is not None:
            cache_key = (current_idx, pcd_mode, tta_rotations, args.num_steps)
            if cache_key in pred_cache:
                pred_norm, t_inf, pcd_input, cur_com, cur_scale, cur_intrinsics = pred_cache[cache_key]
            else:
                # 1. Generate PCD
                if pcd_mode == "mesh" and gt_mesh is not None:
                    pcd_input, cur_com, cur_scale = mesh_to_pcd_simu(gt_mesh, 400)
                    cur_intrinsics = None
                else:
                    pcd_input, cur_com, cur_scale, cur_intrinsics = depth_to_pcd_trtm(depth_raw, 400, is_simu=is_simu_sample)

                # 2. Run Forward Pass
                t0 = time.time()
                pred_norm = run_inference(
                    pipeline, use_fm, pcd_input, q_temp, args.num_steps, device,
                    tta_rotations=tta_rotations
                )
                t_inf = (time.time() - t0) * 1000.0
                pred_cache[cache_key] = (pred_norm, t_inf, pcd_input, cur_com, cur_scale, cur_intrinsics)

        # Draw Overlays on Right Panel
        if pred_norm is not None:
            # 1. Draw input PCD points (gray/cyan small dots)
            uvs_pcd = project_to_canvas(pcd_input, cur_com, cur_scale, cur_intrinsics, is_ortho=is_simu_sample)
            for u, v in uvs_pcd:
                if 0 <= u < TARGET_CANVAS_SIZE and 0 <= v < TARGET_CANVAS_SIZE:
                    cv2.circle(right_view, (u, v), 1, (180, 180, 180), -1)

            # 2. Draw Reference / Ground Truth Mesh (if available & enabled)
            metrics_hud = None
            is_real_gt = ("simu" in sample.split) or (sample.mesh_gt_path and "simu_mesh" in sample.mesh_gt_path)
            ref_label = "GROUND TRUTH" if is_real_gt else "SAVED REF MESH"

            if gt_mesh is not None:
                metrics = compute_metrics(pred_norm, gt_mesh, corner_idx)
                metrics_hud = (
                    f"CD-L1: {metrics['cd_l1']:.4f} | "
                    f"MSE: {metrics['mse']:.4f} | "
                    f"Corner: {metrics['corner_err']:.4f}"
                )
                if show_gt:
                    if is_real_gt:
                        cx_c = TARGET_CANVAS_SIZE / 2.0
                        cy_c = TARGET_CANVAS_SIZE / 2.0
                        u_gt = cx_c + gt_mesh[:, 0] * SCALE_PX_SIMU
                        v_gt = cy_c - gt_mesh[:, 1] * SCALE_PX_SIMU
                        uvs_gt = np.stack([u_gt, v_gt], axis=1).astype(np.int32)
                    else:
                        gt_norm_proj = metrics["gt_norm"].copy()
                        if sample.mesh_gt_path and "pred_mesh" in sample.mesh_gt_path:
                            gt_norm_proj[:, 1] = -gt_norm_proj[:, 1]
                        uvs_gt = project_to_canvas(gt_norm_proj, cur_com, cur_scale, cur_intrinsics)
                    if show_wireframe and not is_edge:
                        draw_wireframe_grid(right_view, uvs_gt, color=(0, 255, 0), thickness=1)
                    elif show_wireframe and is_edge:
                        draw_edge_contour(right_view, uvs_gt, color=(0, 255, 0), thickness=2)

                    # Reference/GT corners (green rings)
                    for c_i in corner_idx:
                        u_gt, v_gt = uvs_gt[c_i]
                        cv2.circle(right_view, (u_gt, v_gt), 7, (0, 255, 0), 2)

            # 3. Draw Predicted Mesh (Current Checkpoint)
            if is_real_gt and gt_mesh is not None:
                com = gt_mesh.mean(axis=0)
                scale = float(np.max(np.linalg.norm(gt_mesh - com, axis=1)))
                pred_metric = pred_norm * scale + com
                cx_c = TARGET_CANVAS_SIZE / 2.0
                cy_c = TARGET_CANVAS_SIZE / 2.0
                u_p = cx_c + pred_metric[:, 0] * SCALE_PX_SIMU
                v_p = cy_c - pred_metric[:, 1] * SCALE_PX_SIMU
                uvs_pred = np.stack([u_p, v_p], axis=1).astype(np.int32)
            else:
                uvs_pred = project_to_canvas(pred_norm, cur_com, cur_scale, cur_intrinsics, is_ortho=is_simu_sample)
            if show_wireframe:
                if is_edge:
                    draw_edge_contour(right_view, uvs_pred, color=(0, 255, 255), thickness=2)
                else:
                    draw_wireframe_grid(right_view, uvs_pred, color=(0, 255, 255), thickness=1)

            # Draw nodes (blue dots)
            for u, v in uvs_pred:
                if 0 <= u < TARGET_CANVAS_SIZE and 0 <= v < TARGET_CANVAS_SIZE:
                    cv2.circle(right_view, (u, v), 2, (255, 100, 0), -1)

            # 4. Draw Predicted Corners (with distinct colors & labels)
            for i, c_i in enumerate(corner_idx):
                u_p, v_p = uvs_pred[c_i]
                c_col = CORNER_COLORS[i % len(CORNER_COLORS)]
                cv2.circle(right_view, (u_p, v_p), 6, c_col, -1)
                cv2.circle(right_view, (u_p, v_p), 8, (255, 255, 255), 1)
                cv2.putText(right_view, CORNER_NAMES[i], (u_p + 9, v_p - 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, c_col, 2)

            # Right Panel HUD Header
            cv2.putText(right_view, "MODEL PREDICTION (Yellow/Blue)", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.putText(right_view, f"Latency: {t_inf:.1f}ms | Steps: {args.num_steps} | TTA: {tta_rotations}x [T]",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if gt_mesh is not None:
                gt_status = f"{ref_label} (Green): {'ON' if show_gt else 'OFF'} [Press G to toggle]"
                cv2.putText(right_view, gt_status, (20, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0) if show_gt else (100, 100, 100), 1)

            pcd_source_str = "PCD: Depth Map [P]" if pcd_mode == "depth" else "PCD: GT Vertices [P]"
            cv2.putText(right_view, pcd_source_str, (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1)

            if metrics_hud:
                cv2.putText(right_view, metrics_hud, (20, 135),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1)

        else:
            # Standalone Dataset Mode (no checkpoint)
            if gt_mesh is not None and show_gt:
                if ("simu" in sample.split) or (sample.mesh_gt_path and "simu_mesh" in sample.mesh_gt_path):
                    cx_c = TARGET_CANVAS_SIZE / 2.0
                    cy_c = TARGET_CANVAS_SIZE / 2.0
                    u_gt = cx_c + gt_mesh[:, 0] * SCALE_PX_SIMU
                    v_gt = cy_c - gt_mesh[:, 1] * SCALE_PX_SIMU
                    uvs_gt = np.stack([u_gt, v_gt], axis=1).astype(np.int32)
                else:
                    gt_norm = gt_mesh.copy()
                    gt_norm -= gt_norm.mean(axis=0)
                    sc = float(np.max(np.linalg.norm(gt_norm, axis=1)))
                    if sc > 1e-6:
                        gt_norm /= sc
                    uvs_gt = project_to_canvas(gt_norm, cur_com, cur_scale, cur_intrinsics)
                if show_wireframe and not is_edge:
                    draw_wireframe_grid(right_view, uvs_gt, color=(0, 255, 0), thickness=1)
                elif show_wireframe and is_edge:
                    draw_edge_contour(right_view, uvs_gt, color=(0, 255, 0), thickness=2)
                for i, c_i in enumerate(corner_idx):
                    u_gt, v_gt = uvs_gt[c_i]
                    cv2.circle(right_view, (u_gt, v_gt), 7, (0, 255, 0), -1)
                    cv2.putText(right_view, CORNER_NAMES[i], (u_gt + 10, v_gt - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            cv2.putText(right_view, "DATASET GROUND TRUTH MESH", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(right_view, "(No --ckpt_dir specified: running in inspection mode)", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        # ── COMBINE PANELS HORIZONTALLY ───────────────────────────────────
        combined = np.hstack((left_view, right_view))

        # Top Information Bar
        top_bar = np.zeros((45, combined.shape[1], 3), dtype=np.uint8)
        if sample.mesh_gt_path and "simu_mesh" in sample.mesh_gt_path:
            mesh_status_str = "SIMU GT (Exact)"
        elif sample.mesh_gt_path and "pred_mesh" in sample.mesh_gt_path:
            mesh_status_str = "PREV MODEL BASELINE (Not GT!)"
        elif gt_mesh is not None:
            mesh_status_str = "YES"
        else:
            mesh_status_str = "NO"

        sample_title = (
            f"Sample [{current_idx + 1} / {len(samples)}]: {sample.sample_id}  |  "
            f"Split: {sample.split}  |  "
            f"Mesh: {mesh_status_str}"
        )
        cv2.putText(top_bar, sample_title, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        combined = np.vstack((top_bar, combined))

        # Bottom Controls Legend Bar
        bot_bar = np.zeros((35, combined.shape[1], 3), dtype=np.uint8)
        controls_str = (
            "[< / > or A / D]: Prev/Next  |  [M]: Left GT Mesh  |  [G]: Right GT Mesh  |  "
            "[W]: Wireframe  |  [P]: PCD Src  |  [R]: Random  |  [T]: TTA  |  [C]: Color  |  [S]: Save  |  [Q]: Quit"
        )
        cv2.putText(bot_bar, controls_str, (20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        combined = np.vstack((combined, bot_bar))

        # Save Feedback
        if save_feedback_timer > 0:
            cv2.putText(
                combined, "SAVED TO trtm_eval_captures/!",
                (combined.shape[1] // 2 - 250, combined.shape[0] // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3
            )
            save_feedback_timer -= 1

        # Display (scaled for typical 1080p laptop displays)
        display_img = cv2.resize(combined, (0, 0), fx=0.85, fy=0.85)
        cv2.imshow(window_name, display_img)

        # ── KEYBOARD HANDLING ─────────────────────────────────────────────
        raw_key = cv2.waitKeyEx(25)
        key = raw_key & 0xFF

        # Arrow keys detection:
        # Linux X11/GTK Left Arrow: 65361 (0xFF51)
        # Windows Left Arrow: 2424832
        # Mac/other Left Arrow: 2
        is_left = (raw_key in [65361, 0xFF51, 2424832, 2]) or (key in [ord('a'), ord('A')])
        is_right = (raw_key in [65363, 0xFF53, 2555904, 3]) or (key in [ord('d'), ord('D')])

        # IMPORTANT: Check navigation FIRST to avoid collision where Linux Left Arrow (81) matches ord('Q') (81)!
        if is_left:
            current_idx = (current_idx - 1) % len(samples)
        elif is_right:
            current_idx = (current_idx + 1) % len(samples)
        elif raw_key in [27, ord('q'), ord('Q')]:  # ESC or Q
            break
        elif key in [ord('m'), ord('M')]:  # Toggle GT mesh on left panel
            show_gt_left = not show_gt_left
            print(f"[View] Left Panel GT Mesh Overlay: {'ON' if show_gt_left else 'OFF'}")
        elif key in [ord('g'), ord('G')]:  # Toggle GT overlay on right panel
            show_gt = not show_gt
            print(f"[View] Right Panel GT Mesh Overlay: {'ON' if show_gt else 'OFF'}")
        elif key in [ord('r'), ord('R')]:  # Random sample
            current_idx = random.randint(0, len(samples) - 1)
        elif key in [ord('p'), ord('P')]:  # Toggle PCD mode
            pcd_mode = "mesh" if (pcd_mode == "depth" and gt_mesh is not None) else "depth"
            print(f"[Mode] PCD Source switched to: {pcd_mode.upper()}")
        elif key in [ord('w'), ord('W')]:  # Toggle Wireframe
            show_wireframe = not show_wireframe
            print(f"[View] Wireframe grid: {'ON' if show_wireframe else 'OFF'}")
        elif key in [ord('t'), ord('T')]:  # Toggle TTA
            tta_rotations = 4 if tta_rotations == 1 else 1
            print(f"[Inference] TTA rotations set to: {tta_rotations}x")
        elif key in [ord('c'), ord('C')]:  # Toggle Color
            if color_raw is not None:
                show_color_left = not show_color_left
                print(f"[View] Left panel set to: {'COLOR' if show_color_left else 'DEPTH'}")
            else:
                print("[View] No RGB color image available for this sample.")
        elif key in [ord('s'), ord('S')]:  # Save screenshot
            save_path = os.path.join(
                "trtm_eval_captures",
                f"eval_{sample.sample_id}_{sample.split.replace('/', '_')}.png"
            )
            cv2.imwrite(save_path, combined)
            print(f"[Save] Saved comparison screenshot to: {save_path}")
            save_feedback_timer = 5

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
