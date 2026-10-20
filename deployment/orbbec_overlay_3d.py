import cv2
import numpy as np
import os
import sys
import time
import argparse
import pickle
import importlib
import torch
from datetime import datetime
from typing import Optional, Tuple, List, Dict
from pyorbbecsdk import *
from scipy.spatial import cKDTree

# Ensure we can import from UniClothDiff
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from omegaconf import OmegaConf
from uniclothdiff.registry import build_scheduler
from uniclothdiff.pipelines.cloth_state_est_pipeline import ClothStateEstPipeline
from uniclothdiff.pipelines.cloth_state_est_fm_pipeline import ClothStateEstFMPipeline

from orbbec_depth_processor import (
    depth_image_to_pcd,
    normalise_pcd,
    subsample_pcd,
    OrbbecDepthProcessor,
    DEPTH_FX_DEFAULT, DEPTH_FY_DEFAULT, DEPTH_CX_DEFAULT, DEPTH_CY_DEFAULT
)

CORNER_LABELS = ["TL", "TR", "BL", "BR"]
CORNER_COLORS = [
    (0, 255, 0),    # Green: TL
    (0, 255, 255),  # Yellow: TR
    (0, 165, 255),  # Orange: BL
    (255, 0, 255),  # Magenta: BR
]


# --- CONFIGURATION ---
ESC_KEY = 27

# Arm masking: TGC (template-guided completion) only activates when at least
# this many pixels are flagged as arm. Guards against depth sensor noise at
# low offset values falsely triggering the correction.
MIN_ARM_PIXELS_FOR_TGC = 500

# TRTM Standards
TARGET_CANVAS_SIZE = 720
IDEAL_CLOTH_SIZE = 480
FLAT_PIXEL_VAL = 192
MANUAL_TABLE_DEPTH = None

# --- NEW TUNING ---
INVERT_SHADING = True       # High folds = Darker pixels
PIXEL_PER_MM = 2.0          # 1cm (10mm) depth = 20 pixels difference

# Calibration Defaults
DEFAULT_CROP_SIZE = 480     # Increased default crop based on your feedback
DEFAULT_THRESHOLD = 110
DEFAULT_OFFSET = 0
DEFAULT_MAX_DEPTH = 1300.0  # Maximum depth cutoff in mm (filters out floor)


def frame_to_bgr_image(frame: VideoFrame):
    width = frame.get_width()
    height = frame.get_height()
    data = np.asanyarray(frame.get_data())
    image = np.resize(data, (height, width, 3))
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def fill_depth_holes(depth_map):
    """Fills 0-value noise in ToF depth maps (Integer)"""
    holes = (depth_map == 0)
    kernel = np.ones((5, 5), np.uint8)
    filled_depth = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, kernel)
    result = depth_map.copy()
    result[holes] = filled_depth[holes]
    return result


def smooth_depth_map(depth_mm):
    """Converts integer mm to smooth float to fix quantization"""
    depth_f = depth_mm.astype(np.float32)
    depth_smooth = cv2.GaussianBlur(depth_f, (5, 5), 0)
    return depth_smooth


def crop_and_resize(img, center_xy, crop_size, target_size=720, bg_color=0):
    """Crops 'crop_size' around center, then RESIZES to 'target_size'."""
    h, w = img.shape[:2]
    cx, cy = center_xy
    is_rgb = len(img.shape) == 3
    half_crop = crop_size // 2

    x1 = cx - half_crop
    y1 = cy - half_crop
    x2 = x1 + crop_size
    y2 = y1 + crop_size

    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - w)
    pad_b = max(0, y2 - h)

    if any([pad_l, pad_t, pad_r, pad_b]):
        val = [bg_color] * 3 if is_rgb else bg_color
        border_type = cv2.BORDER_CONSTANT
        if is_rgb:
            img_padded = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, border_type, value=val)
        else:
            img_padded = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, border_type, value=val)
        x1 += pad_l
        y1 += pad_t
        x2 += pad_l
        y2 += pad_t
        crop = img_padded[y1:y2, x1:x2]
    else:
        crop = img[y1:y2, x1:x2]

    # Use INTER_AREA for RGB (shrink), INTER_CUBIC for depth (preserve values)
    method = cv2.INTER_AREA if is_rgb else cv2.INTER_CUBIC
    return cv2.resize(crop, (target_size, target_size), interpolation=method)


def apply_trtm_shading(depth_map_float, mask_bg, offset_mm=0, max_table_depth=1300.0):
    """Calculates TRTM values using Float Depth + Inverted Logic."""
    valid_pixels = depth_map_float[(depth_map_float > 0) & (depth_map_float < max_table_depth)]

    if MANUAL_TABLE_DEPTH:
        table_depth = MANUAL_TABLE_DEPTH
    elif len(valid_pixels) > 0:
        table_depth = np.percentile(valid_pixels, 98)
    else:
        fallback = depth_map_float[depth_map_float > 0]
        if len(fallback) > 0:
            table_depth = np.percentile(fallback, 98)
        else:
            h, w = depth_map_float.shape
            return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w, 3), dtype=np.uint8), 0

    table_depth += offset_mm

    # Calculate height above table
    height_map = table_depth - depth_map_float

    # --- INVERTED LOGIC ---
    if INVERT_SHADING:
        # Higher fold = darker pixel (subtract from base)
        processed = FLAT_PIXEL_VAL - (height_map * PIXEL_PER_MM)
    else:
        # Higher fold = brighter pixel
        processed = FLAT_PIXEL_VAL + (height_map * PIXEL_PER_MM)

    processed = np.clip(processed, 0, 255).astype(np.uint8)

    # Saving version (white background)
    encoded_save = processed.copy()
    encoded_save[mask_bg] = 255

    # Display version (black background for contrast)
    vis_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    vis_bgr[mask_bg] = [0, 0, 0]

    return encoded_save, vis_bgr, table_depth


def segment_cloth_threshold(image, threshold=30, depth_map=None, max_depth=1300.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if threshold <= 0:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    if depth_map is not None:
        binary[(depth_map > max_depth) & (depth_map > 0)] = 0
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(binary)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            cv2.drawContours(final_mask, [largest], -1, 255, -1)
    if depth_map is not None:
        final_mask[(depth_map > max_depth) & (depth_map > 0)] = 0
    return (final_mask // 255).astype(np.uint8)


def nothing(x):
    pass


def find_config_for_checkpoint(ckpt_dir: str) -> str:
    """Search for config.yml or config.yaml in the ckpt dir or its ancestors."""
    clean = os.path.normpath(ckpt_dir)
    search_dirs = [
        clean,
        os.path.dirname(clean),
        os.path.dirname(os.path.dirname(clean)),
    ]
    for d in search_dirs:
        for fname in ["config.yml", "config.yaml"]:
            p = os.path.join(d, fname)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError(f"Could not find config.yml or config.yaml in or above {ckpt_dir}")


def load_pipeline(config, checkpoint_dir: str, device: torch.device):
    """Load model weights and build the inference pipeline."""
    model_cls = getattr(
        importlib.import_module("uniclothdiff.models"), config.model_cfg.type
    )
    model = model_cls.from_pretrained(checkpoint_dir, subfolder="model")
    model.to(device, dtype=torch.float32).eval()

    diff_dict = OmegaConf.to_container(config.diffusion_cfg)
    scheduler = build_scheduler(diff_dict)

    use_fm      = "FlowMatching" in config.diffusion_cfg.type
    PipelineCls = ClothStateEstFMPipeline if use_fm else ClothStateEstPipeline
    pipeline    = PipelineCls(model=model, scheduler=scheduler)
    pipeline.to(device, dtype=torch.float32)
    pipeline.set_progress_bar_config(disable=True)

    return pipeline, use_fm


def load_template(config, is_edge: bool) -> np.ndarray:
    """Load the template mesh from the .pkl file with path fallback."""
    # Handle relative paths or paths from external servers
    patch_file = config.model_cfg.patch_file
    if not os.path.isabs(patch_file):
        patch_file = os.path.join(_REPO_ROOT, patch_file)
    if not os.path.isfile(patch_file):
        fallback = os.path.join(_REPO_ROOT, "assets", os.path.basename(patch_file))
        if os.path.isfile(fallback):
            patch_file = fallback

    with open(patch_file, "rb") as f:
        data = pickle.load(f)

    pts = data["points"].astype(np.float32)

    if is_edge:
        dataset_type = config.dataset_cfg.get("type", "TRTMDiffusionDataset")
        N = 21 if "TRTM" in dataset_type else 20
        grid     = np.arange(N * N).reshape(N, N)
        contour  = np.unique(np.concatenate([
            grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]
        ]))
        # Only extract contour if the template is a full grid.
        # If it already has the same number of points as the contour, it's already an edge template!
        if len(pts) == N * N:
            pts = pts[contour]
        elif len(pts) != len(contour):
            print(f"Warning: Template has {len(pts)} points, expected {N*N} or {len(contour)}.")

    pts -= pts.mean(axis=0)
    sc   = float(np.max(np.linalg.norm(pts, axis=1)))
    if sc > 1e-6:
        pts /= sc

    return pts


def load_model_package(ckpt_dir: str, device: torch.device, forced_mode: Optional[str] = None) -> dict:
    """
    Load model pipeline, template, and corner/wireframe metadata for a checkpoint.
    """
    clean_ckpt_dir = os.path.normpath(ckpt_dir)
    config_path = find_config_for_checkpoint(clean_ckpt_dir)
    config = OmegaConf.load(config_path)

    # Determine mode: full or edge
    train_mode = config.dataset_cfg.get("train_mode", "")
    exp_name = str(config.get("exp_name", ""))
    if forced_mode == "edge":
        is_edge = True
    elif forced_mode == "full":
        is_edge = False
    elif train_mode == "edge_only" or "edge" in os.path.basename(clean_ckpt_dir).lower() or "edge" in exp_name.lower():
        is_edge = True
    else:
        is_edge = False

    pipeline, use_fm = load_pipeline(config, clean_ckpt_dir, device)
    q_temp = load_template(config, is_edge)

    # Check if template size indicates edge mode
    if len(q_temp) in (76, 80):
        is_edge = True

    dataset_type = config.dataset_cfg.get("type", "TRTMDiffusionDataset")
    grid_size = 21 if "TRTM" in dataset_type else 20

    # Corners: [TL, TR, BL, BR] in camera screen coordinate space
    # N=21: TL=0 (edge: 0), TR=20 (edge: 20), BL=420 (edge: 59), BR=440 (edge: 79)
    # N=20: TL=0 (edge: 0), TR=19 (edge: 19), BL=380 (edge: 56), BR=399 (edge: 75)
    if is_edge:
        corner_idx = [0, 20, 59, 79] if grid_size == 21 else [0, 19, 56, 75]
        # Continuous perimeter loop order for closed contour polyline
        N = grid_size
        grid = np.arange(N * N).reshape(N, N)
        loop_full = np.concatenate([
            grid[0, :],
            grid[1:-1, -1],
            grid[-1, ::-1],
            grid[-2:0:-1, 0]
        ])
        contour_sorted = np.unique(np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]]))
        c_map = {full_idx: edge_idx for edge_idx, full_idx in enumerate(contour_sorted)}
        loop_edge_order = np.array([c_map[idx] for idx in loop_full])
    else:
        corner_idx = [0, 20, 420, 440] if grid_size == 21 else [0, 19, 380, 399]
        loop_edge_order = None

    model_title = f"{'Edge Mode' if is_edge else 'Full Mode'} ({len(q_temp)} pts)"
    exp_title = config.get("exp_name", os.path.basename(os.path.dirname(config_path)))

    print(f"[{model_title}] Loaded from {ckpt_dir}. Exp: {exp_title}, Corners: {corner_idx}")

    return {
        "pipeline": pipeline,
        "use_fm": use_fm,
        "q_temp": q_temp,
        "is_edge": is_edge,
        "corner_idx": corner_idx,
        "loop_edge_order": loop_edge_order,
        "grid_size": grid_size,
        "model_title": model_title,
        "exp_title": exp_title,
        "config": config,
    }


@torch.no_grad()
def run_inference(pipeline, use_fm, pcd_np, q_temp_np, num_steps, device):
    """Run one forward pass of the cloth state estimation pipeline."""
    pcd    = torch.tensor(pcd_np[None],    dtype=torch.float32, device=device)
    q_temp = torch.tensor(q_temp_np[None], dtype=torch.float32, device=device)
    N      = q_temp.shape[1]

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


def _yaw_matrix(theta: float) -> np.ndarray:
    """Return a 3×3 Z-axis rotation matrix for angle theta (radians)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)


def _chamfer_score(pred: np.ndarray, pcd: np.ndarray) -> float:
    """
    One-sided Chamfer distance pred → pcd. Lower = better self-consistency.
    Uses vectorised broadcasting; fast for small meshes (<500 pts).
    """
    diff = pred[:, None, :] - pcd[None, :, :]   # [N, P, 3]
    d2   = (diff ** 2).sum(axis=-1)              # [N, P]
    return float(d2.min(axis=1).mean())


@torch.no_grad()
def run_inference_with_tta(pipeline, use_fm, pcd_np, q_temp_np, num_steps, device,
                           tta_rotations: int = 1):
    """
    Test-Time Augmentation (TTA) inference.

    Rotates the PCD+template by `tta_rotations` evenly-spaced yaw angles,
    runs inference on each, rotates predictions back, and returns the one
    with the lowest self-supervised Chamfer score against the original PCD.

    tta_rotations=1  → plain single forward pass (no TTA overhead).
    tta_rotations=4  → 0/90/180/270°  (recommended for real-robot use).
    tta_rotations=8  → every 45°.
    """
    if tta_rotations <= 1:
        return run_inference(pipeline, use_fm, pcd_np, q_temp_np, num_steps, device)

    angles = [2 * np.pi * k / tta_rotations for k in range(tta_rotations)]
    best_pred, best_score = None, float("inf")

    for theta in angles:
        R  = _yaw_matrix(theta)
        Rt = R.T
        pcd_rot    = pcd_np    @ R
        q_temp_rot = q_temp_np @ R

        pcd_t    = torch.tensor(pcd_rot[None],    dtype=torch.float32, device=device)
        q_temp_t = torch.tensor(q_temp_rot[None], dtype=torch.float32, device=device)
        N        = q_temp_t.shape[1]

        out      = pipeline(
            encoder_hidden_states=pcd_t,
            q_temp=q_temp_t,
            shape=(1, N, 3),
            num_inference_steps=num_steps,
            do_classifier_free_guidance=False,
            call_v2=True,
        )
        pred_rot  = out.frames if isinstance(out.frames, np.ndarray) else out.frames.cpu().numpy()
        pred_orig = pred_rot[0] @ Rt

        score = _chamfer_score(pred_orig, pcd_np)
        if score < best_score:
            best_score = score
    return best_pred


def project_to_2d(pts_3d_mm, fx, fy, cx, cy):
    """Project 3D points in camera frame back to 2D image coordinates."""
    xs = (pts_3d_mm[:, 0] * fx / np.clip(pts_3d_mm[:, 2], 1.0, None)) + cx
    ys = (pts_3d_mm[:, 1] * fy / np.clip(pts_3d_mm[:, 2], 1.0, None)) + cy
    return np.stack([xs, ys], axis=1).astype(np.int32)


def draw_grid_wireframe(img: np.ndarray, uvs: np.ndarray, grid_size: int = 21, color=(0, 230, 255), thickness: int = 1):
    """Draws quadrilateral grid wireframe for N x N vertices on canvas."""
    if len(uvs) != grid_size * grid_size:
        return
    for r in range(0, grid_size):
        row_pts = np.ascontiguousarray(uvs[r * grid_size : (r + 1) * grid_size], dtype=np.int32)
        cv2.polylines(img, [row_pts.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    for c in range(0, grid_size):
        col_pts = np.ascontiguousarray(uvs[c :: grid_size], dtype=np.int32)
        cv2.polylines(img, [col_pts.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_contour_wireframe(img: np.ndarray, uvs: np.ndarray, loop_order: Optional[np.ndarray], color=(0, 255, 255), thickness: int = 2):
    """Draws a continuous closed polyline connecting perimeter vertices in order."""
    if loop_order is not None and len(uvs) == len(loop_order):
        contour_pts = np.ascontiguousarray(uvs[loop_order], dtype=np.int32)
        cv2.polylines(img, [contour_pts.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    else:
        contour_pts = np.ascontiguousarray(uvs, dtype=np.int32)
        cv2.polylines(img, [contour_pts.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def render_model_view(
    base_color_img: np.ndarray,
    pred_mm: Optional[np.ndarray],
    model_pkg: dict,
    cam_intrinsics: tuple,
    arm_mask_2d: Optional[np.ndarray],
    prev_pred_mm: Optional[np.ndarray],
    crop_center: tuple,
    crop_size_val: int,
    t_inf: float = 0.0,
) -> np.ndarray:
    """
    Renders 3D prediction onto a copy of the base color image, projects to 2D,
    applies wireframe/contour, node dots, TGC corners, labels, crops to 720x720,
    and draws HUD info.
    """
    cam_fx, cam_fy, cam_cx, cam_cy = cam_intrinsics
    view_raw = base_color_img.copy()
    h, w = view_raw.shape[:2]

    is_edge = model_pkg["is_edge"]
    q_temp = model_pkg["q_temp"]
    corner_idx = model_pkg["corner_idx"]
    loop_edge_order = model_pkg["loop_edge_order"]
    grid_size = model_pkg["grid_size"]
    model_title = model_pkg["model_title"]

    tgc_active = False
    occluded = []
    n_arm_px = int(arm_mask_2d.sum()) if arm_mask_2d is not None else 0

    if pred_mm is not None:
        uvs = project_to_2d(pred_mm, cam_fx, cam_fy, cam_cx, cam_cy)

        # 1. Wireframe / Contour
        if is_edge:
            draw_contour_wireframe(view_raw, uvs, loop_edge_order, color=(0, 255, 255), thickness=2)
            # Node dots
            for u, v in uvs:
                cv2.circle(view_raw, (u, v), 3, (0, 200, 255), -1)
        else:
            draw_grid_wireframe(view_raw, uvs, grid_size=grid_size, color=(0, 230, 255), thickness=1)
            # Node dots
            for u, v in uvs:
                cv2.circle(view_raw, (u, v), 2, (255, 120, 0), -1)

        # 2. TGC corner completion
        corners_mm_raw = pred_mm[corner_idx]   # [4, 3] in order [TL, TR, BL, BR]
        corners_mm_tgc = corners_mm_raw.copy()

        if arm_mask_2d is not None and n_arm_px >= MIN_ARM_PIXELS_FOR_TGC:
            if prev_pred_mm is not None:
                uvs_c = project_to_2d(prev_pred_mm[corner_idx], cam_fx, cam_fy, cam_cx, cam_cy)
            else:
                uvs_c = project_to_2d(corners_mm_raw, cam_fx, cam_fy, cam_cx, cam_cy)

            visible, occluded = [], []
            for i, (u, v) in enumerate(uvs_c):
                is_arm = False
                for du in range(-5, 6):
                    for dv in range(-5, 6):
                        nu, nv = u + du, v + dv
                        if (0 <= nu < w) and (0 <= nv < h) and arm_mask_2d[nv, nu] > 0:
                            is_arm = True
                            break
                    if is_arm: break
                (occluded if is_arm else visible).append(i)

            if occluded and len(visible) >= 3:
                tgc_active = True
                temp_corners = q_temp[corner_idx].astype(np.float32)
                src = temp_corners[visible]
                dst = corners_mm_raw[visible].astype(np.float32)
                ones = np.ones((len(visible), 1), np.float32)
                src_h = np.hstack([src, ones])
                Mt, _, _, _ = np.linalg.lstsq(src_h, dst, rcond=None)
                for i in occluded:
                    tc_h = np.append(temp_corners[i], 1.0)
                    corners_mm_tgc[i] = tc_h @ Mt

        # 3. Draw corners
        uvs_corners_raw = project_to_2d(corners_mm_raw, cam_fx, cam_fy, cam_cx, cam_cy)
        uvs_corners_tgc = project_to_2d(corners_mm_tgc, cam_fx, cam_fy, cam_cx, cam_cy)

        for i, lbl in enumerate(CORNER_LABELS):
            u_r, v_r = uvs_corners_raw[i]
            u_t, v_t = uvs_corners_tgc[i]
            is_corrected = tgc_active and (i in occluded)
            col = CORNER_COLORS[i]

            # Raw corner
            cv2.circle(view_raw, (u_r, v_r), 5, (0, 0, 255) if is_corrected else col, -1)
            if is_corrected:
                # TGC corner
                cv2.circle(view_raw, (u_t, v_t), 8, (0, 255, 0), -1)
                cv2.circle(view_raw, (u_t, v_t), 9, (255, 255, 255), 1)
                cv2.putText(view_raw, f"{lbl} (TGC)", (u_t + 8, v_t - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.circle(view_raw, (u_r, v_r), 6, (255, 255, 255), 1)
                cv2.putText(view_raw, lbl, (u_r + 8, v_r - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    # Crop and resize to target canvas (720x720)
    final_view = crop_and_resize(view_raw, crop_center, crop_size_val, TARGET_CANVAS_SIZE, bg_color=0)

    # HUD Overlay
    cv2.putText(final_view, model_title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    if pred_mm is not None:
        cv2.putText(final_view, f"Pred: {t_inf:.1f} ms", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)
        arm_hud = f"Arm px: {n_arm_px}  (min={MIN_ARM_PIXELS_FOR_TGC})"
        arm_hud_color = (0, 255, 0) if n_arm_px >= MIN_ARM_PIXELS_FOR_TGC else (0, 140, 255)
        cv2.putText(final_view, arm_hud, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, arm_hud_color, 1)
        tgc_txt = f"TGC: {'ON (' + str(len(occluded)) + ' corr)' if tgc_active else 'off'}"
        cv2.putText(final_view, tgc_txt, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if tgc_active else (140, 140, 140), 1)
    else:
        cv2.putText(final_view, "No Cloth Detected", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    return final_view


def main():
    parser = argparse.ArgumentParser(description="Orbbec calibration & live inference GUI")
    parser.add_argument("--ckpt_dir", default=None, help="Path to checkpoint directory (legacy single-model argument)")
    parser.add_argument("--mode", choices=["edge", "full"], default=None, help="Model mode for --ckpt_dir ('edge' or 'full')")
    parser.add_argument("--ckpt_dir_full", "--ckpt_full", default=None, help="Path to full-mode checkpoint directory (441 points)")
    parser.add_argument("--ckpt_dir_edge", "--ckpt_edge", default=None, help="Path to edge-mode checkpoint directory (80 points)")
    parser.add_argument("--disp_scale", type=float, default=None, help="Display window scaling (default: 0.55 for 3 screens, 0.8 for 2 screens)")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of inference steps")
    parser.add_argument("--ema_alpha", type=float, default=0.3, help="EMA smoothing factor (0.0 to 1.0, lower is smoother, 1.0 is no smoothing)")
    parser.add_argument("--disable_plane", action="store_true", help="Disable RANSAC table plane calibration (falls back to flat table)")
    parser.add_argument("--disable_pruning", action="store_true", help="Disable outlier height pruning")
    parser.add_argument(
        "--tta_rotations", type=int, default=1,
        help=(
            "Test-Time Augmentation: number of yaw rotations per frame. "
            "1 = disabled (default). 4 = 0/90/180/270°. 8 = every 45°."
        )
    )
    parser.add_argument("--max_depth", type=float, default=DEFAULT_MAX_DEPTH, help="Maximum depth cutoff in mm (filters out floor)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Resolve full and edge checkpoints
    full_ckpt = args.ckpt_dir_full
    edge_ckpt = args.ckpt_dir_edge

    if args.ckpt_dir:
        if args.mode == "edge":
            if edge_ckpt is None:
                edge_ckpt = args.ckpt_dir
        elif args.mode == "full":
            if full_ckpt is None:
                full_ckpt = args.ckpt_dir
        else:
            try:
                cfg_p = find_config_for_checkpoint(args.ckpt_dir)
                cfg = OmegaConf.load(cfg_p)
                tm = cfg.dataset_cfg.get("train_mode", "")
                if tm == "edge_only" or "edge" in os.path.basename(os.path.normpath(args.ckpt_dir)).lower():
                    if edge_ckpt is None:
                        edge_ckpt = args.ckpt_dir
                else:
                    if full_ckpt is None:
                        full_ckpt = args.ckpt_dir
            except Exception:
                if full_ckpt is None:
                    full_ckpt = args.ckpt_dir

    model_full_pkg = None
    if full_ckpt:
        print(f"\n[Loading Full Model] {full_ckpt}")
        model_full_pkg = load_model_package(full_ckpt, device, forced_mode="full")

    model_edge_pkg = None
    if edge_ckpt:
        print(f"\n[Loading Edge Model] {edge_ckpt}")
        model_edge_pkg = load_model_package(edge_ckpt, device, forced_mode="edge")

    has_full = model_full_pkg is not None
    has_edge = model_edge_pkg is not None
    has_any_model = has_full or has_edge

    if args.disp_scale is not None:
        scale_disp = args.disp_scale
    else:
        scale_disp = 0.55 if (has_full and has_edge) else 0.8

    if has_full and has_edge:
        print("Dual-model mode active: 3 screens (Depth | Full Mesh | Edge Contour).")
    elif has_full:
        print("Single-model mode active: 2 screens (Depth | Full Mesh).")
    elif has_edge:
        print("Single-model mode active: 2 screens (Depth | Edge Contour).")
    else:
        print("Calibration mode active: 2 screens (Color | Depth).")

    pipeline_sdk = None
    for attempt in range(1, 8):
        try:
            pipeline_sdk = Pipeline()
            break
        except Exception as e:
            print(f"[Orbbec] Camera connection attempt {attempt}/7 ({e}). Retrying in 1.5s...", flush=True)
            time.sleep(1.5)
    if pipeline_sdk is None:
        raise RuntimeError("Could not connect to Orbbec device after 7 attempts.")
    config_sdk = Config()

    try:
        color_profiles = pipeline_sdk.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = None

        for w, h in [(1280, 720), (640, 480)]:
            try:
                color_profile = color_profiles.get_video_stream_profile(w, h, OBFormat.RGB, 30)
                break
            except Exception:
                continue

        if not color_profile:
            for i in range(color_profiles.count()):
                p = color_profiles.get_video_stream_profile(i)
                if p.get_format() == OBFormat.RGB:
                    color_profile = p
                    break

        if not color_profile:
            print("No suitable color profile found.")
            return

        config_sdk.enable_stream(color_profile)
        depth_profiles = pipeline_sdk.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_default_video_stream_profile()
        config_sdk.enable_stream(depth_profile)
        config_sdk.set_align_mode(OBAlignMode.DISABLE)
        pipeline_sdk.start(config_sdk)
        
        # State for temporal smoothing (independent per model)
        prev_pred_mm_full = None
        prev_pred_mm_edge = None
        arm_mask_2d = None   # updated each frame when Arm Offset > 0
        align_filter = AlignFilter(align_to_stream=OBStreamType.DEPTH_STREAM)
        
        # Read true intrinsics for accurate 3D projection
        try:
            intrinsics = depth_profile.get_intrinsic()
            cam_fx = float(intrinsics.fx)
            cam_fy = float(intrinsics.fy)
            cam_cx = float(intrinsics.cx)
            cam_cy = float(intrinsics.cy)
            print(f"Using Live Intrinsics: fx={cam_fx:.1f}, fy={cam_fy:.1f}, cx={cam_cx:.1f}, cy={cam_cy:.1f}")
        except Exception:
            cam_fx, cam_fy, cam_cx, cam_cy = DEPTH_FX_DEFAULT, DEPTH_FY_DEFAULT, DEPTH_CX_DEFAULT, DEPTH_CY_DEFAULT
            print("Could not read intrinsics. Using defaults.")

        processor = OrbbecDepthProcessor(
            fx=cam_fx, fy=cam_fy, cx=cam_cx, cy=cam_cy,
            use_plane_correction=not args.disable_plane,
            use_height_pruning=not args.disable_pruning,
            max_depth_mm=args.max_depth,
        )

    except Exception as e:
        print(f"Init Error: {e}")
        return

    # --- UI SETUP ---
    cv2.namedWindow("Master Calibration", cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Zoom (Crop Size)", "Master Calibration", DEFAULT_CROP_SIZE, 800, nothing)
    cv2.createTrackbar("Threshold (0=Auto)", "Master Calibration", DEFAULT_THRESHOLD, 255, nothing)
    cv2.createTrackbar("Table Offset",     "Master Calibration", DEFAULT_OFFSET,    100, nothing)
    cv2.createTrackbar("Arm Offset (mm)",  "Master Calibration", 0,                 300, nothing)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f"trtm_data_calibrated_{timestamp}"
    os.makedirs(save_folder, exist_ok=True)
    frame_count = 0
    save_feedback_timer = 0

    print("\nReady.")
    print("1. Adjust 'Zoom' until cloth fits the GREEN BOX.")
    print("2. Press 'S' to save.")
    print("3. Press 'Q' or ESC to exit.")

    try:
        empty_frames_count = 0
        while True:
            frames = pipeline_sdk.wait_for_frames(100)
            if not frames:
                empty_frames_count += 1
                if empty_frames_count > 50:
                    print("Camera disconnected or unresponsive. Exiting.")
                    break
                cv2.waitKey(1)
                continue
            
            frames = align_filter.process(frames)
            if not frames:
                empty_frames_count += 1
                if empty_frames_count > 50:
                    print("Camera disconnected or unresponsive. Exiting.")
                    break
                cv2.waitKey(1)
                continue
            
            empty_frames_count = 0
            frames = frames.as_frame_set()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            width = depth_frame.get_width()
            height = depth_frame.get_height()
            depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
            depth_raw_mm = (depth_data * depth_frame.get_depth_scale()).astype(np.uint16)
            color_image = frame_to_bgr_image(color_frame)
            if color_image is None:
                continue

            # --- PROCESS ---
            crop_size_val = cv2.getTrackbarPos("Zoom (Crop Size)", "Master Calibration")
            thresh_val    = cv2.getTrackbarPos("Threshold (0=Auto)", "Master Calibration")
            offset_val    = cv2.getTrackbarPos("Table Offset",     "Master Calibration")
            arm_offset_val = cv2.getTrackbarPos("Arm Offset (mm)", "Master Calibration")
            if crop_size_val < 100:
                crop_size_val = 100

            # 1. Depth processing pipeline
            depth_filled = fill_depth_holes(depth_raw_mm)
            depth_smooth = smooth_depth_map(depth_filled)  # Float smoothing

            # 2. Cloth segmentation and dynamic centre (with automatic floor depth gating)
            binary_mask = segment_cloth_threshold(
                color_image, threshold=thresh_val, depth_map=depth_smooth, max_depth=args.max_depth
            )
            background_mask = (binary_mask == 0)
            M = cv2.moments(binary_mask)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                crop_center = (cx, cy)
            else:
                crop_center = (width // 2, height // 2)

            masked_color = cv2.bitwise_and(color_image, color_image, mask=binary_mask)
            masked_smooth = depth_smooth * binary_mask      # Masked smoothed depth (float)
            masked_raw = cv2.bitwise_and(depth_filled, depth_filled, mask=binary_mask)

            # 3b. Robot arm masking + depth inpainting (enabled when Arm Offset > 0)
            arm_mask_2d = np.zeros((height, width), dtype=np.uint8)   # reset each frame
            prev_mesh_for_arm = prev_pred_mm_full if prev_pred_mm_full is not None else prev_pred_mm_edge
            if arm_offset_val > 0 and np.any(binary_mask > 0):
                arm_candidate = np.zeros((height, width), dtype=bool)
                
                if prev_mesh_for_arm is not None:
                    # Strategy 1: Temporal Mesh Masking (using 3D KDTree)
                    ys, xs = np.where(depth_smooth > 0)
                    if len(ys) > 0:
                        zs = depth_smooth[ys, xs].astype(np.float32)
                        X = (xs.astype(np.float32) - cam_cx) / cam_fx * zs
                        Y = (ys.astype(np.float32) - cam_cy) / cam_fy * zs
                        pts_3d = np.stack([X, Y, zs], axis=1)
                        
                        tree = cKDTree(prev_mesh_for_arm)
                        _, idx = tree.query(pts_3d)
                        nearest_mesh_z = prev_mesh_for_arm[idx, 2]
                        nearest_mesh_X = prev_mesh_for_arm[idx, 0]
                        nearest_mesh_Y = prev_mesh_for_arm[idx, 1]
                        dist_xy_sq = (X - nearest_mesh_X)**2 + (Y - nearest_mesh_Y)**2
                        
                        arm_pts_mask = (zs < nearest_mesh_z - arm_offset_val) & (dist_xy_sq < 10000)
                        arm_candidate[ys[arm_pts_mask], xs[arm_pts_mask]] = True
                else:
                    # Fallback: Estimate cloth surface depth using 40th percentile
                    cloth_depths_valid = depth_smooth[binary_mask > 0]
                    if len(cloth_depths_valid) > 100:
                        cloth_surface_z = float(np.percentile(cloth_depths_valid, 40))
                        arm_mask_bool = (
                            (depth_smooth > 0) &
                            (depth_smooth < cloth_surface_z - arm_offset_val) &
                            (binary_mask > 0)
                        )
                        arm_candidate[arm_mask_bool] = True
                
                if np.any(arm_candidate):
                    arm_mask_2d = arm_candidate.astype(np.uint8)
                    binary_mask[arm_candidate] = 0  # Erase arm from cloth mask
                    masked_smooth = depth_smooth * binary_mask
                    masked_color[arm_candidate] = 0

            # 3. TRTM Inverted Shading
            trtm_save_gray, trtm_vis_bgr, table_z = apply_trtm_shading(
                masked_smooth, background_mask, offset_mm=offset_val, max_table_depth=args.max_depth
            )

            # 4. Generate Views
            final_trtm_view = crop_and_resize(
                trtm_vis_bgr, crop_center, crop_size_val, TARGET_CANVAS_SIZE, bg_color=0
            )
            final_color_view = crop_and_resize(
                masked_color, crop_center, crop_size_val, TARGET_CANVAS_SIZE, bg_color=0
            )

            # Target region guide box on Depth View
            center_px = TARGET_CANVAS_SIZE // 2
            half_ideal = IDEAL_CLOTH_SIZE // 2
            p1 = (center_px - half_ideal, center_px - half_ideal)
            p2 = (center_px + half_ideal, center_px + half_ideal)
            cv2.rectangle(final_trtm_view, p1, p2, (0, 255, 0), 2)
            cv2.putText(
                final_trtm_view, "Target 480px",
                (p1[0], p1[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1
            )
            cv2.putText(
                final_trtm_view, "Calibrated Depth (TRTM)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
            )

            final_full_view = None
            final_edge_view = None

            # 5. Live Inference (if any model is loaded)
            if has_any_model:
                pts_3d = depth_image_to_pcd(masked_smooth, binary_mask, cam_fx, cam_fy, cam_cx, cam_cy)
                
                pred_mm_full = None
                t_inf_full = 0.0
                pred_mm_edge = None
                t_inf_edge = 0.0

                if len(pts_3d) > 0:
                    # Fit Table Plane
                    A, B, C = processor.fit_table_plane(depth_smooth, background_mask, pts_3d)
                    table_plane = (A, B, C)

                    # Convert Z from depth to height coordinates
                    pts_3d_height = pts_3d.copy()
                    z_table = A * pts_3d[:, 0] + B * pts_3d[:, 1] + C
                    pts_3d_height[:, 2] = z_table - pts_3d[:, 2]

                    # Prune outliers (heights < -10mm or heights > 80mm)
                    mask_valid = (pts_3d_height[:, 2] >= -10.0) & (pts_3d_height[:, 2] <= 80.0)
                    if np.any(mask_valid):
                        pts_3d_height = pts_3d_height[mask_valid]
                        pts_3d = pts_3d[mask_valid]

                    pts_norm, com, scale = normalise_pcd(pts_3d_height)
                    pcd_input = subsample_pcd(pts_norm, 400)

                    # A. Full Model Forward Pass
                    if model_full_pkg is not None:
                        t0 = time.time()
                        pred_norm_f = run_inference_with_tta(
                            model_full_pkg["pipeline"], model_full_pkg["use_fm"],
                            pcd_input, model_full_pkg["q_temp"], args.num_steps, device,
                            tta_rotations=args.tta_rotations,
                        )
                        t_inf_full = (time.time() - t0) * 1000.0
                        pred_mm_f = OrbbecDepthProcessor.denormalise_prediction(
                            pred_norm_f, com, scale, table_plane=table_plane
                        )
                        if prev_pred_mm_full is None:
                            prev_pred_mm_full = pred_mm_f
                        else:
                            pred_mm_f = args.ema_alpha * pred_mm_f + (1.0 - args.ema_alpha) * prev_pred_mm_full
                            prev_pred_mm_full = pred_mm_f
                        pred_mm_full = pred_mm_f

                    # B. Edge Model Forward Pass
                    if model_edge_pkg is not None:
                        t0 = time.time()
                        pred_norm_e = run_inference_with_tta(
                            model_edge_pkg["pipeline"], model_edge_pkg["use_fm"],
                            pcd_input, model_edge_pkg["q_temp"], args.num_steps, device,
                            tta_rotations=args.tta_rotations,
                        )
                        t_inf_edge = (time.time() - t0) * 1000.0
                        pred_mm_e = OrbbecDepthProcessor.denormalise_prediction(
                            pred_norm_e, com, scale, table_plane=table_plane
                        )
                        if prev_pred_mm_edge is None:
                            prev_pred_mm_edge = pred_mm_e
                        else:
                            pred_mm_e = args.ema_alpha * pred_mm_e + (1.0 - args.ema_alpha) * prev_pred_mm_edge
                            prev_pred_mm_edge = pred_mm_e
                        pred_mm_edge = pred_mm_e

                cam_intr = (cam_fx, cam_fy, cam_cx, cam_cy)
                if model_full_pkg is not None:
                    final_full_view = render_model_view(
                        masked_color, pred_mm_full, model_full_pkg, cam_intr,
                        arm_mask_2d, prev_pred_mm_full, crop_center, crop_size_val, t_inf=t_inf_full
                    )

                if model_edge_pkg is not None:
                    final_edge_view = render_model_view(
                        masked_color, pred_mm_edge, model_edge_pkg, cam_intr,
                        arm_mask_2d, prev_pred_mm_edge, crop_center, crop_size_val, t_inf=t_inf_edge
                    )

                # Assemble horizontal display
                if has_full and has_edge:
                    # 3 screens: Depth | Full Mode | Edge Mode
                    combined = np.hstack((final_trtm_view, final_full_view, final_edge_view))
                elif has_full:
                    # 2 screens: Depth | Full Mode
                    combined = np.hstack((final_trtm_view, final_full_view))
                elif has_edge:
                    # 2 screens: Depth | Edge Mode
                    combined = np.hstack((final_trtm_view, final_edge_view))
            else:
                # Original 2-window behavior (Color + TRTM Depth)
                combined = np.hstack((final_color_view, final_trtm_view))
            
            # Common UI overlays on top-left of the first window
            cv2.putText(
                combined,
                f"Z: {table_z/10:.1f}cm | Crop: {crop_size_val} | Off: {offset_val}mm",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
            )

            if save_feedback_timer > 0:
                cv2.putText(
                    combined, "SAVED!",
                    (combined.shape[1] // 2 - 100, combined.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4
                )
                save_feedback_timer -= 1

            # Scale the final image so it fits on screen comfortably
            display_img = cv2.resize(combined, (0, 0), fx=scale_disp, fy=scale_disp)
            cv2.imshow("Master Calibration", display_img)

            # --- SAVE ---
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s') or key == ord('S'):
                # TRTM (Inverted, Smooth) → White BG
                trtm_save_final = crop_and_resize(
                    trtm_save_gray, crop_center, crop_size_val, TARGET_CANVAS_SIZE, bg_color=255
                )
                # Raw (Integer) → Black BG
                raw_save_final = crop_and_resize(
                    masked_raw, crop_center, crop_size_val, TARGET_CANVAS_SIZE, bg_color=0
                )
                # Color → Black BG
                color_save_final = crop_and_resize(
                    masked_color, crop_center, crop_size_val, TARGET_CANVAS_SIZE, bg_color=0
                )

                cv2.imwrite(os.path.join(save_folder, f"color_{frame_count:05d}.png"), color_save_final)
                cv2.imwrite(os.path.join(save_folder, f"depth_raw_{frame_count:05d}.png"), raw_save_final)
                cv2.imwrite(os.path.join(save_folder, f"depth_trtm_{frame_count:05d}.png"), trtm_save_final)
                if final_full_view is not None:
                    cv2.imwrite(os.path.join(save_folder, f"pred_full_{frame_count:05d}.png"), final_full_view)
                if final_edge_view is not None:
                    cv2.imwrite(os.path.join(save_folder, f"pred_edge_{frame_count:05d}.png"), final_edge_view)
                cv2.imwrite(os.path.join(save_folder, f"combined_{frame_count:05d}.png"), combined)

                print(f"[{frame_count}] Saved. Crop: {crop_size_val}, Offset: {offset_val} to {save_folder}")
                frame_count += 1
                save_feedback_timer = 5

            if key == ord('q') or key == ESC_KEY:
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline_sdk.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()