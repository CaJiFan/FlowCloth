#!/usr/bin/env python3
"""
rollout_inspector_gui.py
─────────────────────────
Interactive GUI for inspecting recorded cloth manipulation rollouts.
Allows sliding smoothly through saved depth + RGB frames, visualizing
predicted 3D cloth mesh overlays, comparing multiple model checkpoints
side-by-side, and inspecting physical ground-truth reconstruction metrics.

Usage:
  # Inspect the latest recorded rollout:
  python deployment/rollout_inspector_gui.py

  # Inspect a specific rollout folder:
  python deployment/rollout_inspector_gui.py --rollout_dir rollouts/rollout_20260914_172644

Controls:
  [Slider]        : Scrub to any frame (0 to N-1)
  [A] / [D]       : Previous / Next frame (or Left/Right arrow keys)
  [ [ ] / [ ] ]   : Skip -10 / +10 frames
  [Space]         : Play / Pause animation (at real-time FPS)
  [M]             : Cycle through available models (FlowCloth, TRTM, etc.)
  [V]             : Cycle view modes:
                      1: RGB + Mesh Overlay
                      2: Depth Map + Mesh Overlay
                      3: Dual View (RGB + Depth)
                      4: Side-by-Side Model Comparison
  [G]             : Toggle physical ground-truth surface point cloud (green points)
  [C]             : Toggle corner labels (TL, TR, BL, BR)
  [W]             : Toggle mesh wireframe opacity / mode
  [H]             : Toggle HUD overlay (clean view for paper figures)
  [S]             : Save high-resolution screenshot into rollout/screenshots/
  [R]             : Reset to frame 0
  [Q] / [Esc]     : Quit
"""

from __future__ import annotations

import os
import sys
import glob
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add repo root to path
_DEPLOYMENT = Path(__file__).resolve().parent
_REPO_ROOT = _DEPLOYMENT.parent
for p in [str(_DEPLOYMENT), str(_REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ─────────────────────────────────────────────────────────────────────────────
# 2D Projection and Wireframe Drawing Helpers
# ─────────────────────────────────────────────────────────────────────────────

def project_to_2d(pts_3d_mm: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Projects 3D points (mm) in camera frame to 2D image coordinates."""
    zs = np.clip(pts_3d_mm[:, 2], 1.0, None)
    xs = (pts_3d_mm[:, 0] * fx / zs) + cx
    ys = (pts_3d_mm[:, 1] * fy / zs) + cy
    return np.stack([xs, ys], axis=1).astype(np.int32)


def draw_mesh_wireframe(
    img: np.ndarray,
    uvs: np.ndarray,
    color: Tuple[int, int, int] = (0, 230, 255),
    thickness: int = 1,
    draw_nodes: bool = False,
) -> None:
    """
    Renders structured mesh wireframe on canvas.
    Automatically detects grid resolution (51x51 TRTM vs 21x21 FlowCloth vs 80 Edge Loop).
    """
    n_pts = len(uvs)
    h, w = img.shape[:2]

    if n_pts == 2601:
        # TRTM 51x51 grid: draw every 2nd line for clarity
        grid_size, step = 51, 2
    elif n_pts == 441:
        # FlowCloth 21x21 full grid
        grid_size, step = 21, 1
    elif n_pts == 400:
        # VR-Folding 20x20 grid
        grid_size, step = 20, 1
    elif n_pts in (80, 76, 56):
        # Boundary contour loop
        cv2.polylines(img, [uvs.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        if draw_nodes:
            for u, v in uvs:
                if 0 <= u < w and 0 <= v < h:
                    cv2.circle(img, (u, v), 2, (255, 120, 0), -1)
        return
    else:
        # Generic point cloud
        for u, v in uvs[::2]:
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(img, (u, v), 2, color, -1)
        return

    # Draw grid rows
    for r in range(0, grid_size, step):
        row_pts = np.ascontiguousarray(uvs[r * grid_size : (r + 1) * grid_size], dtype=np.int32)
        cv2.polylines(img, [row_pts.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    # Draw grid columns
    for c in range(0, grid_size, step):
        col_pts = np.ascontiguousarray(uvs[c :: grid_size], dtype=np.int32)
        cv2.polylines(img, [col_pts.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    # Interior nodes
    if draw_nodes:
        node_step = 2 if grid_size == 21 else 4
        for u, v in uvs[::node_step]:
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(img, (u, v), 2, (255, 120, 0), -1)


def draw_labeled_corners(img: np.ndarray, uvs_corners: np.ndarray) -> None:
    """Draws labeled circles for the 4 corners: TL (Green), TR (Yellow), BL (Orange), BR (Magenta)."""
    corner_names = ["TL", "TR", "BL", "BR"]
    corner_cols = [
        (0, 255, 0),    # Green: TL
        (0, 255, 255),  # Yellow: TR
        (0, 165, 255),  # Orange: BL
        (255, 0, 255),  # Magenta: BR
    ]
    h, w = img.shape[:2]
    for i, (u, v) in enumerate(uvs_corners):
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(img, (u, v), 7, corner_cols[i], -1, lineType=cv2.LINE_AA)
            cv2.circle(img, (u, v), 9, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            # Label background pill
            lbl = corner_names[i]
            cv2.putText(img, lbl, (u + 10, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, lbl, (u + 10, v - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, corner_cols[i], 1, cv2.LINE_AA)


def colorize_depth_map(depth_mm: np.ndarray, min_z: float = 600.0, max_z: float = 1100.0) -> np.ndarray:
    """Colorizes a 16-bit depth map into a JET heatmap for visual inspection."""
    valid = depth_mm > 0
    if not np.any(valid):
        return np.zeros((depth_mm.shape[0], depth_mm.shape[1], 3), dtype=np.uint8)

    norm = np.clip((depth_mm.astype(np.float32) - min_z) / max(1.0, (max_z - min_z)), 0.0, 1.0)
    depth_uint8 = (norm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    colored[~valid] = [20, 20, 20]
    return colored


# ─────────────────────────────────────────────────────────────────────────────
# Rollout Data Loader
# ─────────────────────────────────────────────────────────────────────────────

class RolloutDataset:
    """Loads all sensory frames and model predictions from a rollout directory."""

    def __init__(self, rollout_dir: str):
        self.rollout_dir = os.path.abspath(rollout_dir)
        if not os.path.isdir(self.rollout_dir):
            raise FileNotFoundError(f"Rollout directory not found: {self.rollout_dir}")

        # 1. Load Camera Intrinsics
        calib_file = os.path.join(self.rollout_dir, "camera_intrinsics.json")
        self.fx, self.fy, self.cx, self.cy = 505.15, 505.15, 320.0, 240.0
        if os.path.isfile(calib_file):
            try:
                with open(calib_file, "r") as f:
                    c = json.load(f)
                self.fx = c.get("fx", self.fx)
                self.fy = c.get("fy", self.fy)
                self.cx = c.get("cx", self.cx)
                self.cy = c.get("cy", self.cy)
            except Exception as e:
                print(f"[RolloutDataset] Warning: Could not read {calib_file}: {e}")

        # 2. Discover Depth and Color frames
        depth_dir = os.path.join(self.rollout_dir, "depth")
        color_dir = os.path.join(self.rollout_dir, "color")

        if os.path.isdir(depth_dir):
            self.depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
        else:
            self.depth_files = sorted(
                glob.glob(os.path.join(self.rollout_dir, "depth_raw_*.png"))
                or glob.glob(os.path.join(self.rollout_dir, "depth_*.png"))
            )

        self.color_files = []
        for d_path in self.depth_files:
            base = os.path.splitext(os.path.basename(d_path))[0]
            idx_str = base.replace("depth_raw_", "").replace("depth_", "").replace("frame_", "")
            c_path = None
            for ext in [".jpg", ".png", ".jpeg"]:
                cands = [
                    os.path.join(color_dir, f"color_{idx_str}{ext}"),
                    os.path.join(color_dir, f"frame_{idx_str}{ext}"),
                    os.path.join(self.rollout_dir, f"color_{idx_str}{ext}"),
                    os.path.join(self.rollout_dir, f"color_raw_{idx_str}{ext}"),
                ]
                for c in cands:
                    if os.path.isfile(c):
                        c_path = c
                        break
                if c_path:
                    break
            self.color_files.append(c_path)

        self.num_frames = len(self.depth_files)
        if self.num_frames == 0:
            raise FileNotFoundError(f"No depth frames found in: {self.rollout_dir}")

        # 3. Discover Model Predictions
        self.models: Dict[str, Dict] = {}
        self._load_models()

    def _load_models(self) -> None:
        """Discovers predictions from root rollout_data.npz and any eval_* subfolders."""
        # A. Root rollout data
        root_npz = os.path.join(self.rollout_dir, "rollout_data.npz")
        if os.path.isfile(root_npz):
            try:
                d = np.load(root_npz, allow_pickle=True)
                metrics = self._load_metrics_csv(os.path.join(self.rollout_dir, "metrics_per_frame.csv"))
                mesh = d["pred_mesh_mm"]
                if mesh.ndim == 3 and mesh.shape[1] == 2601:
                    root_name = "TRTM (Recorded)"
                elif mesh.ndim == 3 and mesh.shape[1] == 441:
                    root_name = "FlowCloth (Recorded)"
                else:
                    root_name = "Recorded Run"
                self.models[root_name] = {
                    "pred_mesh": mesh,
                    "corners": d["corners_mm"],
                    "gt_depth_pcd": d.get("gt_depth_pcd"),
                    "timestamps": d.get("timestamps"),
                    "metrics": metrics,
                }
            except Exception as e:
                print(f"[RolloutDataset] Warning: Could not read root NPZ: {e}")

        # B. Eval subdirectories
        for sub in sorted(os.listdir(self.rollout_dir)):
            sub_path = os.path.join(self.rollout_dir, sub)
            if os.path.isdir(sub_path) and sub.startswith("eval_"):
                sub_npz = os.path.join(sub_path, "rollout_data.npz")
                if os.path.isfile(sub_npz):
                    try:
                        d = np.load(sub_npz, allow_pickle=True)
                        metrics = self._load_metrics_csv(os.path.join(sub_path, "metrics_per_frame.csv"))
                        mesh = d["pred_mesh_mm"]
                        # Clean label: e.g. eval_ablation_fm_... -> FlowCloth (ablation_fm)
                        tag = sub.replace("eval_", "")
                        # Truncate timestamp from tag for cleaner display
                        parts = tag.split("_")
                        if len(parts) > 2 and parts[-2].isdigit() and parts[-1].isdigit():
                            tag = "_".join(parts[:-2])
                        if mesh.ndim == 3 and mesh.shape[1] == 441:
                            display_tag = f"FlowCloth ({tag})"
                        elif mesh.ndim == 3 and mesh.shape[1] == 2601:
                            display_tag = f"TRTM ({tag})"
                        else:
                            display_tag = tag
                        self.models[display_tag] = {
                            "pred_mesh": mesh,
                            "corners": d["corners_mm"],
                            "gt_depth_pcd": d.get("gt_depth_pcd"),
                            "timestamps": d.get("timestamps"),
                            "metrics": metrics,
                        }
                    except Exception as e:
                        print(f"[RolloutDataset] Warning: Could not read {sub_npz}: {e}")

        # Add a Clean Image option
        self.models["None (Clean Sensor)"] = None

    def _load_metrics_csv(self, csv_path: str) -> Optional[List[Dict]]:
        if not os.path.isfile(csv_path):
            return None
        rows = []
        try:
            import csv
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(r)
        except Exception:
            return None
        return rows

    def get_frame(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (color_bgr, depth_uint16) for frame idx."""
        idx = np.clip(idx, 0, self.num_frames - 1)
        # Load Depth
        d_path = self.depth_files[idx]
        depth_raw = cv2.imread(d_path, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            depth_raw = np.zeros((480, 640), dtype=np.uint16)
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]
        if depth_raw.dtype != np.uint16:
            depth_raw = depth_raw.astype(np.uint16)

        # Load Color
        c_path = self.color_files[idx]
        if c_path and os.path.isfile(c_path):
            color_bgr = cv2.imread(c_path)
        else:
            # Fallback colorize depth if color frame missing
            color_bgr = colorize_depth_map(depth_raw)

        # Ensure matching resolutions
        dh, dw = depth_raw.shape[:2]
        if color_bgr.shape[:2] != (dh, dw):
            color_bgr = cv2.resize(color_bgr, (dw, dh))

        return color_bgr, depth_raw


# ─────────────────────────────────────────────────────────────────────────────
# Interactive GUI Application
# ─────────────────────────────────────────────────────────────────────────────

class RolloutInspectorGUI:
    def __init__(self, dataset: RolloutDataset):
        self.dataset = dataset
        self.window_name = "FlowCloth Rollout Inspector"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 800)

        self.current_frame: int = 0
        self.is_playing: bool = False
        self.play_fps: float = 20.0
        self.view_mode: int = 0  # 0: RGB, 1: Depth, 2: Dual (RGB+Depth), 3: Side-by-Side Model Compare
        self.show_gt_pcd: bool = False
        self.show_corners: bool = True
        self.show_wireframe: bool = True
        self.show_hud: bool = True
        self.mesh_alpha: float = 0.85

        self.model_names = list(self.dataset.models.keys())
        self.current_model_idx: int = 0
        self.compare_model_idx: int = min(1, len(self.model_names) - 1)

        self.banner_text: str = ""
        self.banner_timer: int = 0

        # Screenshots output directory
        self.screenshot_dir = os.path.join(self.dataset.rollout_dir, "screenshots")
        os.makedirs(self.screenshot_dir, exist_ok=True)

        # Create Trackbars
        cv2.createTrackbar("Frame", self.window_name, 0, max(1, self.dataset.num_frames - 1), self._on_frame_slider)
        cv2.createTrackbar("Model", self.window_name, 0, max(1, len(self.model_names) - 1), self._on_model_slider)
        cv2.createTrackbar("Alpha %", self.window_name, int(self.mesh_alpha * 100), 100, self._on_alpha_slider)

        print(f"\n[RolloutInspector] Initialized with {self.dataset.num_frames} frames.")
        print(f"[RolloutInspector] Discovered {len(self.model_names)} model prediction sets:")
        for i, m in enumerate(self.model_names):
            print(f"   [{i}] {m}")
        print("\nPress 'H' in window to toggle help/HUD.\n")

    def _on_frame_slider(self, val: int) -> None:
        self.current_frame = np.clip(val, 0, self.dataset.num_frames - 1)

    def _on_model_slider(self, val: int) -> None:
        self.current_model_idx = np.clip(val, 0, len(self.model_names) - 1)

    def _on_alpha_slider(self, val: int) -> None:
        self.mesh_alpha = np.clip(val / 100.0, 0.0, 1.0)

    def render_single_view(
        self,
        base_img: np.ndarray,
        model_key: str,
        frame_idx: int,
        wireframe_color=(0, 230, 255),
    ) -> np.ndarray:
        """Renders mesh and corner overlay onto an image canvas."""
        canvas = base_img.copy()
        overlay = canvas.copy()
        model_data = self.dataset.models.get(model_key)

        if model_data is not None:
            pred_mesh_all = model_data.get("pred_mesh")
            corners_all = model_data.get("corners")

            # Draw Predicted Mesh
            if self.show_wireframe and pred_mesh_all is not None and frame_idx < len(pred_mesh_all):
                mesh_3d = pred_mesh_all[frame_idx]
                uvs_mesh = project_to_2d(mesh_3d, self.dataset.fx, self.dataset.fy, self.dataset.cx, self.dataset.cy)
                draw_mesh_wireframe(overlay, uvs_mesh, color=wireframe_color, thickness=1, draw_nodes=False)

            # Draw Physical Ground Truth Surface (Green points from depth)
            if self.show_gt_pcd and model_data.get("gt_depth_pcd") is not None:
                gt_list = model_data["gt_depth_pcd"]
                if frame_idx < len(gt_list) and gt_list[frame_idx] is not None and len(gt_list[frame_idx]) > 0:
                    p_gt = gt_list[frame_idx]
                    uvs_gt = project_to_2d(p_gt, self.dataset.fx, self.dataset.fy, self.dataset.cx, self.dataset.cy)
                    h, w = canvas.shape[:2]
                    for u, v in uvs_gt[::2]:
                        if 0 <= u < w and 0 <= v < h:
                            cv2.circle(overlay, (u, v), 1, (0, 255, 120), -1)

            # Apply Alpha Blending
            if self.mesh_alpha < 1.0:
                canvas = cv2.addWeighted(overlay, self.mesh_alpha, canvas, 1.0 - self.mesh_alpha, 0)
            else:
                canvas = overlay

            # Draw Corners (Unblended for crispness)
            if self.show_corners and corners_all is not None and frame_idx < len(corners_all):
                c_3d = corners_all[frame_idx]
                uvs_corners = project_to_2d(c_3d, self.dataset.fx, self.dataset.fy, self.dataset.cx, self.dataset.cy)
                draw_labeled_corners(canvas, uvs_corners)

        return canvas

    def draw_hud(self, img: np.ndarray, model_key: str, frame_idx: int) -> None:
        """Draws top information pill and bottom control instructions."""
        h, w = img.shape[:2]

        # ── Top Telemetry Pill ───────────────────────────────────────────────
        model_data = self.dataset.models.get(model_key)
        # Adaptive top bar height
        is_wide = (w > 800)
        th = 44 if is_wide else 58
        cv2.rectangle(img, (10, 8), (w - 10, 8 + th), (20, 20, 20), -1)
        cv2.rectangle(img, (10, 8), (w - 10, 8 + th), (70, 70, 70), 1)

        # Model Tag & Status
        play_icon = "PLAY" if self.is_playing else "PAUSE"
        t_sec = 0.0
        if model_data and model_data.get("timestamps") is not None and frame_idx < len(model_data["timestamps"]):
            ts = model_data["timestamps"]
            t_sec = float(ts[frame_idx] - ts[0])
        else:
            t_sec = frame_idx / self.play_fps

        title_text = f"[{play_icon}]  Frame: {frame_idx:03d}/{self.dataset.num_frames-1:03d} ({t_sec:04.2f}s)  |  Model: {model_key}"

        # Metrics summary
        metrics_text = ""
        if model_data and model_data.get("metrics") is not None and frame_idx < len(model_data["metrics"]):
            m = model_data["metrics"][frame_idx]
            cd = m.get("cd_sym_mm", "")
            f1 = m.get("f1_10mm", "")
            if cd and cd != "nan":
                try:
                    f1_val = float(f1) * 100
                    metrics_text = f"Chamfer: {float(cd):.2f}mm  |  F1@10mm: {f1_val:.1f}%"
                except Exception:
                    metrics_text = f"Chamfer: {float(cd):.2f}mm"

        if is_wide:
            cv2.putText(img, title_text, (22, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
            if metrics_text:
                cv2.putText(img, metrics_text, (w - 380, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 220), 1, cv2.LINE_AA)
        else:
            cv2.putText(img, title_text, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
            if metrics_text:
                cv2.putText(img, metrics_text, (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 220), 1, cv2.LINE_AA)

        # ── Bottom Control Helper ────────────────────────────────────────────
        bh = 26
        cv2.rectangle(img, (0, h - bh), (w, h), (15, 15, 15), -1)
        help_text = "[Space] Play/Pause  |  [A/D] Prev/Next  |  [M] Switch Model  |  [V] View Mode  |  [G] GT Depth PCD  |  [S] Screenshot  |  [Q] Quit"
        cv2.putText(img, help_text, (20, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        # ── Notification Banner (if saved screenshot, etc.) ──────────────────
        if self.banner_timer > 0:
            self.banner_timer -= 1
            bw = 360
            b_x1 = (w - bw) // 2
            b_y1 = h - 70
            cv2.rectangle(img, (b_x1, b_y1), (b_x1 + bw, b_y1 + 34), (20, 50, 20), -1)
            cv2.rectangle(img, (b_x1, b_y1), (b_x1 + bw, b_y1 + 34), (0, 255, 0), 2)
            cv2.putText(img, self.banner_text, (b_x1 + 18, b_y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA)

    def render_frame(self) -> np.ndarray:
        """Composes the final composite image according to selected view mode."""
        color_bgr, depth_raw = self.dataset.get_frame(self.current_frame)
        active_model = self.model_names[self.current_model_idx]

        if self.view_mode == 0:
            # Mode 0: RGB + Selected Model Overlay
            out = self.render_single_view(color_bgr, active_model, self.current_frame)
            if self.show_hud:
                self.draw_hud(out, active_model, self.current_frame)
            return out

        elif self.view_mode == 1:
            # Mode 1: Depth Map + Selected Model Overlay
            depth_vis = colorize_depth_map(depth_raw)
            out = self.render_single_view(depth_vis, active_model, self.current_frame, wireframe_color=(255, 255, 255))
            if self.show_hud:
                self.draw_hud(out, active_model, self.current_frame)
            return out

        elif self.view_mode == 2:
            # Mode 2: Dual View (Left: RGB Overlay | Right: Depth Map Overlay)
            left = self.render_single_view(color_bgr, active_model, self.current_frame)
            depth_vis = colorize_depth_map(depth_raw)
            right = self.render_single_view(depth_vis, active_model, self.current_frame, wireframe_color=(255, 255, 255))
            dual = np.hstack([left, right])
            if self.show_hud:
                self.draw_hud(dual, f"{active_model} (Dual RGB + Depth)", self.current_frame)
            return dual

        elif self.view_mode == 3:
            # Mode 3: Side-by-Side Model Comparison (Model A vs Model B)
            m_a = self.model_names[self.current_model_idx]
            m_b = self.model_names[self.compare_model_idx]
            left = self.render_single_view(color_bgr, m_a, self.current_frame, wireframe_color=(0, 230, 255))
            right = self.render_single_view(color_bgr, m_b, self.current_frame, wireframe_color=(0, 165, 255))

            # Metric pills on left and right panels
            def _draw_model_badge(panel, m_key, color_bgr):
                m_data = self.dataset.models.get(m_key)
                txt = f"Model: {m_key}"
                if m_data and m_data.get("metrics") is not None and self.current_frame < len(m_data["metrics"]):
                    met = m_data["metrics"][self.current_frame]
                    cd_val = met.get("cd_sym_mm", "")
                    if cd_val and cd_val != "nan":
                        txt += f" | Chamfer: {float(cd_val):.1f}mm"
                tw = min(panel.shape[1] - 30, len(txt) * 9 + 20)
                cv2.rectangle(panel, (15, 65), (15 + tw, 95), (20, 20, 20), -1)
                cv2.rectangle(panel, (15, 65), (15 + tw, 95), color_bgr, 1)
                cv2.putText(panel, txt, (22, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color_bgr, 1, cv2.LINE_AA)

            _draw_model_badge(left, m_a, (0, 230, 255))
            _draw_model_badge(right, m_b, (0, 165, 255))

            comp = np.hstack([left, right])
            if self.show_hud:
                self.draw_hud(comp, f"Comparison: {m_a} vs {m_b}", self.current_frame)
            return comp

        return color_bgr

    def save_screenshot(self) -> None:
        """Saves current frame composite to disk."""
        img = self.render_frame()
        active_model = self.model_names[self.current_model_idx].replace(" ", "_").replace("(", "").replace(")", "")
        fname = f"screenshot_frame_{self.current_frame:04d}_{active_model}.png"
        path = os.path.join(self.screenshot_dir, fname)
        cv2.imwrite(path, img)
        self.banner_text = f"Saved: {fname}"
        self.banner_timer = 50
        print(f"[RolloutInspector] Screenshot saved to: {path}")

    def run(self) -> None:
        """Main interaction loop."""
        last_play_time = time.time()

        while True:
            # Playback logic
            if self.is_playing:
                now = time.time()
                if now - last_play_time >= (1.0 / self.play_fps):
                    self.current_frame = (self.current_frame + 1) % self.dataset.num_frames
                    cv2.setTrackbarPos("Frame", self.window_name, self.current_frame)
                    last_play_time = now

            # Render frame
            display_img = self.render_frame()
            cv2.imshow(self.window_name, display_img)

            # Wait key
            key = cv2.waitKey(20) & 0xFF

            if key == 27 or key == ord('q'):
                print("[RolloutInspector] Exiting.")
                break
            elif key == ord(' '):
                self.is_playing = not self.is_playing
            elif key in (ord('d'), 83):  # 'D' or Right Arrow
                self.is_playing = False
                self.current_frame = min(self.dataset.num_frames - 1, self.current_frame + 1)
                cv2.setTrackbarPos("Frame", self.window_name, self.current_frame)
            elif key in (ord('a'), 81):  # 'A' or Left Arrow
                self.is_playing = False
                self.current_frame = max(0, self.current_frame - 1)
                cv2.setTrackbarPos("Frame", self.window_name, self.current_frame)
            elif key == ord(']'):
                self.current_frame = min(self.dataset.num_frames - 1, self.current_frame + 10)
                cv2.setTrackbarPos("Frame", self.window_name, self.current_frame)
            elif key == ord('['):
                self.current_frame = max(0, self.current_frame - 10)
                cv2.setTrackbarPos("Frame", self.window_name, self.current_frame)
            elif key == ord('r'):
                self.current_frame = 0
                cv2.setTrackbarPos("Frame", self.window_name, 0)
            elif key == ord('m'):
                # Cycle model
                self.current_model_idx = (self.current_model_idx + 1) % len(self.model_names)
                cv2.setTrackbarPos("Model", self.window_name, self.current_model_idx)
            elif key == ord('v'):
                # Cycle view mode
                self.view_mode = (self.view_mode + 1) % 4
                v_names = ["RGB Overlay", "Depth Overlay", "Dual RGB+Depth", "Side-by-Side Model Compare"]
                self.banner_text = f"View Mode: {v_names[self.view_mode]}"
                self.banner_timer = 40
            elif key == ord('g'):
                self.show_gt_pcd = not self.show_gt_pcd
                self.banner_text = f"GT Surface PCD: {'ON' if self.show_gt_pcd else 'OFF'}"
                self.banner_timer = 40
            elif key == ord('c'):
                self.show_corners = not self.show_corners
            elif key == ord('w'):
                self.show_wireframe = not self.show_wireframe
            elif key == ord('h'):
                self.show_hud = not self.show_hud
            elif key == ord('s'):
                self.save_screenshot()

        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_rollout(base_dir: str = "rollouts") -> Optional[str]:
    """Finds the most recently modified rollout directory."""
    if not os.path.isdir(base_dir):
        return None
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if d.startswith("rollout_")]
    subdirs = [d for d in subdirs if os.path.isdir(d)]
    if not subdirs:
        return None
    return max(subdirs, key=os.path.getmtime)


def parse_args():
    p = argparse.ArgumentParser(description="Interactive Rollout Inspector & Model Overlay GUI")
    p.add_argument(
        "--rollout_dir", default=None,
        help="Path to rollout directory (default: latest directory in rollouts/)"
    )
    p.add_argument(
        "--fps", type=float, default=20.0,
        help="Playback speed in frames per second (default: 20.0)"
    )
    return p.parse_args()


def main():
    args = parse_args()
    rollout_dir = args.rollout_dir
    if rollout_dir is None:
        rollout_dir = find_latest_rollout()
        if rollout_dir is None:
            print("[RolloutInspector] Error: No rollouts found in rollouts/ directory.")
            print("Please provide --rollout_dir path/to/rollout_YYYYMMDD_HHMMSS")
            sys.exit(1)
        print(f"[RolloutInspector] Using latest rollout: {rollout_dir}")

    dataset = RolloutDataset(rollout_dir)
    gui = RolloutInspectorGUI(dataset)
    gui.play_fps = args.fps
    gui.run()


if __name__ == "__main__":
    main()
