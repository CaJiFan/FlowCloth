#!/usr/bin/env python3
"""
trtm_cloth_inference.py
───────────────────────
End-to-end inference pipeline for real-robot cloth state estimation
with the original TRTM checkpoint (checkpoint_large_size.pt) and an Orbbec depth camera.
Directly adapted from deployment/kinova_cloth_inference.py.

Usage:
  # Live camera inference with live overlay window:
  python deployment/trtm_cloth_inference.py

  # With ROS publishing:
  python deployment/trtm_cloth_inference.py --ros --udp

  # Offline test with a saved depth PNG:
  python deployment/trtm_cloth_inference.py \
      --depth_png trtm_data_calibrated_20260910_103728/depth_raw_00000.png \
      --color_png trtm_data_calibrated_20260910_103728/color_00000.png
"""

from __future__ import annotations

import cv2

import os
import sys
import time
import glob
import json
import argparse
from datetime import datetime
import numpy as np
import torch
from pathlib import Path
from typing import Optional, List, Tuple

# ── project root on sys.path ──────────────────────────────────────────────────
_DEPLOYMENT = Path(__file__).resolve().parent
_REPO_ROOT = _DEPLOYMENT.parent
for p in [str(_DEPLOYMENT), str(_REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from orbbec_depth_processor import OrbbecDepthProcessor
from trtm_gnn_model import TRTMGNN, load_trtm_model
from rollout_recorder import RolloutRecorder

# ─────────────────────────────────────────────────────────────────────────────
# Corner Constants & Grid Helpers for TRTM 51x51 (2601 vertices)
# ─────────────────────────────────────────────────────────────────────────────
# 51x51 template semantic corner indices:
# 2550: row 50, col 0 (x=-1, y=+1) -> Top-Left (TL)
# 2600: row 50, col 50 (x=+1, y=+1) -> Top-Right (TR)
# 0: row 0, col 0 (x=-1, y=-1) -> Bottom-Left (BL)
# 50: row 0, col 50 (x=+1, y=-1) -> Bottom-Right (BR)
TRTM_51x51_CORNERS = [2550, 2600, 0, 50]
CORNER_NAMES = ["Top-Left (TL)", "Top-Right (TR)", "Bottom-Left (BL)", "Bottom-Right (BR)"]
CORNER_COLORS = [
    (0, 255, 0),    # Green: TL
    (0, 255, 255),  # Yellow: TR
    (0, 165, 255),  # Orange: BL
    (255, 0, 255),  # Magenta: BR
]


def draw_grid_wireframe(img: np.ndarray, uvs: np.ndarray, grid_size: int = 51, color=(0, 230, 255), thickness: int = 1):
    """Draws quadrilateral grid wireframe for N x N vertices on canvas."""
    if len(uvs) != grid_size * grid_size:
        return
    for r in range(0, grid_size, 2):
        row_pts = np.ascontiguousarray(uvs[r * grid_size : (r + 1) * grid_size], dtype=np.int32)
        cv2.polylines(img, [row_pts.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    for c in range(0, grid_size, 2):
        col_pts = np.ascontiguousarray(uvs[c :: grid_size], dtype=np.int32)
        cv2.polylines(img, [col_pts.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)



# ─────────────────────────────────────────────────────────────────────────────
# 2D Projection helper
# ─────────────────────────────────────────────────────────────────────────────

def project_to_2d(pts_3d_mm: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Project 3D points in camera frame back to 2D image coordinates."""
    xs = (pts_3d_mm[:, 0] * fx / np.clip(pts_3d_mm[:, 2], 1.0, None)) + cx
    ys = (pts_3d_mm[:, 1] * fy / np.clip(pts_3d_mm[:, 2], 1.0, None)) + cy
    return np.stack([xs, ys], axis=1).astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# Template-guided corner completion (TGC)
# ─────────────────────────────────────────────────────────────────────────────

def template_guided_corner_completion(
    corners_mm: np.ndarray,
    q_temp: np.ndarray,
    corner_idx: list[int],
    arm_mask: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    prev_pred_mm: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Recovers occluded corners under the robot arm via affine template fit."""
    height, width = arm_mask.shape
    temp_corners = q_temp[corner_idx].astype(np.float32)

    uvs_corners = project_to_2d(corners_mm, fx, fy, cx, cy)
    visible, occluded = [], []
    for i, (u, v) in enumerate(uvs_corners):
        in_bounds = (0 <= u < width) and (0 <= v < height)
        is_arm = in_bounds and (arm_mask[v, u] > 0)
        (occluded if is_arm else visible).append(i)

    if not occluded or len(visible) < 3:
        return corners_mm

    src = temp_corners[visible]
    dst = corners_mm[visible].astype(np.float32)
    ones = np.ones((len(visible), 1), dtype=np.float32)
    src_h = np.hstack([src, ones])
    M, _, _, _ = np.linalg.lstsq(src_h, dst, rcond=None)

    corners_out = corners_mm.copy()
    for i in occluded:
        tc_h = np.append(temp_corners[i], 1.0)
        corners_out[i] = tc_h @ M
    return corners_out


# ─────────────────────────────────────────────────────────────────────────────
# ROS publisher (Identical to kinova_cloth_inference.py)
# ─────────────────────────────────────────────────────────────────────────────

class ROSCornerPublisher:
    """
    Publishes 4 cloth corners as a PoseArray.
    Topic: /cloth_corners (geometry_msgs/msg/PoseArray)
    """

    def __init__(self, frame_id: str = "camera_depth_optical_frame", force_udp: bool = False, udp_port: int = 9876):
        self._frame_id = frame_id
        self.use_udp = force_udp
        if not self.use_udp:
            try:
                import rospy
                from geometry_msgs.msg import PoseArray, Pose
                rospy.init_node("cloth_corner_publisher", anonymous=True)
                self._pub = rospy.Publisher("/cloth_corners", PoseArray, queue_size=10)
                self._PoseArray = PoseArray
                self._Pose = Pose
                print(f"[ROS1] Publisher ready → /cloth_corners (frame: {frame_id})")
            except Exception as e:
                print(f"[ROS] Native rospy not available in venv ({e}). Falling back to UDP bridge on port {udp_port}.")
                self.use_udp = True

        if self.use_udp:
            import socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_addr = ("127.0.0.1", udp_port)
            print(f"[UDP] Publisher online → 127.0.0.1:{udp_port} (frame: {frame_id})")

    def publish(self, corners_m: np.ndarray, pred_m: np.ndarray = None):
        if self.use_udp:
            import json
            data = {"frame_id": self._frame_id, "corners": corners_m.tolist()}
            if pred_m is not None:
                idx_down = np.linspace(0, len(pred_m) - 1, min(441, len(pred_m)), dtype=int)
                data["mesh"] = pred_m[idx_down].tolist()
            try:
                self.sock.sendto(json.dumps(data).encode("utf-8"), self.udp_addr)
            except Exception as e:
                print(f"[UDP] sendto error: {e}")
            return

        import rospy
        msg = self._PoseArray()
        msg.header.frame_id = self._frame_id
        msg.header.stamp = rospy.Time.now()
        for pt in corners_m:
            pose = self._Pose()
            pose.position.x = float(pt[0])
            pose.position.y = float(pt[1])
            pose.position.z = float(pt[2])
            pose.orientation.w = 1.0
            msg.poses.append(pose)
        self._pub.publish(msg)

    def destroy(self):
        if self.use_udp and hasattr(self, "sock"):
            self.sock.close()


def find_replay_frames(rollout_dir: str) -> List[Tuple[str, Optional[str]]]:
    """Discovers sorted (depth_path, color_path) pairs from a recorded rollout folder."""
    depth_dir = os.path.join(rollout_dir, "depth")
    color_dir = os.path.join(rollout_dir, "color")

    if os.path.isdir(depth_dir):
        depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    else:
        depth_files = sorted(
            glob.glob(os.path.join(rollout_dir, "depth_raw_*.png"))
            or glob.glob(os.path.join(rollout_dir, "depth_*.png"))
        )

    pairs = []
    for d_path in depth_files:
        base_name = os.path.splitext(os.path.basename(d_path))[0]
        idx_str = base_name.replace("depth_raw_", "").replace("depth_", "").replace("frame_", "")
        c_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidates = [
                os.path.join(color_dir, f"color_{idx_str}{ext}"),
                os.path.join(color_dir, f"frame_{idx_str}{ext}"),
                os.path.join(rollout_dir, f"color_{idx_str}{ext}"),
                os.path.join(rollout_dir, f"color_raw_{idx_str}{ext}"),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    c_path = c
                    break
            if c_path:
                break
        pairs.append((d_path, c_path))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point (Matching kinova_cloth_inference.py)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="TRTM real-robot cloth state estimation pipeline"
    )
    p.add_argument(
        "--ckpt",
        default="/home/cjimenez/Downloads/checkpoint_large_size.pt",
        help="Path to TRTM checkpoint_large_size.pt",
    )
    p.add_argument(
        "--template",
        default=str(_REPO_ROOT / "assets" / "trtm_large" / "flat_state_050.txt"),
        help="Path to flat_state_050.txt (2601 points)",
    )
    p.add_argument(
        "--edges",
        default=str(_REPO_ROOT / "assets" / "trtm_large" / "mesh_edge_idx.txt"),
        help="Path to mesh_edge_idx.txt (7600 edges)",
    )
    p.add_argument("--depth_png", default=None, help="Saved depth_raw_*.png for offline testing")
    p.add_argument("--color_png", default=None, help="Saved color_*.png for offline testing")
    p.add_argument(
        "--replay_rollout", default=None,
        help="Path to a recorded rollout directory to replay and benchmark offline (e.g. rollouts/rollout_...)"
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num_sample_points", type=int, default=400)
    p.add_argument("--crop_size", type=int, default=480)
    p.add_argument("--seg_threshold", type=int, default=120, help="Colour segmentation threshold (0-255, 0=Otsu auto)")
    p.add_argument("--disable_plane", action="store_true")
    p.add_argument("--disable_pruning", action="store_true")
    p.add_argument("--fx", type=float, default=None)
    p.add_argument("--fy", type=float, default=None)
    p.add_argument("--cx", type=float, default=None)
    p.add_argument("--cy", type=float, default=None)
    p.add_argument("--ros", action="store_true", help="Publish corners via ROS/UDP")
    p.add_argument("--udp", action="store_true", help="Force UDP fallback for ROS")
    p.add_argument("--udp_port", type=int, default=9876, help="UDP bridge port")
    p.add_argument("--ros_frame", default="camera_depth_optical_frame")
    p.add_argument(
        "--cam_to_base", default=None, nargs=16, type=float,
        help="16 values of the 4x4 camera-to-robot-base transform (row-major)"
    )
    p.add_argument("--loop", action="store_true", help="Run continuously in offline mode")
    p.add_argument(
        "--show_overlay", action="store_true", default=True,
        help="Display a live OpenCV window showing model prediction on the RGB stream (enabled by default)"
    )
    p.add_argument("--no_gui", action="store_true", help="Run headless without OpenCV window")
    p.add_argument("--save_dir", default=None, help="Directory to save per-frame results")
    p.add_argument(
        "--record_rollout", action="store_true",
        help="Automatically record the rollout from start to finish"
    )
    p.add_argument(
        "--rollout_dir", default="rollouts",
        help="Directory to save rollout data and metrics (default: rollouts/)"
    )
    p.add_argument("--ema_alpha", type=float, default=0.3, help="EMA smoothing factor (0.0 to 1.0)")
    p.add_argument("--arm_depth_offset", type=float, default=0.0, help="Robot arm depth offset mm (0 = disabled)")
    p.add_argument("--inpaint_radius", type=int, default=15)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[TRTM] Device: {device}", flush=True)

    # ── 1. Load TRTM Checkpoint ──────────────────────────────────────────────
    print(f"[TRTM] Loading checkpoint: {args.ckpt}", flush=True)
    model = load_trtm_model(
        ckpt_path=args.ckpt,
        template_path=args.template,
        edge_path=args.edges,
        device=device,
    )
    corner_idx = [2550, 2600, 0, 50]
    print(f"[TRTM] Model loaded successfully | Corner indices: {corner_idx}", flush=True)

    # ── 2. Load template ─────────────────────────────────────────────────────
    q_temp = np.loadtxt(args.template).astype(np.float32)
    print(f"[TRTM] Template shape: {q_temp.shape}", flush=True)

    # ── 3. Initialise preprocessor ───────────────────────────────────────────
    proc_kwargs = dict(
        num_sample_points=args.num_sample_points,
        crop_size=args.crop_size,
        seg_threshold=args.seg_threshold,
        arm_depth_offset_mm=args.arm_depth_offset,
        inpaint_radius=args.inpaint_radius,
        use_plane_correction=not args.disable_plane,
        use_height_pruning=not args.disable_pruning,
    )
    replay_frames = None
    if args.replay_rollout is not None:
        if not os.path.isdir(args.replay_rollout):
            raise FileNotFoundError(f"Rollout directory not found: {args.replay_rollout}")
        replay_frames = find_replay_frames(args.replay_rollout)
        if not replay_frames:
            raise FileNotFoundError(f"No depth frames found in rollout folder: {args.replay_rollout}")
        print(f"[TRTM] Replay mode: discovered {len(replay_frames)} frames in {args.replay_rollout}", flush=True)

        calib_file = os.path.join(args.replay_rollout, "camera_intrinsics.json")
        if os.path.isfile(calib_file):
            try:
                with open(calib_file, "r") as f:
                    calib = json.load(f)
                if args.fx is None and "fx" in calib: proc_kwargs["fx"] = calib["fx"]
                if args.fy is None and "fy" in calib: proc_kwargs["fy"] = calib["fy"]
                if args.cx is None and "cx" in calib: proc_kwargs["cx"] = calib["cx"]
                if args.cy is None and "cy" in calib: proc_kwargs["cy"] = calib["cy"]
                print(f"[TRTM] Loaded calibration from {calib_file}: fx={proc_kwargs.get('fx')}, fy={proc_kwargs.get('fy')}", flush=True)
            except Exception as e:
                print(f"[TRTM] Warning: could not load {calib_file}: {e}")

    if args.fx is not None: proc_kwargs["fx"] = args.fx
    if args.fy is not None: proc_kwargs["fy"] = args.fy
    if args.cx is not None: proc_kwargs["cx"] = args.cx
    if args.cy is not None: proc_kwargs["cy"] = args.cy

    proc = OrbbecDepthProcessor(**proc_kwargs)

    live_mode = (args.depth_png is None and args.replay_rollout is None)
    if live_mode:
        print("[TRTM] Starting live Orbbec camera stream...", flush=True)
        proc.start()
        print("[TRTM] Camera stream started.", flush=True)

    # ── 4. Camera-to-base transform ──────────────────────────────────────────
    cam_to_base = None
    if args.cam_to_base is not None:
        cam_to_base = np.array(args.cam_to_base, dtype=np.float64).reshape(4, 4)
        print(f"[TRTM] Camera→Base transform:\n{cam_to_base}", flush=True)

    # ── 5. Optional ROS publisher ─────────────────────────────────────────────
    ros_pub = None
    if args.ros or args.udp:
        try:
            ros_pub = ROSCornerPublisher(
                frame_id=args.ros_frame,
                force_udp=args.udp,
                udp_port=args.udp_port,
            )
        except Exception as e:
            print(f"[ROS] Could not initialise publisher: {e}")

    # ── 6. Optional save directory & Rollout Recorder ────────────────────────
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    if args.replay_rollout:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        replay_out = os.path.join(args.replay_rollout, f"eval_trtm_{ts_str}")
        recorder = RolloutRecorder(base_dir=args.replay_rollout, auto_start=False)
        recorder.start_recording(custom_out_dir=replay_out)
        recorder.save_raw_frames = False  # Avoid duplicating raw frames during replay
    else:
        recorder = RolloutRecorder(base_dir=args.rollout_dir, auto_start=args.record_rollout)

    # ── 7. Inference loop ────────────────────────────────────────────────────
    frame_idx = 0
    prev_pred_mm = None
    show_gui = (args.show_overlay and not args.no_gui)

    try:
        while True:
            if replay_frames is not None and frame_idx >= len(replay_frames):
                print(f"\n[TRTM] Replay complete: processed all {len(replay_frames)} frames.", flush=True)
                break

            print(f"\n─── Frame {frame_idx} ────────────────────────────────────", flush=True)

            # A. Capture / load depth frame
            t_cap = time.perf_counter()
            if replay_frames is not None:
                d_path, c_path = replay_frames[frame_idx]
                result = proc.process_saved_depth(d_path, c_path, prev_mesh=prev_pred_mm)
            elif live_mode:
                result = proc.capture_and_process(prev_mesh=prev_pred_mm, pump_gui=show_gui)
            else:
                result = proc.process_saved_depth(args.depth_png, args.color_png, prev_mesh=prev_pred_mm)
            print(f"  Capture+preprocess: {(time.perf_counter()-t_cap)*1e3:.1f} ms  "
                  f"| cloth pts: {result['n_cloth_pts']}", flush=True)

            # B. TRTM Inference
            trtm_img = result["trtm_image"]
            if trtm_img.ndim == 2:
                trtm_img_bgr = cv2.cvtColor(trtm_img, cv2.COLOR_GRAY2BGR)
            else:
                trtm_img_bgr = trtm_img
            trtm_resized = cv2.resize(trtm_img_bgr, (224, 224), interpolation=cv2.INTER_AREA)
            trtm_tensor = (
                torch.from_numpy(trtm_resized.transpose(2, 0, 1))
                .float()
                .unsqueeze(0)
                .div(255.0)
                .to(device)
            )

            t_infer = time.perf_counter()
            with torch.no_grad():
                pred_norm = model(trtm_tensor)[0].cpu().numpy()  # [2601, 3]
            dt_infer = (time.perf_counter() - t_infer) * 1000.0
            print(f"  TRTM Inference: {dt_infer:.1f} ms")

            # C. Extract corners (normalised)
            corners_norm = pred_norm[corner_idx]   # [4, 3]

            # D. Back-project to camera frame (mm) via TRTM native unprojection
            half_ideal = 240.0
            center_px = 360.0

            u_canvas = center_px + pred_norm[:, 0] * half_ideal
            v_canvas = center_px - pred_norm[:, 1] * half_ideal

            crop_center = result["crop_center"]
            crop_size_val = float(result["crop_size"])
            scale_ratio = crop_size_val / 720.0

            u_cam = crop_center[0] + (u_canvas - center_px) * scale_ratio
            v_cam = crop_center[1] + (v_canvas - center_px) * scale_ratio

            h_mm = pred_norm[:, 2] * 1000.0

            table_z = float(result["table_depth_mm"])
            table_plane = result.get("table_plane")
            if table_plane is not None:
                A, B, C = table_plane
                z_table_pts = (A * (u_cam - proc.cx) / proc.fx + B * (v_cam - proc.cy) / proc.fy) * table_z + C
            else:
                z_table_pts = table_z

            z_cam = z_table_pts - h_mm
            x_cam = (u_cam - proc.cx) / proc.fx * z_cam
            y_cam = (v_cam - proc.cy) / proc.fy * z_cam
            pred_mm = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)

            # Apply Exponential Moving Average (EMA) for temporal smoothing
            old_pred_mm = prev_pred_mm
            if prev_pred_mm is None:
                prev_pred_mm = pred_mm.copy()
            else:
                pred_mm = args.ema_alpha * pred_mm + (1.0 - args.ema_alpha) * prev_pred_mm
                prev_pred_mm = pred_mm.copy()

            corners_mm = pred_mm[corner_idx]

            # D2. Template-guided corner completion (TGC)
            arm_mask = result.get("arm_mask", None)
            if arm_mask is not None and arm_mask.any():
                corners_mm = template_guided_corner_completion(
                    corners_mm, q_temp, corner_idx, arm_mask,
                    proc.fx, proc.fy, proc.cx, proc.cy,
                    prev_pred_mm=old_pred_mm,
                )

            # E. Optionally transform to robot base frame
            corners_out = corners_mm.copy()
            if cam_to_base is not None:
                ones = np.ones((4, 1), dtype=np.float64)
                corners_h = np.hstack([corners_mm.astype(np.float64), ones])
                corners_base = (cam_to_base @ corners_h.T).T[:, :3]
                corners_out = corners_base.astype(np.float32)

            corners_m = corners_out / 1000.0   # mm → metres

            print(f"  Corner positions (metres, {'base' if cam_to_base is not None else 'camera'} frame):", flush=True)
            labels = ["top-left", "top-right", "bottom-left", "bottom-right"]
            for lbl, c in zip(labels, corners_m):
                print(f"    {lbl:15s}: [{c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f}] m", flush=True)

            # F. ROS publish
            if ros_pub is not None:
                pred_out = pred_mm.copy()
                if cam_to_base is not None:
                    ones = np.ones((len(pred_mm), 1), dtype=np.float64)
                    pred_h = np.hstack([pred_mm.astype(np.float64), ones])
                    pred_base = (cam_to_base @ pred_h.T).T[:, :3]
                    pred_out = pred_base.astype(np.float32)
                pred_m = pred_out / 1000.0
                ros_pub.publish(corners_m, pred_m=pred_m)

            # G. Rollout Recording (captures pred mesh + physical depth GT surface)
            recorder.record_frame(
                frame_idx=frame_idx,
                pred_mm=pred_mm,
                corners_mm=corners_mm,
                masked_depth_mm=result.get("depth_raw_mm"),
                binary_mask=result.get("cloth_mask"),
                cam_intrinsics=(proc.fx, proc.fy, proc.cx, proc.cy),
                table_plane=result.get("table_plane"),
                corners_base_m=corners_m if cam_to_base is not None else None,
                pred_base_m=pred_m if cam_to_base is not None else None,
                corner_idx=corner_idx,
                depth_sensor_raw_mm=result.get("depth_sensor_raw_mm"),
                color_bgr=result.get("color_bgr"),
            )

            # H. Save results if requested
            if args.save_dir:
                save_path = os.path.join(args.save_dir, f"frame_{frame_idx:05d}.npz")
                np.savez(
                    save_path,
                    pred_norm=pred_norm,
                    corners_norm=corners_norm,
                    corners_mm=corners_mm,
                    corners_m=corners_m,
                    com=result["com"],
                    scale=result["scale"],
                )
                print(f"  Saved: {save_path}")

            # I. Live OpenCV Overlay (Press 'R' to start/stop rollout recording)
            if show_gui and "color_bgr" in result:
                color_img = result["color_bgr"].copy()
                overlay_pred_mm = pred_mm

                # Project points to 2D
                uvs_all = project_to_2d(overlay_pred_mm, proc.fx, proc.fy, proc.cx, proc.cy)
                uvs_corners = project_to_2d(corners_mm, proc.fx, proc.fy, proc.cx, proc.cy)

                # Draw mesh wireframe
                draw_grid_wireframe(color_img, uvs_all, grid_size=51, color=(0, 230, 255), thickness=1)

                # Draw mesh nodes (cyan/blue)
                for u, v in uvs_all[::4]:
                    if 0 <= u < color_img.shape[1] and 0 <= v < color_img.shape[0]:
                        cv2.circle(color_img, (u, v), 2, (255, 120, 0), -1)

                # Draw corners (larger, labeled)
                corner_names = ["TL", "TR", "BL", "BR"]
                corner_cols = [(0, 255, 0), (0, 255, 255), (0, 165, 255), (255, 0, 255)]
                for i, (u, v) in enumerate(uvs_corners):
                    if 0 <= u < color_img.shape[1] and 0 <= v < color_img.shape[0]:
                        cv2.circle(color_img, (u, v), 7, corner_cols[i], -1)
                        cv2.circle(color_img, (u, v), 9, (255, 255, 255), 1)
                        cv2.putText(color_img, corner_names[i], (u + 8, v - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, corner_cols[i], 2)

                # Render rollout recording HUD badge (pulsing REC or SAVED banner)
                recorder.draw_overlay_hud(color_img)

                cv2.imshow("Live Cloth Tracking Overlay", color_img)
                key = cv2.waitKey(1) & 0xFF
                if recorder.handle_key(key):
                    pass
                elif key == 27 or key == ord('q'):
                    print("[TRTM] Overlay closed by user.")
                    break

            frame_idx += 1

            if not args.loop and args.depth_png is not None and args.replay_rollout is None:
                break   # single-shot offline test

            if not live_mode and args.replay_rollout is None:
                break   # offline mode: only one frame

    except KeyboardInterrupt:
        print("\n[TRTM] Interrupted by user.")

    finally:
        if recorder.is_recording:
            recorder.stop_recording()
        if live_mode:
            proc.stop()
        if ros_pub is not None:
            ros_pub.destroy()

    print(f"\n[TRTM] Done. Processed {frame_idx} frame(s).")


if __name__ == "__main__":
    main()
