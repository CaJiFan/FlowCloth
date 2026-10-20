"""
rollout_recorder.py
───────────────────
Real-robot rollout recording and reconstruction metrics engine for cloth manipulation.
Captures model predictions alongside physical ground-truth surface and boundary
observations directly from the Orbbec depth camera.

Metrics evaluated:
  - Directed Chamfer (P_obs → Mesh_pred) [mm]: Physical surface coverage
  - Directed Chamfer (Mesh_pred → P_obs) [mm]: Surface precision / hallucination penalty
  - Symmetric Chamfer-L1 [mm]
  - F1-Score @ 10mm (1 cm) and 20mm (2 cm)
  - Boundary Contour Chamfer Distance [mm]
  - Temporal Smoothness / Acceleration Jitter [mm/frame^2]

Exports per rollout (to rollouts/rollout_YYYYMMDD_HHMMSS/):
  - rollout_data.npz
  - metrics_summary.json
  - metrics_per_frame.csv
"""

from __future__ import annotations

import os
import time
import json
import csv
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from scipy.spatial import cKDTree


class RolloutRecorder:
    """
    Manages live recording of cloth state estimation rollouts and computes
    reconstruction metrics against physical depth camera observations.
    """

    def __init__(self, base_dir: str = "rollouts", auto_start: bool = False):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        self.is_recording: bool = False
        self.start_time: float = 0.0
        self.buffer: List[Dict] = []
        self.last_saved_dir: Optional[str] = None
        self.feedback_timer: int = 0
        self.feedback_text: str = ""
        self.current_out_dir: Optional[str] = None
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.record_video: bool = True
        self.save_raw_frames: bool = True

        if auto_start:
            self.start_recording()

    def start_recording(self, custom_out_dir: Optional[str] = None) -> None:
        """Start recording a new rollout."""
        self.is_recording = True
        self.start_time = time.time()
        self.buffer = []
        if custom_out_dir is not None:
            self.current_out_dir = custom_out_dir
        else:
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.current_out_dir = os.path.join(self.base_dir, f"rollout_{ts_str}")
        os.makedirs(self.current_out_dir, exist_ok=True)
        self.video_writer = None
        print(f"\n[RolloutRecorder] >>> RECORDING STARTED at {datetime.now().strftime('%H:%M:%S')} <<<")
        print(f"[RolloutRecorder] Output folder: {self.current_out_dir}")

    def stop_recording(self) -> Optional[str]:
        """Stop recording and evaluate all saved frames."""
        if not self.is_recording:
            return None

        self.is_recording = False
        elapsed = time.time() - self.start_time
        n_frames = len(self.buffer)
        print(f"\n[RolloutRecorder] >>> RECORDING STOPPED ({n_frames} frames in {elapsed:.1f}s) <<<")

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            if self.current_out_dir:
                vid_file = os.path.join(self.current_out_dir, "rollout_video.mp4")
                print(f"[RolloutRecorder] Exported synchronized overlay video: {vid_file}")

        if n_frames == 0:
            print("[RolloutRecorder] Warning: No frames were recorded.")
            return None

        saved_path = self._save_and_evaluate()
        self.last_saved_dir = saved_path
        self.feedback_timer = 90  # display banner for ~3 seconds at 30fps
        self.feedback_text = f"SAVED ({n_frames} frames)"
        self.buffer = []
        self.current_out_dir = None
        return saved_path

    def toggle_recording(self) -> bool:
        """Toggle recording state on/off. Returns current is_recording state."""
        if self.is_recording:
            self.stop_recording()
            return False
        else:
            self.start_recording()
            return True

    def handle_key(self, key: int) -> bool:
        """
        Processes OpenCV keycodes. Pressing 'r' or 'R' toggles recording.
        Returns True if recording state changed.
        """
        if key in (ord('r'), ord('R')):
            self.toggle_recording()
            return True
        return False

    def record_frame(
        self,
        frame_idx: int,
        pred_mm: np.ndarray,
        corners_mm: np.ndarray,
        masked_depth_mm: np.ndarray,
        binary_mask: np.ndarray,
        cam_intrinsics: Tuple[float, float, float, float],
        table_plane: Optional[Tuple[float, float, float]] = None,
        corners_base_m: Optional[np.ndarray] = None,
        pred_base_m: Optional[np.ndarray] = None,
        corner_idx: Optional[List[int]] = None,
        edge_loop_order: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None,
        depth_sensor_raw_mm: Optional[np.ndarray] = None,
        color_bgr: Optional[np.ndarray] = None,
    ) -> None:
        """Buffers the current inference frame and extracts physical ground truth."""
        if not self.is_recording:
            return

        t_now = timestamp if timestamp is not None else time.time()
        fx, fy, cx, cy = cam_intrinsics

        # ── 0. Save raw sensor depth (16-bit PNG) and RGB (JPEG) ──────────────
        if self.save_raw_frames and self.current_out_dir is not None:
            depth_dir = os.path.join(self.current_out_dir, "depth")
            color_dir = os.path.join(self.current_out_dir, "color")
            os.makedirs(depth_dir, exist_ok=True)
            os.makedirs(color_dir, exist_ok=True)

            # Save camera calibration metadata if not already written
            calib_file = os.path.join(self.current_out_dir, "camera_intrinsics.json")
            if not os.path.exists(calib_file):
                with open(calib_file, "w") as f:
                    json.dump({
                        "fx": float(fx),
                        "fy": float(fy),
                        "cx": float(cx),
                        "cy": float(cy)
                    }, f, indent=2)

            # Raw 16-bit millimeter depth (prefer untouched sensor depth; fallback to masked)
            d_raw = depth_sensor_raw_mm if depth_sensor_raw_mm is not None else masked_depth_mm
            if d_raw is not None:
                d_path = os.path.join(depth_dir, f"depth_{frame_idx:05d}.png")
                cv2.imwrite(d_path, d_raw.astype(np.uint16))

            # Color RGB / BGR frame
            if color_bgr is not None:
                c_path = os.path.join(color_dir, f"color_{frame_idx:05d}.jpg")
                cv2.imwrite(c_path, color_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        # ── 1. Extract physical surface ground truth point cloud P_obs ─────────
        pcd_obs = np.zeros((0, 3), dtype=np.float32)
        if masked_depth_mm is not None and binary_mask is not None:
            ys, xs = np.where((binary_mask > 0) & (masked_depth_mm > 0))
            if len(ys) > 0:
                zs = masked_depth_mm[ys, xs].astype(np.float32)
                X = (xs.astype(np.float32) - cx) / fx * zs
                Y = (ys.astype(np.float32) - cy) / fy * zs
                pcd_all = np.stack([X, Y, zs], axis=-1)

                # Keep up to 2000 uniformly sampled physical points per frame
                if len(pcd_all) > 2000:
                    sample_idx = np.random.choice(len(pcd_all), size=2000, replace=False)
                    pcd_obs = pcd_all[sample_idx]
                else:
                    pcd_obs = pcd_all

        # ── 2. Extract physical boundary contour ground truth C_obs ────────────
        contour_obs_3d = np.zeros((0, 3), dtype=np.float32)
        if binary_mask is not None and np.any(binary_mask > 0):
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                if len(largest) >= 10 and masked_depth_mm is not None:
                    c_pts = largest.squeeze(1)  # [K, 2] (u, v)
                    h, w = masked_depth_mm.shape[:2]
                    valid_c_3d = []
                    for u, v in c_pts:
                        if 0 <= u < w and 0 <= v < h:
                            # 3x3 local patch median depth for robustness against edge dropouts
                            patch = masked_depth_mm[max(0, v - 1):min(h, v + 2), max(0, u - 1):min(w, u + 2)]
                            valid_d = patch[patch > 0]
                            if len(valid_d) > 0:
                                z_val = float(np.median(valid_d))
                                x_val = (u - cx) / fx * z_val
                                y_val = (v - cy) / fy * z_val
                                valid_c_3d.append([x_val, y_val, z_val])
                    if len(valid_c_3d) > 0:
                        contour_obs_3d = np.array(valid_c_3d, dtype=np.float32)

        entry = {
            "frame_idx": frame_idx,
            "timestamp": t_now,
            "pred_mm": pred_mm.copy(),
            "corners_mm": corners_mm.copy(),
            "corners_base_m": corners_base_m.copy() if corners_base_m is not None else None,
            "pred_base_m": pred_base_m.copy() if pred_base_m is not None else None,
            "pcd_obs": pcd_obs,
            "contour_obs_3d": contour_obs_3d,
            "table_plane": np.array(table_plane, dtype=np.float32) if table_plane is not None else None,
            "corner_idx": corner_idx,
            "edge_loop_order": edge_loop_order,
        }
        self.buffer.append(entry)

    def _save_and_evaluate(self) -> str:
        """Calculates rollout reconstruction metrics and saves artifacts."""
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self.current_out_dir if self.current_out_dir is not None else os.path.join(
            self.base_dir, f"rollout_{ts_str}"
        )
        os.makedirs(out_dir, exist_ok=True)
        ts_str = os.path.basename(out_dir).replace("rollout_", "")

        n_frames = len(self.buffer)
        timestamps = np.array([e["timestamp"] for e in self.buffer], dtype=np.float64)
        pred_mesh_all = np.stack([e["pred_mm"] for e in self.buffer], axis=0)
        corners_all = np.stack([e["corners_mm"] for e in self.buffer], axis=0)

        # ── Compute per-frame metrics ──────────────────────────────────────────
        rows = []
        cd_obs2pred_list = []
        cd_pred2obs_list = []
        cd_sym_list = []
        f1_10_list = []
        f1_20_list = []
        contour_err_list = []

        for i, entry in enumerate(self.buffer):
            p_pred = entry["pred_mm"]        # [N, 3] mm
            p_obs = entry["pcd_obs"]         # [K, 3] mm
            c_obs = entry["contour_obs_3d"]  # [M, 3] mm

            row = {
                "frame": entry["frame_idx"],
                "time_sec": float(timestamps[i] - timestamps[0]),
                "cd_obs2pred_mm": np.nan,
                "cd_pred2obs_mm": np.nan,
                "cd_sym_mm": np.nan,
                "f1_10mm": np.nan,
                "f1_20mm": np.nan,
                "contour_chamfer_mm": np.nan,
                "jitter_mm": np.nan,
            }

            if len(p_obs) >= 10 and len(p_pred) >= 10:
                tree_pred = cKDTree(p_pred)
                tree_obs = cKDTree(p_obs)

                d_obs2pred = tree_pred.query(p_obs)[0]
                d_pred2obs = tree_obs.query(p_pred)[0]

                cd_obs2pred = float(d_obs2pred.mean())
                cd_pred2obs = float(d_pred2obs.mean())
                cd_sym = float((cd_obs2pred + cd_pred2obs) / 2.0)

                # F1 score @ 10mm
                prec_10 = float((d_pred2obs < 10.0).mean())
                rec_10 = float((d_obs2pred < 10.0).mean())
                f1_10 = (2 * prec_10 * rec_10 / (prec_10 + rec_10)) if (prec_10 + rec_10) > 0 else 0.0

                # F1 score @ 20mm
                prec_20 = float((d_pred2obs < 20.0).mean())
                rec_20 = float((d_obs2pred < 20.0).mean())
                f1_20 = (2 * prec_20 * rec_20 / (prec_20 + rec_20)) if (prec_20 + rec_20) > 0 else 0.0

                row["cd_obs2pred_mm"] = round(cd_obs2pred, 3)
                row["cd_pred2obs_mm"] = round(cd_pred2obs, 3)
                row["cd_sym_mm"] = round(cd_sym, 3)
                row["f1_10mm"] = round(f1_10, 4)
                row["f1_20mm"] = round(f1_20, 4)

                cd_obs2pred_list.append(cd_obs2pred)
                cd_pred2obs_list.append(cd_pred2obs)
                cd_sym_list.append(cd_sym)
                f1_10_list.append(f1_10)
                f1_20_list.append(f1_20)

            # Contour Chamfer Error (mm)
            if len(c_obs) >= 10:
                edge_order = entry.get("edge_loop_order")
                if edge_order is not None and len(edge_order) <= len(p_pred):
                    p_boundary = p_pred[edge_order]
                elif len(p_pred) in (76, 80):
                    p_boundary = p_pred
                else:
                    # Outer perimeter for 21x21 or 20x20
                    N = 21 if len(p_pred) == 441 else (20 if len(p_pred) == 400 else int(np.sqrt(len(p_pred))))
                    grid = np.arange(N * N).reshape(N, N)
                    contour_full = np.unique(np.concatenate([grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]]))
                    p_boundary = p_pred[contour_full]

                tree_c_obs = cKDTree(c_obs)
                tree_c_pred = cKDTree(p_boundary)
                d_b2c = tree_c_obs.query(p_boundary)[0].mean()
                d_c2b = tree_c_pred.query(c_obs)[0].mean()
                contour_err = float((d_b2c + d_c2b) / 2.0)
                row["contour_chamfer_mm"] = round(contour_err, 3)
                contour_err_list.append(contour_err)

            rows.append(row)

        # ── Temporal Jitter / Acceleration ────────────────────────────────────
        jitter_list = []
        if n_frames >= 3:
            for t in range(1, n_frames - 1):
                p_prev = pred_mesh_all[t - 1]
                p_curr = pred_mesh_all[t]
                p_next = pred_mesh_all[t + 1]
                accel = np.linalg.norm(p_next - 2.0 * p_curr + p_prev, axis=-1).mean()
                rows[t]["jitter_mm"] = round(float(accel), 3)
                jitter_list.append(float(accel))

        # ── Export CSV ────────────────────────────────────────────────────────
        csv_path = os.path.join(out_dir, "metrics_per_frame.csv")
        if rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

        # ── Summary JSON ──────────────────────────────────────────────────────
        def _stats(arr):
            if not arr:
                return {"mean": None, "median": None, "std": None}
            return {
                "mean": round(float(np.mean(arr)), 3),
                "median": round(float(np.median(arr)), 3),
                "std": round(float(np.std(arr)), 3),
                "min": round(float(np.min(arr)), 3),
                "max": round(float(np.max(arr)), 3),
            }

        summary = {
            "rollout_timestamp": ts_str,
            "num_frames": n_frames,
            "duration_sec": round(float(timestamps[-1] - timestamps[0]), 2) if n_frames > 1 else 0.0,
            "directed_chamfer_obs_to_pred_mm": _stats(cd_obs2pred_list),
            "directed_chamfer_pred_to_obs_mm": _stats(cd_pred2obs_list),
            "symmetric_chamfer_l1_mm": _stats(cd_sym_list),
            "f1_score_10mm": _stats(f1_10_list),
            "f1_score_20mm": _stats(f1_20_list),
            "contour_chamfer_error_mm": _stats(contour_err_list),
            "temporal_jitter_mm_per_frame2": _stats(jitter_list),
        }

        json_path = os.path.join(out_dir, "metrics_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        # ── Save NPZ ──────────────────────────────────────────────────────────
        npz_path = os.path.join(out_dir, "rollout_data.npz")
        npz_dict = {
            "timestamps": timestamps,
            "pred_mesh_mm": pred_mesh_all,
            "corners_mm": corners_all,
            "frame_indices": np.array([e["frame_idx"] for e in self.buffer], dtype=np.int32),
        }

        if self.buffer[0].get("corners_base_m") is not None:
            npz_dict["corners_base_m"] = np.stack([e["corners_base_m"] for e in self.buffer], axis=0)
        if self.buffer[0].get("pred_base_m") is not None:
            npz_dict["pred_base_m"] = np.stack([e["pred_base_m"] for e in self.buffer], axis=0)

        if self.buffer[0].get("table_plane") is not None:
            npz_dict["table_planes"] = np.stack([e["table_plane"] for e in self.buffer], axis=0)

        npz_dict["gt_depth_pcd"] = np.array([e["pcd_obs"] for e in self.buffer], dtype=object)
        npz_dict["gt_contour_pcd"] = np.array([e["contour_obs_3d"] for e in self.buffer], dtype=object)

        np.savez_compressed(npz_path, **npz_dict)

        # ── Print Console Report ──────────────────────────────────────────────
        print(f"\n=======================================================")
        print(f"       ROLLOUT EVALUATION REPORT (ICRA Metrics)        ")
        print(f"=======================================================")
        print(f"Output Directory : {out_dir}")
        print(f"Frames Recorded  : {n_frames} ({summary['duration_sec']} sec)")
        print(f"-------------------------------------------------------")
        s_cd = summary["directed_chamfer_obs_to_pred_mm"]
        print(f"Surface Coverage (P_obs → Mesh) : {s_cd['mean']} ± {s_cd['std']} mm (median: {s_cd['median']})")
        s_sym = summary["symmetric_chamfer_l1_mm"]
        print(f"Symmetric Chamfer-L1            : {s_sym['mean']} ± {s_sym['std']} mm")
        s_f10 = summary["f1_score_10mm"]
        f10_str = f"{s_f10['mean']*100:.1f}%" if s_f10.get('mean') is not None else "N/A"
        print(f"F1-Score @ 10mm (1 cm)          : {f10_str}")
        s_f20 = summary["f1_score_20mm"]
        f20_str = f"{s_f20['mean']*100:.1f}%" if s_f20.get('mean') is not None else "N/A"
        print(f"F1-Score @ 20mm (2 cm)          : {f20_str}")
        s_cnt = summary["contour_chamfer_error_mm"]
        print(f"Boundary Contour Error          : {s_cnt['mean']} ± {s_cnt['std']} mm")
        s_jit = summary["temporal_jitter_mm_per_frame2"]
        print(f"Temporal Acceleration Jitter    : {s_jit['mean']} ± {s_jit['std']} mm/frame²")
        if self.save_raw_frames and out_dir:
            depth_dir = os.path.join(out_dir, "depth")
            if os.path.isdir(depth_dir):
                n_depth = len(os.listdir(depth_dir))
                print(f"Raw Sensor Frames Saved         : {n_depth} frames ({out_dir}/depth/ & color/)")
        print(f"=======================================================\n")

        return out_dir

    def draw_overlay_hud(self, img: np.ndarray) -> None:
        """Renders recording status pill or saved notification onto OpenCV image."""
        h, w = img.shape[:2]

        if self.is_recording:
            # Record clean overlay video frame (without REC HUD banner)
            if self.record_video and self.current_out_dir is not None:
                if self.video_writer is None:
                    vid_path = os.path.join(self.current_out_dir, "rollout_video.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    self.video_writer = cv2.VideoWriter(vid_path, fourcc, 20.0, (w, h))
                self.video_writer.write(img)

            elapsed = time.time() - self.start_time
            n_frames = len(self.buffer)
            mins = int(elapsed // 60)
            secs = elapsed % 60
            text = f"REC {mins:02d}:{secs:04.1f} ({n_frames}f)"

            # Pulsing red circle
            pulse = int(abs(np.sin(time.time() * 3.0)) * 50)
            red_col = (0, 0, 205 + pulse)

            tw = 220
            th = 32
            x2 = w - 15
            x1 = x2 - tw
            y1 = 15
            y2 = y1 + th

            cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), red_col, 2)
            cv2.circle(img, (x1 + 18, y1 + th // 2), 7, red_col, -1)
            cv2.putText(img, text, (x1 + 34, y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        elif self.feedback_timer > 0:
            self.feedback_timer -= 1
            tw = 240
            th = 32
            x2 = w - 15
            x1 = x2 - tw
            y1 = 15
            y2 = y1 + th
            cv2.rectangle(img, (x1, y1), (x2, y2), (20, 40, 20), -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Vector checkmark icon (avoiding non-ASCII Unicode in Hershey font)
            check_pts = np.array([
                [x1 + 14, y1 + 17],
                [x1 + 19, y1 + 23],
                [x1 + 27, y1 + 11],
            ], dtype=np.int32)
            cv2.polylines(img, [check_pts], isClosed=False, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
