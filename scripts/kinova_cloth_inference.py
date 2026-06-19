"""
kinova_cloth_inference.py
─────────────────────────
End-to-end inference pipeline for real-robot cloth state estimation
with the Kinova Gen3 arm and an Orbbec depth camera.

What this script does
─────────────────────
  1. Load a trained UniClothDiff checkpoint (auto-detects FM vs DDPM).
  2. Load the cloth template mesh from the .pkl file in assets/.
  3. Capture + pre-process a depth frame from the Orbbec camera.
  4. Run the Flow Matching (or DDPM) state estimation pipeline.
  5. Extract the 4 physical corner coordinates of the cloth.
  6. (Optional) Publish corners as a ROS2 topic for Kinova grasping.
  7. (Optional) Visualise the predicted mesh / point cloud.

Quick start
───────────
  # From the UniClothDiff/ project root:
  python scripts/kinova_cloth_inference.py \
      --exp_dir experiments/ablation_fm_sw100.3_s1 \
      --step    250000 \
      --mode    edge

  # Offline test with a saved depth PNG:
  python scripts/kinova_cloth_inference.py \
      --exp_dir experiments/ablation_fm_sw100.3_s1 \
      --step    250000 \
      --mode    edge \
      --depth_png path/to/depth_raw_00000.png \
      --color_png path/to/color_00000.png

  # Continuous loop (for live robot demo):
  python scripts/kinova_cloth_inference.py ... --loop --ros

ROS2 note
─────────
  The --ros flag activates rclpy publishing.  If ROS2 is not installed, the
  script runs without it and just prints corner coordinates to stdout.
  Topic: /cloth_corners   type: geometry_msgs/msg/PoseArray
  Each pose.position contains one corner (x, y, z) in camera frame (metres).
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import argparse
import importlib
import numpy as np
import torch
from pathlib import Path
from typing import Optional

# ── project root on sys.path ──────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from omegaconf import OmegaConf
from uniclothdiff.registry import build_scheduler
from uniclothdiff.pipelines.cloth_state_est_pipeline import ClothStateEstPipeline
from uniclothdiff.pipelines.cloth_state_est_fm_pipeline import ClothStateEstFMPipeline

# ── local preprocessing module ────────────────────────────────────────────────
_SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS))
from orbbec_depth_processor import OrbbecDepthProcessor


# ─────────────────────────────────────────────────────────────────────────────
# Corner index helpers  (mirrors train.py / eval_offline.py conventions)
# ─────────────────────────────────────────────────────────────────────────────

def get_corner_indices(dataset_type: str, is_edge: bool) -> list[int]:
    """
    Return the indices (within the model's output tensor) that correspond
    to the 4 physical corners of the cloth.

    TRTM 21×21 grid:
      full  → [0, 20, 420, 440]   (indices into a 441-node mesh)
      edge  → [0, 20,  59,  79]   (indices into the 80-node contour)

    VR-Folding 20×20 grid:
      full  → [0, 19, 380, 399]
      edge  → [0, 19,  56,  75]
    """
    if "TRTM" in dataset_type:
        return [0, 20, 59, 79] if is_edge else [0, 20, 420, 440]
    else:
        return [0, 19, 56, 75] if is_edge else [0, 19, 380, 399]


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint / pipeline helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_checkpoint(exp_dir: str, step) -> str:
    """Return the checkpoint directory for a given step or 'latest'."""
    ckpts = Path(exp_dir) / "checkpoints"
    if not ckpts.is_dir():
        raise FileNotFoundError(f"No checkpoints/ found in {exp_dir}")
    available = sorted(
        [d for d in ckpts.iterdir() if d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1])
    )
    if not available:
        raise FileNotFoundError(f"No checkpoints in {ckpts}")
    if step == "latest":
        return str(available[-1])
    avail_steps = [int(d.name.split("-")[1]) for d in available]
    idx = min(range(len(avail_steps)), key=lambda i: abs(avail_steps[i] - int(step)))
    chosen = available[idx]
    if avail_steps[idx] != int(step):
        print(f"[warn] Requested step {step} not found; using {chosen.name}")
    return str(chosen)


def load_pipeline(config, checkpoint_dir: str, device: torch.device):
    """
    Load model weights and build the inference pipeline.
    Auto-detects Flow Matching vs DDPM from config.diffusion_cfg.type.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Template loading
# ─────────────────────────────────────────────────────────────────────────────

def load_template(config, is_edge: bool) -> np.ndarray:
    """
    Load the template mesh from the .pkl file specified in config.model_cfg.
    Returns normalised vertex positions as float32 [N, 3].
    """
    patch_file = config.model_cfg.patch_file
    with open(patch_file, "rb") as f:
        data = pickle.load(f)

    # The pkl stores 'points' (all nodes) and 'patch_index' / 'centers'
    pts = data["points"].astype(np.float32)   # [N_full, 3]

    if is_edge:
        # Determine contour from dataset type (default TRTM 21×21 → 80 nodes)
        dataset_type = config.dataset_cfg.get("type", "TRTMDiffusionDataset")
        if "TRTM" in dataset_type:
            N = 21
        else:
            N = 20
        grid     = np.arange(N * N).reshape(N, N)
        contour  = np.unique(np.concatenate([
            grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]
        ]))
        pts = pts[contour]   # → [80, 3] or [76, 3]

    # Normalise to unit scale (same as dataset __getitem__)
    pts -= pts.mean(axis=0)
    sc   = float(np.max(np.linalg.norm(pts, axis=1)))
    if sc > 1e-6:
        pts /= sc

    return pts   # [N, 3]


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    pipeline,
    use_fm: bool,
    pcd_np: np.ndarray,       # [num_pts, 3] float32, normalised
    q_temp_np: np.ndarray,    # [N, 3] float32
    num_steps: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run one forward pass of the cloth state estimation pipeline.

    Returns
    -------
    pred : float32 [N, 3] — predicted mesh in normalised space
    """
    pcd    = torch.tensor(pcd_np[None],    dtype=torch.float32, device=device)   # [1, P, 3]
    q_temp = torch.tensor(q_temp_np[None], dtype=torch.float32, device=device)   # [1, N, 3]
    N      = q_temp.shape[1]

    t0  = time.perf_counter()
    out = pipeline(
        encoder_hidden_states=pcd,
        q_temp=q_temp,
        shape=(1, N, 3),
        num_inference_steps=num_steps,
        do_classifier_free_guidance=False,
        call_v2=True,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0

    pred = out.frames if isinstance(out.frames, np.ndarray) else out.frames.cpu().numpy()
    print(f"  Inference time: {dt_ms:.1f} ms  ({1000/dt_ms:.1f} FPS)")
    return pred[0]   # [N, 3]


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation (optional, requires open3d or matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def visualise_result(
    pcd_norm: np.ndarray,    # [P, 3] input point cloud (normalised)
    pred_norm: np.ndarray,   # [N, 3] predicted mesh (normalised)
    corner_indices: list,
    title: str = "UniClothDiff Prediction",
):
    """Visualise predicted mesh and input PCD. Falls back to matplotlib if open3d is not available."""
    corners = pred_norm[corner_indices]   # [4, 3]

    try:
        import open3d as o3d

        pcd_o3d  = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(pcd_norm)
        pcd_o3d.paint_uniform_color([0.6, 0.6, 0.6])

        mesh_o3d = o3d.geometry.PointCloud()
        mesh_o3d.points = o3d.utility.Vector3dVector(pred_norm)
        mesh_o3d.paint_uniform_color([0.1, 0.6, 0.9])

        corner_o3d = o3d.geometry.PointCloud()
        corner_o3d.points = o3d.utility.Vector3dVector(corners)
        corner_o3d.paint_uniform_color([1.0, 0.0, 0.0])

        # Scale corner spheres for visibility
        spheres = []
        for c in corners:
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            s.translate(c)
            s.paint_uniform_color([1.0, 0.1, 0.1])
            spheres.append(s)

        o3d.visualization.draw_geometries(
            [pcd_o3d, mesh_o3d, corner_o3d] + spheres,
            window_name=title,
            width=900, height=700,
        )

    except ImportError:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

        fig = plt.figure(figsize=(10, 6))
        ax  = fig.add_subplot(111, projection="3d")
        ax.scatter(*pcd_norm.T,  s=2, c="grey",  alpha=0.4, label="Input PCD")
        ax.scatter(*pred_norm.T, s=8, c="cyan",  alpha=0.8, label="Prediction")
        ax.scatter(*corners.T,   s=80, c="red", marker="*", label="Corners", zorder=10)
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 publisher  (stub — activated by --ros flag)
# ─────────────────────────────────────────────────────────────────────────────

class ROSCornerPublisher:
    """
    Thin wrapper around rclpy to publish 4 cloth corners as a PoseArray.
    Topic:  /cloth_corners  (geometry_msgs/msg/PoseArray)
    Frame:  camera_depth_optical_frame (or whatever you set as --ros_frame)
    """

    def __init__(self, frame_id: str = "camera_depth_optical_frame"):
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion

        rclpy.init()
        self._node = Node("cloth_corner_publisher")
        self._pub  = self._node.create_publisher(PoseArray, "/cloth_corners", 10)
        self._frame_id = frame_id
        self._PoseArray = PoseArray
        self._Pose      = Pose
        self._Point     = Point
        self._Quaternion = Quaternion
        from builtin_interfaces.msg import Time as ROSTime
        self._ROSTime = ROSTime
        print(f"[ROS] Publisher ready → /cloth_corners  (frame: {frame_id})")

    def publish(self, corners_m: np.ndarray):
        """
        Publish corners.

        Parameters
        ----------
        corners_m : float32 [4, 3] — corner positions in metres, camera frame
        """
        from std_msgs.msg import Header
        msg = self._PoseArray()
        msg.header.frame_id = self._frame_id
        msg.header.stamp    = self._node.get_clock().now().to_msg()
        for pt in corners_m:
            pose = self._Pose()
            pose.position.x = float(pt[0])
            pose.position.y = float(pt[1])
            pose.position.z = float(pt[2])
            pose.orientation.w = 1.0   # identity rotation
            msg.poses.append(pose)
        self._pub.publish(msg)
        print(f"[ROS] Published corners (m): {corners_m.tolist()}")

    def destroy(self):
        self._node.destroy_node()
        import rclpy
        rclpy.shutdown()


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="UniClothDiff real-robot cloth state estimation pipeline"
    )
    p.add_argument(
        "--exp_dir", required=True,
        help="Experiment directory, e.g. experiments/ablation_fm_sw100.3_s1"
    )
    p.add_argument(
        "--step", default="latest",
        help="Checkpoint step to load (int or 'latest')"
    )
    p.add_argument(
        "--mode", choices=["edge", "full"], default="edge",
        help="Model mode: edge (80 nodes, fast) or full (441 nodes)"
    )
    p.add_argument(
        "--num_steps", type=int, default=None,
        help="Number of inference steps (default: 10 for FM, 100 for DDPM)"
    )
    p.add_argument(
        "--device", default="cuda",
        help="Torch device (cuda or cpu)"
    )
    # Camera / offline
    p.add_argument(
        "--depth_png", default=None,
        help="Path to saved depth_raw_*.png for offline testing (skips live camera)"
    )
    p.add_argument(
        "--color_png", default=None,
        help="Path to saved color_*.png for offline testing"
    )
    p.add_argument(
        "--num_sample_points", type=int, default=400,
        help="PCD sub-sample size (must match training config)"
    )
    p.add_argument(
        "--crop_size", type=int, default=480,
        help="Pixel crop size for the depth ROI"
    )
    p.add_argument(
        "--seg_threshold", type=int, default=100,
        help="Colour segmentation threshold (0-255)"
    )
    # Camera intrinsics (Orbbec depth stream)
    p.add_argument("--fx", type=float, default=None,
                   help="Depth camera focal length x (read from SDK if not set)")
    p.add_argument("--fy", type=float, default=None,
                   help="Depth camera focal length y (read from SDK if not set)")
    p.add_argument("--cx", type=float, default=None,
                   help="Depth camera principal point x")
    p.add_argument("--cy", type=float, default=None,
                   help="Depth camera principal point y")
    # Robot / ROS
    p.add_argument(
        "--ros", action="store_true",
        help="Enable ROS2 corner publisher (/cloth_corners PoseArray)"
    )
    p.add_argument(
        "--ros_frame", default="camera_depth_optical_frame",
        help="ROS TF frame for published corner poses"
    )
    # Camera-to-base transform (4×4, row-major, space-separated)
    p.add_argument(
        "--cam_to_base", default=None, nargs=16, type=float,
        metavar="M",
        help=(
            "16 values of the 4×4 camera-to-robot-base homogeneous transform "
            "(row-major). If provided, corners are expressed in robot base frame. "
            "Example: --cam_to_base 1 0 0 0.5  0 1 0 0  0 0 1 0.3  0 0 0 1"
        )
    )
    # Loop / visualisation
    p.add_argument(
        "--loop", action="store_true",
        help="Run inference continuously until Ctrl-C"
    )
    p.add_argument(
        "--vis", action="store_true",
        help="Visualise prediction with Open3D or matplotlib after each inference"
    )
    p.add_argument(
        "--save_dir", default=None,
        help="Directory to save per-frame results as .npz (pcd, pred, corners)"
    )
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[UniClothDiff] Device: {device}")

    # ── 1. Load config & checkpoint ──────────────────────────────────────────
    config_path = os.path.join(args.exp_dir, "config.yml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.yml not found in {args.exp_dir}")
    config = OmegaConf.load(config_path)

    ckpt_dir = resolve_checkpoint(args.exp_dir, args.step)
    print(f"[UniClothDiff] Loading checkpoint: {ckpt_dir}")

    pipeline, use_fm = load_pipeline(config, ckpt_dir, device)

    is_edge     = (args.mode == "edge")
    dataset_type = config.dataset_cfg.get("type", "TRTMDiffusionDataset")
    corner_idx   = get_corner_indices(dataset_type, is_edge)

    num_steps = args.num_steps
    if num_steps is None:
        num_steps = 10 if use_fm else 100
    print(f"[UniClothDiff] Scheduler: {'Flow Matching' if use_fm else 'DDPM'}  "
          f"| Steps: {num_steps}  | Mode: {args.mode}  "
          f"| Corner indices: {corner_idx}")

    # ── 2. Load template ─────────────────────────────────────────────────────
    q_temp = load_template(config, is_edge)
    print(f"[UniClothDiff] Template shape: {q_temp.shape}")

    # ── 3. Initialise preprocessor ───────────────────────────────────────────
    from orbbec_depth_processor import OrbbecDepthProcessor, DEPTH_FX_DEFAULT, DEPTH_FY_DEFAULT, DEPTH_CX_DEFAULT, DEPTH_CY_DEFAULT

    proc_kwargs = dict(
        num_sample_points = args.num_sample_points,
        crop_size         = args.crop_size,
        seg_threshold     = args.seg_threshold,
    )
    if args.fx is not None: proc_kwargs["fx"] = args.fx
    if args.fy is not None: proc_kwargs["fy"] = args.fy
    if args.cx is not None: proc_kwargs["cx"] = args.cx
    if args.cy is not None: proc_kwargs["cy"] = args.cy

    proc = OrbbecDepthProcessor(**proc_kwargs)

    live_mode = args.depth_png is None
    if live_mode:
        proc.start()

    # ── 4. Camera-to-base transform ──────────────────────────────────────────
    cam_to_base = None
    if args.cam_to_base is not None:
        cam_to_base = np.array(args.cam_to_base, dtype=np.float64).reshape(4, 4)
        print(f"[UniClothDiff] Camera→Base transform:\n{cam_to_base}")

    # ── 5. Optional ROS publisher ─────────────────────────────────────────────
    ros_pub = None
    if args.ros:
        try:
            ros_pub = ROSCornerPublisher(frame_id=args.ros_frame)
        except Exception as e:
            print(f"[ROS] Could not initialise publisher: {e}  (continuing without ROS)")

    # ── 6. Optional save directory ───────────────────────────────────────────
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # ── 7. Inference loop ────────────────────────────────────────────────────
    frame_idx = 0
    try:
        while True:
            print(f"\n─── Frame {frame_idx} ────────────────────────────────────")

            # A. Capture / load depth frame
            t_cap = time.perf_counter()
            if live_mode:
                result = proc.capture_and_process()
            else:
                result = proc.process_saved_depth(args.depth_png, args.color_png)
            print(f"  Capture+preprocess: {(time.perf_counter()-t_cap)*1e3:.1f} ms  "
                  f"| cloth pts: {result['n_cloth_pts']}")

            pcd_np = result["pcd"]   # [num_sample_points, 3] normalised

            # B. Inference
            pred_norm = run_inference(
                pipeline, use_fm, pcd_np, q_temp, num_steps, device
            )   # [N, 3] normalised

            # C. Extract corners (normalised)
            corners_norm = pred_norm[corner_idx]   # [4, 3]

            # D. Back-project to camera frame (mm)
            corners_mm = OrbbecDepthProcessor.denormalise_prediction(
                corners_norm, result["com"], result["scale"]
            )   # [4, 3] mm, camera frame

            # E. Optionally transform to robot base frame
            corners_out = corners_mm.copy()
            if cam_to_base is not None:
                ones        = np.ones((4, 1), dtype=np.float64)
                corners_h   = np.hstack([corners_mm.astype(np.float64), ones])   # [4, 4]
                corners_base = (cam_to_base @ corners_h.T).T[:, :3]              # [4, 3] mm
                corners_out  = corners_base.astype(np.float32)

            corners_m = corners_out / 1000.0   # mm → metres

            print(f"  Corner positions (metres, {'base' if cam_to_base is not None else 'camera'} frame):")
            labels = ["top-left", "top-right", "bottom-left", "bottom-right"]
            for lbl, c in zip(labels, corners_m):
                print(f"    {lbl:15s}: [{c[0]:+.4f}, {c[1]:+.4f}, {c[2]:+.4f}] m")

            # F. ROS publish
            if ros_pub is not None:
                ros_pub.publish(corners_m)

            # G. Save results
            if args.save_dir:
                save_path = os.path.join(args.save_dir, f"frame_{frame_idx:05d}.npz")
                np.savez(
                    save_path,
                    pcd_norm     = pcd_np,
                    pred_norm    = pred_norm,
                    corners_norm = corners_norm,
                    corners_mm   = corners_mm,
                    corners_m    = corners_m,
                    com          = result["com"],
                    scale        = result["scale"],
                )
                print(f"  Saved: {save_path}")

            # H. Visualise
            if args.vis:
                visualise_result(
                    pcd_np, pred_norm, corner_idx,
                    title=f"Frame {frame_idx} — UniClothDiff ({args.mode})"
                )

            frame_idx += 1

            if not args.loop and args.depth_png is not None:
                break   # single-shot offline test

            if not live_mode:
                break   # offline mode: only one frame

    except KeyboardInterrupt:
        print("\n[UniClothDiff] Interrupted by user.")

    finally:
        if live_mode:
            proc.stop()
        if ros_pub is not None:
            ros_pub.destroy()

    print(f"\n[UniClothDiff] Done. Processed {frame_idx} frame(s).")


if __name__ == "__main__":
    main()
