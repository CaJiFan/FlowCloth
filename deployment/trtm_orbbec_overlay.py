#!/usr/bin/env python3
"""
trtm_orbbec_overlay.py
──────────────────────
Interactive Orbbec 3D camera calibration and real-time inference overlay
specifically designed for the original TRTM checkpoint (checkpoint_large_size.pt).
Directly adapted from deployment/orbbec_overlay_3d.py.

Usage:
  # Run calibration GUI with live TRTM tracking:
  python deployment/trtm_orbbec_overlay.py

  # With ROS publishing enabled:
  python deployment/trtm_orbbec_overlay.py --ros --udp
"""

import cv2
# Warm up OpenCV Qt GUI context immediately on main thread before other C++ runtimes load
cv2.namedWindow("Master Calibration", cv2.WINDOW_NORMAL)
cv2.waitKey(1)

import numpy as np
import os
import sys
import time
import argparse
import torch
from datetime import datetime
from pathlib import Path
from pyorbbecsdk import *
from scipy.spatial import cKDTree

# ── Ensure repo root and deployment directory are on sys.path ────────────────
_DEPLOYMENT = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_DEPLOYMENT)
for p in [_DEPLOYMENT, _REPO_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from orbbec_depth_processor import (
    depth_image_to_pcd,
    normalise_pcd,
    subsample_pcd,
    OrbbecDepthProcessor,
    DEPTH_FX_DEFAULT, DEPTH_FY_DEFAULT, DEPTH_CX_DEFAULT, DEPTH_CY_DEFAULT
)
from trtm_gnn_model import TRTMGNN, load_trtm_model
from trtm_cloth_inference import (
    ROSCornerPublisher,
    TRTM_51x51_CORNERS,
    CORNER_NAMES,
    CORNER_COLORS,
    draw_grid_wireframe,
    template_guided_corner_completion,
)


# --- CONFIGURATION ---
ESC_KEY = 27
MIN_ARM_PIXELS_FOR_TGC = 500

# TRTM Standards
TARGET_CANVAS_SIZE = 720
IDEAL_CLOTH_SIZE = 480
FLAT_PIXEL_VAL = 192
MANUAL_TABLE_DEPTH = None

# Shading Tuning
INVERT_SHADING = True       # High folds = Darker pixels
PIXEL_PER_MM = 2.0          # 1cm (10mm) depth = 20 pixels difference

# Calibration Defaults
DEFAULT_CROP_SIZE = 480
DEFAULT_THRESHOLD = 35
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
        img_padded = cv2.copyMakeBorder(
            img, pad_t, pad_b, pad_l, pad_r, border_type, value=val
        )
        crop = img_padded[
            y1 + pad_t : y2 + pad_t,
            x1 + pad_l : x2 + pad_l,
        ]
    else:
        crop = img[y1:y2, x1:x2]

    method = cv2.INTER_AREA if is_rgb else cv2.INTER_CUBIC
    return cv2.resize(crop, (target_size, target_size), interpolation=method)


def apply_trtm_shading(
    depth_map_float, mask_bg, fx=DEPTH_FX_DEFAULT, fy=DEPTH_FY_DEFAULT,
    cx=DEPTH_CX_DEFAULT, cy=DEPTH_CY_DEFAULT, table_plane=None, offset_mm=0, max_table_depth=1300.0
):
    """
    Calculates TRTM values using Float Depth + Inverted Logic + Table Plane Correction.
    Guarantees flat cloth is normalized to FLAT_PIXEL_VAL (192), eliminating tilt artifacts.
    """
    h, w = depth_map_float.shape
    cloth_mask = (~mask_bg) & (depth_map_float > 0) & (depth_map_float < max_table_depth)
    cloth_depths = depth_map_float[cloth_mask]

    if len(cloth_depths) == 0:
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w, 3), dtype=np.uint8), 900.0

    if table_plane is not None:
        A, B, C = table_plane
        ys, xs = np.where(cloth_mask)
        zs = depth_map_float[ys, xs]
        X = (xs.astype(np.float32) - cx) / fx * zs
        Y = (ys.astype(np.float32) - cy) / fy * zs
        z_table_pts = A * X + B * Y + C + offset_mm
        heights = np.clip(z_table_pts - zs, 0.0, 35.0)

        height_map = np.zeros((h, w), dtype=np.float32)
        height_map[ys, xs] = heights
        table_z = float(np.median(z_table_pts))
    else:
        # Table depth: 85th percentile of valid cloth depths represents the flat table contact surface
        table_depth = float(np.percentile(cloth_depths, 85)) + offset_mm
        height_map = np.clip(table_depth - depth_map_float, 0.0, 35.0)
        table_z = table_depth

    # --- INVERTED LOGIC ---
    if INVERT_SHADING:
        processed = FLAT_PIXEL_VAL - (height_map * PIXEL_PER_MM)
    else:
        processed = FLAT_PIXEL_VAL + (height_map * PIXEL_PER_MM)

    processed = np.clip(processed, 0, 255).astype(np.uint8)

    # Saving version (white background)
    encoded_save = processed.copy()
    encoded_save[mask_bg] = 255

    # Display version (black background for contrast)
    vis_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    vis_bgr[mask_bg] = [0, 0, 0]

    return encoded_save, vis_bgr, table_z


def segment_cloth_threshold(image, threshold=35, depth_map=None, max_depth=1300.0):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if threshold <= 0:
        # Automatic Otsu thresholding when slider is at 0
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    # Depth gating: reject floor/background pixels beyond max table depth
    if depth_map is not None:
        binary[(depth_map > max_depth) & (depth_map > 0)] = 0

    # Close first to bridge dark folds, wrinkles, and ToF dropouts
    k_close = np.ones((7, 7), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    k_open = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(binary)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        max_area = cv2.contourArea(largest)
        if max_area > 500:
            for cnt in contours:
                if cv2.contourArea(cnt) >= max(500.0, 0.15 * max_area):
                    cv2.drawContours(final_mask, [cnt], -1, 255, -1)
            # Fill any pinholes within the cloth mask
            cnts_fill, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts_fill:
                cv2.drawContours(final_mask, [c], -1, 255, -1)

    # Post-fill depth gating safety
    if depth_map is not None:
        final_mask[(depth_map > max_depth) & (depth_map > 0)] = 0

    return (final_mask // 255).astype(np.uint8)


def nothing(x):
    pass


def project_to_2d(pts_3d_mm, fx, fy, cx, cy):
    """Project 3D points in camera frame back to 2D image coordinates."""
    xs = (pts_3d_mm[:, 0] * fx / np.clip(pts_3d_mm[:, 2], 1.0, None)) + cx
    ys = (pts_3d_mm[:, 1] * fy / np.clip(pts_3d_mm[:, 2], 1.0, None)) + cy
    return np.stack([xs, ys], axis=1).astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description="TRTM Orbbec calibration & live inference GUI")
    parser.add_argument(
        "--ckpt",
        default="/home/cjimenez/Downloads/checkpoint_large_size.pt",
        help="Path to TRTM checkpoint_large_size.pt",
    )
    parser.add_argument(
        "--template",
        default=os.path.join(_REPO_ROOT, "assets", "trtm_large", "flat_state_050.txt"),
        help="Path to flat_state_050.txt",
    )
    parser.add_argument(
        "--edges",
        default=os.path.join(_REPO_ROOT, "assets", "trtm_large", "mesh_edge_idx.txt"),
        help="Path to mesh_edge_idx.txt",
    )
    parser.add_argument("--ema_alpha", type=float, default=0.3, help="EMA smoothing factor (0.0 to 1.0)")
    parser.add_argument("--disable_plane", action="store_true", help="Disable RANSAC table plane calibration")
    parser.add_argument("--disable_pruning", action="store_true", help="Disable outlier height pruning")
    parser.add_argument("--ros", action="store_true", help="Publish corners via ROS/UDP topic /cloth_corners")
    parser.add_argument("--udp", action="store_true", help="Force UDP socket fallback for ROS")
    parser.add_argument("--udp_port", type=int, default=9876, help="UDP bridge port")
    parser.add_argument("--ros_frame", default="camera_depth_optical_frame")
    parser.add_argument("--max_depth", type=float, default=DEFAULT_MAX_DEPTH, help="Maximum depth cutoff in mm (filters out floor)")
    args = parser.parse_args()

    # Trackbars & initial splash screen on main thread
    cv2.createTrackbar("Zoom (Crop Size)", "Master Calibration", DEFAULT_CROP_SIZE, 800, nothing)
    cv2.createTrackbar("Threshold (0=Auto)", "Master Calibration", DEFAULT_THRESHOLD, 255, nothing)
    cv2.createTrackbar("Table Offset",     "Master Calibration", DEFAULT_OFFSET,    100, nothing)
    cv2.createTrackbar("Arm Offset (mm)",  "Master Calibration", 0,                 300, nothing)
    init_splash = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(init_splash, "TRTM Overlay: Initializing Camera & Model...", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Master Calibration", init_splash)
    cv2.waitKey(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[TRTM] Device: {device}", flush=True)

    # 1. Load TRTM Model
    model = None
    # 51x51 template semantic corner indices:
    # 2550: row 50, col 0 (x=-1, y=+1) -> Top-Left (TL)
    # 2600: row 50, col 50 (x=+1, y=+1) -> Top-Right (TR)
    # 0: row 0, col 0 (x=-1, y=-1) -> Bottom-Left (BL)
    # 50: row 0, col 50 (x=+1, y=-1) -> Bottom-Right (BR)
    corner_idx = [2550, 2600, 0, 50]
    corner_labels = ["TL", "TR", "BL", "BR"]
    if args.ckpt and os.path.isfile(args.ckpt):
        print(f"Loading TRTM checkpoint from: {args.ckpt}", flush=True)
        model = load_trtm_model(
            ckpt_path=args.ckpt,
            template_path=args.template,
            edge_path=args.edges,
            device=device,
        )
        print(f"TRTM Model loaded successfully. Corner indices: {corner_idx}. Inference window enabled.", flush=True)
    else:
        print(f"Warning: Checkpoint not found at {args.ckpt}. Running in pure calibration mode.", flush=True)

    pipeline_sdk = None
    for attempt in range(1, 8):
        try:
            pipeline_sdk = Pipeline()
            break
        except Exception as e:
            print(f"[TRTM] Camera connection attempt {attempt}/7 ({e}). Retrying in 1.5s...", flush=True)
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
            print("No suitable color profile found.", flush=True)
            return

        config_sdk.enable_stream(color_profile)
        depth_profiles = pipeline_sdk.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = depth_profiles.get_default_video_stream_profile()
        config_sdk.enable_stream(depth_profile)
        config_sdk.set_align_mode(OBAlignMode.DISABLE)
        pipeline_sdk.start(config_sdk)

        # State for temporal smoothing
        prev_pred_mm = None
        arm_mask_2d = None
        align_filter = AlignFilter(align_to_stream=OBStreamType.DEPTH_STREAM)

        # Read true intrinsics for accurate 3D projection
        try:
            intrinsics = depth_profile.get_intrinsic()
            cam_fx = float(intrinsics.fx)
            cam_fy = float(intrinsics.fy)
            cam_cx = float(intrinsics.cx)
            cam_cy = float(intrinsics.cy)
            print(f"Using Live Intrinsics: fx={cam_fx:.1f}, fy={cam_fy:.1f}, cx={cam_cx:.1f}, cy={cam_cy:.1f}", flush=True)
        except Exception:
            cam_fx, cam_fy, cam_cx, cam_cy = DEPTH_FX_DEFAULT, DEPTH_FY_DEFAULT, DEPTH_CX_DEFAULT, DEPTH_CY_DEFAULT
            print("Could not read intrinsics. Using defaults.", flush=True)

        processor = OrbbecDepthProcessor(
            fx=cam_fx, fy=cam_fy, cx=cam_cx, cy=cam_cy,
            use_plane_correction=not args.disable_plane,
            use_height_pruning=not args.disable_pruning,
        )

    except Exception as e:
        print(f"Init Error: {e}", flush=True)
        return

    # 3. Optional ROS Publisher
    ros_pub = None
    if args.ros:
        ros_pub = ROSCornerPublisher(
            frame_id=args.ros_frame,
            force_udp=args.udp,
            udp_port=args.udp_port,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f"trtm_data_calibrated_{timestamp}"
    os.makedirs(save_folder, exist_ok=True)
    frame_count = 0
    save_feedback_timer = 0

    print("\nReady.", flush=True)
    print("1. Adjust 'Zoom' until cloth fits the GREEN BOX.", flush=True)
    print("2. Press 'S' to save.", flush=True)
    print("3. Press 'R' to reset temporal EMA smoothing.", flush=True)
    print("4. Press 'Q' or ESC to exit.\n", flush=True)

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
            depth_smooth = smooth_depth_map(depth_filled)

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
            masked_smooth = depth_smooth * binary_mask
            masked_raw = cv2.bitwise_and(depth_filled, depth_filled, mask=binary_mask)

            # 3b. Robot arm masking (when Arm Offset > 0)
            arm_mask_2d = np.zeros((height, width), dtype=np.uint8)
            if arm_offset_val > 0 and np.any(binary_mask > 0):
                arm_candidate = np.zeros((height, width), dtype=bool)
                if prev_pred_mm is not None:
                    ys, xs = np.where(depth_smooth > 0)
                    if len(ys) > 0:
                        zs = depth_smooth[ys, xs].astype(np.float32)
                        X = (xs.astype(np.float32) - cam_cx) / cam_fx * zs
                        Y = (ys.astype(np.float32) - cam_cy) / cam_fy * zs
                        pts_3d = np.stack([X, Y, zs], axis=1)

                        tree = cKDTree(prev_pred_mm)
                        _, idx = tree.query(pts_3d)
                        nearest_mesh_z = prev_pred_mm[idx, 2]
                        nearest_mesh_X = prev_pred_mm[idx, 0]
                        nearest_mesh_Y = prev_pred_mm[idx, 1]
                        dist_xy_sq = (X - nearest_mesh_X)**2 + (Y - nearest_mesh_Y)**2

                        arm_pts_mask = (zs < nearest_mesh_z - arm_offset_val) & (dist_xy_sq < 10000)
                        arm_candidate[ys[arm_pts_mask], xs[arm_pts_mask]] = True
                else:
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
                    binary_mask[arm_candidate] = 0
                    masked_smooth = depth_smooth * binary_mask
                    masked_color[arm_candidate] = 0

            # 3. 3D point cloud & Table Plane fitting (fit plane BEFORE shading to eliminate tilt)
            pts_3d = depth_image_to_pcd(masked_smooth, binary_mask, cam_fx, cam_fy, cam_cx, cam_cy)
            table_plane = None
            if len(pts_3d) > 0 and not args.disable_plane:
                try:
                    table_plane = processor.fit_table_plane(depth_smooth, background_mask, pts_3d)
                except Exception:
                    table_plane = None

            # 4. TRTM Inverted Shading (with plane correction for uniform 192 flat cloth)
            trtm_save_gray, trtm_vis_bgr, table_z = apply_trtm_shading(
                masked_smooth, background_mask,
                fx=cam_fx, fy=cam_fy, cx=cam_cx, cy=cam_cy,
                table_plane=table_plane,
                offset_mm=offset_val, max_table_depth=args.max_depth
            )

            # 5. Generate Views
            final_trtm_view = crop_and_resize(
                trtm_vis_bgr, crop_center, crop_size_val, TARGET_CANVAS_SIZE, bg_color=0
            )
            final_color_view = crop_and_resize(
                masked_color, crop_center, crop_size_val, TARGET_CANVAS_SIZE, bg_color=0
            )

            # Target region guide box
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

            # 6. Live Inference (TRTM GNN)
            if model is not None and np.any(binary_mask > 0):
                # Feed pure TRTM image: WHITE background (bg_color=255), inverted shading, NO green boxes/text
                trtm_model_input = crop_and_resize(
                    trtm_save_gray, crop_center, crop_size_val, TARGET_CANVAS_SIZE, bg_color=255
                )
                if trtm_model_input.ndim == 2:
                    trtm_model_input_bgr = cv2.cvtColor(trtm_model_input, cv2.COLOR_GRAY2BGR)
                else:
                    trtm_model_input_bgr = trtm_model_input

                trtm_resized = cv2.resize(trtm_model_input_bgr, (224, 224), interpolation=cv2.INTER_AREA)
                trtm_tensor = (
                    torch.from_numpy(trtm_resized.transpose(2, 0, 1))
                    .float()
                    .unsqueeze(0)
                    .div(255.0)
                    .to(device)
                )

                t0_infer = time.perf_counter()
                with torch.no_grad():
                    pred_norm = model(trtm_tensor)[0].cpu().numpy()  # [2601, 3]
                t_inf = (time.perf_counter() - t0_infer) * 1000.0

                # Base image to draw the projection on: start directly from final_color_view for zero distortion
                final_inference_view = final_color_view.copy()

                # Native 2D canvas coordinates (Target canvas is 720x720, cloth target is 480px)
                # In TRTM: u = 360 + x * 240, v = 360 - y * 240
                u_canvas = center_px + pred_norm[:, 0] * half_ideal
                v_canvas = center_px - pred_norm[:, 1] * half_ideal
                uvs_canvas = np.stack([u_canvas, v_canvas], axis=1).astype(np.int32)

                # 3D metric coordinates in camera frame (for ROS publishing)
                u_cam = crop_center[0] + (u_canvas - center_px) * (crop_size_val / TARGET_CANVAS_SIZE)
                v_cam = crop_center[1] + (v_canvas - center_px) * (crop_size_val / TARGET_CANVAS_SIZE)
                h_mm = pred_norm[:, 2] * 1000.0
                if table_plane is not None:
                    A, B, C = table_plane
                    z_table_pts = (A * (u_cam - cam_cx) / cam_fx + B * (v_cam - cam_cy) / cam_fy) * table_z + C
                else:
                    z_table_pts = table_z
                z_cam = z_table_pts - h_mm
                x_cam = (u_cam - cam_cx) / cam_fx * z_cam
                y_cam = (v_cam - cam_cy) / cam_fy * z_cam
                pred_mm = np.stack([x_cam, y_cam, z_cam], axis=1).astype(np.float32)

                # Temporal EMA Smoothing
                old_pred_mm = prev_pred_mm
                if prev_pred_mm is None:
                    prev_pred_mm = pred_mm.copy()
                else:
                    pred_mm = args.ema_alpha * pred_mm + (1.0 - args.ema_alpha) * prev_pred_mm
                    prev_pred_mm = pred_mm.copy()

                corners_mm_raw = pred_mm[corner_idx]
                corners_mm_tgc = corners_mm_raw.copy()
                tgc_active = False
                occluded = []

                # Template-guided corner completion (TGC) under robot arm
                n_arm_px = int(arm_mask_2d.sum()) if arm_mask_2d is not None else 0
                if arm_mask_2d is not None and n_arm_px >= MIN_ARM_PIXELS_FOR_TGC:
                    uvs_c = project_to_2d(corners_mm_raw, cam_fx, cam_fy, cam_cx, cam_cy)
                    visible, occluded = [], []
                    for i, (u_c, v_c) in enumerate(uvs_c):
                        is_arm = False
                        for du in range(-5, 6):
                            for dv in range(-5, 6):
                                nu, nv = u_c + du, v_c + dv
                                if (0 <= nu < width) and (0 <= nv < height) and arm_mask_2d[nv, nu] > 0:
                                    is_arm = True
                                    break
                            if is_arm: break
                        (occluded if is_arm else visible).append(i)

                    if occluded and len(visible) >= 3:
                        tgc_active = True
                        q_temp = np.loadtxt(args.template).astype(np.float32)
                        temp_corners = q_temp[corner_idx].astype(np.float32)
                        src = temp_corners[visible]
                        dst = corners_mm_raw[visible].astype(np.float32)
                        ones = np.ones((len(visible), 1), np.float32)
                        src_h = np.hstack([src, ones])
                        Mt, _, _, _ = np.linalg.lstsq(src_h, dst, rcond=None)
                        for i in occluded:
                            tc_h = np.append(temp_corners[i], 1.0)
                            corners_mm_tgc[i] = tc_h @ Mt

                # 8. Draw on the cropped RGB view (RIGHT PANEL)
                # A. 51x51 quadrilateral grid wireframe (cyan)
                draw_grid_wireframe(final_inference_view, uvs_canvas, grid_size=51, color=(0, 230, 255), thickness=1)

                # B. Mesh node points (blue dots)
                for u_pt, v_pt in uvs_canvas[::4]:
                    if 0 <= u_pt < TARGET_CANVAS_SIZE and 0 <= v_pt < TARGET_CANVAS_SIZE:
                        cv2.circle(final_inference_view, (u_pt, v_pt), 2, (255, 120, 0), -1)

                # C. Corners: TL (2550), TR (2600), BL (0), BR (50)
                corners_uv = uvs_canvas[corner_idx]
                for i, lbl in enumerate(corner_labels):
                    u_crn, v_crn = corners_uv[i]
                    is_corrected = tgc_active and (i in occluded)
                    crn_col = (0, 255, 0) if is_corrected else (0, 0, 255)
                    if 0 <= u_crn < TARGET_CANVAS_SIZE and 0 <= v_crn < TARGET_CANVAS_SIZE:
                        cv2.circle(final_inference_view, (u_crn, v_crn), 6, crn_col, -1)
                        cv2.putText(
                            final_inference_view, lbl, (u_crn + 8, v_crn - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, crn_col, 2
                        )

                cv2.putText(final_inference_view, f"Pred: {t_inf:.1f}ms", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                arm_hud = f"Arm px: {n_arm_px}  (min={MIN_ARM_PIXELS_FOR_TGC})"
                arm_hud_color = (0, 255, 0) if n_arm_px >= MIN_ARM_PIXELS_FOR_TGC else (0, 140, 255)
                cv2.putText(final_inference_view, arm_hud, (20, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, arm_hud_color, 1)
                tgc_txt = f"TGC: {'ON (' + str(len(occluded)) + ' corrected)' if tgc_active else 'off'}"
                cv2.putText(final_inference_view, tgc_txt, (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0) if tgc_active else (100, 100, 100), 1)

                # Optional ROS publish
                if ros_pub is not None:
                    active_corners = corners_mm_tgc if tgc_active else corners_mm_raw
                    ros_pub.publish(active_corners / 1000.0, pred_m=pred_mm / 1000.0)

                # Stack horizontally: [LEFT: Calibrated Depth | RIGHT: Live 3D Mesh on Cloth]
                combined = np.hstack((final_trtm_view, final_inference_view))
            else:
                combined = np.hstack((final_color_view, final_trtm_view))

            # Common UI overlays
            cv2.putText(
                combined,
                f"Z: {table_z/10:.1f}cm | Crop: {crop_size_val} | Off: {offset_val}mm",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

            if save_feedback_timer > 0:
                cv2.putText(
                    combined, "SAVED!",
                    (combined.shape[1] // 2 - 100, combined.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4
                )
                save_feedback_timer -= 1

            # Scale to fits comfortably on screen
            display_img = cv2.resize(combined, (0, 0), fx=0.8, fy=0.8)
            cv2.imshow("Master Calibration", display_img)

            # --- SAVE ---
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s') or key == ord('S'):
                # TRTM (Inverted, Smooth) → White BG
                trtm_save_final = crop_and_resize(
                    trtm_save_gray, crop_center, crop_size_val, TARGET_CANVAS_SIZE, bg_color=255
                )
                raw_save_final = crop_and_resize(
                    masked_raw, crop_center, crop_size_val, TARGET_CANVAS_SIZE, bg_color=0
                )
                color_save_final = crop_and_resize(
                    masked_color, crop_center, crop_size_val, TARGET_CANVAS_SIZE, bg_color=0
                )

                cv2.imwrite(os.path.join(save_folder, f"color_{frame_count:05d}.png"), color_save_final)
                cv2.imwrite(os.path.join(save_folder, f"depth_raw_{frame_count:05d}.png"), raw_save_final)
                cv2.imwrite(os.path.join(save_folder, f"depth_trtm_{frame_count:05d}.png"), trtm_save_final)
                print(f"[{frame_count}] Saved. Crop: {crop_size_val}, Offset: {offset_val}")
                frame_count += 1
                save_feedback_timer = 5

            elif key == ord('r') or key == ord('R'):
                prev_pred_mm = None
                print("[User] Reset EMA smoothing.")

            if key == ord('q') or key == ESC_KEY:
                break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline_sdk.stop()
        if ros_pub is not None:
            ros_pub.destroy()
        cv2.destroyAllWindows()
        print("\n[Shutdown] Clean exit completed.")


if __name__ == "__main__":
    main()
