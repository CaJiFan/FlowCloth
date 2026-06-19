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
from pyorbbecsdk import *

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


# --- CONFIGURATION ---
ESC_KEY = 27

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
DEFAULT_THRESHOLD = 100
DEFAULT_OFFSET = 0


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


def apply_trtm_shading(depth_map_float, mask_bg, offset_mm=0):
    """Calculates TRTM values using Float Depth + Inverted Logic."""
    valid_pixels = depth_map_float[depth_map_float > 0]

    if MANUAL_TABLE_DEPTH:
        table_depth = MANUAL_TABLE_DEPTH
    elif len(valid_pixels) > 0:
        table_depth = np.percentile(valid_pixels, 98)
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


def segment_cloth_threshold(image, threshold=30):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.zeros_like(binary)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            cv2.drawContours(final_mask, [largest], -1, 255, -1)
    return (final_mask // 255).astype(np.uint8)


def nothing(x):
    pass


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
    """Load the template mesh from the .pkl file."""
    # Handle relative paths from the repo root
    patch_file = config.model_cfg.patch_file
    if not os.path.isabs(patch_file):
        patch_file = os.path.join(_REPO_ROOT, patch_file)
        
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
        pts = pts[contour]

    pts -= pts.mean(axis=0)
    sc   = float(np.max(np.linalg.norm(pts, axis=1)))
    if sc > 1e-6:
        pts /= sc

    return pts


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


def project_to_2d(pts_3d_mm, fx, fy, cx, cy):
    """Project 3D points in camera frame back to 2D image coordinates."""
    xs = (pts_3d_mm[:, 0] * fx / pts_3d_mm[:, 2]) + cx
    ys = (pts_3d_mm[:, 1] * fy / pts_3d_mm[:, 2]) + cy
    return np.stack([xs, ys], axis=1).astype(np.int32)


def main():
    parser = argparse.ArgumentParser(description="Orbbec calibration & live inference GUI")
    parser.add_argument("--ckpt_dir", default=None, help="Path to checkpoint directory (if provided, enables the 3rd inference window)")
    parser.add_argument("--mode", choices=["edge", "full"], default="edge", help="Model mode")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of inference steps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load model if checkpoint provided
    pipeline = None
    q_temp = None
    use_fm = False
    if args.ckpt_dir:
        print(f"Loading checkpoint from: {args.ckpt_dir}")
        # Find config.yml in the parent experiment dir
        exp_dir = os.path.dirname(os.path.dirname(args.ckpt_dir))
        config_path = os.path.join(exp_dir, "config.yml")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"config.yml not found in {exp_dir}")
        config = OmegaConf.load(config_path)
        
        pipeline, use_fm = load_pipeline(config, args.ckpt_dir, device)
        q_temp = load_template(config, args.mode == "edge")
        print("Model loaded successfully. Inference window enabled.")

    pipeline_sdk = Pipeline()
    config_sdk = Config()

    try:
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
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

    except Exception as e:
        print(f"Init Error: {e}")
        return

    # --- UI SETUP ---
    cv2.namedWindow("Master Calibration")
    cv2.createTrackbar("Zoom (Crop Size)", "Master Calibration", DEFAULT_CROP_SIZE, 800, nothing)
    cv2.createTrackbar("Threshold", "Master Calibration", DEFAULT_THRESHOLD, 100, nothing)
    cv2.createTrackbar("Table Offset", "Master Calibration", DEFAULT_OFFSET, 100, nothing)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_folder = f"trtm_data_calibrated_{timestamp}"
    os.makedirs(save_folder, exist_ok=True)
    frame_count = 0
    save_feedback_timer = 0

    print("\nReady.")
    print("1. Adjust 'Zoom' until cloth fits the GREEN BOX.")
    print("2. Press 'S' to save.")

    try:
        while True:
            frames = pipeline_sdk.wait_for_frames(100)
            if not frames:
                continue
            frames = align_filter.process(frames)
            if not frames:
                continue
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
            thresh_val = cv2.getTrackbarPos("Threshold", "Master Calibration")
            offset_val = cv2.getTrackbarPos("Table Offset", "Master Calibration")
            if crop_size_val < 100:
                crop_size_val = 100

            # 1. Depth processing pipeline
            depth_filled = fill_depth_holes(depth_raw_mm)
            depth_smooth = smooth_depth_map(depth_filled)  # Float smoothing

            # 2. Cloth segmentation and dynamic centre
            binary_mask = segment_cloth_threshold(color_image, threshold=thresh_val)
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

            # 3. TRTM Inverted Shading
            trtm_save_gray, trtm_vis_bgr, table_z = apply_trtm_shading(
                masked_smooth, background_mask, offset_mm=offset_val
            )

            # 4. Generate Views
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

            # 5. Live Inference (if loaded)
            if pipeline is not None:
                # Get point cloud from the cropped cloth mask
                pts_3d = depth_image_to_pcd(masked_smooth, binary_mask, cam_fx, cam_fy, cam_cx, cam_cy)
                
                # Base image to draw the projection on (copy of masked color)
                inference_view_raw = masked_color.copy()
                
                if len(pts_3d) > 0:
                    pts_norm, com, scale = normalise_pcd(pts_3d)
                    pcd_input = subsample_pcd(pts_norm, 400)
                    
                    # Run model
                    t0 = time.time()
                    pred_norm = run_inference(pipeline, use_fm, pcd_input, q_temp, args.num_steps, device)
                    t_inf = (time.time() - t0) * 1000
                    
                    # Back-project to 2D
                    pred_mm = OrbbecDepthProcessor.denormalise_prediction(pred_norm, com, scale)
                    uvs = project_to_2d(pred_mm, cam_fx, cam_fy, cam_cx, cam_cy)
                    
                    # Draw on the raw uncropped image
                    if args.mode == "edge":
                        # The edge nodes form a contour loop
                        cv2.polylines(inference_view_raw, [uvs], isClosed=True, color=(0, 255, 255), thickness=2)
                        # Emphasize corners (assuming TRTM 80 nodes -> 0, 20, 59, 79 or VR 76 nodes -> 0, 19, 56, 75)
                        # This works automatically since the corners are explicitly known in the contour
                        for idx in [0, len(uvs)//4, (len(uvs)//4)*3-1, len(uvs)-1]: # Rough corner approx for visual
                            if idx < len(uvs):
                                cv2.circle(inference_view_raw, tuple(uvs[idx]), 6, (0, 0, 255), -1)
                    else:
                        # Draw full scatter points
                        for u, v in uvs:
                            cv2.circle(inference_view_raw, (u, v), 2, (0, 255, 255), -1)
                            
                    # Crop and resize the inference view to match the others
                    final_inference_view = crop_and_resize(
                        inference_view_raw, crop_center, crop_size_val, TARGET_CANVAS_SIZE, bg_color=0
                    )
                    cv2.putText(final_inference_view, f"Pred: {t_inf:.1f}ms", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    # Empty view if no cloth found
                    final_inference_view = crop_and_resize(
                        inference_view_raw, crop_center, crop_size_val, TARGET_CANVAS_SIZE, bg_color=0
                    )
                
                # Stack all three windows horizontally
                combined = np.hstack((final_color_view, final_trtm_view, final_inference_view))
            else:
                # Original 2-window behavior
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

            cv2.imshow("Master Calibration", combined)

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
                print(f"[{frame_count}] Saved. Crop: {crop_size_val}, Offset: {offset_val}")
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