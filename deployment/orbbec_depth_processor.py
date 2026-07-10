"""
orbbec_depth_processor.py
─────────────────────────
Orbbec camera depth pre-processing pipeline that converts raw depth frames
into normalised 3D point clouds ready for UniClothDiff state estimation.

Pipeline stages
───────────────
  1. Capture depth + color frames from the Orbbec SDK (live mode)
     OR load saved depth PNG files (offline mode).
  2. Fill depth holes (morphological close on uint16 depth).
  3. Gaussian smooth (float32, removes quantisation artefacts).
  4. Cloth segmentation via color thresholding.
  5. TRTM shading encoding (height-map → uint8 grayscale).
  6. Crop & resize to TARGET_CANVAS_SIZE × TARGET_CANVAS_SIZE.
  7. Back-project depth image → 3D point cloud [H×W, 3] in camera frame (mm).
  8. Cloth mask → keep only cloth pixels.
  9. Random subsample to `num_sample_points`.
 10. Normalise: subtract centre-of-mass, divide by max L2 norm.
     Saves the inverse transform (com, scale) so you can recover metric coords.

Returns a dict with:
  - pcd         : np.ndarray [num_sample_points, 3] — normalised, model-ready
  - com         : np.ndarray [3] — centre-of-mass in camera-frame mm
  - scale       : float — max L2 norm before normalisation (mm)
  - crop_center : (cx, cy) — pixel centre of cloth region
  - crop_size   : int — crop size used
  - trtm_image  : np.ndarray [720, 720] uint8 — TRTM-encoded depth (save/debug)
  - depth_raw_mm: np.ndarray [H, W] uint16 — raw depth map (mm) for records

Usage — live camera
───────────────────
  from scripts.orbbec_depth_processor import OrbbecDepthProcessor

  proc = OrbbecDepthProcessor(num_sample_points=400)
  proc.start()
  result = proc.capture_and_process()   # blocks until a valid frame arrives
  proc.stop()
  pcd = result["pcd"]   # [400, 3] float32, normalised

Usage — offline (saved PNG)
────────────────────────────
  proc = OrbbecDepthProcessor(num_sample_points=400)
  result = proc.process_saved_depth(
      depth_path="trtm_data_calibrated_20250101_120000/depth_raw_00000.png",
      color_path="trtm_data_calibrated_20250101_120000/color_00000.png",
  )
"""

from __future__ import annotations

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# TRTM / Camera constants  (must match orbbec_frame.py calibration session)
# ─────────────────────────────────────────────────────────────────────────────

TARGET_CANVAS_SIZE: int   = 720     # pixels — output image size
IDEAL_CLOTH_SIZE:   int   = 480     # pixels — expected cloth extent in canvas
FLAT_PIXEL_VAL:     int   = 192     # TRTM encoding neutral value
PIXEL_PER_MM:       float = 2.0     # encoding scale (1 mm → 2 px difference)
INVERT_SHADING:     bool  = True    # True = higher folds darker

# Default calibration for interactive cropping (adjust to your session values)
DEFAULT_CROP_SIZE:   int = 480
DEFAULT_THRESHOLD:   int = 100
DEFAULT_OFFSET_MM:   int = 0

# Orbbec Astra+ approximate intrinsics for the DEPTH stream at 640×480.
# Replace with your specific device's intrinsics from the SDK if available.
# You can print them by calling: depth_profile.get_intrinsic()
DEPTH_FX_DEFAULT: float = 580.0
DEPTH_FY_DEFAULT: float = 580.0
DEPTH_CX_DEFAULT: float = 320.0
DEPTH_CY_DEFAULT: float = 240.0


# ─────────────────────────────────────────────────────────────────────────────
# Pure-numpy preprocessing helpers (no SDK dependency)
# ─────────────────────────────────────────────────────────────────────────────

def fill_depth_holes(depth_map: np.ndarray) -> np.ndarray:
    """Morphological close on integer depth to fill small 0-value holes."""
    holes  = (depth_map == 0)
    kernel = np.ones((5, 5), np.uint8)
    filled = cv2.morphologyEx(depth_map, cv2.MORPH_CLOSE, kernel)
    result = depth_map.copy()
    result[holes] = filled[holes]
    return result


def smooth_depth_map(depth_mm: np.ndarray) -> np.ndarray:
    """Cast to float32 and apply Gaussian blur to remove quantisation steps."""
    return cv2.GaussianBlur(depth_mm.astype(np.float32), (5, 5), 0)


def segment_cloth_color(color_bgr: np.ndarray, threshold: int = 30) -> np.ndarray:
    """
    Simple luminance-threshold segmentation.
    Returns binary mask uint8 {0, 1} with cloth=1.
    """
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    k = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 500:
            cv2.drawContours(mask, [largest], -1, 255, -1)
    return (mask // 255).astype(np.uint8)


def cloth_centroid(mask: np.ndarray, fallback: Tuple[int, int]) -> Tuple[int, int]:
    """Return (cx, cy) centroid of a binary mask or fallback if mask is empty."""
    M = cv2.moments(mask)
    if M["m00"] != 0:
        return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    return fallback


def crop_and_resize(
    img: np.ndarray,
    center_xy: Tuple[int, int],
    crop_size: int,
    target_size: int = TARGET_CANVAS_SIZE,
    bg_color: int = 0,
) -> np.ndarray:
    """Crop a square region of `crop_size` around `center_xy`, resize to `target_size`."""
    h, w = img.shape[:2]
    is_rgb = img.ndim == 3
    cx, cy = center_xy
    half = crop_size // 2

    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + crop_size, y1 + crop_size

    pl = max(0, -x1); pt = max(0, -y1)
    pr = max(0,  x2 - w); pb = max(0, y2 - h)
    if any([pl, pt, pr, pb]):
        val   = [bg_color] * 3 if is_rgb else bg_color
        btype = cv2.BORDER_CONSTANT
        img   = cv2.copyMakeBorder(img, pt, pb, pl, pr, btype, value=val)
        x1 += pl; y1 += pt; x2 += pl; y2 += pt

    crop   = img[y1:y2, x1:x2]
    method = cv2.INTER_AREA if is_rgb else cv2.INTER_CUBIC
    return cv2.resize(crop, (target_size, target_size), interpolation=method)


def apply_trtm_shading(
    depth_float: np.ndarray,
    bg_mask: np.ndarray,
    offset_mm: float = 0.0,
    manual_table_depth: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert a masked float depth map to TRTM-encoded grayscale.

    Returns
    -------
    encoded_save : uint8 [H,W]  — white-BG version for saving
    vis_bgr      : uint8 [H,W,3]— black-BG BGR for display
    table_depth  : float         — estimated table depth in mm
    """
    valid = depth_float[depth_float > 0]
    if manual_table_depth is not None:
        table_depth = float(manual_table_depth)
    elif len(valid) > 0:
        table_depth = float(np.percentile(valid, 98))
    else:
        h, w = depth_float.shape
        return (np.zeros((h, w), np.uint8),
                np.zeros((h, w, 3), np.uint8),
                0.0)

    table_depth += offset_mm
    height_map  = table_depth - depth_float

    if INVERT_SHADING:
        processed = FLAT_PIXEL_VAL - height_map * PIXEL_PER_MM
    else:
        processed = FLAT_PIXEL_VAL + height_map * PIXEL_PER_MM
    processed = np.clip(processed, 0, 255).astype(np.uint8)

    encoded_save            = processed.copy()
    encoded_save[bg_mask]   = 255
    vis_bgr                 = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
    vis_bgr[bg_mask]        = [0, 0, 0]
    return encoded_save, vis_bgr, table_depth


# ─────────────────────────────────────────────────────────────────────────────
# Back-projection: depth image → 3-D point cloud
# ─────────────────────────────────────────────────────────────────────────────

def depth_image_to_pcd(
    depth_mm: np.ndarray,
    mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """
    Back-project masked depth pixels to 3-D points in camera frame.

    Parameters
    ----------
    depth_mm : float32 [H, W] — depth in millimetres
    mask     : uint8   [H, W] — cloth mask {0,1}
    fx, fy   : focal lengths in pixels
    cx, cy   : principal point in pixels

    Returns
    -------
    pts : float32 [M, 3] — XYZ in millimetres (camera frame, Z = depth)
    """
    h, w = depth_mm.shape
    ys, xs = np.where(mask > 0)
    zs = depth_mm[ys, xs].astype(np.float32)

    # Remove invalid (zero) depth points
    valid = zs > 0
    xs, ys, zs = xs[valid], ys[valid], zs[valid]

    X = (xs.astype(np.float32) - cx) / fx * zs
    Y = (ys.astype(np.float32) - cy) / fy * zs
    return np.stack([X, Y, zs], axis=1)   # [M, 3]


def normalise_pcd(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Normalise a point cloud to zero-mean, unit max-L2-norm.

    Returns
    -------
    pts_norm : float32 [M, 3]
    com      : float32 [3]     — original centre-of-mass
    scale    : float            — max L2 norm before normalisation
    """
    com   = pts.mean(axis=0)
    pts_c = pts - com
    scale = float(np.max(np.linalg.norm(pts_c, axis=1)))
    if scale < 1e-6:
        scale = 1.0
    return (pts_c / scale).astype(np.float32), com, scale


def subsample_pcd(pts: np.ndarray, n: int, rng: np.random.Generator = None) -> np.ndarray:
    """Random sub-sample (with replacement if needed) to exactly n points."""
    rng = rng or np.random.default_rng()
    if len(pts) >= n:
        idx = rng.choice(len(pts), n, replace=False)
    else:
        idx = rng.choice(len(pts), n, replace=True)
    return pts[idx]


# ─────────────────────────────────────────────────────────────────────────────
# Main processor class
# ─────────────────────────────────────────────────────────────────────────────

class OrbbecDepthProcessor:
    """
    Full pre-processing pipeline from raw Orbbec depth to model-ready PCD.

    Parameters
    ----------
    num_sample_points : int
        Number of 3-D points to pass to the model (default 400, from config).
    crop_size : int
        Square crop around cloth centroid before resizing (default 480 px).
    seg_threshold : int
        Luminance threshold for cloth/background segmentation (0-255).
    offset_mm : float
        Additional mm to add to the estimated table depth for TRTM encoding.
    fx, fy, cx, cy : float
        Camera intrinsics for the depth stream. If None, defaults are used.
        You can read these at runtime via `depth_profile.get_intrinsic()`.
    """

    def __init__(
        self,
        num_sample_points: int = 400,
        crop_size: int = DEFAULT_CROP_SIZE,
        seg_threshold: int = DEFAULT_THRESHOLD,
        offset_mm: float = float(DEFAULT_OFFSET_MM),
        fx: float = DEPTH_FX_DEFAULT,
        fy: float = DEPTH_FY_DEFAULT,
        cx: float = DEPTH_CX_DEFAULT,
        cy: float = DEPTH_CY_DEFAULT,
    ):
        self.num_sample_points = num_sample_points
        self.crop_size         = crop_size
        self.seg_threshold     = seg_threshold
        self.offset_mm         = offset_mm
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # Live-capture state (populated by start())
        self._pipeline       = None
        self._align_filter   = None
        self._rng            = np.random.default_rng()

    # ------------------------------------------------------------------
    # Live camera management
    # ------------------------------------------------------------------

    def start(self, color_width: int = 1280, color_height: int = 720):
        """Initialise and start the Orbbec pipeline."""
        try:
            from pyorbbecsdk import (
                Pipeline, Config, OBSensorType, OBFormat, OBAlignMode,
                AlignFilter, OBStreamType
            )
        except ImportError:
            raise ImportError(
                "pyorbbecsdk is not installed. "
                "Cannot use live capture; use process_saved_depth() instead."
            )

        pipeline = Pipeline()
        cfg      = Config()

        # Color stream
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile  = None
        for w, h in [(color_width, color_height), (640, 480)]:
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
            raise RuntimeError("No RGB color profile found on the Orbbec device.")

        cfg.enable_stream(color_profile)

        # Depth stream
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile  = depth_profiles.get_default_video_stream_profile()
        cfg.enable_stream(depth_profile)

        # Try to read real intrinsics from the device
        try:
            intr = depth_profile.get_intrinsic()
            self.fx = float(intr.fx)
            self.fy = float(intr.fy)
            self.cx = float(intr.cx)
            self.cy = float(intr.cy)
            print(f"[OrbbecDepthProcessor] Using device intrinsics: "
                  f"fx={self.fx:.1f}, fy={self.fy:.1f}, "
                  f"cx={self.cx:.1f}, cy={self.cy:.1f}")
        except Exception:
            print("[OrbbecDepthProcessor] Could not read intrinsics from device; "
                  f"using defaults: fx={self.fx}, fy={self.fy}.")

        cfg.set_align_mode(OBAlignMode.DISABLE)
        pipeline.start(cfg)
        self._align_filter = AlignFilter(align_to_stream=OBStreamType.DEPTH_STREAM)
        self._pipeline     = pipeline
        print("[OrbbecDepthProcessor] Camera started.")

    def stop(self):
        """Stop the Orbbec pipeline and release resources."""
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        print("[OrbbecDepthProcessor] Camera stopped.")

    # ------------------------------------------------------------------
    # Core processing (shared between live and offline modes)
    # ------------------------------------------------------------------

    def _process_raw_frames(
        self,
        depth_raw_mm: np.ndarray,   # uint16 [H, W] in mm
        color_bgr:    np.ndarray,   # uint8  [H, W, 3]
    ) -> dict:
        """
        Run the full pre-processing pipeline on raw depth + color arrays.

        Returns a dict with all intermediate and final outputs.
        """
        h, w = depth_raw_mm.shape

        # Stage 1-2: hole-fill + smooth
        depth_filled = fill_depth_holes(depth_raw_mm)
        depth_smooth = smooth_depth_map(depth_filled)

        # Stage 3: cloth segmentation
        binary_mask     = segment_cloth_color(color_bgr, threshold=self.seg_threshold)
        background_mask = (binary_mask == 0)
        crop_center     = cloth_centroid(binary_mask, fallback=(w // 2, h // 2))

        masked_smooth = depth_smooth * binary_mask    # float32
        masked_raw    = depth_filled * binary_mask    # uint16

        # Stage 4: TRTM encoding
        trtm_gray, _trtm_vis, table_z = apply_trtm_shading(
            masked_smooth, background_mask, offset_mm=self.offset_mm
        )

        # Stage 5: Crop & resize (for TRTM image saving/debug)
        trtm_cropped = crop_and_resize(
            trtm_gray, crop_center, self.crop_size, TARGET_CANVAS_SIZE, bg_color=255
        )

        # Stage 6: Back-project depth → 3-D cloud in camera frame (mm)
        # Use the *masked* smooth depth for back-projection for best quality.
        pts_3d = depth_image_to_pcd(
            masked_smooth, binary_mask,
            self.fx, self.fy, self.cx, self.cy
        )

        if len(pts_3d) == 0:
            print("[OrbbecDepthProcessor] WARNING: no valid cloth points found!")
            pts_3d = np.zeros((1, 3), np.float32)

        # Stage 7: Normalise
        pts_norm, com, scale = normalise_pcd(pts_3d)

        # Stage 8: Subsample to model input size
        pcd = subsample_pcd(pts_norm, self.num_sample_points, self._rng)

        return {
            "pcd":          pcd.astype(np.float32),     # [num_sample_points, 3]
            "com":          com.astype(np.float32),     # [3] mm, camera frame
            "scale":        float(scale),               # mm
            "table_depth_mm": float(table_z),
            "crop_center":  crop_center,                # (cx, cy) pixels
            "crop_size":    self.crop_size,
            "trtm_image":   trtm_cropped,               # [720,720] uint8
            "depth_raw_mm": masked_raw,                 # [H,W] uint16
            "cloth_mask":   binary_mask,                # [H,W] uint8
            "n_cloth_pts":  len(pts_3d),
        }

    # ------------------------------------------------------------------
    # Live capture
    # ------------------------------------------------------------------

    def capture_and_process(self, timeout_ms: int = 2000) -> dict:
        """
        Capture one depth+color frame from the live Orbbec camera and process it.

        Parameters
        ----------
        timeout_ms : int
            SDK frame wait timeout in milliseconds.

        Returns
        -------
        dict — same as _process_raw_frames()
        """
        if self._pipeline is None:
            raise RuntimeError("Call start() before capture_and_process().")

        for attempt in range(20):
            frames = self._pipeline.wait_for_frames(timeout_ms // 20)
            if not frames:
                continue
            frames = self._align_filter.process(frames)
            if not frames:
                continue
            frames      = frames.as_frame_set()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # --- extract depth ---
            dh, dw = depth_frame.get_height(), depth_frame.get_width()
            depth_data   = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((dh, dw))
            depth_raw_mm = (depth_data * depth_frame.get_depth_scale()).astype(np.uint16)

            # --- extract color (BGR) ---
            ch, cw = color_frame.get_height(), color_frame.get_width()
            color_data = np.asanyarray(color_frame.get_data())
            color_rgb  = np.resize(color_data, (ch, cw, 3))
            color_bgr  = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

            # Resize colour to match depth resolution
            if color_bgr.shape[:2] != (dh, dw):
                color_bgr = cv2.resize(color_bgr, (dw, dh))

            return self._process_raw_frames(depth_raw_mm, color_bgr)

        raise RuntimeError(
            f"Could not capture a valid frame after {20} attempts "
            f"(total timeout {timeout_ms} ms)."
        )

    # ------------------------------------------------------------------
    # Offline / saved-PNG mode
    # ------------------------------------------------------------------

    def process_saved_depth(
        self,
        depth_path: str,
        color_path: Optional[str] = None,
    ) -> dict:
        """
        Process a saved depth PNG (uint16, values in mm as written by orbbec_frame.py).

        Parameters
        ----------
        depth_path : str
            Path to the raw uint16 depth PNG (saved as depth_raw_*.png).
        color_path : str or None
            Optional path to the corresponding color PNG for segmentation.
            If None, a simple depth-threshold is used instead.

        Returns
        -------
        dict — same as _process_raw_frames()
        """
        depth_raw_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_raw_mm is None:
            raise FileNotFoundError(f"Depth image not found: {depth_path}")
        if depth_raw_mm.dtype != np.uint16:
            depth_raw_mm = depth_raw_mm.astype(np.uint16)

        if color_path is not None and os.path.isfile(color_path):
            color_bgr = cv2.imread(color_path)
        else:
            # Fallback: synthesise a pseudo-color image from depth for segmentation
            print("[OrbbecDepthProcessor] No color image — using depth-based segmentation.")
            depth_vis  = cv2.normalize(depth_raw_mm, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            color_bgr  = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

        # Resize colour to match depth if needed
        dh, dw = depth_raw_mm.shape
        if color_bgr.shape[:2] != (dh, dw):
            color_bgr = cv2.resize(color_bgr, (dw, dh))

        return self._process_raw_frames(depth_raw_mm, color_bgr)

    # ------------------------------------------------------------------
    # Utility: recover metric coordinates from normalised prediction
    # ------------------------------------------------------------------

    @staticmethod
    def denormalise_prediction(
        pred_norm: np.ndarray,   # [N, 3] normalised
        com: np.ndarray,         # [3] mm
        scale: float,            # mm
    ) -> np.ndarray:
        """
        Convert model output (normalised [N,3]) back to camera-frame mm.

        Usage
        -----
        result    = proc.capture_and_process()
        pred_mm   = OrbbecDepthProcessor.denormalise_prediction(
                        pred[0], result["com"], result["scale"]
                    )
        corners_mm = pred_mm[corner_indices]  # [4, 3] in mm, camera frame
        """
        return pred_norm * scale + com


# ─────────────────────────────────────────────────────────────────────────────
# Quick standalone test (no robot, no SDK required)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os, sys, argparse

    parser = argparse.ArgumentParser(
        description="Test OrbbecDepthProcessor on a saved depth PNG."
    )
    parser.add_argument("--depth", required=True, help="Path to depth_raw_*.png")
    parser.add_argument("--color", default=None,  help="Path to color_*.png")
    parser.add_argument("--n",     type=int, default=400, help="num_sample_points")
    parser.add_argument("--fx",    type=float, default=DEPTH_FX_DEFAULT)
    parser.add_argument("--fy",    type=float, default=DEPTH_FY_DEFAULT)
    parser.add_argument("--cx",    type=float, default=DEPTH_CX_DEFAULT)
    parser.add_argument("--cy",    type=float, default=DEPTH_CY_DEFAULT)
    args = parser.parse_args()

    proc = OrbbecDepthProcessor(
        num_sample_points=args.n,
        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
    )
    result = proc.process_saved_depth(args.depth, args.color)

    print("\n── Preprocessing result ──────────────────────────────────")
    print(f"  pcd shape      : {result['pcd'].shape}")
    print(f"  pcd range      : [{result['pcd'].min():.3f}, {result['pcd'].max():.3f}]")
    print(f"  cloth points   : {result['n_cloth_pts']}")
    print(f"  COM (mm)       : {result['com']}")
    print(f"  scale (mm)     : {result['scale']:.1f}")
    print(f"  table depth    : {result['table_depth_mm']:.1f} mm")
    print(f"  TRTM image     : {result['trtm_image'].shape}, dtype={result['trtm_image'].dtype}")
    print("──────────────────────────────────────────────────────────\n")

    # Show TRTM image if display is available
    try:
        cv2.imshow("TRTM Preview", result["trtm_image"])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass

    # Optional: 3-D scatter plot with matplotlib
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        pcd = result["pcd"]
        fig = plt.figure(figsize=(7, 5))
        ax  = fig.add_subplot(111, projection="3d")
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=2, c=pcd[:, 2], cmap="viridis")
        ax.set_title(f"PCD (normalised) — {args.n} points")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()
    except ImportError:
        pass
