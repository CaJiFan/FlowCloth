import sys
import os
import argparse
import numpy as np
import torch
import cv2
from omegaconf import OmegaConf

# Add project root and deployment directory to path
sys.path.insert(0, "/home/cjimenez/Projects/FlowCloth")
sys.path.insert(0, "/home/cjimenez/Projects/FlowCloth/deployment")

from deployment.orbbec_depth_processor import OrbbecDepthProcessor, segment_cloth_color
from deployment.orbbec_overlay_3d import load_pipeline, load_template, run_inference
from uniclothdiff.datasets.trtm_dataset import TRTMDiffusionDataset

def fit_plane_ransac(pts, n_iter=200, threshold=3.0):
    best_coef = None
    best_inliers = 0
    n = len(pts)
    if n < 3:
        return None
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        p1, p2, p3 = pts[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        if abs(normal[2]) < 1e-2:
            continue
        A = -normal[0] / normal[2]
        B = -normal[1] / normal[2]
        C = np.dot(normal, p1) / normal[2]
        z_plane = A * pts[:, 0] + B * pts[:, 1] + C
        dist = np.abs(z_plane - pts[:, 2])
        inliers = np.sum(dist < threshold)
        if inliers > best_inliers:
            best_inliers = inliers
            best_coef = (A, B, C)
    return best_coef

def main():
    parser = argparse.ArgumentParser(description="Test TRTM Model Inference.")
    parser.add_argument(
        "--mode", type=str, default="local", choices=["local", "trtm"],
        help="local: run on raw frames with plane correction. trtm: run on TRTM real-test split."
    )
    parser.add_argument(
        "--sample_dir", type=str,
        default="/home/cjimenez/Projects/FlowCloth/trtm_data_calibrated_20260731_174657",
        help="Directory containing local color_*.png and depth_raw_*.png."
    )
    parser.add_argument(
        "--data_dir", type=str,
        default="/home/cjimenez/Downloads/dataset_trtm/trtm_orbbec_tuning_20260129_120342",
        help="Path to TRTM dataset folder containing real/test."
    )
    parser.add_argument(
        "--idx", type=int, default=0,
        help="Sample index to test when using --mode trtm."
    )
    parser.add_argument(
        "--ckpt_dir", type=str,
        default="/home/cjimenez/Projects/FlowCloth/checkpoints/ablation_fm_sw100.3_s3/checkpoints/checkpoint-best/",
        help="Path to model checkpoint folder."
    )
    parser.add_argument(
        "--steps", type=int, default=10,
        help="Number of inference steps."
    )
    parser.add_argument(
        "--disable_plane", action="store_true",
        help="Disable RANSAC table plane calibration (falls back to flat table)"
    )
    parser.add_argument(
        "--disable_pruning", action="store_true",
        help="Disable outlier height pruning"
    )
    args = parser.parse_args()

    # Load config and model pipeline
    config_path = os.path.join(args.ckpt_dir, "../../config.yml")
    config = OmegaConf.load(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    pipeline, use_fm = load_pipeline(config, args.ckpt_dir, device)
    q_temp = load_template(config, is_edge=False)
    corner_indices = [0, 20, 420, 440]
    corner_names = ["TL", "TR", "BL", "BR"]

    if args.mode == "local":
        print(f"\n=== MODE: LOCAL (Processing raw frames with table plane fitting) ===")
        depth_path = os.path.join(args.sample_dir, "depth_raw_00000.png")
        color_path = os.path.join(args.sample_dir, "color_00000.png")
        
        if not os.path.exists(depth_path) or not os.path.exists(color_path):
            raise FileNotFoundError(f"Could not find local frames in {args.sample_dir}")

        proc = OrbbecDepthProcessor(
            use_plane_correction=not args.disable_plane,
            use_height_pruning=not args.disable_pruning
        )
        depth_raw_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.uint16)
        color_bgr = cv2.imread(color_path)

        result = proc._process_raw_frames(depth_raw_mm, color_bgr)
        pcd_new = result["pcd"]
        com = result["com"]
        scale = result["scale"]
        plane_coef = result["table_plane"]

        if plane_coef is not None:
            A, B, C = plane_coef
            print(f"Fitted Table Plane: Z = {A:.4f} * X + {B:.4f} * Y + {C:.2f}")
        else:
            print("Table plane fitting disabled (using horizontal fallback).")
        print(f"Total processed points: {result['n_cloth_pts']}")

    else:
        print(f"\n=== MODE: TRTM (Testing on TRTM real-test split, sample index={args.idx}) ===")
        # Ensure the template pickle file exists where the dataset loader expects it
        pickle_path = f"{args.data_dir}.pickle"
        if not os.path.exists(pickle_path):
            print(f"Template pickle not found at {pickle_path}. Generating from assets...")
            asset_path = "/home/cjimenez/Projects/FlowCloth/assets/trtm_441_voronoi_template.pkl"
            if not os.path.exists(asset_path):
                raise FileNotFoundError(f"Could not find template asset at {asset_path}")
            import pickle as pkl
            with open(asset_path, "rb") as f_asset:
                asset_data = pkl.load(f_asset)
            # Reformat to expected structure
            trtm_data = {"mesh_pos": asset_data["points"]}
            with open(pickle_path, "wb") as f_trtm:
                pkl.dump(trtm_data, f_trtm)
            print(f"Template pickle generated successfully at {pickle_path}.")

        # We instantiate the TRTM dataset loader
        dataset = TRTMDiffusionDataset(data_dir=args.data_dir, mode="real_test")
        if args.idx < 0 or args.idx >= len(dataset):
            raise IndexError(f"Sample index {args.idx} is out of bounds for dataset of size {len(dataset)}")
            
        sample = dataset[args.idx]
        pcd_new = sample["pcd"].numpy()
        print(f"Loaded normalized PCD from TRTM dataset. Shape: {pcd_new.shape}")
        
        # TRTM dataset samples are already zero-centered and scaled
        com = np.zeros(3, dtype=np.float32)
        scale = 1.0

    # Run inference
    print(f"Running inference with {args.steps} steps...")
    pred_norm = run_inference(pipeline, use_fm, pcd_new, q_temp, args.steps, device)
    
    # Reconstruct metric predictions (for local mode we can map back to real mm, for trtm it's model scale)
    pred_height = pred_norm * scale + com
    
    print("\nCorner Predictions:")
    for idx, name in zip(corner_indices, corner_names):
        h_val = pred_height[idx]
        print(f"  {name}: X={h_val[0]:.2f}, Y={h_val[1]:.2f}, Z={h_val[2]:.2f}")
        
    print(f"Predicted Z range: [{pred_height[:, 2].min():.2f}, {pred_height[:, 2].max():.2f}]")

    # Save predictions
    import pickle
    if args.mode == "local":
        # Save pickle file
        pickle_save_path = os.path.join(args.sample_dir, "prediction.pkl")
        pred_data = {
            "pred_mesh_height": pred_height,
            "pred_corners_height": {
                name: pred_height[idx] for idx, name in zip(corner_indices, corner_names)
            },
            "table_plane": plane_coef
        }
        with open(pickle_save_path, "wb") as f:
            pickle.dump(pred_data, f)
        print(f"\nSaved prediction coordinates to {pickle_save_path}")

        # Reconstruct standard camera frame depth coordinates for 2D projection overlay
        pred_camera = OrbbecDepthProcessor.denormalise_prediction(
            pred_norm, com, scale, table_plane=plane_coef
        )
        
        # Load raw color image and draw projection
        color_path = os.path.join(args.sample_dir, "color_00000.png")
        if os.path.exists(color_path):
            vis = cv2.imread(color_path)
            
            # Project to 2D
            xs = pred_camera[:, 0]
            ys = pred_camera[:, 1]
            zs = np.maximum(pred_camera[:, 2], 1.0)
            u = (xs * proc.fx / zs) + proc.cx
            v = (ys * proc.fy / zs) + proc.cy
            uvs = np.stack([u, v], axis=1).astype(np.int32)
            
            # Draw 21x21 grid lines
            grid_size = 21
            for r in range(grid_size):
                for c in range(grid_size):
                    idx = r * grid_size + c
                    if c < grid_size - 1:
                        cv2.line(vis, tuple(uvs[idx]), tuple(uvs[idx + 1]), (0, 255, 255), 1)
                    if r < grid_size - 1:
                        cv2.line(vis, tuple(uvs[idx]), tuple(uvs[idx + grid_size]), (0, 255, 255), 1)
            
            # Draw corner indicators
            corner_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 0, 255)] # TL (red), TR (green), BL (blue), BR (magenta)
            for idx, col, name in zip(corner_indices, corner_colors, corner_names):
                pt = tuple(uvs[idx])
                cv2.circle(vis, pt, 8, col, -1)
                cv2.putText(vis, name, (pt[0] + 12, pt[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
                
            overlay_save_path = os.path.join(args.sample_dir, "prediction_overlay.png")
            cv2.imwrite(overlay_save_path, vis)
            print(f"Saved visualization overlay to {overlay_save_path}")
    else:
        # TRTM Mode
        pickle_save_path = f"prediction_trtm_{args.idx}.pkl"
        pred_data = {
            "pred_mesh_normalized": pred_height,
            "pred_corners_normalized": {
                name: pred_height[idx] for idx, name in zip(corner_indices, corner_names)
            }
        }
        with open(pickle_save_path, "wb") as f:
            pickle.dump(pred_data, f)
        print(f"\nSaved TRTM prediction coordinates to {pickle_save_path}")

if __name__ == "__main__":
    main()
