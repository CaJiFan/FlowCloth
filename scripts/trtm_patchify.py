import pickle
import numpy as np
import torch
import argparse
from scipy.spatial import cKDTree

def sample_farthest_points(points, num_samples):
    """Simple FPS implementation"""
    N = points.shape[1]
    centroids = torch.zeros(1, num_samples, dtype=torch.long)
    distance = torch.ones(1, N) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long)
    for i in range(num_samples):
        centroids[0, i] = farthest
        centroid = points[0, farthest, :].view(1, 1, 3)
        dist = torch.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def split_point_cloud_with_voronoi(point_cloud, n_patches):
    fps_idx = sample_farthest_points(point_cloud.float(), n_patches)
    centers = point_cloud[0, fps_idx[0], :].unsqueeze(0)
    
    centers_np = centers[0].cpu().numpy()
    tree = cKDTree(centers_np)
    
    patches = [[] for _ in range(n_patches)]
    patches_idx = [[] for _ in range(n_patches)]
    
    points_np = point_cloud[0].cpu().numpy()
    for point_idx, point in enumerate(points_np):
        _, nearest_center_idx = tree.query(point)
        patches[nearest_center_idx].append(point)
        patches_idx[nearest_center_idx].append(point_idx)
        
    patches = [np.array(p, dtype=np.float32) for p in patches if len(p) > 0]
    patches_idx = [np.array(p, dtype=np.int64) for p in patches_idx if len(p) > 0]
    return patches, patches_idx, centers, fps_idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickle_path", type=str, required=True, help="Path to template_square.pickle")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "edge_input"])
    parser.add_argument("--num_patches", type=int, default=16)
    args = parser.parse_args()

    # 1. Load TRTM Pickle
    with open(args.pickle_path, 'rb') as f:
        template_info = pickle.load(f)
    points = template_info['mesh_pos'] # (441, 3)

    # 2. Slice if testing Edge Input
    if args.mode == "edge_input":
        # Get contour (80 points)
        N = int(np.sqrt(len(points)))
        grid = np.arange(N*N).reshape((N, N))
        contour_idx = np.unique(np.concatenate((grid[0,:], grid[-1,:], grid[:,0], grid[:,-1])))
        points = points[contour_idx]
        print(f"Edge mode: Pruned template to {len(points)} nodes.")

    # 3. Patchify
    points_tensor = torch.from_numpy(points[None])
    print(f"Patchifying {points.shape[0]} points into {args.num_patches} Voronoi patches...")
    
    p_pts, p_idx, centers, c_idx = split_point_cloud_with_voronoi(points_tensor, args.num_patches)

    # 4. Save UniClothDiff compatible format
    out_name = f"template_square_{args.mode}_voronoi.pkl"
    data = {
        "patch_points": p_pts,
        "patch_index": p_idx,
        "centers": centers[0].cpu().numpy(),
        "center_idx": c_idx[0].cpu().numpy(),
        "points": points,
    }
    with open(out_name, "wb") as f:
        pickle.dump(data, f)
    print(f"Saved to {out_name}")