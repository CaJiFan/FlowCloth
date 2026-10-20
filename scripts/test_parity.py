import numpy as np
import os
import cv2

# Path to the files on the slurm cluster
mesh_path = "/home/cjimenez/projects/datasets/TRTM/template_square/simu/val/000000.simu_mesh.txt"
depth_path = "/home/cjimenez/projects/datasets/TRTM/template_square/simu/val/000000.simu_depth.png"

def depth_to_point_cloud(depth):
    # TRTM camera intrinsics
    camera_params = {
        'cx': 360.0,
        'cy': 360.0,
        'fx': 360.0 / np.tan(np.deg2rad(60) / 2),
        'fy': 360.0 / np.tan(np.deg2rad(60) / 2)
    }
    
    height, width = depth.shape[:2]
    if len(depth.shape) > 2:
        depth = depth[:, :, 0]
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    
    # Scale depth
    z = depth.astype(np.float32) / 1000.0
    
    x_norm = (x - camera_params['cx']) / camera_params['fx']
    y_norm = (y - camera_params['cy']) / camera_params['fy']
    
    x_3d = x_norm * z
    y_3d = y_norm * z
    
    # MASK OUT BACKGROUND (pixel value 255 corresponds to z = 0.255)
    mask = (z > 0) & (z < 0.250)
    points = np.stack([x_3d[mask], y_3d[mask], z[mask]], axis=1)
    
    return points

def normalise(pts):
    com = pts.mean(axis=0)
    pts_c = pts - com
    scale = np.max(np.linalg.norm(pts_c, axis=1))
    if scale > 1e-6:
        pts_norm = pts_c / scale
    else:
        pts_norm = pts_c
    return pts_norm, com, scale

def main():
    if not os.path.exists(mesh_path) or not os.path.exists(depth_path):
        print("Files do not exist on this host.")
        return
        
    mesh = np.loadtxt(mesh_path).astype(np.float32)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    pcd_raw = depth_to_point_cloud(depth)
    
    # Normalize both
    mesh_norm, mesh_com, mesh_scale = normalise(mesh)
    pcd_norm, pcd_com, pcd_scale = normalise(pcd_raw)
    
    from scipy.spatial import cKDTree
    # Build Tree on X and Y coordinates ONLY
    tree = cKDTree(pcd_norm[:, :2])
    dists, idxs = tree.query(mesh_norm[:, :2])
    matched_pcd = pcd_norm[idxs]
    
    # Compute correlation along each axis
    for i, axis in enumerate(['X', 'Y', 'Z']):
        corr = np.corrcoef(mesh_norm[:, i], matched_pcd[:, i])[0, 1]
        print(f"Axis {axis} correlation (matched by XY): {corr:.4f}")
        # Print a few samples of mesh vs matched PCD
        print(f"  Samples (mesh vs pcd):")
        for j in range(0, len(mesh_norm), len(mesh_norm)//5):
            print(f"    {mesh_norm[j, i]:.4f} vs {matched_pcd[j, i]:.4f}")

if __name__ == "__main__":
    main()
