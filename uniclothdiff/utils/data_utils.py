import numpy as np
import torch

def rotate_point_cloud_z(pcd, q_gt):
    """
    Randomly rotates the point cloud and ground truth around the Z-axis (up vector).
    This forces the model to be rotation-invariant.
    """
    theta = np.random.uniform(0, 2 * np.pi)
    cosval = np.cos(theta)
    sinval = np.sin(theta)
    
    # 3D Rotation Matrix for the Z-axis
    rotation_matrix = np.array([
        [cosval, -sinval, 0],
        [sinval,  cosval, 0],
        [0,       0,      1]
    ], dtype=np.float32)
    
    # Apply identical rotation to both input and target
    rotated_pcd = np.dot(pcd, rotation_matrix)
    rotated_q_gt = np.dot(q_gt, rotation_matrix)
    
    return rotated_pcd, rotated_q_gt

def shift_point_cloud(pcd, q_gt, shift_range=0.05):
    """
    Randomly translates (shifts) the objects in XYZ space.
    shift_range=0.05 means moving it up to 5cm in any direction.
    """
    shifts = np.random.uniform(-shift_range, shift_range, size=(3,)).astype(np.float32)
    
    # Apply identical shift to both input and target
    shifted_pcd = pcd + shifts
    shifted_q_gt = q_gt + shifts
    
    return shifted_pcd, shifted_q_gt

def jitter_point_cloud(pcd, sigma=0.001, clip=0.005):
    """
    Adds microscopic Gaussian noise to the input point cloud only.
    This prevents the model from relying on exact vertex locations.
    NOTE: We do NOT jitter the ground truth (q_gt) because we want the model 
    to predict a perfectly clean, smooth mesh!
    """
    N, C = pcd.shape
    jitter = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip).astype(np.float32)
    
    return pcd + jitter