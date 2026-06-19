import os
import shutil
import torch
from typing import Tuple

def get_model_numel(model: torch.nn.Module) -> Tuple[int, int]:
    num_params = 0
    num_params_trainable = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_params_trainable += p.numel()
    return num_params, num_params_trainable

def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"

import os

def get_project_root():
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    root_indicators = ['.gitignore', 'setup.py', 'README.md']
    
    while True:
        if any(os.path.exists(os.path.join(current_path, indicator)) for indicator in root_indicators):
            return current_path
        
        parent_path = os.path.dirname(current_path)
        
        if parent_path == current_path:
            raise Exception("Not able to find root dir")
        
        current_path = parent_path


def backup_code(exp_dir, ignore_patterns=None, logger=None):
    """
    Backup experiment code
    
    :param exp_dir
    :param ignore_patterns
    """
    
    backup_dir = os.path.join(exp_dir, "code")
    os.makedirs(backup_dir, exist_ok=True)
    
    root_dir = get_project_root()
    
    # Ignore patterns
    if ignore_patterns is None:
        ignore_patterns = [
            "*.pyc", "__pycache__", ".git", ".gitignore", "*.log",
            "*.pth", "*.pt", "*.pkl", "data", "outputs", "experiments", 
            "third_party", "*.bin", "checkpoints", "environment.yml",
            "wandb", "UniClothDiff.egg-info", "README.md"
        ]

    def ignore_files(dir, files):
        return [f for f in files if any(f.endswith(pat) or f == pat for pat in ignore_patterns)]
    
    def custom_copy(src, dst, ignore=None):
        names = os.listdir(src)
        if ignore is not None:
            ignored_names = ignore(src, names)
        else:
            ignored_names = set()
        
        if not os.path.exists(dst):
            os.makedirs(dst)
        
        for name in names:
            if name in ignored_names:
                continue
            srcname = os.path.join(src, name)
            dstname = os.path.join(dst, name)
            if os.path.isdir(srcname):
                custom_copy(srcname, dstname, ignore)
            else:
                shutil.copy2(srcname, dstname)
                logger.info(f"Backing up: {srcname}")
    
    custom_copy(root_dir, backup_dir, ignore=ignore_files)
    
    logger.info(f"\nCode backup at: {backup_dir}")


def find_unused_parameters(model):
    unused_parameters = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused_parameters.append(name)
    return unused_parameters

def get_model_parameters(model):
    name_list = []
    param_list = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            name_list.append(name)
            param_list.append(param)
    return name_list, param_list


import numpy as np

def corner_localization_error(pred, gt, corner_indices=[0, 19, 56, 75]):
    # For edge-only: find which contour indices correspond to cloth corners
    errors = np.linalg.norm(pred[:, corner_indices] - gt[:, corner_indices], axis=-1)
    return errors.mean()  # mean over corners and batch

def chamfer_l1(pred, gt):
    # pred, gt: [B, N, 3]
    # For each pred point, nearest GT; for each GT point, nearest pred
    from scipy.spatial import cKDTree
    B = pred.shape[0]
    total = 0.0
    for b in range(B):
        tree_gt = cKDTree(gt[b])
        tree_pred = cKDTree(pred[b])
        d_pred_to_gt, _ = tree_gt.query(pred[b])
        d_gt_to_pred, _ = tree_pred.query(gt[b])
        total += (d_pred_to_gt.mean() + d_gt_to_pred.mean()) / 2.0
    return total / B

def f_score(pred, gt, threshold=0.02):  # 2cm threshold
    from scipy.spatial import cKDTree
    B = pred.shape[0]
    total = 0.0
    for b in range(B):
        tree_gt = cKDTree(gt[b])
        tree_pred = cKDTree(pred[b])
        d_p2g, _ = tree_gt.query(pred[b])
        d_g2p, _ = tree_pred.query(gt[b])
        precision = (d_p2g < threshold).mean()
        recall    = (d_g2p < threshold).mean()
        if precision + recall > 0:
            total += 2 * precision * recall / (precision + recall)
    return total / B