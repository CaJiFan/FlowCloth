import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from uniclothdiff.registry import DATASETS
from .stratified_split import create_stratified_split
from uniclothdiff.utils.data_utils import rotate_point_cloud_z, shift_point_cloud, jitter_point_cloud

@DATASETS.register_module()
class ClothTrackingDataset(Dataset):
    def __init__(self,
        data_dir, 
        mode='train', 
        train_mode='full', # OPTIONS: 'full', 'edge_loss', 'edge_input'
        num_sample_points=2048,
        do_point_cloud_augmentation=True,
        points_jitter_sigma=0.0005,
        points_drop_ratio=0.0
    ):
        self.data_dir = data_dir
        self.num_sample_points = num_sample_points
        self.mode = mode # "train" or "val"
        self.train_mode = train_mode

        grid_size = 20
        contour = []
        for i in range(grid_size):
            for j in range(grid_size):
                if i == 0 or i == grid_size - 1 or j == 0 or j == grid_size - 1:
                    contour.append(i * grid_size + j)
        
        self.contour_idx = torch.tensor(contour, dtype=torch.long)
        
        # 1. Load all files
        all_xml_files = sorted(glob.glob(f'{data_dir}/Dataset_*/*.xml'))
        # print('all', len(all_xml_files))
        
        # 2. Create a 90/10 Train/Val Split
        split_idx = int(len(all_xml_files) * 0.9) # 90% for training

        # Call our new stratified function
        train_files, val_files = create_stratified_split(all_xml_files, train_ratio=0.8)

        print(len(all_xml_files), len(train_files), len(val_files))
        
        # Assign based on mode
        if self.mode == "train":
            self.xml_files = train_files
        elif self.mode == "val":
            self.xml_files = val_files
        
        # if self.mode == "train":
        #     self.xml_files = all_xml_files[:split_idx]
        # elif self.mode == "val":
        #     self.xml_files = all_xml_files[split_idx:]
        # else:
        #     self.xml_files = all_xml_files # fallback for "full" mode
            
        print(f"[{self.mode.upper()}] Found {len(self.xml_files)} XML sequences. Building index...")
        
        # 1. Parse and cache all sequences in memory (they are very small!)
        self.sequences = {}
        self.valid_pairs = []
        
        for xml_file in self.xml_files:
            try:
                seq_tensor = self._parse_xml(xml_file)
                self.sequences[xml_file] = seq_tensor
                
                # If sequence has 175 frames, we have 174 pairs (t-1 -> t)
                num_frames = seq_tensor.shape[0]
                for t in range(1, num_frames):
                    self.valid_pairs.append((xml_file, t))
            except Exception as e:
                print(f"Skipping {xml_file} due to error: {e}")
                
        print(f"Dataset ready! Total training steps (frame pairs): {len(self.valid_pairs)}")

    def _parse_xml(self, xml_path):
        """Helper to parse XML into [Frames, 400, 3]"""
        tree = ET.parse(xml_path)
        vertices = []
        for pos in tree.getroot().findall('.//position'):
            if pos.get('x') is None: continue
            vertices.append([float(pos.get('x')), float(pos.get('y')), float(pos.get('z'))])
        
        flat_mesh = torch.tensor(np.array(vertices, dtype=np.float32))
        return flat_mesh.view(-1, 400, 3)

    def _simulate_depth_camera(self, mesh):
        """Simulates a Point Cloud (pcd) by sampling the mesh surface"""
        num_verts = mesh.shape[0]
        indices = torch.randint(0, num_verts, (self.num_sample_points,))
        pcd = mesh[indices].clone()
        camera_noise = torch.randn_like(pcd) * 0.002 # 2mm of sensor noise
        return pcd + camera_noise

    def __len__(self):
        return len(self.valid_pairs)

    def __getitem__(self, idx):
        # 1. Look up which file and which frame we are training on
        xml_file, t = self.valid_pairs[idx]
        seq_tensor = self.sequences[xml_file]
        
        # 2. Extract the Temporal Pair
        q_prev = seq_tensor[t - 1].clone() # Frame t-1
        q_gt = seq_tensor[t].clone()       # Frame t
        
        # We center both frames based on the starting position so the model 
        # doesn't have to learn arbitrary table offsets like y = -1.3
        center_of_mass = q_prev.mean(dim=0)
        q_prev -= center_of_mass
        q_gt -= center_of_mass
        q_temp = seq_tensor[0].clone() - center_of_mass 
        
        # Simulate depth sensor seeing the GT frame
        pcd = self._simulate_depth_camera(q_gt)

        max_distance = torch.max(torch.sqrt(torch.sum(q_prev**2, dim=1)))

        if max_distance > 0.0001:
            q_prev = q_prev / max_distance
            q_gt = q_gt / max_distance
            q_temp = q_temp / max_distance
            pcd = pcd / max_distance

        pcd = pcd.numpy() if isinstance(pcd, torch.Tensor) else pcd
        q_gt = q_gt.numpy() if isinstance(q_gt, torch.Tensor) else q_gt
        q_prev = q_prev.numpy() if isinstance(q_prev, torch.Tensor) else q_prev
        q_temp = q_temp.numpy() if isinstance(q_temp, torch.Tensor) else q_temp

        if self.mode == 'train':
            # 1. Random Z-Rotation (Apply to ALL 4 arrays)
            theta = np.random.uniform(0, 2 * np.pi)
            cosval, sinval = np.cos(theta), np.sin(theta)
            rot_matrix = np.array([
                [cosval, -sinval, 0],
                [sinval,  cosval, 0],
                [0,       0,      1]
            ], dtype=np.float32)
            
            pcd = np.dot(pcd, rot_matrix)
            q_gt = np.dot(q_gt, rot_matrix)
            q_prev = np.dot(q_prev, rot_matrix)
            q_temp = np.dot(q_temp, rot_matrix)
            
            # 2. Random Translation (Apply to ALL 4 arrays)
            shifts = np.random.uniform(-0.05, 0.05, size=(3,)).astype(np.float32)
            pcd += shifts
            q_gt += shifts
            q_prev += shifts
            q_temp += shifts
            
            # 3. Random Jitter (Apply to PCD INPUT ONLY)
            jitter = np.clip(0.001 * np.random.randn(*pcd.shape), -0.005, 0.005).astype(np.float32)
            pcd += jitter

        # Convert everything back to PyTorch Tensors
        pcd = torch.tensor(pcd, dtype=torch.float32)
        q_gt = torch.tensor(q_gt, dtype=torch.float32)
        q_prev = torch.tensor(q_prev, dtype=torch.float32)
        q_temp = torch.tensor(q_temp, dtype=torch.float32)

        # Simulate the "Action" (How did the top two corners move?)
        # Corners in a 20x20 grid are typically at indices 0 and 19
        corner_0_move = q_gt[0] - q_prev[0]
        corner_19_move = q_gt[19] - q_prev[19]
        action = torch.cat([corner_0_move, corner_19_move])

        if self.train_mode == "edge_only":
            q_gt_sliced = q_gt[self.contour_idx]
            q_prev_sliced = q_prev[self.contour_idx]
            q_temp_sliced = q_temp[self.contour_idx]

            return {
                "q_gt": q_gt_sliced,       # [76, 3] target for DDPM
                "q_prev": q_prev_sliced,    # [76, 3] previous state
                "pcd": pcd,                 # [2048, 3] depth camera
                "action": action,           # [6]
                "q_temp": q_temp_sliced     # [76, 3] canonical shape
            }
        
        return {
            'q_prev': q_prev,       # [400, 3] - What it looked like
            'q_gt': q_gt,           # [400, 3] - What it looks like now
            'pcd': pcd,             # [2048, 3]- What the camera sees now
            'action': action,       # [6]      - How the human moved the corners
            'q_temp': q_temp,        # [400, 3] - Canonical shape
        }

# ==========================================
# Test the DataLoader!
# ==========================================
if __name__ == "__main__":
    # Point this to the folder containing your XMLs
    DATA_DIR = "../../datasets/VR_Folding/Dataset_01/"
    
    dataset = ClothTrackingDataset(data_dir=DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Grab one batch
    batch = next(iter(dataloader))
    
    print("\n--- Batch Shapes ---")
    print(f"q_prev (t-1):   {batch['q_prev'].shape}")
    print(f"q_gt (t):       {batch['q_gt'].shape}")
    print(f"pcd (Sensor):   {batch['pcd'].shape}")
    print(f"action (Move):  {batch['action'].shape}")
    print(f"q_temp:         {batch['q_temp'].shape}")