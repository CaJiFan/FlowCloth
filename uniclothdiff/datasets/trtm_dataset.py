import os
import pickle
import numpy as np
import glob

import torch
from torch.utils.data import Dataset

from uniclothdiff.registry import DATASETS

@DATASETS.register_module()
class TRTMDiffusionDataset(Dataset):
    def __init__(
        self, 
        data_dir,                       # path to template_square/
        mode='train',                   # 'train' | 'val' | 'real_test'
        train_mode='full',              # 'full' | 'edge_only'
        num_sample_points=400,
        do_point_cloud_augmentation=True,
        points_jitter_sigma=0.002,      # 2 mm sensor noise (same as VR-Folding)
        points_drop_ratio=0.0,
    ):
        self.mode = mode
        self.data_dir = data_dir
        self.train_mode = train_mode
        self.num_sample_points = num_sample_points
        self.do_point_cloud_augmentation = do_point_cloud_augmentation
        self.points_jitter_sigma = points_jitter_sigma
        self.points_drop_ratio = points_drop_ratio

        # ── contour indices for 21×21 grid ────────────────────────────────
        self.contour_idx = self._get_contour_indices_square(21)

        # ── template ──────────────────────────────────────────────────────
        self._setup_template()

        # ── file list ─────────────────────────────────────────────────────
        if mode == 'real_test':
            # Real test: depth PNGs only (no mesh GT → qualitative eval only)
            real_dir = os.path.join(data_dir, 'real', 'test')
            self.data_files = sorted(glob.glob(os.path.join(real_dir, '*.real_depth.png')))
            self.has_gt = False
        else:
            simu_split = 'train' if mode == 'train' else 'val'
            simu_dir = os.path.join(data_dir, 'simu', simu_split)
            self.data_files = sorted(glob.glob(os.path.join(simu_dir, '*.simu_mesh.txt')))
            self.has_gt = True

        self.num_samples = len(self.data_files)
        print(f'[{mode.upper()}] TRTM: {self.num_samples} samples  '
              f'(mode={train_mode}, pcd_src=mesh_vertices)')

    # ──────────────────────────────────────────────────────────────────────
    def _get_contour_indices_square(self, N=21):
        grid = np.arange(N * N).reshape((N, N))
        return np.unique(np.concatenate((
            grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1]
        )))

    def _setup_template(self):
        # data_dir = .../TRTM/template_square
        # pickle lives next to the folder:  .../TRTM/template_square.pickle
        name          = os.path.basename(self.data_dir)          # 'template_square'
        template_path = os.path.join(
            os.path.dirname(self.data_dir),                       # .../TRTM/
            f'{name}.pickle'
        )
        with open(template_path, 'rb') as f:
            info = pickle.load(f)
        pts = info['mesh_pos'].astype(np.float32)        # [441, 3], already centred
        if self.train_mode == 'edge_only':
            pts = pts[self.contour_idx]                  # → [80, 3]
        self.q_template = torch.tensor(pts, dtype=torch.float32)

    # ──────────────────────────────────────────────────────────────────────
    def _mesh_to_pcd(self, mesh: np.ndarray) -> np.ndarray:
        """
        Simulate a depth observation by randomly sampling mesh vertices
        and adding Gaussian noise — identical to VR-Folding's approach.
        This guarantees PCD and mesh are in the same coordinate space.
        """
        N = mesh.shape[0]
        idx = np.random.randint(0, N, size=self.num_sample_points)
        pcd = mesh[idx].copy()
        if self.points_jitter_sigma > 0:
            pcd += (np.random.randn(*pcd.shape) * self.points_jitter_sigma).astype(np.float32)
        if self.points_drop_ratio > 0:
            keep = int(len(pcd) * (1 - self.points_drop_ratio))
            pcd = pcd[np.random.choice(len(pcd), keep, replace=False)]
            # re-sample back to num_sample_points
            idx2 = np.random.randint(0, len(pcd), size=self.num_sample_points)
            pcd = pcd[idx2]
        return pcd

    # ──────────────────────────────────────────────────────────────────────
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.has_gt:
            return self._getitem_simu(idx)
        else:
            return self._getitem_real(idx)

    def _getitem_simu(self, idx):
        try:
            mesh = np.loadtxt(self.data_files[idx]).astype(np.float32)  # [441, 3]

            # ── normalise (same convention as VR-Folding) ─────────────────
            com   = mesh.mean(axis=0)
            mesh -= com

            q_temp = self.q_template.numpy().copy()
            # template is already zero-centred; rescale to match this frame's extent
            scale = float(np.max(np.linalg.norm(mesh, axis=1)))
            if scale > 1e-6:
                mesh   /= scale
                q_temp /= (np.max(np.linalg.norm(q_temp, axis=1)) + 1e-8)

            # ── PCD from mesh vertices ─────────────────────────────────────
            pcd = self._mesh_to_pcd(mesh)

            # ── optional augmentation (train only) ────────────────────────
            if self.do_point_cloud_augmentation and self.mode == 'train':
                theta = np.random.uniform(0, 2 * np.pi)
                c, s  = np.cos(theta), np.sin(theta)
                R = np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float32)
                pcd    = pcd    @ R
                mesh   = mesh   @ R
                q_temp = q_temp @ R
                shift  = np.random.uniform(-0.05, 0.05, 3).astype(np.float32)
                pcd   += shift;  mesh += shift;  q_temp += shift

            # ── edge slicing ──────────────────────────────────────────────
            if self.train_mode == 'edge_only':
                mesh   = mesh[self.contour_idx]

            return {
                'q_gt':   torch.tensor(mesh,   dtype=torch.float32),
                'pcd':    torch.tensor(pcd,    dtype=torch.float32),
                'q_temp': torch.tensor(q_temp, dtype=torch.float32),
            }

        except Exception as e:
            print(f'[TRTM] Error at idx {idx} ({self.data_files[idx]}): {e}')
            N = len(self.contour_idx) if self.train_mode == 'edge_only' else 441
            return {
                'q_gt':   torch.zeros((N, 3),                      dtype=torch.float32),
                'pcd':    torch.zeros((self.num_sample_points, 3), dtype=torch.float32),
                'q_temp': self.q_template,
            }

    def _getitem_real(self, idx):
        """
        Real-world test sample.  No mesh GT — returns pcd and q_temp only.
        Depth images in real/test are 720×720 uint8 with a proprietary
        per-frame-normalised encoding; we return the raw depth for
        qualitative inspection but flag that metrics are unavailable.
        """
        import cv2
        depth_path = self.data_files[idx]
        img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        # Build a rough PCD from the depth image using the known camera params
        # (real camera: 720×720, ~60° FOV based on TRTM calibration)
        h, w = img.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        fov_rad = np.deg2rad(60)
        fx = fy = (w / 2.0) / np.tan(fov_rad / 2)
        d = img[:, :, 0].astype(np.float32)           # single channel
        cloth_mask = d > d.min() + 5                   # remove background
        ys, xs = np.where(cloth_mask)
        zs = d[cloth_mask]                             # raw encoding — not metric!
        pts = np.stack([
            (xs - cx) / fx * zs,
            (ys - cy) / fy * zs,
            zs,
        ], axis=1).astype(np.float32)
        # normalise to zero-mean unit-scale (no mesh GT to anchor COM)
        if len(pts):
            pts -= pts.mean(axis=0)
            sc = np.max(np.linalg.norm(pts, axis=1))
            if sc > 1e-6:
                pts /= sc
        # downsample
        if len(pts) >= self.num_sample_points:
            idx2 = np.random.choice(len(pts), self.num_sample_points, replace=False)
        else:
            idx2 = np.random.choice(len(pts), self.num_sample_points, replace=True)
        pts = pts[idx2]
        return {
            'pcd':    torch.tensor(pts,  dtype=torch.float32),
            'q_temp': self.q_template,
            # no q_gt — caller must check for its absence before computing metrics
        }

