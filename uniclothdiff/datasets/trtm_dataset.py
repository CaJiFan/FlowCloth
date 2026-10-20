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
        pcd_source='depth',             # 'depth' | 'visible_surface' | 'mesh'
        num_sample_points=400,
        do_point_cloud_augmentation=True,
        points_jitter_sigma=0.002,      # 2 mm sensor noise (same as VR-Folding)
        points_drop_ratio=0.0,
    ):
        self.mode = mode
        if not os.path.exists(data_dir):
            for candidate in [
                "/mnt/Data/cjimenez/TRTM/datasets/template_square",
                os.path.expanduser("~/projects/datasets/TRTM/template_square"),
                os.path.expanduser("~/Projects/datasets/TRTM/template_square"),
            ]:
                if os.path.exists(candidate):
                    data_dir = candidate
                    break
        self.data_dir = data_dir
        self.train_mode = train_mode
        self.pcd_source = pcd_source
        self.num_sample_points = num_sample_points
        self.do_point_cloud_augmentation = do_point_cloud_augmentation
        self.points_jitter_sigma = points_jitter_sigma
        self.points_drop_ratio = points_drop_ratio

        # ── contour indices for 21×21 grid ────────────────────────────────
        self.contour_idx = self._get_contour_indices_square(21)

        # ── template & topology ───────────────────────────────────────────
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
              f'(mode={train_mode}, pcd_src={pcd_source})')

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

        raw_edges = info['edge_idx'].astype(np.int64) if 'edge_idx' in info else np.zeros((0, 2), dtype=np.int64)
        raw_corners = info['corner_idx'].astype(np.int64) if 'corner_idx' in info else np.array([0, 20, 440, 420], dtype=np.int64)

        if self.train_mode == 'edge_only':
            pts = pts[self.contour_idx]                  # → [80, 3]
            # Remap edge indices and corner indices to 0..79 contour index space
            in_contour = np.isin(raw_edges[:, 0], self.contour_idx) & np.isin(raw_edges[:, 1], self.contour_idx)
            contour_edges = raw_edges[in_contour]
            c_map = {orig: new for new, orig in enumerate(self.contour_idx)}
            self.edge_idx = np.array([[c_map[e[0]], c_map[e[1]]] for e in contour_edges], dtype=np.int64)
            self.corner_idx = np.array([c_map[c] for c in raw_corners if c in c_map], dtype=np.int64)
        else:
            self.edge_idx = raw_edges
            self.corner_idx = raw_corners

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
        return pcd

    def _depth_to_pcd(self, depth_path: str, mesh: np.ndarray) -> np.ndarray:
        """
        Back-project realistic single-view depth observation (*.simu_depth.png)
        to a 3D point cloud matching the TRTM camera setup.
        """
        import cv2
        try:
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                return self._mesh_to_pcd(mesh)
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            cloth_mask = depth < 250
            ys, xs = np.where(cloth_mask)
            if len(xs) < 10:
                return self._mesh_to_pcd(mesh)
            
            if len(xs) >= self.num_sample_points:
                sel = np.random.choice(len(xs), self.num_sample_points, replace=False)
            else:
                sel = np.random.choice(len(xs), self.num_sample_points, replace=True)
            xs, ys = xs[sel], ys[sel]
            d_vals = depth[ys, xs].astype(np.float32)

            # Metric back-projection: 552x552 image, 187 px/m, table flat at 192
            x = (xs.astype(np.float32) - 276.0) / 187.0
            y = -(ys.astype(np.float32) - 276.0) / 187.0
            z = (192.0 - d_vals) / 228.0
            pcd = np.stack([x, y, z], axis=1).astype(np.float32)
            return pcd
        except Exception:
            return self._mesh_to_pcd(mesh)

    def _visible_surface_to_pcd(self, mesh: np.ndarray) -> np.ndarray:
        """
        Sample point cloud from visible upper surface by culling occluded vertices
        under folds (top-down z-culling).
        """
        grid_res = 0.05  # 5 cm bins
        bins = {}
        for idx, (x, y, z) in enumerate(mesh):
            key = (int(np.floor(x / grid_res)), int(np.floor(y / grid_res)))
            if key not in bins or z > bins[key][1]:
                bins[key] = (idx, z)
        visible_idx = [v[0] for v in bins.values()]
        if len(visible_idx) == 0:
            visible_idx = np.arange(len(mesh))
        chosen = np.random.choice(visible_idx, self.num_sample_points, replace=(len(visible_idx) < self.num_sample_points))
        return mesh[chosen].copy()

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

            # ── PCD generation based on pcd_source ────────────────────────
            if self.pcd_source == 'depth':
                depth_path = self.data_files[idx].replace('.simu_mesh.txt', '.simu_depth.png')
                pcd = self._depth_to_pcd(depth_path, mesh)
            elif self.pcd_source == 'visible_surface':
                pcd = self._visible_surface_to_pcd(mesh)
            else:
                pcd = self._mesh_to_pcd(mesh)

            # ── normalise (same convention as VR-Folding) ─────────────────
            com   = mesh.mean(axis=0)
            mesh -= com
            pcd  -= com

            q_temp = self.q_template.numpy().copy()
            # template is already zero-centred; rescale to match this frame's extent
            scale = float(np.max(np.linalg.norm(mesh, axis=1)))
            if scale > 1e-6:
                mesh   /= scale
                pcd    /= scale
                q_temp /= (np.max(np.linalg.norm(q_temp, axis=1)) + 1e-8)

            # ── optional augmentation (train only) ────────────────────────
            if self.do_point_cloud_augmentation and self.mode == 'train':
                if self.points_jitter_sigma > 0:
                    pcd += (np.random.randn(*pcd.shape) * self.points_jitter_sigma).astype(np.float32)
                if self.points_drop_ratio > 0:
                    keep = int(len(pcd) * (1 - self.points_drop_ratio))
                    if keep > 0:
                        drop_idx = np.random.choice(len(pcd), keep, replace=False)
                        pcd = pcd[drop_idx]
                        resample_idx = np.random.randint(0, len(pcd), size=self.num_sample_points)
                        pcd = pcd[resample_idx]

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
                'q_gt':        torch.tensor(mesh,            dtype=torch.float32),
                'pcd':         torch.tensor(pcd,             dtype=torch.float32),
                'q_temp':      torch.tensor(q_temp,          dtype=torch.float32),
                'edge_idx':    torch.tensor(self.edge_idx,   dtype=torch.long),
                'corner_idx':  torch.tensor(self.corner_idx, dtype=torch.long),
                'contour_idx': torch.tensor(self.contour_idx, dtype=torch.long),
            }

        except Exception as e:
            print(f'[TRTM] Error at idx {idx} ({self.data_files[idx]}): {e}')
            N = len(self.contour_idx) if self.train_mode == 'edge_only' else 441
            return {
                'q_gt':        torch.zeros((N, 3),                      dtype=torch.float32),
                'pcd':         torch.zeros((self.num_sample_points, 3), dtype=torch.float32),
                'q_temp':      self.q_template,
                'edge_idx':    torch.tensor(self.edge_idx,              dtype=torch.long),
                'corner_idx':  torch.tensor(self.corner_idx,            dtype=torch.long),
                'contour_idx': torch.tensor(self.contour_idx,           dtype=torch.long),
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
        if img.ndim == 3:
            d = img[:, :, 0].astype(np.float32)           # single channel
        else:
            d = img.astype(np.float32)
        cloth_mask = d > d.min() + 5                   # remove background
        ys, xs = np.where(cloth_mask)
        zs = d[cloth_mask]                             # raw encoding — not metric!
        pts = np.stack([
            (xs.astype(np.float32) - cx) / fx * zs,
            -(ys.astype(np.float32) - cy) / fy * zs,  # +Y is forward/UP (matching table frame & simulation mesh)
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