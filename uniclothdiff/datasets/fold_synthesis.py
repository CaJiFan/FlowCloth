"""
Procedural fold synthesis for the TRTM square template.

TRTM's simulated split is dominated by crumpled states: ~82% of samples have no
corner folded on top of the cloth, ~2% have two or more, and almost none lie flat
on the table.  Real table-top manipulation (half folds, multi-corner folds) is
therefore out of distribution.  This module generates such states directly:

  1. Start from the flat template, add a smooth low-amplitude wrinkle field.
  2. Apply 1-3 folds.  A fold rotates every point on one side of a line by phi
     about that line, lifted to the top of the current stack (phi = pi gives a
     flat fold with reversed layer order, phi < pi a partially lifted flap).
  3. Render a top-down observation from a dense copy of the surface: keep only
     points on the topmost layer (z-buffer), quantise height like simu_depth.png.

Folding is an isometry, so the ground-truth meshes keep their rest edge lengths.
Outputs are in the raw TRTM mesh units (flat side length = 2), i.e. the same
space as *.simu_mesh.txt and the back-projected *.simu_depth.png.
"""

import numpy as np

N_GRID = 21
UPSAMPLE = 8                         # dense surface: (21-1)*8+1 = 161 points per side
CORNERS = [0, N_GRID - 1, N_GRID * N_GRID - 1, N_GRID * (N_GRID - 1)]
DEPTH_Z_PER_UNIT = 228.0             # simu_depth.png height quantisation (see TRTMDiffusionDataset._depth_to_pcd)
VIS_CELL = 0.02                      # z-buffer cell, > dense point spacing (2/160 = 0.0125)
HINGE_BAND = 0.15                    # points within this distance of the fold line set the hinge height


def _wrinkle_field(uv: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Smooth height field over material coords uv in [0,1]^2 (sum of random sinusoids)."""
    amp = rng.uniform(0.0, 0.02)
    z = np.zeros(len(uv), dtype=np.float32)
    for _ in range(3):
        freq = rng.uniform(0.5, 2.0, size=2) * rng.choice([-1.0, 1.0], size=2)
        phase = rng.uniform(0.0, 2 * np.pi)
        z += np.sin(2 * np.pi * (uv @ freq) + phase).astype(np.float32)
    return amp * z / 3.0


def _fold_line(X: np.ndarray, kind: str, rng: np.random.Generator):
    """Return (unit normal n, offset c); points with xy.n - c > 0 are folded over."""
    xy = X[:, :2]
    centre = xy.mean(axis=0)
    if kind == "corner":
        k = CORNERS[rng.integers(4)]
        n = xy[k] - centre
        n /= np.linalg.norm(n) + 1e-8
        c = xy[k] @ n - rng.uniform(0.2, 0.9)
    elif kind == "axis":
        n = np.eye(2)[rng.integers(2)] * rng.choice([-1.0, 1.0])
        proj = xy @ n
        c = 0.5 * (proj.min() + proj.max()) + rng.uniform(-0.15, 0.15) * np.ptp(proj)
    else:
        theta = rng.uniform(0.0, 2 * np.pi)
        n = np.array([np.cos(theta), np.sin(theta)])
        c = np.quantile(xy @ n, rng.uniform(0.2, 0.8))
    return n.astype(np.float32), np.float32(c)


def _apply_fold(X: np.ndarray, n: np.ndarray, c: float, phi: float, thickness: float) -> np.ndarray:
    """Rotate the side xy.n > c by phi about the fold line, hinged on top of the local stack."""
    s = X[:, :2] @ n - c
    moving = s > 0
    if not moving.any() or moving.all():
        return X
    near_line = np.abs(s) < HINGE_BAND
    h = X[near_line, 2].max() + 0.5 * thickness if near_line.any() else X[:, 2].max() + 0.5 * thickness
    sm, w = s[moving], X[moving, 2] - h
    s_new = sm * np.cos(phi) - w * np.sin(phi)
    w_new = sm * np.sin(phi) + w * np.cos(phi)
    X = X.copy()
    X[moving, :2] += (s_new - sm)[:, None] * n[None, :]
    X[moving, 2] = h + w_new
    return X


def _visible_points(dense: np.ndarray, num_points: int, eps: float, rng: np.random.Generator) -> np.ndarray:
    """Top-down z-buffer: keep dense points within eps of the highest point in their cell."""
    ij = np.floor(dense[:, :2] / VIS_CELL).astype(np.int64)
    ij -= ij.min(axis=0)
    key = ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]
    _, inv = np.unique(key, return_inverse=True)
    zmax = np.full(inv.max() + 1, -np.inf, dtype=np.float32)
    np.maximum.at(zmax, inv, dense[:, 2])
    pts = dense[dense[:, 2] >= zmax[inv] - eps]
    sel = rng.choice(len(pts), num_points, replace=len(pts) < num_points)
    pts = pts[sel].copy()
    pts[:, 2] = np.round(pts[:, 2] * DEPTH_Z_PER_UNIT) / DEPTH_Z_PER_UNIT
    return pts


def synthesize_folded_sample(template: np.ndarray, num_points: int, rng: np.random.Generator):
    """
    Args:
        template:   [441, 3] flat TRTM template (mesh_pos from template_square.pickle).
        num_points: number of observed points to return.
    Returns:
        mesh: [441, 3] folded ground-truth mesh, raw TRTM units.
        pcd:  [num_points, 3] top-down visible surface points, raw TRTM units.
    """
    # ── nodes + dense surface share every operation, so they stay consistent ──
    grid = template.reshape(N_GRID, N_GRID, 3).astype(np.float32)
    t = np.linspace(0.0, N_GRID - 1, (N_GRID - 1) * UPSAMPLE + 1)
    i0 = np.clip(np.floor(t).astype(int), 0, N_GRID - 2)
    f = (t - i0).astype(np.float32)
    rows = grid[i0] * (1 - f)[:, None, None] + grid[i0 + 1] * f[:, None, None]
    dense = rows[:, i0] * (1 - f)[None, :, None] + rows[:, i0 + 1] * f[None, :, None]
    dense = dense.reshape(-1, 3)

    uv_nodes = np.stack(np.meshgrid(np.linspace(0, 1, N_GRID), np.linspace(0, 1, N_GRID), indexing="ij"), -1).reshape(-1, 2)
    uv_dense = np.stack(np.meshgrid(np.linspace(0, 1, len(t)), np.linspace(0, 1, len(t)), indexing="ij"), -1).reshape(-1, 2)
    X = np.concatenate([grid.reshape(-1, 3), dense], axis=0)
    X[:, 2] = _wrinkle_field(np.concatenate([uv_nodes, uv_dense]).astype(np.float32), rng)

    # ── 1-3 folds ──────────────────────────────────────────────────────────
    thickness = rng.uniform(0.006, 0.02)
    num_folds = rng.choice([1, 2, 3], p=[0.55, 0.30, 0.15])
    for k in range(num_folds):
        kind = rng.choice(["corner", "axis", "random"], p=[0.45, 0.35, 0.20])
        n, c = _fold_line(X[: N_GRID * N_GRID], kind, rng)
        # only the last fold may be a partially lifted flap; earlier ones lie flat
        last = k == num_folds - 1
        phi = rng.uniform(0.55 * np.pi, np.pi) if last and rng.random() < 0.25 else np.pi
        X = _apply_fold(X, n, c, phi, thickness)

    mesh = X[: N_GRID * N_GRID].copy()
    pcd = _visible_points(X[N_GRID * N_GRID:], num_points, eps=0.5 * thickness, rng=rng)
    return mesh, pcd
