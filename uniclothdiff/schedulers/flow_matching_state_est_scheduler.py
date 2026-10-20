"""
Flow Matching scheduler for cloth state estimation.

Key difference from DDPM:
  DDPM: noisy_x = sqrt(alpha_t)*x + sqrt(1-alpha_t)*eps   → predict eps
  FM:   x_t     = (1-t)*x_0 + t*x_1                       → predict velocity v = x_1 - x_0

Training loss:  MSE(model(x_t, q_temp, pcd, t*T), v_target)
Inference:      Euler ODE  x_{t+h} = x_t + h * v_pred  (10-50 steps)

The model architecture (TransformerStateEstV3Model) is reused unchanged.
Timestep is scaled to [0, num_train_timesteps] so AdaLayerNorm sees the same
integer range it would see during DDPM training.
"""

import numpy as np
import torch
import torch.nn.functional as F
from uniclothdiff.registry import SCHEDULERS


@SCHEDULERS.register_module()
class FlowMatching_StateEst:
    """
    Conditional Flow Matching scheduler with multi-objective physics and boundary-aware loss:
      L_total = L_vel + lambda_shape * L_recon + lambda_corner * L_corner + lambda_contour * L_contour + lambda_edge * L_edge
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shape_loss_weight: float = 1.0,
        corner_loss_weight: float = 0.0,
        contour_loss_weight: float = 0.0,
        edge_loss_weight: float = 0.0,
        laplacian_loss_weight: float = 0.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shape_loss_weight = shape_loss_weight
        self.corner_loss_weight = corner_loss_weight
        self.contour_loss_weight = contour_loss_weight
        self.edge_loss_weight = edge_loss_weight
        self.laplacian_loss_weight = laplacian_loss_weight

        # Logging attributes
        self.last_loss_noise = 0.0
        self.last_loss_vel = 0.0
        self.last_loss_shape = 0.0
        self.last_loss_corner = 0.0
        self.last_loss_contour = 0.0
        self.last_loss_edge = 0.0
        self.last_loss_laplacian = 0.0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_losses_with_cfg(
        self,
        model,
        input,                  # x_1: ground-truth mesh  [B, N, 3]
        pc_encoder=None,        # unused, kept for interface compatibility
        model_kwargs=None,
        noise=None,
        weight_dtype=None,
        generator=None
    ):
        batch_size = input.shape[0]
        device = input.device

        # 1. Extract conditioning inputs
        contour_idx = None
        edge_idx = None
        corner_idx = None
        if model_kwargs is not None:
            points = model_kwargs.pop('pcd')          # [B, P, 3]
            q_temp  = model_kwargs.pop('q_temp')      # [B, N, 3]
            model_kwargs.pop('q_prev', None)          # not used in state-est
            contour_idx = model_kwargs.pop('contour_idx', None)
            edge_idx = model_kwargs.pop('edge_idx', None)
            corner_idx = model_kwargs.pop('corner_idx', None)

        # 2. Sample x_0 ~ N(0, I)  and  t ~ U[0, 1]
        x_0 = torch.randn_like(input) if noise is None else noise
        t_continuous = torch.rand(batch_size, device=device)          # [B]  in [0, 1)

        # 3. Linear interpolation: x_t = (1-t)*x_0 + t*x_1
        t_view = t_continuous.view(-1, 1, 1)
        x_t = (1.0 - t_view) * x_0 + t_view * input

        # 4. Target velocity: the straight-line direction from noise to data
        v_target = input - x_0                        # [B, N, 3]

        # 5. Scale t to integer range so AdaLayerNorm works identically to DDPM
        t_int = (t_continuous * self.num_train_timesteps).long().clamp(0, self.num_train_timesteps - 1)

        # 6. Build model input: concat [x_t, q_temp] along feature dim
        sample_input = torch.cat([x_t, q_temp], dim=-1)   # [B, N, 6]

        model_output = model(
            hidden_states=sample_input,
            timestep=t_int,
            encoder_hidden_states=points,
        ).sample                                           # [B, N, 3]  predicted velocity

        model_output = model_output.float()
        v_target     = v_target.float()
        x_t_fp32     = x_t.float()
        input_fp32   = input.float()

        # Direct reconstruction of state x_1 from velocity:
        # Since x_t = (1-t)*x_0 + t*x_1 and v = x_1 - x_0,
        # x_1 = x_t + (1-t)*v
        pred_x1 = x_t_fp32 + (1.0 - t_view.expand_as(x_t_fp32)) * model_output
        target_x1 = input_fp32

        # Extract active indices (handle batched [B, ...] vs unbatched tensor)
        c_idx = None
        if contour_idx is not None:
            c_idx = contour_idx[0].long() if contour_idx.ndim == 2 else contour_idx.long()

        # Handle edge-only mode slicing if model operates on full mesh but target is contour
        if model_output.shape[1] != input.shape[1] and c_idx is not None:
            v_pred_active = model_output[:, c_idx, :]
            v_target_active = v_target[:, c_idx, :]
            pred_x1_active = pred_x1[:, c_idx, :]
            target_x1_active = target_x1
        else:
            v_pred_active = model_output
            v_target_active = v_target
            pred_x1_active = pred_x1
            target_x1_active = target_x1

        # 1. Velocity prediction MSE: L_vel = ||v_pred - v_target||^2
        loss_vel = F.mse_loss(v_pred_active, v_target_active)
        total_loss = loss_vel

        # 2. Shape / Reconstruction MSE: L_recon = ||pred_x1 - target_x1||^2
        if self.shape_loss_weight > 0:
            loss_shape = F.mse_loss(pred_x1_active, target_x1_active)
            total_loss = total_loss + self.shape_loss_weight * loss_shape
        else:
            loss_shape = torch.tensor(0.0, device=device)

        # Automatic resolution of corner_idx and edge_idx if not explicitly provided in model_kwargs
        num_nodes = pred_x1_active.shape[1]
        if corner_idx is None:
            if num_nodes == 441:
                corner_idx = torch.tensor([0, 20, 420, 440], device=device)
            elif num_nodes == 80:
                corner_idx = torch.tensor([0, 20, 59, 79], device=device)
            elif num_nodes == 400:
                corner_idx = torch.tensor([0, 19, 380, 399], device=device)
            elif num_nodes == 76:
                corner_idx = torch.tensor([0, 19, 56, 75], device=device)

        if edge_idx is None:
            if num_nodes == 441:
                if not hasattr(self, "_cached_edges_441") or self._cached_edges_441.device != device:
                    N = 21
                    h = [[r * N + c, r * N + c + 1] for r in range(N) for c in range(N - 1)]
                    v = [[r * N + c, (r + 1) * N + c] for r in range(N - 1) for c in range(N)]
                    self._cached_edges_441 = torch.tensor(h + v, dtype=torch.long, device=device)
                edge_idx = self._cached_edges_441
            elif num_nodes == 80:
                if not hasattr(self, "_cached_edges_80") or self._cached_edges_80.device != device:
                    ring = [[i, (i + 1) % 80] for i in range(80)]
                    self._cached_edges_80 = torch.tensor(ring, dtype=torch.long, device=device)
                edge_idx = self._cached_edges_80
            elif num_nodes == 400:
                if not hasattr(self, "_cached_edges_400") or self._cached_edges_400.device != device:
                    N = 20
                    h = [[r * N + c, r * N + c + 1] for r in range(N) for c in range(N - 1)]
                    v = [[r * N + c, (r + 1) * N + c] for r in range(N - 1) for c in range(N)]
                    self._cached_edges_400 = torch.tensor(h + v, dtype=torch.long, device=device)
                edge_idx = self._cached_edges_400

        # 3. Corner MSE: L_corner
        if self.corner_loss_weight > 0 and corner_idx is not None:
            corners = corner_idx[0].long() if corner_idx.ndim == 2 else corner_idx.long()
            pred_corners = pred_x1_active[:, corners, :]
            target_corners = target_x1_active[:, corners, :]
            loss_corner = F.mse_loss(pred_corners, target_corners)
            total_loss = total_loss + self.corner_loss_weight * loss_corner
        else:
            loss_corner = torch.tensor(0.0, device=device)

        # 4. Contour MSE: L_contour
        if self.contour_loss_weight > 0:
            if c_idx is not None and pred_x1.shape[1] > len(c_idx):
                pred_contour = pred_x1[:, c_idx, :]
                target_contour = target_x1[:, c_idx, :]
                loss_contour = F.mse_loss(pred_contour, target_contour)
            else:
                loss_contour = loss_shape
            total_loss = total_loss + self.contour_loss_weight * loss_contour
        else:
            loss_contour = torch.tensor(0.0, device=device)

        # 5. Inextensibility / Edge-length preservation loss: L_edge
        if self.edge_loss_weight > 0 and edge_idx is not None:
            edges = edge_idx[0].long() if edge_idx.ndim == 3 else edge_idx.long()
            e_i = edges[:, 0]
            e_j = edges[:, 1]
            pred_edge_diff = pred_x1_active[:, e_i, :] - pred_x1_active[:, e_j, :]
            pred_edge_len = torch.sqrt(torch.sum(pred_edge_diff ** 2, dim=-1) + 1e-8)
            target_edge_diff = target_x1_active[:, e_i, :] - target_x1_active[:, e_j, :]
            target_edge_len = torch.sqrt(torch.sum(target_edge_diff ** 2, dim=-1) + 1e-8)
            loss_edge = F.mse_loss(pred_edge_len, target_edge_len)
            total_loss = total_loss + self.edge_loss_weight * loss_edge
        else:
            loss_edge = torch.tensor(0.0, device=device)

        # 6. Topology-based Laplacian grid smoothness loss: L_laplacian
        if self.laplacian_loss_weight > 0:
            num_nodes = pred_x1_active.shape[1]
            if num_nodes in (441, 400):
                # 2D Grid Laplacian (interior nodes: harmonic mean of 4 orthogonal neighbors)
                N = 21 if num_nodes == 441 else 20
                grid = pred_x1_active.view(batch_size, N, N, 3)
                center = grid[:, 1:-1, 1:-1, :]
                north  = grid[:, :-2,  1:-1, :]
                south  = grid[:, 2:,   1:-1, :]
                west   = grid[:, 1:-1, :-2,  :]
                east   = grid[:, 1:-1, 2:,   :]
                lapl = center - 0.25 * (north + south + west + east)
                loss_laplacian = torch.mean(lapl ** 2)
            elif num_nodes in (80, 76):
                # 1D Closed Contour Ring Laplacian on straight edge nodes (excluding 90-degree corners)
                N = 21 if num_nodes == 80 else 20
                cache_key = f"_cached_neigh_{num_nodes}"
                if not hasattr(self, cache_key) or getattr(self, cache_key).device != device:
                    grid = np.arange(N * N).reshape((N, N))
                    contour_idx = np.unique(np.concatenate((grid[0, :], grid[-1, :], grid[:, 0], grid[:, -1])))
                    c_map = {orig: new for new, orig in enumerate(contour_idx)}
                    top = [[c_map[grid[0, c]], c_map[grid[0, c+1]]] for c in range(N - 1)]
                    bottom = [[c_map[grid[-1, c]], c_map[grid[-1, c+1]]] for c in range(N - 1)]
                    left = [[c_map[grid[r, 0]], c_map[grid[r+1, 0]]] for r in range(N - 1)]
                    right = [[c_map[grid[r, -1]], c_map[grid[r+1, -1]]] for r in range(N - 1)]
                    all_edges = top + bottom + left + right
                    adj = {i: [] for i in range(num_nodes)}
                    for u, v in all_edges:
                        adj[u].append(v)
                        adj[v].append(u)
                    neigh = torch.tensor([[adj[i][0], adj[i][1]] for i in range(num_nodes)], dtype=torch.long, device=device)
                    corners = [c_map[0], c_map[N - 1], c_map[N * N - 1], c_map[N * (N - 1)]]
                    non_corners = torch.tensor([i for i in range(num_nodes) if i not in corners], dtype=torch.long, device=device)
                    setattr(self, cache_key, neigh)
                    setattr(self, f"_cached_non_corners_{num_nodes}", non_corners)
                
                neigh = getattr(self, cache_key)
                non_corners = getattr(self, f"_cached_non_corners_{num_nodes}")
                lapl = pred_x1_active[:, non_corners, :] - 0.5 * (pred_x1_active[:, neigh[non_corners, 0], :] + pred_x1_active[:, neigh[non_corners, 1], :])
                loss_laplacian = torch.mean(lapl ** 2)
            else:
                loss_laplacian = torch.tensor(0.0, device=device)
            total_loss = total_loss + self.laplacian_loss_weight * loss_laplacian
        else:
            loss_laplacian = torch.tensor(0.0, device=device)

        # Expose per-component losses for external logging
        self.last_loss_noise = float(loss_vel.detach().cpu().item())
        self.last_loss_vel = self.last_loss_noise
        self.last_loss_shape = float(loss_shape.detach().cpu().item())
        self.last_loss_corner = float(loss_corner.detach().cpu().item())
        self.last_loss_contour = float(loss_contour.detach().cpu().item())
        self.last_loss_edge = float(loss_edge.detach().cpu().item())
        self.last_loss_laplacian = float(loss_laplacian.detach().cpu().item())

        return total_loss

    # ------------------------------------------------------------------
    # Inference helpers (used by FlowMatchingStateEstPipeline)
    # ------------------------------------------------------------------

    def euler_step(self, x_t: torch.Tensor, v_pred: torch.Tensor, dt: float) -> torch.Tensor:
        """Single Euler step: x_{t+dt} = x_t + dt * v_pred."""
        return x_t + dt * v_pred

    def scale_model_input(self, sample: torch.Tensor, t) -> torch.Tensor:
        """No-op: FM does not scale the input (unlike DDPM's init_noise_sigma)."""
        return sample