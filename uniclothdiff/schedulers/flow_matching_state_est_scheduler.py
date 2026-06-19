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

import torch
import torch.nn.functional as F
from uniclothdiff.registry import SCHEDULERS


@SCHEDULERS.register_module()
class FlowMatching_StateEst:
    """
    Minimal Conditional Flow Matching scheduler with the same training interface
    as DDPM_StateEst so train.py requires no structural changes.
    """

    def __init__(self, num_train_timesteps: int = 1000, shape_loss_weight: float = 1.0):
        self.num_train_timesteps = num_train_timesteps
        self.shape_loss_weight = shape_loss_weight

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
        if model_kwargs is not None:
            points = model_kwargs.pop('pcd')          # [B, P, 3]
            q_temp  = model_kwargs.pop('q_temp')      # [B, N, 3]
            model_kwargs.pop('q_prev', None)          # not used in state-est
            contour_idx = model_kwargs.pop('contour_idx', None)

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

        # 6. Build model input: concat [x_t, q_temp] along feature dim  (same as DDPM path)
        sample_input = torch.cat([x_t, q_temp], dim=-1)   # [B, N, 6]

        model_output = model(
            hidden_states=sample_input,
            timestep=t_int,
            encoder_hidden_states=points,
        ).sample                                           # [B, N, 3]  predicted velocity

        model_output = model_output.float()
        v_target     = v_target.float()
       
        # edge only mode
        if contour_idx is not None:
            c_idx        = contour_idx[0].long()
            model_output = model_output[:, c_idx, :]
            v_target     = v_target[:, c_idx, :]

        # 7. Dual loss: velocity prediction + direct x1 reconstruction
        loss_vel = F.mse_loss(model_output, v_target)
        total_loss = loss_vel

        if self.shape_loss_weight > 0:
            x_t          = x_t.float()
            input_fp32   = input.float()
            if contour_idx is not None:
                # Reconstruct predicted x_1 from velocity for shape loss
                x_t_sliced         = x_t[:, c_idx, :]
                t_view_sliced      = t_view.expand_as(x_t_sliced)
                pred_x1_sliced     = x_t_sliced + (1.0 - t_view[:, :, :].expand_as(x_t_sliced)) * model_output
                target_x1_sliced   = input_fp32[:, c_idx, :]
                loss_shape = F.mse_loss(pred_x1_sliced, target_x1_sliced)
            else:
                pred_x1   = x_t + (1.0 - t_view.expand_as(x_t)) * model_output
                target_x1 = input_fp32
                loss_shape = F.mse_loss(pred_x1, target_x1)

            total_loss += self.shape_loss_weight * loss_shape
        # Expose per-component losses for external logging (use same attribute names
        # as DDPM scheduler so training loop can log generically)
        try:
            self.last_loss_noise = float(loss_vel.detach().cpu().item())
        except Exception:
            self.last_loss_noise = None

        if self.shape_loss_weight > 0:
            try:
                self.last_loss_shape = float(loss_shape.detach().cpu().item())
            except Exception:
                self.last_loss_shape = None
        else:
            self.last_loss_shape = 0.0
        
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
