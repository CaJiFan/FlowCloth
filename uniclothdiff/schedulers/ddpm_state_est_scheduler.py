import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from uniclothdiff.registry import SCHEDULERS
import torch.nn as nn
import torch.nn.functional as F

def get_contributing_params(y, top_level=True):
    nf = y.grad_fn.next_functions if top_level else y.next_functions
    for f, _ in nf:
        try:
            yield f.variable
        except AttributeError:
            pass  # node has no tensor
        if f is not None:
            yield from get_contributing_params(f, top_level=False)

@SCHEDULERS.register_module()
class DDPM_StateEst(DDPMScheduler):
    def __init__(self, num_train_timesteps: int = 1000, shape_loss_weight: float = 1.0, **kwargs):
        # Accept and forward extra scheduler kwargs (e.g. `beta_schedule`) to
        # the parent `DDPMScheduler` which may require them depending on
        # the installed `diffusers` version.
        super().__init__(num_train_timesteps=num_train_timesteps, **kwargs)
        self.shape_loss_weight = shape_loss_weight

    def training_losses(self, model, input, model_kwargs=None, noise=None, weight_dtype=None):
        # sample random timestep t
        batch_size = input.shape[0]
        t = torch.randint(0, self.config.num_train_timesteps, 
                                  (batch_size,), device=input.device)
        if noise is None:
            noise = torch.randn_like(input).to(weight_dtype).to(input.device)
        noisy_input = self.add_noise(input, noise, timesteps=t).to(weight_dtype).to(input.device)
        
        # should be moved into the model forward
        if model_kwargs is not None:
            q_prev = model_kwargs.pop('q_prev')
            q_mask = model_kwargs.pop('mask')
            action = model_kwargs.pop('action')
            
        sample_input = torch.cat([q_prev, noisy_input], dim=1)
        _, num_frames, _, _, _ = sample_input.shape
        q_mask = q_mask.repeat(1, num_frames, 1, 1, 1)

        sample_input = torch.cat([sample_input, q_mask], dim=2)

        action = action

        sample_input = sample_input.permute(0, 2, 1, 3, 4) # reshape for latte
        
        model_output = model(
            hidden_states=sample_input,
            timestep=t,
            encoder_hidden_states=action
        ).sample
        
        
        
        loss = torch.mean(((model_output - noise)**2).reshape(model_output.shape[0], -1), dim=1)
        loss = loss.mean()
        
        return loss
    
    def point2dict(self, points, grid_size=0.01):
        B, N, C = points.shape
        offset = []
        for i in range(B):
            offset += [(i+1) * N]
        offset = torch.tensor(offset).int().to(points.device)
        points = points.view(-1, C)
        data_dict = {
            "feat": points,
            "coord": points[:, :3],
            "grid_size": grid_size,
            "offset": offset
        }
        return data_dict
    
    def ori_training_losses_with_cfg(
        self, 
        model,
        input, 
        pc_encoder=None,
        model_kwargs=None, 
        noise=None, 
        weight_dtype=None, 
        generator=None,
    ):
        # sample random timestep t
        batch_size = input.shape[0]
        t = torch.randint(
            0, self.config.num_train_timesteps, 
            (batch_size,), device=input.device
        )
        contour_idx = None

        if model_kwargs is not None:
            points = model_kwargs.pop('pcd')
            q_temp = model_kwargs.pop('q_temp')
            contour_idx = model_kwargs.pop('contour_idx', None)
            
        if noise is None:
            noise = torch.randn_like(input).to(weight_dtype).to(input.device)

        noisy_input = self.add_noise(input, noise, timesteps=t).to(weight_dtype).to(input.device)

        colors = torch.ones_like(points, dtype=torch.float32, device=points.device) * 0.4
        encoder_hidden_states = torch.cat([points, colors], dim=-1)
        
        if pc_encoder is not None:
            encoder_hidden_states = pc_encoder(
                encoder_hidden_states[:, :, :3].contiguous(),
                encoder_hidden_states[:, :, 3:].contiguous()
            )
            
            encoder_hidden_states = encoder_hidden_states / encoder_hidden_states.norm(dim=-1, keepdim=True)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
        # concate with template mesh vertex
        sample_input = torch.cat([q_temp.unsqueeze(1), noisy_input.unsqueeze(1)], dim=1)
        
        # repeat encoder hidden states
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(sample_input.shape[1], dim=0)
        model_output = model(
            hidden_states=sample_input,
            timestep=t,
            encoder_hidden_states=encoder_hidden_states,
        ).sample

        # --- EDGE LOSS IMPLEMENTATION ---
        if contour_idx is not None:
            # We take the contour from the first batch item (it's the same for all items in batch)
            c_idx = contour_idx[0].long()
            model_output = model_output[:, c_idx, :]
            if self.config.prediction_type == 'sample':
                input = input[:, c_idx, :]
            else:
                noise = noise[:, c_idx, :]

        if self.config.prediction_type == 'sample':
            loss = torch.mean(
                ((model_output - input) ** 2).reshape(model_output.shape[0], -1),
                dim=1
            )
        else:
            loss = torch.mean(((model_output - noise)**2).reshape(model_output.shape[0], -1), dim=1)
        
        loss = loss.mean()
        
        return loss
    
    def training_losses_with_cfg(
        self, 
        model,
        input, 
        pc_encoder=None,
        model_kwargs=None, 
        noise=None, 
        weight_dtype=None, 
        generator=None,
    ):
        batch_size = input.shape[0]
        t = torch.randint(
            0, self.config.num_train_timesteps, 
            (batch_size,), device=input.device
        )
        
        # 1. Extract inputs (Keep them as default PyTorch Float32!)
        if model_kwargs is not None:
            points = model_kwargs.pop('pcd')
            q_temp = model_kwargs.pop('q_temp')
            model_kwargs.pop('q_prev', None)  # not used in state estimation
            contour_idx = model_kwargs.pop('contour_idx', None)
            
        if noise is None:
            noise = torch.randn_like(input)

        # 2. Add noise
        noisy_input = self.add_noise(input, noise, timesteps=t)

        # concate with template mesh vertex
        sample_input = torch.cat([noisy_input, q_temp], dim=-1)
        
        # repeat encoder hidden states
        model_output = model(
            hidden_states=sample_input,
            timestep=t,
            encoder_hidden_states=points,
        ).sample

        model_output = model_output.float()
        noisy_input = noisy_input.float()
        noise = noise.float()
        
        # Handle edge_loss mode slicing
        if contour_idx is not None:
            c_idx = contour_idx[0].long()
            model_output = model_output[:, c_idx, :]
            noise = noise[:, c_idx, :]

        loss_noise = F.mse_loss(model_output, noise)
        total_loss = loss_noise

        # shape loss
        if self.shape_loss_weight > 0:
            target_x0 = input.float()
            alpha_prod_t = self.alphas_cumprod[t].to(device=input.device, dtype=torch.float32).view(-1, 1, 1)
            beta_prod_t = 1.0 - alpha_prod_t
            
            # Mathematically reconstruct x0 safely in high-precision Float32
            pred_x0 = (noisy_input - beta_prod_t ** 0.5 * model_output) / (alpha_prod_t ** 0.5 + 1e-8)

            if contour_idx is not None:
                target_x0 = target_x0[:, c_idx, :]
                pred_x0 = pred_x0[:, c_idx, :]

            loss_shape = F.mse_loss(pred_x0, target_x0)
            total_loss += self.shape_loss_weight * loss_shape
        
        # Expose per-component losses for external logging (detached floats)
        try:
            self.last_loss_noise = float(loss_noise.detach().cpu().item())
        except Exception:
            # Fallback if something unexpected happens
            self.last_loss_noise = None

        if self.shape_loss_weight > 0:
            try:
                self.last_loss_shape = float(loss_shape.detach().cpu().item())
            except Exception:
                self.last_loss_shape = None
        else:
            self.last_loss_shape = 0.0

        return total_loss