"""
Flow Matching inference pipeline for cloth state estimation.

Replaces DDPM's reverse Markov chain with a simple Euler ODE:

    x_0 ~ N(0, I)
    for t in linspace(0, 1, num_steps):
        v = model(x_t, q_temp, pcd, t * T)
        x_{t+dt} = x_t + dt * v
    return x_1  ≈  q_gt

Typical num_inference_steps: 20-50  (vs 1000 for DDPM, ~20x faster)
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import torch

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor

from uniclothdiff.models.transformer_state_est_v3 import TransformerStateEstV3Model
from uniclothdiff.schedulers.flow_matching_state_est_scheduler import FlowMatching_StateEst


@dataclass
class FlowMatchingPipelineOutput(BaseOutput):
    frames: Union[torch.FloatTensor, np.ndarray]
    result_tensor: torch.FloatTensor


class ClothStateEstFMPipeline(DiffusionPipeline):
    """
    Flow Matching pipeline for cloth state estimation.
    Shares the same call signature as ClothStateEstPipeline._call_v2
    so train.py needs only a type check to swap between them.
    """

    def __init__(
        self,
        model: TransformerStateEstV3Model,
        scheduler: FlowMatching_StateEst,
    ):
        super().__init__()
        self.register_modules(model=model, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        encoder_hidden_states: torch.FloatTensor,   # pcd  [B, P, 3]
        q_temp: torch.FloatTensor,                   # template mesh  [B, N, 3]
        shape: tuple,                                # (B, N, 3)
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        return_dict: bool = True,
        # kept for API compatibility with ClothStateEstPipeline
        call_v2: bool = True,
        do_classifier_free_guidance: bool = False,
        guidance_scale: float = 1.0,
        **kwargs,
    ):
        device = self._execution_device
        dtype  = encoder_hidden_states.dtype

        # ---- 1. Sample initial noise x_0 ~ N(0, I) ---- 
        x = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        # ---- 2. Build uniform time grid t: 0 → 1 ----
        dt = 1.0 / num_inference_steps
        # integer timestep scaling (same as training)
        T  = self.scheduler.num_train_timesteps

        with self.progress_bar(total=num_inference_steps) as pbar:
            for i in range(num_inference_steps):
                t_continuous = i / num_inference_steps            # [0, 1)
                t_int = int(t_continuous * T)
                t_tensor = torch.tensor(
                    [t_int] * shape[0], device=device, dtype=torch.long
                )

                # Concat [x_t, q_temp] → model input  [B, N, 6]
                model_input = torch.cat([x, q_temp], dim=-1)

                v_pred = self.model(
                    model_input,
                    timestep=t_tensor,
                    encoder_hidden_states=encoder_hidden_states,
                )[0]                                              # [B, N, 3]

                # Euler step
                x = x + dt * v_pred
                pbar.update()

        self.maybe_free_model_hooks()

        result_tensor = x
        frames = x.cpu().float().numpy()

        if not return_dict:
            return frames, result_tensor

        return FlowMatchingPipelineOutput(
            frames=frames,
            result_tensor=result_tensor,
        )
