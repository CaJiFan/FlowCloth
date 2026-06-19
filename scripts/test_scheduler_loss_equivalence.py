#!/usr/bin/env python3
"""
Quick test to compare OG DDPM scheduler vs modified DDPM scheduler (shape_loss_weight=0)
and to compare FM with/without shape loss. Run from repo root with:

  PYTHONPATH=UniClothDiff python3 scripts/test_scheduler_loss_equivalence.py

The script constructs a fake batch, a trivial fake model, and calls
`training_losses_with_cfg` on each scheduler with the same RNG seed and noise
so results are comparable.
"""
import os
import sys
from types import SimpleNamespace

import torch

# Ensure package import works when run from repo root with PYTHONPATH set.
try:
    from uniclothdiff.schedulers import (
        og_ddpm_state_est_scheduler as og_ddpm_mod,
        ddpm_state_est_scheduler as mod_ddpm_mod,
        flow_matching_state_est_scheduler as fm_mod,
    )
except Exception as e:
    print("Failed to import schedulers from package. Make sure to run with:\n  PYTHONPATH=UniClothDiff python3 scripts/test_scheduler_loss_equivalence.py")
    raise


class FakeModel(torch.nn.Module):
    """Minimal stand-in model returning a deterministic `.sample` tensor.

    The real schedulers only use the `.sample` attribute, so we return a
    zero-tensor shaped like [B, N, 3].
    """
    def __call__(self, hidden_states=None, timestep=None, encoder_hidden_states=None):
        B = hidden_states.shape[0]
        N = hidden_states.shape[1]
        device = hidden_states.device
        sample = torch.zeros((B, N, 3), dtype=hidden_states.dtype, device=device)
        return SimpleNamespace(sample=sample)


def run_compare():
    torch.set_num_threads(1)
    device = torch.device("cpu")

    B = 2
    N = 400
    P = 400

    # deterministic fake batch
    torch.manual_seed(0)
    target = torch.randn(B, N, 3, dtype=torch.float32, device=device)
    q_temp = torch.randn(B, N, 3, dtype=torch.float32, device=device)
    pcd = torch.randn(B, P, 3, dtype=torch.float32, device=device)

    # shared noise (pass explicitly to avoid internal RNG differences)
    torch.manual_seed(1)
    noise = torch.randn_like(target)

    fake_model = FakeModel()

    # Instantiate schedulers
    og_sched = og_ddpm_mod.OG_DDPM_StateEst(num_train_timesteps=1000)
    mod_sched_zero = mod_ddpm_mod.DDPM_StateEst(num_train_timesteps=1000, shape_loss_weight=0.0)
    mod_sched_one = mod_ddpm_mod.DDPM_StateEst(num_train_timesteps=1000, shape_loss_weight=1.0)

    fm_sched_one = fm_mod.FlowMatching_StateEst(num_train_timesteps=1000, shape_loss_weight=1.0)
    fm_sched_zero = fm_mod.FlowMatching_StateEst(num_train_timesteps=1000, shape_loss_weight=0.0)

    # Prepare identical model_kwargs (use copies for each call to avoid in-place pops)
    model_kwargs_base = {"pcd": pcd, "q_temp": q_temp}

    # Call each scheduler with the SAME RNG state for sampling `t` internally.
    seed = 1234

    torch.manual_seed(seed)
    loss_og = og_sched.training_losses_with_cfg(
        fake_model, target, None, model_kwargs_base.copy(), noise=noise
    )

    torch.manual_seed(seed)
    loss_mod_zero = mod_sched_zero.training_losses_with_cfg(
        fake_model, target, None, model_kwargs_base.copy(), noise=noise
    )

    torch.manual_seed(seed)
    loss_mod_one = mod_sched_one.training_losses_with_cfg(
        fake_model, target, None, model_kwargs_base.copy(), noise=noise
    )

    torch.manual_seed(seed)
    loss_fm_one = fm_sched_one.training_losses_with_cfg(
        fake_model, target, None, model_kwargs_base.copy(), noise=noise
    )

    torch.manual_seed(seed)
    loss_fm_zero = fm_sched_zero.training_losses_with_cfg(
        fake_model, target, None, model_kwargs_base.copy(), noise=noise
    )

    print("Results (scalar losses):")
    print(f"OG DDPM loss         : {loss_og.item():.8f}")
    print(f"Mod DDPM (w=0) loss  : {loss_mod_zero.item():.8f}")
    print(f"Mod DDPM (w=1) loss  : {loss_mod_one.item():.8f}")
    print(f"FM (w=1) loss        : {loss_fm_one.item():.8f}")
    print(f"FM (w=0) loss        : {loss_fm_zero.item():.8f}")

    print("\nPairwise differences:")
    print(f"OG vs Mod(w=0) abs diff: {abs(loss_og - loss_mod_zero).item():.12e}")
    print(f"OG vs Mod(w=1) abs diff: {abs(loss_og - loss_mod_one).item():.12e}")
    print(f"FM(w=1) vs FM(w=0) abs diff: {abs(loss_fm_one - loss_fm_zero).item():.12e}")

    # Simple check: if OG and Mod(w=0) are very close, user can drop OG config
    tol = 1e-6
    if float(abs(loss_og - loss_mod_zero)) < tol:
        print("\nOG and modified DDPM (shape_weight=0) produce numerically equivalent losses (within tol).\n")
    else:
        print("\nOG and modified DDPM (shape_weight=0) differ — keep OG config for strict baseline or investigate small implementation differences.\n")


if __name__ == "__main__":
    run_compare()
