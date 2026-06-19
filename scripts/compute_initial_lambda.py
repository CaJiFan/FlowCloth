#!/usr/bin/env python3
"""
Compute initial shape-loss weight (lambda) from a small batch.

Usage (from repo root):

  PYTHONPATH=UniClothDiff python3 scripts/compute_initial_lambda.py \
      --config configs/ablation/train_ddpm_shape_w1.yaml \
      --method ddpm --device cuda --batches 2

Outputs per-batch `loss_noise`, `loss_shape`, the ratio `lambda0 = loss_noise / loss_shape`,
and a recommended sweep list around `lambda0`.

Note: this script requires a CUDA GPU because the model's forward uses `.cuda()`.
"""
import argparse
from omegaconf import OmegaConf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from uniclothdiff.registry import build_model, build_dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='configs/ablation/train_ddpm_shape_w1.yaml')
    p.add_argument('--method', type=str, choices=['ddpm', 'fm'], default='ddpm')
    p.add_argument('--device', type=str, choices=['cuda','cpu'], default='cuda')
    p.add_argument('--batches', type=int, default=1, help='Number of batches to average over')
    p.add_argument('--batch_size', type=int, default=1)
    return p.parse_args()


def compute_on_batch_ddpm(sched, model, batch, device):
    # expects batch to contain 'q_gt', 'pcd', 'q_temp'
    x = batch['q_gt'].to(device)
    points = batch['pcd'].to(device)
    q_temp = batch['q_temp'].to(device)

    B = x.shape[0]

    # sample timestep t
    T = getattr(getattr(sched, 'config', None), 'num_train_timesteps', None)
    if T is None:
        T = getattr(sched, 'num_train_timesteps', 1000)
    t = torch.randint(0, T, (B,), device=device)

    noise = torch.randn_like(x, device=device)
    noisy_x = sched.add_noise(x, noise, timesteps=t)

    sample_input = torch.cat([noisy_x, q_temp.to(noisy_x.dtype)], dim=-1)
    model_out = model(hidden_states=sample_input, timestep=t, encoder_hidden_states=points).sample
    model_out = model_out.float()
    noise_f = noise.float()

    loss_noise = F.mse_loss(model_out, noise_f).item()

    # reconstruct x0 and shape loss
    alpha_prod_t = sched.alphas_cumprod[t].to(device=device, dtype=torch.float32).view(-1, 1, 1)
    beta_prod_t = 1.0 - alpha_prod_t
    pred_x0 = (noisy_x.float() - (beta_prod_t ** 0.5) * model_out) / (alpha_prod_t ** 0.5 + 1e-8)
    loss_shape = F.mse_loss(pred_x0, x.float()).item()

    return loss_noise, loss_shape


def compute_on_batch_fm(sched, model, batch, device):
    x = batch['q_gt'].to(device)
    points = batch['pcd'].to(device)
    q_temp = batch['q_temp'].to(device)

    B = x.shape[0]
    # sample noise x0 & continuous t
    x0 = torch.randn_like(x, device=device)
    t_cont = torch.rand(B, device=device)
    t_view = t_cont.view(-1, 1, 1)
    x_t = (1.0 - t_view) * x0 + t_view * x

    v_target = x - x0
    # scale t to integer range for model's AdaLN
    T = getattr(getattr(sched, 'num_train_timesteps', None), 'num_train_timesteps', None)
    # some schedulers store num_train_timesteps directly
    if hasattr(sched, 'num_train_timesteps'):
        T = sched.num_train_timesteps
    if T is None:
        T = 1000
    t_int = (t_cont * T).long().clamp(0, T - 1)

    sample_input = torch.cat([x_t, q_temp.to(x_t.dtype)], dim=-1)
    model_out = model(hidden_states=sample_input, timestep=t_int, encoder_hidden_states=points).sample
    model_out = model_out.float()

    loss_vel = F.mse_loss(model_out, v_target.float()).item()

    # reconstruct x1 and shape loss
    pred_x1 = x_t.float() + (1.0 - t_view.expand_as(x_t)) * model_out
    loss_shape = F.mse_loss(pred_x1, x.float()).item()

    return loss_vel, loss_shape


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    if device.type == 'cpu' and args.device == 'cuda':
        raise RuntimeError('CUDA not available — this script requires a GPU because the model forward uses .cuda() internally.')

    # build model and dataset
    model = build_model(OmegaConf.to_container(cfg.model_cfg))
    model.to(device)
    model.eval()

    dataset_cfg = OmegaConf.to_container(cfg.dataset_cfg)
    dataset_cfg['mode'] = 'val'
    dataset = build_dataset(dataset_cfg)

    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # instantiate scheduler class directly
    if args.method == 'ddpm':
        from uniclothdiff.schedulers import ddpm_state_est_scheduler as ddpm_mod
        sched = ddpm_mod.DDPM_StateEst(num_train_timesteps=cfg.diffusion_cfg.num_train_timesteps,
                                        shape_loss_weight=cfg.diffusion_cfg.get('shape_loss_weight', 1.0))
    else:
        from uniclothdiff.schedulers import flow_matching_state_est_scheduler as fm_mod
        sched = fm_mod.FlowMatching_StateEst(num_train_timesteps=cfg.diffusion_cfg.num_train_timesteps,
                                             shape_loss_weight=cfg.diffusion_cfg.get('shape_loss_weight', 1.0))

    sched_device = device

    losses_noise = []
    losses_shape = []

    it = iter(dl)
    for i in range(args.batches):
        try:
            batch = next(it)
        except StopIteration:
            print('Dataset exhausted')
            break

        if args.method == 'ddpm':
            ln, ls = compute_on_batch_ddpm(sched, model, batch, device)
            print(f'Batch {i}: loss_noise={ln:.6e}, loss_shape={ls:.6e}')
        else:
            ln, ls = compute_on_batch_fm(sched, model, batch, device)
            print(f'Batch {i}: loss_vel={ln:.6e}, loss_shape={ls:.6e}')

        losses_noise.append(ln)
        losses_shape.append(ls)

    if len(losses_noise) == 0:
        print('No batches processed')
        return

    avg_noise = sum(losses_noise) / len(losses_noise)
    avg_shape = sum(losses_shape) / len(losses_shape)

    lambda0 = avg_noise / (avg_shape + 1e-12)

    print('\n---- Summary ----')
    print(f'Avg noise loss : {avg_noise:.6e}')
    print(f'Avg shape loss : {avg_shape:.6e}')
    print(f'Initial lambda0 (noise/shape) : {lambda0:.6e}\n')

    # Suggested sweep around lambda0
    sweep = [lambda0 / 10.0, lambda0 / 2.0, lambda0, lambda0 * 2.0, lambda0 * 10.0]
    # clamp to reasonable bounds
    sweep = [max(s, 1e-12) for s in sweep]
    print('Suggested sweep (lambda values):')
    print(', '.join([f'{s:.3e}' for s in sweep]))


if __name__ == '__main__':
    main()
