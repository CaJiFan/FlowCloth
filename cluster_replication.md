# Cluster Server Replication Guide: FlowCloth Laplacian Loss & Ablation Suite

This guide contains the exact changes, new configurations, and Slurm scripts needed to reproduce the retraining setup on your Slurm-based cluster server.

> **Instruction for Antigravity on the Cluster Server**:  
> If you are reading this on the cluster server, apply all code changes, config files, and Slurm scripts described in this document to the repository.

---

## Table of Contents
1. [Overview of Changes](#1-overview-of-changes)
2. [Code Modifications](#2-code-modifications)
   - [A. Scheduler: `uniclothdiff/schedulers/flow_matching_state_est_scheduler.py`](#a-scheduler-flow_matching_state_est_schedulerpy)
   - [B. CLI Arguments: `args.py`](#b-cli-arguments-argspy)
   - [C. Training Loop: `train.py`](#c-training-loop-trainpy)
   - [D. Transformer Assets Fallback: `uniclothdiff/models/transformer_state_est_v3.py`](#d-transformer-assets-fallback-transformer_state_est_v3py)
   - [E. Dataset Path Fallback: `uniclothdiff/datasets/trtm_dataset.py`](#e-dataset-path-fallback-trtm_datasetpy)
3. [Base Config Updates](#3-base-config-updates)
   - [`configs/train_state_est_full_fm_trtm.yaml`](#configstrain_state_est_full_fm_trtmyaml)
   - [`configs/train_state_est_edge_fm_trtm.yaml`](#configstrain_state_est_edge_fm_trtmyaml)
4. [Ablation Configs (`configs/ablation/`)](#4-ablation-configs-configsablation)
   - [Full-Mesh Models (441 Nodes)](#full-mesh-models-441-nodes)
   - [Contour-FlowCloth Models (80 Nodes)](#contour-flowcloth-models-80-nodes)
5. [Slurm Scripts (`slurm/`)](#5-slurm-scripts-slurm)
   - [`slurm/run_ablation_fm_contour.slurm` (Run First)](#slurmrun_ablation_fm_contourslurm)
   - [`slurm/run_ablation_fm_full.slurm` (Queue Up Next)](#slurmrun_ablation_fm_fullslurm)
   - [`slurm/run_ablation_fm.slurm`](#slurmrun_ablation_fmslurm)

---

## 1. Overview of Changes

1. **4-Term Consolidated Loss Formulation**:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{vel}} + 100.0 \cdot \mathcal{L}_{\text{shape}} + \lambda_{\text{lapl}} \cdot \mathcal{L}_{\text{laplacian}} + 2.5 \cdot \mathcal{L}_{\text{corner}}$$
   - Replaced $\mathcal{L}_{\text{edge}}$ ($0.0$) with $\mathcal{L}_{\text{laplacian}}$: removes square rest-length rigidity that forced the model to curl corners during half-folds.
   - Disabled redundant $\mathcal{L}_{\text{contour}}$ ($0.0$).
   - Kept modest $\mathcal{L}_{\text{corner}}$ ($2.5$) for grasp point precision.
2. **Topology-Based Discrete Laplacian**:
   - **Full Mesh (441 nodes / 21×21)**: 2D 5-point grid Laplacian on interior nodes.
   - **Contour Mesh (80 nodes / edge-only)**: 1D closed-contour perimeter Laplacian on straight-edge nodes (excluding the 4 $90^\circ$ corners so straight edges remain identically 0 error under arbitrary scaling).
3. **Training & Early Stopping**:
   - **1000 Epochs**: Set to `--num_train_epochs 1000`.
   - **Hybrid Patience**: `--use_hybrid_patience_loss` ($\text{corner\_error} + 1.5 \cdot \text{chamfer\_l1}$) with `--patience_limit 8`.
   - **Parallel 1-GPU Slurm Scripts**: Runs 3 weight variants (`w1.0`, `w5.0`, `w10.0`) concurrently on a single 48GB Lovelace GPU using unique ports and `--per_gpu_batch_size 16 --gradient_accumulation_steps 2` (effective batch size 32, ~22 GB total peak VRAM).

---

## 2. Code Modifications

### A. Scheduler: `uniclothdiff/schedulers/flow_matching_state_est_scheduler.py`
1. Add `import numpy as np` to top imports.
2. In `__init__`, add:
   ```python
   laplacian_loss_weight: float = 0.0,
   ```
   and store `self.laplacian_loss_weight = laplacian_loss_weight`, plus `self.last_loss_laplacian = 0.0`.
3. In `training_losses_with_cfg(...)`, implement the multi-objective losses:
   - Compute `loss_corner` (weight `self.corner_loss_weight`).
   - Compute `loss_contour` (weight `self.contour_loss_weight`).
   - Compute `loss_edge` (weight `self.edge_loss_weight`).
   - Compute `loss_laplacian` (weight `self.laplacian_loss_weight`):
     ```python
     if self.laplacian_loss_weight > 0:
         num_nodes = pred_x1_active.shape[1]
         if num_nodes in (441, 400):
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

     self.last_loss_laplacian = float(loss_laplacian.detach().cpu().item())
     ```

### B. CLI Arguments: `args.py`
Add CLI overrides for the loss terms:
```python
parser.add_argument("--shape_loss_weight", type=float, default=None, help="Override shape loss weight")
parser.add_argument("--laplacian_loss_weight", type=float, default=None, help="Override laplacian loss weight")
parser.add_argument("--corner_loss_weight", type=float, default=None, help="Override corner loss weight")
parser.add_argument("--contour_loss_weight", type=float, default=None, help="Override contour loss weight")
parser.add_argument("--edge_loss_weight", type=float, default=None, help="Override edge loss weight")
```

### C. Training Loop: `train.py`
1. **Pass CLI overrides into `diffusion_scheduler`**:
   ```python
   diff_cfg = OmegaConf.to_container(config.diffusion_cfg)
   for k in ["shape_loss_weight", "laplacian_loss_weight", "corner_loss_weight", "contour_loss_weight", "edge_loss_weight"]:
       if getattr(config, k, None) is not None:
           diff_cfg[k] = getattr(config, k)
   diffusion_scheduler = build_scheduler(diff_cfg)
   ```
2. **Safe Pipeline unwrapping**:
   ```python
   sched = accelerator.unwrap_model(diffusion_scheduler) if isinstance(diffusion_scheduler, torch.nn.Module) else diffusion_scheduler
   ```
3. **Telemetry & Logging**:
   Add `"lapl"` to `pbar_updates` and `"train/loss_laplacian"` to `accelerator.log`.
4. **Fix `max_train_steps` termination**:
   ```python
   if config.max_train_steps is None or config.max_train_steps <= 0:
       config.max_train_steps = config.num_train_epochs * num_update_step_per_epoch
   else:
       config.num_train_epochs = math.ceil(config.max_train_steps / num_update_step_per_epoch)
   ```
   and break when `global_step >= config.max_train_steps`.

### D. Transformer Assets Fallback: `uniclothdiff/models/transformer_state_est_v3.py`
Ensure `import os` is present, and add a local assets fallback if `patch_file` path differs across systems:
```python
if not os.path.exists(patch_file):
    alt_patch = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "assets", os.path.basename(patch_file))
    if os.path.exists(alt_patch):
        patch_file = alt_patch
```

### E. Dataset Path Fallback: `uniclothdiff/datasets/trtm_dataset.py`
In `__init__`, check fallback dataset paths:
```python
if not os.path.exists(data_dir):
    for candidate in [
        "/mnt/Data/cjimenez/TRTM/datasets/template_square",
        os.path.expanduser("~/projects/datasets/TRTM/template_square"),
        os.path.expanduser("~/Projects/datasets/TRTM/template_square"),
    ]:
        if os.path.exists(candidate):
            data_dir = candidate
            break
```

---

## 3. Base Config Updates

### `configs/train_state_est_full_fm_trtm.yaml`
Update `diffusion_cfg`:
```yaml
diffusion_cfg:
  type: "FlowMatching_StateEst"
  num_train_timesteps: 1000
  shape_loss_weight: 100.0
  laplacian_loss_weight: 5.0
  corner_loss_weight: 2.5
  contour_loss_weight: 0.0
  edge_loss_weight: 0.0
```

### `configs/train_state_est_edge_fm_trtm.yaml`
Update `diffusion_cfg`:
```yaml
diffusion_cfg:
  type: "FlowMatching_StateEst"
  num_train_timesteps: 1000
  shape_loss_weight: 100.0
  laplacian_loss_weight: 5.0
  corner_loss_weight: 2.5
  contour_loss_weight: 0.0
  edge_loss_weight: 0.0
```

---

## 4. Ablation Configs (`configs/ablation/`)

### Full-Mesh Models (441 Nodes)

#### `configs/ablation/train_fm_laplacian.yaml` & `configs/ablation/train_fm_laplacian_w5.0.yaml`
```yaml
model_cfg:
  type: "TransformerStateEstV3Model"
  num_groups: 64
  group_size: 16
  in_channels: 6
  out_channels: 3
  num_layers: 6
  attention_head_dim: 72
  patchified_input: true
  num_in_frames: 2
  num_out_frames: 1
  patch_file: /home/cjimenez/projects/TrackAnyCloth/UniClothDiff/assets/trtm_441_voronoi_template.pkl

diffusion_cfg:
  type: "FlowMatching_StateEst"
  num_train_timesteps: 1000
  shape_loss_weight: 100.0
  laplacian_loss_weight: 5.0
  corner_loss_weight: 2.5
  contour_loss_weight: 0.0
  edge_loss_weight: 0.0

wandb_cfg:
  project_name: "UniClothDiff_TRTM"
  exp_name: "ablation_fm_laplacian_w5.0"

dataset_cfg:
  type: "TRTMDiffusionDataset"
  data_dir: "/home/cjimenez/projects/datasets/TRTM/template_square"
  train_mode: "full"
  num_sample_points: 400
  do_point_cloud_augmentation: true
  points_jitter_sigma: 0.002
  points_drop_ratio: 0.0
```

#### `configs/ablation/train_fm_laplacian_w1.0.yaml`
Same as above, with:
```yaml
diffusion_cfg:
  laplacian_loss_weight: 1.0
wandb_cfg:
  exp_name: "ablation_fm_laplacian_w1.0"
```

#### `configs/ablation/train_fm_laplacian_w10.0.yaml`
Same as above, with:
```yaml
diffusion_cfg:
  laplacian_loss_weight: 10.0
wandb_cfg:
  exp_name: "ablation_fm_laplacian_w10.0"
```

---

### Contour-FlowCloth Models (80 Nodes)

#### `configs/ablation/train_fm_edge_laplacian.yaml` & `configs/ablation/train_fm_edge_laplacian_w5.0.yaml`
```yaml
model_cfg:
  type: "TransformerStateEstV3Model"
  num_groups: 32
  group_size: 8
  in_channels: 6
  out_channels: 3
  num_layers: 6
  attention_head_dim: 72
  patchified_input: true
  num_in_frames: 2
  num_out_frames: 1
  patch_file: /home/cjimenez/projects/TrackAnyCloth/UniClothDiff/assets/trtm_80_voronoi_template.pkl

diffusion_cfg:
  type: "FlowMatching_StateEst"
  num_train_timesteps: 1000
  shape_loss_weight: 100.0
  laplacian_loss_weight: 5.0
  corner_loss_weight: 2.5
  contour_loss_weight: 0.0
  edge_loss_weight: 0.0

wandb_cfg:
  project_name: "UniClothDiff_TRTM"
  exp_name: "ablation_fm_edge_laplacian_w5.0"

dataset_cfg:
  type: "TRTMDiffusionDataset"
  data_dir: "/home/cjimenez/projects/datasets/TRTM/template_square"
  train_mode: "edge_only"
  num_sample_points: 400
  do_point_cloud_augmentation: true
  points_jitter_sigma: 0.002
  points_drop_ratio: 0.0
```

#### `configs/ablation/train_fm_edge_laplacian_w1.0.yaml`
Same as above, with:
```yaml
diffusion_cfg:
  laplacian_loss_weight: 1.0
wandb_cfg:
  exp_name: "ablation_fm_edge_laplacian_w1.0"
```

#### `configs/ablation/train_fm_edge_laplacian_w10.0.yaml`
Same as above, with:
```yaml
diffusion_cfg:
  laplacian_loss_weight: 10.0
wandb_cfg:
  exp_name: "ablation_fm_edge_laplacian_w10.0"
```

---

## 5. Slurm Scripts (`slurm/`)

### `slurm/run_ablation_fm_contour.slurm`
```bash
#!/bin/bash
#SBATCH --job-name=uni_ablation_contour
#SBATCH --output=logs/ablation_contour_%j.out
#SBATCH --error=logs/ablation_contour_%j.err
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=48G
#SBATCH --time=48:00:00

CONDA_ENV=jae
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

mkdir -p logs

EXPERIMENTS=(
  "configs/ablation/train_fm_edge_laplacian_w1.0.yaml|fm_edge_lapl_w1.0"
  "configs/ablation/train_fm_edge_laplacian_w5.0.yaml|fm_edge_lapl_w5.0"
  "configs/ablation/train_fm_edge_laplacian_w10.0.yaml|fm_edge_lapl_w10.0"
)

SEED=1
BASE_PORT=29510
PIDS=()

for i in "${!EXPERIMENTS[@]}"; do
  CFG_PAIR=${EXPERIMENTS[$i]}
  CFG=$(echo $CFG_PAIR | cut -d'|' -f1)
  NAME=$(echo $CFG_PAIR | cut -d'|' -f2)
  EXP_NAME="ablation_${NAME}_s${SEED}"
  PORT=$((BASE_PORT + i))
  LOG_FILE="logs/${EXP_NAME}.log"

  accelerate launch \
    --main_process_port ${PORT} \
    --num_processes=1 \
    --mixed_precision="bf16" \
    train.py \
    --config ${CFG} \
    --seed ${SEED} \
    --exp_name ${EXP_NAME} \
    --per_gpu_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 1000 \
    --use_hybrid_patience_loss \
    --patience_limit 8 \
    --learning_rate 1e-4 \
    --lr_warmup_steps 3000 \
    --lr_scheduler "cosine" \
    --allow_tf32 \
    --checkpointing_steps 25000 \
    --checkpoints_total_limit 1 \
    --sampling_steps 10000 \
    --num_workers 8 \
    --report_to wandb \
    --resume_from_checkpoint "latest" > "${LOG_FILE}" 2>&1 &

  PIDS+=($!)
  sleep 10
done

for pid in "${PIDS[@]}"; do
  wait $pid
done
```

### `slurm/run_ablation_fm_full.slurm`
```bash
#!/bin/bash
#SBATCH --job-name=uni_ablation_full
#SBATCH --output=logs/ablation_full_%j.out
#SBATCH --error=logs/ablation_full_%j.err
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=48G
#SBATCH --time=48:00:00

CONDA_ENV=jae
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

mkdir -p logs

EXPERIMENTS=(
  "configs/ablation/train_fm_laplacian_w1.0.yaml|fm_lapl_w1.0"
  "configs/ablation/train_fm_laplacian_w5.0.yaml|fm_lapl_w5.0"
  "configs/ablation/train_fm_laplacian_w10.0.yaml|fm_lapl_w10.0"
)

SEED=1
BASE_PORT=29500
PIDS=()

for i in "${!EXPERIMENTS[@]}"; do
  CFG_PAIR=${EXPERIMENTS[$i]}
  CFG=$(echo $CFG_PAIR | cut -d'|' -f1)
  NAME=$(echo $CFG_PAIR | cut -d'|' -f2)
  EXP_NAME="ablation_${NAME}_s${SEED}"
  PORT=$((BASE_PORT + i))
  LOG_FILE="logs/${EXP_NAME}.log"

  accelerate launch \
    --main_process_port ${PORT} \
    --num_processes=1 \
    --mixed_precision="bf16" \
    train.py \
    --config ${CFG} \
    --seed ${SEED} \
    --exp_name ${EXP_NAME} \
    --per_gpu_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 1000 \
    --use_hybrid_patience_loss \
    --patience_limit 8 \
    --learning_rate 1e-4 \
    --lr_warmup_steps 3000 \
    --lr_scheduler "cosine" \
    --allow_tf32 \
    --checkpointing_steps 25000 \
    --checkpoints_total_limit 1 \
    --sampling_steps 10000 \
    --num_workers 8 \
    --report_to wandb \
    --resume_from_checkpoint "latest" > "${LOG_FILE}" 2>&1 &

  PIDS+=($!)
  sleep 10
done

for pid in "${PIDS[@]}"; do
  wait $pid
done
```
