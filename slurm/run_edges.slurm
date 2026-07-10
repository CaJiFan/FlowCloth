#!/bin/bash
#
# --- Slurm Configuration ---
#SBATCH --job-name=UniCloth_Tracker
#SBATCH --output=logs/unicloth_tracker_%j.txt
#SBATCH --error=logs/unicloth_tracker_%j.err   
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40                  # 64 CPUs (14 dataloader workers * 2 GPUs + overhead)
#SBATCH --mem=120G                            # 120GB system memory
#SBATCH --time=72:00:00                       # Allow up to 3 days just in case
#SBATCH --gres=gpu:lovelace:1                 # Request EXACTLY ONE L40S GPU

echo "=========================================================="
echo "Starting UniClothDiff Distributed Training"
echo "Running on node: ${HOSTNAME}"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================================="

# Activate the correct conda environment based on your error traces
eval "$(conda shell.bash hook)"
conda activate jae

# --- Admin Required L40S Fixes ---
# (NCCL_P2P_DISABLE is especially crucial for multi-L40S setups to prevent PCIe crashes!)
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export MALLOC_ARENA_MAX=1

echo "🚀 Launching Accelerate Distributed Training on 2 GPUs..."

# --- Launch Distributed Training ---
# No '&' or 'wait' needed here. Accelerate acts as a master node 
# and automatically spawns/manages the sub-processes for both GPUs.

MODE=edge # edge
accelerate launch \
  --num_processes=1 \
  --mixed_precision="bf16" \
  train.py \
  --config="configs/train_state_est_${MODE}.yaml" \
  --exp_name="vr_diff_${MODE}" \
  --per_gpu_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=2000 \
  --num_train_epochs=600 \
  --checkpointing_steps=25000 \
  --sampling_steps=2000 \
  --checkpoints_total_limit=3 \
  --num_workers=16



echo "✅ Training completed successfully!"
conda deactivate