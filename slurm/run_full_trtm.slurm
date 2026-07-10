#!/bin/bash
#
# --- Slurm Configuration ---
#SBATCH --job-name=UniCloth_Diff_Full_TRTM
#SBATCH --output=logs/unicloth_tracker_%j.txt
#SBATCH --error=logs/unicloth_tracker_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=120G
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:lovelace:1

echo "=========================================================="
echo "Starting UniClothDiff (DDPM) TRTM Full Training"
echo "Running on node: ${HOSTNAME}"
echo "Allocated GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "=========================================================="

eval "$(conda shell.bash hook)"
conda activate jae

export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export MALLOC_ARENA_MAX=1

accelerate launch \
  --num_processes=1 \
  --mixed_precision="bf16" \
  train.py \
  --config="configs/train_state_est_full_trtm.yaml" \
  --exp_name="trtm_diff_full" \
  --per_gpu_batch_size=32 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-5 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=2000 \
  --num_train_epochs=600 \
  --checkpointing_steps=25000 \
  --sampling_steps=2000 \
  --checkpoints_total_limit=2 \
  --num_workers=16 \
  --resume_from_checkpoint="latest"

echo "✅ Training completed successfully!"
conda deactivate
