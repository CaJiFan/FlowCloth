#!/bin/bash
#
# --- Slurm Configuration ---
#SBATCH --job-name=UniCloth_Eval
#SBATCH --output=logs/unicloth_eval_%j.txt
#SBATCH --error=logs/unicloth_eval_%j.err   
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40                  # 64 CPUs (14 dataloader workers * 2 GPUs + overhead)
#SBATCH --mem=120G                            # 120GB system memory
#SBATCH --time=72:00:00                       # Allow up to 3 days just in case
#SBATCH --gres=gpu:lovelace:1                 # Request EXACTLY TWO L40S GPUs

echo "=========================================================="
echo "Starting UniClothDiff Evaluation"
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

python3 scripts/eval_clothdiff.py

echo "✅ Evaluation completed successfully!"
conda deactivate