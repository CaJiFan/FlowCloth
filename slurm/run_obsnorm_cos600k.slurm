#!/bin/bash
#SBATCH --job-name=uni_obsnorm_cos600k
#SBATCH --output=logs/obsnorm_cos600k_%j.out
#SBATCH --error=logs/obsnorm_cos600k_%j.err
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=96G
#SBATCH --time=72:00:00

CONDA_ENV=jae
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024
export MALLOC_ARENA_MAX=1
# wandb.init currently fails server-side (CommError); log offline, upload later with `wandb sync`.
# Override at submit time with: sbatch --export=ALL,WANDB_MODE=online ...
export WANDB_MODE=${WANDB_MODE:-offline}

# Auto-detect if Slurm assigned a GPU hijacked by an unmanaged rogue process (<30GB free)
CURRENT_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -n 1)
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  CURRENT_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" 2>/dev/null || echo "$CURRENT_FREE")
fi

echo "Current assigned GPU free VRAM: ${CURRENT_FREE} MiB"

if [ -z "$CURRENT_FREE" ] || [ "$CURRENT_FREE" -lt 30000 ]; then
  echo "WARNING: Assigned GPU has only ${CURRENT_FREE}MiB free (rogue process detected!)."
  BEST_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -k2 -n -r | head -n 1 | awk -F', ' '{print $1}')
  BEST_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${BEST_GPU}" 2>/dev/null)
  if [ -n "$BEST_GPU" ] && [ "$BEST_FREE" -ge 30000 ]; then
    echo "Automatically switching to cleanest GPU ${BEST_GPU} (${BEST_FREE} MiB free)..."
    export CUDA_VISIBLE_DEVICES=${BEST_GPU}
  else
    echo "ERROR: No GPU on this node has sufficient free memory (>= 30GB). Aborting to prevent OOM crash."
    exit 1
  fi
fi

mkdir -p logs

# N (observation normalisation) with a cosine LR schedule that actually decays: 600k steps
# (current runs decay over 900k steps but early-stop around 300k-1M, i.e. barely annealed).
# Patience 30 evaluations (300k steps) so the schedule, not early stopping, sets the length.
#   Full mesh and contour variant share the GPU.
EXPERIMENTS=(
  "configs/ablation/train_fm_obsnorm.yaml|fm_obsnorm_cos600k"
  "configs/ablation/train_fm_edge_obsnorm.yaml|fm_edge_obsnorm_cos600k"
)

SEED=1
BASE_PORT=29620
PIDS=()

echo "Launching ${#EXPERIMENTS[@]} parallel cos600k jobs on 1 Lovelace GPU..."

for i in "${!EXPERIMENTS[@]}"; do
  CFG_PAIR=${EXPERIMENTS[$i]}
  CFG=$(echo $CFG_PAIR | cut -d'|' -f1)
  NAME=$(echo $CFG_PAIR | cut -d'|' -f2)
  EXP_NAME="ablation_${NAME}_s${SEED}"
  PORT=$((BASE_PORT + i))
  LOG_FILE="logs/${EXP_NAME}.log"

  echo "Starting ${EXP_NAME} on port ${PORT} (config: ${CFG}, log: ${LOG_FILE})..."

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
    --num_train_epochs 999999 \
    --max_train_steps 600000 \
    --use_hybrid_patience_loss \
    --patience_limit 30 \
    --learning_rate 1e-4 \
    --lr_warmup_steps 3000 \
    --lr_scheduler "cosine" \
    --allow_tf32 \
    --checkpointing_steps 25000 \
    --checkpoints_total_limit 1 \
    --sampling_steps 10000 \
    --max_val_samples 1000 \
    --val_inference_steps 10 \
    --num_workers 8 \
    --report_to wandb \
    --resume_from_checkpoint "latest" > "${LOG_FILE}" 2>&1 &

  PIDS+=($!)
  sleep 10
done

echo "All ${#EXPERIMENTS[@]} background processes running (PIDs: ${PIDS[*]}). Waiting for completion..."

FAILED=0
for pid in "${PIDS[@]}"; do
  wait $pid || FAILED=1
done

if [ $FAILED -ne 0 ]; then
  echo "ERROR: One or more parallel training runs failed. Check logs in logs/!"
  exit 1
fi

echo "All experiments finished successfully."
