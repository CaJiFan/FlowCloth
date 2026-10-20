#!/bin/bash
#SBATCH --job-name=uni_ablation_parallel
#SBATCH --output=logs/ablation_parallel_%j.out
#SBATCH --error=logs/ablation_parallel_%j.err
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=48G
#SBATCH --time=48:00:00

# Conda environment
CONDA_ENV=jae
eval "$(conda shell.bash hook)"
conda activate ${CONDA_ENV}

mkdir -p logs

# Experiments to run in parallel on this single 48GB GPU (1 seed each for rapid testing)
EXPERIMENTS=(
  "configs/ablation/train_fm_laplacian_w1.0.yaml|fm_lapl_w1.0"
  "configs/ablation/train_fm_laplacian_w5.0.yaml|fm_lapl_w5.0"
  "configs/ablation/train_fm_laplacian_w10.0.yaml|fm_lapl_w10.0"
)

# Run 1 seed per config as requested for fastest real-world turnaround
SEED=1

echo "=========================================================="
echo "Launching ${#EXPERIMENTS[@]} ablation configs in PARALLEL on 1 GPU"
echo "GPU Model: Lovelace (48GB VRAM) | Seed: ${SEED}"
echo "=========================================================="

PIDS=()
BASE_PORT=29500

for i in "${!EXPERIMENTS[@]}"; do
  CFG_PAIR=${EXPERIMENTS[$i]}
  CFG=$(echo $CFG_PAIR | cut -d'|' -f1)
  NAME=$(echo $CFG_PAIR | cut -d'|' -f2)
  EXP_NAME="ablation_${NAME}_s${SEED}"
  PORT=$((BASE_PORT + i))
  LOG_FILE="logs/${EXP_NAME}.log"

  echo "[Process $i] Starting: ${EXP_NAME} (config=${CFG}, port=${PORT}) -> ${LOG_FILE}"

  # Memory optimization for concurrent multi-process training on 1 GPU:
  # per_gpu_batch_size=16 with gradient_accumulation_steps=2 preserves effective batch size 32
  # while keeping peak VRAM per process ~7 GB (3 processes ≈ 22 GB total, well inside 48 GB).
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
    --patience_limit 10 \
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

  PID=$!
  PIDS+=($PID)
  echo "[Process $i] Launched PID ${PID}"

  # Small staggering pause to avoid simultaneous disk I/O burst during template loading
  sleep 10
done

echo "All ${#PIDS[@]} processes launched in background. Waiting for completion..."

# Wait for all background training processes to finish
FAIL=0
for pid in "${PIDS[@]}"; do
  wait $pid || FAIL=1
done

if [ $FAIL -eq 0 ]; then
  echo "All parallel training runs finished successfully!"
else
  echo "One or more parallel training runs encountered an error. Check logs/ for details."
  exit 1
fi
