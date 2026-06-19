#!/bin/bash
#SBATCH --job-name=uni_ablation
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err
#SBATCH --gres=gpu:lovelace:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --array=0-5%1

# Edit the conda env name below to match your environment (e.g., `jae`)
CONDA_ENV=jae

# Map array tasks to experiment configs
EXPERIMENTS=(
  # "configs/ablation/train_fm_no_shape.yaml|fm_sw0"
  # "configs/ablation/train_fm_shape_w1.003e0.yaml|fm_sw1.003"
  # "configs/ablation/train_fm_shape_w5.016e0.yaml|fm_sw5.016"
  # "configs/ablation/train_fm_shape_w1.003e1.yaml|fm_sw10.03"
  "configs/ablation/train_fm_shape_w2.006e1.yaml|fm_sw20.06"
  "configs/ablation/train_fm_shape_w1.003e2.yaml|fm_sw100.3"
)

IDX=${SLURM_ARRAY_TASK_ID}
CFG_PAIR=${EXPERIMENTS[$IDX]}
CFG=$(echo $CFG_PAIR | cut -d'|' -f1)
NAME=$(echo $CFG_PAIR | cut -d'|' -f2)

echo "Running experiment: $NAME (config=$CFG)"

# prepare env - adjust to your environment setup
eval "$(conda shell.bash hook)"
conda activate jae

mkdir -p logs

# Seeds to run
for SEED in 1 2 3; do
  EXP_NAME="ablation_${NAME}_s${SEED}"
  accelerate launch \
    --num_processes=1 \
    --mixed_precision="bf16" \
    train.py \
    --config ${CFG} \
    --seed ${SEED} \
    --exp_name ${EXP_NAME} \
    --per_gpu_batch_size 32 \
    --num_train_epochs 600 \
    --checkpointing_steps 25000 \
    --checkpoints_total_limit 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler "cosine" \
    --learning_rate 5e-5 \
    --sampling_steps=10000 \
    --lr_warmup_steps=2000 \
    --num_workers=16 \
    --report_to wandb \
    --resume_from_checkpoint="latest"

  # small pause between seeds
  sleep 5
done
