#!/bin/bash
#SBATCH --job-name=r4_extra
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=3,4
#SBATCH --output=slurm_logs/r4x_%A_%a.out
#SBATCH --error=slurm_logs/r4x_%A_%a.err

set -e

cd /scratch/users/k25056661/project
source env_eve.sh

W=0.0003
SEED=${SLURM_ARRAY_TASK_ID}
TAG="r4_w${W}_s${SEED}"

echo "node:   $(hostname)"
echo "start:  $(date)"
echo "task:   weight ${W} seed ${SEED} tag ${TAG}"

python -u train_baseline.py \
  --steps 300000 \
  --steps_per_epoch 6000 \
  --seed ${SEED} \
  --force_penalty ${W} \
  --tag ${TAG}

echo "end: $(date)"
