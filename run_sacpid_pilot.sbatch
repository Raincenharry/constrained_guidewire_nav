#!/bin/bash
#SBATCH --job-name=sacpid_p
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm_logs/sacpidp_%A_%a.out
#SBATCH --error=slurm_logs/sacpidp_%A_%a.err

set -e

cd /scratch/users/k25056661/project
source env_eve.sh

# Pilot for the warmup fix. Budgets are now anchored on the measured
# unconstrained mean of 148.7 rather than on successful episodes only.
# 120 is 0.8x and 90 is 0.6x. Warmup 20 puts the constraint on after the
# success plateau at 113k steps rather than during early exploration.
DVALS=(120 90)
D_INDEX=$(( SLURM_ARRAY_TASK_ID / 2 ))
SEED=$(( SLURM_ARRAY_TASK_ID % 2 ))
D=${DVALS[$D_INDEX]}
TAG="sacpid_w20_d${D}_s${SEED}"

echo "node:   $(hostname)"
echo "start:  $(date)"
echo "task:   ${SLURM_ARRAY_TASK_ID} -> d ${D} seed ${SEED} warmup 20 tag ${TAG}"

python -u train_constrained.py \
  --steps 300000 \
  --steps_per_epoch 6000 \
  --seed ${SEED} \
  --cost_limit ${D} \
  --warmup_epochs 20 \
  --tag ${TAG}

echo "end: $(date)"
