#!/bin/bash
#SBATCH --job-name=sacpid
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-19%8
#SBATCH --output=slurm_logs/sacpid_%A_%a.out
#SBATCH --error=slurm_logs/sacpid_%A_%a.err

set -e

cd /scratch/users/k25056661/project
source env_eve.sh

# task 0 to 19: d index is task / 5, seed is task modulo 5
DVALS=(10 40 60 160)
D_INDEX=$(( SLURM_ARRAY_TASK_ID / 5 ))
SEED=$(( SLURM_ARRAY_TASK_ID % 5 ))
D=${DVALS[$D_INDEX]}
TAG="sacpid_d${D}_s${SEED}"

echo "node:   $(hostname)"
echo "start:  $(date)"
echo "task:   ${SLURM_ARRAY_TASK_ID} -> d ${D} seed ${SEED} tag ${TAG}"

python -u train_constrained.py \
  --steps 300000 \
  --steps_per_epoch 6000 \
  --seed ${SEED} \
  --cost_limit ${D} \
  --warmup_epochs 5 \
  --tag ${TAG}

echo "end: $(date)"
