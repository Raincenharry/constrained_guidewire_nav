#!/bin/bash
#SBATCH --job-name=r4tip
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1%2
#SBATCH --output=slurm_logs/r4tip_%A_%a.out
#SBATCH --error=slurm_logs/r4tip_%A_%a.err
set -e
cd /scratch/users/k25056661/project
source env_eve.sh
# Condition 2 on the tip signal, w=0.03, collapse boundary test.
# w0.01 evaluated healthy at stuck 0.008 and 0.000, which falsified the
# prediction that the boundary sits between 0.003 and 0.01. w0.03 is the
# replacement prediction. If it collapses, the R4 weight axis and the SACPID
# effective lambda axis are the same axis and one mechanism covers both
# conditions.
# Fixed weight, two seeds. No index arithmetic by design.
W=0.03
SEED=$SLURM_ARRAY_TASK_ID
TAG="r4tip_w${W}_s${SEED}"
echo "node:   $(hostname)"
echo "start:  $(date)"
echo "task:   ${SLURM_ARRAY_TASK_ID} -> weight ${W} seed ${SEED} cost_fn tip tag ${TAG}"
python -u train_baseline.py \
  --steps 300000 \
  --steps_per_epoch 6000 \
  --seed ${SEED} \
  --cost_fn tip \
  --force_penalty ${W} \
  --tag ${TAG}
echo "end: $(date)"
