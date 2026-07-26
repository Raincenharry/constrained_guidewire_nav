#!/bin/bash
#SBATCH --job-name=tip_d30x
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm_logs/tipd30x_%A_%a.out
#SBATCH --error=slurm_logs/tipd30x_%A_%a.err
set -e
cd /scratch/users/k25056661/project
source env_eve.sh
# Extends the tip force pilot at d = 30 to seeds 2, 3 and 4, bringing the
# condition to five seeds so it matches R1 and R4 on n.
# d = 30 is 0.40x of the five seed tip anchor mean of 74.6 and 0.64x of the
# five seed median of 47. Seeds 0 and 1 are already done as job 36034070.
SEEDS=(2 3 4)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
TAG="sacpid_tip_w20_d30_s${SEED}"
echo "node:   $(hostname)"
echo "start:  $(date)"
echo "task:   ${SLURM_ARRAY_TASK_ID} -> d 30 seed ${SEED} warmup 20 cost_fn tip tag ${TAG}"
python -u train_constrained.py \
  --steps 300000 \
  --steps_per_epoch 6000 \
  --seed ${SEED} \
  --cost_fn tip \
  --cost_limit 30 \
  --warmup_epochs 20 \
  --tag ${TAG}
echo "end: $(date)"
