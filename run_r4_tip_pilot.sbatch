#!/bin/bash
#SBATCH --job-name=r4tip
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm_logs/r4tip_%A_%a.out
#SBATCH --error=slurm_logs/r4tip_%A_%a.err
set -e
cd /scratch/users/k25056661/project
source env_eve.sh
# Condition 2 on the tip signal. The published 0.01 and the swept 0.0003 were
# both chosen against max along device, where episode cost runs about 185.
# Tip episode cost runs about 75 with a different distribution, so the weight
# does not carry across and has to be found again.
# Three weights, two seeds each. Take the one landing nearest the SACPID tip
# cost region of 20 to 30 forward to five seeds.
WEIGHTS=(0.0003 0.001 0.003)
W_INDEX=$(( SLURM_ARRAY_TASK_ID / 2 ))
SEED=$(( SLURM_ARRAY_TASK_ID % 2 ))
W=${WEIGHTS[$W_INDEX]}
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
