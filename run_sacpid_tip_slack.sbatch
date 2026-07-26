#!/bin/bash
#SBATCH --job-name=tip_slack
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-1
#SBATCH --output=slurm_logs/tipslack_%A_%a.out
#SBATCH --error=slurm_logs/tipslack_%A_%a.err
set -e
cd /scratch/users/k25056661/project
source env_eve.sh
# Slack control on the tip signal. d = 150 is 2.0x the five seed tip anchor
# of 74.6 and sits above the pre activation cost of both pilot seeds, 31.6
# and 113.8, so the constraint cannot bind for either.
# Purpose: the max signal d = 160 control came in at 0.082 mean success
# across five seeds against an R1 baseline of 0.138, at a budget that should
# never have bound. If that was the max signal binding after all, these runs
# return to baseline. If they do not, SACPID costs performance independent
# of the budget and every condition 1 against condition 3 success gap is
# confounded. Seeds 0 and 1 match the pilot so the pairs are comparable.
SEED=${SLURM_ARRAY_TASK_ID}
TAG="sacpid_tip_w20_d150_s${SEED}"
echo "node:   $(hostname)"
echo "start:  $(date)"
echo "task:   ${SLURM_ARRAY_TASK_ID} -> d 150 seed ${SEED} warmup 20 cost_fn tip tag ${TAG}"
python -u train_constrained.py \
  --steps 300000 \
  --steps_per_epoch 6000 \
  --seed ${SEED} \
  --cost_fn tip \
  --cost_limit 150 \
  --warmup_epochs 20 \
  --tag ${TAG}
echo "end: $(date)"
