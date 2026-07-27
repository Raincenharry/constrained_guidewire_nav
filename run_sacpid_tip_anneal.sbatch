#!/bin/bash
#SBATCH --job-name=tip_anneal
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm_logs/tipanneal_%A_%a.out
#SBATCH --error=slurm_logs/tipanneal_%A_%a.err
set -e
cd /scratch/users/k25056661/project
source env_eve.sh
# Annealing control on the tip signal. d descends linearly from 150 to 20
# across epochs 20 to 30, then holds at 20 for the rest of the run.
# Final budget 20 matches the fixed d20 runs exactly, so the only difference
# between this and sacpid_tip_w20_d20 is the approach path onto the budget.
# Purpose: the fixed d20 runs split by seed, s0 held its reach at R 0.98 while
# s1 lost 60 percent at R 0.39, and the 26 July dose response showed the loss
# is driven by binding severity B, how far above budget the policy sat when
# the constraint activated. Annealing from above keeps B near 1 throughout,
# so the prediction is that it lifts the hard seeds toward s0 without costing
# s0 anything. Start value 150 is above every seed's pre activation cost.
SEED=${SLURM_ARRAY_TASK_ID}
TAG="sacpid_tip_anneal_d20_s${SEED}"
echo "node:   $(hostname)"
echo "start:  $(date)"
echo "task:   ${SLURM_ARRAY_TASK_ID} -> anneal 150->20 seed ${SEED} warmup 20 end 30 cost_fn tip tag ${TAG}"
python -u train_constrained.py \
  --steps 300000 \
  --steps_per_epoch 6000 \
  --seed ${SEED} \
  --cost_fn tip \
  --cost_limit 20 \
  --warmup_epochs 20 \
  --anneal_from 150 \
  --anneal_end_epoch 30 \
  --tag ${TAG}
echo "end: $(date)"