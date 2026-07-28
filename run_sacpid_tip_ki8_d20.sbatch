#!/bin/bash
#SBATCH --job-name=tip_ki8
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm_logs/tipki8_%A_%a.out
#SBATCH --error=slurm_logs/tipki8_%A_%a.err
set -e
cd /scratch/users/k25056661/project
source env_eve.sh
# Integral gain arm on the tip signal. Fixed d=20, warmup 20, ki lowered one
# order from the OmniSafe default of 1e-7 to 1e-8. Everything else matches
# sacpid_tip_w20_d20 exactly, so the only difference is the integral gain.
#
# Purpose: ddpg_pid.py calls pid_update from _update, which runs once per
# gradient step rather than once per epoch, so the integral accumulates about
# 6000 times per epoch against a single epoch level Jc. Measured on WSL at
# 500 calls per epoch at steps_per_epoch 500. The published gains assume one
# call per epoch, so the effective integral gain here is inflated by roughly
# 6000x. The 28 July anneal run showed lambda still climbing while the error
# shrank to 2, which is windup rather than proportional response, and 11.11
# showed the policy survives at lambda below 0.009 and breaks above 0.024.
#
# Prediction: peak lambda below 0.009, stuck below 0.10, ep_cost converging
# onto 20 from above rather than undershooting. Seed 1 collapsed to 22.2 mm
# and zero success at the default gain and should land near seed 0 at 188 mm.
SEED=${SLURM_ARRAY_TASK_ID}
TAG="sacpid_tip_ki8_d20_s${SEED}"
echo "node:   $(hostname)"
echo "start:  $(date)"
echo "task:   ${SLURM_ARRAY_TASK_ID} -> fixed d20 seed ${SEED} warmup 20 ki 1e-8 cost_fn tip tag ${TAG}"
python -u train_constrained.py \
  --steps 300000 \
  --steps_per_epoch 6000 \
  --seed ${SEED} \
  --cost_fn tip \
  --cost_limit 20 \
  --warmup_epochs 20 \
  --pid_ki 1e-8 \
  --tag ${TAG}
echo "end: $(date)"