#!/bin/bash
#SBATCH --job-name=sacpid_tip
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm_logs/sacpidtip_%A_%a.out
#SBATCH --error=slurm_logs/sacpidtip_%A_%a.err
set -e
cd /scratch/users/k25056661/project
source env_eve.sh
# Tip force pilot. The constraint now hinges on tip node force, not max
# along device. Tip force stays in the paper's regime (mean 0.24 to 0.29 N,
# max under 1 N), so budgets are anchored on the deterministic tip probe
# episode cost of 35.5. d 30 is 0.85x and d 20 is 0.56x of that.
# Warmup 20 keeps the constraint off until the success plateau at 113k.
DVALS=(30 20)
D_INDEX=$(( SLURM_ARRAY_TASK_ID / 2 ))
SEED=$(( SLURM_ARRAY_TASK_ID % 2 ))
D=${DVALS[$D_INDEX]}
TAG="sacpid_tip_w20_d${D}_s${SEED}"
echo "node:   $(hostname)"
echo "start:  $(date)"
echo "task:   ${SLURM_ARRAY_TASK_ID} -> d ${D} seed ${SEED} warmup 20 cost_fn tip tag ${TAG}"
python -u train_constrained.py \
  --steps 300000 \
  --steps_per_epoch 6000 \
  --seed ${SEED} \
  --cost_fn tip \
  --cost_limit ${D} \
  --warmup_epochs 20 \
  --tag ${TAG}
echo "end: $(date)"
