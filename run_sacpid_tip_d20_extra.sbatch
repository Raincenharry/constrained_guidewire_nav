#!/bin/bash
#SBATCH --job-name=tip_d20x
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/tipd20x_%A_%a.out
#SBATCH --error=slurm_logs/tipd20x_%A_%a.err
set -e
cd /scratch/users/k25056661/project
source env_eve.sh
# Extends the tip force pilot at d = 20 to seeds 2, 3 and 4, bringing the
# condition to five seeds so it matches R1 and the R4 arms on n.
# d20 currently rests on two seeds that disagree completely, 180.1 mm at
# tip cost 11.7 against a collapse to 60.5 mm, so the arm is not reportable
# as it stands and the constructive claim rests on one seed.
#
# The array index IS the seed, deliberately. run_sacpid_tip_d30_extra maps
# 0,1,2 onto 2,3,4 and that indirection makes --dependency=aftercorr
# impossible, because aftercorr pairs task index to task index.
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "submit as an array, the array index is the seed:"
  echo "  sbatch --array=2-4 run_sacpid_tip_d20_extra.sbatch"
  exit 1
fi
SEED=$SLURM_ARRAY_TASK_ID
TAG="sacpid_tip_w20_d20_s${SEED}"
echo "node:   $(hostname)"
echo "start:  $(date)"
echo "task:   ${SLURM_ARRAY_TASK_ID} -> d 20 seed ${SEED} warmup 20 cost_fn tip tag ${TAG}"
python -u train_constrained.py \
  --steps 300000 \
  --steps_per_epoch 6000 \
  --seed ${SEED} \
  --cost_fn tip \
  --cost_limit 20 \
  --warmup_epochs 20 \
  --tag ${TAG}
echo "end: $(date)"