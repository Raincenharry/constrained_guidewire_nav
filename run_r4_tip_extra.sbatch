#!/bin/bash
#SBATCH --job-name=r4tipx
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/r4tipx_%A_%a.out
#SBATCH --error=slurm_logs/r4tipx_%A_%a.err
set -e
cd /scratch/users/k25056661/project
source env_eve.sh
# Adds seeds to an existing R4 tip weight. The weight comes in on the
# command line rather than from an index into an array, because the index
# arithmetic in run_r4_tip_w01.sbatch silently produces an empty weight for
# any task above 1.
#
# The array index IS the seed, so the evaluation can be chained with
# --dependency=aftercorr.
if [ -z "$W" ]; then
  echo "W not set. Submit like:"
  echo "  sbatch --export=ALL,W=0.003 --array=2 run_r4_tip_extra.sbatch"
  exit 1
fi
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
  echo "submit as an array, the array index is the seed"
  exit 1
fi
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