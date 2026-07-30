#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/eval_%A_%a.out
#SBATCH --error=slurm_logs/eval_%A_%a.err

set -e

cd /scratch/users/k25056661/project
source env_eve.sh

if [ -z "$PREFIX" ]; then
  echo "PREFIX not set. Submit like:"
  echo "  sbatch --export=ALL,PREFIX=r4_w0.0003 --array=0-2 run_eval_any.sbatch"
  exit 1
fi

S=$SLURM_ARRAY_TASK_ID
RUNDIR="runs_${PREFIX}_s${S}"

if [ ! -d "$RUNDIR" ]; then
  echo "no run directory $RUNDIR"
  exit 1
fi

if [ -z "$EPOCH_OVERRIDE" ] && [ -z "$(find "$RUNDIR" -name 'epoch-50.pt' -print -quit)" ]; then
  echo "no epoch-50.pt under $RUNDIR, run is not finished"
  echo "set EPOCH_OVERRIDE to evaluate an earlier epoch on purpose"
  exit 1
fi

# Highest epoch present rather than a hardcoded epoch-50, because a timed out
# run stops short. The epoch used is printed and must match across every
# condition being compared, otherwise the comparison has a hidden variable.
#
# EPOCH_OVERRIDE picks a specific checkpoint instead, for the early stopping
# tests. It also changes the tag, so an override run can never overwrite the
# headline epoch-50 evaluation of the same run.
if [ -n "$EPOCH_OVERRIDE" ]; then
  CKPT=$(find "$RUNDIR" -name "epoch-${EPOCH_OVERRIDE}.pt" | head -1)
  if [ -z "$CKPT" ]; then
    echo "no epoch-${EPOCH_OVERRIDE}.pt under $RUNDIR, available:"
    find "$RUNDIR" -name "epoch-*.pt" -printf "%f\n" | sort -n
    exit 1
  fi
  TAG="${PREFIX}_s${S}_ep${EPOCH_OVERRIDE}"
else
  CKPT=$(find "$RUNDIR" -name "epoch-*.pt" -printf "%f %p\n" \
         | sed 's/^epoch-//' | sort -n | tail -1 | awk '{print $2}')
  if [ -z "$CKPT" ]; then
    echo "no checkpoint under $RUNDIR"
    exit 1
  fi
  TAG="${PREFIX}_s${S}"
fi
EPOCH=$(basename "$CKPT" .pt | sed 's/epoch-//')

echo "node:   $(hostname)"
echo "start:  $(date)"
echo "prefix: $PREFIX   seed: $S"
echo "ckpt:   $CKPT"
echo "tag:    $TAG"
echo "EPOCH USED: $EPOCH"

rm -f "evals/${TAG}_train_seed100.csv" "evals/${TAG}_test_seed100.csv"

python -u evaluate.py --checkpoint "$CKPT" --episodes 100 --pool train --tag ${TAG} --seed 100
python -u evaluate.py --checkpoint "$CKPT" --episodes 100 --pool test  --tag ${TAG} --seed 100

echo "end: $(date)"
