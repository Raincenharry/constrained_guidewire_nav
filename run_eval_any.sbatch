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

# More than one timestamped run directory means a resubmission happened and
# the checkpoints come from two different trainings. Stop rather than guess.
NRUNS=$(find "$RUNDIR" -type d -name "torch_save" | wc -l)
if [ "$NRUNS" -ne 1 ]; then
  echo "expected 1 torch_save directory under $RUNDIR, found $NRUNS"
  find "$RUNDIR" -type d -name "torch_save"
  exit 1
fi

# Highest epoch present rather than a hardcoded epoch-50, because a timed out
# run stops short. The epoch used is printed and must match across every
# condition being compared, otherwise the comparison has a hidden variable.
CKPT=$(find "$RUNDIR" -name "epoch-*.pt" -printf "%f %p\n" \
       | sed 's/^epoch-//' | sort -n | tail -1 | awk '{print $2}')
if [ -z "$CKPT" ]; then
  echo "no checkpoint under $RUNDIR"
  exit 1
fi
EPOCH=$(basename "$CKPT" .pt | sed 's/epoch-//')

echo "node:   $(hostname)"
echo "start:  $(date)"
echo "prefix: $PREFIX   seed: $S"
echo "ckpt:   $CKPT"
echo "EPOCH USED: $EPOCH"

python -u evaluate.py --checkpoint "$CKPT" --episodes 100 --pool train --tag ${PREFIX}_s${S} --seed 100
python -u evaluate.py --checkpoint "$CKPT" --episodes 100 --pool test  --tag ${PREFIX}_s${S} --seed 100

echo "end: $(date)"
