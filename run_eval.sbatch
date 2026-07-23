#!/bin/bash
#SBATCH --job-name=r1_eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/r1_eval_s%a_%j.out
#SBATCH --error=slurm_logs/r1_eval_s%a_%j.err
set -e
cd /scratch/users/k25056661/project
source env_eve.sh

S=$SLURM_ARRAY_TASK_ID
CKPT=$(ls -t runs_r1_baseline_s${S}/SAC-*/*/torch_save/epoch-50.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
  echo "no epoch-50 checkpoint for seed $S"
  exit 1
fi

echo "node:  $(hostname)"
echo "start: $(date)"
echo "seed:  $S"
echo "ckpt:  $CKPT"

python -u evaluate.py --checkpoint "$CKPT" --episodes 100 --pool train --tag r1_s${S} --seed 100
python -u evaluate.py --checkpoint "$CKPT" --episodes 100 --pool test  --tag r1_s${S} --seed 100

echo "end: $(date)"
