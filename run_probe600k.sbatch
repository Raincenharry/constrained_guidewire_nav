#!/bin/bash
#SBATCH --job-name=r1_base
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/r1_probe_%j.out
#SBATCH --error=slurm_logs/r1_probe_%j.err

set -e

cd /scratch/users/k25056661/project
source env_eve.sh

echo "node:  $(hostname)"
echo "start: $(date)"

python -u train_baseline.py --steps 600000 --seed 7 --tag r1_probe600k

echo "end: $(date)"

