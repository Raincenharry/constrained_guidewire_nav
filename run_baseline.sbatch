#!/bin/bash
#SBATCH --job-name=r1_base
#SBATCH --partition=bmeis_teach_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/r1_base_%j.out
#SBATCH --error=slurm_logs/r1_base_%j.err

set -e

cd /scratch/users/k25056661/project
source env_eve.sh

echo "node:  $(hostname)"
echo "start: $(date)"

python -u train_baseline.py --steps 300000 --seed 0 --tag r1_baseline

echo "end: $(date)"

