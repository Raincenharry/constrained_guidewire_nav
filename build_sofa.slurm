#!/bin/bash -l
#SBATCH --job-name=sofa_build
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/users/k25056661/project/logs/%x_%j.out

module load anaconda3/2022.10-gcc-13.2.0
source activate /scratch/users/k25056661/.conda/envs/eve

cd /scratch/users/k25056661/project

echo "build started $(date)"
cmake --build sofa_build --target install -j 16
echo "build finished $(date)"
