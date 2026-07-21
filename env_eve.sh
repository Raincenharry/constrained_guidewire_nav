#!/bin/bash
# Source this once per session:  source env_eve.sh

module load anaconda3/2022.10-gcc-13.2.0
source activate /scratch/users/k25056661/.conda/envs/eve

export SOFA_ROOT=/scratch/users/k25056661/project/sofa_install
export PYTHONPATH=$SOFA_ROOT/plugins/SofaPython3/lib/python3/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=$SOFA_ROOT/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "eve env ready"
echo "  SOFA_ROOT=$SOFA_ROOT"
echo "  python:  $(which python)"
