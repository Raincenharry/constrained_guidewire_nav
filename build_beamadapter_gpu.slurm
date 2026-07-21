#!/bin/bash -l
#SBATCH --job-name=beam_build
#SBATCH --partition=bmeis_teach_gpu
#SBATCH --account=er_grp_bmeis_teach_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/users/k25056661/project/logs/%x_%j.out

module load anaconda3/2022.10-gcc-13.2.0
source activate /scratch/users/k25056661/.conda/envs/eve
cd /scratch/users/k25056661/project
CONDA=/scratch/users/k25056661/.conda/envs/eve
echo "build started $(date)"
cmake -S sofa_src -B sofa_build -G Ninja -D CMAKE_BUILD_TYPE=Release -D CMAKE_C_COMPILER=$CC -D CMAKE_CXX_COMPILER=$CXX -D CMAKE_PREFIX_PATH=$CONDA -D OpenGL_GL_PREFERENCE=GLVND -D OPENGL_INCLUDE_DIR=$CONDA/include -D OPENGL_opengl_LIBRARY=$CONDA/lib/libGL.so -D OPENGL_glx_LIBRARY=$CONDA/lib/libGL.so -D CMAKE_INSTALL_PREFIX=/scratch/users/k25056661/project/sofa_install -D Python_EXECUTABLE=$CONDA/bin/python -D pybind11_DIR=$($CONDA/bin/python -c "import pybind11; print(pybind11.get_cmake_dir())") -D SOFA_FETCH_SOFAPYTHON3=False -D SOFA_FETCH_BEAMADAPTER=False -D SOFA_EXTERNAL_DIRECTORIES=/scratch/users/k25056661/project/BeamAdapter_src -D SOFA_BUILD_TESTS=OFF -D EIGEN3_INCLUDE_DIR=$CONDA/include/eigen3 -D EIGEN3_VERSION_OK=TRUE
cmake --build sofa_build --target install -j 16
echo "build finished $(date)"
