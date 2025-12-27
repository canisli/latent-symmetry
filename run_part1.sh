#!/bin/bash
#SBATCH -p iaifi_gpu
#SBATCH --ntasks=1
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=32GB
#SBATCH -o logs/%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e logs/%j.err  # File to which STDERR will be written, %j inserts jobid

echo "$@"

# load modules
source ~/venvs/venv_3.12/bin/activate
export MATPLOTLIBRC=$HOME/.matplotlib/matplotlibrc

module load cuda/12.4.1-fasrc01
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/n/sw/helmod-rocky8/apps/Core/cuda/12.4.1-fasrc01/cuda
export TF_GPU_ALLOCATOR=cuda_malloc_async

# run code
export NUMBA_DISABLE_JIT=1
export HYDRA_FULL_ERROR=1
cd ~/latent-symmetry
python benchmark.py --num-hidden-layers 4 --hidden-dim 256 --seeds 1-100