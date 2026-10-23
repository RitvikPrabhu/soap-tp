#!/bin/bash
# Coordinator submits only when no other allocation is running.
#SBATCH -A gen150
#SBATCH -J torch-eigh-control
#SBATCH -p batch
#SBATCH --qos=debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=7
#SBATCH -t 00:10:00
#SBATCH -o output/custom-eigh/torch-benchmark-%j.out
#SBATCH -e output/custom-eigh/torch-benchmark-%j.err
set -euo pipefail
module reset
module load PrgEnv-gnu cpe/26.03 rocm/7.0.2 craype-accel-amd-gfx90a
export MPICH_GPU_SUPPORT_ENABLED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$PWD/build/elpa-install/rocm/lib:$PWD/build/math-install/lib:${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
srun -N1 -n1 --ntasks-per-node=1 -c7 --gpus-per-task=1 \
    --gpu-bind=closest --kill-on-bad-exit=1 \
    .venv/bin/python research/frontier_eigh/torch_benchmark.py --device rocm "$@"
