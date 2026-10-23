#!/bin/bash
# Submit only when the node coordinator approves; tiny correctness, not timing.
#SBATCH -A gen150
#SBATCH -J eigh-torch
#SBATCH -p batch
#SBATCH --qos=debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 00:15:00
#SBATCH -o output/custom-eigh/torch-loop2/job-%j.out
#SBATCH -e output/custom-eigh/torch-loop2/job-%j.err
set -euo pipefail
module reset
module load PrgEnv-gnu cpe/26.03 rocm/7.0.2 craype-accel-amd-gfx90a
export MPICH_GPU_SUPPORT_ENABLED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$PWD/build/elpa-install/rocm/lib:$PWD/build/math-install/lib:${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
destination="output/custom-eigh/torch-loop2/batch-$SLURM_JOB_ID"
mkdir -p "$destination"
# One Python process on one GCD runs the torch reference and serially launches
# eight-rank custom solves. No concurrent jobs or nested srun steps.
.venv/bin/python research/frontier_eigh/torch_compare.py \
    --device rocm --run-custom --n 97 --seeds 42 43 44 --pr 2 --pc 4 \
    --binary "${EIGH_BIN:-build/custom-eigh/pool-eigh}" \
    --launcher 'srun -N1 -n8 --ntasks-per-node=8 -c7 --gpus-per-task=1 --gpu-bind=closest --kill-on-bad-exit=1' \
    --output "$destination" "$@"
