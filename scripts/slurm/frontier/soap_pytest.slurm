#!/bin/bash
#SBATCH -A gen150
#SBATCH -J soap-test
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -t 02:00:00
#SBATCH -o output/test-soap-%j.out
#SBATCH -e output/test-soap-%j.err
#SBATCH --qos=debug

set -euo pipefail

module reset
module load PrgEnv-gnu
module load cpe/26.03
module load rocm/7.0.2
module load slate/2025.05.28-gpu-mpi-omp
module load netlib-scalapack/2.2.2-mpi

export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

REPO_ROOT="${SOAP_TP_ROOT:-${SLURM_SUBMIT_DIR}}"
cd "${REPO_ROOT}"
source .venv/bin/activate && source build/bindings.env
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

srun -N 1 -n 8 --ntasks-per-node=8 -c 7 --gpus-per-task=1 --gpu-bind=closest python -u tests/unit/test_soap_gpu.py