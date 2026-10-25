#!/bin/bash
#SBATCH -A gen150
#SBATCH -J soap-tp-scaling
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=7
#SBATCH -t 02:00:00

set -euo pipefail

if (( $# != 7 )); then
    echo "Usage: sbatch $0 MATRIX_SIZE RANKS BLOCK_SIZE STEPS SHARD_DIM BASIS REFRESH_INTERVAL" >&2
    exit 2
fi

MATRIX_SIZE="$1"
RANKS="$2"
BLOCK_SIZE="$3"
STEPS="$4"
SHARD_DIM="$5"
BASIS_IMPLEMENTATION="$6"
BASIS_REFRESH_INTERVAL="$7"

if [[ "${SLURM_NTASKS:-}" != "${RANKS}" ]]; then
    echo "Expected ${RANKS} Slurm tasks, got ${SLURM_NTASKS:-unset}" >&2
    exit 2
fi

module reset
module load PrgEnv-gnu
module load cpe/26.03
module load rocm/7.0.2
module load craype-accel-amd-gfx90a
module load slate/2025.05.28-gpu-mpi-omp

# The submit wrapper sets --chdir to the repository. Slurm executes a copied
# script from /var/spool, so BASH_SOURCE cannot be used to find the repository.
REPO_ROOT="${SOAP_TP_ROOT:-${PWD}}"
if [[ ! -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    echo "SOAP-TP repository not found at ${REPO_ROOT}" >&2
    exit 2
fi

export LD_LIBRARY_PATH="${REPO_ROOT}/build/elpa-install/rocm/lib:${REPO_ROOT}/build/math-install/lib:${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export MPICH_GPU_SUPPORT_ENABLED=1
export SLATE_GPU_AWARE_MPI=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# SOAP_PROFILE serializes complete tensors. The timing runner below is the
# profiler for scaling runs and keeps that expensive diagnostic path disabled.
export SOAP_PROFILE=0
unset SOAP_PROFILE_STEPS

cd "${REPO_ROOT}"
source .venv/bin/activate

echo "SOAP-TP scaling: N=${MATRIX_SIZE}, ranks/GPUs=${RANKS}, block=${BLOCK_SIZE}, steps=${STEPS}, shard_dim=${SHARD_DIM}, basis=${BASIS_IMPLEMENTATION}"

srun \
    --nodes=1 \
    --ntasks="${RANKS}" \
    --ntasks-per-node="${RANKS}" \
    --cpus-per-task=7 \
    --gpus-per-task=1 \
    --gpu-bind=closest \
    --kill-on-bad-exit=1 \
    python -u scripts/run_soap_tp_timing_gpu.py \
        --size "${MATRIX_SIZE}" \
        --world-size "${RANKS}" \
        --block-size "${BLOCK_SIZE}" \
        --steps "${STEPS}" \
        --shard-dim "${SHARD_DIM}" \
        --basis-implementation "${BASIS_IMPLEMENTATION}" \
        --basis-refresh-interval "${BASIS_REFRESH_INTERVAL}"
