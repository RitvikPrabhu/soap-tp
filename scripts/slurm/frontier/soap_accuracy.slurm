#!/bin/bash
#SBATCH -A gen150
#SBATCH -J soap-accuracy
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -t 02:00:00
#SBATCH -o soap-accuracy-%j.out
#SBATCH -e soap-accuracy-%j.err

set -euo pipefail

module reset
module load PrgEnv-gnu
module load cpe/26.03
module load rocm/7.0.2
module load craype-accel-amd-gfx90a
module load slate/2025.05.28-gpu-mpi-omp

REPO_ROOT="${SOAP_TP_ROOT:-${SLURM_SUBMIT_DIR}}"
export LD_LIBRARY_PATH="${REPO_ROOT}/build/elpa-install/rocm/lib:${REPO_ROOT}/build/math-install/lib:${CRAY_LD_LIBRARY_PATH:-}:${LD_LIBRARY_PATH:-}"
export MPICH_GPU_SUPPORT_ENABLED=1
export SLATE_GPU_AWARE_MPI=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export SOAP_TP_TEST_MPI_RANKS=8

cd "${REPO_ROOT}"
source .venv/bin/activate
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

OUTPUT_DIR="${REPO_ROOT}/output"
case "${1:-}" in
    0)
        SHARD_DIM=0
        SHARD_LABEL=row
        ;;
    1)
        SHARD_DIM=1
        SHARD_LABEL=col
        ;;
    *)
        echo "Usage: sbatch $0 {0|1}" >&2
        exit 2
        ;;
esac
LOG_FILE="${OUTPUT_DIR}/soap-accuracy-${SLURM_JOB_ID}-${SHARD_LABEL}.log"
PLOT_FILE="${OUTPUT_DIR}/soap-accuracy-${SLURM_JOB_ID}-${SHARD_LABEL}.png"
mkdir -p "${OUTPUT_DIR}"

set +e
srun \
    --nodes=1 \
    --ntasks=8 \
    --ntasks-per-node=8 \
    --cpus-per-task=7 \
    --gpus-per-task=1 \
    --gpu-bind=closest \
    --kill-on-bad-exit=1 \
    python scripts/run_soap_accuracy_gpu.py \
        --sizes 8 16 32 64 128 256 512 1024 2048 4096 8192 \
        --max-block-size 512 \
        --rtol 1e-5 \
        --shard-dim "${SHARD_DIM}" \
        --basis-implementations eigh elpa \
        --device cuda:0 > "${LOG_FILE}" 2>&1
TEST_STATUS=$?
python scripts/plot_soap_accuracy.py \
    --shard-dim "${SHARD_DIM}" "${LOG_FILE}" "${PLOT_FILE}"
PLOT_STATUS=$?
set -e

echo "Shard: ${SHARD_LABEL} (dim ${SHARD_DIM})"
echo "SOAP log: ${LOG_FILE}"
echo "Accuracy plot: ${PLOT_FILE}"

if (( PLOT_STATUS != 0 )); then
    exit "${PLOT_STATUS}"
fi
exit "${TEST_STATUS}"
