#!/bin/bash
#SBATCH -A gen150
#SBATCH -J soap-accuracy
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH -t 02:00:00
#SBATCH --qos=debug
#SBATCH -o soap-accuracy-%j.out
#SBATCH -e soap-accuracy-%j.err

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
export SOAP_TP_TEST_MPI_RANKS=8

REPO_ROOT="${SOAP_TP_ROOT:-${SLURM_SUBMIT_DIR}}"
cd "${REPO_ROOT}"
source .venv/bin/activate
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

OUTPUT_DIR="${REPO_ROOT}/output"
LOG_FILE="${OUTPUT_DIR}/soap-accuracy-${SLURM_JOB_ID}.log"
PLOT_FILE="${OUTPUT_DIR}/soap-accuracy-${SLURM_JOB_ID}.png"
mkdir -p "${OUTPUT_DIR}"

set +e
python -m pytest -s tests/unit/test_soap.py > "${LOG_FILE}" 2>&1
TEST_STATUS=$?
python scripts/plot_soap_accuracy.py "${LOG_FILE}" "${PLOT_FILE}"
PLOT_STATUS=$?
set -e

echo "SOAP log: ${LOG_FILE}"
echo "Accuracy plot: ${PLOT_FILE}"

if (( PLOT_STATUS != 0 )); then
    exit "${PLOT_STATUS}"
fi
exit "${TEST_STATUS}"
