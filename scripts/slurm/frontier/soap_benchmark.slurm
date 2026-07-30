#!/bin/bash
#
# Tune the sizes on Frontier; strong-scaling sizes must be multiples of 840.
# Example:
# sbatch -A PROJECT -N 1 --ntasks-per-node=8 -c 7 -t 02:00:00 \
#   scripts/slurm/frontier/soap_benchmark.slurm \
#   --sizes 3360,6720 --weak-base-size 3360 \
#   --output-dir soap_plots --log-file plot_data.csv

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${SOAP_BENCHMARK_PYTHON:-${ROOT}/.venv/bin/python}"

export MPICH_GPU_SUPPORT_ENABLED="${MPICH_GPU_SUPPORT_ENABLED:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"
exec "${PYTHON}" -u scripts/frontier_soap_plots.py "$@"
