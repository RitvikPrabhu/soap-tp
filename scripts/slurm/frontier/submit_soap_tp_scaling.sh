#!/bin/bash
# Submit one SOAP-TP strong-scaling job for every requested rank/GPU count.
#
# Usage:
#   bash scripts/slurm/frontier/submit_soap_tp_scaling.sh MATRIX_SIZE RANKS...
#
# Example:
#   bash scripts/slurm/frontier/submit_soap_tp_scaling.sh 8192 1 2 4 8

set -euo pipefail

if (( $# < 2 )); then
    echo "Usage: $0 MATRIX_SIZE RANKS..." >&2
    echo "Example: $0 8192 1 2 4 8" >&2
    exit 2
fi

MATRIX_SIZE="$1"
shift
RANK_COUNTS=("$@")

if [[ ! "${MATRIX_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "MATRIX_SIZE must be a positive integer" >&2
    exit 2
fi

declare -A SEEN_RANKS=()
MAX_GRID_AXIS=1
for RANKS in "${RANK_COUNTS[@]}"; do
    if [[ ! "${RANKS}" =~ ^[1-9][0-9]*$ ]] || (( RANKS > 8 )); then
        echo "Each rank count must be an integer from 1 through 8" >&2
        exit 2
    fi
    if [[ -n "${SEEN_RANKS[${RANKS}]:-}" ]]; then
        echo "Duplicate rank count: ${RANKS}" >&2
        exit 2
    fi
    SEEN_RANKS[${RANKS}]=1
    if (( MATRIX_SIZE % RANKS != 0 )); then
        echo "Matrix size ${MATRIX_SIZE} must be divisible by rank count ${RANKS}" >&2
        exit 2
    fi

    GRID_ROWS=1
    for (( CANDIDATE = 1; CANDIDATE * CANDIDATE <= RANKS; CANDIDATE++ )); do
        if (( RANKS % CANDIDATE == 0 )); then
            GRID_ROWS="${CANDIDATE}"
        fi
    done
    GRID_COLUMNS=$(( RANKS / GRID_ROWS ))
    if (( GRID_COLUMNS > MAX_GRID_AXIS )); then
        MAX_GRID_AXIS="${GRID_COLUMNS}"
    fi
done

STEPS="${SOAP_SCALING_STEPS:-10}"
SHARD_DIM="${SOAP_SCALING_SHARD_DIM:-0}"
BASIS_IMPLEMENTATION="${SOAP_SCALING_BASIS_IMPLEMENTATION:-elpa}"
BASIS_REFRESH_INTERVAL="${SOAP_SCALING_BASIS_REFRESH_INTERVAL:-10}"
MAX_BLOCK_SIZE="${SOAP_SCALING_MAX_BLOCK_SIZE:-512}"

for VALUE_NAME in STEPS BASIS_REFRESH_INTERVAL MAX_BLOCK_SIZE; do
    VALUE="${!VALUE_NAME}"
    if [[ ! "${VALUE}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${VALUE_NAME} must be a positive integer" >&2
        exit 2
    fi
done
if [[ "${SHARD_DIM}" != 0 && "${SHARD_DIM}" != 1 ]]; then
    echo "SOAP_SCALING_SHARD_DIM must be 0 or 1" >&2
    exit 2
fi
if [[ "${BASIS_IMPLEMENTATION}" != elpa && "${BASIS_IMPLEMENTATION}" != eigh ]]; then
    echo "SOAP_SCALING_BASIS_IMPLEMENTATION must be elpa or eigh" >&2
    exit 2
fi

BLOCK_SIZE=$(( MATRIX_SIZE / MAX_GRID_AXIS ))
if (( BLOCK_SIZE > MAX_BLOCK_SIZE )); then
    BLOCK_SIZE="${MAX_BLOCK_SIZE}"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
WORKER="${SCRIPT_DIR}/soap_tp_scaling.slurm"
OUTPUT_ROOT="${SOAP_SCALING_OUTPUT_ROOT:-${REPO_ROOT}/output/strong-scaling}"
if [[ "${OUTPUT_ROOT}" != /* ]]; then
    OUTPUT_ROOT="${REPO_ROOT}/${OUTPUT_ROOT}"
fi
RUN_DIR="${OUTPUT_ROOT}/n${MATRIX_SIZE}-$(date +%Y%m%d-%H%M%S)"
mkdir -p "${OUTPUT_ROOT}"
mkdir "${RUN_DIR}"
GIT_COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"

printf '%s\n' \
    "submitted_at=$(date --iso-8601=seconds)" \
    "repository=${REPO_ROOT}" \
    "git_commit=${GIT_COMMIT}" \
    "matrix_size=${MATRIX_SIZE}" \
    "ranks=${RANK_COUNTS[*]}" \
    "block_size=${BLOCK_SIZE}" \
    "steps=${STEPS}" \
    "shard_dim=${SHARD_DIM}" \
    "basis_implementation=${BASIS_IMPLEMENTATION}" \
    "basis_refresh_interval=${BASIS_REFRESH_INTERVAL}" \
    > "${RUN_DIR}/submission.txt"

printf 'ranks\tjob_id\tstdout\tstderr\n' > "${RUN_DIR}/manifest.tsv"

for RANKS in "${RANK_COUNTS[@]}"; do
    JOB_NAME="soap-n${MATRIX_SIZE}-r${RANKS}"
    STDOUT="${RUN_DIR}/${JOB_NAME}-j%j.out"
    STDERR="${RUN_DIR}/${JOB_NAME}-j%j.err"
    JOB_REPLY="$(
        sbatch \
            --parsable \
            --chdir="${REPO_ROOT}" \
            --nodes=1 \
            --ntasks="${RANKS}" \
            --ntasks-per-node="${RANKS}" \
            --cpus-per-task=7 \
            --job-name="${JOB_NAME}" \
            --output="${STDOUT}" \
            --error="${STDERR}" \
            "${WORKER}" \
            "${MATRIX_SIZE}" \
            "${RANKS}" \
            "${BLOCK_SIZE}" \
            "${STEPS}" \
            "${SHARD_DIM}" \
            "${BASIS_IMPLEMENTATION}" \
            "${BASIS_REFRESH_INTERVAL}"
    )"
    JOB_ID="${JOB_REPLY%%;*}"
    FINAL_STDOUT="${STDOUT//%j/${JOB_ID}}"
    FINAL_STDERR="${STDERR//%j/${JOB_ID}}"
    printf '%s\t%s\t%s\t%s\n' \
        "${RANKS}" "${JOB_ID}" "${FINAL_STDOUT}" "${FINAL_STDERR}" \
        >> "${RUN_DIR}/manifest.tsv"
    echo "Submitted ${JOB_NAME} as job ${JOB_ID}"
done

echo "All logs and submission metadata: ${RUN_DIR}"
