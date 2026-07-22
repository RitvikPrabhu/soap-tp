#!/usr/bin/env bash

# Rebuild only the two pybind extensions against selected ELPA and SLATE
# installations. Native libraries are never built by this script.
set -eo pipefail

usage() {
    echo "Usage: scripts/rebuild_bindings.sh <cpu|cuda|rocm> [--force]"
}

PROFILE="${1:-}"
case "${PROFILE}" in
    cpu|cuda|rocm) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
shift

BUILD_ARGS=(build_ext --inplace)
case "${1:-}" in
    "") ;;
    --force) BUILD_ARGS+=(--force) ;;
    *) usage >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${SOAP_TP_BUILD_ROOT:-${ROOT}/build}"
PYTHON="${PYTHON:-python3}"
ELPA_PREFIX="${ELPA_PREFIX:-${BUILD_ROOT}/elpa-install/${PROFILE}}"
SLATE_PREFIX="${SLATE_PREFIX:-${BUILD_ROOT}/slate-install/${PROFILE}}"

export ELPA_PREFIX SLATE_PREFIX
export ELPA_PROFILE="${PROFILE}"
export SLATE_PROFILE="${PROFILE}"
export SOAP_TP_BUILD_ELPA_BINDINGS=1
export SOAP_TP_BUILD_SLATE_BINDINGS=1

cd "${ROOT}"
"${PYTHON}" setup.py "${BUILD_ARGS[@]}"
