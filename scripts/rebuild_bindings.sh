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

MPI_CXX="${MPICXX:-$(command -v mpicxx || command -v mpic++ || true)}"
if [[ -z "${MPI_CXX}" ]]; then
    echo "An MPI C++ compiler wrapper is required (mpicxx or mpic++)." >&2
    echo "Set MPICXX to select a non-default MPI installation." >&2
    exit 1
fi

if ! MPI_COMPILE_FLAGS="$("${MPI_CXX}" --showme:compile 2>/dev/null)" ||
   ! MPI_LINK_FLAGS="$("${MPI_CXX}" --showme:link 2>/dev/null)"; then
    echo "Could not query compile and link flags from ${MPI_CXX}." >&2
    echo "Set MPICXX to an Open MPI C++ compiler wrapper." >&2
    exit 1
fi

# Keep the selected host compiler (which may provide OpenMP for SLATE), while
# adding the MPI headers and libraries required by the ELPA binding.
export CPPFLAGS="${MPI_COMPILE_FLAGS}${CPPFLAGS:+ ${CPPFLAGS}}"
export LDFLAGS="${MPI_LINK_FLAGS}${LDFLAGS:+ ${LDFLAGS}}"
export ELPA_PREFIX SLATE_PREFIX
export ELPA_PROFILE="${PROFILE}"
export SLATE_PROFILE="${PROFILE}"
export SOAP_TP_BUILD_ELPA_BINDINGS=1
export SOAP_TP_BUILD_SLATE_BINDINGS=1

cd "${ROOT}"
"${PYTHON}" setup.py "${BUILD_ARGS[@]}"
