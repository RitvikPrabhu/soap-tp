#!/usr/bin/env bash

# macOS still ships Bash 3.2, where `set -u` treats an empty array expansion as
# an unset variable. Keep this entry point compatible with both macOS and the
# newer Bash versions normally found on clusters.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-cpu}"

usage() {
    cat <<'EOF'
Usage: ./scripts/install.sh [cpu|cuda|rocm] [pip install options]

Examples:
  ./scripts/install.sh cpu
  ./scripts/install.sh cuda
  ./scripts/install.sh rocm --editable
  ./scripts/install.sh cpu --skip-elpa
  ./scripts/install.sh cpu --skip-slate
  PYTHON=$HOME/venvs/soap/bin/python ./scripts/install.sh cuda

Before running on a cluster, load its compiler, MPI, and (for GPU builds)
CUDA or ROCm modules. Useful overrides:

  BUILD_JOBS=16                 parallel native build jobs
  ELPA_PREFIX=/stable/path      where this script installs and finds ELPA
  SLATE_PREFIX=/stable/path     where this script installs SLATE
  MATH_PREFIX=/stable/path      where fallback OpenBLAS/ScaLAPACK are installed
  SCALAPACK_LDFLAGS='...'       use a cluster-provided ScaLAPACK
  ELPA_CONFIGURE_ARGS='...'     extra ELPA configure flags
  SLATE_CMAKE_ARGS='...'        extra SLATE CMake options
  CC=... CXX=... FC=...         explicit MPI-aware compiler wrappers
EOF
}

if [[ "${PROFILE}" == "-h" || "${PROFILE}" == "--help" ]]; then
    usage
    exit 0
fi

case "${PROFILE}" in
    cpu|cuda|rocm) ;;
    *) usage >&2; exit 2 ;;
esac
shift $(( $# > 0 ? 1 : 0 ))

EDITABLE=0
SKIP_ELPA=0
SKIP_SLATE=0
PIP_ARGS=()
for argument in "$@"; do
    case "${argument}" in
        --editable) EDITABLE=1 ;;
        --skip-elpa) SKIP_ELPA=1 ;;
        --skip-slate) SKIP_SLATE=1 ;;
        *) PIP_ARGS+=("${argument}") ;;
    esac
done

PYTHON="${PYTHON:-python3}"
if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON}" >&2
    exit 1
fi
export PYTHON
"${PYTHON}" -m pip --version >/dev/null
if ! "${PYTHON}" -c "import torch" >/dev/null 2>&1; then
    echo "PyTorch is not installed in ${PYTHON}'s environment." >&2
    echo "Activate/install the cluster's CPU, CUDA, or ROCm PyTorch build first." >&2
    exit 1
fi

if [[ "${PROFILE}" == "cuda" ]] && \
   ! "${PYTHON}" -c "import torch, sys; sys.exit(torch.version.cuda is None)"; then
    echo "The selected Python environment does not contain a CUDA PyTorch build." >&2
    exit 1
fi

if [[ "${PROFILE}" == "rocm" ]] && \
   ! "${PYTHON}" -c "import torch, sys; sys.exit(torch.version.hip is None)"; then
    echo "The selected Python environment does not contain a ROCm PyTorch build." >&2
    exit 1
fi

# Respect a complete user-provided compiler triplet. Otherwise select common
# MPI wrapper names used by workstation MPI installations and Cray systems.
if [[ -n "${CC:-}" && -n "${CXX:-}" && -n "${FC:-}" ]]; then
    export CC CXX FC
elif [[ -n "${CC:-}" || -n "${CXX:-}" || -n "${FC:-}" ]]; then
    echo "Ignoring an incomplete CC/CXX/FC override and selecting MPI wrappers."
    if command -v mpicc >/dev/null 2>&1 && \
       command -v mpicxx >/dev/null 2>&1 && \
       command -v mpifort >/dev/null 2>&1; then
        export CC=mpicc CXX=mpicxx FC=mpifort
    elif command -v cc >/dev/null 2>&1 && \
         command -v CC >/dev/null 2>&1 && \
         command -v ftn >/dev/null 2>&1; then
        export CC=cc CXX=CC FC=ftn
    else
        echo "No complete MPI compiler wrapper triplet was found." >&2
        exit 1
    fi
elif command -v mpicc >/dev/null 2>&1 && \
     command -v mpicxx >/dev/null 2>&1 && \
     command -v mpifort >/dev/null 2>&1; then
    export CC=mpicc CXX=mpicxx FC=mpifort
elif command -v cc >/dev/null 2>&1 && \
     command -v CC >/dev/null 2>&1 && \
     command -v ftn >/dev/null 2>&1; then
    export CC=cc CXX=CC FC=ftn
else
    echo "No MPI compiler wrappers found." >&2
    echo "Load the cluster MPI module, or set CC, CXX, and FC explicitly." >&2
    exit 1
fi

if [[ "${PROFILE}" == "cuda" ]] && \
   ! command -v nvcc >/dev/null 2>&1 && \
   [[ ! -x "${CUDA_HOME:-/nonexistent}/bin/nvcc" ]]; then
    echo "CUDA profile requested, but nvcc was not found." >&2
    echo "Load the cluster CUDA module or set CUDA_HOME." >&2
    exit 1
fi

if [[ "${PROFILE}" == "rocm" ]] && \
   ! command -v hipcc >/dev/null 2>&1 && \
   [[ ! -x "${ROCM_PATH:-/nonexistent}/bin/hipcc" ]]; then
    echo "ROCm profile requested, but hipcc was not found." >&2
    echo "Load the cluster ROCm module or set ROCM_PATH." >&2
    exit 1
fi

export SOAP_TP_BUILD_ROOT="${SOAP_TP_BUILD_ROOT:-${ROOT}/build}"
export ELPA_PREFIX="${ELPA_PREFIX:-${SOAP_TP_BUILD_ROOT}/elpa-install/${PROFILE}}"
export SLATE_PREFIX="${SLATE_PREFIX:-${SOAP_TP_BUILD_ROOT}/slate-install/${PROFILE}}"
export MATH_PREFIX="${MATH_PREFIX:-${SOAP_TP_BUILD_ROOT}/math-install}"
export ELPA_PROFILE="${PROFILE}"
export SLATE_PROFILE="${PROFILE}"

echo "Building profile: ${PROFILE}"
echo "MPI C compiler: ${CC}"
echo "MPI C++ compiler: ${CXX}"
echo "MPI Fortran compiler: ${FC}"
echo "ELPA prefix: ${ELPA_PREFIX}"
echo "SLATE prefix: ${SLATE_PREFIX}"

if [[ "${SKIP_ELPA}" == "1" ]]; then
    if [[ ! -d "${ELPA_PREFIX}/include" ]]; then
        echo "--skip-elpa was requested, but ${ELPA_PREFIX} is not installed." >&2
        exit 1
    fi
    echo "Using the existing ELPA installation."
else
    "${ROOT}/scripts/build_elpa.sh" "${PROFILE}"
fi

if [[ "${SKIP_SLATE}" == "1" ]]; then
    if [[ ! -f "${SLATE_PREFIX}/include/slate/slate.hh" ]] || \
       { ! compgen -G "${SLATE_PREFIX}/lib/libslate.*" >/dev/null && \
         ! compgen -G "${SLATE_PREFIX}/lib64/libslate.*" >/dev/null; }; then
        echo "--skip-slate was requested, but ${SLATE_PREFIX} is not installed." >&2
        exit 1
    fi
    echo "Using the existing SLATE installation."
else
    "${ROOT}/scripts/build_slate.sh" "${PROFILE}"
fi

export SOAP_TP_BUILD_ELPA_BINDINGS=1
export SOAP_TP_BUILD_SLATE_BINDINGS=1
if [[ "${EDITABLE}" == "1" ]]; then
    "${PYTHON}" -m pip install --no-build-isolation --editable "${ROOT}" "${PIP_ARGS[@]}"
else
    "${PYTHON}" -m pip install --no-build-isolation "${ROOT}" "${PIP_ARGS[@]}"
fi

EXPECTED_BACKEND=none
case "${PROFILE}" in
    cuda) EXPECTED_BACKEND=cuda ;;
    rocm) EXPECTED_BACKEND=rocm ;;
esac
export EXPECTED_BACKEND

"${PYTHON}" -c '
import os
from soap_tp import elpa_bindings, slate_bindings

expected = os.environ["EXPECTED_BACKEND"]
backends = {
    "ELPA": elpa_bindings.compiled_gpu_backend(),
    "SLATE": slate_bindings.compiled_gpu_backend(),
}
for library, actual in backends.items():
    if actual != expected:
        raise SystemExit(
            f"{library} binding reports {actual!r}, expected {expected!r}"
        )
elpa_backend = backends["ELPA"]
slate_backend = backends["SLATE"]
print(
    "soap-tp installed successfully "
    f"(ELPA backend: {elpa_backend}, SLATE backend: {slate_backend})"
)
'

echo "Keep ${ELPA_PREFIX} available: the installed extension loads ELPA from there."
echo "Keep ${SLATE_PREFIX} available: the installed extension loads SLATE from there."
