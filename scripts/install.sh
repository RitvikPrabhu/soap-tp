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
    if command -v cc >/dev/null 2>&1 && \
         command -v CC >/dev/null 2>&1 && \
         command -v ftn >/dev/null 2>&1; then
        export CC=cc CXX=CC FC=ftn
    elif command -v mpicc >/dev/null 2>&1 && \
         command -v mpicxx >/dev/null 2>&1 && \
         command -v mpifort >/dev/null 2>&1; then
        export CC=mpicc CXX=mpicxx FC=mpifort
    else
        echo "No complete MPI compiler wrapper triplet was found." >&2
        exit 1
    fi
elif command -v cc >/dev/null 2>&1 && \
     command -v CC >/dev/null 2>&1 && \
     command -v ftn >/dev/null 2>&1; then
    export CC=cc CXX=CC FC=ftn
elif command -v mpicc >/dev/null 2>&1 && \
     command -v mpicxx >/dev/null 2>&1 && \
     command -v mpifort >/dev/null 2>&1; then
    export CC=mpicc CXX=mpicxx FC=mpifort
else
    echo "No MPI compiler wrappers found." >&2
    echo "Load the cluster MPI module, or set CC, CXX, and FC explicitly." >&2
    exit 1
fi

# Python environments can carry older runtime libraries than the compiler
# module. Ask the active C++ wrapper for its own library search path so native
# modules and pybind extensions use one compatible ABI.
SOAP_TP_CXX_LIBRARY_PATH=""
if [[ "$(uname -s)" == "Linux" ]]; then
    compiler_library_path="$(
        "${CXX}" -print-search-dirs 2>/dev/null | \
            sed -n 's/^libraries: =//p' || true
    )"
    old_ifs="${IFS}"
    IFS=:
    for directory in ${compiler_library_path}; do
        directory="${directory#=}"
        if [[ -d "${directory}" ]]; then
            directory="$(cd "${directory}" && pwd -P)"
            case ":${SOAP_TP_CXX_LIBRARY_PATH}:" in
                *":${directory}:"*) ;;
                *) SOAP_TP_CXX_LIBRARY_PATH="${SOAP_TP_CXX_LIBRARY_PATH:+${SOAP_TP_CXX_LIBRARY_PATH}:}${directory}" ;;
            esac
        fi
    done
    IFS="${old_ifs}"
fi
if [[ -n "${SOAP_TP_CXX_LIBRARY_PATH}" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":${SOAP_TP_CXX_LIBRARY_PATH}:"*) ;;
        *) LD_LIBRARY_PATH="${SOAP_TP_CXX_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
    export LD_LIBRARY_PATH
fi
export SOAP_TP_CXX_LIBRARY_PATH

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

prefix_is_usable() {
    local package="$1"
    local prefix="$2"

    case "${package}" in
        elpa)
            compgen -G "${prefix}/include/elpa-*/elpa/elpa.h" >/dev/null && \
                { compgen -G "${prefix}/lib/libelpa.*" >/dev/null || \
                  compgen -G "${prefix}/lib64/libelpa.*" >/dev/null; }
            ;;
        slate)
            [[ -f "${prefix}/include/slate/slate.hh" ]] && \
                { compgen -G "${prefix}/lib/libslate.*" >/dev/null || \
                  compgen -G "${prefix}/lib64/libslate.*" >/dev/null; }
            ;;
    esac
}

discover_prefix() {
    local package="$1"
    local candidate
    local pkgconfig_prefix
    local IFS=:

    if command -v pkg-config >/dev/null 2>&1 && \
       pkgconfig_prefix="$(pkg-config --variable=prefix "${package}" 2>/dev/null)" && \
       [[ -n "${pkgconfig_prefix}" ]] && \
       prefix_is_usable "${package}" "${pkgconfig_prefix}"; then
        printf '%s\n' "${pkgconfig_prefix}"
        return
    fi

    for candidate in ${CMAKE_PREFIX_PATH:-}; do
        if prefix_is_usable "${package}" "${candidate}"; then
            printf '%s\n' "${candidate}"
            return
        fi
    done
}

export SOAP_TP_BUILD_ROOT="${SOAP_TP_BUILD_ROOT:-${ROOT}/build}"
if [[ "${SKIP_ELPA}" == "1" && -z "${ELPA_PREFIX:-}" ]]; then
    ELPA_PREFIX="$(discover_prefix elpa)"
fi
if [[ "${SKIP_SLATE}" == "1" && -z "${SLATE_PREFIX:-}" ]]; then
    SLATE_PREFIX="$(discover_prefix slate)"
fi
export ELPA_PREFIX="${ELPA_PREFIX:-${SOAP_TP_BUILD_ROOT}/elpa-install/${PROFILE}}"
export SLATE_PREFIX="${SLATE_PREFIX:-${SOAP_TP_BUILD_ROOT}/slate-install/${PROFILE}}"
export MATH_PREFIX="${MATH_PREFIX:-${SOAP_TP_BUILD_ROOT}/math-install}"
export ELPA_PROFILE="${PROFILE}"
export SLATE_PROFILE="${PROFILE}"

echo "Building profile: ${PROFILE}"
echo "MPI C compiler: ${CC}"
echo "MPI C++ compiler: ${CXX}"
echo "MPI Fortran compiler: ${FC}"
if [[ -n "${SOAP_TP_CXX_LIBRARY_PATH}" ]]; then
    echo "C++ library path: ${SOAP_TP_CXX_LIBRARY_PATH}"
fi
echo "ELPA prefix: ${ELPA_PREFIX}"
echo "SLATE prefix: ${SLATE_PREFIX}"

NATIVE_TARGETS=()
if [[ "${SKIP_ELPA}" == "1" ]]; then
    if ! prefix_is_usable elpa "${ELPA_PREFIX}"; then
        echo "--skip-elpa was requested, but no usable ELPA installation" >&2
        echo "was found in the active package search path." >&2
        exit 1
    fi
    echo "Using the existing ELPA installation."
else
    NATIVE_TARGETS+=(elpa)
fi

if [[ "${SKIP_SLATE}" == "1" ]]; then
    if ! prefix_is_usable slate "${SLATE_PREFIX}"; then
        echo "--skip-slate was requested, but no usable SLATE installation" >&2
        echo "was found in the active package search path." >&2
        exit 1
    fi
    echo "Using the existing SLATE installation."
else
    NATIVE_TARGETS+=(slate)
fi

if (( ${#NATIVE_TARGETS[@]} > 0 )); then
    "${ROOT}/scripts/build_native.sh" "${PROFILE}" "${NATIVE_TARGETS[@]}"
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

BUILD_CONFIG="${SOAP_TP_BUILD_CONFIG:-${SOAP_TP_BUILD_ROOT}/bindings.env}"
BUILD_CONFIG_TEMP="${BUILD_CONFIG}.tmp.$$"
mkdir -p "$(dirname "${BUILD_CONFIG}")"
{
    printf 'export SOAP_TP_PROFILE=%q\n' "${PROFILE}"
    printf 'export SOAP_TP_PYTHON=%q\n' "${PYTHON}"
    printf 'export ELPA_PREFIX=%q\n' "${ELPA_PREFIX}"
    printf 'export SLATE_PREFIX=%q\n' "${SLATE_PREFIX}"
    printf 'export CC=%q\n' "${CC}"
    printf 'export CXX=%q\n' "${CXX}"
    printf 'export FC=%q\n' "${FC}"
    printf 'export SOAP_TP_CXX_LIBRARY_PATH=%q\n' "${SOAP_TP_CXX_LIBRARY_PATH}"
    if [[ -n "${SOAP_TP_CXX_LIBRARY_PATH}" ]]; then
        printf '%s\n' 'case ":${LD_LIBRARY_PATH:-}:" in'
        printf '%s\n' '    *":${SOAP_TP_CXX_LIBRARY_PATH}:"*) ;;'
        printf '%s\n' '    *) export LD_LIBRARY_PATH="${SOAP_TP_CXX_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;'
        printf '%s\n' 'esac'
    fi
} >"${BUILD_CONFIG_TEMP}"
mv "${BUILD_CONFIG_TEMP}" "${BUILD_CONFIG}"
echo "Binding build configuration: ${BUILD_CONFIG}"
