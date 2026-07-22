#!/usr/bin/env bash

# Build the pinned ELPA and SLATE submodules. The caller supplies the compiler,
# MPI, math, and GPU environment; this script does not load or discover them.
set -eo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/build_native.sh <cpu|cuda|rocm> [--skip-elpa] [--skip-slate]

Builds ELPA and SLATE into build/. Use a skip flag when that library is already
available and set ELPA_PREFIX or SLATE_PREFIX to its installation prefix.

Environment:
  BUILD_JOBS           Parallel jobs (default: 8)
  SOAP_TP_BUILD_ROOT   Build directory (default: <repo>/build)
  ELPA_PREFIX          ELPA install prefix
  SLATE_PREFIX         SLATE install prefix
  ELPA_CONFIGURE_ARGS  Additional ELPA configure arguments
  SLATE_CMAKE_ARGS     Additional SLATE CMake arguments
EOF
}

PROFILE="${1:-}"
case "${PROFILE}" in
    cpu|cuda|rocm) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
shift

BUILD_ELPA=1
BUILD_SLATE=1
while (( $# > 0 )); do
    case "$1" in
        --skip-elpa) BUILD_ELPA=0 ;;
        --skip-slate) BUILD_SLATE=0 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
    shift
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ROOT="${SOAP_TP_BUILD_ROOT:-${ROOT}/build}"
ELPA_PREFIX="${ELPA_PREFIX:-${BUILD_ROOT}/elpa-install/${PROFILE}}"
SLATE_PREFIX="${SLATE_PREFIX:-${BUILD_ROOT}/slate-install/${PROFILE}}"
JOBS="${BUILD_JOBS:-8}"
export ELPA_PREFIX SLATE_PREFIX

case "${PROFILE}" in
    cpu)
        ELPA_GPU_ARGS=()
        SLATE_GPU_BACKEND=none
        ;;
    cuda)
        ELPA_GPU_ARGS=(
            --enable-nvidia-gpu-kernels
            --enable-gpu-streams=nvidia
        )
        SLATE_GPU_BACKEND=cuda
        ;;
    rocm)
        ELPA_GPU_ARGS=(
            --enable-amd-gpu-kernels
            --enable-gpu-streams=amd
            --with-rocsolver=yes
        )
        SLATE_GPU_BACKEND=hip
        ;;
esac

read -r -a EXTRA_ELPA_ARGS <<<"${ELPA_CONFIGURE_ARGS:-}"
read -r -a EXTRA_SLATE_ARGS <<<"${SLATE_CMAKE_ARGS:-}"

if [[ "${BUILD_ELPA}" == "1" ]]; then
    ELPA_SOURCE="${ROOT}/third_party/elpa"
    ELPA_BUILD="${BUILD_ROOT}/elpa/${PROFILE}"

    git -C "${ROOT}" submodule update --init third_party/elpa
    (cd "${ELPA_SOURCE}" && ./autogen.sh)
    mkdir -p "${ELPA_BUILD}" "${ELPA_PREFIX}"
    (
        cd "${ELPA_BUILD}"
        "${ELPA_SOURCE}/configure" \
            --prefix="${ELPA_PREFIX}" \
            --with-mpi=yes \
            --with-test-programs=no \
            "${ELPA_GPU_ARGS[@]}" \
            "${EXTRA_ELPA_ARGS[@]}"
    )
    make -C "${ELPA_BUILD}" -j"${JOBS}" install
fi

if [[ "${BUILD_SLATE}" == "1" ]]; then
    SLATE_SOURCE="${ROOT}/third_party/slate"
    SLATE_BUILD="${BUILD_ROOT}/slate/${PROFILE}"

    git -C "${ROOT}" submodule update --init --recursive third_party/slate
    cmake -S "${SLATE_SOURCE}" -B "${SLATE_BUILD}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${SLATE_PREFIX}" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTING=OFF \
        -Dbuild_tests=OFF \
        -Dgpu_backend="${SLATE_GPU_BACKEND}" \
        "${EXTRA_SLATE_ARGS[@]}"
    cmake --build "${SLATE_BUILD}" --parallel "${JOBS}"
    cmake --install "${SLATE_BUILD}"
fi

printf 'ELPA_PREFIX=%s\nSLATE_PREFIX=%s\n' "${ELPA_PREFIX}" "${SLATE_PREFIX}"
