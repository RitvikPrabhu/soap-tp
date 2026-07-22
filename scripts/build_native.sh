#!/usr/bin/env bash

# Build the pinned math, ELPA, and SLATE submodules. The caller supplies the
# compiler, MPI, and GPU environment; this script does not load or discover it.
set -eo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/build_native.sh <cpu|cuda|rocm> [--skip-elpa] [--skip-slate]

Builds ELPA and SLATE into build/. Use a skip flag when that library is already
available and set ELPA_PREFIX or SLATE_PREFIX to its installation prefix.
OpenBLAS and ScaLAPACK are built from the pinned submodules when either native
library is selected.

Environment:
  BUILD_JOBS           Parallel jobs (default: 8)
  ELPA_BUILD_JOBS      ELPA build jobs (default: 1)
  SOAP_TP_BUILD_ROOT   Build directory (default: <repo>/build)
  MATH_PREFIX          OpenBLAS/ScaLAPACK install prefix
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
MATH_PREFIX="${MATH_PREFIX:-${BUILD_ROOT}/math-install}"
ELPA_PREFIX="${ELPA_PREFIX:-${BUILD_ROOT}/elpa-install/${PROFILE}}"
SLATE_PREFIX="${SLATE_PREFIX:-${BUILD_ROOT}/slate-install/${PROFILE}}"
JOBS="${BUILD_JOBS:-8}"
ELPA_JOBS="${ELPA_BUILD_JOBS:-1}"
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

if [[ "${BUILD_ELPA}" == "1" || "${BUILD_SLATE}" == "1" ]]; then
    OPENBLAS_SOURCE="${ROOT}/third_party/openblas"
    SCALAPACK_SOURCE="${ROOT}/third_party/scalapack"
    OPENBLAS_BUILD="${BUILD_ROOT}/math/openblas"
    SCALAPACK_BUILD="${BUILD_ROOT}/math/scalapack"

    cmake -S "${OPENBLAS_SOURCE}" -B "${OPENBLAS_BUILD}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${MATH_PREFIX}" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_STATIC_LIBS=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_WITHOUT_LAPACKE=ON \
        -DBUILD_TESTING=OFF
    cmake --build "${OPENBLAS_BUILD}" \
        --target openblas_static openblas_shared --parallel "${JOBS}"
    cmake --install "${OPENBLAS_BUILD}"

    cmake -S "${SCALAPACK_SOURCE}" -B "${SCALAPACK_BUILD}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${MATH_PREFIX}" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DSCALAPACK_BUILD_TESTS=OFF \
        -DBLAS_LIBRARIES="${MATH_PREFIX}/lib/libopenblas.a" \
        -DLAPACK_LIBRARIES="${MATH_PREFIX}/lib/libopenblas.a"
    cmake --build "${SCALAPACK_BUILD}" --parallel "${JOBS}"
    cmake --install "${SCALAPACK_BUILD}"
fi

if [[ "${BUILD_ELPA}" == "1" ]]; then
    ELPA_SOURCE="${ROOT}/third_party/elpa"
    ELPA_BUILD="${BUILD_ROOT}/elpa/${PROFILE}"

    (cd "${ELPA_SOURCE}" && ./autogen.sh)
    mkdir -p "${ELPA_BUILD}" "${ELPA_PREFIX}"
    (
        cd "${ELPA_BUILD}"
        SCALAPACK_LDFLAGS="-L${MATH_PREFIX}/lib -lscalapack ${MATH_PREFIX}/lib/libopenblas.a" \
        "${ELPA_SOURCE}/configure" \
            --prefix="${ELPA_PREFIX}" \
            --with-mpi=yes \
            --with-test-programs=no \
            "${ELPA_GPU_ARGS[@]}" \
            "${EXTRA_ELPA_ARGS[@]}"
    )
    make -C "${ELPA_BUILD}" -j"${ELPA_JOBS}" install
fi

if [[ "${BUILD_SLATE}" == "1" ]]; then
    SLATE_SOURCE="${ROOT}/third_party/slate"
    SLATE_BUILD="${BUILD_ROOT}/slate/${PROFILE}"

    cmake -S "${SLATE_SOURCE}" -B "${SLATE_BUILD}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${SLATE_PREFIX}" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTING=OFF \
        -Dbuild_tests=OFF \
        -Dgpu_backend="${SLATE_GPU_BACKEND}" \
        -DBLAS_LIBRARIES="-L${MATH_PREFIX}/lib;-lopenblas" \
        -DLAPACK_LIBRARIES="-L${MATH_PREFIX}/lib;-lopenblas" \
        "${EXTRA_SLATE_ARGS[@]}"
    cmake --build "${SLATE_BUILD}" --parallel "${JOBS}"
    cmake --install "${SLATE_BUILD}"
fi

printf 'MATH_PREFIX=%s\nELPA_PREFIX=%s\nSLATE_PREFIX=%s\n' \
    "${MATH_PREFIX}" "${ELPA_PREFIX}" "${SLATE_PREFIX}"
