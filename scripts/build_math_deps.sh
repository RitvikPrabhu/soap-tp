#!/usr/bin/env bash

# Internal fallback for the native solver builds.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENBLAS="${ROOT}/third_party/openblas"
SCALAPACK="${ROOT}/third_party/scalapack"
BUILD_ROOT="${SOAP_TP_BUILD_ROOT:-${ROOT}/build}"
BUILD="${BUILD_ROOT}/math"
PREFIX="${MATH_PREFIX:-${BUILD_ROOT}/math-install}"
JOBS="${BUILD_JOBS:-8}"

MPI_CC="${CC:-mpicc}"
MPI_FC="${FC:-mpifort}"

fix_openblas_install_name() {
    if [[ "$(uname -s)" != "Darwin" ]]; then
        return
    fi

    # OpenBLAS's experimental CMake build records its build-tree path as the
    # dylib ID. Replace it with the persistent installed path before consumers
    # link, otherwise SLATE would keep depending on build/math/openblas.
    for library in "${PREFIX}/lib"/libopenblas.*.dylib; do
        if [[ -f "${library}" && ! -L "${library}" ]]; then
            install_name_tool -id "${library}" "${library}"
        fi
    done
}

if [[ -f "${PREFIX}/lib/libopenblas.a" && \
      -f "${PREFIX}/lib/libscalapack.a" ]] && \
   { [[ -f "${PREFIX}/lib/libopenblas.so" ]] || \
     [[ -f "${PREFIX}/lib/libopenblas.dylib" ]]; }; then
    fix_openblas_install_name
    exit 0
fi

if [[ ! -f "${OPENBLAS}/CMakeLists.txt" || \
      ! -f "${SCALAPACK}/CMakeLists.txt" ]]; then
    git -C "${ROOT}" submodule update --init \
        third_party/openblas third_party/scalapack
fi

mkdir -p "${BUILD}/openblas" "${BUILD}/scalapack" "${PREFIX}"

echo "Building OpenBLAS."
cmake -S "${OPENBLAS}" -B "${BUILD}/openblas" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_C_COMPILER="${MPI_CC}" \
    -DCMAKE_Fortran_COMPILER="${MPI_FC}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_STATIC_LIBS=ON \
    -DBUILD_SHARED_LIBS=ON \
    -DBUILD_WITHOUT_LAPACKE=ON \
    -DBUILD_TESTING=OFF
cmake --build "${BUILD}/openblas" \
    --target openblas_static openblas_shared --parallel "${JOBS}"
cmake --install "${BUILD}/openblas"
fix_openblas_install_name

echo "Building ScaLAPACK."
cmake -S "${SCALAPACK}" -B "${BUILD}/scalapack" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_C_COMPILER="${MPI_CC}" \
    -DCMAKE_Fortran_COMPILER="${MPI_FC}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DSCALAPACK_BUILD_TESTS=OFF \
    -DBLAS_LIBRARIES="${PREFIX}/lib/libopenblas.a" \
    -DLAPACK_LIBRARIES="${PREFIX}/lib/libopenblas.a"
cmake --build "${BUILD}/scalapack" --parallel "${JOBS}"
cmake --install "${BUILD}/scalapack"

echo "Math libraries installed in ${PREFIX}"
