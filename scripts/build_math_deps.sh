#!/usr/bin/env bash

# Internal fallback for build_elpa.sh.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENBLAS="${ROOT}/third_party/openblas"
SCALAPACK="${ROOT}/third_party/scalapack"
BUILD="${ROOT}/build/math"
PREFIX="${ROOT}/build/math-install"
JOBS="${BUILD_JOBS:-8}"

MPI_CC="${CC:-mpicc}"
MPI_FC="${FC:-mpifort}"

if [[ -f "${PREFIX}/lib/libopenblas.a" && -f "${PREFIX}/lib/libscalapack.a" ]]; then
    exit 0
fi

mkdir -p "${BUILD}/openblas" "${BUILD}/scalapack" "${PREFIX}"

echo "Building OpenBLAS."
cmake -S "${OPENBLAS}" -B "${BUILD}/openblas" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="${PREFIX}" \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DCMAKE_C_COMPILER=cc \
    -DCMAKE_Fortran_COMPILER="${MPI_FC}" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DBUILD_STATIC_LIBS=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_WITHOUT_LAPACKE=ON \
    -DBUILD_TESTING=OFF
cmake --build "${BUILD}/openblas" --target openblas_static --parallel "${JOBS}"
cmake --install "${BUILD}/openblas"

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
