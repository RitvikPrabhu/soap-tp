#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-cpu}"
SOURCE="${ROOT}/third_party/elpa"
BUILD="${ROOT}/build/elpa/${PROFILE}"
PREFIX="${ROOT}/build/elpa-install/${PROFILE}"
MATH_PREFIX="${ROOT}/build/math-install"
JOBS="${BUILD_JOBS:-8}"

if command -v mpicc >/dev/null && command -v mpicxx >/dev/null && command -v mpifort >/dev/null; then
    MPI_CC=mpicc
    MPI_CXX=mpicxx
    MPI_FC=mpifort
elif command -v cc >/dev/null && command -v CC >/dev/null && command -v ftn >/dev/null; then
    MPI_CC=cc
    MPI_CXX=CC
    MPI_FC=ftn
else
    echo "Load an MPI module that provides mpicc, mpicxx, and mpifort." >&2
    exit 1
fi
export CC="${MPI_CC}" CXX="${MPI_CXX}" FC="${MPI_FC}"

CONFIGURE_ARGS=(--prefix="${PREFIX}" --with-mpi=yes --with-test-programs=no)
case "${PROFILE}" in
    cpu) ;;
    rocm) CONFIGURE_ARGS+=(--enable-amd-gpu-kernels --enable-gpu-streams=amd --with-rocsolver=yes) ;;
    cuda) CONFIGURE_ARGS+=(--enable-nvidia-gpu-kernels --enable-gpu-streams=nvidia) ;;
    *) echo "Usage: $0 [cpu|rocm|cuda]" >&2; exit 1 ;;
esac

if [[ "$(uname -m)" == "arm64" || "$(uname -m)" == "aarch64" ]]; then
    CONFIGURE_ARGS+=(--disable-sse-kernels --disable-sse-assembly-kernels)
    CONFIGURE_ARGS+=(--disable-avx-kernels --disable-avx2-kernels --disable-avx512-kernels)
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
    CONFIGURE_ARGS+=(--disable-affinity-checking)
fi

if [[ ! -f "${SOURCE}/configure.ac" ]]; then
    git -C "${ROOT}" submodule update --init --recursive
fi

if [[ ! -x "${SOURCE}/configure" ]]; then
    command -v autoreconf >/dev/null || {
        echo "Install or load Autoconf, Automake, and Libtool, then retry." >&2
        exit 1
    }
    (cd "${SOURCE}" && ./autogen.sh)
fi

# Use a loaded ScaLAPACK module when possible; otherwise build our pinned copy.
if [[ -z "${SCALAPACK_LDFLAGS:-}" ]]; then
    mkdir -p "${ROOT}/build"
    cat >"${ROOT}/build/scalapack_probe.f90" <<'EOF'
program probe
  call blacs_pinfo(i, n)
end program probe
EOF

    for flags in -lscalapack -lscalapack-openmpi -lmpiscalapack; do
        if "${MPI_FC}" "${ROOT}/build/scalapack_probe.f90" ${flags} -o "${ROOT}/build/scalapack_probe" >/dev/null 2>&1; then
            SCALAPACK_LDFLAGS="${flags}"
            break
        fi
    done
fi

if [[ -z "${SCALAPACK_LDFLAGS:-}" ]]; then
    "${ROOT}/scripts/build_math_deps.sh"
    SCALAPACK_LDFLAGS="-L${MATH_PREFIX}/lib -lscalapack -lopenblas"
fi
export SCALAPACK_LDFLAGS

mkdir -p "${BUILD}" "${PREFIX}"
(cd "${BUILD}" && "${SOURCE}/configure" "${CONFIGURE_ARGS[@]}")

MAKE_ARGS=(-j"${JOBS}")
if [[ "$(uname -s)" == "Darwin" ]]; then
    MAKE_ARGS=(-j1 "FORTRAN_CPP=${MPI_FC} -E -cpp -P -traditional -Wall -Werror")
fi

make -C "${BUILD}" "${MAKE_ARGS[@]}"
make -C "${BUILD}" install

echo "ELPA installed in ${PREFIX}"
