#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-cpu}"
SOURCE="${ROOT}/third_party/elpa"
BUILD_ROOT="${SOAP_TP_BUILD_ROOT:-${ROOT}/build}"
BUILD="${BUILD_ROOT}/elpa/${PROFILE}"
PREFIX="${ELPA_PREFIX:-${BUILD_ROOT}/elpa-install/${PROFILE}}"
MATH_PREFIX="${MATH_PREFIX:-${BUILD_ROOT}/math-install}"
JOBS="${BUILD_JOBS:-8}"

if [[ -n "${CC:-}" && -n "${CXX:-}" && -n "${FC:-}" ]]; then
    MPI_CC="${CC}"
    MPI_CXX="${CXX}"
    MPI_FC="${FC}"
elif command -v mpicc >/dev/null && command -v mpicxx >/dev/null && command -v mpifort >/dev/null; then
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

if [[ -n "${ELPA_CONFIGURE_ARGS:-}" ]]; then
    # This escape hatch is useful for cluster-specific settings such as CUDA
    # compute capabilities and nonstandard toolkit prefixes. Arguments may not
    # contain whitespace.
    read -r -a EXTRA_CONFIGURE_ARGS <<<"${ELPA_CONFIGURE_ARGS}"
    CONFIGURE_ARGS+=("${EXTRA_CONFIGURE_ARGS[@]}")
fi

if [[ "$(uname -m)" == "arm64" || "$(uname -m)" == "aarch64" ]]; then
    CONFIGURE_ARGS+=(--disable-sse-kernels --disable-sse-assembly-kernels)
    CONFIGURE_ARGS+=(--disable-avx-kernels --disable-avx2-kernels --disable-avx512-kernels)
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
    CONFIGURE_ARGS+=(--disable-affinity-checking)
fi

if [[ ! -f "${SOURCE}/configure.ac" || \
      ! -f "${ROOT}/third_party/openblas/CMakeLists.txt" || \
      ! -f "${ROOT}/third_party/scalapack/CMakeLists.txt" ]]; then
    git -C "${ROOT}" submodule update --init \
        third_party/elpa third_party/openblas third_party/scalapack
fi

if [[ ! -x "${SOURCE}/configure" ]]; then
    command -v autoreconf >/dev/null || {
        echo "Install or load Autoconf, Automake, and Libtool, then retry." >&2
        exit 1
    }
    (cd "${SOURCE}" && ./autogen.sh)
fi

# Use a loaded ScaLAPACK module when possible; otherwise build our pinned copy.
# Probe all three interfaces ELPA needs.  A ScaLAPACK shared library can link a
# BLACS-only program through transitive dependencies even when the development
# link names for BLAS and LAPACK are unavailable; accepting that partial link
# line makes ELPA's later dgemm/dlarrv configure checks fail.
if [[ -z "${SCALAPACK_LDFLAGS:-}" ]]; then
    mkdir -p "${BUILD_ROOT}"
    cat >"${BUILD_ROOT}/scalapack_probe.f90" <<'EOF'
program probe
  call blacs_pinfo(i, n)
  call dgemm
  call dlarrv
end program probe
EOF

    for flags in \
        "-lscalapack -lopenblas" \
        "-lscalapack-openmpi -lopenblas" \
        "-lmpiscalapack -lopenblas" \
        "-lscalapack -llapack -lblas" \
        "-lscalapack-openmpi -llapack -lblas" \
        "-lmpiscalapack -llapack -lblas"; do
        read -r -a LINK_FLAGS <<<"${flags}"
        if "${MPI_FC}" "${BUILD_ROOT}/scalapack_probe.f90" "${LINK_FLAGS[@]}" -o "${BUILD_ROOT}/scalapack_probe" >/dev/null 2>&1; then
            SCALAPACK_LDFLAGS="${flags}"
            break
        fi
    done
fi

if [[ -z "${SCALAPACK_LDFLAGS:-}" ]]; then
    SOAP_TP_BUILD_ROOT="${BUILD_ROOT}" MATH_PREFIX="${MATH_PREFIX}" \
        "${ROOT}/scripts/build_math_deps.sh"
    # Keep ELPA's fallback self-contained with the static OpenBLAS archive.
    # The shared fallback exists for C++ consumers such as SLATE, which need
    # its recorded Fortran runtime dependencies.
    SCALAPACK_LDFLAGS="-L${MATH_PREFIX}/lib -lscalapack ${MATH_PREFIX}/lib/libopenblas.a"
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
