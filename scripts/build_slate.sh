#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-cpu}"
SOURCE="${ROOT}/third_party/slate"
BUILD_ROOT="${SOAP_TP_BUILD_ROOT:-${ROOT}/build}"
BUILD="${BUILD_ROOT}/slate/${PROFILE}"
PREFIX="${SLATE_PREFIX:-${BUILD_ROOT}/slate-install/${PROFILE}}"
MATH_PREFIX="${MATH_PREFIX:-${BUILD_ROOT}/math-install}"
JOBS="${BUILD_JOBS:-8}"

case "${PROFILE}" in
    cpu) GPU_BACKEND=none ;;
    cuda) GPU_BACKEND=cuda ;;
    rocm) GPU_BACKEND=hip ;;
    *) echo "Usage: $0 [cpu|cuda|rocm]" >&2; exit 1 ;;
esac

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

if [[ ! -f "${SOURCE}/CMakeLists.txt" || \
      ! -f "${SOURCE}/blaspp/CMakeLists.txt" || \
      ! -f "${SOURCE}/lapackpp/CMakeLists.txt" ]]; then
    git -C "${ROOT}" submodule update --init --recursive third_party/slate
fi

if ! command -v cmake >/dev/null 2>&1; then
    echo "CMake 3.18 or newer is required to build SLATE." >&2
    exit 1
fi

CMAKE_ARGS=(
    -S "${SOURCE}"
    -B "${BUILD}"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="${PREFIX}"
    -DCMAKE_INSTALL_LIBDIR=lib
    -DCMAKE_INSTALL_RPATH="${PREFIX}/lib"
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
    -DCMAKE_C_COMPILER="${MPI_CC}"
    -DCMAKE_CXX_COMPILER="${MPI_CXX}"
    -DCMAKE_Fortran_COMPILER="${MPI_FC}"
    -DBUILD_SHARED_LIBS=ON
    -DBUILD_TESTING=OFF
    -Dbuild_tests=OFF
    -Dgpu_backend="${GPU_BACKEND}"
)

# Prefer an explicitly selected BLAS/LAPACK. If build_elpa.sh created the
# pinned OpenBLAS fallback, reuse that exact library for SLATE as well. SLATE
# is linked by a C++ compiler, so use the shared fallback: it records the
# Fortran runtime needed by OpenBLAS instead of losing it behind a static
# archive.
OPENBLAS_SHARED=""
for candidate in \
    "${MATH_PREFIX}/lib/libopenblas.so" \
    "${MATH_PREFIX}/lib/libopenblas.dylib"; do
    if [[ -f "${candidate}" ]]; then
        OPENBLAS_SHARED="${candidate}"
        break
    fi
done

if [[ -z "${BLAS_LIBRARIES:-}" && \
      -z "${OPENBLAS_SHARED}" && \
      -f "${MATH_PREFIX}/lib/libopenblas.a" ]]; then
    SOAP_TP_BUILD_ROOT="${BUILD_ROOT}" MATH_PREFIX="${MATH_PREFIX}" \
        "${ROOT}/scripts/build_math_deps.sh"
    for candidate in \
        "${MATH_PREFIX}/lib/libopenblas.so" \
        "${MATH_PREFIX}/lib/libopenblas.dylib"; do
        if [[ -f "${candidate}" ]]; then
            OPENBLAS_SHARED="${candidate}"
            break
        fi
    done
fi

if [[ -n "${BLAS_LIBRARIES:-}" ]]; then
    CMAKE_ARGS+=("-DBLAS_LIBRARIES=${BLAS_LIBRARIES}")
elif [[ -n "${OPENBLAS_SHARED}" ]]; then
    CMAKE_ARGS+=("-DBLAS_LIBRARIES=${OPENBLAS_SHARED}")
    CMAKE_ARGS+=("-DLAPACK_LIBRARIES=${OPENBLAS_SHARED}")
    CMAKE_ARGS+=("-DCMAKE_INSTALL_RPATH=${PREFIX}/lib;${MATH_PREFIX}/lib")
fi
if [[ -n "${LAPACK_LIBRARIES:-}" ]]; then
    CMAKE_ARGS+=("-DLAPACK_LIBRARIES=${LAPACK_LIBRARIES}")
fi

if [[ -n "${SLATE_CMAKE_ARGS:-}" ]]; then
    # This escape hatch supports site-specific package paths and GPU
    # architectures. Arguments may not contain whitespace.
    read -r -a EXTRA_CMAKE_ARGS <<<"${SLATE_CMAKE_ARGS}"
    CMAKE_ARGS+=("${EXTRA_CMAKE_ARGS[@]}")
fi

mkdir -p "${BUILD}" "${PREFIX}"
cmake "${CMAKE_ARGS[@]}"
cmake --build "${BUILD}" --parallel "${JOBS}"
cmake --install "${BUILD}"

if [[ "$(uname -s)" == "Darwin" && -n "${OPENBLAS_SHARED}" ]]; then
    # Upgrade installations created with the older static-only fallback too.
    # Their BLAS++/LAPACK++ dylibs can retain OpenBLAS's former build-tree ID
    # even after OpenBLAS itself has been fixed and reinstalled.
    OPENBLAS_INSTALL_NAME="$(otool -D "${OPENBLAS_SHARED}" | sed -n '2p')"
    for library in "${PREFIX}/lib"/*.dylib; do
        if [[ ! -f "${library}" || -L "${library}" ]]; then
            continue
        fi
        while IFS= read -r dependency; do
            case "${dependency}" in
                "${BUILD_ROOT}/math/openblas/lib/"libopenblas.*.dylib)
                    install_name_tool -change \
                        "${dependency}" "${OPENBLAS_INSTALL_NAME}" "${library}"
                    ;;
            esac
        done < <(otool -L "${library}" | awk 'NR > 1 { print $1 }')
    done
fi

if [[ "$(uname -s)" == "Darwin" ]]; then
    # Make SLATE and PyTorch resolve the same OpenMP runtime.
    if TORCH_LIB_DIR="$("${PYTHON:-python3}" -c \
        'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")' \
        2>/dev/null)" && [[ -f "${TORCH_LIB_DIR}/libomp.dylib" ]]; then
        for library in "${PREFIX}/lib"/*.dylib; do
            [[ -f "${library}" && ! -L "${library}" ]] || continue
            dependency="$(otool -L "${library}" | awk '$1 ~ /libomp[.]dylib/ { print $1; exit }')"
            [[ -n "${dependency}" ]] || continue
            [[ "${dependency}" == "@rpath/libomp.dylib" ]] || \
                install_name_tool -change \
                    "${dependency}" "@rpath/libomp.dylib" "${library}"
            install_name_tool -delete_rpath \
                "${TORCH_LIB_DIR}" "${library}" 2>/dev/null || :
            install_name_tool -add_rpath "${TORCH_LIB_DIR}" "${library}"
        done
    fi
fi

echo "SLATE installed in ${PREFIX}"
