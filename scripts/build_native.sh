#!/usr/bin/env bash

# Build the native libraries that are not supplied by the active environment.
# ELPA automatically builds the pinned math fallback only when no loaded
# ScaLAPACK/BLAS combination passes its link probe.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-cpu}"
shift $(( $# > 0 ? 1 : 0 ))
TARGETS=("$@")

usage() {
    cat <<'EOF'
Usage: scripts/build_native.sh [cpu|cuda|rocm] [elpa] [slate] [math]

Build one or more native dependencies. With no targets, builds ELPA and SLATE.
The normal entry point is install.sh, which calls this script only for libraries
not selected through --skip-elpa or --skip-slate.

Completed installations are reused. Set SOAP_TP_FORCE_NATIVE_BUILD=1 to rebuild
a native library in place.
EOF
}

case "${PROFILE}" in
    cpu|cuda|rocm) ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

if (( ${#TARGETS[@]} == 0 )); then
    TARGETS=(elpa slate)
fi
for target in "${TARGETS[@]}"; do
    case "${target}" in
        elpa|slate|math) ;;
        *) usage >&2; exit 2 ;;
    esac
done

BUILD_ROOT="${SOAP_TP_BUILD_ROOT:-${ROOT}/build}"
MATH_PREFIX="${MATH_PREFIX:-${BUILD_ROOT}/math-install}"
JOBS="${BUILD_JOBS:-8}"
export SOAP_TP_BUILD_ROOT="${BUILD_ROOT}" MATH_PREFIX

if [[ -n "${CC:-}" && -n "${CXX:-}" && -n "${FC:-}" ]]; then
    MPI_CC="${CC}"
    MPI_CXX="${CXX}"
    MPI_FC="${FC}"
elif command -v cc >/dev/null 2>&1 && \
     command -v CC >/dev/null 2>&1 && \
     command -v ftn >/dev/null 2>&1; then
    MPI_CC=cc
    MPI_CXX=CC
    MPI_FC=ftn
elif command -v mpicc >/dev/null 2>&1 && \
     command -v mpicxx >/dev/null 2>&1 && \
     command -v mpifort >/dev/null 2>&1; then
    MPI_CC=mpicc
    MPI_CXX=mpicxx
    MPI_FC=mpifort
else
    echo "Load an MPI compiler environment or set CC, CXX, and FC." >&2
    exit 1
fi
export CC="${MPI_CC}" CXX="${MPI_CXX}" FC="${MPI_FC}"

fix_openblas_install_name() {
    if [[ "$(uname -s)" != "Darwin" ]]; then
        return
    fi
    for library in "${MATH_PREFIX}/lib"/libopenblas.*.dylib; do
        if [[ -f "${library}" && ! -L "${library}" ]]; then
            install_name_tool -id "${library}" "${library}"
        fi
    done
}

build_math() {
    local openblas="${ROOT}/third_party/openblas"
    local scalapack="${ROOT}/third_party/scalapack"
    local build="${BUILD_ROOT}/math"

    if [[ -f "${MATH_PREFIX}/lib/libopenblas.a" && \
          -f "${MATH_PREFIX}/lib/libscalapack.a" ]] && \
       { [[ -f "${MATH_PREFIX}/lib/libopenblas.so" ]] || \
         [[ -f "${MATH_PREFIX}/lib/libopenblas.dylib" ]]; }; then
        fix_openblas_install_name
        return
    fi

    if [[ ! -f "${openblas}/CMakeLists.txt" || \
          ! -f "${scalapack}/CMakeLists.txt" ]]; then
        git -C "${ROOT}" submodule update --init \
            third_party/openblas third_party/scalapack
    fi

    mkdir -p "${build}/openblas" "${build}/scalapack" "${MATH_PREFIX}"

    echo "Building OpenBLAS."
    cmake -S "${openblas}" -B "${build}/openblas" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${MATH_PREFIX}" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_C_COMPILER="${MPI_CC}" \
        -DCMAKE_Fortran_COMPILER="${MPI_FC}" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_STATIC_LIBS=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_WITHOUT_LAPACKE=ON \
        -DBUILD_TESTING=OFF
    cmake --build "${build}/openblas" \
        --target openblas_static openblas_shared --parallel "${JOBS}"
    cmake --install "${build}/openblas"
    fix_openblas_install_name

    echo "Building ScaLAPACK."
    cmake -S "${scalapack}" -B "${build}/scalapack" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${MATH_PREFIX}" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_C_COMPILER="${MPI_CC}" \
        -DCMAKE_Fortran_COMPILER="${MPI_FC}" \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DBUILD_SHARED_LIBS=OFF \
        -DSCALAPACK_BUILD_TESTS=OFF \
        -DBLAS_LIBRARIES="${MATH_PREFIX}/lib/libopenblas.a" \
        -DLAPACK_LIBRARIES="${MATH_PREFIX}/lib/libopenblas.a"
    cmake --build "${build}/scalapack" --parallel "${JOBS}"
    cmake --install "${build}/scalapack"

    echo "Math libraries installed in ${MATH_PREFIX}"
}

build_elpa() {
    local source="${ROOT}/third_party/elpa"
    local build="${BUILD_ROOT}/elpa/${PROFILE}"
    local prefix="${ELPA_PREFIX:-${BUILD_ROOT}/elpa-install/${PROFILE}}"
    local automake_libdir
    local candidate
    local cuda_libdir=""
    local cuda_root=""
    local flags
    local rocm_libdir=""
    local rocm_root=""
    local configure_args=(--prefix="${prefix}" --with-mpi=yes --with-test-programs=no)
    local make_args=(-j"${JOBS}")
    local link_flags=()
    local extra_configure_args=()

    if [[ "${SOAP_TP_FORCE_NATIVE_BUILD:-0}" != "1" ]] && \
       compgen -G "${prefix}/include/elpa-*/elpa/elpa.h" >/dev/null && \
       { compgen -G "${prefix}/lib/libelpa.*" >/dev/null || \
         compgen -G "${prefix}/lib64/libelpa.*" >/dev/null; }; then
        echo "Reusing ELPA in ${prefix}"
        return
    fi

    case "${PROFILE}" in
        cpu) ;;
        rocm)
            configure_args+=(
                --enable-amd-gpu-kernels
                --enable-gpu-streams=amd
                --with-rocsolver=yes
                --disable-sse-kernels
                --disable-sse-assembly-kernels
                --disable-avx-kernels
                --disable-avx2-kernels
                --disable-avx512-kernels
            )
            rocm_root="${ROCM_PATH:-${HIP_PATH:-}}"
            if [[ -z "${rocm_root}" ]] && command -v hipcc >/dev/null 2>&1; then
                rocm_root="$(cd "$(dirname "$(command -v hipcc)")/.." && pwd)"
            fi
            for candidate in "${rocm_root}/lib" "${rocm_root}/lib64"; do
                if compgen -G "${candidate}/librocblas.*" >/dev/null; then
                    rocm_libdir="${candidate}"
                    break
                fi
            done
            if [[ -z "${rocm_libdir}" ]]; then
                echo "The active ROCm environment does not expose librocblas." >&2
                exit 1
            fi
            CPPFLAGS="${CPPFLAGS:+${CPPFLAGS} }-I${rocm_root}/include"
            LDFLAGS="${LDFLAGS:+${LDFLAGS} }-L${rocm_libdir}"
            export CPPFLAGS LDFLAGS
            ;;
        cuda)
            configure_args+=(
                --enable-nvidia-gpu-kernels
                --enable-gpu-streams=nvidia
                --disable-sse-kernels
                --disable-sse-assembly-kernels
                --disable-avx-kernels
                --disable-avx2-kernels
                --disable-avx512-kernels
            )
            cuda_root="${CUDA_HOME:-${CUDA_PATH:-}}"
            if [[ -z "${cuda_root}" ]] && command -v nvcc >/dev/null 2>&1; then
                cuda_root="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
            fi
            for candidate in "${cuda_root}/lib64" "${cuda_root}/lib"; do
                if compgen -G "${candidate}/libcublas.*" >/dev/null; then
                    cuda_libdir="${candidate}"
                    break
                fi
            done
            if [[ -z "${cuda_libdir}" ]]; then
                echo "The active CUDA environment does not expose libcublas." >&2
                exit 1
            fi
            configure_args+=("--with-cuda-path=${cuda_root}")
            CPPFLAGS="${CPPFLAGS:+${CPPFLAGS} }-I${cuda_root}/include"
            LDFLAGS="${LDFLAGS:+${LDFLAGS} }-L${cuda_libdir}"
            export CPPFLAGS LDFLAGS
            ;;
    esac

    if [[ -n "${ELPA_CONFIGURE_ARGS:-}" ]]; then
        read -r -a extra_configure_args <<<"${ELPA_CONFIGURE_ARGS}"
        configure_args+=("${extra_configure_args[@]}")
    fi
    if [[ "$(uname -m)" == "arm64" || "$(uname -m)" == "aarch64" ]]; then
        configure_args+=(
            --disable-sse-kernels
            --disable-sse-assembly-kernels
            --disable-avx-kernels
            --disable-avx2-kernels
            --disable-avx512-kernels
        )
    fi
    if [[ "$(uname -s)" == "Darwin" ]]; then
        configure_args+=(--disable-affinity-checking)
        make_args=(-j1 "FORTRAN_CPP=${MPI_FC} -E -cpp -P -traditional -Wall -Werror")
    fi

    if [[ ! -f "${source}/configure.ac" ]]; then
        git -C "${ROOT}" submodule update --init third_party/elpa
    fi
    if [[ ! -x "${source}/configure" ]]; then
        if ! command -v autoreconf >/dev/null 2>&1 || \
           ! command -v automake >/dev/null 2>&1; then
            echo "Install or load Autoconf, Automake, and Libtool." >&2
            exit 1
        fi
        automake_libdir="$(automake --print-libdir)"
        if [[ ! -f "${automake_libdir}/install-sh" ]]; then
            echo "Automake install-sh was not found under ${automake_libdir}." >&2
            exit 1
        fi
        if [[ ! -f "${source}/install-sh" ]]; then
            cp "${automake_libdir}/install-sh" "${source}/install-sh"
        fi
        (cd "${source}" && ./autogen.sh)
    fi

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
            read -r -a link_flags <<<"${flags}"
            if "${MPI_FC}" "${BUILD_ROOT}/scalapack_probe.f90" \
                "${link_flags[@]}" -o "${BUILD_ROOT}/scalapack_probe" \
                >/dev/null 2>&1; then
                SCALAPACK_LDFLAGS="${flags}"
                break
            fi
        done
    fi
    if [[ -z "${SCALAPACK_LDFLAGS:-}" ]]; then
        build_math
        SCALAPACK_LDFLAGS="-L${MATH_PREFIX}/lib -lscalapack ${MATH_PREFIX}/lib/libopenblas.a"
    fi
    export SCALAPACK_LDFLAGS

    mkdir -p "${build}" "${prefix}"
    (cd "${build}" && "${source}/configure" "${configure_args[@]}")
    make -C "${build}" "${make_args[@]}"
    make -C "${build}" install
    echo "ELPA installed in ${prefix}"
}

build_slate() {
    local source="${ROOT}/third_party/slate"
    local build="${BUILD_ROOT}/slate/${PROFILE}"
    local prefix="${SLATE_PREFIX:-${BUILD_ROOT}/slate-install/${PROFILE}}"
    local gpu_backend
    local openblas_shared=""
    local openblas_install_name
    local dependency
    local library
    local candidate
    local torch_lib_dir
    local cmake_args=()
    local extra_cmake_args=()

    if [[ "${SOAP_TP_FORCE_NATIVE_BUILD:-0}" != "1" ]] && \
       [[ -f "${prefix}/include/slate/slate.hh" ]] && \
       { compgen -G "${prefix}/lib/libslate.*" >/dev/null || \
         compgen -G "${prefix}/lib64/libslate.*" >/dev/null; }; then
        echo "Reusing SLATE in ${prefix}"
        return
    fi

    case "${PROFILE}" in
        cpu) gpu_backend=none ;;
        cuda) gpu_backend=cuda ;;
        rocm) gpu_backend=hip ;;
    esac

    if [[ ! -f "${source}/CMakeLists.txt" || \
          ! -f "${source}/blaspp/CMakeLists.txt" || \
          ! -f "${source}/lapackpp/CMakeLists.txt" ]]; then
        git -C "${ROOT}" submodule update --init --recursive third_party/slate
    fi

    cmake_args=(
        -S "${source}"
        -B "${build}"
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_INSTALL_PREFIX="${prefix}"
        -DCMAKE_INSTALL_LIBDIR=lib
        -DCMAKE_INSTALL_RPATH="${prefix}/lib"
        -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON
        -DCMAKE_C_COMPILER="${MPI_CC}"
        -DCMAKE_CXX_COMPILER="${MPI_CXX}"
        -DCMAKE_Fortran_COMPILER="${MPI_FC}"
        -DBUILD_SHARED_LIBS=ON
        -DBUILD_TESTING=OFF
        -Dbuild_tests=OFF
        -Dgpu_backend="${gpu_backend}"
    )

    for candidate in \
        "${MATH_PREFIX}/lib/libopenblas.so" \
        "${MATH_PREFIX}/lib/libopenblas.dylib"; do
        if [[ -f "${candidate}" ]]; then
            openblas_shared="${candidate}"
            break
        fi
    done
    if [[ -z "${BLAS_LIBRARIES:-}" && \
          -z "${openblas_shared}" && \
          -f "${MATH_PREFIX}/lib/libopenblas.a" ]]; then
        build_math
        for candidate in \
            "${MATH_PREFIX}/lib/libopenblas.so" \
            "${MATH_PREFIX}/lib/libopenblas.dylib"; do
            if [[ -f "${candidate}" ]]; then
                openblas_shared="${candidate}"
                break
            fi
        done
    fi

    if [[ -n "${BLAS_LIBRARIES:-}" ]]; then
        cmake_args+=("-DBLAS_LIBRARIES=${BLAS_LIBRARIES}")
    elif [[ -n "${openblas_shared}" ]]; then
        cmake_args+=("-DBLAS_LIBRARIES=${openblas_shared}")
        cmake_args+=("-DLAPACK_LIBRARIES=${openblas_shared}")
        cmake_args+=("-DCMAKE_INSTALL_RPATH=${prefix}/lib;${MATH_PREFIX}/lib")
    fi
    if [[ -n "${LAPACK_LIBRARIES:-}" ]]; then
        cmake_args+=("-DLAPACK_LIBRARIES=${LAPACK_LIBRARIES}")
    fi
    if [[ -n "${SLATE_CMAKE_ARGS:-}" ]]; then
        read -r -a extra_cmake_args <<<"${SLATE_CMAKE_ARGS}"
        cmake_args+=("${extra_cmake_args[@]}")
    fi

    mkdir -p "${build}" "${prefix}"
    cmake "${cmake_args[@]}"
    cmake --build "${build}" --parallel "${JOBS}"
    cmake --install "${build}"

    if [[ "$(uname -s)" == "Darwin" && -n "${openblas_shared}" ]]; then
        openblas_install_name="$(otool -D "${openblas_shared}" | sed -n '2p')"
        for library in "${prefix}/lib"/*.dylib; do
            if [[ ! -f "${library}" || -L "${library}" ]]; then
                continue
            fi
            while IFS= read -r dependency; do
                case "${dependency}" in
                    "${BUILD_ROOT}/math/openblas/lib/"libopenblas.*.dylib)
                        install_name_tool -change \
                            "${dependency}" "${openblas_install_name}" "${library}"
                        ;;
                esac
            done < <(otool -L "${library}" | awk 'NR > 1 { print $1 }')
        done
    fi

    if [[ "$(uname -s)" == "Darwin" ]]; then
        if torch_lib_dir="$("${PYTHON:-python3}" -c \
            'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")' \
            2>/dev/null)" && [[ -f "${torch_lib_dir}/libomp.dylib" ]]; then
            for library in "${prefix}/lib"/*.dylib; do
                [[ -f "${library}" && ! -L "${library}" ]] || continue
                dependency="$(otool -L "${library}" | awk \
                    '$1 ~ /libomp[.]dylib/ { print $1; exit }')"
                [[ -n "${dependency}" ]] || continue
                [[ "${dependency}" == "@rpath/libomp.dylib" ]] || \
                    install_name_tool -change \
                        "${dependency}" "@rpath/libomp.dylib" "${library}"
                install_name_tool -delete_rpath \
                    "${torch_lib_dir}" "${library}" 2>/dev/null || :
                install_name_tool -add_rpath "${torch_lib_dir}" "${library}"
            done
        fi
    fi

    echo "SLATE installed in ${prefix}"
}

for target in "${TARGETS[@]}"; do
    case "${target}" in
        math) build_math ;;
        elpa) build_elpa ;;
        slate) build_slate ;;
    esac
done
