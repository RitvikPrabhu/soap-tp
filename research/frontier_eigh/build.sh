#!/bin/bash
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
mkdir -p "$ROOT/build/custom-eigh"
ELPA="$ROOT/build/elpa-install/rocm"
ELPA_INCLUDE=$(find "$ELPA/include" -name elpa.h -printf '%h\n')
hipcc -O3 -std=c++17 --offload-arch=gfx90a \
    $(CC --cray-print-opts=all) \
    -I"$(dirname "$ELPA_INCLUDE")" \
    "$ROOT/research/frontier_eigh/eigh.cpp" \
    -L"$ROOT/build/math-install/lib" -lscalapack -lopenblas \
    -L"$ELPA/lib" -lelpa -Wl,-rpath,"$ELPA/lib" \
    -Wl,-rpath,"$ROOT/build/math-install/lib" \
    -lrocsolver -lrocblas -lrccl -lgfortran \
    -o "$ROOT/build/custom-eigh/eigh"
