#!/bin/bash
# Run with the same Frontier modules as half_micro.slurm; compilation only.
set -euo pipefail
ROOT=$(git rev-parse --show-toplevel)
mkdir -p "$ROOT/build/custom-eigh"
ROCM_ROOT=$(dirname "$(dirname "$(command -v hipcc)")")
extra=()
if grep -q 'rocblas_gemm_flags_fp16_alt_impl ' "$ROCM_ROOT/include/rocblas/internal/rocblas-types.h"; then
    extra+=(-DHALF_MICRO_ALT=1)
fi
set -x
hipcc -O3 -std=c++17 --offload-arch=gfx90a "${extra[@]}" \
    "$ROOT/research/frontier_eigh/half_micro.cpp" -lrocblas -ldl \
    -o "$ROOT/build/custom-eigh/half_micro"
sha256sum "$ROOT/build/custom-eigh/half_micro" "$ROOT/research/frontier_eigh/half_micro.cpp"
