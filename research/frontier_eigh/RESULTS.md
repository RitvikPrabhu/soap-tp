# Custom distributed eigh on Frontier: experiment record

Start: 2026-10-22 17:35 EDT. Deadline: 21:35 EDT. Branch: `research/custom-eigh-frontier-20261022`.

## Architectural evidence

- [cuSOLVERMp syevd](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermpsyevd): distributed 2D block-cyclic device storage; SYTRD, divide-and-conquer STEDC, ORMTR stages; host and device workspaces. A proprietary two-stage reduction cannot be inferred from this interface.
- ScaLAPACK `third_party/scalapack/SRC/pdsyevd.f` follows PDSYTRD → replicated d/e → PDSTEDC → PDORMTR. `pdstedc.f` supports independent process-row and process-column counts, square *blocks*, and distributed host eigenvectors. Workspace ~two local dense matrices plus O(n).
- SLATE `third_party/slate/src/heev.cc` uses dense-to-band → root band-to-tridiagonal → distributed stedc → two backtransforms. Its QR tree requires a square process grid, so it is not an applicable rectangular-grid implementation.
- ELPA `third_party/elpa/src/elpa2/` provides a better distributed second-stage reference: redistributed narrow band, neighbor boundary exchanges, retained bulge-chasing reflectors, GPU blocked backtransforms. Dense-to-band is GEMM-rich; band chasing can remain on CPUs initially.
- [rocSOLVER syevd source](https://github.com/ROCm/rocSOLVER/blob/develop/library/src/lapack/roclapack_syevd_heevd.hpp): local SYTRD → STEDC → ORMTR. Local tiles are not independent eigenproblems.

## Experiments

### E0: establish correctness (17:44–17:45 EDT)

Hypothesis: direct distributed Householder → PDSTEDC → reverse reflector application is a workable rectangular-grid baseline.

- Driver `eigh.cpp`: FP64/FP32 GPU local matrices, rocBLAS GEMV, custom fused rank-2/rank-1 kernels, host-staged MPI row/column broadcasts/reductions, PDSTEDC distributed FP64 CPU solve. No full dense gather; only vectors replicated.
- Job **5735939**, one node frontier00270: 19 cases, n=97, block=8, 1/2/4/8 GCDs, all eight grids in checkpoint. All passed. Raw `output/custom-eigh/job-5735939.{out,err}`.
- Exact distributed GPU SUMMA checks AQ and QᵀQ; host distributed PDTRAN prepares Qᵀ. Eigenvalues checked against CPU LAPACK only for n≤512.
- Example 8x1 FP32: residual 2.21e-7, orthogonality 3.35e-6, relative max eigenvalue error 1.62e-7.
- Example 2x4 lowrank FP64: residual 1.27e-15, orthogonality 1.35e-14, eigenvalue error 9.94e-16.
- Tiny local n=2/31 edge/reference tests passed. Login GPU timings are explicitly NOT performance evidence.
- Reference integration failure: ELPA GPU API with host eigenvalue pointer returned success but invalid zeros; changed reference buffer to device, then n=31 ELPA2 residual 7.94e-16. No existing code was changed.

Conclusion: keep baseline and proceed to phase/grid/block/precision measurements. Baseline snapshot `build/custom-eigh/baseline-eigh`.
