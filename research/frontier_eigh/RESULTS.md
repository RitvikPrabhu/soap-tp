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

### E1: grid, tile, precision, collective baseline (17:46–17:48 EDT)

Job **5735963**, frontier01681, completed in 70 seconds. n=2048, block=64 unless noted, two trials (trial 0 includes library warmup, trial 1 shown). All residual/orthogonality checks passed. Raw `output/custom-eigh/job-5735963.{out,err}`.

| GCDs / grid | FP64 seconds | FP32 seconds |
|---|---:|---:|
| 1 / 1x1 | 1.1180 | 0.9451 |
| 2 / 1x2 | 0.9650 | 0.8152 |
| 4 / 2x2 | 0.8068 | 0.7009 |
| 8 / 1x8 | 0.6854 | 0.6179 |
| 8 / 2x4 | 0.7293 | 0.6345 |
| 8 / 4x2 | 0.7726 | 0.6670 |
| 8 / 8x1 | 0.9377 | 0.7945 |

- 2x4 FP64 breakdown: reduction 0.4168, PDSTEDC+handoff 0.1106, backtransform 0.2017 seconds. FP32: 0.3487 / 0.1059 / 0.1798.
- ELPA1 2x4: FP64 0.3230, FP32 0.2890 s. ELPA2: FP64 0.2130, FP32 0.1611 s. Each call includes ELPA handle/GPU setup; setup not cached. Large cold-start costs are explicitly excluded from these warm comparisons.
- Block=32/64/128/256: custom FP64 0.7216/0.7293/0.7350/0.7609 s. Small blocks help the CPU divide-and-conquer at this size, little impact on reduction.
- GPU-aware MPI 2x4 FP64: 0.6771 s (0.4063 reduction / 0.1037 tridiagonal / 0.1671 back), versus host-staged 0.7293. Keep as an explicit experimentally supported option. Residual 5.45e-15; orthogonality 2.36e-13; eigenvalue error versus ELPA2 1.66e-15.
- Caution: original baseline `comm` field sums both trials and excludes library-internal communication. Later executable resets it per trial. This is measured MPI call wall time, not isolated network time or bytes.

Conclusion: reduction is largest phase, but backtransform remains substantial; skinny 1x8 currently wins because of baseline communication choices, not proof it will win for blocked algorithms. ELPA2 already strongly suggests a two-stage path is worthwhile.

### E2: active trailing-submatrix GEMV (17:49–17:51 EDT)

Hypothesis: the baseline unnecessarily reads completed rows/columns in each GEMV. Restrict GEMV and fused rank-2 update to local suffixes of the active trailing submatrix while preserving the same algorithm.

- Snapshot `build/custom-eigh/active-eigh`; job **5735992** compares baseline and active versions on the same node, n=2048/4096, 2x4/4x2, FP64/FP32, GPU-aware MPI, two trials.
- Tiny login correctness n=31 passed; FP16 input/FP32 compute lowrank example exposes precision risk: original residual 1.96e-4 but quantized-problem residual 1.93e-7; minimum eigenvalue changes from +1.01e-7 to -1.12e-4. Quantized-input arithmetic is NOT a native FP16 eigensolver.
- Precision sweep submission initially rejected by debug QoS maximum concurrent submit limit while E1 was running. No job ID or allocation was created by that rejected submission; submit after active job completes.
- Completed job 5735992, frontier01560, 79 s, all cases passed. Paired n4096 2x4 FP64 reduction 1.3074→1.1689 s, total 2.3540→2.2196 s; FP32 reduction 0.9497→0.8858 s, total 1.8375→1.7660 s. 4x2 FP64 total 2.4976→2.3812 s, FP32 1.9380→1.8703 s. Keep: modest consistent win, larger at n4096 than n2048.

### E3: compact-WY distributed backtransform (17:55–17:58 EDT)

Hypothesis: accumulate consecutive Householders into I−VTVᵀ and apply through GEMM, replacing n reflector broadcasts/reductions and bandwidth-bound rank-1 updates by ~n/back_block blocked operations. Reflector panels remain row-distributed; only small T/Gram matrices replicated. GPU-aware panel broadcasts avoid host copies.

- `--back-block 1` retains unblocked control; candidate widths 16/32/64/128. T constructed in FP64 from a GPU distributed Gram matrix; GEMMs use requested FP64 or FP32.
- Tiny n31 b8 back_block13 tests passed FP64/FP32, diagonal and lowrank, deliberately crossing distribution-tile boundaries.
- Next job `wy` includes rectangular-grid correctness, the full precision sweep and n4096 timing/width comparisons.
- Submitted job **5736013** with immutable `wy-eigh` snapshot. Pending priority at 17:55; lowered its requested limit from 15 to 5 minutes (suite expected <3 min) to keep backfill request small.
- Completed successfully on frontier01560 in 2m10s. All 8 rectangular-grid cases, 16 precision cases and 13 performance configurations passed solver correctness. n4096 final warm trial:

| Back block | FP64 back / total (s) | FP32 back / total (s) |
|---|---:|---:|
| 1 | 0.5835 / 2.2396 | 0.4169 / 1.7594 |
| 16 | 0.2473 / 1.9160 | 0.2148 / 1.5685 |
| 32 | 0.1279 / 1.7885 | 0.1102 / 1.4516 |
| 64 | 0.0766 / 1.7419 | 0.0659 / 1.4150 |
| 128 | 0.0655 / 1.7263 | 0.0532 / 1.4003 |

- Keep compact-WY: ~8.9x FP64 and ~7.8x FP32 backtransform speedups. FP64 n4096 residual 7.81e-15 / orthogonality 4.88e-13; ELPA2 same job total 0.7809 s FP64 / 0.5526 s FP32. Reduction and CPU tridiagonal stage still dominate.
- Communication finding **revises E1's direction**: at n4096/back32, host-staged MPI won: 1.6369 s total / 0.0905 s back, versus GPU-aware 1.7885 / 0.1279 s. No universally preferred collective path; compare again at target sizes. Do not default to GPU-aware MPI merely because it sounds faster.
- Precision sweep n257 lowrank (true λmin≈1.001e-7): FP64 residual 1.66e-15 / smallest-eigenvalue relative error 1.32e-9; FP32 residual 3.94e-7 / λmin error 23.4%; FP16-rounded input residual 2.14e-4, λmin=−3.89e-5, **122 negative eigenvalues**; BF16 residual 1.68e-3, λmin=−3.65e-4, **123 negatives**. The quantized problems themselves solve to ~3.8e-7 residual.
- Well-conditioned random SPD n257: FP32 original residual 3.07e-7; FP16 input 1.67e-4; BF16 input 1.23e-3. Low precision is a task-dependent accuracy decision, not automatically invalid, but cannot preserve the low-rank SOAP-like small spectrum in this test.

### E4: delayed panel updates (prepared, not yet performance-validated)

Hypothesis: delay a panel's rank-2 updates and apply A₂₂←A₂₂−[V W][W V]ᵀ through GPU GEMM. Correct each panel column and Av for the still-pending low-rank updates. This is still **one-stage tridiagonalization**, not dense-to-band. It retains the sequential GEMV chain and adds a small coefficient reduction; measure whether saved bandwidth outweighs that latency.

- Option `--panel`; 1 is the direct update control. Panel matrices use both row- and column-oriented vector layouts, independently respecting rectangular grids. Tiny n31 b8 panel7 back_block13 FP64/FP32/lowrank/diagonal passed.
- Snapshot `build/custom-eigh/panel-eigh`; `panel` suite prepared for after E3.
- Link provenance check: `LD_DEBUG=bindings` confirms actual `PDSTEDC`, BLACS and CPU LAPACK/BLAS resolve to **Cray LibSci 26.03**, because the Frontier compiler wrapper links it before the local reference libraries. Source review used reference ScaLAPACK; measured backend is LibSci's compatible distributed CPU implementation. ELPA's host BLAS also resolves to LibSci. No claim that reference OpenBLAS supplied measured CPU flops.
- Submitted job **5736055**, 5-minute one-node limit, includes rectangular-grid n97 tests, n4096 widths1/8/16/32/64 FP64/FP32, and paired n8192 direct/panel32 with ELPA2 eigenvalue reference.
