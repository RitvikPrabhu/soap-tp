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

### E4: delayed panel updates (slower; not selected)

Hypothesis: delay a panel's rank-2 updates and apply A₂₂←A₂₂−[V W][W V]ᵀ through GPU GEMM. Correct each panel column and Av for the still-pending low-rank updates. This is still **one-stage tridiagonalization**, not dense-to-band. It retains the sequential GEMV chain and adds a small coefficient reduction; measure whether saved bandwidth outweighs that latency.

- Option `--panel`; 1 is the direct update control. Panel matrices use both row- and column-oriented vector layouts, independently respecting rectangular grids. Tiny n31 b8 panel7 back_block13 FP64/FP32/lowrank/diagonal passed.
- Snapshot `build/custom-eigh/panel-eigh`; `panel` suite prepared for after E3.
- Link provenance check: `LD_DEBUG=bindings` confirms actual `PDSTEDC`, BLACS and CPU LAPACK/BLAS resolve to **Cray LibSci 26.03**, because the Frontier compiler wrapper links it before the local reference libraries. Source review used reference ScaLAPACK; measured backend is LibSci's compatible distributed CPU implementation. ELPA's host BLAS also resolves to LibSci. No claim that reference OpenBLAS supplied measured CPU flops.
- Submitted job **5736055**, 5-minute one-node limit, includes rectangular-grid n97 tests, n4096 widths1/8/16/32/64 FP64/FP32, and paired n8192 direct/panel32 with ELPA2 eigenvalue reference.
- Completed 18:04 EDT, frontier01560. All correctness checks passed. n4096 FP64 panel1/8/16/32/64 total1.787/1.903/1.941/1.979/2.117s; FP321.456/1.593/1.617/1.667/1.777s. At n8192 panel1/32 total6.539/6.750s, reduction3.614/3.825s. FP64 residual1.11e-14/1.09e-14; eigenvalue agreement with ELPA2 2.79e-15/2.27e-15.
- Conclusion: select panel1. Delayed updates do not remove the sequential GEMV dependency and add coefficient reductions; GEMM alone is insufficient. Retain this explicitly non-default experimental control as evidence distinguishing this approach from genuine two-stage reduction, not as a performance improvement.

### E5: tridiagonal component, transport, native-half bulk

Hypotheses: ELPA's public GPU tridiagonal component may beat LibSci PDSTEDC without surrendering custom dense stages; RCCL may improve device collectives; native FP16 GEMM/FP32 accumulation may be the lowest useful bulk arithmetic.

- Job **5736114**, 6-minute one-node limit, immutable `half-eigh` binary. Includes all rectangular ELPA-tridiagonal cases, RCCL rectangles, paired transports/components n4096, n8192 ELPA component plus full ELPA2 eigenvalue reference, native-half random/lowrank checks.
- ELPA's component API uses host d/e and distributed local host Q; internal GPU work is ELPA-owned. Full-solver GPU-pointer API differs. FP64 tridiagonal solve remains in FP32 dense mode.
- Native-half prototype converts operands for GEMM through rocBLAS GEMM_EX, retains FP32 state/accumulation. Tiny n31 failed strict orthogonality (0.00535), so this is an accuracy experiment, not a working low-precision solver. Batch explicitly records such failures rather than weakening acceptance.
- Job5736114 completed 18:09 EDT, frontier00002. All ELPA-component and RCCL correctness cases passed; all four naive-half cases failed, as expected. n257 lowrank half/panel16 even produced120 negative eigenvalues.

| n4096 FP64 transport | LibSci tri / total(s) | ELPA tri / total(s) |
|---|---:|---:|
| Host MPI | .4669 / 1.6231 | .1213 / 1.2822 |
| Device MPI | .4740 / 1.7322 | .1180 / 1.3770 |
| Synchronized RCCL | .4674 / 1.6540 | .1024 / 1.3061 |

- n8192 host MPI with ELPA tridiagonal: total4.0769, reduction3.3114, tri.4858, back.2743s; deviceMPI4.2689/3.5894/.4942/.1800s. FP64 residual1.11e-14, orth9.67e-13, eigenvalue error2.79e-15 vsELPA2. Select ELPA component for performance mode, retain independent LibSci baseline. This is **not** wrapping full ELPA eigh: both dense stages are custom.
- Source check confirms `elpa1/elpa_solve_tridi_impl_public.F90` stages host d/e/localQ to GPUs. `solve_tridi/solve_tridi_single_problem_template.F90` uses ROCm rocSOLVER STEDC for leaves when available; recursive GPU merges combine subproblems. Dense Z remains distributed.

### E6: scaled host norm (candidate)

Hypothesis: per-element `hypot` in replicated Householder generation is expensive. Replace it by a two-pass max-scaled FP64 norm, retaining overflow-safe scaling and one final `hypot` with alpha. Tiny n31 lowrank FP64 and random FP32 passed. Snapshot `norm-eigh`.

- Paired job **5736133**, frontier00642: n4096 FP64 hostMPI/ELPAtri reduction1.0852→.9420s, total1.2890→1.1429s. n8192 reduction3.3076→2.6659s, total4.0703→3.4223s. Keep scaled norm; numerical checks passed.

### E7: scaled split-FP16 bulk arithmetic

Hypothesis: three FP16-input/FP32-accumulate GEMMs for AhiBhi+AhiBlo+AloBhi may preserve FP32-like accuracy while using faster matrix instructions. State and reduction GEMV remain FP32; this is **not** an FP16 eigensolver.

- Unscaled split failed tiny n31 (orth.00208). [AMD documents MI200 FP16 MFMA flushing subnormals](https://rocm.docs.amd.com/projects/rocBLAS/en/latest/conceptual/rocblas-design-notes.html). Corrected representation: high is rounded FP16 normal-or-zero, low is4096×residual; cross-term alpha is divided by4096. This prevents low corrections being represented directly as FP16 subnormals. Low×low is omitted, so this remains an approximation.
- Tiny n31 random residual1.71e-7/orth2.21e-6; lowrank2.94e-7/2.39e-6. Job5736133 distributed n257 grid4x2 random/lowrank/indefinite all pass. Lowrank residual3.44e-7/orth8.34e-6, but smallest-eigenvalue error30.6% persists from FP32 reduction/input.
- n8192 passes: split residual1.34e-6/orth.000155, versus FP32 1.37e-6/.000264. Backtransform .1892 vs .1506s; total2.7468 vs2.7034s. Not selected for speed. Retained as the lowest tested native bulk arithmetic with FP32-like normwise accuracy; not a guarantee for small eigenvalues or arbitrary dynamic range.

### E8: synchronization and disjoint row-vector collectives (candidates)

- `async-eigh`: remove redundant device-wide sync before blocking host copies; RCCL reductions/broadcasts enqueue on the same default HIP stream as rocBLAS. Host reads and phase boundaries synchronize as required. **No claim of algorithmic communication/computation overlap**; this first tests eliminating avoidable synchronization. `comm` becomes host call/enqueue wall time for RCCL, not device communication duration.
- `gather-eigh`: replace zero-padded length-n MPI_SUM across process rows by an Allgatherv of disjoint local row pieces, then unpack cyclic blocks. This is only O(n) vector replication, never a dense gather. Used for both Householder columns and w vectors. Tiny local ELPA/RCCL passed; rectangular and performance checks in `transport` suite must precede selection.
- Job **5736154** completed18:20 EDT, frontier02829, all cases passed. n8192 FP64 paired total seconds:

| Binary | Host MPI | Device MPI | RCCL |
|---|---:|---:|---:|
| Scaled norm, synchronized | 3.4645 | 3.6800 | 3.4402 |
| Remove redundant sync | 3.3610 | 3.6732 | 3.2435 |
| +row-vector Allgatherv | 3.2376 | 3.5250 | **3.1420** |

- Keep both changes. Best reduction2.5166/ELPAtri.4897/back.1301s, residual9.66e-15/orth8.11e-13. Default-stream RCCL is ordered with rocBLAS; this removes synchronization but does not implement look-ahead overlap.

### E9: tuning, SOAP-relevant subspace validation (submitted)

- Job **5736176**, five-minute limit, immutable `subspace-eigh`, varies matrix block32/64/128/256 and backtransform32/64/128/256 at8192; adds low-precision n257 top-eight subspace metrics and ELPA1/2 internal stage profiles.
- SOAP caller inspection: `soap.py` initializes eigenvectors and discards eigenvalues; later refreshes use QR. FP32 input is promoted to FP64 in the existing ELPA path and results cast back. There is no inverse eigenvalue in this initializer. Therefore loss of tiny eigenvalues alone does **not** prove an optimizer failure; basis/subspace and training/update errors matter.
- Validation now reports `||(I-RRᵀ)Qtop||F/||Qtop||F` for eight dominant reference vectors and the normalized boundary eigengap; suppresses this metric when the reference boundary is degenerate. Collects only n×8 thin panels, not the full eigenvector matrix. Tiny n31 FP16-rounded lowrank top-subspace8.50e-5 despite negative eigenvalues; clustered boundary correctly reports skipped. Full SOAP training/update comparison is not yet performed.

### E10: genuine custom dense-to-band reduction (planned)

Hypothesis: factor independent tall panels using distributed Householder QR, then apply two-sided compact-WY updates through GEMM, eliminating one full trailing-matrix GEMV per reflector. Keep full matrices distributed and preserve first-stage reflectors separately. Initially reuse only ELPA's banded-input mode for distributed band→tridiagonal/solve/second-stage backtransform, then our own first-stage backtransform. ELPA source explicitly skips its dense→band and last backtransform when `bandwidth` is set; this path will be runtime-validated, not assumed correct from source alone.
