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

### E9: tuning, SOAP-relevant subspace validation

- Job **5736176**, five-minute limit, immutable `subspace-eigh`, varies matrix block32/64/128/256 and backtransform32/64/128/256 at8192; adds low-precision n257 top-eight subspace metrics and ELPA1/2 internal stage profiles.
- SOAP caller inspection: `soap.py` initializes eigenvectors and discards eigenvalues; later refreshes use QR. FP32 input is promoted to FP64 in the existing ELPA path and results cast back. There is no inverse eigenvalue in this initializer. Therefore loss of tiny eigenvalues alone does **not** prove an optimizer failure; basis/subspace and training/update errors matter.
- Validation now reports `||(I-RRᵀ)Qtop||F/||Qtop||F` for eight dominant reference vectors and the normalized boundary eigengap; suppresses this metric when the reference boundary is degenerate. Collects only n×8 thin panels, not the full eigenvector matrix. Tiny n31 FP16-rounded lowrank top-subspace8.50e-5 despite negative eigenvalues; clustered boundary correctly reports skipped. Full SOAP training/update comparison is not yet performed.
- Completed18:26, frontier02826, all checks passed. n8192 FP64 matrixblock32/64/128/256 total3.167/3.171/3.156/3.141s: small differences, no material block winner. Back32/64/128/256 phase.220/.140/.130/.302s. Keep128; FP32 back64 and128 effectively tied(.0709/.0719s), total2.492/2.476s.
- n257 top-eight subspace residuals, reference boundary gaps random5.15e-4/lowrank.909:

| Dense/input precision | Random SPD | Lowrank SPD |
|---|---:|---:|
| FP64 | 1.57e-13 | 1.21e-15 |
| FP32 | 3.17e-5 | 1.19e-7 |
| FP16-rounded / FP32 solve | .01569 | 3.88e-5 |
| BF16-rounded / FP32 solve | .04153 | 3.34e-4 |

- Dominant-subspace accuracy is eigengap-dependent: FP16-rounded lowrank has many negative small eigenvalues yet good dominant subspace; the closer random-SPD boundary rotates much more. No universal minimum SOAP precision is established without update/training metrics.
- Profiled ELPA2 n8192 total3.785s includes timer/output overhead, so not an uninstrumented competitive result. Internal tridi→band backtransform~1.525s is substantial; stage totals must be compared rather than assuming any two-stage solver wins. Full uninstrumented controls remain necessary.

### E10: genuine custom dense-to-band reduction (submitted)

Hypothesis: factor independent tall panels using distributed Householder QR, then apply two-sided compact-WY updates through GEMM, eliminating one full trailing-matrix GEMV per reflector. Keep full matrices distributed and preserve first-stage reflectors separately. Initially reuse only ELPA's banded-input mode for distributed band→tridiagonal/solve/second-stage backtransform, then our own first-stage backtransform. ELPA source explicitly skips its dense→band and last backtransform when `bandwidth` is set; this path will be runtime-validated, not assumed correct from source alone.

- Implemented `--band b` requiring band==distributionblock initially. Custom CPU distributed tall-panel QR uses scaled FP64 scalar formation and short column-communicator reductions. GPU forms V/T/W: W=AVT−½VTᵀ(VᵀAVT); two rocBLAS GEMMs update A←A−VWᵀ−WVᵀ. Store first-stage Householders in a separate **distributed** GPU matrix; only O(n×band) panels are replicated transiently. No full dense gather.
- `tridiag` timing now means the complete ELPA banded-input pipeline (CPU distributed band→tridiagonal, distributed eigensolve, Q2 backtransform), not just eigensolution; `back` is our Q1 backtransform. Use `--elpa-times 1` for internal separation. FP32 band mode uses ELPA's FP32 band/tri path, unlike one-stage's FP64 tridiagonal component.
- Tiny login31 b8/back13 FP64 random/lowrank/diagonal and FP32 random passed; FP64 residual1.16e-15/orth8.68e-15 and FP32 5.51e-7/3.89e-6. Includes partial last panel. Source-to-band correctness is checked indirectly by full original AQ−QΛ and reference spectrum, not by inspecting triangular zeros alone.
- Job **5736208**, six-minute one-node limit, immutable `band-eigh`. Distributed97 rectangles/families precede target4096/8192 band32/64/128 versus one-stage controls, both precisions; all large cases use full ELPA2 reference plus thin-subspace validation. Last case tests split-FP16 bulk on band64 and records strict-check outcome.
- Job timed out at6m19s on frontier03617. **All34 executed cases passed**; the final split-half step was denied by Slurm and never ran. Its script failure marker is not a numerical failure. n8192 FP64 one-stage3.145s versus band32 5.965s (reduction2.282/band pipeline3.544/back.139), band64 6.687s. FP32 band32/64/128 totals5.610/5.327/7.984s. The first two-stage implementation loses overall despite Level-3 updates. No automatic selection of this path.

### E11: use LibSci BLAS for narrow CPU-panel QR

Hypothesis: nested CPU dot/rank-one loops, rather than GPU two-sided updates, dominate the first custom dense-to-band stage. Replace just these loops with LibSci DGEMV/DGER; promote the narrow panel to FP64 in FP32 mode as well, retaining requested GPU state precision. Scalar norm formation was already FP64.

- Job **5736310**,8-minute one-node request, snapshot `blas-panel-eigh`, paired against `band-eigh` on the same node at4096/8192, then widths16/64, uninstrumented full ELPA1/2 controls, split-half band32. Initial8192 FP64 band32 reduction2.313→1.012s, total5.982→4.709s; CPU QR.223s. This is a real improvement but does not yet beat one-stage. Full outcomes pending.
- Completed5m45s on frontier03585; all25 cases passed. Keep BLAS panel QR. n4096 band32 FP64 total1.279→.936s and FP32 1.030→.708s. n8192 band64 is better than32: FP64 total3.962s(reduction1.010/pipeline2.817/back.134), FP32 2.948s(.759/2.116/.072). Band16 loses6.390/5.096s. FP32 promotion also changes the band pipeline's numerical input, so not all FP32 total gain can be attributed solely to faster CPU BLAS.
- Uninstrumented full ELPA1 n8192 FP64/FP32 **2.137/1.607s**, ELPA2 **3.717/2.643s**. Thus current custom one-stage beats ELPA2 at this configuration, but **does not beat ELPA1**. Split-half band32 passes(residual6.53e-6/orth.000484) but total3.587 versus native FP32 3.504s; not selected for speed.

### E12: FP32 tridiagonal component and target-size scaling (prepared)

Hypothesis: FP32 dense reduction need not pay for FP64 distributed tridiagonal eigensolution. Add explicit `--tridiag elpa32` using the public FP32 component, while preserving FP64 `elpa` and independent LibSci controls. FP64 scalar generation remains deliberate. Test reference eigenvalues/residual/orthogonality and dominant subspaces, not just timings.

- Added `managed_gpu_peak_mib` instrumentation: maximum over ranks of this driver's live `Device` buffers during each solver trial. Includes A/Q and our temporaries; **excludes library/RCCL internal allocations and all host memory**, so it is not a total-device-memory high-water mark. Reset before each trial; reduce outside timed solve. Does not affect solver algorithms.
- Prepared `scaling` suite: tiny FP32-component rectangular checks, n8192 grids1x1/1x2/2x2/1x8/2x4/4x2/8x1 in FP64 and FP32, paired FP64/FP32 tridiagonal component at2x4, split-half, single-GCD rocSOLVER controls. Three trials per performance configuration; full ELPA2 spectrum/subspace reference at2x4 and4x2, distributed residual/orthogonality everywhere. No dense gather.
- Tiny login31 random/lowrank/indefinite/clustered/diagonal FP32-component checks passed after adding the new backend to argument validation. Lowrank smallest-eigenvalue relative error.745 despite residual2.69e-7: do not select FP32 tridiagonal work when accurate tiny eigenvalues matter.
- Job **5736348**, snapshot `tri32-eigh`, completed6m21s on frontier02306; all32 cases passed. n8192 final warm timings(s):

| Grid/GCDs | FP64 total / reduction / tri / back | FP32 total / reduction / tri / back |
|---|---:|---:|
|1x1/1|7.495 /6.124 /.969 /.376|4.402 /3.519 /.696 /.169|
|1x2/2|5.201 /4.193 /.766 /.224|3.324 /2.640 /.545 /.126|
|2x2/4|3.500 /2.884 /.457 /.150|2.409 /2.016 /.300 /.085|
|1x8/8|3.378 /2.306 /.934 /.132|2.347 /1.750 /.496 /.098|
|2x4/8|3.140 /2.526 /.479 /.129|2.300 /1.946 /.279 /.072|
|4x2/8|**2.938** /2.538 /.285 /.109|**2.112** /1.842 /.197 /.069|
|8x1/8|3.506 /3.180 /.230 /.090|2.180 /1.950 /.157 /.069|

- FP64 speedup1→best8 is2.55x, FP32 2.08x; substantial reduction/communication floor remains. Rectangular4x2 wins overall, although1x8 has fastest FP64 reduction. FP64 residual9.3–9.7e-15/orth7.2–9.6e-13; FP32 residual1.4–1.6e-6/orth.00033–.00047.
- Paired2x4 FP32 total2.475(FP64 tri)→2.300(FP32 tri), tri.460→.279; select FP32 component for normwise FP32 mode. Split-half adds.035s at8192(back.106 versus.072), still no speed win. Eight-GCD managed peak4x2 FP64/FP32 138.59/69.30MiB, excludes library internals.
- Single-GCD rocSOLVER8192 FP64/FP32 **2.838/1.491s**, faster than this custom eight-GCD baseline. No dense gather is performed by this local control. This reinforces that custom solver overhead, not nominal GPU arithmetic throughput alone, must improve before claiming a practical win.

### E13: SOAP basis precision sensitivity (CPU experiment)

- `soap_precision.py`, command in its docstring, seeds42/43/44,96×64,20 frozen-basis updates, condition numbers100/1e6/1e9. Uses actual Adam primitive and SOAP epsilon/projection conventions; all update arithmetic FP64 isolates basis changes. Reference solves the same FP32 preconditioner promoted to FP64, matching existing ELPA input precision. FP16/BF16 are input-rounding experiments, not low-precision eigensolver arithmetic. Raw45 rows in `soap_precision.csv`.
- Signed/permuted-basis control maximum update error6.52e-13; comparisons are sign/permutation invariant. At condition100, final-update FP32 error7.3–8.3e-6, FP16-rounded.0032–.0039, BF16-rounded.0235–.0306. First update can be much more sensitive: FP32 max.0030, FP16.101, BF16.292.
- At condition1e6, final-update FP32 error.0678–.0731, FP16.423–.445, BF16.677–.680. At1e9 FP32.311–.354, FP16.875–.880, BF16.989–1.019. Near-degenerate/poorly conditioned bases can materially alter coordinate-wise Adam despite small eigensolver residuals. This is sensitivity evidence, **not** a training-loss regression, custom-solver comparison, or full distributed SOAP validation. No universal lowest SOAP precision is established.

### E14: GPU-resident Householder vector candidate (prepared)

Hypothesis: remove the per-reflector host vector transfers/replicated CPU scalar loops that remain in the one-stage reduction. Opt-in `--device-vectors 1` keeps scaled FP64 norms and scalar reductions on GPUs, uses RCCL row broadcasts/column reductions, padded vector AllGather to transpose block-cyclic row/column layouts, GPU W formation, fused rank2 updates, D2D reflector storage. Only O(n) d/e/tau return to host at reduction end. Existing distributed tridiagonal/backtransform stages are shared; baseline remains default.

- Snapshot `device-vectors-eigh`, SHA2560942e84ec8e17588499ca4873928e7ba744f4dd793917f4bbaf213fcf10b0345. Tiny login31 random/lowrank/diagonal/indefinite FP64/32 plus2boundary and31FP32 elpa32 passed. Representative FP64 residual6.54e-16/orth6.22e-15; FP32 residual1.71e-7/orth1.92e-6. No multirank or performance claim yet. Requires RCCL/GPU-MPI/panel1/custom and rejects band mode.
- Prepared `device-active-eigh` with active-suffix rank2 launch instead of a full local-matrix launch, and complete direct RCCL enqueue accounting. Both tiny precisions pass. Job **5736421**,15-minute one-node request, snapshotSHA16526d14aaa585d85be65b17ab93b9eba5335c1df9d61aaa8dd97afb1bb7fdee, running frontier03167. All24 tiny rectangular/family cases passed before performance. Includes pairedCPU/GPU8192, old full-launch GPU control, and16384CPU/GPU/ELPA1/ELPA2/band64. Fixed snapshots are immutable.
- First8192 2x4 FP64 GPU-vector result **loses**: total6.962/reduction6.322s versus CPU-vector3.149/2.530. Residual2.72e-15/orth7.59e-13 still correct. Do not select this GPU path merely because it removes host copies. Remaining outcomes pending.
- Completed job5736421 in14m20s; all43 executed cases passed. At8192/4x2, CPU-vector FP64/32 totals2.917/2.099s versus GPU-vector5.041/4.105s. Active-only GPU update improves2x4 FP64 reduction6.528→6.322s, but does not explain most of the gap. No reason to select the initial GPU-vector candidate.

| n16384, eight GCDs, 4x2 | FP64 total (s) | FP32 total (s) |
|---|---:|---:|
| Custom one-stage, CPU vectors | 11.553 | 7.434 |
| Custom one-stage, initial GPU vectors | 17.802 | 14.503 |
| Full ELPA1 | **6.177** | **4.571** |
| Full ELPA2 | 19.403 | 14.071 |
| Custom band64 + ELPA banded component | 20.263 | 14.860 |

- CPU-vector16384 reduction/tri/back FP64=10.045/1.074/.409s, FP32=6.317/.886/.213s. Custom one-stage GPU-vector FP64 reference eigenvalue error2.07e-15, residual3.78e-15, orthogonality1.35e-12; FP32 reference error6.04e-7, residual1.99e-6, orthogonality.000659. CPU-vector residuals1.35e-14/1.99e-6. The banded pipeline, not our GEMM-rich dense-to-band stage, dominates the two-stage loss at this size (16.431/12.328s).

### E15: eliminate serial GPU scalar lookup (next candidate)

Hypothesis: Stage1's thread0 scans all local rows to locate just alpha and the diagonal. This introduces an avoidable serial O(n/pr) GPU dependency chain per reflector. Rows are sorted, so one lower-bound lookup suffices for both k and k+1; test a logarithmic lookup against the frozen active-vector binary. This is a hypothesis about the measured slowdown, not an established complete cause. Also investigate stage-specific tridiagonal grids: measured8192 tri.934s on1x8 versus.230s on8x1 suggests distributed host-Z redistribution could pay for itself while keeping the faster reduction grid.
- Lookup implemented in commit `cfd5c76`, snapshot `device-indexed-eigh` SHA37d517ade4bc8545b21cc9a80a7bb9a8be4017ff3925dcb4f5c3c8a22db795e9. Tiny31/2 FP64/32 pass with the same metrics as the frozen active-vector candidate; unequal ownership remains for batch validation.
- Independent `--tri-pr` option implemented using a persistent alternate Grid, ELPA tridiagonal solve on that grid, then distributed LibSci PDGEMR2D of host-local Z into the main layout. Only Z changes distribution; reflector A and backtransform stay on the main grid. No dense gather. `retile` is a subset of `tridiag`, including packing and communication. Extra grid setup is outside timed trials; per-solve redistribution/allocations are inside.
- Snapshot `retile-eigh` SHA2a62c0ba53eb80a6757632385113753b3f67bedc5f5f79baeee981cd63e6400b includes lookup+stage-grid support. Tiny single-rank31 ELPA FP64/32 paths pass, but those use identical grids and do **not** validate PDGEMR2D. Prepared `indexed` batch suite first gates97 unequal-ownership main grids1x8/2x4/4x2/8x1→tri8x1 for CPU/GPU vectors and FP64/32, then paired lookup timings, all8Ggrid GPU/ELPA1 controls, and1x8→8x1 performance.
- Job5736491 running onfrontier03167,12-minute limit. All16 uneven rectangular n97 redistribution checks passed. Paired8192/2x4 FP64 reduction6.305→3.295s after replacing only the serial scalar scan; this confirms the scan was a substantial bottleneck, not the whole remaining performance problem. Indexed4x2 GPU-vector total3.554s still loses the CPU-vector baseline. Complete grid/FP32/stage-grid outcomes pending.
- Completed8m10s, all40 cases passed. The following are **warm medians** (two post-warmup trials), not final/minimum timings:

| n8192 eight GCDs | FP64 (s) | FP32 (s) |
|---|---:|---:|
| GPU indexed vectors, main1x8 / tri1x8 | 3.111 | 2.314 |
| GPU indexed vectors, main1x8 / tri8x1 | **2.543** | 2.141 |
| CPU vectors, main1x8 / tri1x8 | 3.412 | 2.359 |
| CPU vectors, main1x8 / tri8x1 | 2.842 | 2.139 |
| Full ELPA1,4x2 | **1.952** | **1.563** |

- Keep the stage-grid option, especially for skinny main grids; it is not universally profitable. GPU1x8→8x1 final redistribution.0646/.0687s, included in tridiag.3397/.3304s. Indexed GPU4x2 and8x1 still lose CPU-vector controls. FP64 best custom8192 now uses GPU vectors/main1x8/tri8x1; FP32 old CPU4x2 remains slightly faster (~2.11s). FullELPA1 is still faster than either. No blanket claim that GPU residency or square grids win.

### E16: pack rank-local GPU vectors (candidate)

Hypothesis: the faster CPU-vector baseline still uploads full replicated V/W to every GPU. For panel1, pack just each rank's row and column pieces into contiguous V/W buffers. V now needs one H2D call rather than two, and the rank2 kernel uses direct local indices. Total uploaded elements per nontrivial step change from2n+nc to2nr+2nc; this is a40% reduction on4x2, but a6% increase on1x8 (with fewer copy calls). CPU scalar formation and communication are unchanged. Panel controls and the shared backtransform are preserved.

- Snapshot `packed-eigh` SHA530960a08f7aa43481ecf5f6158371ecc2a141e9f35f92aa63564f6ab0f4dc5d. Tiny31 random/lowrank/indefinite/diagonal in both precisions and delayed-panel/unblocked control pass. FP64 random residual7.58e-16/orth6.16e-15; FP32 random1.71e-7/2.12e-6. These are login correctness checks, not performance evidence.
- Prepared `packed` batch suite gates97 grids1/2/4/8G and all transports/backtransform controls; then pairs `retile-eigh` versus `packed-eigh` at8192 on1x8/2x4/4x2 with tri8x1. Separate rank0-only rocprofv3 kernel/copy traces compare CPU/GPU vector paths. Instrumented totals must not be used as competitive measurements.
- Job5736545 completed6m48s onfrontier00194; all34 cases passed, including two instrumented cases. Keep packing: paired8192 warm medians below use tri8x1 for both binaries. Main4x2's extra redistribution is not profitable; do not infer this is the best4x2 total.

| Main grid | FP64 before→after (s) | FP32 before→after (s) |
|---|---:|---:|
|1x8|2.855→2.811|2.175→2.060|
|2x4|3.078→2.930|2.308→2.179|
|4x2|3.121→3.060|2.291→2.163|

- Rank0-only rocprofv3 trace at8192/4x2/FP64, whole run including validation/reference work: CPU-vector GEMV.599s, rank2.407s, RCCL kernels.141s; GPU-vector GEMV.631s, rank2.370s, RCCL kernels**1.426s**, scalar kernels.291s. There are8,446 versus57,593 RCCL kernels. Kernel durations are instrumented GPU intervals, include waiting, and are not isolated network time or an additive decomposition of uninstrumented solver wall time. Raw ROCpd databases: `output/custom-eigh/profile-5736545-v{0,1}/rank0_results.db`. GPU collective latency, not just host copies, is material.

### E17: use the custom GPU solver's bases in SOAP-style updates (prepared)

Hypothesis: low solver residual alone does not establish acceptable SOAP updates. Extend the same96×64,20-update,three-seed/condition experiment to bases exported by our actual custom GPU implementation, including split-half bulk arithmetic. Original preconditioners remain FP32 promoted toFP64, as in the existing SOAP path; update arithmetic remains FP64 to isolate basis changes.

- Tiny binary input/export paths are explicitly limited to2≤n≤512. Normal-size A/Q stay distributed. FP64 andFP32 independent3×3 roundtrip tests compare exported AQ−QΛ, QᵀQ and eigenvalues to NumPy: residual4.10e-16/2.30e-7, orth8.85e-16/2.08e-7. Existing45-row CPU study reproduces exactly after adding the optional GPU backend.
- Snapshot `basis-eigh` SHA57ae5f1dfef02daddb9afaaea42780942beb733d0eb17f511d02e0213e06c8ae; `soap` suite prepared for one compute-node GCD, not yet submitted. Six variants yield54 rows. This remains a frozen-basis update experiment, not training-loss validation or a distributed SOAP backend.
- First attempt job5736707 TIMEOUT5m20s onfrontier09032, **no GPU solves/results**: PyTorch import was still reading modules over the shared home filesystem. Process/strace inspection showed repeated filesystem metadata reads under `torch/distributed`, not an eigensolver deadlock. Increasing the running time limit was denied by Slurm; no allocation change occurred. Removed the compute-node Python orchestrator. New workflow: login-CPU `--prepare` writes18 tiny covariance inputs (~1MiB); a pure-shell batch runs90 compiled GPU solves; login-CPU `--bases` computes54 update-study rows, reusing FP64 bases for the sign/permutation control. Prepared corpus in `output/custom-eigh/soap-input`; new job not yet submitted.

### E18: remove rank2 index division (candidate)

Hypothesis: rank2 is the second-largest custom compute kernel in the trace. Its flattened64-bit runtime quotient/remainder can be replaced by a128×2 thread block and a2D row/column launch. This preserves numerical operations and rectangular ownership; tail masks independently bound rows and columns. It also removes the obsolete global-vector rank2 kernel superseded by E16.

- Snapshot `twod-eigh` SHAe7984d83d8509a4ec0601a0c3cb8d250306e533c1bbf6bb78868b6528f2428b6. Tiny2/31, FP64/32, CPU/GPU vectors all pass; FP32 random residual1.71e-7/orth2.12e-6. No performance claim yet.
- Job5736758 submitted12min, suite `twod`:16 uneven rectangular checks, paired8192 CPU4x2/GPU1x8→tri8x1 against `basis-eigh`, then16384 updated candidates in both precisions with full ELPA2 reference. Latest source changes affect only rank2 indexing; frozen precision-study binary remains unchanged.
- Completed9m18s onfrontier09155; all28 cases passed. Paired8192 warm medians before→after: CPU4x2 FP64 2.819→2.822s, FP32 1.987→1.988s; GPU1x8→tri8x1 FP64 2.534→2.543s, FP32 2.136→2.105s. This is essentially neutral, not a claimed speedup; retain direct rectangular indexing without claiming its theoretical instruction savings improved performance.
- Updated16384 warm medians: CPU4x2 FP64/32 11.173/7.023s; GPU1x8→tri8x1 10.643/7.456s. GPU FP64 residual3.79e-15, orth1.79e-12, spectrum error1.65e-15. CPU FP32 residual1.99e-6, orth6.60e-4, spectrum error6.05e-7. These are independently validated against distributed FP64 ELPA2, not inferred from smaller tests.

### E19: gather scaled norm statistics once (candidate)

Hypothesis: the GPU-resident path pays collective and kernel latency for separate maximum and scaled-sum reductions. Compute a local max/SSQ pair, gather four scalars per process row (scale, SSQ, unique alpha, unique diagonal), and merge robustly on the GPU. This removes one collective and one kernel per reflector without using an unsafe unscaled norm. CPU ownership arithmetic replaces GPU scalar searches; the dot kernel visits only active rows.

- Snapshot `fused-eigh` SHA4628d316d3b5385a7cc162c5c5b9ecd0fd9da09800f7e32510179a7adff4c5b7. Tiny31 random/lowrank/diagonal/indefinite FP64/32 all pass; random residual6.51e-16/1.71e-7 and orthogonality6.61e-15/2.12e-6. Not yet a measured performance selection.
- `fused` suite gates uneven rectangular ownership and four matrix families, then pairs the frozen `twod-eigh` control with this candidate on three8192 grids and tests16384/main1x8→tri8x1. The allgather increases scalar payload slightly, so latency reduction is a hypothesis to measure rather than a guaranteed win.
- Job5736836 COMPLETED6m57 onfrontier01244; all26 cases passed. **Keep fusion**: paired8192 warm medians FP64/32 before→after:1x8→tri8x1 2.551→2.219 /2.112→1.770s;2x4 3.919→3.458 /3.172→2.732s;4x2 3.559→3.185 /2.633→2.167s. At16384/main1x8→tri8x1 warm9.429/6.257s; final FP64 phases7.280/1.426/.697s and FP32 4.819/.911/.441s. FP64 residual3.79e-15/orth1.79e-12/spectrum1.65e-15; FP32 1.98e-6/.000898/5.79e-7. Norm fusion changes reduction, not the dominant asymptotic GEMV work.

### E20: fuse row/column vector transformations (candidate)

Hypothesis: form or correct both row- and column-oriented copies of each vector in one HIP kernel after gathering the raw vector, rather than launching formation, unpacking, and initialization separately. This removes three launches per reflector, including W zeroing, while preserving FP32 rounding of W before the rank2 update. Padding is never interpreted as a valid global element. It does not alter communication payload or the scalar algorithm.

- Snapshot `vector-fusion-eigh` SHAffce00b7e454aed11ae6804e2921f26f8b5364882c465a7d11684137841a16ea. Tiny2/31 random/diagonal/lowrank/indefinite in FP64/32 pass. `vector-fusion` suite will compare this against frozen `fused-eigh`; not yet selected by multirank/performance evidence.
- Staged GPU SOAP-basis job **5736815** submitted8min using frozen `basis-eigh`;90 compiled singleton solves, no compute-node Torch imports. Results still pending.

### E21: separate solve precision from basis storage precision

Job5736815 COMPLETED1m23 onfrontier01375; all90 GPU solves passed. Login-CPU analysis of the exported bases produces81 rows in `soap_gpu_precision.csv`:20 frozen-basis updates, seeds42/43/44,96×64 gradients, actual Adam primitive. No training-loss claim. Signed/permuted controls agree within1.61e-13. Median final update errors across three seeds:

| Basis source/storage | condition100 | condition1e6 | condition1e9 |
|---|---:|---:|---:|
| Custom FP32 solve |6.22e-6|.07798|.32797|
| Split-half bulk, FP32 state |6.24e-6|.07798|.32798|
| FP16-rounded input / FP32 solve |.00382|.43509|.87806|
| BF16-rounded input / FP32 solve |.02935|.67733|.99348|
| **Custom FP64 solve, FP32 basis storage** |**8.89e-8**|**2.76e-6**|**4.73e-5**|
| Custom FP64 solve, FP16 basis storage |.000710|.02392|.29772|
| Custom FP64 solve, BF16 basis storage |.005625|.17792|.74815|

FP32 basis storage after an FP64 solve is materially safer here than solving in FP32; this directly supports the existing SOAP cast-after-solve boundary rather than assuming all stages need FP64. Maximum error over any of the20 updates for FP64→FP32 storage reaches1.54e-4 at condition1e9. FP16 storage reaches.379 and BF16.808 there. These storage controls round exported bases and compute updates in FP64, isolating storage error; they are not native low-precision GPU projection benchmarks. FP32 input matrices are the same in every variant, and their smallest eigenvalues may already be negative at condition1e9.

### E22: remove reflector column copy round-trips (candidate)

Hypothesis: broadcast the owner GPU's matrix column directly into V, and store the completed reflector as part of vector formation. Column k is already excluded from the active GEMV/rank2 update, so early storage does not change that update's inputs. This removes two D2D copy submissions per reflector on the owning process column, with unchanged RCCL communication and arithmetic.

- Snapshot `direct-eigh` SHAc7f313e14282f2e37cab2629269871e609077284692ceac70a07511280076801. Tiny31 four families/both precisions pass with identical numerical metrics to the norm-fused control. `direct` suite will gate97 rectangles/families then compare `fused-eigh`, `vector-fusion-eigh`, and `direct-eigh` on the same node, isolating E20 and E22 separately. Not yet a performance selection.
- E19 job **5736836** currently running; all12 correctness gates passed. Initial paired8192 FP64 main1x8→tri8x1 warm2.551→2.219s confirms scalar-collective fusion helps. Full FP32/16384 outcomes pending. This justifies continuing launch/collective overhead work rather than implementing a speculative new eigensolver stage.

### E23: form scalar coefficients inside the vector kernel (candidate)

Hypothesis: a separate one-thread kernel solely merging a few gathered norm statistics adds another launch dependency. Merge those statistics once per vector CTA in shared memory, with only CTA0 publishing d/e/tau. Each CTA computes the same tiny robust scalar problem; no cross-CTA barrier or global scalar read is needed during vector formation. Later kernels read tau only after this kernel completes. One kernel/name and one scalar workspace entry disappear.

- Snapshot `merged-eigh` SHA75c673ccedcf8e9435588345cf7eda83b6d12d4c26798570b113030d6b8bd969. Tiny31 four families/both precisions pass with unchanged metrics. Batch `merged` will gate uneven rectangles, pair direct→merged at8192, test16384, refresh1/2/4/8G scaling and same-node fullELPA1 controls. No performance claim yet.
- Job **5736889**,14min, `direct` suite onfrontier02621 currently compares E20/E22 frozen snapshots. No snapshots are rebuilt while jobs use them.
