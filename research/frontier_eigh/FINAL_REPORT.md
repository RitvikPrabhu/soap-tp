# Frontier distributed eigh: research outcome

## Third loop update: actual FP16 throughput (in progress)

The authorized third window is October23,08:29–12:29EDT, on dedicated local branch `research/custom-eigh-fp16-20261023`, based on fc09aa3. No production SOAP files were changed, and nothing was pushed or merged. A usage-limit pause interrupted work around09:26–10:09; the deadline was not extended.

**Current conclusion:** FP16 arithmetic is implemented and tested, but FP32 remains the fastest accurate custom mode. A tuned **native FP32** panel kernel gives the useful new speed improvement. It is not evidence for full-FP16 eigh or a robust half-compute speedup.

### Algorithm, ownership and hardware execution

The selected algorithm is unchanged: power-of-two normalization, our one-stage distributed Householder A→T, ELPA's **tridiagonal-only** distributed divide-and-conquer, precision-preserving redistribution, our compact-WY eigenvector backtransform, then eigenvalue rescaling. Delayed panels are still one-stage reduction, not a band/bulge-chasing algorithm. Full A and Q remain distributed; only tiny correctness exports gather them.

```diagram
┌────────────────────┐     ┌───────────────────────┐
│ Distributed A      │────▶│ Our Householder A → T │
│ HIP / block cyclic │     │ HIP + rocBLAS + RCCL  │
└────────────────────┘     └──────────┬────────────┘
                                     ▼
                          ┌───────────────────────┐
                          │ ELPA tridiagonal D&C  │
                          │ Distributed Z and λ   │
                          └──────────┬────────────┘
                                     ▼
                          ┌───────────────────────┐
                          │ Our compact-WY QZ     │
                          │ Distributed output Q  │
                          └───────────────────────┘
```

One rank per MI250X GCD; all batch experiments use at most one Frontier node. ScaLAPACK2D block-cyclic storage supports1×8,2×4,4×2,8×1, including uneven tile counts. Main1×8 / tridiagonal8×1, tile64 and backtransform256 remain the target configurations. At8192 use panel1. At16384 the new candidate is panel16 with explicit installed rocBLAS solution `-606624864`. IDs are backend/version-specific, not portable tuning constants: support is checked per shrinking descriptor, unsupported edges use the standard kernel, and actual selected/default counts are printed. No numerical precision changes in this optimization.

HIP owns matrix/vector kernels and robust FP64 reduction scalars; rocBLAS supplies FP32/FP64 GEMV/GEMM. Same-stream RCCL handles GPU vector/panel collectives, with GPU-aware MPI and LibSci/ScaLAPACK handling other distributed work and redistribution. CPU FP64 DTRMV constructs compact-T. rocSOLVER is used for local controls/experimental Jacobi pivots, not our selected dense reduction. No native FP16 tridiagonal eigensolver is claimed.

### New measured evidence

Job5742774 passes all26cases (16uneven-grid gates and ten paired target-size/control cases). Warm medians, seconds:

| n / panel | Native heuristic | Native selected kernel | Full ELPA1,4×2 |
|---|---:|---:|---:|
|8192 /16|1.5855|1.5981|1.6086|
|8192 /32|1.6527|1.6644|same control|
|16384 /16|4.7926|**4.6730**|**4.4311**|
|16384 /32|4.8406|4.7652|same control|

Panel1 remains the better8192 configuration (prior baseline1.519s); do not select a losing tuned panel merely because its isolated GEMM wins. At16384, panel16 kernel selection reduces matched total by2.5%. FullELPA1 remains faster. Fresh selected-source scaling and mixed controls are pending in E52, not established by the historical scaling curve below.

The16384 selected-panel16 final trial has reduction3.7556s, tridiagonal.6120s including retile.1369s, backtransform.2542s; managed GPU peak279.2MiB/GCD excludes library internals/pool reservations/host arrays. Setup1.1543s is outside total. Host-enqueue communication timing is not isolated network/device time. Warm total range4.6220–4.7241s (two samples, not a confidence interval).

Original-A residual9.846e-7, raw Frobenius orthogonality8.978e-4 (normalized7.014e-6), relative Q-norm error3.568e-8, max eigenvalue error/spectral radius8.262e-7. These are unchanged at printed precision versus the heuristic with the same panel. Same-input ROCm torch gates for selected source are pending; prior mixed-panel gates passed4/4 ordinary-scale2×4 cases and12/12 opposite-grid scale-stress cases. Extreme-scale torch itself fails/is inaccurate, so those cases use CPUFP64 plus explicit precision floors, not a claimed measured torch match.

### What actual lower precision achieved

- Split-FP16 uses FP16 operands/MFMA, FP32 accumulation/state and three algebraic products in two GEMM launches. It is **computation**, not just FP16 storage, but not a fully FP16 eigensolver. Prior8192/16384 totals1.520/5.002s do not beat native1.519/4.818s.
- New fused rocWMMA correction kernels retain FP32 accumulation and use one C epilogue. Shared64×64 tiling makes16384 panel GEMM.512ms versus defaultSGEMM.866ms, but tuned nativeFP32 is **.284ms**. Selective-half panel32 full solve4.806s versus native4.837s is only0.6%, and does not beat selected-native panel16. Backtransform fused-half kernels lose. No half-speed victory is selected.
- One-launch K3 concatenation improves the split implementation but loses native backtransform. Its bounded packing can overflow beyond|high|≈1023.5 and omits low×low; no generic dynamic-range claim or solver integration.
- Native one-product FP16/BF16 did not meet retained FP32-level gates. Split-BF16 also fails a stronger Q-norm case. Lowest operand precision demonstrated under the bounded strict gates is corrected FP16; fastest supported state/solve precision remains FP32. Ill-conditioned SOAP application evidence still favors FP64 solve→FP32 basis storage.
- Public cuSOLVERMp documents FP32/FP64 SYEVD/SYTRD/STEDC, not a native half eigensolver. Actual repository SOAP initialization uses FP64 ELPA then FP32 storage. ROCm torch Half eigh is unsupported. There is no verified CUDA-half baseline to claim parity with.

### Rejected experiments and decision

HIP graphs pass20cases but8192 total14.28s versus1.51s: capture alone≈12.67s overwhelms replay. FP32 Jacobi pivots plus FP64 refinement pass after exact local seed normalization fixes a reproduced rocSOLVER scale problem, but2048≈1.91s versus FP64 pivots.572s;74full512clusters and382size256solves erase the intended savings. Both complex paths were removed from selected source, retained in experimental commits/binaries. Two-launch GEMV loses at both target sizes; a single-launch micro follow-up also loses large local shapes and was not integrated.

An initial tuning job5741930 was invalid: our caller treated all negative IDs as default. Its apparent tuning gains are explicitly discarded. Corrected selectors were verified using actual rocprof dispatches in5742092, then timed uninstrumented in5742174 with independent sampled FP64 dot-product checks. Sampled micro accuracy is not substituted for full-solver validation.

The dominant limitation remains sequential reduction/GEMV/collective dependencies (~81%of final phases), not bulk GEMM peak precision. cuSOLVERMp's stage separation,2D ownership and distributed backtransform translate well; proprietary kernels/NCCL tuning and undocumented half/two-stage assumptions do not. The promising substantial next step is a competitive complete two-stage band chase/Q2 schedule, not another unmeasured low-precision substitution. Earlier custom A→B plus ELPA B→T/Q2 was correct but slow (~14.86s at16384); SLATE was studied, not benchmarked, because the inspected grid/root-band design conflicts with requirements. Practical production adoption still favors tuning/extending ELPA1; retain custom code for algorithm control and research.

### Reproduction and delivery

Current selected-source snapshot: `build/custom-eigh/selected-panel-eigh`, SHA256 `4b802180e912b3da52bca189c08e1dd8e92f2b0685db2673f1ddcc188c3d505e`. Build/commands are in README; E44–E52 in RESULTS record hypotheses, failures and exact raw paths. Final job/file/commit inventories and refreshed scaling are still pending. All remaining work must stop by12:29EDT; no background resume or renewed budget is implied.

## Historical final update: second research loop, FP32 / actual FP16 computation

**Recommendation:** retain the custom native-FP32 path as the fastest measured custom mode. Mixed split-FP16 computation meets the tested accuracy gates but is not consistently faster. Do not claim a full-FP16 eigensolver, universal PyTorch equivalence, or superiority to a tuned distributed ELPA1 backend. The existing SOAP backend is unchanged.

The second authorized window was October23,00:15–04:15EDT. A usage-limit pause interrupted active work around02:08; the last queued job finished02:11:20. The scheduled resume after05:03 was used only for reporting, removing rejected/unmeasured candidates and verifying the selected build, not restarting experimentation. Consequently the full four-hour active research opportunity was not used. All work remains on local branch `research/custom-eigh-frontier-20261022`; nothing was pushed or merged.

### Best implementation and hardware

- **Eight MI250X GCDs on one Frontier node**, one MPI rank/GCD: four physical MI250X packages. Earlier scaling also used1/2/4GCDs; no multi-node jobs. Tiny login-node checks used an MI210 and are not performance evidence.
- Real symmetric FP64/FP32, full distributed eigenvectors. ScaLAPACK2D block-cyclic, block64; fastest main grid1×8, independent tridiagonal grid8×1, backtransform block256. At8192 use panel1; at16384 use delayed panel32. Correctness includes1×8,2×4,4×2,8×1 and unequal local block counts. No full A/Q gather except tiny≤512 references/exports.
- Exact power-of-two normalization → custom GPU one-stage Householder A→T → ELPA **tridiagonal-only** distributed divide-and-conquer → precision-preserving distributed Z redistribution → custom compact-WY backtransform → restored eigenvalue scale. Delayed panel32 remains one-stage, not A→B→T.
- HIP kernels form robust FP64 norms/dots/scalars and requested-precision vectors/updates; rocBLAS GEMV and GEMM perform dense products. Same-stream RCCL broadcasts/AllReduce/AllGather handle device vectors/panels; GPU-aware MPI and LibSci BLACS/PSGEMR2D/PDGEMR2D handle other distributed work. Small Gram/T factors use host FP64 DTRMV: it beat all three custom GPU-T candidates. ELPA's public tridiagonal interface stages host-local arrays and executes its own distributed/GPU work internally. rocSOLVER supplies local reference and experimental Jacobi pivots, not the selected dense reduction.
- Selected immutable executable `build/custom-eigh/blas-t-eigh`, SHA256 `d7b05fc3b2123527870b9b9528434dd3a6b50b970592a1f1c63c6a569dd013b1`. Final `build.sh` rebuild is byte-identical; see `.amp/in/artifacts/selected-build.log`. Exact commands are in [README](README.md).

### Performance, phases and scaling

Seconds, one cold plus two warm trials per case; custom entries are warm medians. The two-sample ranges are not confidence intervals. Setup, reference solves and validation are excluded; per-solve workspaces/ELPA handles are included. Persistent grid/RCCL/outer rocBLAS setup is outside timing.

| Mode | GCDs | n8192 | n16384 |
|---|---:|---:|---:|
| First-loop custom native FP32 |8|1.711|6.155|
| **Selected custom native FP32** |**8**|**1.51895**|**4.81769**|
| Selected mixed split-FP16 |8|1.51982|5.00233|
| Actual ROCm torch FP32 |1|1.50559–1.50700|13.61405–13.61606|
| Full ELPA1 FP32, earlier controls |8|1.561|4.440|

Native FP32 improves the prior custom baseline by about11%/22%. The torch row is **not an equal-hardware speedup comparison**: custom uses eight GCDs, torch one. Custom still loses slightly to single-GCD torch at8192; it is about2.83× faster at16384 while using eight times as many GCDs. Full ELPA1 controls were not refreshed in a final paired job before the interruption, so no custom-versus-ELPA1 victory is established. Final FP64 target-size timing was not refreshed either; first-loop FP64 is historical below.

Selected final-trial phases from job5738707:

| n / mode | Reduction | Tridiagonal, including retile | Retile subset | Backtransform | Driver-managed peak MiB/GCD |
|---|---:|---:|---:|---:|---:|
|8192 FP32|1.2660|.1835|.0416|.0606|74.8|
|8192 split-FP16|1.2634|.1769|.0419|.0805|114.8|
|16384 FP32|3.9094|.6062|.1358|.2547|281.5|
|16384 split-FP16|3.9980|.6284|.1353|.3245|425.5|

Do not add retile again to tridiagonal time. Phase values are the final trial, not phase medians; totals above are warm medians. Memory excludes library/RCCL internals, host arrays and cached HIP pool reservation. Asynchronous `comm` is host enqueue/MPI call time, not isolated device-network duration.

Earlier first-loop n8192 FP32 strong scaling1/2/4/8GCD: **3.753/3.052/2.062/1.787s** (~2.10× at8GCD,26%parallel efficiency). This is an older configuration, not a fresh scaling curve of the final winner. No new multi-node or final-winner scaling claim is made.

### Accuracy against real ROCm PyTorch

Actual torch is2.10.0+rocm7.0, HIP7.0.51831, MI250X on compute nodes. Its Half eigh is unsupported (`linalg_eigh_cuda not implemented for Half`; the CUDA-named error is ROCm's compatibility API, not an NVIDIA build).

For an orthonormal n×n eigenvector matrix, its Frobenius norm is √n, not1. We measure relative norm error `abs(||Q||F−√n)/√n`, plus original-input residual, normalized orthogonality `||QᵀQ−I||F/√n`, eigenvalue error and sign/rotation-invariant eigenspaces. Norm agreement alone cannot validate eigenvectors. Target-size checks use the same deterministic matrix and thin FP64 torch spectrum/top-eight reference; no distributed full-Q gather.

| n / mode | Relative residual | Orthogonality / √n | Relative Q norm error |
|---|---:|---:|---:|
|8192 torch FP32|7.16e-7|4.03e-6|3.67e-8|
|8192 custom FP32|1.42e-6|5.26e-6|3.04e-8|
|8192 custom split-FP16|1.37e-6|4.06e-6|3.05e-8|
|16384 torch FP32|8.74e-7|5.44e-6|4.62e-8|
|16384 custom FP32|9.32e-7|7.03e-6|3.83e-8|
|16384 custom split-FP16|1.74e-6|5.21e-6|3.82e-8|

All target-size Q-norm errors are below FP32 epsilon (~1.19e-7). Custom residual/orthogonality is comparable, not numerically identical to torch (up to about2× these residuals). Custom max eigenvalue error scaled by spectral radius is4.49e-7/6.60e-7(native8192/16384), versus torch9.39e-7/1.27e-6. This differs from relative Frobenius eigenvalue error; do not compare unlike columns. Top-eight leakage custom native5.44e-5/2.12e-4 is reported with separating gaps, not judged against epsilon.

Tiny same-input tests independently compare CPU FP64 torch and real ROCm FP32 torch on exactly FP32-representable inputs, including repeated/clustered eigenvalues. Job5738338: four families×three seeds×FP64/FP32/split-half, **36/36accepted on2×4**. Scale-stress jobs5738675(2×4,seed42) and5738814(4×2,seed43): each **36/36custom accepted** across four families×scales1e-20/1/1e20. Across the two, native/split max residual1.06e-6/1.15e-6 and max relative Q norm1.67e-7/1.51e-7. The latter can exceed one epsilon; the documented policy is bounded by8eps32 plus measured torch-relative gates, not a universal one-epsilon promise. Eigenspace leakage can reach~.009 in sensitive clustered/ill-conditioned cases and is gap-aware.

At extreme scales torch itself returns inaccurate FP32 solutions at1e-20 and fails convergence at1e20 on all four tested families. These custom cases are checked against CPU FP64 with explicit precision-floor limits and **do not claim a measured torch match**. Comparison jobs correctly exit nonzero to flag reference problems. Ordinary-scale cases establish the torch-relative comparison. Our reproduced n33 small-scale custom residual.732 was isolated to FP32 tridiagonal work and fixed by normalization (2.57e-7 afterward); exact internal ELPA failure mechanism remains unproven.

### Computation precision, not just storage

| Mode | A/Q state | GEMV / scalar / tridiagonal | Bulk GEMM computation | Outcome |
|---|---|---|---|---|
|Native FP64|FP64|FP64 / FP64 / FP64|FP64|Correct baseline|
|Native FP32|FP32|FP32 / FP64 / FP32|FP32|Fastest measured custom mode|
|Split-FP16|FP32|FP32 / FP64 / FP32|FP16 operands, FP32 accumulation, three algebraic products in two launches|Tested accuracy, no robust total speed win|
|Naive FP16/BF16|FP32|Higher-precision core as above|One low-precision product|Fails accuracy; not selected|
|Split-BF16|FP32|Higher-precision core as above|BF16 high+residual, FP32 accumulation|Earlier normwise checks pass; stronger Q-norm case fails; not selected|

Split-half stores high plus4096-scaled-low operands; packing combines both cross terms into the second GEMM, omitting low×low. It explicitly addresses MI200 FP16 MFMA subnormal flushing. At16384 panel32 this is genuine lower-precision reduction-update computation as well as backtransform computation. It is **not fully FP16 state, reduction or tridiagonal solving**, and is not a guarantee for arbitrary dynamic range. `--quant` and post-solve basis casts are separately labeled input/storage-only controls. SOAP accuracy still requires application-level evidence: previous sensitive frozen-basis updates favor FP64 solve→FP32 storage, even when solver norms pass.

### Experiments that changed the decision

- Retained direct GPU column/layout reuse, smaller-communicator panel coefficients and panel32 at16384. At8192 panel1 remains faster.
- Native FP32 redistribution removes FP64 conversion buffers;16384 retile .222→.139s and measured tridiagonal .878→.629s. Exact power-of-two normalization fixes scale sensitivity.
- Joint split-half packing improves16384 mixed warm5.622→5.506s in its paired test, but native5.339s in that job still wins.
- CPU FP64 DTRMV beats global/LDS/shared-Gram GPU compact-T; selected back256. Do not preserve GPU residency at the cost of measured speed.
- RCCL default versus LL8192 is1.620/1.603s, too small/variable to force a policy. Simple loses; LL128 Broadcast fails. Retain automatic defaults and same-stream execution, not a claim of completed asynchronous look-ahead.
- Block Jacobi with FP64 pivots and installed analytic pivot diagonals reaches accurate2048 native/split~.573s; FP32 pivots are less accurate, localSYEVJ takes~3.147s. Not a target-size winner. Lower-only reduction, MRRR, fused update+AV and threaded LibSci also lose or fail accuracy. Fused-AV source removed; frozen binaries preserve the experiment. Unbenchmarked cache candidate removed.
- Existing genuine custom A→B plus ELPA B→T/Q2 prototype loses badly at16384 (FP32~14.860s). A GEMM-rich first stage alone is insufficient. SLATE was studied, not benchmarked, because its inspected square-grid/root-band design conflicts with the required ownership constraints.

The main bottleneck remains the sequential bandwidth-heavy reduction and per-reflector collectives (about81–83%of the selected final-trial phases). The most promising substantial custom follow-up is a competitive complete two-stage B→T/Q2 path, using the working A→B boundary; first profile the band chase and Q2 schedule rather than assume tensor throughput fixes them. A cheaper next measurement is a final-winner scaling/paired ELPA1 refresh before investing further.

cuSOLVERMp's public SYTRD/STEDC/ORMTR stage separation, distributed eigenvectors and2D ownership translate well. Its proprietary implementation and NCCL tuning do not translate literally to ROCm; no undocumented two-stage cuSOLVERMp claim is made. RCCL/device layout and precision-preserving redistribution help here, whereas low-precision peak GEMM rates do not address the dominant GEMV/collective dependency.

### Jobs, files, commits and final delivery

[jobs.csv](jobs.csv) inventories **all43 submitted one-node jobs across both loops** (24first loop,19second):37COMPLETED,3FAILED,2TIMEOUT,1CANCELLED. The three failed comparison jobs expose torch-reference problems, not failed accepted custom results.5738463 was canceled for a duplicate suite selection, not a numerical result. Two earlier timeouts are explained in the first-loop history below. No job remains active.958main-suite RESULT rows include947passed and11explicit rejected native-half/BF16/MRRR cases; failed Jacobi calls may terminate before RESULT and are documented separately. The three complete tiny comparison suites add108/108accepted custom rows; do not count these as main-suite RESULT rows.

Every job's state, elapsed time, suite and raw stdout/stderr path is in jobs.csv; [RESULTS](RESULTS.md) records E0–E43 hypotheses and outcomes. Main records, binary hashes and raw artifact inventory are refreshed in [metrics.csv](metrics.csv), [binaries.sha256](binaries.sha256), [artifacts.tsv](artifacts.tsv). Torch comparison/reference CSVs and backend/acceptance JSON remain under `output/custom-eigh/torch-loop2/` and `output/custom-eigh/torch-large/`. No generated raw binaries/matrices are committed.

All created tracked files are `RESEARCH_CHECKPOINT.md` and these under `research/frontier_eigh/`: `eigh.cpp`, `jacobi.hpp`, `build.sh`, `run.slurm`, `summarize.py`, `soap_precision.py`, `torch_compare.py`, `torch.slurm`, `torch_benchmark.py`, `torch_benchmark.slurm`, `README.md`, `RESULTS.md`, `FINAL_REPORT.md`, `metrics.csv`, `jobs.csv`, `soap_precision.csv`, `soap_gpu_precision.csv`, `binaries.sha256`, `artifacts.tsv`. Complete local commit and file inventories (including the final commit) are generated after committing in `output/custom-eigh/commits.tsv` and `files.txt`; regenerate with `git log --reverse --format='%H%x09%s' 505abff..HEAD` and `git diff --name-status 505abff..HEAD`. Loop2 implementation commits before final consolidation:95863f4,8a04e96,264ef64. Scope is only the research directory/checkpoint; unrelated dirty scripts/tests/core remain untouched.

This delivers a standalone research implementation and reproducible evidence, **not** Python/SOAP integration, complex Hermitian support, full training equivalence, best-possible solver proof, or multi-node scaling. Practical adoption still favors tuning/extending ELPA1; custom native FP32 is worth retaining for controlled algorithm research, but mixed FP16 currently gives no measured reason to replace it.

---

## First-loop report (historical configurations and measurements)

Research window: October 22, 2026, 17:35–21:35 EDT maximum. The last experiment finished21:07EDT; reporting resumed00:00EDT after a usage-limit pause, with no further experiments. Work is confined to local branch `research/custom-eigh-frontier-20261022`, based on `505abff`. No push, merge, deployment, or modification of the existing SOAP solver was performed.

## 1. Outcome and recommendation

We built a working **custom distributed real symmetric FP64/FP32 eigensolver with full eigenvectors**, including our own reduction and backtransformation, arbitrary rectangular process grids, and distributed numerical checks. This is not just a full-ELPA wrapper. The fastest custom path currently uses ELPA **only for the tridiagonal eigensolution**, with an independent LibSci PDSTEDC path retained.

The measured winner among our algorithms is **one-stage Householder reduction with GPU-resident vectors**, not our genuine two-stage prototype. The custom solver remains slower than the best measured full ELPA1 configuration, especially at 16384. For an immediate SOAP backend, the evidence favors **using/extending ELPA1**, while keeping this custom implementation as an independently controllable research path. It does not yet justify replacing the existing backend or scaling this implementation beyond one node.

The lowest broadly defensible precision boundary in the tested ill-conditioned SOAP-style examples is **FP64 eigensolve followed by FP32 basis storage**. FP32 solves are useful for less sensitive problems, but do not preserve all optimizer updates. Native FP16 bulk arithmetic was tested, not merely input rounding; naive half fails accuracy, and a scaled three-product half implementation passes normwise tests without winning performance. No universal FP16/BF16 SOAP solver is established.

## 2. Exact selected algorithm and library ownership

```diagram
┌───────────────────────────────────────────┐
│ Distributed GPU A, 2-D block cyclic        │
└────────────────────┬──────────────────────┘
                     │ custom Householder reduction
                     │ HIP + rocBLAS GEMV + RCCL
                     ▼
┌───────────────────────────────────────────┐
│ O(n) d/e + distributed retained reflectors │
└────────────────────┬──────────────────────┘
                     │ ELPA tridiagonal divide-and-conquer
                     │ or independent LibSci PDSTEDC
                     ▼
┌───────────────────────────────────────────┐
│ Distributed Z, optionally on another grid  │
└────────────────────┬──────────────────────┘
                     │ distributed redistribution if needed
                     │ custom compact-WY / rocBLAS GEMM
                     ▼
┌───────────────────────────────────────────┐
│ Distributed GPU Q, replicated eigenvalues  │
└───────────────────────────────────────────┘
```

For each reflector, the selected reduction:

1. Broadcasts the owner's local column across its process row directly from GPU A.
2. Computes robust local FP64 max/scaled-sum-of-squares statistics, gathers the small statistics across process rows, and forms the Householder scalar on the GPU.
3. Forms row/column layouts of V through padded column-communicator RCCL AllGather and a fused conversion kernel, storing the completed reflector in A.
4. Computes the active trailing GEMV, reduces partial AV across process columns, forms W with a distributed FP64 dot, and performs the local two-sided rank-2 update.
5. Copies only d/e/tau to the host at the reduction boundary. There are no per-reflector host vector transfers in this loop.

The backtransform groups 128 reflectors, constructs the compact triangular factor from distributed Gram matrices, and applies blocks through GEMM. Reflector panels stay on device with the selected GPU transport; small Gram/T factors pass through the host. The tridiagonal component accepts host-local arrays and performs its own distributed GPU work internally; rocSOLVER supplies local leaf eigensolution inside the ROCm-enabled ELPA implementation. Our code does **not** implement its secular equation or distributed merge algorithm.

HIP supplies allocations, copies, custom norms/vector formation and fused updates. rocBLAS supplies GEMV/GEMM and GEMM_EX for the half study. RCCL supplies selected reduction collectives. Cray GPU-aware MPI is an independent measured transport and is also available for backtransform collectives; MPI handles communicators, scalar/panel exchanges and validation. LibSci supplies BLACS, distributed redistribution/transpose, optional PDSTEDC, and CPU panel BLAS. The single-GCD reference directly calls rocSOLVER SYEVD.

Our HIP/rocBLAS/RCCL work uses the default HIP stream; ELPA owns its internal execution. Removing redundant synchronization improved performance, but this is **not** a completed asynchronous look-ahead or independent-stream overlap algorithm.

## 3. Distribution, rectangular grids and memory

For global zero-based `(i,j)` and tile width `b`, owner coordinates are `((i/b) mod pr, (j/b) mod pc)` with integer division. MPI rank is `r*pc+c`. Local storage is column-major. Process rows and columns are independent; square tiles do not imply a square process grid.

One rank is bound to each MI250X GCD. Tested grids include **1x1, 1x2, 2x1, 2x2, 1x8, 2x4, 4x2 and 8x1**, including uneven dimensions such as n97/b8 and n1025/b32. The selected target-size main grid is 1x8; an independent **8x1 tridiagonal grid** plus distributed redistribution is faster overall than retaining the 1x8 tridiagonal grid. This is an empirical algorithm-specific choice, not a restriction on 2x4/4x2 support.

A, retained reflectors, Z and Q remain distributed. Only O(n) vectors, small Gram matrices and narrow panels are replicated. There is **no full dense gather for normal solves**. Tiny input/reference/export paths are explicitly restricted to n≤512. Eight-column dominant-subspace checks do not gather full Q. Empty local ownership is rejected; n/block must provide at least one row and column block to every requested grid coordinate.

Both triangles are stored. The one-stage solver needs distributed A/Q plus O(n) and narrow-panel GPU workspaces, and host-local tridiagonal vectors/eigenvectors. The two-stage prototype adds a distributed dense reflector matrix. `managed_gpu_peak_mib` measures only this driver's live GPU buffers, not ELPA/RCCL/rocBLAS internals or host memory. Therefore it is a lower bound, not a total memory high-water mark.

## 4. Performance, scaling and numerical evidence

Selected-binary job **5737042 completed in13m40s**, frontier03637, with all24 cases passing. Main grid1x8, tridiagonal grid8x1, tile64, compact-WY128, one rank/GCD:

| n | Custom FP64 warm median(s) | Custom FP32 warm median(s) | Full ELPA1 FP64(s),4x2 | Full ELPA1 FP32(s),4x2 |
|---:|---:|---:|---:|---:|
|8192|2.152|1.787|**1.943**|**1.561**|
|16384|9.350|6.161|**6.188**|**4.440**|

These are the synchronous-allocation controls, which also supply the scaling measurements below. The final improvement, stream-ordered allocation, reduces paired custom warm totals to **2.133/1.711s at8192** and **9.249/6.155s at16384**, FP64/FP32, in job5737139. Final backtransform improves .133→.114/.096→.084s at8192 and .709→.631/.438→.404s at16384. Numerical metrics are unchanged. Select this allocation option, but do not claim it closes the ELPA1 gap. Cached pool capacity is excluded from the driver's logical live-buffer metric.

FP32 is **actual lower-precision computation**, not an FP64 solve followed by a cast: matrix state, GEMV/rank2, tridiagonal eigensolution and backtransform use FP32, with robust norm/dot scalar work deliberately retained in FP64. Its speedup over this FP64 custom path is1.20x at8192 and1.52x at16384. The earlier independent single-GCD rocSOLVER8192 control took2.838/1.491s FP64/FP32; the FP32 distributed custom path does not beat that local control.

| n / precision | Reduction(s) | Tridiagonal + redistribution(s) | Backtransform(s) | Residual | Orthogonality | Max relative eigenvalue error |
|---|---:|---:|---:|---:|---:|---:|
|8192 / FP64|1.645|.337|.133|2.74e-15|9.14e-13|1.34e-15|
|8192 / FP32|1.347|.357|.096|1.42e-6|4.71e-4|3.94e-7|
|16384 / FP64|7.192|1.421|.706|3.79e-15|1.79e-12|1.65e-15|
|16384 / FP32|4.734|.908|.441|1.98e-6|8.98e-4|5.79e-7|

Phase values are the **final warm trial**, while the preceding table reports the median of both warm totals. Allocations/copies and phase-boundary bookkeeping explain the small difference between phase sums and total. Redistribution is included in tridiagonal time: final16384FP64/32 retile cost is.356/.232s. Setup is reported separately inCSV and is about.3s at8192 and1.2s at16384 for these eight-GCD cases. Driver-managed GPU peak is138.83/69.43MiB per rank at8192 and533.41/266.74MiB at16384 FP64/32, excluding library-internal allocations and host state.

### Strong scaling at8192, selected algorithm

| GCDs / main grid / tridiagonal grid | FP64 warm median(s) | FP32 warm median(s) |
|---|---:|---:|
|1 /1x1 /1x1|6.700|3.753|
|2 /1x2 /2x1|4.838|3.052|
|4 /1x4 /4x1|3.267|2.062|
|8 /1x8 /8x1|2.152|1.787|

Speedup1→8 is **3.11x FP64 /2.10x FP32**, not near-linear scaling. The same solver algorithm in job5736889 measured8192 main2x4 FP64/32 **3.336/2.608s** and main4x2 **3.089/2.036s**, retaining each main grid for the tridiagonal phase. Both rectangular grids work;1x8 plus a separate8x1 tridiagonal grid wins here. Earlier CPU-vector controls also benchmarked8x1 at3.506/2.180s; those are a different reduction implementation, not a selected-binary8x1 timing.

At16384/4x2, revisiting CPU-vector delayed updates was worthwhile: panel1/8/32 warm medians were **11.279/9.800/9.779s FP64** and **7.064/6.646/6.961s FP32**. This reverses their smaller-size performance ordering, but none beats the selected GPU-vector path on1x8. Normwise/reference checks passed for every panel width. Keep the option as a measured alternative, not the default winner.

All meaningful timings come from Frontier compute-node MI250X GCDs, not the login MI210. Performance cases use three trials: one cold and two warm. Reported warm medians have only two samples and are not confidence intervals. Setup/input generation, validation and reference solves are outside `total`; per-call workspaces and ELPA handles are inside. Persistent grid/RCCL setup and the outer rocBLAS handle are outside both `setup` and `total`. Raw logs retain every trial and executable hash; [metrics.csv](metrics.csv) includes cold/warm ranges and final-trial phase timings.

Correctness checks calculate exact distributed FP64 SUMMA products for residual and orthogonality, compare eigenvalues against LAPACK at n≤512 and distributed FP64 ELPA2 at larger reference cases, and optionally compare sign/rotation-invariant dominant subspaces. No elementwise eigenvector comparison is used. Random SPD, diagonal, clustered, lowrank, indefinite and zero families are exercised. Zero-norm relative errors are reported as absolute errors rather than 0/0; degenerate selected-subspace boundaries are skipped explicitly.

### Two-stage result and important rejected approaches

Our two-stage prototype genuinely implements distributed tall-panel Householder QR and two-sided GPU compact-WY A→B. ELPA's explicit banded-input mode supplies only B→T, eigensolution and Q2 backtransformation; our Q1 backtransform follows. Source and runtime checks confirm the dense-to-band and final Q1 stages of ELPA are skipped.

At n16384/4x2, custom band64+ELPA banded pipeline took **20.263s FP64 / 14.860s FP32**, with the banded pipeline alone **16.431/12.328s**. A profiled n8192 ELPA2 run spent about1.525s in Q2 backtransformation. Thus a GEMM-rich first stage is not enough, and it would be wrong to blame the entire loss on CPU bulge chasing. The tested ELPA installation has OpenMP disabled; all controls use one host BLAS thread. These measurements do not establish the best achievable tuned two-stage implementation.

Other evidence-driven decisions:

- Compact-WY backtransform reduced n4096 FP64 back time .584→.066s (~8.9x).
- Removing a serial GPU scalar lookup reduced paired n8192/2x4 reduction6.305→3.295s; merely moving vectors to the GPU was initially much slower.
- Combining robust norm statistics into one collective improved n8192/1x8 totals2.551→2.219s FP64 and2.112→1.770s FP32.
- GPU-aware MPI did not win uniformly. Same-stream RCCL, vector-layout fusion and direct reflector storage helped the selected path.
- Splitting scalars into every vector CTA removed a launch but gave no reduction benefit on the fastest grid; this E23 change was reverted. It did help slower layouts slightly, without changing their selection.
- Delayed one-stage panels lost at4096/8192 but improved the CPU-vector baseline at16384; they still did not beat the GPU-vector winner.
- Raw FP16 bulk GEMM fails orthogonality. Three-product scaled split-half passes tested normwise criteria but is slower than native FP32, e.g. n8192 total2.747 vs2.703s in its paired experiment.

### Lowest precision for the SOAP use case

Computation precision and storage precision are separate controls:

| Mode | Dense state | Reduction GEMV/update | Norm/dot scalars | Tridiagonal solve | Large backtransform GEMM |
|---|---|---|---|---|---|
| FP64 | FP64 | FP64 | FP64 | FP64 | FP64 |
| FP32 +elpa32 | FP32 | FP32 | FP64 | FP32 | FP32 |
| Split-FP16 bulk | FP32 | FP32 | FP64 | FP32 | FP16 operands, FP32 accumulate, three products |
| Native/split-BF16 bulk | FP32 | FP32 | FP64 | FP32 | BF16 operands, FP32 accumulate, one/three products |
| Input rounding only | requested FP64/FP32 | unchanged | unchanged | unchanged | unchanged |
| Basis storage cast | unchanged solve | unchanged | unchanged | unchanged | unchanged solve; only final Q rounded |

In the two-stage prototype, mixed GEMM modes also change the dense-to-band trailing updates. They do not claim a fully FP16/BF16 reduction or tridiagonal solver. The actual repository CUDA path (`factorizations.py:490–547`, `elpa_bindings.cpp:61–171`) promotes distributed FP32 state to an FP64 ELPA solve and then casts the basis back. That behavior supplies a reference boundary, not evidence that CUDA uses native half eigensolving.

Job5736815 produced90 validated **actual custom GPU eigenbases**. The81-row [soap_gpu_precision.csv](soap_gpu_precision.csv) covers seeds42/43/44,96×64 matrices, three condition numbers, and20 frozen-basis updates using the actual Adam primitive and SOAP epsilon convention. All update arithmetic is FP64 to isolate basis sensitivity. This is neither a training-loss experiment nor native low-precision projection throughput measurement.

| Basis solve/storage | Median final update error, condition100 | condition1e6 | condition1e9 |
|---|---:|---:|---:|
| FP32 solve |6.22e-6|.07798|.32797|
| Split-half bulk / FP32 state |6.24e-6|.07798|.32798|
| FP16-rounded input / FP32 solve |.00382|.43509|.87806|
| BF16-rounded input / FP32 solve |.02935|.67733|.99348|
| **FP64 solve / FP32 basis storage** |**8.89e-8**|**2.76e-6**|**4.73e-5**|
| FP64 solve / FP16 basis storage |.000710|.02392|.29772|
| FP64 solve / BF16 basis storage |.005625|.17792|.74815|

The signed/permuted-basis control agrees to1.61e-13. Maximum error over any update for FP64→FP32 basis storage is1.54e-4 at condition1e9. No acceptance threshold for optimizer updates was supplied, so these are measured tradeoffs, not a universal safety certificate. At extreme conditioning, the common FP32 input itself can already have negative small eigenvalues. Small global eigensolver residuals do not bound coordinate-wise Adam sensitivity in nearly degenerate eigenspaces.

`--quant fp16|bf16` is **input rounding only**. `--half-bulk 1|2` actually uses FP16-input/FP32-accumulate GEMM but leaves state, GEMV and scalar work at higher precision. Split-half uses high+4096-scaled-low representations and three products, omitting low×low, with explicit treatment of MI200 FP16 MFMA subnormal flushing. `--bf16-bulk 1|2` implements native/split BF16 GEMM operands with FP32 accumulation; BF16's exponent range avoids the FP16 residual scaling. Neither is a full FP16/BF16 eigensolver or an arbitrary-dynamic-range guarantee.

**Extended compute experiment, job5737122:** all six native-BF16 rectangular cases fail strict checks; all six split-BF16 cases pass. Same-node8192 FP64/FP32/splitFP16/splitBF16 warm totals are **2.157/1.715/1.755/1.739s**. Split-BF16 residual/orthogonality/eigenvalue error are **1.95e-6/.001193/3.94e-7**, versus FP32 **1.42e-6/.000471/3.94e-7**. Mixed dense-to-band also passes257lowrank and2048random checks, but2048splitBF16 total.1750s loses FP32.1715s. This is genuine lower computation in reduction updates and backtransform, not only storage.

**Allocation follow-up, job5737139:** all22 cases pass. With the pool enabled,8192 FP32/splitFP16/splitBF16 totals are **1.711/1.709/1.719s**, but mixed backtransforms remain slower (.110/.111 vs.084s). The tiny total tie does not establish a half-compute speedup: tridiagonal timings also differ. At16384 splitFP16 takes **6.264s vs FP32 6.155s**. Thus the best practical measured compute path remains native FP32 where its accuracy is sufficient; split-FP16/BF16 are tested alternatives, not performance winners. FP64 remains the conservative solve choice for the sensitive SOAP examples.

## 5. cuSOLVERMp comparison and next algorithmic step

The useful cuSOLVERMp ideas translated cleanly: 2-D block-cyclic GPU state, explicit reduction/eigensolution/backtransform boundaries, distributed divide-and-conquer, and retaining distributed eigenvectors throughout. Separating the tridiagonal grid from the reduction grid was particularly valuable on this implementation. [Public cuSOLVERMp documentation](https://docs.nvidia.com/cuda/cusolvermp/usage/functions.html#cusolvermpsyevd) describes SYTRD/STEDC/ORMTR; it does **not** establish a proprietary two-stage internal implementation, and none is claimed here.

What did not translate as an automatic win: GPU-resident vectors alone, GPU-aware MPI alone, square grids, delayed GEMM updates without removing the sequential dependency, or theoretical half-precision peak throughput. [AMD's MI250 architecture](https://rocm.docs.amd.com/en/docs-6.4.2/conceptual/gpu-arch/mi250.html) supports native lower-precision matrix instructions, but this one-stage reduction is dominated by matrix-vector traffic and collective latency, not GEMM peak. Three half products plus packing did not repay their costs.

The largest remaining bottleneck is the **sequential, bandwidth-heavy one-stage reduction and its per-reflector collectives**; the tridiagonal solve/grid redistribution, GEMM backtransform and host-staged small Gram/T factors are secondary. The rank0-only ROCpd study measured substantial RCCL kernel waiting/latency; these instrumented intervals are not isolated network time or additive solver phase timings. No byte-accurate communication-volume study or real look-ahead overlap is claimed. Reduce-scatter is not a drop-in replacement while every process column needs replicated W; it needs a different ownership/update schedule.

The most promising next **custom-solver** optimization is a complete two-stage path that improves the measured **B→T plus Q2 application** bottleneck, not another dense-to-band GEMM-only rewrite. Use the working distributed A→B implementation as the boundary, study/tune ELPA's band and Q2 schedules, then replace the limiting component with controlled batched GPU bulge-chase/backtransform work. That is a substantial follow-up, not something proven by this loop. For the immediate practical backend, tune/cache ELPA1 handles/workspaces and preserve FP64-solve→FP32-storage until application-level training evidence supports lower solve precision.

SLATE source was studied but not benchmarked: its current `heev` implementation requires a square process grid and gathers the band to root, conflicting with the requested rectangular distributed design. LAPACK/ScaLAPACK and rocSOLVER stage implementations were also studied; see [RESULTS.md](RESULTS.md) for exact source paths and the experiments that changed decisions.

## 6. Reproducibility, jobs, files and commits

Build and exact selected execution commands are in [README.md](README.md). The tested stack is ROCm7.0.2/gfx90a, Cray MPICH9.1, LibSci26.03 and ELPA2026.02.002. Existing installations in `build/` are required. Link tracing verifies LibSci supplies CPU BLAS/BLACS/PDSTEDC despite reference-library archives also appearing on the link line.

- [RESULTS.md](RESULTS.md): hypotheses, changes, configurations, measurements and retain/reject decisions E0–E27.
- [metrics.csv](metrics.csv): all parsed solves, including explicit expected failures, executable hashes and commands. Consult `check`, not just the existence of a RESULT row.
- [jobs.csv](jobs.csv): **all submitted job IDs**, suite, state, elapsed/start/end, node, result count and raw log paths.
- [binaries.sha256](binaries.sha256): immutable executable snapshots, including rejected candidates.
- [artifacts.tsv](artifacts.tsv): generated-file sizes and SHA256 values; raw logs/profiles/basis files are retained under `output/custom-eigh/`, executable/build artifacts under `build/custom-eigh/`. Generated binary/raw data are intentionally not committed.
- [../../RESEARCH_CHECKPOINT.md](../../RESEARCH_CHECKPOINT.md): original budget, resume instructions, pitfalls and final active-process state.

The selected executable is `build/custom-eigh/pool-eigh`, SHA256 `418dae0168f10d5054f4e18ed8ba25c966f7d0220259027a966f5a42c534af64`, with `--async-alloc 1` and native FP64/FP32 compute. `final-eigh` retains the synchronous-allocation scaling control. The reduction matches E22, with the zero-validation fix retained and E23 reverted; later changes add BF16 controls and the measured allocation option. Frozen historical executables are never overwritten while jobs use them.

Only one one-node debug-QoS job was submitted/running at a time. **All24 submitted jobs are listed in jobs.csv**:22completed, two timed out.5736208 completed34 valid solves before the last step could launch;5736707 completed no solves because shared-filesystem PyTorch imports consumed the allocation. The staged replacement5736815 completed in83s. A concurrent-submit rejection and denied running-time extensions created no new allocations. No multi-node job was justified or submitted. Across all674 parsed solves,664pass and10are expected naive FP16/BF16 failures; no case is unconfirmed and no job remains active.

Tracked files created on this branch are `RESEARCH_CHECKPOINT.md` and the following under `research/frontier_eigh/`: `eigh.cpp`, `build.sh`, `run.slurm`, `summarize.py`, `soap_precision.py`, `README.md`, `RESULTS.md`, `FINAL_REPORT.md`, `metrics.csv`, `jobs.csv`, `soap_precision.csv`, `soap_gpu_precision.csv`, `binaries.sha256`, and `artifacts.tsv`. None existed in the branch base. Unrelated pre-existing dirty scripts/tests/core were left alone.

The complete local commit and changed-file inventories, generated **after** the final commit so they include it, are retained as `output/custom-eigh/commits.tsv` and `output/custom-eigh/files.txt`. They can be regenerated without any environment dependencies:

```bash
git log --reverse --format='%H%x09%s' 505abff..HEAD
git diff --name-status 505abff..HEAD
```

## 7. Limits of this result

This is a standalone **real symmetric** research solver, not complex Hermitian support, a Python extension, a distributed SOAP integration, a full training run, or a cuSOLVERMp performance-equivalence claim. Tridiagonal eigensolution and the experimental second-stage pipeline still use established library components. No multi-node scaling, true look-ahead overlap, asynchronous multi-stream schedule, fully BF16-state eigensolver, or accurate total-device-memory high-water measurement was completed. These limitations do not invalidate the measured rectangular-grid custom dense stages, but they matter when deciding whether to deploy or continue.
