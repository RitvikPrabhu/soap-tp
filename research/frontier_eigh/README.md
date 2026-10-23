# Custom distributed symmetric eigensolver on Frontier

Experimental branch: `research/custom-eigh-frontier-20261022`. Existing SOAP implementations and behavior are unchanged. This standalone driver implements **real** FP64/FP32 symmetric eigendecomposition, not complex Hermitian eigendecomposition. It is not yet a Python/SOAP backend.

The final findings are in [FINAL_REPORT.md](FINAL_REPORT.md), with experiment-by-experiment evidence in [RESULTS.md](RESULTS.md); restart state and the original four-hour budget are in [../../RESEARCH_CHECKPOINT.md](../../RESEARCH_CHECKPOINT.md). Performance decisions must follow those measurements, not the theoretical attractiveness of a kernel or transport.

## Reproduce

From the repository root on Frontier, with the existing GPU ELPA and math installations in `build/`:

```bash
module reset
module load PrgEnv-gnu cpe/26.03 rocm/7.0.2 craype-accel-amd-gfx90a
bash research/frontier_eigh/build.sh
mkdir -p output/custom-eigh
EIGH_BIN=build/custom-eigh/eigh sbatch --time=00:05:00 \
    research/frontier_eigh/run.slurm case 8 \
    --n 8192 --block 64 --pr 1 --pc 8 --back-block 256 \
    --precision fp32 --tri-pr 8 --tridiag elpa32 --panel 1 --device-vectors 1 \
    --gpu-mpi 1 --rccl 1 --async-alloc 1 --repeat 3 --reference 1
```

This is the selected custom one-stage FP32 configuration, not a claim that it beats full ELPA. At16384 change `--n 16384 --panel 32`; retain back256. Add `--half-bulk 2` for the tested mixed-FP16 bulk compute alternative (not a performance winner). For FP64 use `--precision fp64 --tridiag elpa`. Frozen selected executable `build/custom-eigh/blas-t-eigh` has SHA256 `d7b05fc3b2123527870b9b9528434dd3a6b50b970592a1f1c63c6a569dd013b1`; the final source rebuild matches it byte-for-byte. For rectangular controls, use `--pr 2 --pc 4` or `--pr 4 --pc 2`, with or without the separate tridiagonal grid. For the two-stage candidate, remove `--device-vectors 1 --tri-pr 8` and add `--band 64`. Initially **bandwidth must equal the distribution block size**. Full ELPA controls use `--solver elpa1` or `--solver elpa2` without device-vector/stage-grid options; local rocSOLVER uses `--solver rocsolver` on one rank. `--elpa-times 1` prints internal stage profiles, but adds instrumentation/output overhead; do not substitute these totals for uninstrumented controls.

For FP32 dense state, use `--precision fp32`. `--tridiag elpa` retains FP64 tridiagonal work; explicit `--tridiag elpa32` tests the FP32 component. Normwise correctness does not guarantee accurate small eigenvalues or a stable SOAP basis in poorly separated subspaces.

Batch scripts use account `gen150`, one node, one rank per GCD, seven CPU cores per rank, and one OpenMP/BLAS thread. They do not launch multi-node jobs. Matrices are deterministic functions of global indices (no external data or stochastic seed). Every `COMMAND` line records material configuration, followed by the executable SHA256 and all trial timings. First trials include library/kernel warmup; subsequent trials are the warm comparisons. Setup, input generation, exact validation and optional reference solves are outside `total`; per-call solver workspaces/handles are inside `total`. `setup` measures matrix generation, A/Q allocation and initial upload; persistent MPI/BLACS/RCCL grid initialization and the outer rocBLAS handle creation are outside both fields.

```bash
python3 research/frontier_eigh/summarize.py output/custom-eigh/job-*.out > results.csv
```

The CSV extractor keeps each case's final-trial phases and numerical metrics, plus cold timing and the median/min/max of subsequent warm trials. Single-trial correctness cases have no warm statistics. Raw stdout preserves every trial. `check` marks explicit expected failures, successful continuation/completion, or an unconfirmed final case in an incomplete log. Native-half failure rows are not successful solves. Earlier jobs predate some metadata fields.

Tested stack: ROCm7.0.2/gfx90a, Cray MPICH9.1, LibSci26.03, ELPA2026.02.002. Link tracing confirms CPU PDSTEDC/BLACS/BLAS resolve to **LibSci**, despite reference ScaLAPACK/OpenBLAS archives also being on the link line. Login-node MI210 runs are tiny debugging tests only; all performance evidence is compute-node MI250X GCDs.

## Distribution and ownership

For zero-based global `(i,j)`, owner coordinates are `(floor(i/b) mod pr, floor(j/b) mod pc)`. MPI rank is `row*pc+column`. Local row index is `floor(i/(b*pr))*b+i mod b`, and similarly for columns. Local storage is column-major with leading dimension equal to the local row count. Tiles are square; **process grids need not be square**. Both triangles are stored.

Separate row/column communicators have independent sizes. A column panel is broadcast across process columns. Contributions to AV reduce across process columns; disjoint pieces of a global row vector gather across process rows. Compact-WY products reduce across process rows. Full A and Q are always block-cyclic distributed. Tests requiring empty local ownership are rejected explicitly; choose enough blocks for the requested grid.

There is no dense gather in a multi-rank solve. Replication is limited to vectors, small Gram/T matrices, and transient narrow panels. Validation uses distributed FP64 GPU SUMMA and distributed PDTRAN; only tiny n≤512 CPU LAPACK references construct a full matrix on rank zero. Optional larger references use distributed ELPA2. Dominant-subspace validation replicates only eight selected columns, not full Q.

## Algorithms and component boundaries

**One stage:** custom Householder similarity reduction A→T; distributed divide-and-conquer for T; custom reverse compact-WY backtransform Q=H₀H₁…Z. The active trailing local GEMV uses rocBLAS; a fused HIP rank-2 kernel updates A. FP64 scalar formation is max-scaled; vectors and reflectors are retained with distributed A. `--tridiag scalapack` selects FP64 LibSci PDSTEDC, while `elpa` selects only ELPA's public FP64 distributed tridiagonal component. FP32 dense mode can instead select `elpa32` for FP32 tridiagonal work.

`--device-vectors 1` is the selected one-stage reduction requiring `--gpu-mpi 1 --rccl 1`. HIP kernels form scaled FP64 norms and scalars, requested-precision V/W, and the rank2 update. Padded RCCL AllGather transposes row-distributed vectors into column layout even with unequal local row counts. `--panel >1` delays trailing updates and applies them with GEMM; it still forms reflectors sequentially and is not a two-stage reduction. Coefficients use the row or column layout with the smaller communicator. Panel1 wins8192; panel32 wins16384. Per-reflector host copies disappear during reduction; d/e/tau and the final diagonal extraction return at the phase boundary. Before reduction, a GPU max plus one MPI_MAX scalar chooses exact power-of-two matrix normalization; eigenvalue scale is restored afterward. This fixes a reproduced small-magnitude FP32 tridiagonal failure without changing the eigenspaces. With GPU communication enabled, the shared backtransform copies/broadcasts reflector panels directly on device; only small Gram/T factors pass through the host. CPU FP64 BLAS DTRMV builds T; three custom GPU-T candidates were slower and removed. Host-staged panel transport remains an independent control.

`--tri-pr 8` independently chooses eight process rows for ELPA's tridiagonal component, with process columns inferred from world size. Distributed Z returns through LibSci PSGEMR2D for FP32 or PDGEMR2D for FP64, retaining its actual precision; neither A nor full Q is gathered. This tests whether a faster tridiagonal grid offsets the redistribution cost. `tridiag` includes this redistribution and `retile` separately measures its host elapsed time (packing/communication together). Main A/reflector/backtransform ownership remains unchanged. The optional stage grid is created once outside timed trials, like the main grid.

**Two stage:** custom distributed tall-panel Householder QR followed by custom GPU two-sided compact-WY updates A→B. With Q₁=I−VTVᵀ, compute W=AVT−½VTᵀ(VᵀAVT), then A←A−VWᵀ−WVᵀ. CPU panels use only scalar/short-vector MPI reductions; large updates use rocBLAS GEMM. First-stage Householders occupy a separate distributed GPU matrix. ELPA's explicit `bandwidth` mode skips **both** its dense→band stage and its final band→full backtransform. It supplies B→T, distributed eigensolution and Q₂ backtransform; our compact-WY implementation then applies Q₁. ELPA's band chase is distributed CPU work; tridiagonal leaves/merges and specialized Q₂ backtransform use its GPU implementation. FP32 band mode uses the FP32 ELPA path.

In two-stage logs, `reduction` is our dense→band stage, `tridiag` includes the **entire** banded-input pipeline (band chase, eigensolve, Q₂ backtransform), and `back` is our Q₁ application. ELPA profiles separate those internal phases. This component reuse is intentional; the custom implementation does not claim to own bulge chasing or the secular equation yet.

rocBLAS, HIP kernels and RCCL share the default HIP stream. RCCL collectives are enqueued without redundant device-wide synchronization; blocking host reads and phase boundaries supply required waits. GPU-aware MPI synchronizes producers before handing it device buffers. Host-staged MPI remains an independent measured control. This is synchronization avoidance, **not** a completed look-ahead/overlap implementation. `comm` measures MPI/RCCL host call wall time, excludes library-internal traffic, and for asynchronous RCCL does not measure device communication duration.

`managed_gpu_peak_mib` counts the maximum live bytes in this driver's `Device` buffers, then takes the maximum over ranks. It excludes library/RCCL internal allocations and host storage; it is not a total GPU-memory high-water measurement. It is reduced outside the timed solve.

## Precision and correctness

Every solve checks original-problem Frobenius residual, orthogonality, and reference eigenvalue agreement when available. Input quantization additionally reports the rounded-problem residual separately. Tiny references use LAPACK; larger `--reference 1` cases use distributed FP64 ELPA2. Global eigenvalue error and smallest-eigenvalue relative error are distinct. Reference comparison uses no elementwise eigenvector matching. `--matrix zero` is an explicit regression: undefined relative zero-norm errors are reported as absolute errors, with no changed normalization for nonzero problems. Degenerate selected-subspace boundaries are skipped rather than compared elementwise.

`top_subspace` is `||(I−RRᵀ)Qtop||F/||Qtop||F` for the eight dominant reference vectors. It is invariant to their signs and rotations within the selected subspace. `top_gap` reports the normalized separating eigengap; degenerate boundaries skip this comparison. Its accuracy depends strongly on that gap.

`--quant fp16|bf16` rounds **input only**, retaining requested dense arithmetic. `--half-bulk 1` is actual FP16-input GEMM with FP32 accumulation/state and fails the strict tested accuracy criteria. `--half-bulk 2` represents operands as high plus4096-scaled low halves, computing three FP16-input/FP32-accumulate products in two launches: high×high, then both cross terms packed along the contraction dimension. It omits low×low. MI200 flushes FP16 MFMA subnormals, so small high values are routed into the scaled low part. This is mixed arithmetic, not FP16 storage/compute throughout, and is not guaranteed for arbitrary dynamic range. Baseline GEMV and scalar work remain higher precision. Mixed computation affects delayed reduction GEMMs as well as large backtransform products, not only basis storage.

`--bf16-bulk 1|2` similarly selects **actual BF16-input/FP32-accumulate GEMM**, not BF16 input rounding or basis storage. Mode1 is the native single product; mode2 uses high+residual BF16 operands and three products, omitting low×low. BF16 does not need the FP16 residual scaling. These modes require FP32 state and cannot be combined with FP16 bulk. In one-stage mode they change the two large backtransform GEMMs; in band mode they also change dense-to-band trailing updates. Reduction GEMV remains FP32, robust norm/dot scalars remain FP64, and `--tridiag elpa32` selects FP32 tridiagonal work. Native BF16 fails strict checks; split-BF16 passes tested rectangular and8192 checks but is not a performance winner and has larger orthogonality error. Reproduce the bounded computation-precision comparison with `sbatch --time=00:08:00 research/frontier_eigh/run.slurm low-compute` after rebuilding.

Those BF16 passes used first-loop normwise gates. The stronger second-loop relative Q-norm check rejects a split-BF16 lowrank case (9.8e-7 exceeds8eps32), so BF16 is **not selected for the PyTorch-level accuracy goal**. `torch_compare.py`/`torch.slurm` compare FP64, FP32 and actual split-half bulk computation against same-input CPU FP64 and ROCm torch references, including gap-aware invariant eigenspaces. Pass `EIGH_BIN=build/custom-eigh/blas-t-eigh` to select the final measured binary; their default pool snapshot is the historical first-loop control. `torch_benchmark.py`/`.slurm` independently measure single-GCD target-size torch and export only its spectrum/top-eight vectors. Torch Half eigh is unsupported; it is never silently promoted to make a half baseline.

An orthonormal n×n basis has Frobenius norm √n. The relative Q-norm gate is explicitly bounded by8eps32, alongside residual/orthogonality/eigenvalue/eigenspace checks; it does not promise every error is below one epsilon. Selected8192/16384 native and split-half Q-norm errors are below eps32. At extreme scales1e-20/1e20, torch FP32 itself can be inaccurate/fail convergence. Such cases use CPU FP64/precision-floor checks, are marked as lacking a measured torch match, and preserve the comparison job's nonzero exit status for reference failures.

`--async-alloc 1` uses stream-ordered HIP allocation/free and retains the process-local default memory pool across synchronization points. Paired measurements improve backtransform time in all tested FP64/FP32/mixed cases, with small total gains. This does not change arithmetic or introduce a new stream. The driver-managed logical-byte metric excludes cached pool capacity; actual GPU memory reservation can be larger. `pool-eigh` is the first-loop baseline, not the current selected executable; `final-eigh` retains the earlier synchronous-allocation scaling control.

The existing SOAP initializer consumes eigenvectors and discards eigenvalues; its FP32 input is currently promoted to FP64 for ELPA and cast back afterward. Thus tiny eigenvalue relative errors alone neither prove nor disprove acceptable optimizer behavior. Low precision must ultimately be judged on eigengaps/subspaces **and SOAP update or training behavior**. This loop does not claim that a solver-only residual establishes training equivalence.

`soap_precision.py` isolates basis sensitivity on tiny96×64 CPU examples using the actual `adam_update` primitive, SOAP's adjusted epsilon, and dense forward/back projections. It compares frozen FP64, FP32, FP16-rounded-input and BF16-rounded-input bases over20 updates at three condition numbers/seeds. All update arithmetic is FP64, deliberately isolating basis changes. A signed/permuted-basis control must return the same update. This is **not** an end-to-end distributed SOAP or training experiment, and does not use the custom solver's eigenvectors.

To use actual custom GPU bases, first generate the tiny input corpus on a login CPU, run only compiled singleton MPI solves on one compute-node GCD, then analyze the exported bases on the login CPU. This avoids expensive shared-filesystem PyTorch imports on compute nodes. It includes the three-product split-half backtransform, still not training or a distributed SOAP backend:

```bash
PYTHONPATH=src .venv/bin/python research/frontier_eigh/soap_precision.py --prepare output/custom-eigh/soap-input
EIGH_BIN=build/custom-eigh/eigh sbatch --time=00:08:00 research/frontier_eigh/run.slurm soap
# After JOBID completes:
PYTHONPATH=src .venv/bin/python research/frontier_eigh/soap_precision.py --bases output/custom-eigh/soap-JOBID
```

Job5736815 supplied90 validated GPU bases; `soap_gpu_precision.csv` contains81 update-study rows, including FP64-solve bases rounded afterward to FP32/FP16/BF16. This separates solve accuracy from basis-storage accuracy. At condition1e9, median final update error is.328 for an FP32 solve versus4.73e-5 for FP64 solve→FP32 basis storage. FP16 basis storage gives.298. These controls retain FP64 update arithmetic and do not establish low-precision projection throughput or training equivalence. See RESULTS E21 for all conditions and worst-step errors.

Tiny test I/O is explicitly restricted to2≤n≤512: `--input path` reads exactly n² native FP64 column-major values; `--export-basis path` writes n ascending FP64 eigenvalues followed by n² FP64 column-major eigenvectors **after** numerical validation. Dense input replication and full-Q export are prohibited for larger problems. Normal distributed solves do not use these paths.
