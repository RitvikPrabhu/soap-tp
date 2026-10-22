# Custom distributed symmetric eigensolver on Frontier

Experimental branch: `research/custom-eigh-frontier-20261022`. Existing SOAP implementations and behavior are unchanged. This standalone driver implements **real** FP64/FP32 symmetric eigendecomposition, not complex Hermitian eigendecomposition. It is not yet a Python/SOAP backend.

The experiment-by-experiment evidence is in [RESULTS.md](RESULTS.md); restart state and the original four-hour budget are in [../../RESEARCH_CHECKPOINT.md](../../RESEARCH_CHECKPOINT.md). Performance decisions must follow those measurements, not the theoretical attractiveness of a kernel or transport.

## Reproduce

From the repository root on Frontier, with the existing GPU ELPA and math installations in `build/`:

```bash
module reset
module load PrgEnv-gnu cpe/26.03 rocm/7.0.2 craype-accel-amd-gfx90a
bash research/frontier_eigh/build.sh
mkdir -p output/custom-eigh
EIGH_BIN=build/custom-eigh/eigh sbatch --time=00:05:00 \
    research/frontier_eigh/run.slurm case 8 \
    --n 8192 --block 64 --pr 2 --pc 4 --back-block 128 \
    --tridiag elpa --gpu-mpi 1 --rccl 1 --repeat 3 --reference 1
```

For the two-stage candidate, add `--band 64`. Initially **bandwidth must equal the distribution block size**. Full ELPA controls use `--solver elpa1` or `--solver elpa2`; local rocSOLVER uses `--solver rocsolver` on one rank. `--elpa-times 1` prints internal stage profiles, but adds instrumentation/output overhead; do not substitute these totals for uninstrumented controls.

For FP32 dense state, use `--precision fp32`. `--tridiag elpa` retains FP64 tridiagonal work; explicit `--tridiag elpa32` tests the FP32 component. Normwise correctness does not guarantee accurate small eigenvalues or a stable SOAP basis in poorly separated subspaces.

Batch scripts use account `gen150`, one node, one rank per GCD, seven CPU cores per rank, and one OpenMP/BLAS thread. They do not launch multi-node jobs. Matrices are deterministic functions of global indices (no external data or stochastic seed). Every `COMMAND` line records material configuration, followed by the executable SHA256 and all trial timings. First trials include library/kernel warmup; subsequent trials are the warm comparisons. Setup, input generation, exact validation and optional reference solves are outside `total`; per-call solver workspaces/handles are inside `total`.

```bash
python3 research/frontier_eigh/summarize.py output/custom-eigh/job-*.out > results.csv
```

The CSV extractor keeps each case's final-trial phases and numerical metrics, plus cold timing and the median/min/max of subsequent warm trials. Single-trial correctness cases have no warm statistics. Raw stdout preserves every trial. Native-half expected-failure rows are not successful solves: consult the `*_CHECK FAILED` markers and stderr. Earlier jobs predate some metadata fields.

Tested stack: ROCm7.0.2/gfx90a, Cray MPICH9.1, LibSci26.03, ELPA2026.02.002. Link tracing confirms CPU PDSTEDC/BLACS/BLAS resolve to **LibSci**, despite reference ScaLAPACK/OpenBLAS archives also being on the link line. Login-node MI210 runs are tiny debugging tests only; all performance evidence is compute-node MI250X GCDs.

## Distribution and ownership

For zero-based global `(i,j)`, owner coordinates are `(floor(i/b) mod pr, floor(j/b) mod pc)`. MPI rank is `row*pc+column`. Local row index is `floor(i/(b*pr))*b+i mod b`, and similarly for columns. Local storage is column-major with leading dimension equal to the local row count. Tiles are square; **process grids need not be square**. Both triangles are stored.

Separate row/column communicators have independent sizes. A column panel is broadcast across process columns. Contributions to AV reduce across process columns; disjoint pieces of a global row vector gather across process rows. Compact-WY products reduce across process rows. Full A and Q are always block-cyclic distributed. Tests requiring empty local ownership are rejected explicitly; choose enough blocks for the requested grid.

There is no dense gather in a multi-rank solve. Replication is limited to vectors, small Gram/T matrices, and transient narrow panels. Validation uses distributed FP64 GPU SUMMA and distributed PDTRAN; only tiny n≤512 CPU LAPACK references construct a full matrix on rank zero. Optional larger references use distributed ELPA2. Dominant-subspace validation replicates only eight selected columns, not full Q.

## Algorithms and component boundaries

**One stage:** custom Householder similarity reduction A→T; distributed divide-and-conquer for T; custom reverse compact-WY backtransform Q=H₀H₁…Z. The active trailing local GEMV uses rocBLAS; a fused HIP rank-2 kernel updates A. FP64 scalar formation is max-scaled; vectors and reflectors are retained with distributed A. `--tridiag scalapack` selects FP64 LibSci PDSTEDC, while `elpa` selects only ELPA's public FP64 distributed tridiagonal component. FP32 dense mode can instead select `elpa32` for FP32 tridiagonal work.

**Two stage:** custom distributed tall-panel Householder QR followed by custom GPU two-sided compact-WY updates A→B. With Q₁=I−VTVᵀ, compute W=AVT−½VTᵀ(VᵀAVT), then A←A−VWᵀ−WVᵀ. CPU panels use only scalar/short-vector MPI reductions; large updates use rocBLAS GEMM. First-stage Householders occupy a separate distributed GPU matrix. ELPA's explicit `bandwidth` mode skips **both** its dense→band stage and its final band→full backtransform. It supplies B→T, distributed eigensolution and Q₂ backtransform; our compact-WY implementation then applies Q₁. ELPA's band chase is distributed CPU work; tridiagonal leaves/merges and specialized Q₂ backtransform use its GPU implementation. FP32 band mode uses the FP32 ELPA path.

In two-stage logs, `reduction` is our dense→band stage, `tridiag` includes the **entire** banded-input pipeline (band chase, eigensolve, Q₂ backtransform), and `back` is our Q₁ application. ELPA profiles separate those internal phases. This component reuse is intentional; the custom implementation does not claim to own bulge chasing or the secular equation yet.

rocBLAS, HIP kernels and RCCL share the default HIP stream. RCCL collectives are enqueued without redundant device-wide synchronization; blocking host reads and phase boundaries supply required waits. GPU-aware MPI synchronizes producers before handing it device buffers. Host-staged MPI remains an independent measured control. This is synchronization avoidance, **not** a completed look-ahead/overlap implementation. `comm` measures MPI/RCCL host call wall time, excludes library-internal traffic, and for asynchronous RCCL does not measure device communication duration.

`managed_gpu_peak_mib` counts the maximum live bytes in this driver's `Device` buffers, then takes the maximum over ranks. It excludes library/RCCL internal allocations and host storage; it is not a total GPU-memory high-water measurement. It is reduced outside the timed solve.

## Precision and correctness

Every solve checks original-problem Frobenius residual, orthogonality, and reference eigenvalue agreement when available. Input quantization additionally reports the rounded-problem residual separately. Tiny references use LAPACK; larger `--reference 1` cases use distributed FP64 ELPA2. Global eigenvalue error and smallest-eigenvalue relative error are distinct. Reference comparison uses no elementwise eigenvector matching.

`top_subspace` is `||(I−RRᵀ)Qtop||F/||Qtop||F` for the eight dominant reference vectors. It is invariant to their signs and rotations within the selected subspace. `top_gap` reports the normalized separating eigengap; degenerate boundaries skip this comparison. Its accuracy depends strongly on that gap.

`--quant fp16|bf16` rounds **input only**, retaining requested dense arithmetic. `--half-bulk 1` is actual FP16-input GEMM with FP32 accumulation/state and fails the strict tested accuracy criteria. `--half-bulk 2` represents operands as high plus4096-scaled low halves, computing three FP16-input/FP32-accumulate products; it omits low×low. MI200 flushes FP16 MFMA subnormals, so small high values are routed into the scaled low part. This is mixed arithmetic, not FP16 storage/compute throughout, and is not guaranteed for arbitrary dynamic range. Baseline GEMV and scalar work remain higher precision.

The existing SOAP initializer consumes eigenvectors and discards eigenvalues; its FP32 input is currently promoted to FP64 for ELPA and cast back afterward. Thus tiny eigenvalue relative errors alone neither prove nor disprove acceptable optimizer behavior. Low precision must ultimately be judged on eigengaps/subspaces **and SOAP update or training behavior**. This loop does not claim that a solver-only residual establishes training equivalence.

`soap_precision.py` isolates basis sensitivity on tiny96×64 CPU examples using the actual `adam_update` primitive, SOAP's adjusted epsilon, and dense forward/back projections. It compares frozen FP64, FP32, FP16-rounded-input and BF16-rounded-input bases over20 updates at three condition numbers/seeds. All update arithmetic is FP64, deliberately isolating basis changes. A signed/permuted-basis control must return the same update. This is **not** an end-to-end distributed SOAP or training experiment, and does not use the custom solver's eigenvectors.
