# Frontier custom distributed eigh research checkpoint

- Objective: custom controllable AMD distributed eigensolver, rectangular 2D block-cyclic grids, then lowest useful precision.
- Budget: up to four hours, started 2026-10-22 17:35 EDT; stop by 21:35 EDT (01:35 UTC Oct 23).
- Branch: `research/custom-eigh-frontier-20261022`, created from `research/frontier-distributed-eigh` at `505abff`.
- Pre-existing dirty files in scripts and tests and untracked `core`/Slurm debug scripts belong to other work. Do not stage or modify them.
- Initial state: ROCm 7.0.2 available after modules; login GPU is MI210, not a Frontier compute-node MI250X GCD. No initial jobs running.
- Environment: `module reset; module load PrgEnv-gnu cpe/26.03 rocm/7.0.2 craype-accel-amd-gfx90a`. Account `gen150`, batch partition, one node, at most eight ranks/GCDs. Existing math and GPU ELPA installs in `build/`.
- Baseline design: custom HIP/rocBLAS one-stage Householder similarity reduction, distributed CPU ScaLAPACK PDSTEDC, custom distributed GPU backtransform. Square tiles, arbitrary rectangular MPI grid. No full dense gather.
- Evidence: PDSTEDC supports rectangular grids; SLATE heev explicitly requires square grids and gathers the band to root. ELPA two-stage has distributed CPU bulge chasing and GPU dense-to-band/backtransforms. cuSOLVERMp publicly documents SYTRD/STEDC/ORMTR stages but does not disclose a two-stage internal reduction.
- 17:45 EDT (~10 min elapsed, ~3h50 remaining): custom standalone `eigh.cpp` implemented and compiled. Tiny login tests n=2/31 passed FP64, FP32, lowrank, diagonal and rocSOLVER comparison. These are debugging results, not performance data.
- Job 5735939: completed successfully, one node, 19 correctness cases n=97 b=8. Grids 1x1, 1x2, 2x1, 2x2, 1x8, 2x4, 4x2, 8x1; FP64/FP32 random, FP64 diagonal/clustered/lowrank. FP64 residual ~1e-15; FP32 ~2e-7. Exact distributed SUMMA residual/orthogonality and tiny LAPACK eigenvalue reference.
- Fixed experimental ELPA reference: GPU-pointer API requires eigenvalues on device as well as A/Q; initial host eigenvalue buffer produced zero eigenvalues despite ELPA success. Fixed and tiny ELPA2 passes.
- Baseline executable snapshot: `build/custom-eigh/baseline-eigh`. Build with `bash research/frontier_eigh/build.sh`; batch runner `research/frontier_eigh/run.slurm` supports smoke/baseline/case suites, `EIGH_BIN` selects snapshot.
- Next: measure n=2048 grid/precision/block and GPU-aware MPI choices, ELPA1/2 comparison; optimize based on phase timings. No performance conclusions yet.

Detailed experiment record will be kept in `research/frontier_eigh/RESULTS.md` and raw batch logs under `output/custom-eigh/`.
