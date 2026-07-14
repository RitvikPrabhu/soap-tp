# Building ELPA

ELPA, OpenBLAS, and ScaLAPACK are pinned Git submodules. MPI and the compiler toolchain come from the machine or active cluster modules.

There are two build scripts:

- `scripts/build_elpa.sh` is the normal entry point. It initializes missing submodules, finds the MPI wrappers, uses an available ScaLAPACK installation, builds the pinned OpenBLAS and ScaLAPACK fallback when necessary, and then builds ELPA.
- `scripts/build_math_deps.sh` builds only the pinned OpenBLAS and ScaLAPACK fallback under `build/math-install`. `build_elpa.sh` calls it automatically, so it normally does not need to be run directly.

Before running either script, load or install MPI, Autoconf, Automake, and Libtool. CMake 3.26 or newer is required when building the pinned math libraries. CUDA or ROCm libraries must be supplied for the corresponding GPU profile.

A CPU build from a fresh checkout is:

```bash
git clone --recurse-submodules https://github.com/RitvikPrabhu/soap-tp.git
cd soap-tp
./scripts/build_elpa.sh cpu
```

For a normal build, run only `build_elpa.sh`. It recognizes `mpicc`/`mpicxx`/`mpifort` and Cray `cc`/`CC`/`ftn`. If a loaded module provides ScaLAPACK, that installation is used. Otherwise the math build runs automatically before ELPA.

Run the math script directly only to prepare or troubleshoot the fallback libraries separately:

```bash
./scripts/build_math_deps.sh
```

That command does not build ELPA. Run `./scripts/build_elpa.sh cpu` afterward to finish the CPU build.

GPU builds use the same command with a different profile:

```bash
./scripts/build_elpa.sh rocm
./scripts/build_elpa.sh cuda
```

The scripts write build files under `build/`. Math libraries are installed under `build/math-install`, and ELPA is installed under `build/elpa-install/<profile>`. Re-running the same command reuses completed work.
