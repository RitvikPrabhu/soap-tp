# Building and installing soap-tp

ELPA, SLATE, OpenBLAS, and ScaLAPACK are pinned Git submodules. SLATE recursively
pins the BLAS++, LAPACK++, and TestSweeper revisions it uses. The cluster
supplies MPI, compilers, and GPU toolkits because their module names and ABI
versions are machine-specific.

## One-command installation

After activating the desired Python environment and loading the cluster modules,
run `scripts/install.sh`. It initializes the native sources if needed, finds the
MPI wrappers, builds a ScaLAPACK/OpenBLAS fallback when the cluster does not
provide ScaLAPACK, builds ELPA and SLATE, compiles both pybind11 extensions,
installs the Python package, and imports the extensions as a final check.

A CPU installation from a fresh checkout is:

```bash
git clone --recurse-submodules https://github.com/RitvikPrabhu/soap-tp.git
cd soap-tp
./scripts/install.sh cpu
```

The script installs into the Python selected by `PYTHON`, defaulting to
`python3`. For example:

```bash
PYTHON="$HOME/venvs/soap/bin/python" ./scripts/install.sh cpu
```

PyTorch must already be installed in that environment. This is intentionally a
prerequisite: CUDA/ROCm PyTorch distributions are cluster-specific, and the
installer must not silently replace a site-supported GPU build with a generic
wheel. The installer can use the pybind11 headers bundled with PyTorch, so the
native package build does not need network access.

The normal native prerequisites are MPI, Autoconf, Automake, Libtool, a Fortran
compiler, and a C++17 compiler with OpenMP support. CMake 3.18 or newer is
required for SLATE; CMake 3.26 or newer is required when the pinned fallback
math libraries must be built. The installer recognizes
`mpicc`/`mpicxx`/`mpifort` and Cray `cc`/`CC`/`ftn`. A complete custom compiler
triplet can be provided through `CC`, `CXX`, and `FC`.

## GPU clusters

For an NVIDIA cluster, load that site's MPI and CUDA modules and select the
CUDA profile:

```bash
module load <site-compiler> <site-mpi> <site-cuda>
./scripts/install.sh cuda
```

For an AMD cluster, load MPI and ROCm and select the ROCm profile:

```bash
module load <site-compiler> <site-mpi> <site-rocm>
./scripts/install.sh rocm
```

The module commands are placeholders; use the names documented by each cluster.
Those modules generally need to be loaded both when installing and when running
the package so that MPI and GPU runtime libraries remain available.

ELPA and the pinned SLATE revision default to CUDA compute capability `sm_60`.
Set the architectures needed by the target cluster when that default is
unsuitable:

```bash
ELPA_CONFIGURE_ARGS="--with-NVIDIA-GPU-compute-capability=sm_80,sm_90" \
SLATE_CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=80;90" \
    ./scripts/install.sh cuda
```

Other site-specific ELPA configure options and SLATE CMake definitions can be
passed through the same variables. For example, a nonstandard CUDA toolkit can
be selected for ELPA with `--with-cuda-path=/path/to/cuda` and for SLATE with
`-DCUDAToolkit_ROOT=/path/to/cuda`.

## Installation locations

By default, native outputs live inside the checkout:

```text
build/math-install
build/elpa-install/cpu
build/elpa-install/cuda
build/elpa-install/rocm
build/slate-install/cpu
build/slate-install/cuda
build/slate-install/rocm
```

The Python extensions record the selected ELPA and SLATE library directories as
runtime search paths. Therefore, do not delete those native installations after
installing the package. When the fallback math libraries are used, SLATE also
records `MATH_PREFIX/lib` in its runtime search path. Keep all selected prefixes
available. On a cluster with a persistent software directory, use explicit
prefixes:

```bash
ELPA_PREFIX="$HOME/.local/soap-tp/elpa/cuda" \
SLATE_PREFIX="$HOME/.local/soap-tp/slate/cuda" \
MATH_PREFIX="$HOME/.local/soap-tp/math" \
    ./scripts/install.sh cuda
```

The same command can be rerun after source changes; completed native build work
is reused. Pass `--editable` for a development Python installation:

```bash
./scripts/install.sh cpu --editable
```

If both native solvers succeeded but a later packaging step failed, reuse those
installations without recompiling either solver:

```bash
./scripts/install.sh cpu --skip-elpa --skip-slate
```

Each flag can also be used independently. A skipped solver must already exist
at its selected prefix.

Additional arguments are forwarded to `pip install`, so `--user` and similar
site options remain available.

## How SLATE is built

SLATE is built as a native C++ library with its bundled BLAS++ and LAPACK++
revisions. The profile selects `gpu_backend=none`, `cuda`, or `hip` for CPU,
CUDA, and ROCm respectively. `SLATE_CMAKE_ARGS` is appended to the CMake command
for site-specific settings, and `BLAS_LIBRARIES`/`LAPACK_LIBRARIES` can select
explicit vendor math libraries. When the pinned OpenBLAS fallback already
exists, SLATE reuses it automatically.

## How the Python extensions are built

`scripts/install.sh` eventually runs:

```bash
python -m pip install --no-build-isolation /path/to/soap-tp
```

Pip reads `pyproject.toml`, selects setuptools as the build backend, and then
setuptools reads the extension definition in `setup.py`. You should not normally
run `setup.py` yourself.

Before invoking pip, the installer exports:

```text
SOAP_TP_BUILD_ELPA_BINDINGS=1
SOAP_TP_BUILD_SLATE_BINDINGS=1
ELPA_PREFIX=/path/to/the/selected/elpa/installation
ELPA_PROFILE=cpu, cuda, or rocm
SLATE_PREFIX=/path/to/the/selected/slate/installation
SLATE_PROFILE=cpu, cuda, or rocm
CC=<MPI C compiler wrapper>
CXX=<MPI C++ compiler wrapper>
FC=<MPI Fortran compiler wrapper>
```

`setup.py` uses the two prefixes to discover the native headers and libraries:

```text
ELPA headers:  $ELPA_PREFIX/include/elpa-*/elpa/elpa.h
ELPA library:  $ELPA_PREFIX/lib/libelpa.* (or lib64/libelpa.*)
SLATE headers: $SLATE_PREFIX/include/slate/slate.hh
SLATE library: $SLATE_PREFIX/lib/libslate.* (or lib64/libslate.*)
```

It supplies those include directories to the MPI C++ compiler, links the
extensions to their native libraries, and records both library directories as
runtime search paths. It builds:

```text
soap_tp.elpa_bindings
soap_tp.slate_bindings
```

This is why the C++ source can contain:

```cpp
#include <elpa/elpa.h>
```

The include statement only names the header. `setup.py` supplies the directory
where the compiler searches for it.

The installer uses `--no-build-isolation` so builds on network-restricted
cluster login nodes can reuse the active environment. `setup.py` uses a normal
`pybind11` installation when present; otherwise it uses the pybind11 headers
bundled with the already-installed PyTorch package.

Running plain `pip install .` without the installer does not build ELPA or
SLATE. If both native installations already exist, the equivalent manual package
command for both extensions is:

```bash
SOAP_TP_BUILD_ELPA_BINDINGS=1 \
SOAP_TP_BUILD_SLATE_BINDINGS=1 \
ELPA_PREFIX=/absolute/path/to/elpa \
SLATE_PREFIX=/absolute/path/to/slate \
ELPA_PROFILE=cpu SLATE_PROFILE=cpu \
CC=mpicc CXX=mpicxx FC=mpifort \
    python -m pip install --no-build-isolation .
```

That command only builds and installs the Python package and extensions. The
recommended `scripts/install.sh` entry point performs both native solver builds
first and passes these values automatically.

## Running the SLATE binding

`slate_power_iteration_qr_float` uses the selected SLATE build profile. A CPU
installation wraps host memory and runs the host-task backend. CUDA and ROCm
installations wrap device memory and run SLATE's device backend. The caller does
not pass a runtime target.

GPU execution uses one MPI rank per accelerator. For a node with eight GPUs,
launch eight ranks and configure the scheduler or MPI launcher to expose one
different GPU to each rank. For example, the equivalent Slurm allocation is
typically requested with `--ntasks-per-node=8 --gpus-per-task=1` and the site's
GPU-binding option. NVIDIA ranks use `CUDA_VISIBLE_DEVICES`; AMD ranks use
`ROCR_VISIBLE_DEVICES`. Each rank must pass pointers belonging to its assigned
accelerator.

The operation is collective over `MPI_COMM_WORLD`; `process_rows * process_cols`
must equal the world size. MPI must already be initialized, normally by importing
`mpi4py.MPI`.

The three buffers are raw integer addresses. Each must describe a distinct,
column-major, rank-local ScaLAPACK-style `float32` allocation with the requested
2D block-cyclic distribution. Individual row-major PyTorch tile tensors must be
packed into that layout before calling the binding.

On macOS, the SLATE build rewrites its OpenMP dependency to use the active
PyTorch installation's `libomp.dylib`. This prevents Torch and SLATE from
initializing two OpenMP runtimes in one Python process.

## Lower-level scripts

`scripts/build_elpa.sh` builds ELPA without installing the Python package.
`scripts/build_slate.sh` builds SLATE, BLAS++, and LAPACK++ without installing
the Python package.
`scripts/build_math_deps.sh` builds only the pinned OpenBLAS/ScaLAPACK fallback.
They are primarily useful for troubleshooting; a normal installation should use
`scripts/install.sh`.
