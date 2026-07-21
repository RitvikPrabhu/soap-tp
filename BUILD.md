# Building and installing soap-tp

ELPA, SLATE, OpenBLAS, and ScaLAPACK are pinned Git submodules. SLATE recursively
pins the BLAS++, LAPACK++, and TestSweeper revisions it uses. The cluster
supplies MPI, compilers, and GPU toolkits because their module names and ABI
versions are machine-specific.

## Two-command setup

The top-level installer owns the complete development environment. It fetches
submodules only for native libraries that it builds. From a fresh checkout,
CPU setup is:

```bash
git clone https://github.com/RitvikPrabhu/soap-tp.git && cd soap-tp
./install.sh cpu
```

It creates or reuses `.venv`, installs CPU PyTorch and the remaining Python
dependencies, fetches sources for the native libraries selected for building,
builds a ScaLAPACK/OpenBLAS fallback when the machine does not provide one,
builds ELPA and SLATE, compiles both extensions, and installs the checkout in
editable mode. It is safe to rerun and reuses completed native builds.

Activate the resulting environment when working in a new shell:

```bash
source .venv/bin/activate
```

The normal native prerequisites are MPI, Autoconf, Automake, Libtool, a Fortran
compiler, and a C++17 compiler with OpenMP support. CMake 3.18 or newer is
required for SLATE; CMake 3.26 or newer is required when the pinned fallback
math libraries must be built. The installer recognizes
`mpicc`/`mpicxx`/`mpifort` and Cray `cc`/`CC`/`ftn`. A complete custom compiler
triplet can be provided through `CC`, `CXX`, and `FC`. The bootstrap checks
these tools before installing anything and prints a Homebrew or Ubuntu/Debian
package command when they are missing.

To use an existing environment instead of `.venv`, select its interpreter:

```bash
PYTHON="$HOME/venvs/soap/bin/python" ./install.sh cpu
```

Pass `--no-editable` when a snapshot installation is preferable.

### Preconfigured or offline Python environments

`scripts/install.sh` is the lower-level entry point for environments where
PyTorch is already provisioned and Python packages must not be downloaded. It
builds the native dependencies and package but does not create a virtual
environment or install PyTorch:

```bash
PYTHON="$HOME/venvs/soap/bin/python" ./scripts/install.sh cpu --editable
```

The bootstrap reuses PyTorch from the selected environment or derives the
PyTorch wheel index from the active CUDA or ROCm toolkit. `TORCH_INDEX_URL`
remains available as an override.

## GPU clusters

For an NVIDIA cluster, load that site's MPI and CUDA modules and select the
CUDA profile:

```bash
module load <site-compiler> <site-mpi> <site-cuda>
./install.sh cuda
```

For an AMD cluster, load MPI and ROCm and select the ROCm profile:

```bash
module load <site-compiler> <site-mpi> <site-rocm>
./install.sh rocm
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
    ./install.sh cuda
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
    ./install.sh cuda
```

The same command can be rerun after source changes; completed native build work
is reused. The top-level installer uses an editable Python installation by
default.

If both native solvers succeeded but a later packaging step failed, reuse those
installations without recompiling either solver:

```bash
./install.sh cpu --skip-elpa --skip-slate
```

Each flag can also be used independently. A skipped solver must already exist
at its selected prefix.

The same flags select libraries supplied by environment modules. Load the
module and skip that build; the prefix is discovered from the active package
search path:

```bash
./install.sh rocm --skip-slate
```

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

The public functions in `soap_tp.ops` validate the buffers and call focused
SLATE operations for symmetric multiplication, QR, and forward/backward basis
rotation. A CPU installation wraps host memory and runs the host-task backend.
CUDA and ROCm installations wrap device memory and run SLATE's device backend.
The caller does not pass a runtime target.

GPU execution uses one MPI rank per accelerator. For a node with eight GPUs,
launch eight ranks and configure the scheduler or MPI launcher to expose one
different GPU to each rank. For example, the equivalent Slurm allocation is
typically requested with `--ntasks-per-node=8 --gpus-per-task=1` and the site's
GPU-binding option. NVIDIA ranks use `CUDA_VISIBLE_DEVICES`; AMD ranks use
`ROCR_VISIBLE_DEVICES`. Each rank must pass pointers belonging to its assigned
accelerator.

The operations are collective over `MPI_COMM_WORLD`;
`process_rows * process_cols` must equal the world size. MPI must already be
initialized, normally by importing `mpi4py.MPI`.

Native buffers are distinct, column-major, rank-local ScaLAPACK-style
`float32` allocations with the requested 2D block-cyclic distribution. Use
`soap_tp.ops.allocate_2d_block_cyclic` rather than packing individual tile
tensors manually.

On macOS, the SLATE build rewrites its OpenMP dependency to use the active
PyTorch installation's `libomp.dylib`. This prevents Torch and SLATE from
initializing two OpenMP runtimes in one Python process.

## Lower-level scripts

`scripts/build_native.sh` builds any requested combination of ELPA, SLATE, and
the pinned OpenBLAS/ScaLAPACK fallback without installing the Python package.
It is primarily useful for troubleshooting; a normal installation should use
the top-level `./install.sh`, which requests only libraries not selected with a
`--skip-*` option.

After installation, use `scripts/rebuild_bindings.sh` for incremental pybind
development. The installer records its profile, native prefixes, Python, and
compiler wrappers in `build/bindings.env`, so rebuilding performs no dependency
discovery and can be called from any working directory. For example:

```bash
/path/to/soap-tp/scripts/rebuild_bindings.sh
```

Pass `--force` when a compiler flag changed or a full extension rebuild is
otherwise required. Ordinary C++ source changes are detected incrementally.
