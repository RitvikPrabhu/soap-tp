# Building and installing soap-tp

ELPA, OpenBLAS, and ScaLAPACK are pinned Git submodules. The cluster supplies
MPI, compilers, and GPU toolkits because their module names and ABI versions are
machine-specific.

## One-command installation

After activating the desired Python environment and loading the cluster modules,
run `scripts/install.sh`. It initializes the native sources if needed, finds the
MPI wrappers, builds a ScaLAPACK/OpenBLAS fallback when the cluster does not
provide ScaLAPACK, builds ELPA, compiles the pybind11 extension, installs the
Python package, and imports the extension as a final check.

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

The normal native prerequisites are MPI, Autoconf, Automake, Libtool, and a
Fortran compiler. CMake 3.26 or newer is needed only when the fallback math
libraries must be built. The installer recognizes `mpicc`/`mpicxx`/`mpifort`
and Cray `cc`/`CC`/`ftn`. A complete custom compiler triplet can be provided
through `CC`, `CXX`, and `FC`.

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

ELPA defaults to CUDA compute capability `sm_60`. Set the architectures needed
by the target cluster when that default is unsuitable:

```bash
ELPA_CONFIGURE_ARGS="--with-NVIDIA-GPU-compute-capability=sm_80,sm_90" \
    ./scripts/install.sh cuda
```

Other site-specific ELPA configure options can be passed the same way. For
example, a nonstandard CUDA toolkit can be selected with
`--with-cuda-path=/path/to/cuda`.

## Installation locations

By default, native outputs live inside the checkout:

```text
build/math-install
build/elpa-install/cpu
build/elpa-install/cuda
build/elpa-install/rocm
```

The Python extension records the selected ELPA library directory as its runtime
search path. Therefore, do not delete that ELPA installation after installing
the package. On a cluster with a persistent software directory, use explicit
prefixes:

```bash
ELPA_PREFIX="$HOME/.local/soap-tp/elpa/cuda" \
MATH_PREFIX="$HOME/.local/soap-tp/math" \
    ./scripts/install.sh cuda
```

The same command can be rerun after source changes; completed native build work
is reused. Pass `--editable` for a development Python installation:

```bash
./scripts/install.sh cpu --editable
```

If ELPA succeeded but a later packaging step failed, rerun only the Python part
without recompiling ELPA:

```bash
./scripts/install.sh cpu --skip-elpa
```

Additional arguments are forwarded to `pip install`, so `--user` and similar
site options remain available.

## How the Python extension is built

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
ELPA_PREFIX=/path/to/the/selected/elpa/installation
ELPA_PROFILE=cpu, cuda, or rocm
CC=<MPI C compiler wrapper>
CXX=<MPI C++ compiler wrapper>
FC=<MPI Fortran compiler wrapper>
```

`setup.py` uses `ELPA_PREFIX` to discover two different things:

```text
ELPA headers:  $ELPA_PREFIX/include/elpa-*/elpa/elpa.h
ELPA library:  $ELPA_PREFIX/lib/libelpa.* (or lib64/libelpa.*)
```

It gives the parent of `elpa/elpa.h` to the C++ compiler as an include
directory, links the extension with `-lelpa`, and records the ELPA library
directory as a runtime search path. It then builds the extension as:

```text
soap_tp.elpa_bindings
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

Running plain `pip install .` without the installer does not build ELPA. If an
ELPA installation already exists, the equivalent manual package command is:

```bash
SOAP_TP_BUILD_ELPA_BINDINGS=1 \
ELPA_PREFIX=/absolute/path/to/elpa \
CC=mpicc CXX=mpicxx FC=mpifort \
    python -m pip install --no-build-isolation .
```

That command only builds and installs the Python package and extension. The
recommended `scripts/install.sh` entry point performs the ELPA build first and
passes these values automatically.

## Lower-level scripts

`scripts/build_elpa.sh` builds ELPA without installing the Python package.
`scripts/build_math_deps.sh` builds only the pinned OpenBLAS/ScaLAPACK fallback.
They are primarily useful for troubleshooting; a normal installation should use
`scripts/install.sh`.
