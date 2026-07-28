# Building soap-tp

soap-tp does not load modules, select compilers, install system packages, or
choose a PyTorch wheel. Prepare those for the target machine before building.
Clone with `--recurse-submodules`; the build script does not run Git commands.

## Required system software

- MPI with C, C++, and Fortran compiler wrappers
- CMake
- Autoconf, Automake, Libtool, and Make
- CUDA for a `cuda` build or ROCm for a `rocm` build

The build scripts use the compiler environment that is already active. They do
not choose compiler executables. Set `CC`, `CXX`, and `FC` before running them
only when the loaded environment requires explicit compiler selection.

## Python packages

All required Python packages are listed in `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install the appropriate CPU, CUDA, or ROCm PyTorch wheel before that command
when the default PyTorch package is not correct for the machine.

`requirements.txt` builds mpi4py from source so it links to the active MPI
instead of using a generic MPI wheel. Set `MPI4PY_BUILD_MPICC` to the active
MPI C wrapper when explicit compiler selection is required:

```bash
MPI4PY_BUILD_MPICC="$CC" python -m pip install -r requirements.txt
```

## Native libraries

Build both pinned libraries with one command. The script also builds the pinned
OpenBLAS and ScaLAPACK submodules that they link against:

```bash
./scripts/build_native.sh cpu
./scripts/build_native.sh cuda
./scripts/build_native.sh rocm
```

Only run the line matching the desired build. Outputs go to:

```text
build/elpa-install/<profile>
build/slate-install/<profile>
build/math-install
```

All three profiles build the same source. The profile selects the ELPA and
SLATE backends and the native libraries linked into the Python extensions. A
CPU-only machine does not need CUDA, ROCm, NCCL, or RCCL:

```bash
./scripts/build_native.sh cpu
./scripts/rebuild_bindings.sh cpu
```

The script accepts two omission flags:

```bash
./scripts/build_native.sh rocm --skip-elpa
./scripts/build_native.sh rocm --skip-slate
./scripts/build_native.sh rocm --skip-elpa --skip-slate
```

When a library is omitted, give its existing prefix explicitly. The same
prefix is then used to build the bindings:

```bash
SLATE_PREFIX=/path/to/slate ./scripts/build_native.sh rocm --skip-slate
SLATE_PREFIX=/path/to/slate ./scripts/rebuild_bindings.sh rocm
```

The equivalent variables for ELPA are `ELPA_PREFIX` and `--skip-elpa`.

Optional native build settings are deliberately explicit:

```text
BUILD_JOBS
ELPA_BUILD_JOBS
SOAP_TP_BUILD_ROOT
MATH_PREFIX
ELPA_PREFIX
SLATE_PREFIX
ELPA_CONFIGURE_ARGS
SLATE_CMAKE_ARGS
```

The script never searches for another math-library installation. It always
uses the repository's pinned OpenBLAS and ScaLAPACK sources.

### Optional GPU collectives

CUDA and ROCm builds use MPI by default. ELPA can additionally use NCCL or RCCL
for GPU collectives when the corresponding library is installed:

```bash
ELPA_CONFIGURE_ARGS="--enable-gpu-ccl=nccl --with-nccl-path=/path/to/nccl" \
    ./scripts/build_native.sh cuda

ELPA_CONFIGURE_ARGS="--enable-gpu-ccl=rccl" \
    ./scripts/build_native.sh rocm
```

Omit `--with-nccl-path` when the active compiler and dynamic linker already
find NCCL. This vendored ELPA release parses `--with-rccl-path`, but does not
apply it to the compiler or linker flags; make RCCL discoverable through the
active module environment, `CPPFLAGS`, and `LDFLAGS` instead.

When CCL support is compiled in, ELPA enables `use_ccl` by default and creates
its own parent, process-row, and process-column communicators during GPU setup.
ELPA still uses MPI to bootstrap those communicators. Its CCL detector expects
one MPI rank per GPU and every rank on a node to see the node's GPU set; map
each local rank to a distinct device. If ranks see only their individually
masked GPU, ELPA falls back to MPI collectives. CCL support cannot be combined
with ELPA OpenMP (`--enable-openmp`) or CUDA-aware MPI
(`--enable-cuda-aware-mpi`). After CCL is active, communicator or collective
failures terminate the MPI job rather than switching back to MPI.

## Python bindings

After the native build, compile the pybind extensions in place:

```bash
./scripts/rebuild_bindings.sh cpu
./scripts/rebuild_bindings.sh cuda
./scripts/rebuild_bindings.sh rocm
```

Only run the line matching the native build. The same binding source supports
host tensors in the `cpu` profile and device tensors in the `cuda` and `rocm`
profiles. Use the same profile, compiler variables, and external prefixes used
for the native libraries. Add `--force` only when a full extension rebuild is
needed:

```bash
./scripts/rebuild_bindings.sh rocm --force
```

Make the source tree importable:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

The dynamic loader must find the selected ELPA and SLATE libraries, the pinned
math libraries, and, for GPU builds, the CUDA or ROCm and optional NCCL or RCCL
runtime libraries. Prefer the machine's module or loader configuration. When
needed on Linux, add the local prefixes explicitly:

```bash
PROFILE=cuda
export LD_LIBRARY_PATH="$PWD/build/elpa-install/$PROFILE/lib:$PWD/build/slate-install/$PROFILE/lib:$PWD/build/math-install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Use `DYLD_LIBRARY_PATH` instead on macOS. Also include non-system GPU or CCL
library directories when their installation is not already discoverable.
