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

## Python bindings

After the native build, compile the pybind extensions in place:

```bash
./scripts/rebuild_bindings.sh cpu
```

Use the same profile, compiler variables, and external prefixes used for the
native libraries. Add `--force` only when a full extension rebuild is needed:

```bash
./scripts/rebuild_bindings.sh rocm --force
```

Make the source tree importable:

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
```

The native library modules used during the build must also be available at
runtime.
