# soap-tp

## Getting started

Clone the repository with its pinned ELPA, SLATE, OpenBLAS, and ScaLAPACK
submodules. SLATE's BLAS++, LAPACK++, and TestSweeper dependencies are pinned
recursively:

```bash
git clone --recurse-submodules https://github.com/RitvikPrabhu/soap-tp.git
cd soap-tp
```

Load or install a C++17/OpenMP and Fortran toolchain, MPI, CMake, Autoconf,
Automake, and Libtool. On a GPU cluster, also load its CUDA or ROCm modules.
Then create a Python environment and install the PyTorch build appropriate for
that machine:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# Install the site's CPU, CUDA, or ROCm PyTorch build here.
```

Build ELPA, SLATE, their native dependencies, and the Python extensions with the
matching profile:

```bash
./scripts/install.sh cpu       # CPU
./scripts/install.sh cuda      # NVIDIA GPU
./scripts/install.sh rocm      # AMD GPU
```

See [BUILD.md](BUILD.md) for system prerequisites, cluster modules, GPU
architecture options, installation prefixes, and troubleshooting.
