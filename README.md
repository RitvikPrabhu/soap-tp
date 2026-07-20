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

## Current SOAP building blocks

The repository now provides the operations needed to assemble a SOAP pipeline;
it intentionally does not hide them behind an optimizer class yet. Import the
supported surface from `soap_tp.ops`:

- allocate packed, column-major 2D block-cyclic buffers;
- update `G @ G.T` and `G.T @ G` preconditioners with an EMA;
- initialize descending eigenbases with ELPA;
- refresh bases with one SLATE power iteration and QR;
- rotate gradients and updates with SLATE;
- apply the local Adam update;
- permute optimizer state when bases are reordered; and
- redistribute between TP shards and the packed 2D block-cyclic layout.

Both contiguous row shards (`shard_dim=0`) and contiguous column shards
(`shard_dim=1`) follow the same sequence:

```text
TP gradient shard
  -> update left/right preconditioners
  -> redistribute to packed 2D block-cyclic storage
  -> rotate forward
  -> Adam update
  -> rotate backward
  -> redistribute to the original TP shard layout
```

At initialization, call `initialize_basis_2d_block_cyclic_` once for each
preconditioner. At the chosen refresh interval, call
`refresh_bases_and_transport_optimizer_state_`; it moves momentum out of the
old bases, refreshes both bases, reorders variance, and moves momentum into the
new bases.

### First-stage constraints

The first pipeline should keep these constraints explicit:

- all operations use the default `torch.distributed` world;
- native ELPA/SLATE calls use `MPI_COMM_WORLD`, with identical MPI and Torch
  ranks;
- `process_grid_shape=(Pr, Pc)` is row-major and `Pr * Pc == world_size`;
- TP redistribution currently requires equal shards, so the sharded global
  dimension must be divisible by `world_size`;
- ELPA basis initialization requires every rank to own at least one local row
  and column; and
- packed native buffers use `float32` and should be created with
  `allocate_2d_block_cyclic`.

There is no command-line entry point yet. The next layer can be a small
pipeline that owns these buffers and calls the operations in the order above.
