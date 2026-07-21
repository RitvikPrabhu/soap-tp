# soap-tp

## Getting started

Clone the repository and run the top-level installer:

```bash
git clone https://github.com/RitvikPrabhu/soap-tp.git && cd soap-tp
./install.sh cpu
```

The installer creates `.venv`, installs the CPU PyTorch and Python dependencies,
initializes all pinned submodules, builds ELPA, SLATE, OpenBLAS/ScaLAPACK when
needed, and installs the checkout in editable mode. Activate the completed
environment with:

```bash
source .venv/bin/activate
```

The machine must provide MPI, CMake, Autotools, Libtool, and C++/Fortran
compilers. If anything is missing, the installer prints the appropriate
Homebrew or Ubuntu/Debian package command.

On GPU clusters, first load the site's compiler, MPI, and CUDA or ROCm modules.
If PyTorch is not already installed in the active environment, provide the
site-compatible wheel index:

```bash
TORCH_INDEX_URL="<site CUDA wheel index>" ./install.sh cuda
TORCH_INDEX_URL="<site ROCm wheel index>" ./install.sh rocm
```

See [BUILD.md](BUILD.md) for system prerequisites, cluster modules, GPU
architecture options, installation prefixes, and troubleshooting.

## SOAP step

`soap_tp.soap_step` is the framework-neutral entry point for a tensor-parallel
optimizer integration. Give it a local gradient shard and a persistent
per-parameter state dictionary:

```python
from soap_tp import soap_step

soap_state = {}

update_shard = soap_step(
    gradient_shard,
    soap_state,
    global_shape=(rows, columns),
    shard_dim=1,
    block_size=128,
    process_grid_shape=(2, 4),
)
local_parameter.add_(
    update_shard.to(local_parameter.dtype),
    alpha=-learning_rate,
)
```

Reuse the same `soap_state` dictionary for that parameter on every step. The
function allocates distributed state on its first call, initializes and
refreshes the bases at the configured interval, and returns the normalized
local update. Its first call is a warmup that returns zero, matching the
[original public SOAP optimizer](https://github.com/nikhilvyas/SOAP/blob/main/soap.py).
It has no Megatron or other training-framework dependency.

The lower-level operations remain available from `soap_tp.ops`:

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
  -> redistribute to packed 2D block-cyclic storage
  -> rotate forward
  -> Adam update
  -> rotate backward
  -> redistribute to the original TP shard layout
  -> update left/right preconditioners
  -> refresh bases when due
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

Every rank must call `soap_step` for parameters in the same collective order.
