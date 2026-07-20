from functools import lru_cache
from typing import Iterator, Literal, Optional, Sequence, Tuple

import torch
import torch.distributed as dist


def ceil_div(a: int, b: int) -> int:
    if b <= 0:
        raise ValueError(f"b must be positive, got {b}.")
    return (a + b - 1) // b


def num_blocks(dim: int, block_size: int) -> int:
    if dim < 0:
        raise ValueError(f"dim must be non-negative, got {dim}.")
    return ceil_div(dim, block_size)


def block_bounds(block_id: int, block_size: int, dim: int) -> Tuple[int, int]:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    start = block_id * block_size
    end = min(start + block_size, dim)
    return start, end


def block_cyclic_owner_rank(
    block_row: int,
    block_col: int,
    process_grid_shape: Tuple[int, int],
) -> int:
    Pr, Pc = process_grid_shape

    if Pr <= 0 or Pc <= 0:
        raise ValueError(
            f"process_grid_shape must be positive, got {process_grid_shape}."
        )

    owner_pr = block_row % Pr
    owner_pc = block_col % Pc

    return owner_pr * Pc + owner_pc


@lru_cache(maxsize=None)
def block_cyclic_indices(
    size: int,
    block_size: int,
    process: int,
    process_count: int,
) -> Tuple[int, ...]:
    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")
    if process_count <= 0:
        raise ValueError(
            f"process_count must be positive, got {process_count}."
        )
    if process < 0 or process >= process_count:
        raise ValueError(
            f"process must be in [0, {process_count}), got {process}."
        )

    return tuple(
        index
        for index in range(size)
        if (index // block_size) % process_count == process
    )


def validate_process_grid(
    process_grid_shape: Tuple[int, int],
    world_size: int,
) -> None:
    process_rows, process_columns = process_grid_shape
    if process_rows <= 0 or process_columns <= 0:
        raise ValueError(
            "process_grid_shape must contain positive dimensions, got "
            f"{process_grid_shape}."
        )
    if process_rows * process_columns != world_size:
        raise ValueError(
            f"process_grid_shape={process_grid_shape} does not match "
            f"world_size={world_size}."
        )


def local_2d_block_cyclic_shape(
    global_shape: Tuple[int, int],
    block_size: int,
    process_grid_shape: Tuple[int, int],
    rank: int,
) -> Tuple[int, int]:
    rows, columns = global_shape
    process_rows, process_columns = process_grid_shape
    validate_process_grid(process_grid_shape, process_rows * process_columns)
    if rank < 0 or rank >= process_rows * process_columns:
        raise ValueError(
            f"rank must be in [0, {process_rows * process_columns}), got {rank}."
        )

    process_row = rank // process_columns
    process_column = rank % process_columns
    return (
        len(block_cyclic_indices(rows, block_size, process_row, process_rows)),
        len(
            block_cyclic_indices(
                columns,
                block_size,
                process_column,
                process_columns,
            )
        ),
    )


def validate_2d_block_cyclic_buffer(
    name: str,
    matrix: torch.Tensor,
    global_shape: Tuple[int, int],
    block_size: int,
    process_grid_shape: Tuple[int, int],
    rank: int,
) -> Tuple[int, int]:
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2D, got {matrix.ndim}D.")
    if matrix.stride(0) != 1:
        raise ValueError(
            f"{name} must use column-major local storage, got "
            f"stride {matrix.stride()}."
        )

    local_rows, local_columns = local_2d_block_cyclic_shape(
        global_shape,
        block_size,
        process_grid_shape,
        rank,
    )
    if matrix.size(0) < max(1, local_rows):
        raise ValueError(
            f"{name} has {matrix.size(0)} storage rows, expected at least "
            f"{max(1, local_rows)}."
        )
    if matrix.size(1) < max(1, local_columns):
        raise ValueError(
            f"{name} has {matrix.size(1)} storage columns, expected at least "
            f"{max(1, local_columns)}."
        )
    if matrix.stride(1) < max(1, matrix.size(0)):
        raise ValueError(
            f"{name} has invalid leading dimension {matrix.stride(1)}."
        )
    return local_rows, local_columns


def allocate_2d_block_cyclic(
    global_shape: Tuple[int, int],
    block_size: int,
    process_grid_shape: Tuple[int, int],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Allocate a zeroed column-major local 2D block-cyclic buffer.

    This first-stage API intentionally uses the default distributed world.
    """
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first.")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    validate_process_grid(process_grid_shape, world_size)
    local_rows, local_columns = local_2d_block_cyclic_shape(
        global_shape,
        block_size,
        process_grid_shape,
        rank,
    )
    leading_dimension = max(1, local_rows)
    storage_columns = max(1, local_columns)
    return torch.zeros(
        (storage_columns, leading_dimension),
        dtype=dtype,
        device=device,
    ).T


def block_cyclic_tile_views(
    matrix: torch.Tensor,
    global_shape: Tuple[int, int],
    block_size: int,
    process_grid_shape: Tuple[int, int],
    rank: int,
    *,
    mode: Literal["lower", "full"] = "full",
) -> dict[Tuple[int, int], torch.Tensor]:
    """Return views of the global tiles owned by ``rank``."""
    if mode not in {"lower", "full"}:
        raise ValueError(f"mode must be 'lower' or 'full', got {mode!r}.")

    local_rows, local_columns = validate_2d_block_cyclic_buffer(
        "matrix",
        matrix,
        global_shape,
        block_size,
        process_grid_shape,
        rank,
    )
    rows, columns = global_shape
    process_rows, process_columns = process_grid_shape
    process_row = rank // process_columns
    process_column = rank % process_columns
    block_rows = num_blocks(rows, block_size)
    block_columns = num_blocks(columns, block_size)

    row_offsets = {}
    offset = 0
    for block_row in range(process_row, block_rows, process_rows):
        row_offsets[block_row] = offset
        start, end = block_bounds(block_row, block_size, rows)
        offset += end - start
    if offset != local_rows:
        raise RuntimeError("local block-row offsets are inconsistent.")

    column_offsets = {}
    offset = 0
    for block_column in range(
        process_column,
        block_columns,
        process_columns,
    ):
        column_offsets[block_column] = offset
        start, end = block_bounds(block_column, block_size, columns)
        offset += end - start
    if offset != local_columns:
        raise RuntimeError("local block-column offsets are inconsistent.")

    views = {}
    for block_row, block_column in iter_2d_block_cyclic_blocks_owned_by_rank(
        block_rows,
        block_columns,
        process_grid_shape,
        rank,
    ):
        if mode == "lower" and block_row < block_column:
            continue
        row_start, row_end = block_bounds(block_row, block_size, rows)
        column_start, column_end = block_bounds(
            block_column,
            block_size,
            columns,
        )
        local_row = row_offsets[block_row]
        local_column = column_offsets[block_column]
        views[(block_row, block_column)] = matrix[
            local_row : local_row + row_end - row_start,
            local_column : local_column + column_end - column_start,
        ]
    return views


def iter_2d_block_cyclic_blocks_owned_by_rank(
    num_block_rows: int,
    num_block_cols: int,
    process_grid_shape: Tuple[int, int],
    rank: int,
) -> Iterator[Tuple[int, int]]:
    Pr, Pc = process_grid_shape

    if Pr <= 0 or Pc <= 0:
        raise ValueError(
            f"process_grid_shape must be positive, got {process_grid_shape}."
        )

    world_size = Pr * Pc

    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}.")

    pr = rank // Pc
    pc = rank % Pc

    for bi in range(pr, num_block_rows, Pr):
        for bj in range(pc, num_block_cols, Pc):
            yield bi, bj


def iter_lower_2d_block_cyclic_blocks_owned_by_rank(
    num_blocks: int,
    process_grid_shape: Tuple[int, int],
    rank: int,
) -> Iterator[Tuple[int, int]]:
    for bi, bj in iter_2d_block_cyclic_blocks_owned_by_rank(
        num_blocks,
        num_blocks,
        process_grid_shape,
        rank,
    ):
        if bi >= bj:
            yield bi, bj


def column_shard_offsets(shard_sizes: Sequence[int]) -> Tuple[int, ...]:
    offsets = [0]

    for shard_size in shard_sizes:
        if shard_size < 0:
            raise ValueError(f"shard sizes must be non-negative, got {shard_size}.")
        offsets.append(offsets[-1] + shard_size)

    return tuple(offsets)


def iter_column_panel_fragments(
    column_start: int,
    column_end: int,
    shard_offsets: Sequence[int],
) -> Iterator[Tuple[int, int, int, int]]:

    if column_start < 0 or column_end < column_start:
        raise ValueError(
            f"invalid column range [{column_start}, {column_end})."
        )

    if len(shard_offsets) < 2:
        raise ValueError("shard_offsets must contain at least one shard.")

    if column_end > shard_offsets[-1]:
        raise ValueError(
            f"column range [{column_start}, {column_end}) exceeds the shards."
        )

    for source_rank, (shard_start, shard_end) in enumerate(
        zip(shard_offsets, shard_offsets[1:])
    ):
        fragment_start = max(column_start, shard_start)
        fragment_end = min(column_end, shard_end)

        if fragment_start < fragment_end:
            yield (
                source_rank,
                fragment_start - shard_start,
                fragment_end - shard_start,
                fragment_start - column_start,
            )


def get_column_panel_from_col_shards(
    A_col_shard: torch.Tensor,
    row_start: int,
    row_end: int,
    column_start: int,
    column_end: int,
    shard_offsets: Sequence[int],
    destination_rank: int,
    rank: int,
    panel_buffer: torch.Tensor,
    transfer_buffer: torch.Tensor,
) -> Optional[torch.Tensor]:

    destination_start = shard_offsets[destination_rank]
    destination_end = shard_offsets[destination_rank + 1]

    if destination_start <= column_start and column_end <= destination_end:
        if rank == destination_rank:
            return A_col_shard[
                row_start:row_end,
                column_start - destination_start : column_end - destination_start,
            ]
        return None

    rows = row_end - row_start
    columns = column_end - column_start
    panel = None
    if rank == destination_rank:
        panel = panel_buffer[: rows * columns].view(rows, columns)

    for source_rank, local_start, local_end, panel_start in (
        iter_column_panel_fragments(column_start, column_end, shard_offsets)
    ):
        width = local_end - local_start

        if source_rank == destination_rank:
            if rank == destination_rank:
                panel[:, panel_start : panel_start + width].copy_(
                    A_col_shard[row_start:row_end, local_start:local_end]
                )
            continue

        transfer = transfer_buffer[: rows * width]

        if rank == source_rank:
            transfer.view(rows, width).copy_(
                A_col_shard[row_start:row_end, local_start:local_end]
            )
            dist.isend(transfer, dst=destination_rank).wait()
        elif rank == destination_rank:
            dist.irecv(transfer, src=source_rank).wait()
            panel[:, panel_start : panel_start + width].copy_(
                transfer.view(rows, width)
            )

    return panel
