from typing import Iterator, Optional, Sequence, Tuple

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
