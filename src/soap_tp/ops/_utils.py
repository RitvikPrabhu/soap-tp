from typing import Iterator, Tuple


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
