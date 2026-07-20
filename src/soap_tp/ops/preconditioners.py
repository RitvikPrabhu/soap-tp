from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from ._utils import (
    block_bounds,
    block_cyclic_tile_views,
    block_cyclic_owner_rank,
    column_shard_offsets,
    get_column_panel_from_col_shards,
    iter_2d_block_cyclic_blocks_owned_by_rank,
    iter_lower_2d_block_cyclic_blocks_owned_by_rank,
    num_blocks,
)


def update_left_preconditioner_from_col_shards(
    G_local: Tensor,  # shape is [m, n_local]
    L_prev: Optional[Tensor],
    beta: float,
    *,
    TP_group: Optional[ProcessGroup] = None,  # Process group for tensor parallelism
) -> Tensor:
    """Update one row shard of the left Gram preconditioner.

    The global gradient ``G`` has shape ``[m, n]`` and is partitioned by
    columns across a tensor-parallel process group. Each rank contributes
    ``G_r @ G_r.T``. Summing those contributions gives ``G @ G.T``, whose
    rows are reduce-scattered in process-group rank order.

    Args:
        G_local: This rank's column shard ``G_r``, with shape
            ``[m, n_local]``. All ranks must use the same ``m``, dtype, and
            device type; local column counts do not affect the reduction
            shape.
        L_prev: Previous local row shard, with shape ``[m / p, m]`` for
            process-group size ``p``. It is updated in place. If ``None``, a
            zero tensor with the input dtype and device is allocated.
        beta: Weight of the previous value in
            ``beta * L_prev + (1 - beta) * L_current``.
        TP_group: Group over which column contributions are summed and rows
            are scattered. ``None`` selects ``dist.group.WORLD``.

    Returns:
        The same tensor object as ``L_prev`` when supplied, otherwise the new
        local shard. Group rank ``r`` receives rows
        ``[r * (m / p):(r + 1) * (m / p)]`` of the interpolated left Gram
        matrix, with shape ``[m / p, m]``.

    Raises:
        ValueError: If ``TP_group`` is ``None`` and the default process group
            has not been initialized.
        RuntimeError: If distributed communication is unavailable or the
            tensors do not satisfy PyTorch's matrix, interpolation, or
            reduce-scatter requirements.

    Contract:
        Preconditions:
            - ``torch.distributed`` is initialized, and every rank in
              ``TP_group`` calls this function in the same collective order.
            - ``m`` is divisible by the process-group size.
            - A supplied ``L_prev`` has the documented local shape and is
              compatible with ``G_local`` for dtype and device.

        Guarantees:
            - ``L_prev`` is mutated in place and returned.
            - With ``beta == 0``, concatenating returned shards in group-rank
              order equals the dense result ``G @ G.T``.
            - Partial final column shards are mathematically valid because
              every local contribution has shape ``[m, m]``.

        Invariants:
            - The global, uninterpolated result is symmetric and positive
              semidefinite.
            - Scaling every column shard by ``alpha`` scales the current Gram
              contribution by ``alpha**2``; zero input contributes zero.

        Unchecked assumptions:
            - ``beta`` lies in ``[0, 1]`` when a convex interpolation is
              intended; the function does not validate its range.
            - Collective tensor support depends on the selected backend,
              dtype, and device.
    """

    if TP_group is None:
        TP_group = dist.group.WORLD

    m = G_local.size(0)

    local_contribution = G_local @ G_local.T

    world_size = dist.get_world_size(group=TP_group)

    rows_per_rank = m // world_size
    expected_shape = (rows_per_rank, m)

    if L_prev is None:
        L_prev = torch.zeros(expected_shape, device=G_local.device, dtype=G_local.dtype)

    output = torch.zeros_like(L_prev)

    dist.reduce_scatter_tensor(
        output,
        local_contribution,
        op=dist.ReduceOp.SUM,
        group=TP_group,
    )

    L_prev.lerp_(output, 1.0 - beta)
    return L_prev


def update_right_preconditioner_from_col_shards(
    G_local: Tensor,
    R_prev: Optional[Tensor],
    beta: float,
    *,
    TP_group: Optional[ProcessGroup] = None,
) -> Tensor:
    """Update one row shard of the right Gram preconditioner.

    The global gradient ``G`` has shape ``[m, n]`` and is evenly partitioned
    by columns across a tensor-parallel group. Rank ``r`` returns the rows of
    ``G.T @ G`` corresponding to its local columns. Concatenating rank outputs
    in process-group order reconstructs the complete right Gram matrix.

    Args:
        G_local: This rank's column shard, with shape ``[m, n_local]``. Shards
            may be non-contiguous on entry. All ranks must use the same ``m``,
            ``n_local``, dtype, and compatible devices.
        R_prev: Previous local row shard, with shape
            ``[n_local, p * n_local]`` for group size ``p``. It is updated in
            place. If ``None``, a zero tensor with the input dtype and device
            is allocated.
        beta: Weight of the previous value in
            ``beta * R_prev + (1 - beta) * R_current``.
        TP_group: Group whose column shards form ``G``. ``None`` selects
            ``dist.group.WORLD``.

    Returns:
        The same tensor object as ``R_prev`` when supplied, otherwise the new
        local shard. Its shape is ``[n_local, p * n_local]`` and its rows are
        the block-row of ``G.T @ G`` owned by this rank's local columns.

    Raises:
        ValueError: If ``TP_group`` is ``None`` and the default process group
            has not been initialized.
        RuntimeError: If distributed communication is unavailable or the
            tensors do not satisfy PyTorch's matrix, point-to-point, or
            interpolation requirements.

    Contract:
        Preconditions:
            - ``torch.distributed`` is initialized, and every rank in
              ``TP_group`` calls this function in the same communication order.
            - The global column count is ``n == p * n_local``; all ranks have
              equal-width column shards and equal row counts.
            - A supplied ``R_prev`` has the documented local shape and is
              compatible with ``G_local`` for dtype and device.

        Guarantees:
            - ``R_prev`` is mutated in place and returned.
            - With ``beta == 0``, concatenating outputs in group-rank order
              equals the dense result ``G.T @ G``.
            - A non-contiguous ``G_local`` is accepted without changing its
              mathematical values.

        Invariants:
            - The global, uninterpolated result is symmetric and positive
              semidefinite.
            - Scaling all shards by ``alpha`` scales the current Gram matrix by
              ``alpha**2``; zero input produces a zero current matrix.

        Unchecked assumptions:
            - ``beta`` lies in ``[0, 1]`` when a convex interpolation is
              intended; the function does not validate its range.
            - Collective tensor support depends on the selected backend,
              dtype, and device.

        Unresolved assumptions:
            - Unequal column-shard widths are not represented by the returned
              shape and are not established as supported.
    """

    if TP_group is None:
        TP_group = dist.group.WORLD

    G_local = G_local.contiguous()

    m = G_local.size(0)
    n_local = G_local.size(1)

    world_size = dist.get_world_size(group=TP_group)
    tp_rank = dist.get_rank(group=TP_group)

    expected_shape = (n_local, world_size * n_local)

    if R_prev is None:
        R_prev = torch.zeros(
            expected_shape,
            device=G_local.device,
            dtype=G_local.dtype,
        )

    R_current = torch.zeros_like(R_prev)

    send_to = (tp_rank - 1) % world_size
    recv_from = (tp_rank + 1) % world_size

    recv_buffers = (torch.empty_like(G_local), torch.empty_like(G_local))

    current_G = G_local
    current_owner = tp_rank

    for step in range(world_size):
        if step < world_size - 1:
            next_G = recv_buffers[step % 2]
            reqs = dist.batch_isend_irecv(
                [
                    dist.P2POp(
                        dist.irecv, next_G, group=TP_group, group_peer=recv_from
                    ),
                    dist.P2POp(
                        dist.isend, current_G, group=TP_group, group_peer=send_to
                    ),
                ]
            )

        block = G_local.T @ current_G
        col_start = current_owner * n_local
        R_current[:, col_start : col_start + n_local] = block

        if step < world_size - 1:
            for req in reqs:
                req.wait()
            current_G = next_G
            current_owner = (current_owner + 1) % world_size

    R_prev.lerp_(R_current, 1.0 - beta)
    return R_prev


def _update_aat_from_column_shards_2d_block_cyclic(
    A_col_shard: torch.Tensor,
    left_preconditioner: dict[tuple[int, int], torch.Tensor],
    beta: float,
    block_size: int,
    process_grid_shape: tuple[int, int],
    mode: Literal["lower", "full"] = "lower",
) -> dict[tuple[int, int], torch.Tensor]:
    """EMA-update owned tiles of ``A @ A.T`` from column shards of ``A``."""

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first.")

    if A_col_shard.ndim != 2:
        raise ValueError(f"A_col_shard must be 2D, got {A_col_shard.ndim}D.")

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    if mode not in {"lower", "full"}:
        raise ValueError(f"mode must be 'lower' or 'full', got {mode!r}.")

    Pr, Pc = process_grid_shape
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if Pr * Pc != world_size:
        raise ValueError(
            f"process_grid_shape={process_grid_shape} does not match "
            f"world_size={world_size}."
        )

    m, local_n = A_col_shard.shape
    device = A_col_shard.device
    dtype = A_col_shard.dtype

    nblocks = num_blocks(m, block_size)
    if mode == "lower":
        owned_block_iterator = iter_lower_2d_block_cyclic_blocks_owned_by_rank(
            nblocks,
            process_grid_shape,
            rank,
        )
    else:
        owned_block_iterator = iter_2d_block_cyclic_blocks_owned_by_rank(
            nblocks,
            nblocks,
            process_grid_shape,
            rank,
        )

    owned_blocks = tuple(owned_block_iterator)

    for owned_bi, owned_bj in owned_blocks:
        owned_i0, owned_i1 = block_bounds(owned_bi, block_size, m)
        owned_j0, owned_j1 = block_bounds(owned_bj, block_size, m)
        owned_key = (owned_bi, owned_bj)
        expected_shape = (owned_i1 - owned_i0, owned_j1 - owned_j0)

        if owned_key not in left_preconditioner:
            raise ValueError(
                f"left_preconditioner is missing owned tile {owned_key}."
            )

        owned_tile = left_preconditioner[owned_key]

        if tuple(owned_tile.shape) != expected_shape:
            raise ValueError(
                f"left_preconditioner[{owned_key}] has shape "
                f"{tuple(owned_tile.shape)}, expected {expected_shape}."
            )

        if owned_tile.device != device:
            raise ValueError(
                f"left_preconditioner[{owned_key}] is on {owned_tile.device}, "
                f"expected {device}."
            )

        if owned_tile.dtype != dtype:
            raise ValueError(
                f"left_preconditioner[{owned_key}] has dtype "
                f"{owned_tile.dtype}, expected {dtype}."
            )

    current_flat = torch.empty(
        block_size * block_size,
        device=device,
        dtype=dtype,
    )
    transfer_flat = torch.empty_like(current_flat)

    for bi in range(nblocks):
        i0, i1 = block_bounds(bi, block_size, m)
        Ai = A_col_shard[i0:i1, :]
        tile_rows = i1 - i0

        for bj in range(bi + 1):
            j0, j1 = block_bounds(bj, block_size, m)
            Aj = A_col_shard[j0:j1, :]
            tile_cols = j1 - j0

            dst = block_cyclic_owner_rank(
                block_row=bi,
                block_col=bj,
                process_grid_shape=process_grid_shape,
            )
            key = (bi, bj)
            current = current_flat[: tile_rows * tile_cols].view(
                tile_rows,
                tile_cols,
            )

            if local_n == 0:
                current.zero_()
            else:
                torch.mm(Ai, Aj.T, out=current)

            dist.reduce(current, dst=dst, op=dist.ReduceOp.SUM)
            if rank == dst:
                left_preconditioner[key].lerp_(current, 1.0 - beta)

            if mode == "lower" or bi == bj:
                continue

            upper_key = (bj, bi)
            upper_dst = block_cyclic_owner_rank(
                block_row=bj,
                block_col=bi,
                process_grid_shape=process_grid_shape,
            )

            if upper_dst == dst:
                if rank == dst:
                    left_preconditioner[upper_key].lerp_(
                        current.T,
                        1.0 - beta,
                    )
                continue

            transfer = transfer_flat[: tile_rows * tile_cols].view(
                tile_rows,
                tile_cols,
            )
            request = None

            if rank == dst:
                send_tile = current.contiguous()
                request = dist.isend(send_tile, dst=upper_dst)
            elif rank == upper_dst:
                request = dist.irecv(transfer, src=dst)

            if request is not None:
                request.wait()

            if rank == upper_dst:
                left_preconditioner[upper_key].lerp_(
                    transfer.T,
                    1.0 - beta,
                )

    return left_preconditioner


def _update_ata_from_column_shards_2d_block_cyclic(
    A_col_shard: torch.Tensor,
    right_preconditioner: dict[tuple[int, int], torch.Tensor],
    beta: float,
    block_size: int,
    process_grid_shape: tuple[int, int],
    mode: Literal["lower", "full"] = "lower",
) -> dict[tuple[int, int], torch.Tensor]:
    """EMA-update owned tiles of ``A.T @ A`` from column shards of ``A``."""

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first.")

    if A_col_shard.ndim != 2:
        raise ValueError(f"A_col_shard must be 2D, got {A_col_shard.ndim}D.")

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    if mode not in {"lower", "full"}:
        raise ValueError(f"mode must be 'lower' or 'full', got {mode!r}.")

    Pr, Pc = process_grid_shape
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if Pr * Pc != world_size:
        raise ValueError(
            f"process_grid_shape={process_grid_shape} does not match "
            f"world_size={world_size}."
        )

    m, local_n = A_col_shard.shape
    device = A_col_shard.device
    dtype = A_col_shard.dtype

    local_shape = torch.tensor([m, local_n], device=device, dtype=torch.int64)
    gathered_shapes = [torch.empty_like(local_shape) for _ in range(world_size)]
    dist.all_gather(gathered_shapes, local_shape)

    row_counts = [shape[0].item() for shape in gathered_shapes]
    if any(rows != m for rows in row_counts):
        raise ValueError("A_col_shard must have the same row count on every rank.")

    shard_offsets = column_shard_offsets([shape[1].item() for shape in gathered_shapes])
    n = shard_offsets[-1]
    nblocks = num_blocks(n, block_size)

    if mode == "lower":
        owned_block_iterator = iter_lower_2d_block_cyclic_blocks_owned_by_rank(
            nblocks,
            process_grid_shape,
            rank,
        )
    else:
        owned_block_iterator = iter_2d_block_cyclic_blocks_owned_by_rank(
            nblocks,
            nblocks,
            process_grid_shape,
            rank,
        )

    for bi, bj in owned_block_iterator:
        i0, i1 = block_bounds(bi, block_size, n)
        j0, j1 = block_bounds(bj, block_size, n)
        key = (bi, bj)
        expected_shape = (i1 - i0, j1 - j0)

        if key not in right_preconditioner:
            raise ValueError(
                f"right_preconditioner is missing owned tile {key}."
            )

        out = right_preconditioner[key]

        if tuple(out.shape) != expected_shape:
            raise ValueError(
                f"right_preconditioner[{key}] has shape {tuple(out.shape)}, "
                f"expected {expected_shape}."
            )

        if out.device != device:
            raise ValueError(
                f"right_preconditioner[{key}] is on {out.device}, expected {device}."
            )

        if out.dtype != dtype:
            raise ValueError(
                f"right_preconditioner[{key}] has dtype {out.dtype}, expected {dtype}."
            )

    if n == 0:
        return right_preconditioner
    if m == 0:
        for tile in right_preconditioner.values():
            tile.mul_(beta)
        return right_preconditioner

    panel_rows = min(block_size, m)
    panel_cols = min(block_size, n)
    panel_capacity = panel_rows * panel_cols
    transfer_capacity = max(panel_capacity, panel_cols * panel_cols)

    panel_i_flat = torch.empty(panel_capacity, device=device, dtype=dtype)
    panel_j_flat = torch.empty_like(panel_i_flat)
    transfer_flat = torch.empty(transfer_capacity, device=device, dtype=dtype)
    current_flat = torch.empty(
        block_size * block_size,
        device=device,
        dtype=dtype,
    )

    for bi in range(nblocks):
        i0, i1 = block_bounds(bi, block_size, n)

        for bj in range(bi + 1):
            j0, j1 = block_bounds(bj, block_size, n)
            destination = block_cyclic_owner_rank(
                block_row=bi,
                block_col=bj,
                process_grid_shape=process_grid_shape,
            )
            tile_rows = i1 - i0
            tile_cols = j1 - j0
            current = current_flat[: tile_rows * tile_cols].view(
                tile_rows,
                tile_cols,
            )
            if rank == destination:
                current.zero_()

            for row_start in range(0, m, panel_rows):
                row_end = min(row_start + panel_rows, m)
                Ai = get_column_panel_from_col_shards(
                    A_col_shard,
                    row_start,
                    row_end,
                    i0,
                    i1,
                    shard_offsets,
                    destination,
                    rank,
                    panel_i_flat,
                    transfer_flat,
                )

                if bi == bj:
                    if rank == destination:
                        current.addmm_(Ai.T, Ai)
                    continue

                Aj = get_column_panel_from_col_shards(
                    A_col_shard,
                    row_start,
                    row_end,
                    j0,
                    j1,
                    shard_offsets,
                    destination,
                    rank,
                    panel_j_flat,
                    transfer_flat,
                )

                if rank == destination:
                    current.addmm_(Ai.T, Aj)

            if rank == destination:
                right_preconditioner[(bi, bj)].lerp_(
                    current,
                    1.0 - beta,
                )

            if mode == "lower" or bi == bj:
                continue

            upper_key = (bj, bi)
            upper_destination = block_cyclic_owner_rank(
                block_row=bj,
                block_col=bi,
                process_grid_shape=process_grid_shape,
            )

            if upper_destination == destination:
                if rank == destination:
                    right_preconditioner[upper_key].lerp_(
                        current.T,
                        1.0 - beta,
                    )
                continue

            transfer = transfer_flat[: tile_rows * tile_cols].view(
                tile_rows,
                tile_cols,
            )
            request = None

            if rank == destination:
                send_tile = current.contiguous()
                request = dist.isend(send_tile, dst=upper_destination)
            elif rank == upper_destination:
                request = dist.irecv(transfer, src=destination)

            if request is not None:
                request.wait()

            if rank == upper_destination:
                right_preconditioner[upper_key].lerp_(
                    transfer.T,
                    1.0 - beta,
                )

    return right_preconditioner


def _global_shape_from_tp_shard(
    G_local: Tensor,
    shard_dim: Literal[0, 1],
) -> tuple[int, int]:
    world_size = dist.get_world_size()
    local_shape = torch.tensor(
        G_local.shape,
        dtype=torch.int64,
        device=G_local.device,
    )
    shapes = [torch.empty_like(local_shape) for _ in range(world_size)]
    dist.all_gather(shapes, local_shape)
    dimensions = [tuple(int(value) for value in shape.tolist()) for shape in shapes]
    replicated_dim = 1 - shard_dim
    replicated_size = dimensions[0][replicated_dim]
    if any(shape[replicated_dim] != replicated_size for shape in dimensions):
        raise ValueError(
            f"non-sharded dimension {replicated_dim} must match on every rank."
        )

    result = list(dimensions[0])
    result[shard_dim] = sum(shape[shard_dim] for shape in dimensions)
    return result[0], result[1]


def _packed_preconditioner_views(
    name: str,
    preconditioner: Tensor,
    size: int,
    block_size: int,
    process_grid_shape: tuple[int, int],
    G_local: Tensor,
) -> dict[tuple[int, int], Tensor]:
    if preconditioner.dtype != torch.float32:
        raise ValueError(
            f"{name} must use float32 for ELPA and SLATE, got "
            f"{preconditioner.dtype}."
        )
    if preconditioner.device != G_local.device:
        raise ValueError(
            f"{name} is on {preconditioner.device}, expected {G_local.device}."
        )
    return block_cyclic_tile_views(
        preconditioner,
        (size, size),
        block_size,
        process_grid_shape,
        dist.get_rank(),
        mode="full",
    )


@torch.no_grad()
def update_left_preconditioner_2d_block_cyclic_(
    G_local: Tensor,
    left_preconditioner: Tensor,
    beta: float,
    block_size: int,
    process_grid_shape: tuple[int, int],
    *,
    shard_dim: Literal[0, 1],
) -> Tensor:
    """Update packed ``L = beta * L + (1 - beta) * G @ G.T`` in place."""
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first.")
    if G_local.ndim != 2:
        raise ValueError(f"G_local must be 2D, got {G_local.ndim}D.")
    if shard_dim not in (0, 1):
        raise ValueError(f"shard_dim must be 0 or 1, got {shard_dim}.")
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}.")

    rows, _ = _global_shape_from_tp_shard(G_local, shard_dim)
    tiles = _packed_preconditioner_views(
        "left_preconditioner",
        left_preconditioner,
        rows,
        block_size,
        process_grid_shape,
        G_local,
    )
    G_float = G_local.to(dtype=torch.float32)
    if shard_dim == 1:
        _update_aat_from_column_shards_2d_block_cyclic(
            G_float,
            tiles,
            beta,
            block_size,
            process_grid_shape,
            "full",
        )
    else:
        _update_ata_from_column_shards_2d_block_cyclic(
            G_float.transpose(0, 1).contiguous(),
            tiles,
            beta,
            block_size,
            process_grid_shape,
            "full",
        )
    return left_preconditioner


@torch.no_grad()
def update_right_preconditioner_2d_block_cyclic_(
    G_local: Tensor,
    right_preconditioner: Tensor,
    beta: float,
    block_size: int,
    process_grid_shape: tuple[int, int],
    *,
    shard_dim: Literal[0, 1],
) -> Tensor:
    """Update packed ``R = beta * R + (1 - beta) * G.T @ G`` in place."""
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first.")
    if G_local.ndim != 2:
        raise ValueError(f"G_local must be 2D, got {G_local.ndim}D.")
    if shard_dim not in (0, 1):
        raise ValueError(f"shard_dim must be 0 or 1, got {shard_dim}.")
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}.")

    _, columns = _global_shape_from_tp_shard(G_local, shard_dim)
    tiles = _packed_preconditioner_views(
        "right_preconditioner",
        right_preconditioner,
        columns,
        block_size,
        process_grid_shape,
        G_local,
    )
    G_float = G_local.to(dtype=torch.float32)
    if shard_dim == 1:
        _update_ata_from_column_shards_2d_block_cyclic(
            G_float,
            tiles,
            beta,
            block_size,
            process_grid_shape,
            "full",
        )
    else:
        _update_aat_from_column_shards_2d_block_cyclic(
            G_float.transpose(0, 1).contiguous(),
            tiles,
            beta,
            block_size,
            process_grid_shape,
            "full",
        )
    return right_preconditioner
