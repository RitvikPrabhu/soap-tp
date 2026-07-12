from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from ._utils import (
    block_bounds,
    block_cyclic_owner_rank,
    column_shard_offsets,
    get_column_panel_from_col_shards,
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


def lerp_preconditioner_2DblockCyclic_lower_(
    preconditioner_prev: dict[tuple[int, int], Tensor],
    preconditioner_current: dict[tuple[int, int], Tensor],
    beta: float,
) -> dict[tuple[int, int], Tensor]:
    """Interpolate matching lower-triangular block dictionaries in place.

    For each tile key ``k``, this computes
    ``prev[k] = beta * prev[k] + (1 - beta) * current[k]``. The dictionary and
    its existing tile tensors retain their identities.

    Args:
        preconditioner_prev: Mutable mapping from ``(block_row, block_col)`` to
            previous tile tensors. Each tile is modified in place.
        preconditioner_current: Mapping with exactly the same keys and
            tensor-compatible current tiles. It is read but not intentionally
            mutated.
        beta: Previous-value weight in ``[0, 1]``. Zero selects the current
            tiles; one preserves the previous tiles.

    Returns:
        ``preconditioner_prev`` itself after all tile values are interpolated.

    Raises:
        ValueError: If the two dictionaries do not have identical key sets.
        RuntimeError: If corresponding tensors are incompatible for in-place
            ``Tensor.lerp_``, including incompatible shapes, dtypes, devices,
            or unsupported overlapping storage.

    Contract:
        Preconditions:
            - Corresponding tile tensors are compatible with in-place linear
              interpolation.

        Guarantees:
            - The returned dictionary is the same object as
              ``preconditioner_prev``.
            - Existing previous tile tensors are updated in place according to
              the stated interpolation formula.
            - An empty pair of dictionaries is returned unchanged.

        Invariants:
            - ``beta == 0`` yields the current tile values, and ``beta == 1``
              preserves the previous tile values.

        Unchecked assumptions:
            - ``beta`` is not range-checked.
            - Keys are not checked to satisfy ``block_row >= block_col``.
            - Distinct current and previous tiles are assumed not to use an
              unsupported partially overlapping storage layout.
    """

    if preconditioner_prev.keys() != preconditioner_current.keys():
        raise ValueError(
            "preconditioner_prev and preconditioner_current must have "
            "the same tile keys."
        )

    for key, current_tile in preconditioner_current.items():
        preconditioner_prev[key].lerp_(current_tile, 1.0 - beta)

    return preconditioner_prev


def update_left_preconditioner_from_col_shards_2DblockCyclic_lower(
    A_col_shard: torch.Tensor,
    left_preconditioner: dict[tuple[int, int], torch.Tensor],
    block_size: int,
    process_grid_shape: tuple[int, int],
) -> dict[tuple[int, int], torch.Tensor]:
    """Compute owned lower-triangular blocks of the left Gram matrix.

    The global matrix ``A`` has shape ``[m, n]`` and is column-sharded across
    the default distributed world. This function sums each rank's local
    ``A_col_shard @ A_col_shard.T`` contribution and stores only blocks
    ``(bi, bj)`` with ``bi >= bj``. For a process grid ``(Pr, Pc)``, block
    ``(bi, bj)`` is owned by row-major rank
    ``(bi % Pr) * Pc + (bj % Pc)``.

    Args:
        A_col_shard: This rank's columns of ``A``, with shape
            ``[m, n_local]``. Ranks may have different ``n_local`` values but
            must agree on ``m``, dtype, and compatible devices.
        left_preconditioner: Mutable dictionary of this rank's owned tiles,
            keyed by ``(block_row, block_col)``. Missing owned tiles are
            allocated; existing owned tiles are overwritten in place.
        block_size: Positive row and column extent of a full tile. A final tile
            may be smaller, so ``m`` need not be divisible by ``block_size``.
        process_grid_shape: Positive ``(Pr, Pc)`` grid interpreted in row-major
            rank order. Its product must equal the default world size.

    Returns:
        The same ``left_preconditioner`` dictionary. For every lower-triangular
        block owned by this rank, tile ``(bi, bj)`` equals the corresponding
        block of ``A @ A.T`` and has the input dtype and device. A rank that
        owns no blocks receives no newly allocated tiles.

    Raises:
        RuntimeError: If ``torch.distributed`` is not initialized, or if a
            collective or tensor operation fails.
        ValueError: If ``A_col_shard`` is not two-dimensional, ``block_size``
            is not positive, ``Pr * Pc`` differs from the world size, or an
            existing owned tile has the wrong shape, dtype, or device.

    Contract:
        Preconditions:
            - Every world rank calls the function in the same order with the
              same ``m``, ``block_size``, and ``process_grid_shape``.
            - Local shards collectively represent a column partition of one
              global matrix and are compatible with the distributed backend.

        Guarantees:
            - Every lower-triangular block has exactly one owner under the
              stated mapping.
            - Owned tiles are allocated or overwritten; their tensor objects
              are reused when already present and valid.
            - Zero-width local shards contribute zero. Partial edge tiles are
              supported without padding.

        Invariants:
            - The represented lower triangle agrees with the symmetric,
              positive-semidefinite dense matrix ``A @ A.T``.
            - Scaling all shards by ``alpha`` scales every tile by
              ``alpha**2``; all-zero shards produce zero tiles.
            - The mathematical result is independent of any legal process-grid
              shape, although ownership changes with the grid.

        Unchecked assumptions:
            - ``Pr`` and ``Pc`` are positive before ownership is evaluated;
              only their product is checked directly by this function.
            - The input dictionary contains only tiles intended for this rank;
              unrelated or stale keys are not removed.
            - All ranks have the same row count. This is required for matching
              collective sequences but is not checked collectively here.
    """

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first.")

    if A_col_shard.ndim != 2:
        raise ValueError(f"A_col_shard must be 2D, got {A_col_shard.ndim}D.")

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

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
    owned_blocks = set(
        iter_lower_2d_block_cyclic_blocks_owned_by_rank(
            nblocks,
            process_grid_shape,
            rank,
        )
    )

    tmp_flat = torch.empty(
        block_size * block_size,
        device=device,
        dtype=dtype,
    )

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

            if key in owned_blocks:
                if key not in left_preconditioner:
                    left_preconditioner[key] = torch.empty(
                        tile_rows,
                        tile_cols,
                        device=device,
                        dtype=dtype,
                    )

                out = left_preconditioner[key]

                expected_shape = (tile_rows, tile_cols)
                if tuple(out.shape) != expected_shape:
                    raise ValueError(
                        f"left_preconditioner[{key}] has shape "
                        f"{tuple(out.shape)}, expected {expected_shape}."
                    )

                if out.device != device:
                    raise ValueError(
                        f"left_preconditioner[{key}] is on {out.device}, "
                        f"expected {device}."
                    )

                if out.dtype != dtype:
                    raise ValueError(
                        f"left_preconditioner[{key}] has dtype {out.dtype}, "
                        f"expected {dtype}."
                    )

            else:
                out = tmp_flat[: tile_rows * tile_cols].view(tile_rows, tile_cols)

            if local_n == 0:
                out.zero_()
            else:
                torch.mm(Ai, Aj.T, out=out)

            dist.reduce(out, dst=dst, op=dist.ReduceOp.SUM)

    return left_preconditioner


def update_right_preconditioner_from_col_shards_2DBlockCyclic_lower(
    A_col_shard: torch.Tensor,
    right_preconditioner: dict[tuple[int, int], torch.Tensor],
    block_size: int,
    process_grid_shape: tuple[int, int],
) -> dict[tuple[int, int], torch.Tensor]:
    """Compute owned lower-triangular blocks of the right Gram matrix.

    The global matrix ``A`` has shape ``[m, n]`` and is column-sharded across
    the default distributed world, with uneven and empty column shards
    permitted. This function forms lower-triangular blocks of ``A.T @ A``.
    For process grid ``(Pr, Pc)``, block ``(bi, bj)`` is owned by row-major rank
    ``(bi % Pr) * Pc + (bj % Pc)``.

    Args:
        A_col_shard: This rank's contiguous range of global columns, with shape
            ``[m, n_local]``. All ranks must use the same ``m``, dtype, and
            compatible devices; ``n_local`` may vary by rank.
        right_preconditioner: Mutable dictionary of this rank's owned tiles,
            keyed by ``(block_row, block_col)``. Missing owned tiles are
            allocated; existing owned tiles are zeroed and refilled in place.
        block_size: Positive row and column extent of a full tile. The global
            column count need not be divisible by it.
        process_grid_shape: Positive ``(Pr, Pc)`` grid interpreted in row-major
            rank order. Its product must equal the default world size.

    Returns:
        The same ``right_preconditioner`` dictionary. Each owned key ``(bi,
        bj)`` with ``bi >= bj`` maps to the corresponding block of ``A.T @ A``
        on the input device and with the input dtype. A rank may own no tiles.

    Raises:
        RuntimeError: If ``torch.distributed`` is not initialized, or if a
            collective, point-to-point, or tensor operation fails.
        ValueError: If ``A_col_shard`` is not two-dimensional, ``block_size``
            is not positive, ``Pr * Pc`` differs from the world size, ranks
            report different row counts, or an existing owned tile has the
            wrong shape, dtype, or device.

    Contract:
        Preconditions:
            - Every world rank calls the function in the same communication
              order with the same ``block_size`` and ``process_grid_shape``.
            - Rank-ordered local shards are contiguous column intervals whose
              concatenation is the global matrix ``A``.
            - Tensor dtypes and devices are compatible with matrix operations
              and the selected distributed backend.

        Guarantees:
            - Global shard widths are gathered, so uneven and zero-width column
              shards contribute according to their rank-ordered offsets.
            - Every lower-triangular block has exactly one owner under the
              stated mapping, and owned tile tensors are reused when valid.
            - Partial edge tiles are supported. If ``m == 0``, owned tiles are
              zero; if ``n == 0``, no new tiles are allocated.

        Invariants:
            - The represented lower triangle agrees with the symmetric,
              positive-semidefinite dense matrix ``A.T @ A``.
            - Scaling all shards by ``alpha`` scales every tile by
              ``alpha**2``; zero input produces zero tiles.
            - The mathematical result is independent of row-panel boundaries
              and any legal process-grid shape; only ownership changes.

        Unchecked assumptions:
            - ``Pr`` and ``Pc`` are positive before ownership is evaluated;
              only their product is checked directly by this function.
            - The input dictionary contains only tiles intended for this rank;
              unrelated or stale keys are not removed.

        Unresolved assumptions:
            - Cross-rank dtype and device mismatches are not diagnosed with a
              dedicated validation error; backend behavior determines failure.
    """

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first.")

    if A_col_shard.ndim != 2:
        raise ValueError(f"A_col_shard must be 2D, got {A_col_shard.ndim}D.")

    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

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

    for bi, bj in iter_lower_2d_block_cyclic_blocks_owned_by_rank(
        nblocks,
        process_grid_shape,
        rank,
    ):
        i0, i1 = block_bounds(bi, block_size, n)
        j0, j1 = block_bounds(bj, block_size, n)
        key = (bi, bj)
        expected_shape = (i1 - i0, j1 - j0)

        if key not in right_preconditioner:
            right_preconditioner[key] = torch.empty(
                expected_shape,
                device=device,
                dtype=dtype,
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

        out.zero_()

    if m == 0 or n == 0:
        return right_preconditioner

    panel_rows = min(block_size, m)
    panel_cols = min(block_size, n)
    panel_capacity = panel_rows * panel_cols

    panel_i_flat = torch.empty(panel_capacity, device=device, dtype=dtype)
    panel_j_flat = torch.empty_like(panel_i_flat)
    transfer_flat = torch.empty_like(panel_i_flat)

    for bi in range(nblocks):
        i0, i1 = block_bounds(bi, block_size, n)

        for bj in range(bi + 1):
            j0, j1 = block_bounds(bj, block_size, n)
            destination = block_cyclic_owner_rank(
                block_row=bi,
                block_col=bj,
                process_grid_shape=process_grid_shape,
            )

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
                        right_preconditioner[(bi, bj)].addmm_(Ai.T, Ai)
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
                    right_preconditioner[(bi, bj)].addmm_(Ai.T, Aj)

    return right_preconditioner
