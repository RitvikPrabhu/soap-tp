from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from ._utils import block_bounds, block_cyclic_owner_rank, num_blocks


def update_left_preconditioner_from_col_shards(
    G_local: Tensor,  # shape is [m, n_local]
    L_prev: Optional[Tensor],
    beta: float,
    *,
    TP_group: Optional[ProcessGroup] = None,  # Process group for tensor parallelism
) -> Tensor:

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

    if TP_group is None:
        TP_group = dist.group.WORLD

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


def update_left_preconditioner_from_col_shards_2DblockCyclic_lower(
    A_col_shard: torch.Tensor,
    left_preconditioner: dict[tuple[int, int], torch.Tensor],
    block_size: int,
    process_grid_shape: tuple[int, int],
) -> dict[tuple[int, int], torch.Tensor]:

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

            if dst == rank:
                key = (bi, bj)

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
