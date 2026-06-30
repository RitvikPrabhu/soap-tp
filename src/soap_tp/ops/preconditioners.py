from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup


def update_left_preconditioner_from_col_shards(
    G_local: Tensor, #shape is [m, n_local]
    L_prev: Optional[Tensor],
    beta: float,
    *,
    TP_group: Optional[ProcessGroup] = None #Process group for tensor parallelism
) -> Tensor:
    
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    
    if G_local.ndim != 2:
        raise ValueError(f"G_local must be 2D, got shape {tuple(G_local.shape)}")
    
    if TP_group is None:
        TP_group = dist.group.WORLD

    m = G_local.size(0)

    local_contribution = G_local @ G_local.T

    
    world_size = dist.get_world_size(group=TP_group)

    if m % world_size != 0:
        raise ValueError(
            f"reduce_scatter=True requires m to be divisible by TP world size. "
            f"Got m={m}, world_size={world_size}."
        )
    
    rows_per_rank = m // world_size
    expected_shape = (rows_per_rank, m)

    if L_prev is None:
        L_prev = torch.zeros(expected_shape, device=G_local.device, dtype=G_local.dtype)

    if tuple(L_prev.shape) != expected_shape:
        raise ValueError(
            f"For reduce_scatter=True, L_prev must have shape {expected_shape}, "
            f"got {tuple(L_prev.shape)}."
        )

    output = torch.zeros_like(L_prev)

    input_chunks = list(local_contribution.chunk(world_size, dim=0))

    dist.reduce_scatter(
        output,
        input_chunks,
        op=dist.ReduceOp.SUM,
        group=TP_group,
    )

    L_prev.lerp_(output, 1.0 - beta)
    return L_prev


def update_right_preconditioner_from_col_shards(
    G_local: Tensor, #shape is [m, n_local]
    R_prev: Optional[Tensor],
    beta: float,
    *,
    TP_group: Optional[ProcessGroup] = None #Process group for tensor parallelism
) -> Tensor:
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    
    if G_local.ndim != 2:
        raise ValueError(f"G_local must be 2D, got shape {tuple(G_local.shape)}")
    
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

    if tuple(R_prev.shape) != expected_shape:
        raise ValueError(
            f"R_prev must have shape {expected_shape}, got {tuple(R_prev.shape)}."
        )

    R_current = torch.zeros_like(R_prev)
    # Ring direction:
    #
    # Rank r starts with G_r.
    # Then it receives G_{r+1}, G_{r+2}, ..., wrapping around.
    # To get that order, rank r receives from r+1 and sends to r-1.

    send_to = (tp_rank - 1) % world_size
    recv_from = (tp_rank + 1) % world_size

    current_G = G_local.contiguous()
    current_owner = tp_rank

    for step in range(world_size):
        block = G_local.T @ current_G  # [n_local, n_local]

        col_start = current_owner * n_local
        col_end = col_start + n_local
        R_current[:, col_start:col_end] = block

        if step == world_size - 1:
            break

        recv_G = torch.empty_like(G_local)

        recv_op = dist.P2POp(
            dist.irecv,
            recv_G,
            group=TP_group,
            group_peer=recv_from,
        )

        send_op = dist.P2POp(
            dist.isend,
            current_G,
            group=TP_group,
            group_peer=send_to,
        )

        reqs = dist.batch_isend_irecv([recv_op, send_op])

        for req in reqs:
            req.wait()

        current_G = recv_G
        current_owner = (current_owner + 1) % world_size

    R_prev.lerp_(R_current, 1.0 - beta)
    return R_prev

