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
    TP_group: Optional[ProcessGroup] = None,
    reduce_scatter: bool = False, #reduce scatter or all_reduce
) -> Tensor:
    """
    Update the left preconditioner from column shards of G.

    Args:
        G_local: Local shard of G, shape is [m, n_local].
        L_prev: Previous left preconditioner, shape is [m, m].
        beta: Weight for the previous preconditioner.
        TP_group: Process group for tensor parallelism.
        reduce_scatter: Whether to use reduce_scatter or all_reduce.
    """
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be in [0, 1], got {beta}")
    
    if G_local.ndim != 2:
        raise ValueError(f"G_local must be 2D, got shape {tuple(G_local.shape)}")
    
    if TP_group is None:
        TP_group = dist.group.WORLD

    m = G_local.size(0)

    local_contribution = G_local @ G_local.T

    if reduce_scatter:
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

    #If it is not reduce_scatter, we do all_reduce
    if L_prev is None:
        L_prev = torch.zeros(
            m,
            m,
            device=G_local.device,
            dtype=G_local.dtype,
        )

    expected_shape = (m, m)

    if tuple(L_prev.shape) != expected_shape:
        raise ValueError(
            f"For reduce_scatter=False, L_prev must have shape {expected_shape}, "
            f"got {tuple(L_prev.shape)}."
        )

    output = local_contribution.clone()

    dist.all_reduce(
        output,
        op=dist.ReduceOp.SUM,
        group=TP_group,
    )

    L_prev.lerp_(output, 1.0 - beta)
    return L_prev

