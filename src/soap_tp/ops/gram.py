from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist


def distributed_gtg_from_row_shards(
    g_local: torch.Tensor,
    *,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Compute replicated ``G.T @ G`` when ``G`` is row-sharded.

    Shape contract:
        full G:      [m, n]
        local shard: [m_local, n]
        output:      [n, n], replicated on every rank

    First implementation target:
        local = g_local.T @ g_local
        output = all_reduce_sum(local)
    """
    raise NotImplementedError("Implement after test_reference_gram_ops is passing")


def distributed_ggt_from_col_shards(
    g_local: torch.Tensor,
    *,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Compute replicated ``G @ G.T`` when ``G`` is column-sharded.

    Shape contract:
        full G:      [m, n]
        local shard: [m, n_local]
        output:      [m, m], replicated on every rank

    First implementation target:
        local = g_local @ g_local.T
        output = all_reduce_sum(local)
    """
    raise NotImplementedError("Implement after test_reference_gram_ops is passing")


def distributed_ggt_from_row_shards_blocked(
    g_local: torch.Tensor,
    *,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Compute block-row view of ``G @ G.T`` when ``G`` is row-sharded.

    This is communication-heavy because each rank needs other row shards to form
    off-diagonal blocks. Keep it separate from the cheap all-reduce Gram paths.
    """
    raise NotImplementedError("Define desired output layout before implementing")


def distributed_gtg_from_col_shards_blocked(
    g_local: torch.Tensor,
    *,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Compute block-column/block-row view of ``G.T @ G`` when ``G`` is column-sharded.

    This is communication-heavy because off-diagonal blocks require cross-shard dot products.
    """
    raise NotImplementedError("Define desired output layout before implementing")
