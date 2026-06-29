from __future__ import annotations

import torch


def shard_rows(g: torch.Tensor, world_size: int) -> list[torch.Tensor]:
    """Split ``G`` along rows for tests.

    Uses torch.tensor_split so uneven sizes are handled at the fixture level.
    """
    return list(torch.tensor_split(g, world_size, dim=0))


def shard_cols(g: torch.Tensor, world_size: int) -> list[torch.Tensor]:
    """Split ``G`` along columns for tests."""
    return list(torch.tensor_split(g, world_size, dim=1))
