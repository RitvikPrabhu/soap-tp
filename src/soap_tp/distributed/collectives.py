from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist


def all_reduce_sum(tensor: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
    """Return a summed copy of `tensor` across the process group.

    Contract:
        - Input is not modified.
        - Output has same shape/dtype/device as input.
        - Every rank receives the same value.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")
    out = tensor.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
    return out


def all_gather_tensor(tensor: torch.Tensor, group: Optional[dist.ProcessGroup] = None) -> list[torch.Tensor]:
    """Gather same-shaped tensors from every TP rank.

    For unequal shards, add a padded/all_gather_v variant later. Do not hide padding
    logic inside operation code.
    """
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")
    world_size = dist.get_world_size(group=group)
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor, group=group)
    return gathered
