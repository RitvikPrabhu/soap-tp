from dataclasses import dataclass
from typing import Optional

import torch.distributed as dist


@dataclass(frozen=True)
class TPContext:
    """Tensor-parallel process metadata.

    Keep this tiny. Do not let optimizer state leak into the distributed context.
    """

    world_size: int
    rank: int
    group: Optional[dist.ProcessGroup] = None
    shard_axis: str = "row"  # "row" or "col"

    @staticmethod
    def from_default_group(shard_axis: str) -> "TPContext":
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before creating TPContext")
        if shard_axis not in {"row", "col"}:
            raise ValueError(f"unsupported shard_axis={shard_axis!r}")
        return TPContext(
            world_size=dist.get_world_size(),
            rank=dist.get_rank(),
            group=None,
            shard_axis=shard_axis,
        )
