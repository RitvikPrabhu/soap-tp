from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from soap_tp.steps.col_step import col_tp_soap_step
from soap_tp.steps.row_step import row_tp_soap_step
from soap_tp.steps.state import MatrixSOAPState


def matrix_soap_tp_step(
    param_local: torch.Tensor,
    grad_local: torch.Tensor,
    state: MatrixSOAPState,
    *,
    lr: float,
    shard_axis: str,
    group: Optional[dist.ProcessGroup] = None,
) -> tuple[torch.Tensor, MatrixSOAPState]:
    """Dispatch to row-wise or column-wise TP SOAP step."""
    if shard_axis == "row":
        return row_tp_soap_step(param_local, grad_local, state, lr=lr, group=group)
    if shard_axis == "col":
        return col_tp_soap_step(param_local, grad_local, state, lr=lr, group=group)
    raise ValueError(f"unsupported shard_axis={shard_axis!r}")
