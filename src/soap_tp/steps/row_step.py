from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from soap_tp.steps.state import MatrixSOAPState


def row_tp_soap_step(
    param_local: torch.Tensor,
    grad_local: torch.Tensor,
    state: MatrixSOAPState,
    *,
    lr: float,
    group: Optional[dist.ProcessGroup] = None,
) -> tuple[torch.Tensor, MatrixSOAPState]:
    """One SOAP update for a row-sharded matrix parameter.

    First-version orchestration target:
        1. compute cheap right Gram ``G.T @ G`` via all-reduce
        2. compute/refresh right basis
        3. handle left basis layout explicitly
        4. project grad
        5. Adam in projected basis
        6. project update back
        7. return updated local parameter shard

    Keep every math operation delegated to ``soap_tp.ops``.
    """
    raise NotImplementedError("Wire after op-level tests pass")
