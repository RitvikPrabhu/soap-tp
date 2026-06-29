from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist


def symmetric_eigh_replicated(
    matrix: torch.Tensor,
    *,
    group: Optional[dist.ProcessGroup] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Eigen-decompose a replicated symmetric matrix.

    Returns:
        eigenvalues, eigenvectors

    First version may simply call torch.linalg.eigh locally on every rank after
    Gram matrices are replicated. Later versions can replace this with distributed
    eigensolver experiments.
    """
    raise NotImplementedError("Implement local torch.linalg.eigh baseline first")
