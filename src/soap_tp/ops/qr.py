from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist


def qr_basis_refresh_replicated(
    basis: torch.Tensor,
    gram: torch.Tensor,
    *,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Refresh an orthogonal basis using one power/QR-style update.

    Intended SOAP-style contract:
        updated = gram @ basis
        q, _ = torch.linalg.qr(updated)
        return q

    Keep this replicated first. Distributed QR is a separate research/optimization path.
    """
    raise NotImplementedError("Implement replicated QR baseline first")
