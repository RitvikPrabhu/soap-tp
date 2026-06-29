from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AdamBasisState:
    exp_avg: torch.Tensor
    exp_avg_sq: torch.Tensor
    step: int = 0


def adam_update_in_basis(
    grad_hat: torch.Tensor,
    state: AdamBasisState,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> tuple[torch.Tensor, AdamBasisState]:
    """Compute Adam-style update in the projected SOAP basis.

    Keep this separate from distributed SOAP logic so the numerical issue is isolated.
    """
    raise NotImplementedError("Implement after projection tests exist")
