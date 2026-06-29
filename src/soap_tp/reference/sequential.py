from __future__ import annotations

import torch


def gram_ggt(g: torch.Tensor) -> torch.Tensor:
    """Sequential golden ``G @ G.T``."""
    return g @ g.T


def gram_gtg(g: torch.Tensor) -> torch.Tensor:
    """Sequential golden ``G.T @ G``."""
    return g.T @ g


def eigh_symmetric(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential golden symmetric eigendecomposition."""
    return torch.linalg.eigh(matrix)


def project_gradient(g: torch.Tensor, q_left: torch.Tensor, q_right: torch.Tensor) -> torch.Tensor:
    """Sequential golden ``Q_L.T @ G @ Q_R``."""
    return q_left.T @ g @ q_right


def project_back(update_hat: torch.Tensor, q_left: torch.Tensor, q_right: torch.Tensor) -> torch.Tensor:
    """Sequential golden ``Q_L @ update_hat @ Q_R.T``."""
    return q_left @ update_hat @ q_right.T


def one_step_reference_placeholder(g: torch.Tensor) -> torch.Tensor:
    """Placeholder for a full one-step SOAP golden update.

    Replace this with the exact math once hyperparameters and state layout are frozen.
    """
    raise NotImplementedError("Define one-step SOAP reference once step semantics are fixed")
