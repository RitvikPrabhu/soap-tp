from __future__ import annotations

import torch


def project_gradient_matrix(
    g_local: torch.Tensor,
    q_left: torch.Tensor,
    q_right: torch.Tensor,
    *,
    shard_axis: str,
) -> torch.Tensor:
    """Project local gradient shard into SOAP basis.

    Full-matrix math:
        G_hat = Q_L.T @ G @ Q_R

    Sharding contract must be explicit:
        - row shard: local G is [m_local, n]; Q_L access/layout is nontrivial.
        - col shard: local G is [m, n_local]; Q_R access/layout is nontrivial.

    Do not implement until basis storage layout is decided.
    """
    raise NotImplementedError("Define Q_L/Q_R storage layout before implementing")


def project_update_back_matrix(
    update_hat_local: torch.Tensor,
    q_left: torch.Tensor,
    q_right: torch.Tensor,
    *,
    shard_axis: str,
) -> torch.Tensor:
    """Map SOAP-basis update back to parameter-gradient space.

    Full-matrix math:
        Delta = Q_L @ update_hat @ Q_R.T
    """
    raise NotImplementedError("Define Q_L/Q_R storage layout before implementing")
