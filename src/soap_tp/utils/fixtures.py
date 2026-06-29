from __future__ import annotations

import torch


def make_tiny_gradient(dtype: torch.dtype = torch.float64, device: str | torch.device = "cpu") -> torch.Tensor:
    """Deterministic non-symmetric small matrix for correctness tests."""
    return torch.tensor(
        [
            [1.0, 2.0, -1.0],
            [0.5, -3.0, 4.0],
            [2.0, 0.0, 1.5],
            [-1.0, 1.0, 0.25],
        ],
        dtype=dtype,
        device=device,
    )
