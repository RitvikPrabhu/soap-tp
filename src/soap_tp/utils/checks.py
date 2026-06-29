from __future__ import annotations

import torch


def assert_matrix(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be rank-2, got shape={tuple(tensor.shape)}")


def assert_same_shape(name_a: str, a: torch.Tensor, name_b: str, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {name_a}{tuple(a.shape)} vs {name_b}{tuple(b.shape)}")
