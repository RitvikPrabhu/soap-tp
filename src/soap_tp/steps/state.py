from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class MatrixSOAPState:
    """State for one matrix parameter under SOAP-TP.

    Store optimizer state separately from process-group context.
    """

    q_left: Optional[torch.Tensor] = None
    q_right: Optional[torch.Tensor] = None
    left_gram: Optional[torch.Tensor] = None
    right_gram: Optional[torch.Tensor] = None
    exp_avg: Optional[torch.Tensor] = None
    exp_avg_sq: Optional[torch.Tensor] = None
    step: int = 0
    metadata: dict = field(default_factory=dict)
