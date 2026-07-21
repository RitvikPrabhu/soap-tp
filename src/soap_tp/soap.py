from __future__ import annotations

import math
from typing import Any, Literal, MutableMapping

import torch
from torch import Tensor

from .ops import (
    adam_update,
    allocate_2d_block_cyclic,
    initialize_basis_2d_block_cyclic_,
    redistribute_2d_block_cyclic_to_tp_shard,
    redistribute_tp_shard_to_2d_block_cyclic,
    refresh_bases_and_transport_optimizer_state_,
    rotate_2d_block_cyclic_,
    update_left_preconditioner_2d_block_cyclic_,
    update_right_preconditioner_2d_block_cyclic_,
)


@torch.no_grad()
def soap_step(
    gradient_shard: Tensor,
    state: MutableMapping[str, Any],
    *,
    global_shape: tuple[int, int],
    shard_dim: Literal[0, 1],
    block_size: int,
    process_grid_shape: tuple[int, int],
    preconditioner_beta: float = 0.95,
    beta1: float = 0.95,
    beta2: float = 0.95,
    eps: float = 1e-8,
    basis_refresh_interval: int = 10,
    elpa_binding: Any | None = None,
    slate_binding: Any | None = None,
) -> Tensor:
    """Run one SOAP update for a row- or column-sharded matrix gradient.

    ``state`` is a mutable per-parameter dictionary. It is populated lazily on
    the first call and reused on later calls, which makes this function usable
    from an external optimizer integration without depending on that framework.

    The first call initializes the preconditioners and bases and returns a zero
    update, matching the reference SOAP optimizer. Later calls return the
    normalized ``float32`` SOAP update for this rank's shard. Applying a
    learning rate, weight decay, and the update to the parameter remains the
    caller's responsibility.
    """
    if basis_refresh_interval <= 0:
        raise ValueError("basis_refresh_interval must be positive.")

    rows, columns = global_shape
    initializing = "left_basis" not in state

    if initializing:
        state_shapes = {
            "left_preconditioner": (rows, rows),
            "right_preconditioner": (columns, columns),
            "left_basis": (rows, rows),
            "right_basis": (columns, columns),
            "left_work": (rows, rows),
            "right_work": (columns, columns),
            "momentum": global_shape,
            "variance": global_shape,
        }
        for name, shape in state_shapes.items():
            state[name] = allocate_2d_block_cyclic(
                shape,
                block_size,
                process_grid_shape,
                device=gradient_shard.device,
            )

    gradient_float = gradient_shard.detach().to(torch.float32).contiguous()

    if initializing:
        update_left_preconditioner_2d_block_cyclic_(
            gradient_float,
            state["left_preconditioner"],
            preconditioner_beta,
            block_size,
            process_grid_shape,
            shard_dim=shard_dim,
        )
        update_right_preconditioner_2d_block_cyclic_(
            gradient_float,
            state["right_preconditioner"],
            preconditioner_beta,
            block_size,
            process_grid_shape,
            shard_dim=shard_dim,
        )
        initialize_basis_2d_block_cyclic_(
            state["left_preconditioner"],
            state["left_basis"],
            state["left_work"],
            torch.empty(rows, dtype=torch.float32, device=gradient_shard.device),
            rows,
            block_size,
            process_grid_shape,
            elpa_binding=elpa_binding,
        )
        initialize_basis_2d_block_cyclic_(
            state["right_preconditioner"],
            state["right_basis"],
            state["right_work"],
            torch.empty(
                columns,
                dtype=torch.float32,
                device=gradient_shard.device,
            ),
            columns,
            block_size,
            process_grid_shape,
            elpa_binding=elpa_binding,
        )
        state["step"] = 0
        return torch.zeros_like(gradient_float)

    step = int(state["step"]) + 1
    packed_gradient = redistribute_tp_shard_to_2d_block_cyclic(
        gradient_float,
        global_shape,
        block_size,
        process_grid_shape,
        shard_dim=shard_dim,
    )
    rotate_2d_block_cyclic_(
        packed_gradient,
        state["left_basis"],
        state["right_basis"],
        global_shape,
        block_size,
        process_grid_shape,
        direction="forward",
        slate_binding=slate_binding,
    )
    packed_update = adam_update(
        packed_gradient,
        state["momentum"],
        state["variance"],
        step,
        beta1,
        beta2,
        eps / math.sqrt(1.0 - beta2**step),
    )
    rotate_2d_block_cyclic_(
        packed_update,
        state["left_basis"],
        state["right_basis"],
        global_shape,
        block_size,
        process_grid_shape,
        direction="backward",
        slate_binding=slate_binding,
    )
    update_shard = redistribute_2d_block_cyclic_to_tp_shard(
        packed_update,
        global_shape,
        block_size,
        process_grid_shape,
        shard_dim=shard_dim,
    )

    update_left_preconditioner_2d_block_cyclic_(
        gradient_float,
        state["left_preconditioner"],
        preconditioner_beta,
        block_size,
        process_grid_shape,
        shard_dim=shard_dim,
    )
    update_right_preconditioner_2d_block_cyclic_(
        gradient_float,
        state["right_preconditioner"],
        preconditioner_beta,
        block_size,
        process_grid_shape,
        shard_dim=shard_dim,
    )
    if step % basis_refresh_interval == 0:
        refresh_bases_and_transport_optimizer_state_(
            state["momentum"],
            state["variance"],
            state["left_preconditioner"],
            state["right_preconditioner"],
            state["left_basis"],
            state["right_basis"],
            state["left_work"],
            state["right_work"],
            global_shape,
            block_size,
            process_grid_shape,
            slate_binding=slate_binding,
        )

    state["step"] = step
    return update_shard


__all__ = ["soap_step"]
