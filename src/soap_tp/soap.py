from __future__ import annotations

import json
import math
import os
from typing import Any, Literal, MutableMapping

import torch
import torch.distributed as dist
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
from .ops._utils import block_cyclic_indices


def _soap_profile_enabled(step: int) -> bool:
    if os.environ.get("SOAP_PROFILE", "").lower() not in {"1", "true", "yes"}:
        return False
    requested_steps = os.environ.get("SOAP_PROFILE_STEPS")
    if not requested_steps:
        return True
    return step in {
        int(requested_step.strip())
        for requested_step in requested_steps.split(",")
        if requested_step.strip()
    }


def _soap_profile_tensor(stage: str, step: int, tensor: Tensor) -> None:
    if dist.get_rank() != 0:
        return
    detached = tensor.detach()
    print(
        "SOAP_PROFILE "
        + json.dumps(
            {
                "impl": "tp",
                "step": step,
                "stage": stage,
                "shape": list(detached.shape),
                "dtype": str(detached.dtype).removeprefix("torch."),
                "values": detached.cpu().reshape(-1).tolist(),
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


def _soap_profile_tp_shard(
    stage: str,
    step: int,
    tensor: Tensor,
    shard_dim: Literal[0, 1],
) -> None:
    if not _soap_profile_enabled(step):
        return
    shards = [
        torch.empty_like(tensor)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(shards, tensor.contiguous())
    _soap_profile_tensor(stage, step, torch.cat(shards, dim=shard_dim))


def _soap_profile_2d_block_cyclic(
    stage: str,
    step: int,
    tensor: Tensor,
    global_shape: tuple[int, int],
    block_size: int,
    process_grid_shape: tuple[int, int],
) -> None:
    if not _soap_profile_enabled(step):
        return
    rank = dist.get_rank()
    process_rows, process_columns = process_grid_shape
    process_row = rank // process_columns
    process_column = rank % process_columns
    rows, columns = global_shape
    row_indices = block_cyclic_indices(
        rows,
        block_size,
        process_row,
        process_rows,
    )
    column_indices = block_cyclic_indices(
        columns,
        block_size,
        process_column,
        process_columns,
    )
    full_tensor = tensor.new_zeros(global_shape)
    if row_indices and column_indices:
        row_index = torch.tensor(
            row_indices,
            dtype=torch.long,
            device=tensor.device,
        )
        column_index = torch.tensor(
            column_indices,
            dtype=torch.long,
            device=tensor.device,
        )
        full_tensor[row_index[:, None], column_index] = tensor[
            : len(row_indices),
            : len(column_indices),
        ]
    dist.all_reduce(full_tensor)
    _soap_profile_tensor(stage, step, full_tensor)


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
    profile_step = 0 if initializing else int(state["step"]) + 1
    _soap_profile_tp_shard(
        "gradient",
        profile_step,
        gradient_float,
        shard_dim,
    )

    if initializing:
        update_left_preconditioner_2d_block_cyclic_(
            gradient_float,
            state["left_preconditioner"],
            preconditioner_beta,
            block_size,
            process_grid_shape,
            shard_dim=shard_dim,
        )
        _soap_profile_2d_block_cyclic(
            "left_preconditioner",
            profile_step,
            state["left_preconditioner"],
            (rows, rows),
            block_size,
            process_grid_shape,
        )
        update_right_preconditioner_2d_block_cyclic_(
            gradient_float,
            state["right_preconditioner"],
            preconditioner_beta,
            block_size,
            process_grid_shape,
            shard_dim=shard_dim,
        )
        _soap_profile_2d_block_cyclic(
            "right_preconditioner",
            profile_step,
            state["right_preconditioner"],
            (columns, columns),
            block_size,
            process_grid_shape,
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
        _soap_profile_2d_block_cyclic(
            "left_basis",
            profile_step,
            state["left_basis"],
            (rows, rows),
            block_size,
            process_grid_shape,
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
        _soap_profile_2d_block_cyclic(
            "right_basis",
            profile_step,
            state["right_basis"],
            (columns, columns),
            block_size,
            process_grid_shape,
        )
        state["step"] = 0
        zero_update = torch.zeros_like(gradient_float)
        _soap_profile_tp_shard(
            "returned_update",
            profile_step,
            zero_update,
            shard_dim,
        )
        return zero_update

    step = int(state["step"]) + 1
    _soap_profile_2d_block_cyclic(
        "left_preconditioner",
        profile_step,
        state["left_preconditioner"],
        (rows, rows),
        block_size,
        process_grid_shape,
    )
    _soap_profile_2d_block_cyclic(
        "right_preconditioner",
        profile_step,
        state["right_preconditioner"],
        (columns, columns),
        block_size,
        process_grid_shape,
    )
    _soap_profile_2d_block_cyclic(
        "left_basis",
        profile_step,
        state["left_basis"],
        (rows, rows),
        block_size,
        process_grid_shape,
    )
    _soap_profile_2d_block_cyclic(
        "right_basis",
        profile_step,
        state["right_basis"],
        (columns, columns),
        block_size,
        process_grid_shape,
    )
    _soap_profile_2d_block_cyclic(
        "momentum_before_adam",
        profile_step,
        state["momentum"],
        global_shape,
        block_size,
        process_grid_shape,
    )
    _soap_profile_2d_block_cyclic(
        "variance_before_adam",
        profile_step,
        state["variance"],
        global_shape,
        block_size,
        process_grid_shape,
    )
    packed_gradient = redistribute_tp_shard_to_2d_block_cyclic(
        gradient_float,
        global_shape,
        block_size,
        process_grid_shape,
        shard_dim=shard_dim,
    )
    _soap_profile_2d_block_cyclic(
        "gradient_2d",
        profile_step,
        packed_gradient,
        global_shape,
        block_size,
        process_grid_shape,
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
    _soap_profile_2d_block_cyclic(
        "projected_gradient",
        profile_step,
        packed_gradient,
        global_shape,
        block_size,
        process_grid_shape,
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
    _soap_profile_2d_block_cyclic(
        "momentum_after_adam",
        profile_step,
        state["momentum"],
        global_shape,
        block_size,
        process_grid_shape,
    )
    _soap_profile_2d_block_cyclic(
        "variance_after_adam",
        profile_step,
        state["variance"],
        global_shape,
        block_size,
        process_grid_shape,
    )
    _soap_profile_2d_block_cyclic(
        "coordinate_update",
        profile_step,
        packed_update,
        global_shape,
        block_size,
        process_grid_shape,
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
    _soap_profile_2d_block_cyclic(
        "backrotated_update",
        profile_step,
        packed_update,
        global_shape,
        block_size,
        process_grid_shape,
    )
    update_shard = redistribute_2d_block_cyclic_to_tp_shard(
        packed_update,
        global_shape,
        block_size,
        process_grid_shape,
        shard_dim=shard_dim,
    )
    _soap_profile_tp_shard(
        "returned_update",
        profile_step,
        update_shard,
        shard_dim,
    )

    update_left_preconditioner_2d_block_cyclic_(
        gradient_float,
        state["left_preconditioner"],
        preconditioner_beta,
        block_size,
        process_grid_shape,
        shard_dim=shard_dim,
    )
    _soap_profile_2d_block_cyclic(
        "left_preconditioner_after_step",
        profile_step,
        state["left_preconditioner"],
        (rows, rows),
        block_size,
        process_grid_shape,
    )
    update_right_preconditioner_2d_block_cyclic_(
        gradient_float,
        state["right_preconditioner"],
        preconditioner_beta,
        block_size,
        process_grid_shape,
        shard_dim=shard_dim,
    )
    _soap_profile_2d_block_cyclic(
        "right_preconditioner_after_step",
        profile_step,
        state["right_preconditioner"],
        (columns, columns),
        block_size,
        process_grid_shape,
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

    _soap_profile_2d_block_cyclic(
        "left_basis_after_step",
        profile_step,
        state["left_basis"],
        (rows, rows),
        block_size,
        process_grid_shape,
    )
    _soap_profile_2d_block_cyclic(
        "right_basis_after_step",
        profile_step,
        state["right_basis"],
        (columns, columns),
        block_size,
        process_grid_shape,
    )
    _soap_profile_2d_block_cyclic(
        "momentum_after_step",
        profile_step,
        state["momentum"],
        global_shape,
        block_size,
        process_grid_shape,
    )
    _soap_profile_2d_block_cyclic(
        "variance_after_step",
        profile_step,
        state["variance"],
        global_shape,
        block_size,
        process_grid_shape,
    )
    state["step"] = step
    return update_shard


__all__ = ["soap_step"]
