import math
from typing import Literal, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup

from ._utils import block_cyclic_indices


@torch.no_grad()
def adam_update(
    gradient: Tensor,
    momentum: Tensor,
    variance: Tensor,
    step: int,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> Tensor:
    """Update local Adam state and return the bias-corrected update."""
    if step < 1:
        raise ValueError(f"step must be positive, got {step}.")
    if not 0.0 <= beta1 < 1.0:
        raise ValueError(f"beta1 must be in [0, 1), got {beta1}.")
    if not 0.0 <= beta2 < 1.0:
        raise ValueError(f"beta2 must be in [0, 1), got {beta2}.")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")
    if gradient.shape != momentum.shape or gradient.shape != variance.shape:
        raise ValueError(
            "gradient, momentum, and variance must have identical shapes."
        )

    momentum.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
    variance.mul_(beta2).addcmul_(
        gradient, gradient, value=1.0 - beta2
    )

    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    denominator = variance.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
    return denominator.reciprocal_().mul_(momentum).div_(bias_correction1)


def _validated_permutation(
    order: Sequence[int] | Tensor,
    size: int,
    name: str,
) -> Tuple[int, ...]:
    if isinstance(order, Tensor):
        if order.ndim != 1:
            raise ValueError(f"{name} must be 1D, got {order.ndim}D.")
        values = tuple(int(index) for index in order.tolist())
    else:
        values = tuple(int(index) for index in order)

    if len(values) != size or sorted(values) != list(range(size)):
        raise ValueError(f"{name} must be a permutation of range({size}).")
    return values


def redistribute_2d_block_cyclic_to_tp_shard(
    matrix: Tensor,
    global_shape: Tuple[int, int],
    block_size: int,
    process_grid_shape: Tuple[int, int],
    *,
    shard_dim: Literal[0, 1],
    group: Optional[ProcessGroup] = None,
) -> Tensor:
    """Redistribute a packed 2D block-cyclic matrix into equal TP shards.

    ``shard_dim=0`` returns a contiguous row shard with shape
    ``[global_rows / world_size, global_columns]``. ``shard_dim=1`` returns a
    contiguous column shard with shape
    ``[global_rows, global_columns / world_size]``.
    """
    if group is None:
        group = dist.group.WORLD

    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got {matrix.ndim}D.")
    if shard_dim not in (0, 1):
        raise ValueError(f"shard_dim must be 0 or 1, got {shard_dim}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    process_rows, process_columns = process_grid_shape
    rows, columns = global_shape

    if rows < 0 or columns < 0:
        raise ValueError(f"global_shape must be non-negative, got {global_shape}.")
    if process_rows <= 0 or process_columns <= 0:
        raise ValueError(
            "process_grid_shape must contain positive dimensions, got "
            f"{process_grid_shape}."
        )
    if process_rows * process_columns != world_size:
        raise ValueError("process grid must contain every process-group rank.")

    global_dimensions = (rows, columns)
    sharded_size = global_dimensions[shard_dim]
    if sharded_size % world_size != 0:
        raise ValueError(
            f"global dimension {shard_dim} must be divisible by the world size."
        )

    shard_size = sharded_size // world_size
    destination_start = rank * shard_size
    destination_end = destination_start + shard_size
    process_row = rank // process_columns
    process_column = rank % process_columns
    local_rows = block_cyclic_indices(
        rows, block_size, process_row, process_rows
    )
    local_columns = block_cyclic_indices(
        columns, block_size, process_column, process_columns
    )
    if matrix.size(0) < len(local_rows) or matrix.size(1) < len(local_columns):
        raise ValueError(
            f"matrix shape {tuple(matrix.shape)} is smaller than its logical "
            f"block-cyclic shape {(len(local_rows), len(local_columns))}."
        )
    logical_matrix = matrix[: len(local_rows), : len(local_columns)]

    send_parts = []
    input_splits = []
    local_sharded_indices = (local_rows, local_columns)[shard_dim]
    for destination in range(world_size):
        start = destination * shard_size
        end = start + shard_size
        positions = [
            position for position, index in enumerate(local_sharded_indices)
            if start <= index < end
        ]
        if positions:
            index = torch.tensor(positions, device=matrix.device)
            part = (
                logical_matrix.index_select(shard_dim, index)
                .contiguous()
                .view(-1)
            )
        else:
            part = matrix.new_empty(0)
        send_parts.append(part)
        input_splits.append(part.numel())

    send_buffer = torch.cat(send_parts)
    output_splits = []
    source_layouts = []
    for source in range(world_size):
        source_row = source // process_columns
        source_column = source % process_columns
        source_rows = block_cyclic_indices(
            rows, block_size, source_row, process_rows
        )
        source_columns = tuple(
            column
            for column in block_cyclic_indices(
                columns, block_size, source_column, process_columns
            )
        )
        if shard_dim == 0:
            source_rows = tuple(
                row
                for row in source_rows
                if destination_start <= row < destination_end
            )
        else:
            source_columns = tuple(
                column
                for column in source_columns
                if destination_start <= column < destination_end
            )
        output_splits.append(len(source_rows) * len(source_columns))
        source_layouts.append((source_rows, source_columns))

    receive_buffer = matrix.new_empty(sum(output_splits))
    dist.all_to_all_single(
        receive_buffer,
        send_buffer,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )

    output_shape = [rows, columns]
    output_shape[shard_dim] = shard_size
    output = matrix.new_empty(tuple(output_shape))
    offset = 0
    for size, (source_rows, source_columns) in zip(
        output_splits, source_layouts
    ):
        if size:
            output_rows = (
                [row - destination_start for row in source_rows]
                if shard_dim == 0
                else source_rows
            )
            output_columns = (
                [column - destination_start for column in source_columns]
                if shard_dim == 1
                else source_columns
            )
            row_index = torch.tensor(output_rows, device=matrix.device)
            column_index = torch.tensor(
                output_columns,
                device=matrix.device,
            )
            output[row_index[:, None], column_index] = receive_buffer[
                offset : offset + size
            ].view(len(source_rows), len(source_columns))
        offset += size

    return output


def redistribute_tp_shard_to_2d_block_cyclic(
    matrix: Tensor,
    global_shape: Tuple[int, int],
    block_size: int,
    process_grid_shape: Tuple[int, int],
    *,
    shard_dim: Literal[0, 1],
    group: Optional[ProcessGroup] = None,
) -> Tensor:
    """Redistribute equal row or column TP shards into a packed matrix.

    The returned tensor stores this rank's 2D block-cyclic entries in a
    column-major local buffer suitable for the SLATE bindings. Its logical
    shape is determined by the process-grid coordinates; a dummy row or column
    is allocated when a rank owns an empty local dimension.
    """
    if group is None:
        group = dist.group.WORLD

    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got {matrix.ndim}D.")
    if shard_dim not in (0, 1):
        raise ValueError(f"shard_dim must be 0 or 1, got {shard_dim}.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    process_rows, process_columns = process_grid_shape
    rows, columns = global_shape

    if rows < 0 or columns < 0:
        raise ValueError(f"global_shape must be non-negative, got {global_shape}.")
    if process_rows <= 0 or process_columns <= 0:
        raise ValueError(
            "process_grid_shape must contain positive dimensions, got "
            f"{process_grid_shape}."
        )
    if process_rows * process_columns != world_size:
        raise ValueError("process grid must contain every process-group rank.")

    global_dimensions = (rows, columns)
    sharded_size = global_dimensions[shard_dim]
    if sharded_size % world_size != 0:
        raise ValueError(
            f"global dimension {shard_dim} must be divisible by the world size."
        )

    shard_size = sharded_size // world_size
    expected_shape = [rows, columns]
    expected_shape[shard_dim] = shard_size
    if tuple(matrix.shape) != tuple(expected_shape):
        raise ValueError(
            f"matrix has shape {tuple(matrix.shape)}, expected "
            f"{tuple(expected_shape)} for shard_dim={shard_dim}."
        )

    source_start = rank * shard_size
    source_end = source_start + shard_size

    send_parts = []
    input_splits = []
    for destination in range(world_size):
        destination_row = destination // process_columns
        destination_column = destination % process_columns
        destination_rows = block_cyclic_indices(
            rows, block_size, destination_row, process_rows
        )
        destination_columns = block_cyclic_indices(
            columns, block_size, destination_column, process_columns
        )

        if shard_dim == 0:
            selected_rows = tuple(
                row
                for row in destination_rows
                if source_start <= row < source_end
            )
            row_positions = [row - source_start for row in selected_rows]
            selected_columns = destination_columns
            column_positions = selected_columns
        else:
            selected_rows = destination_rows
            row_positions = selected_rows
            selected_columns = tuple(
                column
                for column in destination_columns
                if source_start <= column < source_end
            )
            column_positions = [
                column - source_start for column in selected_columns
            ]

        if selected_rows and selected_columns:
            row_index = torch.tensor(row_positions, device=matrix.device)
            column_index = torch.tensor(column_positions, device=matrix.device)
            part = (
                matrix.index_select(0, row_index)
                .index_select(1, column_index)
                .contiguous()
                .view(-1)
            )
        else:
            part = matrix.new_empty(0)
        send_parts.append(part)
        input_splits.append(part.numel())

    process_row = rank // process_columns
    process_column = rank % process_columns
    local_rows = block_cyclic_indices(
        rows, block_size, process_row, process_rows
    )
    local_columns = block_cyclic_indices(
        columns, block_size, process_column, process_columns
    )
    local_row_positions = {row: position for position, row in enumerate(local_rows)}
    local_column_positions = {
        column: position for position, column in enumerate(local_columns)
    }

    output_splits = []
    source_layouts = []
    for source in range(world_size):
        start = source * shard_size
        end = start + shard_size
        source_rows = (
            tuple(row for row in local_rows if start <= row < end)
            if shard_dim == 0
            else local_rows
        )
        source_columns = (
            tuple(column for column in local_columns if start <= column < end)
            if shard_dim == 1
            else local_columns
        )
        output_splits.append(len(source_rows) * len(source_columns))
        source_layouts.append((source_rows, source_columns))

    receive_buffer = matrix.new_empty(sum(output_splits))
    send_buffer = torch.cat(send_parts)
    dist.all_to_all_single(
        receive_buffer,
        send_buffer,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )

    leading_dimension = max(1, len(local_rows))
    storage_columns = max(1, len(local_columns))
    output = matrix.new_empty((storage_columns, leading_dimension)).T
    logical_output = output[: len(local_rows), : len(local_columns)]
    offset = 0
    for size, (source_rows, source_columns) in zip(
        output_splits, source_layouts
    ):
        if size:
            row_index = torch.tensor(
                [local_row_positions[row] for row in source_rows],
                device=matrix.device,
            )
            column_index = torch.tensor(
                [local_column_positions[column] for column in source_columns],
                device=matrix.device,
            )
            logical_output[row_index[:, None], column_index] = receive_buffer[
                offset : offset + size
            ].view(len(source_rows), len(source_columns))
        offset += size

    return output


@torch.no_grad()
def permute_2d_block_cyclic_(
    matrix: Tensor,
    global_shape: Tuple[int, int],
    row_order: Sequence[int] | Tensor,
    column_order: Sequence[int] | Tensor,
    block_size: int,
    process_grid_shape: Tuple[int, int],
    *,
    group: Optional[ProcessGroup] = None,
) -> Tensor:
    """Apply global row and column permutations to a packed matrix in place.

    The result follows ``matrix[row_order][:, column_order]`` semantics without
    gathering the global matrix. Every rank must provide identical orders and
    call this collective in the same sequence.
    """
    if group is None:
        group = dist.group.WORLD

    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2D, got {matrix.ndim}D.")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}.")

    rows, columns = global_shape
    if rows < 0 or columns < 0:
        raise ValueError(f"global_shape must be non-negative, got {global_shape}.")
    row_order = _validated_permutation(row_order, rows, "row_order")
    column_order = _validated_permutation(
        column_order, columns, "column_order"
    )

    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    process_rows, process_columns = process_grid_shape
    if process_rows <= 0 or process_columns <= 0:
        raise ValueError(
            "process_grid_shape must contain positive dimensions, got "
            f"{process_grid_shape}."
        )
    if process_rows * process_columns != world_size:
        raise ValueError("process grid must contain every process-group rank.")

    if world_size > 1:
        local_order = torch.tensor(
            row_order + column_order,
            device=matrix.device,
            dtype=torch.int64,
        )
        minimum_order = local_order.clone()
        maximum_order = local_order.clone()
        dist.all_reduce(minimum_order, op=dist.ReduceOp.MIN, group=group)
        dist.all_reduce(maximum_order, op=dist.ReduceOp.MAX, group=group)
        if not torch.equal(minimum_order, maximum_order):
            raise ValueError(
                "row_order and column_order must match on every rank."
            )

    process_row = rank // process_columns
    process_column = rank % process_columns
    local_rows = block_cyclic_indices(
        rows, block_size, process_row, process_rows
    )
    local_columns = block_cyclic_indices(
        columns, block_size, process_column, process_columns
    )
    if matrix.size(0) < len(local_rows) or matrix.size(1) < len(local_columns):
        raise ValueError(
            f"matrix shape {tuple(matrix.shape)} is smaller than its logical "
            f"block-cyclic shape {(len(local_rows), len(local_columns))}."
        )

    inverse_row_order = [0] * rows
    for new_row, old_row in enumerate(row_order):
        inverse_row_order[old_row] = new_row
    inverse_column_order = [0] * columns
    for new_column, old_column in enumerate(column_order):
        inverse_column_order[old_column] = new_column

    logical_matrix = matrix[: len(local_rows), : len(local_columns)]
    send_parts = []
    input_splits = []
    for destination in range(world_size):
        source_row_positions = []
        source_column_positions = []
        for local_row, old_row in enumerate(local_rows):
            new_row = inverse_row_order[old_row]
            owner_row = (new_row // block_size) % process_rows
            for local_column, old_column in enumerate(local_columns):
                new_column = inverse_column_order[old_column]
                owner_column = (new_column // block_size) % process_columns
                if owner_row * process_columns + owner_column == destination:
                    source_row_positions.append(local_row)
                    source_column_positions.append(local_column)

        if source_row_positions:
            row_index = torch.tensor(
                source_row_positions,
                device=matrix.device,
            )
            column_index = torch.tensor(
                source_column_positions,
                device=matrix.device,
            )
            part = logical_matrix[row_index, column_index].contiguous()
        else:
            part = matrix.new_empty(0)
        send_parts.append(part)
        input_splits.append(part.numel())

    local_row_positions = {row: index for index, row in enumerate(local_rows)}
    local_column_positions = {
        column: index for index, column in enumerate(local_columns)
    }
    output_splits = []
    destination_layouts = []
    for source in range(world_size):
        source_row = source // process_columns
        source_column = source % process_columns
        source_rows = block_cyclic_indices(
            rows, block_size, source_row, process_rows
        )
        source_columns = block_cyclic_indices(
            columns, block_size, source_column, process_columns
        )
        destination_rows = []
        destination_columns = []
        for old_row in source_rows:
            new_row = inverse_row_order[old_row]
            owner_row = (new_row // block_size) % process_rows
            for old_column in source_columns:
                new_column = inverse_column_order[old_column]
                owner_column = (new_column // block_size) % process_columns
                if owner_row == process_row and owner_column == process_column:
                    destination_rows.append(local_row_positions[new_row])
                    destination_columns.append(
                        local_column_positions[new_column]
                    )

        output_splits.append(len(destination_rows))
        destination_layouts.append(
            (destination_rows, destination_columns)
        )

    send_buffer = torch.cat(send_parts)
    receive_buffer = matrix.new_empty(sum(output_splits))
    dist.all_to_all_single(
        receive_buffer,
        send_buffer,
        output_split_sizes=output_splits,
        input_split_sizes=input_splits,
        group=group,
    )

    permuted = matrix.new_empty((len(local_rows), len(local_columns)))
    offset = 0
    for size, (destination_rows, destination_columns) in zip(
        output_splits, destination_layouts
    ):
        if size:
            row_index = torch.tensor(destination_rows, device=matrix.device)
            column_index = torch.tensor(
                destination_columns,
                device=matrix.device,
            )
            permuted[row_index, column_index] = receive_buffer[
                offset : offset + size
            ]
        offset += size

    logical_matrix.copy_(permuted)
    return matrix
