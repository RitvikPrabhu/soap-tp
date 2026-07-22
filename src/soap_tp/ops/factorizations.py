from __future__ import annotations

from typing import Any, Literal, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from ._utils import (
    block_cyclic_indices,
    validate_2d_block_cyclic_buffer,
    validate_process_grid,
)
from .optimizer import permute_2d_block_cyclic_


def _native_binding(
    binding: Optional[Any],
    module_name: str,
) -> Any:
    if binding is not None:
        return binding
    if module_name == "slate_bindings":
        from soap_tp import slate_bindings

        return slate_bindings
    from soap_tp import elpa_bindings

    return elpa_bindings


def _validate_native_backend(binding: Any, device: torch.device) -> None:
    backend = binding.compiled_gpu_backend()
    if backend not in {"none", "cuda", "rocm"}:
        raise RuntimeError(f"unsupported native backend {backend!r}.")
    expected_device = "cpu" if backend == "none" else "cuda"
    if device.type != expected_device:
        raise ValueError(
            f"native backend {backend!r} requires {expected_device} buffers, "
            f"got {device.type}."
        )
    if backend == "cuda" and (
        torch.version.cuda is None or torch.version.hip is not None
    ):
        raise RuntimeError("the CUDA binding requires a CUDA PyTorch build.")
    if backend == "rocm" and torch.version.hip is None:
        raise RuntimeError("the ROCm binding requires a ROCm PyTorch build.")


def _validate_world(process_grid_shape: Tuple[int, int]) -> tuple[int, int]:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first.")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    validate_process_grid(process_grid_shape, world_size)
    return rank, world_size


def _validate_float_buffer(
    name: str,
    matrix: Tensor,
    global_shape: Tuple[int, int],
    block_size: int,
    process_grid_shape: Tuple[int, int],
    rank: int,
) -> tuple[int, int]:
    if matrix.dtype != torch.float32:
        raise ValueError(f"{name} must use float32, got {matrix.dtype}.")
    return validate_2d_block_cyclic_buffer(
        name,
        matrix,
        global_shape,
        block_size,
        process_grid_shape,
        rank,
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validated_native_binding(
    binding: Optional[Any],
    module_name: str,
    device: torch.device,
    rank: int,
    world_size: int,
) -> Any:
    binding = _native_binding(binding, module_name)
    mpi_rank, mpi_world_size = binding.mpi_world_rank_and_size()
    if (mpi_rank, mpi_world_size) != (rank, world_size):
        raise RuntimeError(
            "Torch and MPI worlds must have identical ranks and sizes; "
            f"got Torch {(rank, world_size)} and MPI "
            f"{(mpi_rank, mpi_world_size)}."
        )
    _validate_native_backend(binding, device)
    return binding


@torch.no_grad()
def rotate_2d_block_cyclic_(
    matrix: Tensor,
    Q_left: Tensor,
    Q_right: Tensor,
    global_shape: Tuple[int, int],
    block_size: int,
    process_grid_shape: Tuple[int, int],
    *,
    direction: Literal["forward", "backward"],
    slate_binding: Optional[Any] = None,
) -> Tensor:
    """Rotate a packed matrix in place.

    ``forward`` computes ``Q_left.T @ matrix @ Q_right`` and ``backward``
    computes ``Q_left @ matrix @ Q_right.T``.
    """
    if direction not in {"forward", "backward"}:
        raise ValueError(
            f"direction must be 'forward' or 'backward', got {direction!r}."
        )
    rank, world_size = _validate_world(process_grid_shape)
    rows, columns = global_shape
    _validate_float_buffer(
        "matrix",
        matrix,
        global_shape,
        block_size,
        process_grid_shape,
        rank,
    )
    _validate_float_buffer(
        "Q_left",
        Q_left,
        (rows, rows),
        block_size,
        process_grid_shape,
        rank,
    )
    _validate_float_buffer(
        "Q_right",
        Q_right,
        (columns, columns),
        block_size,
        process_grid_shape,
        rank,
    )
    if len({matrix.device, Q_left.device, Q_right.device}) != 1:
        raise ValueError("matrix and bases must share a device.")
    if len({matrix.data_ptr(), Q_left.data_ptr(), Q_right.data_ptr()}) != 3:
        raise ValueError("matrix and basis buffers must not overlap.")

    binding = _validated_native_binding(
        slate_binding,
        "slate_bindings",
        matrix.device,
        rank,
        world_size,
    )
    function = (
        binding.slate_forward_rotation_float
        if direction == "forward"
        else binding.slate_backward_rotation_float
    )
    process_rows, process_columns = process_grid_shape
    _synchronize(matrix.device)
    function(
        Q_left.data_ptr(),
        matrix.data_ptr(),
        Q_right.data_ptr(),
        rows,
        columns,
        Q_left.stride(1),
        matrix.stride(1),
        Q_right.stride(1),
        block_size,
        process_rows,
        process_columns,
    )
    _synchronize(matrix.device)
    return matrix


@torch.no_grad()
def estimated_eigenvalue_order_2d_block_cyclic_(
    preconditioner: Tensor,
    Q: Tensor,
    work: Tensor,
    size: int,
    block_size: int,
    process_grid_shape: Tuple[int, int],
    *,
    slate_binding: Optional[Any] = None,
) -> Tensor:
    """Return the descending order of ``diag(Q.T @ P @ Q)``.

    ``work`` is overwritten with ``P @ Q``.
    """
    rank, world_size = _validate_world(process_grid_shape)
    buffers = (
        ("preconditioner", preconditioner),
        ("Q", Q),
        ("work", work),
    )
    logical_shape = None
    for name, matrix in buffers:
        shape = _validate_float_buffer(
            name,
            matrix,
            (size, size),
            block_size,
            process_grid_shape,
            rank,
        )
        logical_shape = shape if logical_shape is None else logical_shape
        if matrix.device != preconditioner.device:
            raise ValueError("preconditioner, Q, and work must share a device.")
    if len({matrix.data_ptr() for _, matrix in buffers}) != len(buffers):
        raise ValueError("preconditioner, Q, and work must not overlap.")
    if not (
        preconditioner.stride(1) == Q.stride(1) == work.stride(1)
    ):
        raise ValueError("preconditioner, Q, and work LDAs must match.")

    binding = _validated_native_binding(
        slate_binding,
        "slate_bindings",
        preconditioner.device,
        rank,
        world_size,
    )
    process_rows, process_columns = process_grid_shape
    _synchronize(preconditioner.device)
    binding.slate_symmetric_multiply_float(
        preconditioner.data_ptr(),
        Q.data_ptr(),
        work.data_ptr(),
        size,
        preconditioner.stride(1),
        block_size,
        process_rows,
        process_columns,
    )
    _synchronize(preconditioner.device)

    local_rows, local_columns = logical_shape
    process_column = rank % process_columns
    global_columns = block_cyclic_indices(
        size,
        block_size,
        process_column,
        process_columns,
    )
    local_estimates = (
        Q[:local_rows, :local_columns]
        * work[:local_rows, :local_columns]
    ).sum(dim=0)
    estimates = torch.zeros(
        size,
        dtype=torch.float32,
        device=preconditioner.device,
    )
    if global_columns:
        column_index = torch.tensor(
            global_columns,
            dtype=torch.long,
            device=preconditioner.device,
        )
        estimates.index_copy_(0, column_index, local_estimates)
    dist.all_reduce(estimates, op=dist.ReduceOp.SUM)
    return torch.argsort(estimates, descending=True)


@torch.no_grad()
def qr_2d_block_cyclic_(
    matrix: Tensor,
    Q: Tensor,
    size: int,
    block_size: int,
    process_grid_shape: Tuple[int, int],
    *,
    slate_binding: Optional[Any] = None,
) -> Tensor:
    """Overwrite ``Q`` with the distributed QR factor of ``matrix``."""
    rank, world_size = _validate_world(process_grid_shape)
    for name, buffer in (("matrix", matrix), ("Q", Q)):
        _validate_float_buffer(
            name,
            buffer,
            (size, size),
            block_size,
            process_grid_shape,
            rank,
        )
    if matrix.device != Q.device:
        raise ValueError("matrix and Q must share a device.")
    if matrix.data_ptr() == Q.data_ptr():
        raise ValueError("matrix and Q must not overlap.")

    binding = _validated_native_binding(
        slate_binding,
        "slate_bindings",
        matrix.device,
        rank,
        world_size,
    )
    process_rows, process_columns = process_grid_shape
    _synchronize(matrix.device)
    binding.slate_qr_float(
        matrix.data_ptr(),
        Q.data_ptr(),
        size,
        matrix.stride(1),
        Q.stride(1),
        block_size,
        process_rows,
        process_columns,
    )
    _synchronize(matrix.device)
    return Q


@torch.no_grad()
def power_iteration_qr_2d_block_cyclic_(
    preconditioner: Tensor,
    Q: Tensor,
    work: Tensor,
    size: int,
    block_size: int,
    process_grid_shape: Tuple[int, int],
    *,
    slate_binding: Optional[Any] = None,
) -> Tensor:
    """Sort one power iterate, QR it into ``Q``, and return the sort order."""
    order = estimated_eigenvalue_order_2d_block_cyclic_(
        preconditioner,
        Q,
        work,
        size,
        block_size,
        process_grid_shape,
        slate_binding=slate_binding,
    )
    permute_2d_block_cyclic_(
        work,
        (size, size),
        tuple(range(size)),
        order,
        block_size,
        process_grid_shape,
    )
    qr_2d_block_cyclic_(
        work,
        Q,
        size,
        block_size,
        process_grid_shape,
        slate_binding=slate_binding,
    )
    return order


@torch.no_grad()
def initialize_basis_2d_block_cyclic_(
    preconditioner: Tensor,
    Q: Tensor,
    work: Tensor,
    eigenvalues: Tensor,
    size: int,
    block_size: int,
    process_grid_shape: Tuple[int, int],
    *,
    elpa_binding: Optional[Any] = None,
) -> Tensor:
    """Initialize a descending eigenbasis with ELPA.

    The packed preconditioner must contain both triangles. ``work`` is
    overwritten because ELPA may destroy its input.
    """
    rank, world_size = _validate_world(process_grid_shape)
    local_rows = local_columns = 0
    for name, matrix in (
        ("preconditioner", preconditioner),
        ("Q", Q),
        ("work", work),
    ):
        rows, columns = _validate_float_buffer(
            name,
            matrix,
            (size, size),
            block_size,
            process_grid_shape,
            rank,
        )
        local_rows, local_columns = rows, columns
        if matrix.stride(1) != max(1, rows):
            raise ValueError(
                f"{name} must use the exact ELPA leading dimension "
                f"{max(1, rows)}, got {matrix.stride(1)}."
            )
        if matrix.device != preconditioner.device:
            raise ValueError("preconditioner, Q, and work must share a device.")
    empty_ownership = torch.tensor(
        int(local_rows == 0 or local_columns == 0),
        dtype=torch.int32,
        device=preconditioner.device,
    )
    dist.all_reduce(empty_ownership, op=dist.ReduceOp.MAX)
    if empty_ownership.item():
        raise ValueError(
            "ELPA initialization requires every rank to own at least one "
            "local row and column."
        )
    if len({preconditioner.data_ptr(), Q.data_ptr(), work.data_ptr()}) != 3:
        raise ValueError("preconditioner, Q, and work must not overlap.")
    if eigenvalues.shape != (size,) or eigenvalues.dtype != torch.float32:
        raise ValueError(
            f"eigenvalues must be float32 with shape {(size,)}, got "
            f"{eigenvalues.dtype} {tuple(eigenvalues.shape)}."
        )
    if not eigenvalues.is_contiguous():
        raise ValueError("eigenvalues must be contiguous.")
    if eigenvalues.device != preconditioner.device:
        raise ValueError("eigenvalues must share the matrix device.")

    binding = _validated_native_binding(
        elpa_binding,
        "elpa_bindings",
        preconditioner.device,
        rank,
        world_size,
    )
    # ELPA returns eigenpairs in ascending eigenvalue order. Solving for -A
    # makes that order correspond to descending eigenvalues of A, avoiding a
    # distributed permutation of the eigenvector columns afterward.
    work.copy_(preconditioner).neg_()
    process_rows, process_columns = process_grid_shape
    _synchronize(preconditioner.device)
    binding.elpa_eigenvectors_2d_block_cyclic_float(
        work.data_ptr(),
        eigenvalues.data_ptr(),
        Q.data_ptr(),
        size,
        local_rows,
        local_columns,
        block_size,
        process_rows,
        process_columns,
    )
    _synchronize(preconditioner.device)

    eigenvalues.neg_()
    return Q


@torch.no_grad()
def refresh_bases_and_transport_optimizer_state_(
    momentum: Tensor,
    variance: Tensor,
    left_preconditioner: Tensor,
    right_preconditioner: Tensor,
    Q_left: Tensor,
    Q_right: Tensor,
    left_work: Tensor,
    right_work: Tensor,
    global_shape: Tuple[int, int],
    block_size: int,
    process_grid_shape: Tuple[int, int],
    *,
    slate_binding: Optional[Any] = None,
) -> tuple[Tensor, Tensor]:
    """Refresh both bases while preserving SOAP optimizer-state semantics."""
    rotate_2d_block_cyclic_(
        momentum,
        Q_left,
        Q_right,
        global_shape,
        block_size,
        process_grid_shape,
        direction="backward",
        slate_binding=slate_binding,
    )
    rows, columns = global_shape
    left_order = power_iteration_qr_2d_block_cyclic_(
        left_preconditioner,
        Q_left,
        left_work,
        rows,
        block_size,
        process_grid_shape,
        slate_binding=slate_binding,
    )
    right_order = power_iteration_qr_2d_block_cyclic_(
        right_preconditioner,
        Q_right,
        right_work,
        columns,
        block_size,
        process_grid_shape,
        slate_binding=slate_binding,
    )
    permute_2d_block_cyclic_(
        variance,
        global_shape,
        left_order,
        right_order,
        block_size,
        process_grid_shape,
    )
    rotate_2d_block_cyclic_(
        momentum,
        Q_left,
        Q_right,
        global_shape,
        block_size,
        process_grid_shape,
        direction="forward",
        slate_binding=slate_binding,
    )
    return left_order, right_order
