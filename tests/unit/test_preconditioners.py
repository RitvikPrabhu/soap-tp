import math
import os
import socket
import pytest

from mpi4py import MPI
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from soap_tp.ops._utils import (
    allocate_2d_block_cyclic,
    block_cyclic_indices,
)
from soap_tp.ops.preconditioners import (
    update_left_preconditioner_2d_block_cyclic_,
    update_right_preconditioner_2d_block_cyclic_,
)

SEED = 42
BLOCK_SIZE = 2
RELATIVE_L2_RTOL = 1e-5
GRADIENT_SHAPES = (
    (8, 8),
    (8, 12),
    (12, 8),
    (9, 13),
    (13, 9),
)


def _setup_distributed():
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()

    if torch.cuda.is_available():
        shared_world = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
        try:
            local_rank = shared_world.Get_rank()
            local_size = shared_world.Get_size()
            visible_device = next(
                (
                    (name, os.environ[name])
                    for name in (
                        "CUDA_VISIBLE_DEVICES",
                        "ROCR_VISIBLE_DEVICES",
                        "HIP_VISIBLE_DEVICES",
                    )
                    if os.environ.get(name)
                ),
                None,
            )
            visible_devices = shared_world.allgather(visible_device)
        finally:
            shared_world.Free()

        device_count = torch.cuda.device_count()
        if device_count == 1:
            if local_size > 1 and (
                None in visible_devices or len(set(visible_devices)) != local_size
            ):
                raise RuntimeError(
                    "one GPU is visible to every local MPI rank, but the "
                    "launcher did not assign a distinct GPU to each rank"
                )
            device_index = 0
        elif local_size <= device_count:
            device_index = local_rank
        else:
            raise RuntimeError(
                f"{local_size} local MPI ranks have only {device_count} visible GPUs"
            )
        device = torch.device("cuda", device_index)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    if rank == 0:
        address = os.environ.get("MASTER_ADDR", "127.0.0.1")
        if "MASTER_PORT" in os.environ:
            port = int(os.environ["MASTER_PORT"])
        else:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind((address, 0))
                port = sock.getsockname()[1]
        rendezvous = address, port
    else:
        rendezvous = None

    address, port = MPI.COMM_WORLD.bcast(rendezvous, root=0)
    os.environ["MASTER_ADDR"] = address
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "nccl" if device.type == "cuda" else "gloo",
        rank=rank,
        world_size=world_size,
    )
    return rank, world_size, device


def _make_gradient(m, n, dtype=torch.float32, seed=SEED):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.normal(mean=0, std=2, size=(m, n), generator=generator, dtype=dtype)


def _reference_left_preconditioner(gradient):
    """Compute the left preconditioner using the reference implementation."""
    return gradient @ gradient.T


def _reference_right_preconditioner(gradient):
    """Compute the right preconditioner using the reference implementation."""
    return gradient.T @ gradient


def _run_row_sharded_left_precondtioner_2d_block_cyclic(
    gradient,
    *,
    beta=0.0,
    initial_value=0.0,
):
    """Run the left preconditioner update in a row-sharded manner."""
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first")
    if gradient.ndim != 2:
        raise ValueError(f"gradient must be 2D, got {gradient.ndim}D")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    process_rows = math.isqrt(world_size)
    while world_size % process_rows:
        process_rows -= 1
    process_grid_shape = (process_rows, world_size // process_rows)

    rows = gradient.size(0)
    local_gradient = torch.tensor_split(
        gradient,
        world_size,
        dim=0,
    )[rank].contiguous()
    local_preconditioner = allocate_2d_block_cyclic(
        (rows, rows),
        BLOCK_SIZE,
        process_grid_shape,
        dtype=torch.float32,
        device=gradient.device,
    )
    local_preconditioner.fill_(initial_value)

    update_left_preconditioner_2d_block_cyclic_(
        local_gradient,
        local_preconditioner,
        beta=beta,
        block_size=BLOCK_SIZE,
        process_grid_shape=process_grid_shape,
        shard_dim=0,
    )

    process_columns = process_grid_shape[1]
    process_row = rank // process_columns
    process_column = rank % process_columns
    global_rows = block_cyclic_indices(
        rows,
        BLOCK_SIZE,
        process_row,
        process_rows,
    )
    global_columns = block_cyclic_indices(
        rows,
        BLOCK_SIZE,
        process_column,
        process_columns,
    )

    full_preconditioner = local_preconditioner.new_zeros((rows, rows))
    if global_rows and global_columns:
        row_index = torch.tensor(global_rows, device=gradient.device)
        column_index = torch.tensor(global_columns, device=gradient.device)
        full_preconditioner[row_index[:, None], column_index] = local_preconditioner[
            : len(global_rows),
            : len(global_columns),
        ]

    dist.reduce(full_preconditioner, dst=0, op=dist.ReduceOp.SUM)
    return full_preconditioner if rank == 0 else None


def _run_col_sharded_left_precondtioner_2d_block_cyclic(gradient):
    """Run the left preconditioner update in a column-sharded manner."""
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first")
    if gradient.ndim != 2:
        raise ValueError(f"gradient must be 2D, got {gradient.ndim}D")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    process_rows = math.isqrt(world_size)
    while world_size % process_rows:
        process_rows -= 1
    process_grid_shape = (process_rows, world_size // process_rows)

    rows = gradient.size(0)
    local_gradient = torch.tensor_split(
        gradient,
        world_size,
        dim=1,
    )[rank].contiguous()
    local_preconditioner = allocate_2d_block_cyclic(
        (rows, rows),
        BLOCK_SIZE,
        process_grid_shape,
        dtype=torch.float32,
        device=gradient.device,
    )

    update_left_preconditioner_2d_block_cyclic_(
        local_gradient,
        local_preconditioner,
        beta=0.0,
        block_size=BLOCK_SIZE,
        process_grid_shape=process_grid_shape,
        shard_dim=1,
    )

    process_columns = process_grid_shape[1]
    process_row = rank // process_columns
    process_column = rank % process_columns
    global_rows = block_cyclic_indices(
        rows,
        BLOCK_SIZE,
        process_row,
        process_rows,
    )
    global_columns = block_cyclic_indices(
        rows,
        BLOCK_SIZE,
        process_column,
        process_columns,
    )

    full_preconditioner = local_preconditioner.new_zeros((rows, rows))
    if global_rows and global_columns:
        row_index = torch.tensor(global_rows, device=gradient.device)
        column_index = torch.tensor(global_columns, device=gradient.device)
        full_preconditioner[row_index[:, None], column_index] = local_preconditioner[
            : len(global_rows),
            : len(global_columns),
        ]

    dist.reduce(full_preconditioner, dst=0, op=dist.ReduceOp.SUM)
    return full_preconditioner if rank == 0 else None


def _run_row_sharded_right_precondtioner_2d_block_cyclic(gradient):
    """Run the right preconditioner update in a row-sharded manner."""
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first")
    if gradient.ndim != 2:
        raise ValueError(f"gradient must be 2D, got {gradient.ndim}D")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    process_rows = math.isqrt(world_size)
    while world_size % process_rows:
        process_rows -= 1
    process_grid_shape = (process_rows, world_size // process_rows)

    columns = gradient.size(1)
    local_gradient = torch.tensor_split(
        gradient,
        world_size,
        dim=0,
    )[rank].contiguous()
    local_preconditioner = allocate_2d_block_cyclic(
        (columns, columns),
        BLOCK_SIZE,
        process_grid_shape,
        dtype=torch.float32,
        device=gradient.device,
    )

    update_right_preconditioner_2d_block_cyclic_(
        local_gradient,
        local_preconditioner,
        beta=0.0,
        block_size=BLOCK_SIZE,
        process_grid_shape=process_grid_shape,
        shard_dim=0,
    )

    process_columns = process_grid_shape[1]
    process_row = rank // process_columns
    process_column = rank % process_columns
    global_rows = block_cyclic_indices(
        columns,
        BLOCK_SIZE,
        process_row,
        process_rows,
    )
    global_columns = block_cyclic_indices(
        columns,
        BLOCK_SIZE,
        process_column,
        process_columns,
    )

    full_preconditioner = local_preconditioner.new_zeros((columns, columns))
    if global_rows and global_columns:
        row_index = torch.tensor(global_rows, device=gradient.device)
        column_index = torch.tensor(global_columns, device=gradient.device)
        full_preconditioner[row_index[:, None], column_index] = local_preconditioner[
            : len(global_rows),
            : len(global_columns),
        ]

    dist.reduce(full_preconditioner, dst=0, op=dist.ReduceOp.SUM)
    return full_preconditioner if rank == 0 else None


def _run_col_sharded_right_precondtioner_2d_block_cyclic(gradient):
    """Run the right preconditioner update in a column-sharded manner."""
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first")
    if gradient.ndim != 2:
        raise ValueError(f"gradient must be 2D, got {gradient.ndim}D")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    process_rows = math.isqrt(world_size)
    while world_size % process_rows:
        process_rows -= 1
    process_grid_shape = (process_rows, world_size // process_rows)

    columns = gradient.size(1)
    local_gradient = torch.tensor_split(
        gradient,
        world_size,
        dim=1,
    )[rank].contiguous()
    local_preconditioner = allocate_2d_block_cyclic(
        (columns, columns),
        BLOCK_SIZE,
        process_grid_shape,
        dtype=torch.float32,
        device=gradient.device,
    )

    update_right_preconditioner_2d_block_cyclic_(
        local_gradient,
        local_preconditioner,
        beta=0.0,
        block_size=BLOCK_SIZE,
        process_grid_shape=process_grid_shape,
        shard_dim=1,
    )

    process_columns = process_grid_shape[1]
    process_row = rank // process_columns
    process_column = rank % process_columns
    global_rows = block_cyclic_indices(
        columns,
        BLOCK_SIZE,
        process_row,
        process_rows,
    )
    global_columns = block_cyclic_indices(
        columns,
        BLOCK_SIZE,
        process_column,
        process_columns,
    )

    full_preconditioner = local_preconditioner.new_zeros((columns, columns))
    if global_rows and global_columns:
        row_index = torch.tensor(global_rows, device=gradient.device)
        column_index = torch.tensor(global_columns, device=gradient.device)
        full_preconditioner[row_index[:, None], column_index] = local_preconditioner[
            : len(global_rows),
            : len(global_columns),
        ]

    dist.reduce(full_preconditioner, dst=0, op=dist.ReduceOp.SUM)
    return full_preconditioner if rank == 0 else None


def _compare_preconditioner(actual, reference):
    actual_flat = actual.to(torch.float64).reshape(-1)
    reference_flat = reference.to(torch.float64).reshape(-1)
    difference_norm = float(
        torch.linalg.vector_norm(actual_flat - reference_flat)
    )
    reference_norm = float(torch.linalg.vector_norm(reference_flat))
    tensors_are_finite = bool(
        torch.isfinite(actual_flat).all()
        and torch.isfinite(reference_flat).all()
    )
    if not tensors_are_finite:
        relative_l2_error = math.inf
    elif reference_norm == 0:
        relative_l2_error = 0.0 if difference_norm == 0 else math.inf
    else:
        relative_l2_error = difference_norm / reference_norm

    if relative_l2_error > RELATIVE_L2_RTOL:
        raise AssertionError(
            f"relative L2 error {relative_l2_error:.6e} exceeds "
            f"rtol={RELATIVE_L2_RTOL:.6e} "
            f"(||distributed-reference||₂={difference_norm:.6e}, "
            f"||reference||₂={reference_norm:.6e})"
        )
    torch.testing.assert_close(
        actual,
        reference,
        rtol=RELATIVE_L2_RTOL,
        atol=RELATIVE_L2_RTOL,
    )
    return relative_l2_error


def _test_row_sharded_left_preconditioner_2d_block_cyclic(gradient):
    actual = _run_row_sharded_left_precondtioner_2d_block_cyclic(
        gradient
    )
    if dist.get_rank() != 0:
        return None
    reference = _reference_left_preconditioner(gradient.float())
    return _compare_preconditioner(actual, reference)


def _test_row_sharded_right_preconditioner_2d_block_cyclic(gradient):
    actual = _run_row_sharded_right_precondtioner_2d_block_cyclic(
        gradient
    )
    if dist.get_rank() != 0:
        return None
    reference = _reference_right_preconditioner(gradient.float())
    return _compare_preconditioner(actual, reference)


def _test_col_sharded_left_preconditioner_2d_block_cyclic(gradient):
    actual = _run_col_sharded_left_precondtioner_2d_block_cyclic(
        gradient
    )
    if dist.get_rank() != 0:
        return None
    reference = _reference_left_preconditioner(gradient.float())
    return _compare_preconditioner(actual, reference)


def _test_col_sharded_right_preconditioner_2d_block_cyclic(gradient):
    actual = _run_col_sharded_right_precondtioner_2d_block_cyclic(
        gradient
    )
    if dist.get_rank() != 0:
        return None
    reference = _reference_right_preconditioner(gradient.float())
    return _compare_preconditioner(actual, reference)


@pytest.fixture(scope="module")
def distributed_world():
    rank, world_size, device = _setup_distributed()
    try:
        yield rank, world_size, device
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.fixture(scope="module")
def preconditioner_l2_logs(distributed_world, output_folder):
    rank, _world_size, _device = distributed_world
    if output_folder is None:
        return None

    paths = {
        "left_row": output_folder / "left_row_l2.log",
        "right_row": output_folder / "right_row_l2.log",
        "left_col": output_folder / "left_col_l2.log",
        "right_col": output_folder / "right_col_l2.log",
    }
    if rank == 0:
        for path in paths.values():
            path.write_text("matrix_shape relative_l2_error\n")
    return paths


def _write_l2_log(path, shape, relative_l2_error):
    if path is None or dist.get_rank() != 0:
        return
    rows, columns = shape
    with path.open("a") as stream:
        stream.write(
            f"{rows}x{columns} {relative_l2_error:.6e}\n"
        )


@pytest.mark.parametrize(
    "shape",
    GRADIENT_SHAPES,
    ids=lambda shape: f"{shape[0]}x{shape[1]}",
)
def test_row_sharded_left_preconditioner_2d_block_cyclic(
    shape,
    distributed_world,
    preconditioner_l2_logs,
):
    _rank, _world_size, device = distributed_world
    gradient = _make_gradient(*shape).to(device)
    error = _test_row_sharded_left_preconditioner_2d_block_cyclic(
        gradient
    )
    path = (
        None
        if preconditioner_l2_logs is None
        else preconditioner_l2_logs["left_row"]
    )
    _write_l2_log(path, shape, error)


@pytest.mark.parametrize(
    "shape",
    GRADIENT_SHAPES,
    ids=lambda shape: f"{shape[0]}x{shape[1]}",
)
def test_row_sharded_right_preconditioner_2d_block_cyclic(
    shape,
    distributed_world,
    preconditioner_l2_logs,
):
    _rank, _world_size, device = distributed_world
    gradient = _make_gradient(*shape).to(device)
    error = _test_row_sharded_right_preconditioner_2d_block_cyclic(
        gradient
    )
    path = (
        None
        if preconditioner_l2_logs is None
        else preconditioner_l2_logs["right_row"]
    )
    _write_l2_log(path, shape, error)


@pytest.mark.parametrize(
    "shape",
    GRADIENT_SHAPES,
    ids=lambda shape: f"{shape[0]}x{shape[1]}",
)
def test_col_sharded_left_preconditioner_2d_block_cyclic(
    shape,
    distributed_world,
    preconditioner_l2_logs,
):
    _rank, _world_size, device = distributed_world
    gradient = _make_gradient(*shape).to(device)
    error = _test_col_sharded_left_preconditioner_2d_block_cyclic(
        gradient
    )
    path = (
        None
        if preconditioner_l2_logs is None
        else preconditioner_l2_logs["left_col"]
    )
    _write_l2_log(path, shape, error)


@pytest.mark.parametrize(
    "shape",
    GRADIENT_SHAPES,
    ids=lambda shape: f"{shape[0]}x{shape[1]}",
)
def test_col_sharded_right_preconditioner_2d_block_cyclic(
    shape,
    distributed_world,
    preconditioner_l2_logs,
):
    _rank, _world_size, device = distributed_world
    gradient = _make_gradient(*shape).to(device)
    error = _test_col_sharded_right_preconditioner_2d_block_cyclic(
        gradient
    )
    path = (
        None
        if preconditioner_l2_logs is None
        else preconditioner_l2_logs["right_col"]
    )
    _write_l2_log(path, shape, error)


def test_row_sharded_left_preconditioner_nonzero_beta(
    distributed_world,
):
    rank, _world_size, device = distributed_world
    beta = 0.25
    initial_value = 2.0
    gradient = _make_gradient(9, 13, seed=SEED + 100).to(device)

    actual = _run_row_sharded_left_precondtioner_2d_block_cyclic(
        gradient,
        beta=beta,
        initial_value=initial_value,
    )
    if rank == 0:
        current = _reference_left_preconditioner(gradient.float())
        previous = torch.full_like(current, initial_value)
        reference = previous.lerp(current, 1.0 - beta)
        _compare_preconditioner(actual, reference)
