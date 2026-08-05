import math
import os
import socket

import pytest
from mpi4py import MPI
import torch
import torch.distributed as dist

from soap_tp.ops._utils import (
    allocate_2d_block_cyclic,
    block_cyclic_tile_views,
)
from soap_tp.ops.factorizations import (
    refresh_bases_and_transport_optimizer_state_,
    rotate_2d_block_cyclic_,
)


SEED = 42
BLOCK_SIZE = 2
RELATIVE_L2_RTOL = 1e-5
MATRIX_SHAPES = (
    (8, 8),
    (8, 12),
    (12, 8),
    (9, 13),
    (13, 9),
)


def _setup_distributed():
    from soap_tp import slate_bindings

    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()
    backend = slate_bindings.compiled_gpu_backend()

    if backend == "none":
        device = torch.device("cpu")
    elif backend in {"cuda", "rocm"}:
        if not torch.cuda.is_available():
            raise RuntimeError(f"SLATE uses {backend}, but PyTorch cannot see a GPU")

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
        raise RuntimeError(f"unsupported SLATE backend {backend!r}")

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


def _process_grid_shape(world_size):
    process_rows = math.isqrt(world_size)
    while world_size % process_rows:
        process_rows -= 1
    return process_rows, world_size // process_rows


def _make_matrix(rows, columns, device, seed=SEED):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.normal(
        mean=0,
        std=2,
        size=(rows, columns),
        generator=generator,
        dtype=torch.float32,
    ).to(device)


def _make_orthogonal(size, device, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    matrix = torch.randn(
        size,
        size,
        generator=generator,
        dtype=torch.float32,
    )
    Q, _ = torch.linalg.qr(matrix)
    return Q.to(device)


def _make_preconditioner(size, device, seed):
    values = _make_matrix(size, size, device, seed)
    off_diagonal = torch.tril(values, diagonal=-1)
    preconditioner = off_diagonal + off_diagonal.T
    preconditioner.diagonal().copy_(preconditioner.abs().sum(dim=1) + 1)
    return preconditioner


def _pack_full_matrix_2d_block_cyclic(
    full_matrix,
    local_matrix,
    block_size,
    process_grid_shape,
    rank,
):
    for (block_row, block_column), local_view in block_cyclic_tile_views(
        local_matrix,
        tuple(full_matrix.shape),
        block_size,
        process_grid_shape,
        rank,
        mode="full",
    ).items():
        row_start = block_row * block_size
        column_start = block_column * block_size
        local_view.copy_(
            full_matrix[
                row_start : row_start + local_view.size(0),
                column_start : column_start + local_view.size(1),
            ]
        )


def _gather_matrix_2d_block_cyclic(
    local_matrix,
    global_shape,
    block_size,
    process_grid_shape,
    rank,
):
    full_matrix = local_matrix.new_zeros(global_shape)
    for (block_row, block_column), local_view in block_cyclic_tile_views(
        local_matrix,
        global_shape,
        block_size,
        process_grid_shape,
        rank,
        mode="full",
    ).items():
        row_start = block_row * block_size
        column_start = block_column * block_size
        full_matrix[
            row_start : row_start + local_view.size(0),
            column_start : column_start + local_view.size(1),
        ].copy_(local_view)

    dist.reduce(full_matrix, dst=0, op=dist.ReduceOp.SUM)
    return full_matrix if rank == 0 else None


def _run_rotate_2d_block_cyclic(
    matrix,
    Q_left,
    Q_right,
    *,
    direction,
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    process_grid_shape = _process_grid_shape(world_size)
    global_shape = tuple(matrix.shape)

    local_matrix = allocate_2d_block_cyclic(
        global_shape,
        BLOCK_SIZE,
        process_grid_shape,
        dtype=torch.float32,
        device=matrix.device,
    )
    local_Q_left = allocate_2d_block_cyclic(
        (matrix.size(0), matrix.size(0)),
        BLOCK_SIZE,
        process_grid_shape,
        dtype=torch.float32,
        device=matrix.device,
    )
    local_Q_right = allocate_2d_block_cyclic(
        (matrix.size(1), matrix.size(1)),
        BLOCK_SIZE,
        process_grid_shape,
        dtype=torch.float32,
        device=matrix.device,
    )

    for full, local in (
        (matrix, local_matrix),
        (Q_left, local_Q_left),
        (Q_right, local_Q_right),
    ):
        _pack_full_matrix_2d_block_cyclic(
            full,
            local,
            BLOCK_SIZE,
            process_grid_shape,
            rank,
        )

    rotate_2d_block_cyclic_(
        local_matrix,
        local_Q_left,
        local_Q_right,
        global_shape,
        BLOCK_SIZE,
        process_grid_shape,
        direction=direction,
    )
    return _gather_matrix_2d_block_cyclic(
        local_matrix,
        global_shape,
        BLOCK_SIZE,
        process_grid_shape,
        rank,
    )


def _run_refresh_bases_and_transport_optimizer_state(
    momentum,
    variance,
    left_preconditioner,
    right_preconditioner,
    Q_left,
    Q_right,
):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    process_grid_shape = _process_grid_shape(world_size)
    global_shape = tuple(momentum.shape)
    rows, columns = global_shape

    def allocate(shape):
        return allocate_2d_block_cyclic(
            shape,
            BLOCK_SIZE,
            process_grid_shape,
            dtype=torch.float32,
            device=momentum.device,
        )

    local_momentum = allocate(global_shape)
    local_variance = allocate(global_shape)
    local_left_preconditioner = allocate((rows, rows))
    local_right_preconditioner = allocate((columns, columns))
    local_Q_left = allocate((rows, rows))
    local_Q_right = allocate((columns, columns))
    left_work = allocate((rows, rows))
    right_work = allocate((columns, columns))

    for full, local in (
        (momentum, local_momentum),
        (variance, local_variance),
        (left_preconditioner, local_left_preconditioner),
        (right_preconditioner, local_right_preconditioner),
        (Q_left, local_Q_left),
        (Q_right, local_Q_right),
    ):
        _pack_full_matrix_2d_block_cyclic(
            full,
            local,
            BLOCK_SIZE,
            process_grid_shape,
            rank,
        )

    left_order, right_order = refresh_bases_and_transport_optimizer_state_(
        local_momentum,
        local_variance,
        local_left_preconditioner,
        local_right_preconditioner,
        local_Q_left,
        local_Q_right,
        left_work,
        right_work,
        global_shape,
        BLOCK_SIZE,
        process_grid_shape,
    )

    results = {}
    for name, local, shape in (
        ("momentum", local_momentum, global_shape),
        ("variance", local_variance, global_shape),
        ("Q_left", local_Q_left, (rows, rows)),
        ("Q_right", local_Q_right, (columns, columns)),
    ):
        results[name] = _gather_matrix_2d_block_cyclic(
            local,
            shape,
            BLOCK_SIZE,
            process_grid_shape,
            rank,
        )
    return left_order, right_order, results


def _compare_matrix(actual, reference):
    actual_flat = actual.to(torch.float64).reshape(-1)
    reference_flat = reference.to(torch.float64).reshape(-1)
    difference_norm = float(torch.linalg.vector_norm(actual_flat - reference_flat))
    reference_norm = float(torch.linalg.vector_norm(reference_flat))
    tensors_are_finite = bool(
        torch.isfinite(actual_flat).all() and torch.isfinite(reference_flat).all()
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
        rtol=2e-4,
        atol=2e-4,
    )
    return relative_l2_error


def _test_forward_rotation(matrix, Q_left, Q_right):
    actual = _run_rotate_2d_block_cyclic(
        matrix,
        Q_left,
        Q_right,
        direction="forward",
    )
    if dist.get_rank() != 0:
        return None
    reference = Q_left.T @ matrix @ Q_right
    return _compare_matrix(actual, reference)


def _test_backward_rotation(matrix, Q_left, Q_right):
    actual = _run_rotate_2d_block_cyclic(
        matrix,
        Q_left,
        Q_right,
        direction="backward",
    )
    if dist.get_rank() != 0:
        return None
    reference = Q_left @ matrix @ Q_right.T
    return _compare_matrix(actual, reference)


@pytest.fixture(scope="module")
def distributed_world():
    mpi_world_size = MPI.COMM_WORLD.Get_size()
    if mpi_world_size < 2:
        pytest.skip("test requires multiple MPI ranks")
    rank, world_size, device = _setup_distributed()
    try:
        yield rank, world_size, device
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.fixture(scope="module")
def rotation_l2_logs(distributed_world, output_folder):
    rank, _world_size, _device = distributed_world
    if output_folder is None:
        return None

    paths = {
        "forward": output_folder / "forward_rotation_l2.log",
        "backward": output_folder / "backward_rotation_l2.log",
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
        stream.write(f"{rows}x{columns} {relative_l2_error:.6e}\n")


def _rotation_problem(shape, device):
    rows, columns = shape
    matrix = _make_matrix(
        rows,
        columns,
        device,
        seed=SEED + rows * 100 + columns,
    )
    Q_left = _make_orthogonal(
        rows,
        device,
        seed=SEED + 1000 + rows,
    )
    Q_right = _make_orthogonal(
        columns,
        device,
        seed=SEED + 2000 + columns,
    )
    return matrix, Q_left, Q_right


@pytest.mark.parametrize(
    "shape",
    MATRIX_SHAPES,
    ids=lambda shape: f"{shape[0]}x{shape[1]}",
)
def test_rotate_2d_block_cyclic_forward(
    shape,
    distributed_world,
    rotation_l2_logs,
):
    rank, _world_size, device = distributed_world
    matrix, Q_left, Q_right = _rotation_problem(shape, device)
    error = _test_forward_rotation(matrix, Q_left, Q_right)
    path = None if rotation_l2_logs is None else rotation_l2_logs["forward"]
    if rank == 0:
        _write_l2_log(path, shape, error)


@pytest.mark.parametrize(
    "shape",
    MATRIX_SHAPES,
    ids=lambda shape: f"{shape[0]}x{shape[1]}",
)
def test_rotate_2d_block_cyclic_backward(
    shape,
    distributed_world,
    rotation_l2_logs,
):
    rank, _world_size, device = distributed_world
    matrix, Q_left, Q_right = _rotation_problem(shape, device)
    error = _test_backward_rotation(matrix, Q_left, Q_right)
    path = None if rotation_l2_logs is None else rotation_l2_logs["backward"]
    if rank == 0:
        _write_l2_log(path, shape, error)


@pytest.mark.parametrize(
    "shape",
    MATRIX_SHAPES,
    ids=lambda shape: f"{shape[0]}x{shape[1]}",
)
def test_refresh_bases_and_transport_optimizer_state(shape, distributed_world):
    rank, _world_size, device = distributed_world
    rows, columns = shape
    momentum = _make_matrix(
        rows,
        columns,
        device,
        seed=SEED + 3000,
    )
    variance = _make_matrix(
        rows,
        columns,
        device,
        seed=SEED + 4000,
    ).square()
    left_preconditioner = _make_preconditioner(
        rows,
        device,
        seed=SEED + 5000,
    )
    right_preconditioner = _make_preconditioner(
        columns,
        device,
        seed=SEED + 6000,
    )
    Q_left = _make_orthogonal(
        rows,
        device,
        seed=SEED + 7000,
    )
    Q_right = _make_orthogonal(
        columns,
        device,
        seed=SEED + 8000,
    )

    actual_left_order, actual_right_order, actual = (
        _run_refresh_bases_and_transport_optimizer_state(
            momentum,
            variance,
            left_preconditioner,
            right_preconditioner,
            Q_left,
            Q_right,
        )
    )
    if rank != 0:
        return

    parameter_momentum = Q_left @ momentum @ Q_right.T
    left_estimates = torch.diag(Q_left.T @ left_preconditioner @ Q_left)
    right_estimates = torch.diag(Q_right.T @ right_preconditioner @ Q_right)
    expected_left_order = torch.argsort(
        left_estimates,
        descending=True,
    )
    expected_right_order = torch.argsort(
        right_estimates,
        descending=True,
    )
    expected_Q_left, _ = torch.linalg.qr(
        left_preconditioner @ Q_left[:, expected_left_order]
    )
    expected_Q_right, _ = torch.linalg.qr(
        right_preconditioner @ Q_right[:, expected_right_order]
    )
    expected_momentum = expected_Q_left.T @ parameter_momentum @ expected_Q_right
    expected_variance = variance.index_select(
        0,
        expected_left_order,
    ).index_select(1, expected_right_order)

    torch.testing.assert_close(
        actual_left_order,
        expected_left_order,
    )
    torch.testing.assert_close(
        actual_right_order,
        expected_right_order,
    )

    left_signs = torch.sign(torch.sum(expected_Q_left * actual["Q_left"], dim=0))
    right_signs = torch.sign(torch.sum(expected_Q_right * actual["Q_right"], dim=0))
    left_signs[left_signs == 0] = 1
    right_signs[right_signs == 0] = 1

    _compare_matrix(
        actual["Q_left"] * left_signs,
        expected_Q_left,
    )
    _compare_matrix(
        actual["Q_right"] * right_signs,
        expected_Q_right,
    )
    _compare_matrix(
        actual["momentum"] * left_signs[:, None] * right_signs[None, :],
        expected_momentum,
    )
    _compare_matrix(actual["variance"], expected_variance)

    torch.testing.assert_close(
        actual["Q_left"].T @ actual["Q_left"],
        torch.eye(rows, device=device),
        rtol=1e-4,
        atol=1e-4,
    )

    torch.testing.assert_close(
        actual["Q_right"].T @ actual["Q_right"],
        torch.eye(columns, device=device),
        rtol=1e-4,
        atol=1e-4,
    )
