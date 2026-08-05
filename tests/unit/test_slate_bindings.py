"""Tests for distributed SLATE operations and the fixed-basis SOAP pipeline."""

import importlib.util
import math
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import pytest

from mpi4py import MPI
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load

from soap_tp.ops._utils import (
    allocate_2d_block_cyclic,
    block_cyclic_tile_views,
)
from soap_tp.ops.factorizations import power_iteration_qr_2d_block_cyclic_


SEED = 42
BLOCK_SIZE = 2
RELATIVE_L2_RTOL = 1e-5
MATRIX_SIZES = (8, 9, 12, 13)


def _make_preconditioner(size, dtype=torch.float32, seed=SEED):
    generator = torch.Generator()
    generator.manual_seed(seed)

    values = torch.normal(
        mean=0,
        std=2,
        size=(size, size),
        generator=generator,
        dtype=dtype,
    )

    off_diagonal = torch.tril(values, diagonal=-1)
    preconditioner = off_diagonal + off_diagonal.T

    row_sums = preconditioner.abs().sum(dim=1)
    preconditioner.diagonal().copy_(row_sums + 1)

    return preconditioner


def _make_orthogonal(size, device, seed):
    generator = torch.Generator()
    generator.manual_seed(seed)

    matrix = torch.randn(
        size,
        size,
        dtype=torch.float32,
        generator=generator,
    ).to(device)
    Q, _ = torch.linalg.qr(matrix)
    return Q


# Convert the full PyTorch tensor into local 2D block cyclic storage
def _pack_full_matrix_2d_block_cyclic(
    full_matrix,
    local_matrix,
    block_size,
    process_grid_shape,
    rank,
):
    views = block_cyclic_tile_views(
        local_matrix,
        tuple(full_matrix.shape),
        block_size,
        process_grid_shape,
        rank,
        mode="full",
    )

    for (block_row, block_column), local_view in views.items():
        global_row_start = block_row * block_size
        global_row_end = min(global_row_start + block_size, full_matrix.size(0))
        global_column_start = block_column * block_size
        global_column_end = min(global_column_start + block_size, full_matrix.size(1))

        local_view.copy_(
            full_matrix[
                global_row_start:global_row_end, global_column_start:global_column_end
            ]
        )


# Convert the local 2D block cyclic storage into a full PyTorch tensor
def _unpack_matrix_2d_block_cyclic(
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
        global_row_start = block_row * block_size
        global_column_start = block_column * block_size
        full_matrix[
            global_row_start : global_row_start + local_view.size(0),
            global_column_start : global_column_start + local_view.size(1),
        ].copy_(local_view)

    dist.all_reduce(full_matrix)
    return full_matrix


def _setup_mpi(device):
    rank = MPI.COMM_WORLD.Get_rank()
    world_size = MPI.COMM_WORLD.Get_size()

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

    return rank, world_size


@torch.no_grad()
def _reference_power_iteration_qr(preconditioner, orthogonal):
    matrix = preconditioner.float()
    basis = orthogonal.float()

    estimated_eigenvalues = torch.diag(basis.T @ matrix @ basis)
    order = torch.argsort(estimated_eigenvalues, descending=True)

    Q, _ = torch.linalg.qr(matrix @ basis[:, order])

    return order, Q


def _run_soaptp_power_iteration_qr_2d_block_cyclic(
    preconditioner, orthogonal, rank, world_size
):
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first")
    if (rank, world_size) != (dist.get_rank(), dist.get_world_size()):
        raise ValueError("rank and world_size must match torch.distributed")
    if preconditioner.ndim != 2 or preconditioner.size(0) != preconditioner.size(1):
        raise ValueError("preconditioner must be a square matrix")
    if orthogonal.shape != preconditioner.shape:
        raise ValueError("orthogonal must have the same shape as preconditioner")
    if preconditioner.dtype != torch.float32 or orthogonal.dtype != torch.float32:
        raise ValueError("preconditioner and orthogonal must use float32")
    if preconditioner.device != orthogonal.device:
        raise ValueError("preconditioner and orthogonal must share a device")

    process_rows = math.isqrt(world_size)
    while world_size % process_rows:
        process_rows -= 1
    process_grid_shape = (process_rows, world_size // process_rows)
    global_shape = tuple(preconditioner.shape)
    size = preconditioner.size(0)

    local_preconditioner = allocate_2d_block_cyclic(
        global_shape,
        BLOCK_SIZE,
        process_grid_shape,
        dtype=preconditioner.dtype,
        device=preconditioner.device,
    )
    local_orthogonal = allocate_2d_block_cyclic(
        global_shape,
        BLOCK_SIZE,
        process_grid_shape,
        dtype=orthogonal.dtype,
        device=orthogonal.device,
    )
    work = allocate_2d_block_cyclic(
        global_shape,
        BLOCK_SIZE,
        process_grid_shape,
        dtype=preconditioner.dtype,
        device=preconditioner.device,
    )

    _pack_full_matrix_2d_block_cyclic(
        preconditioner,
        local_preconditioner,
        BLOCK_SIZE,
        process_grid_shape,
        rank,
    )
    _pack_full_matrix_2d_block_cyclic(
        orthogonal,
        local_orthogonal,
        BLOCK_SIZE,
        process_grid_shape,
        rank,
    )

    order = power_iteration_qr_2d_block_cyclic_(
        local_preconditioner,
        local_orthogonal,
        work,
        size,
        BLOCK_SIZE,
        process_grid_shape,
    )
    Q = _unpack_matrix_2d_block_cyclic(
        local_orthogonal,
        global_shape,
        BLOCK_SIZE,
        process_grid_shape,
        rank,
    )
    return order, Q


def _test_power_iteration_qr(preconditioner, orthogonal):
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        order_reference, Q_reference = _reference_power_iteration_qr(
            preconditioner,
            orthogonal,
        )
        Q_reference = Q_reference.contiguous()
    else:
        order_reference = torch.empty(
            preconditioner.size(0),
            dtype=torch.long,
            device=preconditioner.device,
        )
        Q_reference = torch.empty_like(preconditioner)

    # Every rank needs the oracle values for the same local assertions, but
    # only rank zero performs the reference power iteration and QR.
    dist.broadcast(order_reference, src=0)
    dist.broadcast(Q_reference, src=0)

    order_distributed, Q_distributed = _run_soaptp_power_iteration_qr_2d_block_cyclic(
        preconditioner,
        orthogonal,
        rank,
        world_size,
    )

    torch.testing.assert_close(order_distributed, order_reference)

    # A real QR factor is unique only up to an independent sign per column.
    signs = torch.sign(torch.sum(Q_reference * Q_distributed, dim=0))
    signs[signs == 0] = 1
    Q_distributed_aligned = Q_distributed * signs

    distributed_flat = Q_distributed_aligned.to(torch.float64).reshape(-1)
    reference_flat = Q_reference.to(torch.float64).reshape(-1)
    difference_norm = float(torch.linalg.vector_norm(distributed_flat - reference_flat))
    reference_norm = float(torch.linalg.vector_norm(reference_flat))
    tensors_are_finite = bool(
        torch.isfinite(distributed_flat).all() and torch.isfinite(reference_flat).all()
    )
    if not tensors_are_finite:
        relative_l2_error = math.inf
    elif reference_norm == 0:
        relative_l2_error = 0.0 if difference_norm == 0 else math.inf
    else:
        relative_l2_error = difference_norm / reference_norm
    if relative_l2_error > RELATIVE_L2_RTOL:
        raise AssertionError(
            "distributed Q does not match the reference Q: "
            f"relative L2 error {relative_l2_error:.6e} exceeds "
            f"rtol={RELATIVE_L2_RTOL:.6e} "
            f"(||distributed-reference||₂={difference_norm:.6e}, "
            f"||reference||₂={reference_norm:.6e})."
        )

    torch.testing.assert_close(
        Q_distributed_aligned,
        Q_reference,
        rtol=3e-3,
        atol=3e-3,
    )

    # Verify that the distributed QR output is orthogonal.
    identity = torch.eye(
        Q_distributed.size(0),
        dtype=Q_distributed.dtype,
        device=Q_distributed.device,
    )

    torch.testing.assert_close(
        Q_distributed.T @ Q_distributed,
        identity,
        rtol=1e-4,
        atol=1e-4,
    )
    return relative_l2_error


@pytest.fixture(scope="module")
def mpi_world():
    from soap_tp import slate_bindings

    backend = slate_bindings.compiled_gpu_backend()
    if backend == "none":
        device = torch.device("cpu")
    elif backend in {"cuda", "rocm"}:
        if not torch.cuda.is_available():
            raise RuntimeError(f"SLATE uses {backend}, but PyTorch cannot see a GPU")

        shared_world = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)
        try:
            local_rank = shared_world.Get_rank()
        finally:
            shared_world.Free()

        device_count = torch.cuda.device_count()
        device_index = 0 if device_count == 1 else local_rank
        if device_index >= device_count:
            raise RuntimeError(f"local MPI rank {local_rank} has no visible GPU")
        device = torch.device("cuda", device_index)
        torch.cuda.set_device(device)
    else:
        raise RuntimeError(f"unsupported SLATE backend {backend!r}")

    rank, world_size = _setup_mpi(device)
    try:
        yield rank, world_size, device
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.fixture(scope="module")
def slate_l2_log(mpi_world, output_folder):
    rank, _world_size, _device = mpi_world
    if output_folder is None:
        return None

    path = output_folder / "slate_l2.log"
    if rank == 0:
        path.write_text("matrix_shape relative_l2_error\n")
    return path


@pytest.mark.parametrize("size", MATRIX_SIZES)
def test_power_iteration_qr(size, mpi_world, slate_l2_log):
    rank, _world_size, device = mpi_world

    preconditioner = _make_preconditioner(
        size,
        dtype=torch.float32,
        seed=SEED + size,
    ).to(device)

    orthogonal = _make_orthogonal(
        size,
        device,
        seed=SEED + 1000 + size,
    )

    relative_l2_error = _test_power_iteration_qr(
        preconditioner,
        orthogonal,
    )
    if rank == 0 and slate_l2_log is not None:
        with slate_l2_log.open("a") as stream:
            stream.write(
                f"{size}x{size} {relative_l2_error:.6e}\n"
            )
