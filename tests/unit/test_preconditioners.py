import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from soap_tp.ops._utils import (
    allocate_2d_block_cyclic,
    block_cyclic_indices,
)
from soap_tp.ops.preconditioners import (
    update_left_preconditioner_2d_block_cyclic_,
    update_left_preconditioner_from_col_shards,
    update_right_preconditioner_2d_block_cyclic_,
    update_right_preconditioner_from_col_shards,
)


WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "4"))


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _device_for_rank(rank, world_size):
    if torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def _init(rank, world_size, port, device):
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl" if device.type == "cuda" else "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )


def _grid(world_size):
    process_rows = int(world_size**0.5)
    while world_size % process_rows:
        process_rows -= 1
    return process_rows, world_size // process_rows


def _gradient(rows, columns, device):
    values = torch.arange(
        rows * columns,
        dtype=torch.float64,
        device=device,
    )
    return ((values.reshape(rows, columns) % 13) - 6.0) / 3.0


def _gather_packed(
    local,
    global_shape,
    block_size,
    process_grid_shape,
    rank,
):
    rows, columns = global_shape
    process_rows, process_columns = process_grid_shape
    process_row = rank // process_columns
    process_column = rank % process_columns
    global_rows = block_cyclic_indices(
        rows,
        block_size,
        process_row,
        process_rows,
    )
    global_columns = block_cyclic_indices(
        columns,
        block_size,
        process_column,
        process_columns,
    )

    result = torch.zeros(
        global_shape,
        dtype=local.dtype,
        device=local.device,
    )
    owners = torch.zeros_like(result)
    if global_rows and global_columns:
        row_index = torch.tensor(global_rows, device=local.device)
        column_index = torch.tensor(global_columns, device=local.device)
        result[row_index[:, None], column_index] = local[
            : len(global_rows), : len(global_columns)
        ]
        owners[row_index[:, None], column_index] = 1
    dist.all_reduce(result)
    dist.all_reduce(owners)
    torch.testing.assert_close(owners, torch.ones_like(owners))
    return result


def _run_packed_update_test(rank, world_size, port):
    device = _device_for_rank(rank, world_size)
    _init(rank, world_size, port, device)
    try:
        rows = world_size + 1
        columns = 2 * world_size + 1
        block_size = 3
        process_grid_shape = _grid(world_size)
        gradient = _gradient(rows, columns, device)
        beta = 0.25

        for shard_dim in (0, 1):
            local_gradient = torch.tensor_split(
                gradient,
                world_size,
                dim=shard_dim,
            )[rank]
            left = allocate_2d_block_cyclic(
                (rows, rows),
                block_size,
                process_grid_shape,
                device=device,
            )
            right = allocate_2d_block_cyclic(
                (columns, columns),
                block_size,
                process_grid_shape,
                device=device,
            )
            left.fill_(2.0)
            right.fill_(2.0)

            left_result = update_left_preconditioner_2d_block_cyclic_(
                local_gradient,
                left,
                beta,
                block_size,
                process_grid_shape,
                shard_dim=shard_dim,
            )
            right_result = update_right_preconditioner_2d_block_cyclic_(
                local_gradient,
                right,
                beta,
                block_size,
                process_grid_shape,
                shard_dim=shard_dim,
            )
            assert left_result is left
            assert right_result is right
            assert left.stride(0) == right.stride(0) == 1
            assert left.dtype == right.dtype == torch.float32

            actual_left = _gather_packed(
                left,
                (rows, rows),
                block_size,
                process_grid_shape,
                rank,
            )
            actual_right = _gather_packed(
                right,
                (columns, columns),
                block_size,
                process_grid_shape,
                rank,
            )
            gradient_float = gradient.float()
            expected_left = torch.full_like(actual_left, 2.0).lerp(
                gradient_float @ gradient_float.T,
                1.0 - beta,
            )
            expected_right = torch.full_like(actual_right, 2.0).lerp(
                gradient_float.T @ gradient_float,
                1.0 - beta,
            )
            torch.testing.assert_close(actual_left, expected_left)
            torch.testing.assert_close(actual_right, expected_right)
    finally:
        dist.destroy_process_group()


def _run_dense_reference_test(rank, world_size, port):
    device = _device_for_rank(rank, world_size)
    _init(rank, world_size, port, device)
    try:
        gradient = _gradient(2 * world_size, 2 * world_size, device)
        local = gradient.chunk(world_size, dim=1)[rank]
        left = update_left_preconditioner_from_col_shards(
            local,
            None,
            beta=0.0,
        )
        right = update_right_preconditioner_from_col_shards(
            local,
            None,
            beta=0.0,
        )
        left_parts = [torch.empty_like(left) for _ in range(world_size)]
        right_parts = [torch.empty_like(right) for _ in range(world_size)]
        dist.all_gather(left_parts, left)
        dist.all_gather(right_parts, right)
        torch.testing.assert_close(torch.cat(left_parts), gradient @ gradient.T)
        torch.testing.assert_close(torch.cat(right_parts), gradient.T @ gradient)
    finally:
        dist.destroy_process_group()


def _run_validation_test(rank, world_size, port):
    device = _device_for_rank(rank, world_size)
    _init(rank, world_size, port, device)
    try:
        process_grid_shape = _grid(world_size)
        gradient = torch.ones(2, 2, device=device)
        buffer = allocate_2d_block_cyclic(
            (2, 2),
            1,
            process_grid_shape,
            device=device,
        )
        functions = (
            update_left_preconditioner_2d_block_cyclic_,
            update_right_preconditioner_2d_block_cyclic_,
        )
        for function in functions:
            with unittest.TestCase().assertRaises(ValueError):
                function(
                    gradient,
                    buffer,
                    -0.1,
                    1,
                    process_grid_shape,
                    shard_dim=1,
                )
            with unittest.TestCase().assertRaises(ValueError):
                function(
                    gradient,
                    buffer,
                    0.0,
                    1,
                    process_grid_shape,
                    shard_dim=2,
                )
    finally:
        dist.destroy_process_group()


class TestPreconditioners(unittest.TestCase):
    def test_packed_row_and_column_shards(self):
        mp.spawn(
            _run_packed_update_test,
            args=(WORLD_SIZE, _free_port()),
            nprocs=WORLD_SIZE,
            join=True,
        )

    def test_dense_reference_kernels(self):
        mp.spawn(
            _run_dense_reference_test,
            args=(WORLD_SIZE, _free_port()),
            nprocs=WORLD_SIZE,
            join=True,
        )

    def test_validation(self):
        mp.spawn(
            _run_validation_test,
            args=(WORLD_SIZE, _free_port()),
            nprocs=WORLD_SIZE,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
