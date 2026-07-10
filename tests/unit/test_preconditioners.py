import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


from soap_tp.ops.preconditioners import (
    update_left_preconditioner_from_col_shards,
    update_right_preconditioner_from_col_shards,
    update_left_preconditioner_from_col_shards_2DblockCyclic_lower,
    update_right_preconditioner_from_col_shards_2DBlockCyclic_lower,
)


WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "4"))


def _gradient(world_size, device="cpu"):
    rows = 2 * world_size
    cols = 2 * world_size
    values = torch.arange(rows * cols, dtype=torch.float64, device=device)
    return ((values.reshape(rows, cols) % 13) - 6.0) / 3.0


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _init_process_group(rank, world_size, port, backend):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def _device_for_rank(rank, world_size):
    if torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def _backend_for_device(device):
    return "nccl" if device.type == "cuda" else "gloo"


def _process_grid_shape(world_size):
    process_rows = int(world_size**0.5)
    while world_size % process_rows != 0:
        process_rows -= 1
    return process_rows, world_size // process_rows


def _block_cyclic_owner(block_row, block_col, process_grid_shape):
    process_rows, process_cols = process_grid_shape
    return (block_row % process_rows) * process_cols + (block_col % process_cols)


def _block_bounds(block_id, block_size, dim):
    start = block_id * block_size
    return start, min(start + block_size, dim)


def _run_update_left_preconditioner_test(rank, world_size, port):
    device = _device_for_rank(rank, world_size)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    _init_process_group(rank, world_size, port, _backend_for_device(device))

    try:
        G = _gradient(world_size, device=device)
        G_local = G.chunk(world_size, dim=1)[rank].contiguous()

        L_local = update_left_preconditioner_from_col_shards(
            G_local,
            None,
            beta=0.0,
        )

        gathered = [torch.empty_like(L_local) for _ in range(world_size)]
        dist.all_gather(gathered, L_local)

        combined = torch.cat(gathered, dim=0)
        expected = G @ G.T

        torch.testing.assert_close(combined.cpu(), expected.cpu())
        assert L_local.device == G_local.device
    finally:
        dist.destroy_process_group()


def _run_update_right_preconditioner_test(rank, world_size, port):
    device = _device_for_rank(rank, world_size)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    _init_process_group(rank, world_size, port, _backend_for_device(device))

    try:
        G = _gradient(world_size, device=device)
        G_local = G.chunk(world_size, dim=1)[rank].contiguous()

        R_local = update_right_preconditioner_from_col_shards(
            G_local,
            None,
            beta=0.0,
        )

        gathered = [torch.empty_like(R_local) for _ in range(world_size)]
        dist.all_gather(gathered, R_local)

        combined = torch.cat(gathered, dim=0)
        expected = G.T @ G

        torch.testing.assert_close(combined.cpu(), expected.cpu())
        assert R_local.device == G_local.device
    finally:
        dist.destroy_process_group()


def _run_update_left_preconditioner_2d_block_cyclic_lower_test(
    rank,
    world_size,
    port,
    process_grid_shape,
):
    device = _device_for_rank(rank, world_size)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    _init_process_group(rank, world_size, port, _backend_for_device(device))

    try:
        block_size = 2
        G = _gradient(world_size, device=device)
        G_local = G.chunk(world_size, dim=1)[rank].contiguous()

        local_tiles = update_left_preconditioner_from_col_shards_2DblockCyclic_lower(
            G_local,
            {},
            block_size,
            process_grid_shape,
        )

        payload = {
            "rank": rank,
            "keys": sorted(local_tiles),
            "device_ok": all(
                tile.device == G_local.device for tile in local_tiles.values()
            ),
            "tiles": {key: tile.cpu() for key, tile in local_tiles.items()},
        }

        gathered = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(payload, gathered, dst=0)

        if rank == 0:
            expected = (G @ G.T).cpu()
            combined = torch.full_like(expected, float("nan"))
            nblocks = (expected.size(0) + block_size - 1) // block_size

            for rank_payload in gathered:
                payload_rank = rank_payload["rank"]
                expected_keys = [
                    (bi, bj)
                    for bi in range(nblocks)
                    for bj in range(bi + 1)
                    if _block_cyclic_owner(bi, bj, process_grid_shape) == payload_rank
                ]

                assert rank_payload["keys"] == expected_keys
                assert rank_payload["device_ok"]

                for (bi, bj), tile in rank_payload["tiles"].items():
                    i0, i1 = _block_bounds(bi, block_size, expected.size(0))
                    j0, j1 = _block_bounds(bj, block_size, expected.size(1))
                    expected_tile = expected[i0:i1, j0:j1]

                    torch.testing.assert_close(tile, expected_tile)
                    combined[i0:i1, j0:j1] = tile

            lower_mask = torch.tril(torch.ones_like(expected, dtype=torch.bool))
            torch.testing.assert_close(combined[lower_mask], expected[lower_mask])
    finally:
        dist.destroy_process_group()


def _run_update_right_preconditioner_2d_block_cyclic_lower_test(rank, world_size, port):
    device = _device_for_rank(rank, world_size)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    _init_process_group(rank, world_size, port, _backend_for_device(device))

    try:
        block_size = 2
        process_grid_shape = _process_grid_shape(world_size)
        G = _gradient(world_size, device=device)
        G_local = G.chunk(world_size, dim=1)[rank].contiguous()

        local_tiles = update_right_preconditioner_from_col_shards_2DBlockCyclic_lower(
            G_local,
            {},
            block_size,
            process_grid_shape,
        )

        payload = {
            "rank": rank,
            "keys": sorted(local_tiles),
            "device_ok": all(
                tile.device == G_local.device for tile in local_tiles.values()
            ),
            "tiles": {key: tile.cpu() for key, tile in local_tiles.items()},
        }

        gathered = [None for _ in range(world_size)] if rank == 0 else None
        dist.gather_object(payload, gathered, dst=0)

        if rank == 0:
            expected = (G.T @ G).cpu()
            combined = torch.full_like(expected, float("nan"))
            nblocks = (expected.size(0) + block_size - 1) // block_size

            for rank_payload in gathered:
                payload_rank = rank_payload["rank"]
                expected_keys = [
                    (bi, bj)
                    for bi in range(nblocks)
                    for bj in range(bi + 1)
                    if _block_cyclic_owner(bi, bj, process_grid_shape) == payload_rank
                ]

                assert rank_payload["keys"] == expected_keys
                assert rank_payload["device_ok"]

                for (bi, bj), tile in rank_payload["tiles"].items():
                    i0, i1 = _block_bounds(bi, block_size, expected.size(0))
                    j0, j1 = _block_bounds(bj, block_size, expected.size(1))
                    expected_tile = expected[i0:i1, j0:j1]

                    torch.testing.assert_close(tile, expected_tile)
                    combined[i0:i1, j0:j1] = tile

            lower_mask = torch.tril(torch.ones_like(expected, dtype=torch.bool))
            torch.testing.assert_close(combined[lower_mask], expected[lower_mask])
    finally:
        dist.destroy_process_group()


class TestPreconditionerDistributedCorrectness(unittest.TestCase):
    def test_update_left_preconditioner_from_col_shards(self):
        mp.spawn(
            _run_update_left_preconditioner_test,
            args=(WORLD_SIZE, _free_port()),
            nprocs=WORLD_SIZE,
            join=True,
        )

    def test_update_left_preconditioner_from_col_shards_2d_block_cyclic_lower_rectangular_grid(
        self,
    ):
        process_grid_shape = (2, 3)
        world_size = process_grid_shape[0] * process_grid_shape[1]
        mp.spawn(
            _run_update_left_preconditioner_2d_block_cyclic_lower_test,
            args=(world_size, _free_port(), process_grid_shape),
            nprocs=world_size,
            join=True,
        )

    def test_update_right_preconditioner_from_col_shards(self):
        mp.spawn(
            _run_update_right_preconditioner_test,
            args=(WORLD_SIZE, _free_port()),
            nprocs=WORLD_SIZE,
            join=True,
        )

    def test_update_right_preconditioner_from_col_shards_2d_block_cyclic_lower(self):
        mp.spawn(
            _run_update_right_preconditioner_2d_block_cyclic_lower_test,
            args=(WORLD_SIZE, _free_port()),
            nprocs=WORLD_SIZE,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
