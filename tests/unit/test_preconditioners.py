import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


from soap_tp.ops.preconditioners import (
    lerp_preconditioner_2DblockCyclic_lower_,
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


def _run_update_left_preconditioner_2d_block_cyclic_lower_test(rank, world_size, port):
    device = _device_for_rank(rank, world_size)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    _init_process_group(rank, world_size, port, _backend_for_device(device))

    try:
        block_size = 2
        process_grid_shape = _process_grid_shape(world_size)
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


def _run_dense_edge_test(rank, world_size, port):
    device = _device_for_rank(rank, world_size)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    _init_process_group(rank, world_size, port, _backend_for_device(device))

    try:
        G = _gradient(world_size, device=device)
        G_local = G.chunk(world_size, dim=1)[rank]
        assert not G_local.is_contiguous()
        beta = 0.25
        left_previous = torch.full(
            (G.size(0) // world_size, G.size(0)),
            2.0,
            dtype=G.dtype,
            device=device,
        )
        right_previous = torch.full(
            (G_local.size(1), G.size(1)), 2.0, dtype=G.dtype, device=device
        )

        left_result = update_left_preconditioner_from_col_shards(
            G_local, left_previous, beta
        )
        right_result = update_right_preconditioner_from_col_shards(
            G_local, right_previous, beta
        )
        left_expected = torch.full_like(G @ G.T, 2.0).lerp(
            G @ G.T, 1.0 - beta
        )
        right_expected = torch.full_like(G.T @ G, 2.0).lerp(
            G.T @ G, 1.0 - beta
        )
        left_rows = G.size(0) // world_size
        right_rows = G_local.size(1)
        torch.testing.assert_close(
            left_result,
            left_expected[rank * left_rows : (rank + 1) * left_rows],
        )
        torch.testing.assert_close(
            right_result,
            right_expected[rank * right_rows : (rank + 1) * right_rows],
        )
        assert left_result is left_previous
        assert right_result is right_previous
    finally:
        dist.destroy_process_group()


def _assert_owned_tiles(tiles, expected, block_size, process_grid_shape, rank):
    nblocks = (expected.size(0) + block_size - 1) // block_size
    expected_keys = [
        (bi, bj)
        for bi in range(nblocks)
        for bj in range(bi + 1)
        if _block_cyclic_owner(bi, bj, process_grid_shape) == rank
    ]
    assert sorted(tiles) == expected_keys
    for (bi, bj), tile in tiles.items():
        i0, i1 = _block_bounds(bi, block_size, expected.size(0))
        j0, j1 = _block_bounds(bj, block_size, expected.size(1))
        assert tile.shape == (i1 - i0, j1 - j0)
        torch.testing.assert_close(tile, expected[i0:i1, j0:j1])


def _run_block_cyclic_edge_test(rank, world_size, port):
    device = _device_for_rank(rank, world_size)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    _init_process_group(rank, world_size, port, _backend_for_device(device))

    try:
        rows = world_size + 1
        columns = 2 * world_size + 1
        block_size = 3
        process_grid_shape = _process_grid_shape(world_size)
        values = torch.arange(rows * columns, dtype=torch.float64, device=device)
        A = ((values.reshape(rows, columns) % 11) - 5.0) / 3.0
        A_local = torch.tensor_split(A, world_size, dim=1)[rank].contiguous()

        left_tiles = update_left_preconditioner_from_col_shards_2DblockCyclic_lower(
            A_local, {}, block_size, process_grid_shape
        )
        right_tiles = update_right_preconditioner_from_col_shards_2DBlockCyclic_lower(
            A_local, {}, block_size, process_grid_shape
        )
        _assert_owned_tiles(
            left_tiles, A @ A.T, block_size, process_grid_shape, rank
        )
        _assert_owned_tiles(
            right_tiles, A.T @ A, block_size, process_grid_shape, rank
        )
    finally:
        dist.destroy_process_group()


def _run_block_cyclic_validation_test(rank, world_size, port):
    device = _device_for_rank(rank, world_size)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    _init_process_group(rank, world_size, port, _backend_for_device(device))

    try:
        functions = (
            update_left_preconditioner_from_col_shards_2DblockCyclic_lower,
            update_right_preconditioner_from_col_shards_2DBlockCyclic_lower,
        )
        for function in functions:
            with unittest.TestCase().assertRaises(ValueError):
                function(torch.ones(3, device=device), {}, 2, (1, world_size))
            with unittest.TestCase().assertRaises(ValueError):
                function(torch.ones(2, 2, device=device), {}, 0, (1, world_size))
            with unittest.TestCase().assertRaises(ValueError):
                function(torch.ones(2, 2, device=device), {}, 2, (1, world_size + 1))

        inconsistent_rows = 2 + (rank == world_size - 1)
        with unittest.TestCase().assertRaises(ValueError):
            update_right_preconditioner_from_col_shards_2DBlockCyclic_lower(
                torch.ones(inconsistent_rows, 1, device=device),
                {},
                2,
                (1, world_size),
            )
    finally:
        dist.destroy_process_group()


class TestBlockCyclicInterpolation(unittest.TestCase):
    # Tests: beta endpoints and midpoint interpolation across multiple tiles.
    # Expected: tile values become 9, 5, or 1 for beta 0, 0.5, or 1, while the
    # original dictionary and tile objects are returned and mutated in place.
    def test_lerp_endpoints_midpoint_and_identity(self):
        for beta, expected in ((0.0, 9.0), (0.5, 5.0), (1.0, 1.0)):
            with self.subTest(beta=beta):
                previous = {
                    (0, 0): torch.ones(2, 2),
                    (1, 0): torch.ones(1, 2),
                }
                original_tiles = dict(previous)
                current = {
                    (0, 0): torch.full((2, 2), 9.0),
                    (1, 0): torch.full((1, 2), 9.0),
                }
                result = lerp_preconditioner_2DblockCyclic_lower_(
                    previous, current, beta
                )
                self.assertIs(result, previous)
                for key, tile in result.items():
                    self.assertIs(tile, original_tiles[key])
                    torch.testing.assert_close(tile, torch.full_like(tile, expected))

    # Tests: empty mappings and the explicit equal-key-set requirement.
    # Expected: empty inputs return the original empty mapping; mismatched keys
    # raise ValueError without modifying the previous tile.
    def test_lerp_empty_and_mismatched_keys(self):
        empty = {}
        self.assertIs(
            lerp_preconditioner_2DblockCyclic_lower_(empty, {}, 0.5), empty
        )
        previous = {(0, 0): torch.tensor([1.0])}
        before = previous[(0, 0)].clone()
        with self.assertRaises(ValueError):
            lerp_preconditioner_2DblockCyclic_lower_(
                previous, {(1, 0): torch.tensor([2.0])}, 0.5
            )
        torch.testing.assert_close(previous[(0, 0)], before)


class TestPreconditionerDistributedCorrectness(unittest.TestCase):
    # Tests: dense interpolation into existing tensors using non-contiguous
    # local column shards for both left and right preconditioners.
    # Expected: returned tensors retain their identities and equal the correct
    # beta-weighted row shards of G @ G.T and G.T @ G.
    def test_dense_accumulation_and_noncontiguous_shards(self):
        mp.spawn(
            _run_dense_edge_test,
            args=(WORLD_SIZE, _free_port()),
            nprocs=WORLD_SIZE,
            join=True,
        )

    # Tests: uneven column shards and partial final blocks in both block-cyclic
    # Gram computations.
    # Expected: every rank receives exactly its owned lower-triangular tiles,
    # each equal to the corresponding dense Gram block.
    def test_uneven_shards_and_partial_blocks(self):
        mp.spawn(
            _run_block_cyclic_edge_test,
            args=(WORLD_SIZE, _free_port()),
            nprocs=WORLD_SIZE,
            join=True,
        )

    # Tests: documented validation for tensor rank, positive block size,
    # process-grid size, and equal row counts in the right block-cyclic update.
    # Expected: every malformed input raises ValueError on every participating
    # rank without producing tiles.
    def test_block_cyclic_input_validation(self):
        mp.spawn(
            _run_block_cyclic_validation_test,
            args=(WORLD_SIZE, _free_port()),
            nprocs=WORLD_SIZE,
            join=True,
        )

    # Tests: standard distributed left Gram computation from equal-width shards.
    # Expected: concatenating local results equals G @ G.T on the input device.
    def test_update_left_preconditioner_from_col_shards(self):
        mp.spawn(
            _run_update_left_preconditioner_test,
            args=(WORLD_SIZE, _free_port()),
            nprocs=WORLD_SIZE,
            join=True,
        )

    # Tests: standard lower-triangular block ownership for the left Gram matrix.
    # Expected: every owned tile equals its block of G @ G.T, and all lower
    # entries are represented exactly once across ranks.
    def test_update_left_preconditioner_from_col_shards_2d_block_cyclic_lower(self):
        mp.spawn(
            _run_update_left_preconditioner_2d_block_cyclic_lower_test,
            args=(WORLD_SIZE, _free_port()),
            nprocs=WORLD_SIZE,
            join=True,
        )

    # Tests: standard distributed right Gram computation from equal-width shards.
    # Expected: concatenating local results equals G.T @ G on the input device.
    def test_update_right_preconditioner_from_col_shards(self):
        mp.spawn(
            _run_update_right_preconditioner_test,
            args=(WORLD_SIZE, _free_port()),
            nprocs=WORLD_SIZE,
            join=True,
        )

    # Tests: standard lower-triangular block ownership for the right Gram matrix.
    # Expected: every owned tile equals its block of G.T @ G, and all lower
    # entries are represented exactly once across ranks.
    def test_update_right_preconditioner_from_col_shards_2d_block_cyclic_lower(self):
        mp.spawn(
            _run_update_right_preconditioner_2d_block_cyclic_lower_test,
            args=(WORLD_SIZE, _free_port()),
            nprocs=WORLD_SIZE,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
