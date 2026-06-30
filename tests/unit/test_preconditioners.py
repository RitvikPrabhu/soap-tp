import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from soap_tp.ops.preconditioners import (
    update_left_preconditioner_from_col_shards,
    update_right_preconditioner_from_col_shards,
)


WORLD_SIZE = 4


def _gradient(device="cpu"):
    return torch.tensor(
        [
            [1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 1.5, 0.0],
            [0.5, -3.0, 4.0, 1.0, -1.0, 2.5, 0.25, -0.5],
            [2.0, 0.0, 1.5, -2.0, 0.75, -1.5, 3.0, 1.0],
            [-1.0, 1.0, 0.25, 3.0, 2.0, 0.5, -2.5, 4.0],
        ],
        dtype=torch.float64,
        device=device,
    )


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _init_process_group(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _left_worker(rank, world_size, port):
    _init_process_group(rank, world_size, port)
    try:
        g = _gradient()
        g_local = g.chunk(world_size, dim=1)[rank].contiguous()
        beta = 0.25
        l_prev = torch.full((g.shape[0] // world_size, g.shape[0]), 2.0, dtype=g.dtype)
        l_prev_before = l_prev.clone()

        actual = update_left_preconditioner_from_col_shards(
            g_local,
            l_prev,
            beta,
            TP_group=dist.group.WORLD,
        )

        expected_current = (g @ g.T).chunk(world_size, dim=0)[rank]
        expected = beta * l_prev_before + (1.0 - beta) * expected_current

        torch.testing.assert_close(actual, expected)
        assert actual.data_ptr() == l_prev.data_ptr()
    finally:
        dist.destroy_process_group()


def _right_worker(rank, world_size, port):
    _init_process_group(rank, world_size, port)
    try:
        g = _gradient()
        g_local = g.chunk(world_size, dim=1)[rank].contiguous()
        beta = 0.5
        r_prev = torch.full((g_local.shape[1], g.shape[1]), -1.0, dtype=g.dtype)
        r_prev_before = r_prev.clone()

        actual = update_right_preconditioner_from_col_shards(
            g_local,
            r_prev,
            beta,
            TP_group=dist.group.WORLD,
        )

        expected_current = g_local.T @ g
        expected = beta * r_prev_before + (1.0 - beta) * expected_current

        torch.testing.assert_close(actual, expected)
        assert actual.data_ptr() == r_prev.data_ptr()
    finally:
        dist.destroy_process_group()


class TestPreconditionerDistributedCorrectness(unittest.TestCase):
    def test_update_left_preconditioner_matches_full_gram_on_four_ranks(self):
        mp.spawn(_left_worker, args=(WORLD_SIZE, _free_port()), nprocs=WORLD_SIZE, join=True)

    def test_update_right_preconditioner_matches_full_gram_on_four_ranks(self):
        mp.spawn(_right_worker, args=(WORLD_SIZE, _free_port()), nprocs=WORLD_SIZE, join=True)


if __name__ == "__main__":
    unittest.main()
