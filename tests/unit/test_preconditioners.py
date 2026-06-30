import os
import socket
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from soap_tp.ops.preconditioners import update_left_preconditioner_from_col_shards


def _gradient(dtype=torch.float64, device="cpu"):
    return torch.tensor(
        [
            [1.0, 2.0, -1.0, 0.5],
            [0.5, -3.0, 4.0, 1.0],
            [2.0, 0.0, 1.5, -2.0],
            [-1.0, 1.0, 0.25, 3.0],
        ],
        dtype=dtype,
        device=device,
    )


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _distributed_worker(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        device = torch.device("cuda", rank)
        g = _gradient(dtype=torch.float32, device=device)
        local = g.chunk(world_size, dim=1)[rank].contiguous()
        expected_full = g @ g.T

        actual_full = update_left_preconditioner_from_col_shards(
            local,
            None,
            beta=0.0,
            TP_group=dist.group.WORLD,
        )
        torch.testing.assert_close(actual_full, expected_full)

        actual_chunk = update_left_preconditioner_from_col_shards(
            local,
            None,
            beta=0.0,
            TP_group=dist.group.WORLD,
            reduce_scatter=True,
        )
        expected_chunk = expected_full.chunk(world_size, dim=0)[rank]
        torch.testing.assert_close(actual_chunk, expected_chunk)
    finally:
        dist.destroy_process_group()


class TestUpdateLeftPreconditionerFromColShards(unittest.TestCase):
    def test_all_reduce_initializes_and_returns_full_preconditioner(self):
        g = _gradient(dtype=torch.float32)
        local = g[:, :2].contiguous()
        expected_full = g @ g.T
        group = object()

        def fake_all_reduce(output, op=None, group=None):
            self.assertIs(op, dist.ReduceOp.SUM)
            self.assertIs(group, group_handle)
            torch.testing.assert_close(output, local @ local.T)
            output.copy_(expected_full)

        group_handle = group
        with mock.patch("soap_tp.ops.preconditioners.dist.all_reduce", side_effect=fake_all_reduce):
            actual = update_left_preconditioner_from_col_shards(
                local,
                None,
                beta=0.0,
                TP_group=group_handle,
            )

        self.assertEqual(actual.shape, (g.shape[0], g.shape[0]))
        self.assertEqual(actual.dtype, g.dtype)
        self.assertEqual(actual.device, g.device)
        torch.testing.assert_close(actual, expected_full)

    def test_all_reduce_applies_ema_update_in_place(self):
        g = _gradient()
        local = g[:, :2].contiguous()
        previous = torch.full((g.shape[0], g.shape[0]), 2.0, dtype=g.dtype)
        previous_before = previous.clone()
        reduced = g @ g.T
        beta = 0.25

        with mock.patch(
            "soap_tp.ops.preconditioners.dist.all_reduce",
            side_effect=lambda output, **_: output.copy_(reduced),
        ):
            actual = update_left_preconditioner_from_col_shards(
                local,
                previous,
                beta=beta,
                TP_group=object(),
            )

        self.assertIs(actual, previous)
        torch.testing.assert_close(actual, beta * previous_before + (1.0 - beta) * reduced)

    def test_all_reduce_beta_one_preserves_previous_value(self):
        g = _gradient()
        previous = torch.randn(g.shape[0], g.shape[0], dtype=g.dtype)
        previous_before = previous.clone()

        with mock.patch(
            "soap_tp.ops.preconditioners.dist.all_reduce",
            side_effect=lambda output, **_: output.fill_(123.0),
        ):
            actual = update_left_preconditioner_from_col_shards(
                g[:, :2].contiguous(),
                previous,
                beta=1.0,
                TP_group=object(),
            )

        self.assertIs(actual, previous)
        torch.testing.assert_close(actual, previous_before)

    def test_reduce_scatter_initializes_and_returns_row_chunk(self):
        g = _gradient(dtype=torch.float32)
        local = g[:, :2].contiguous()
        expected_chunk = (g @ g.T).chunk(2, dim=0)[0]
        group = object()

        def fake_reduce_scatter(output, input_list, op=None, group=None):
            self.assertIs(op, dist.ReduceOp.SUM)
            self.assertIs(group, group_handle)
            self.assertEqual(len(input_list), 2)
            local_chunks = list((local @ local.T).chunk(2, dim=0))
            for actual_chunk, expected_local_chunk in zip(input_list, local_chunks):
                torch.testing.assert_close(actual_chunk, expected_local_chunk)
            output.copy_(expected_chunk)

        group_handle = group
        with (
            mock.patch("soap_tp.ops.preconditioners.dist.get_world_size", return_value=2),
            mock.patch("soap_tp.ops.preconditioners.dist.reduce_scatter", side_effect=fake_reduce_scatter),
        ):
            actual = update_left_preconditioner_from_col_shards(
                local,
                None,
                beta=0.0,
                TP_group=group_handle,
                reduce_scatter=True,
            )

        self.assertEqual(actual.shape, expected_chunk.shape)
        self.assertEqual(actual.dtype, g.dtype)
        self.assertEqual(actual.device, g.device)
        torch.testing.assert_close(actual, expected_chunk)

    def test_reduce_scatter_applies_ema_update_in_place(self):
        g = _gradient()
        local = g[:, :2].contiguous()
        previous = torch.full((2, g.shape[0]), -1.0, dtype=g.dtype)
        previous_before = previous.clone()
        reduced_chunk = (g @ g.T).chunk(2, dim=0)[0]
        beta = 0.5

        with (
            mock.patch("soap_tp.ops.preconditioners.dist.get_world_size", return_value=2),
            mock.patch(
                "soap_tp.ops.preconditioners.dist.reduce_scatter",
                side_effect=lambda output, input_list, **_: output.copy_(reduced_chunk),
            ),
        ):
            actual = update_left_preconditioner_from_col_shards(
                local,
                previous,
                beta=beta,
                TP_group=object(),
                reduce_scatter=True,
            )

        self.assertIs(actual, previous)
        torch.testing.assert_close(actual, beta * previous_before + (1.0 - beta) * reduced_chunk)

    def test_rejects_invalid_beta(self):
        g = _gradient()
        for beta in (-0.01, 1.01):
            with self.subTest(beta=beta):
                with self.assertRaisesRegex(ValueError, "beta must be in"):
                    update_left_preconditioner_from_col_shards(g, None, beta=beta)

    def test_rejects_non_matrix_gradient(self):
        with self.assertRaisesRegex(ValueError, "G_local must be 2D"):
            update_left_preconditioner_from_col_shards(torch.ones(2, 2, 2), None, beta=0.9)

    def test_all_reduce_rejects_wrong_previous_shape_before_collective(self):
        g = _gradient()
        with mock.patch("soap_tp.ops.preconditioners.dist.all_reduce") as all_reduce:
            with self.assertRaisesRegex(ValueError, "reduce_scatter=False"):
                update_left_preconditioner_from_col_shards(
                    g,
                    torch.zeros(g.shape[0], g.shape[0] + 1, dtype=g.dtype),
                    beta=0.9,
                    TP_group=object(),
                )
        all_reduce.assert_not_called()

    def test_reduce_scatter_requires_even_row_split_before_collective(self):
        g = _gradient()[:3]
        with (
            mock.patch("soap_tp.ops.preconditioners.dist.get_world_size", return_value=2),
            mock.patch("soap_tp.ops.preconditioners.dist.reduce_scatter") as reduce_scatter,
        ):
            with self.assertRaisesRegex(ValueError, "requires m to be divisible"):
                update_left_preconditioner_from_col_shards(
                    g,
                    None,
                    beta=0.9,
                    TP_group=object(),
                    reduce_scatter=True,
                )
        reduce_scatter.assert_not_called()

    def test_reduce_scatter_rejects_wrong_previous_shape_before_collective(self):
        g = _gradient()
        with (
            mock.patch("soap_tp.ops.preconditioners.dist.get_world_size", return_value=2),
            mock.patch("soap_tp.ops.preconditioners.dist.reduce_scatter") as reduce_scatter,
        ):
            with self.assertRaisesRegex(ValueError, "L_prev must have shape"):
                update_left_preconditioner_from_col_shards(
                    g,
                    torch.zeros(g.shape[0], g.shape[0], dtype=g.dtype),
                    beta=0.9,
                    TP_group=object(),
                    reduce_scatter=True,
                )
        reduce_scatter.assert_not_called()


@unittest.skipUnless(
    dist.is_available() and dist.is_nccl_available() and torch.cuda.device_count() >= 2,
    "requires at least two CUDA GPUs and NCCL",
)
class TestUpdateLeftPreconditionerFromColShardsDistributed(unittest.TestCase):
    def test_real_collectives_on_two_gpus(self):
        mp.spawn(
            _distributed_worker,
            args=(2, _free_port()),
            nprocs=2,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
