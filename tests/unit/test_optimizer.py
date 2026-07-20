import os
import socket
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from soap_tp.ops.factorizations import (
    refresh_bases_and_transport_optimizer_state_,
)
from soap_tp.ops.optimizer import (
    adam_update,
    permute_2d_block_cyclic_,
    redistribute_2d_block_cyclic_to_tp_shard,
    redistribute_tp_shard_to_2d_block_cyclic,
)


WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "4"))


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_redistribution_test(rank, world_size, port):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        rows, columns = 2 * world_size, 3 * world_size
        block_size = 5
        global_matrix = torch.arange(
            rows * columns, dtype=torch.float32
        ).reshape(rows, columns)
        process_grids = (
            (1, world_size),
            (2, world_size // 2),
            (world_size, 1),
        )

        for process_grid in process_grids:
            process_row = rank // process_grid[1]
            process_column = rank % process_grid[1]
            local_rows = [
                index
                for index in range(rows)
                if (index // block_size) % process_grid[0] == process_row
            ]
            local_columns = [
                index
                for index in range(columns)
                if (index // block_size) % process_grid[1] == process_column
            ]
            row_index = torch.tensor(local_rows, dtype=torch.long)
            column_index = torch.tensor(local_columns, dtype=torch.long)
            expected_block_cyclic = global_matrix.index_select(
                0, row_index
            ).index_select(1, column_index)

            for shard_dim in (0, 1):
                shard_size = global_matrix.size(shard_dim) // world_size
                shard_start = rank * shard_size
                tp_shard = global_matrix.narrow(
                    shard_dim, shard_start, shard_size
                ).contiguous()

                block_cyclic = redistribute_tp_shard_to_2d_block_cyclic(
                    tp_shard,
                    global_matrix.shape,
                    block_size,
                    process_grid,
                    shard_dim=shard_dim,
                )
                assert block_cyclic.stride(0) == 1
                torch.testing.assert_close(
                    block_cyclic[: len(local_rows), : len(local_columns)],
                    expected_block_cyclic,
                )

                leading_dimension = max(1, len(local_rows) + 2)
                storage_columns = max(1, len(local_columns))
                padded = torch.full(
                    (storage_columns, leading_dimension), torch.nan
                ).T
                padded[: len(local_rows), : len(local_columns)].copy_(
                    block_cyclic[: len(local_rows), : len(local_columns)]
                )

                result = redistribute_2d_block_cyclic_to_tp_shard(
                    padded,
                    global_matrix.shape,
                    block_size,
                    process_grid,
                    shard_dim=shard_dim,
                )
                torch.testing.assert_close(result, tp_shard)
                assert torch.isnan(padded[len(local_rows) :, :]).all()

            row_order = tuple(reversed(range(rows)))
            column_order = tuple(
                list(range(1, columns, 2)) + list(range(0, columns, 2))
            )
            column_shard = global_matrix.chunk(world_size, dim=1)[rank]
            block_cyclic = redistribute_tp_shard_to_2d_block_cyclic(
                column_shard,
                global_matrix.shape,
                block_size,
                process_grid,
                shard_dim=1,
            )
            result = permute_2d_block_cyclic_(
                block_cyclic,
                global_matrix.shape,
                row_order,
                column_order,
                block_size,
                process_grid,
            )
            assert result is block_cyclic
            expected = global_matrix[list(row_order)][:, list(column_order)]
            torch.testing.assert_close(
                result[: len(local_rows), : len(local_columns)],
                expected.index_select(0, row_index).index_select(
                    1, column_index
                ),
            )
    finally:
        dist.destroy_process_group()


def _pack_column_major(matrix, lda_padding=0):
    leading_dimension = max(1, matrix.size(0) + lda_padding)
    output = torch.full(
        (max(1, matrix.size(1)), leading_dimension),
        torch.nan,
        dtype=matrix.dtype,
        device=matrix.device,
    ).T
    output[: matrix.size(0), : matrix.size(1)].copy_(matrix)
    return output


class _FakeSlateBinding:
    def __init__(self, tensors):
        self.tensors = {tensor.data_ptr(): tensor for tensor in tensors}
        self.calls = []

    @staticmethod
    def mpi_world_rank_and_size():
        return 0, 1

    @staticmethod
    def compiled_gpu_backend():
        return "none"

    def slate_backward_rotation_float(
        self,
        q_left,
        matrix,
        q_right,
        rows,
        columns,
        *_args,
    ):
        self.calls.append("backward")
        Q_left = self.tensors[q_left][:rows, :rows]
        X = self.tensors[matrix][:rows, :columns]
        Q_right = self.tensors[q_right][:columns, :columns]
        X.copy_(Q_left @ X.clone() @ Q_right.T)

    def slate_symmetric_multiply_float(
        self,
        preconditioner,
        basis,
        work,
        size,
        *_args,
    ):
        self.calls.append("symmetric_multiply")
        A = self.tensors[preconditioner][:size, :size]
        Q = self.tensors[basis][:size, :size]
        Y = self.tensors[work][:size, :size]
        Y.copy_(A @ Q.clone())

    def slate_qr_float(
        self,
        matrix,
        basis,
        size,
        *_args,
    ):
        self.calls.append("qr")
        Y = self.tensors[matrix][:size, :size]
        Q = self.tensors[basis][:size, :size]
        Q_new, _ = torch.linalg.qr(Y.clone())
        Q.copy_(Q_new)

    def slate_forward_rotation_float(
        self,
        q_left,
        matrix,
        q_right,
        rows,
        columns,
        *_args,
    ):
        self.calls.append("forward")
        Q_left = self.tensors[q_left][:rows, :rows]
        X = self.tensors[matrix][:rows, :columns]
        Q_right = self.tensors[q_right][:columns, :columns]
        X.copy_(Q_left.T @ X.clone() @ Q_right)


def _run_basis_refresh_test(rank, world_size, port):
    assert rank == 0
    assert world_size == 1
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )
    try:
        rows, columns = 4, 3
        block_size = 2
        process_grid = (1, 1)
        left_preconditioner_global = torch.tensor(
            [
                [6.0, 0.2, 0.1, 0.0],
                [0.2, 5.0, 0.3, 0.1],
                [0.1, 0.3, 4.0, 0.4],
                [0.0, 0.1, 0.4, 3.0],
            ],
            dtype=torch.float32,
        )
        right_preconditioner_global = torch.tensor(
            [
                [4.0, 0.2, 0.1],
                [0.2, 3.0, 0.3],
                [0.1, 0.3, 2.0],
            ],
            dtype=torch.float32,
        )
        Q_left_global, _ = torch.linalg.qr(
            torch.tensor(
                [
                    [1.0, 0.2, 0.3, 0.4],
                    [0.1, 1.0, 0.2, 0.3],
                    [0.2, 0.1, 1.0, 0.2],
                    [0.3, 0.2, 0.1, 1.0],
                ]
            )
        )
        Q_right_global, _ = torch.linalg.qr(
            torch.tensor(
                [
                    [1.0, 0.2, 0.3],
                    [0.1, 1.0, 0.2],
                    [0.2, 0.1, 1.0],
                ]
            )
        )
        momentum_global = torch.arange(
            rows * columns, dtype=torch.float32
        ).reshape(rows, columns) / 7.0
        variance_global = torch.arange(
            1, rows * columns + 1, dtype=torch.float32
        ).reshape(rows, columns)

        expected_parameter_momentum = (
            Q_left_global @ momentum_global @ Q_right_global.T
        )
        left_estimates = torch.diag(
            Q_left_global.T
            @ left_preconditioner_global
            @ Q_left_global
        )
        right_estimates = torch.diag(
            Q_right_global.T
            @ right_preconditioner_global
            @ Q_right_global
        )
        left_order = torch.argsort(
            left_estimates,
            descending=True,
        ).tolist()
        right_order = torch.argsort(
            right_estimates,
            descending=True,
        ).tolist()
        sorted_Q_left = Q_left_global[:, list(left_order)]
        sorted_Q_right = Q_right_global[:, list(right_order)]
        expected_Q_left, _ = torch.linalg.qr(
            left_preconditioner_global @ sorted_Q_left
        )
        expected_Q_right, _ = torch.linalg.qr(
            right_preconditioner_global @ sorted_Q_right
        )
        expected_momentum = (
            expected_Q_left.T
            @ expected_parameter_momentum
            @ expected_Q_right
        )
        expected_variance = variance_global[list(left_order)][
            :, list(right_order)
        ]

        momentum = _pack_column_major(momentum_global, 2)
        variance = _pack_column_major(variance_global, 2)
        left_preconditioner = _pack_column_major(
            left_preconditioner_global, 1
        )
        right_preconditioner = _pack_column_major(
            right_preconditioner_global, 1
        )
        Q_left = _pack_column_major(Q_left_global, 1)
        Q_right = _pack_column_major(Q_right_global, 1)
        left_work = _pack_column_major(torch.zeros(rows, rows), 1)
        right_work = _pack_column_major(
            torch.zeros(columns, columns), 1
        )
        binding = _FakeSlateBinding(
            (
                momentum,
                variance,
                left_preconditioner,
                right_preconditioner,
                Q_left,
                Q_right,
                left_work,
                right_work,
            )
        )

        refresh_bases_and_transport_optimizer_state_(
            momentum,
            variance,
            left_preconditioner,
            right_preconditioner,
            Q_left,
            Q_right,
            left_work,
            right_work,
            (rows, columns),
            block_size,
            process_grid,
            slate_binding=binding,
        )

        assert binding.calls == [
            "backward",
            "symmetric_multiply",
            "qr",
            "symmetric_multiply",
            "qr",
            "forward",
        ]
        torch.testing.assert_close(momentum[:rows, :columns], expected_momentum)
        torch.testing.assert_close(variance[:rows, :columns], expected_variance)
        torch.testing.assert_close(Q_left[:rows, :rows], expected_Q_left)
        torch.testing.assert_close(
            Q_right[:columns, :columns], expected_Q_right
        )
        assert torch.isnan(momentum[rows:, :]).all()
        assert torch.isnan(variance[rows:, :]).all()
    finally:
        dist.destroy_process_group()


class TestAdamUpdate(unittest.TestCase):
    def test_updates_state_and_returns_bias_corrected_update(self):
        gradient = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        momentum = torch.tensor([[0.2, 0.3], [-0.4, 0.5]])
        variance = torch.tensor([[0.7, 0.8], [0.9, 1.0]])
        expected_momentum = momentum * 0.8 + gradient * 0.2
        expected_variance = variance * 0.9 + gradient.square() * 0.1
        expected_update = (expected_momentum / (1 - 0.8**3)) / (
            expected_variance.sqrt() / (1 - 0.9**3) ** 0.5 + 1e-6
        )

        update = adam_update(
            gradient,
            momentum,
            variance,
            step=3,
            beta1=0.8,
            beta2=0.9,
            eps=1e-6,
        )

        torch.testing.assert_close(momentum, expected_momentum)
        torch.testing.assert_close(variance, expected_variance)
        torch.testing.assert_close(update, expected_update)

    def test_rejects_invalid_hyperparameters_and_shapes(self):
        gradient = torch.ones(2, 2)
        state = torch.zeros_like(gradient)
        for arguments in (
            {"step": 0},
            {"step": 1, "beta1": 1.0},
            {"step": 1, "beta2": -0.1},
            {"step": 1, "eps": 0.0},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    adam_update(
                        gradient,
                        state.clone(),
                        state.clone(),
                        **arguments,
                    )

        with self.assertRaises(ValueError):
            adam_update(
                gradient,
                torch.zeros(2, 1),
                state,
                step=1,
            )


class TestRedistribution(unittest.TestCase):
    def test_tp_shard_block_cyclic_round_trip_for_both_dimensions(self):
        self.assertEqual(WORLD_SIZE, 4)
        mp.spawn(
            _run_redistribution_test,
            args=(WORLD_SIZE, _free_port()),
            nprocs=WORLD_SIZE,
            join=True,
        )


class TestBasisRefresh(unittest.TestCase):
    def test_transports_momentum_and_reorders_variance(self):
        mp.spawn(
            _run_basis_refresh_test,
            args=(1, _free_port()),
            nprocs=1,
            join=True,
        )


if __name__ == "__main__":
    unittest.main()
