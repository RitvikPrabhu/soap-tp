import unittest
from unittest.mock import patch

import torch

from soap_tp.ops import factorizations


def _padded_column_major(size, values=None):
    buffer = torch.full((size, size + 2), torch.nan).T
    if values is not None:
        buffer[:size, :size].copy_(values)
    return buffer


class _PoisonElpaBinding:
    def __getattr__(self, name):
        raise AssertionError(f"eigh unexpectedly accessed ELPA attribute {name!r}")


class TestInitializeBasis(unittest.TestCase):
    def test_eigh_returns_descending_eigenbasis_without_accessing_elpa(self):
        matrix = torch.tensor(
            [
                [4.0, 1.0, 0.0],
                [1.0, 3.0, 1.0],
                [0.0, 1.0, 2.0],
            ]
        )
        preconditioner = _padded_column_major(3, matrix)
        Q = _padded_column_major(3)
        work = _padded_column_major(3)
        eigenvalues = torch.empty(3)

        with (
            patch.object(
                factorizations.dist,
                "is_initialized",
                return_value=True,
            ),
            patch.object(factorizations.dist, "get_rank", return_value=0),
            patch.object(
                factorizations.dist,
                "get_world_size",
                return_value=1,
            ),
            patch.object(factorizations.dist, "reduce") as reduce,
            patch.object(factorizations.dist, "broadcast") as broadcast,
            patch.object(
                factorizations.dist,
                "all_reduce",
                side_effect=AssertionError("eigh used an ELPA-only collective"),
            ),
        ):
            result = factorizations.initialize_basis_2d_block_cyclic_(
                preconditioner,
                Q,
                work,
                eigenvalues,
                3,
                2,
                (1, 1),
                implementation="eigh",
                elpa_binding=_PoisonElpaBinding(),
            )

        self.assertIs(result, Q)
        reduce.assert_called_once()
        self.assertEqual(broadcast.call_count, 2)
        logical_Q = Q[:3, :3]
        torch.testing.assert_close(
            matrix @ logical_Q,
            logical_Q * eigenvalues.unsqueeze(0),
        )
        torch.testing.assert_close(logical_Q.T @ logical_Q, torch.eye(3))
        self.assertTrue(torch.all(eigenvalues[:-1] >= eigenvalues[1:]))
        self.assertTrue(torch.isnan(Q[3:, :]).all())

    def test_default_implementation_is_elpa(self):
        matrix = torch.eye(2)
        preconditioner = _padded_column_major(2, matrix)
        Q = _padded_column_major(2)
        work = _padded_column_major(2)

        with (
            patch.object(
                factorizations.dist,
                "is_initialized",
                return_value=True,
            ),
            patch.object(factorizations.dist, "get_rank", return_value=0),
            patch.object(
                factorizations.dist,
                "get_world_size",
                return_value=1,
            ),
            self.assertRaisesRegex(ValueError, "exact ELPA leading dimension"),
        ):
            factorizations.initialize_basis_2d_block_cyclic_(
                preconditioner,
                Q,
                work,
                torch.empty(2),
                2,
                1,
                (1, 1),
                elpa_binding=_PoisonElpaBinding(),
            )

    def test_rejects_unknown_implementation_before_distributed_validation(self):
        matrix = torch.empty((1, 1)).T
        with self.assertRaisesRegex(
            ValueError,
            "implementation must be 'elpa' or 'eigh'",
        ):
            factorizations.initialize_basis_2d_block_cyclic_(
                matrix,
                matrix.clone(),
                matrix.clone(),
                torch.empty(1),
                1,
                1,
                (1, 1),
                implementation="unknown",
            )


if __name__ == "__main__":
    unittest.main()
