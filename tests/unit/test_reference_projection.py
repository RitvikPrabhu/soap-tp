import unittest

import torch

from soap_tp.reference.sequential import project_back, project_gradient
from soap_tp.utils.fixtures import make_tiny_gradient


class TestReferenceProjection(unittest.TestCase):
    def test_projection_round_trip_with_identity_basis(self):
        g = make_tiny_gradient()
        q_left = torch.eye(g.shape[0], dtype=g.dtype)
        q_right = torch.eye(g.shape[1], dtype=g.dtype)
        projected = project_gradient(g, q_left, q_right)
        restored = project_back(projected, q_left, q_right)
        torch.testing.assert_close(restored, g)

    def test_projection_round_trip_with_qr_basis(self):
        g = make_tiny_gradient()
        q_left, _ = torch.linalg.qr(torch.randn(g.shape[0], g.shape[0], dtype=g.dtype))
        q_right, _ = torch.linalg.qr(torch.randn(g.shape[1], g.shape[1], dtype=g.dtype))
        projected = project_gradient(g, q_left, q_right)
        restored = project_back(projected, q_left, q_right)
        torch.testing.assert_close(restored, g, rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
