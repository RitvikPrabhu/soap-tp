import unittest

import torch

from soap_tp.reference.sequential import gram_ggt, gram_gtg
from soap_tp.utils.fixtures import make_tiny_gradient


class TestReferenceGramOps(unittest.TestCase):
    def test_ggt_shape_and_value(self):
        g = make_tiny_gradient()
        actual = gram_ggt(g)
        expected = g @ g.T
        self.assertEqual(actual.shape, (g.shape[0], g.shape[0]))
        torch.testing.assert_close(actual, expected)

    def test_gtg_shape_and_value(self):
        g = make_tiny_gradient()
        actual = gram_gtg(g)
        expected = g.T @ g
        self.assertEqual(actual.shape, (g.shape[1], g.shape[1]))
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
