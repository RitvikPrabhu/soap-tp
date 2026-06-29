import os
import unittest

import torch.distributed as dist

from soap_tp.steps.dispatch import matrix_soap_tp_step
from soap_tp.steps.state import MatrixSOAPState
from soap_tp.utils.fixtures import make_tiny_gradient
from soap_tp.utils.sharding import shard_cols, shard_rows


@unittest.skip("Enable after row_tp_soap_step and col_tp_soap_step are implemented")
class TestOneSOAPStepContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not dist.is_available():
            raise unittest.SkipTest("torch.distributed is unavailable")
        if not dist.is_initialized():
            dist.init_process_group(backend=os.environ.get("SOAP_TP_BACKEND", "gloo"))
        cls.rank = dist.get_rank()
        cls.world_size = dist.get_world_size()

    def test_row_sharded_one_step_matches_single_device_reference(self):
        g = make_tiny_gradient()
        p = g.clone() * 0.0
        grad_local = shard_rows(g, self.world_size)[self.rank].contiguous()
        param_local = shard_rows(p, self.world_size)[self.rank].contiguous()
        out_local, state = matrix_soap_tp_step(
            param_local,
            grad_local,
            MatrixSOAPState(),
            lr=1e-3,
            shard_axis="row",
        )
        # TODO: gather out_local and compare against sequential one-step reference.
        self.assertIsNotNone(out_local)
        self.assertIsInstance(state, MatrixSOAPState)

    def test_col_sharded_one_step_matches_single_device_reference(self):
        g = make_tiny_gradient()
        p = g.clone() * 0.0
        grad_local = shard_cols(g, self.world_size)[self.rank].contiguous()
        param_local = shard_cols(p, self.world_size)[self.rank].contiguous()
        out_local, state = matrix_soap_tp_step(
            param_local,
            grad_local,
            MatrixSOAPState(),
            lr=1e-3,
            shard_axis="col",
        )
        # TODO: gather out_local and compare against sequential one-step reference.
        self.assertIsNotNone(out_local)
        self.assertIsInstance(state, MatrixSOAPState)


if __name__ == "__main__":
    unittest.main()
