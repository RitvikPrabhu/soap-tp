import os
import unittest

import torch
import torch.distributed as dist

from soap_tp.ops.gram import distributed_ggt_from_col_shards, distributed_gtg_from_row_shards
from soap_tp.reference.sequential import gram_ggt, gram_gtg
from soap_tp.utils.fixtures import make_tiny_gradient
from soap_tp.utils.sharding import shard_cols, shard_rows


@unittest.skip("Enable after distributed Gram stubs are implemented")
class TestDistributedGramContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not dist.is_available():
            raise unittest.SkipTest("torch.distributed is unavailable")
        if not dist.is_initialized():
            dist.init_process_group(backend=os.environ.get("SOAP_TP_BACKEND", "gloo"))
        cls.rank = dist.get_rank()
        cls.world_size = dist.get_world_size()

    def test_gtg_from_row_shards_matches_sequential(self):
        g = make_tiny_gradient()
        local = shard_rows(g, self.world_size)[self.rank].contiguous()
        actual = distributed_gtg_from_row_shards(local)
        expected = gram_gtg(g)
        torch.testing.assert_close(actual, expected)

    def test_ggt_from_col_shards_matches_sequential(self):
        g = make_tiny_gradient()
        local = shard_cols(g, self.world_size)[self.rank].contiguous()
        actual = distributed_ggt_from_col_shards(local)
        expected = gram_ggt(g)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
