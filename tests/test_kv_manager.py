import torch
import unittest

from llm_infer.core import BlocksManager, KVCacheManager


class TestKVManager(unittest.TestCase):

    def test_blocks_manager(self):
        max_seq = 32
        max_blocks_per_seq = 32
        blocks_elts = 64
        sequences = [i for i in range(max_seq)]

        manager = BlocksManager(31, )

    def test_kv_manager(self):
        blocks = 200
        tokens_per_block = 32
        max_blocks_per_seq = 16
        dims_per_head1 = 64
        dims_per_head2 = 128
        memory_pool_1 = torch.zeros(2,
                                    blocks,
                                    tokens_per_block,
                                    dims_per_head1,
                                    dtype=torch.float16,
                                    device="cuda")
        memory_pool_2 = torch.zeros(2,
                                    blocks,
                                    tokens_per_block,
                                    dims_per_head2,
                                    dtype=torch.float16,
                                    device="cuda")
        manager = KVCacheManager(memory_pools=[memory_pool_1, memory_pool_2],
                                 blocks=blocks,
                                 tokens_per_block=tokens_per_block,
                                 max_blocks_per_seq=max_blocks_per_seq)
        manager.add_sequence(seq_idx=0, context_len=30)
        manager.add_sequence(seq_idx=1, context_len=10)
        manager.add_sequence(seq_idx=2, context_len=129)

        allocated_blocks = manager.blocks_manager.allocated_blocks
        self.assertEqual(len(allocated_blocks), 3)
        self.assertEqual(len(allocated_blocks[0]), 1)
        self.assertEqual(len(allocated_blocks[1]), 1)
        self.assertEqual(len(allocated_blocks[2]), 5)
        

if __name__ == "__main__":
    unittest.main()
