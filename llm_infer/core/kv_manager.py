import math
import torch

from typing import List, Dict


class Block:

    def __init__(self, block_idx: int):
        # The index of a Block.
        self.block_idx = block_idx
        # The reference count of a Block.
        self.ref_count = 0

    def add_link(self):
        self.ref_count += 1

    def remove_link(self):
        self.ref_count -= 1

    def has_link(self) -> bool:
        return self.ref_count > 0


class BlocksManager:
    _sizeof = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        max_blocks_per_seq: int = 128,
    ) -> None:
        self.num_layers = num_layers
        self.num_blcoks = num_blocks
        self.block_size = block_size
        self.max_blocks_per_seq = max_blocks_per_seq

        # Initialize all free blocks.
        self.free_blocks: List[Block] = []
        for bi in range(num_blocks):
            self.free_blocks.append(Block(bi))

        # To store the blocks allocated.
        self.allocated_blocks: Dict[int, List[Block]] = dict()

    def get_mem_pointer(self, block_idx: int, pool: torch.Tensor,
                        elts_per_block: int) -> int:
        return pool.data_ptr(
        ) + block_idx * elts_per_block * self._sizeof[pool.dtype]

    def has_free_block(self) -> bool:
        return len(self.free_blocks) > 0

    def allocate(self, seq_idx: int) -> None:
        if not self.has_free_block():
            print("Not engough blocks to allocate.")

        block: Block = self.free_blocks.pop(0)
        block.add_link()
        if seq_idx not in self.allocated_blocks:
            self.allocated_blocks.update({seq_idx: [block]})
        else:
            self.allocated_blocks[seq_idx].append(block)

    def get_kv_offset(self, block_idx: int, field_idx: int):
        # filed_idx should be 0 (k) or 1 (v)
        return block_idx * self.num_layers * 2 + field_idx

    def get_offset_array(self) -> None:
        # (seq_idx, 2, max_blocks_per_seq)
        offset_array = torch.empty(len(self.allocated_blocks), 2,
                                   self.max_blocks_per_seq)

        k_idx = 0
        v_idx = 1
        for seq_idx, blocks in self.allocated_blocks.items():
            for block_idx, block in enumerate(blocks[seq_idx]):
                block: Block
                for x_idx in [k_idx, v_idx]:
                    offset_array[seq_idx][x_idx][
                        block_idx] = self.get_kv_offset(
                            self, block.block_idx, x_idx)
        offset_array = torch.tensor(offset_array, dtype=torch.int32)

        return offset_array


class KVCacheManager:

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        tokens_per_block: int,
        max_blocks_per_seq: int,
    ) -> None:
        self.tokens_per_block = tokens_per_block
        self.blocks_manager = BlocksManager(num_layers, num_blocks, block_size,
                                            max_blocks_per_seq)

    def add_sequence(self, seq_idx: int, context_len: int) -> None:
        context_blocks = math.ceil(context_len / self.tokens_per_block)
        for _ in range(context_blocks):
            self.blocks_manager.allocate(seq_idx)
