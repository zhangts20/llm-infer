import os
import glob
import torch
import torch.nn as nn
import torch.distributed
from transformers import AutoTokenizer
from typing import List
from torch.distributed import all_reduce, ProcessGroup

__all__ = ["load_tokenizer", "initialize_torch_distributed"]


def initialize_torch_distributed():
    # Get rank and world_size using torchrun --nproc-per-node xx.py.
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl",
                                             world_size=world_size,
                                             rank=rank)

    return torch.distributed.group.WORLD, rank, world_size


def load_tokenizer(
    model_id: str,
    trust_remote_code: bool = True,
) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_weight_files(model_id: str) -> List[str]:
    return glob.glob(model_id + "/*.bin")


class FastLinear(nn.Linear):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: torch.Tensor = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is not None:
            return torch.addmm(self.bias, x, self.weight.T)
        return torch.matmul(x, self.weight.T)


class TensorParallelColLinear(FastLinear):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: ProcessGroup,
        bias: torch.Tensor = None,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        tp_size = process_group.size()
        assert out_features % tp_size == 0, \
            f"The out feature ({out_features}) couldn't divided by tp size ({tp_size})."
        out_features = out_features // tp_size
        super().__init__(in_features, out_features, bias, device, dtype)


class TensorParallelRowLinear(FastLinear):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        process_group: ProcessGroup,
        bias: torch.Tensor = None,
        device: torch.device = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.process_group = process_group
        self.tp_size = process_group.size()
        assert in_features % self.tp_size == 0, \
            f"The in feature ({in_features}) couldn't divided by tp size ({self.tp_size})."
        in_features = in_features // self.tp_size
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super(TensorParallelRowLinear, self).forward(x)
        all_reduce(out, group=self.process_group)

        return out
