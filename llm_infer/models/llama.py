import math
import torch
import torch.nn.functional as F
import torch.distributed
from tqdm import tqdm
from accelerate import init_empty_weights
from typing import List, Tuple, Dict

from llm_infer.models.base import Model
from llm_infer.utils.model_utils import (
    get_weight_files,
    FastLinear,
    TensorParallelColLinear,
    TensorParallelRowLinear,
)
from llm_infer.utils.time import calculate_time


class LLamaRotaryEmbedding(torch.nn.Module):

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        dtype: torch.dtype = torch.float16,
        device: torch.device = "cuda",
    ) -> None:
        super(LLamaRotaryEmbedding, self).__init__()
        # theta
        self.inv_freq = \
            1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.dtype = dtype
        self.device = device

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        position_ids: torch.Tensor,
        seq_len: int = None,
    ) -> torch.Tensor:
        # [0, 1, ..., seq_len-1]
        t = torch.arange(seq_len,
                         device=self.device,
                         dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # https://zhuanlan.zhihu.com/p/647109286
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos().to(self.dtype)[:seq_len]
        sin_cached = emb.sin().to(self.dtype)[:seq_len]

        # apply
        cos_cached = cos_cached.squeeze(1).squeeze(0)
        sin_cached = sin_cached.squeeze(1).squeeze(0)
        cos_cached = cos_cached[position_ids].unsqueeze(1)
        sin_cached = sin_cached[position_ids].unsqueeze(1)

        q_embed = (Q * cos_cached) + (self._rotate_half(Q) * sin_cached)
        k_embed = (K * cos_cached) + (self._rotate_half(K) * sin_cached)

        return q_embed, k_embed

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]

        return torch.cat((-x2, x1), dim=-1)


class LlamaMLP(torch.nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        process_group: torch.distributed.ProcessGroup,
    ) -> None:
        super(LlamaMLP, self).__init__()
        if process_group is None:
            self.gate_proj = TensorParallelColLinear(hidden_size,
                                                     intermediate_size,
                                                     process_group)
            self.up_proj = TensorParallelColLinear(hidden_size,
                                                   intermediate_size,
                                                   process_group)
            self.down_proj = TensorParallelRowLinear(intermediate_size,
                                                     hidden_size,
                                                     process_group)
        else:
            self.gate_proj = FastLinear(hidden_size, intermediate_size)
            self.up_proj = FastLinear(hidden_size, intermediate_size)
            self.down_proj = FastLinear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaAttention(torch.nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        process_group: torch.distributed.ProcessGroup,
    ) -> None:
        super(LlamaAttention, self).__init__()
        self.process_group = process_group
        if process_group is None:
            self.tp_size = 1
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads
            self.head_dim = hidden_size // num_attention_heads
            self.q_proj = FastLinear(hidden_size, hidden_size)
            self.k_proj = FastLinear(hidden_size, hidden_size)
            self.v_proj = FastLinear(hidden_size, hidden_size)
            self.o_proj = FastLinear(hidden_size, hidden_size)
            self.rotary_emb = LLamaRotaryEmbedding(self.head_dim)
        else:
            self.tp_size = process_group.size()
            self.hidden_size = hidden_size
            self.num_attention_heads = num_attention_heads // self.tp_size
            self.head_dim = hidden_size // num_attention_heads
            self.q_proj = TensorParallelColLinear(hidden_size, hidden_size,
                                                  process_group)
            self.k_proj = TensorParallelColLinear(hidden_size, hidden_size,
                                                  process_group)
            self.v_proj = TensorParallelColLinear(hidden_size, hidden_size,
                                                  process_group)
            self.o_proj = TensorParallelRowLinear(hidden_size, hidden_size,
                                                  process_group)
            self.rotary_emb = LLamaRotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: Tuple[torch.Tensor],
    ) -> torch.Tensor:
        Q: torch.Tensor = self.q_proj(x)
        K: torch.Tensor = self.k_proj(x)
        V: torch.Tensor = self.v_proj(x)

        # Multi-Head Attention.
        batch_size, seq_len, _ = x.shape
        # (batch_size, seq_len, hidden_size) to (batch_size, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_attention_heads,
                   self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_attention_heads,
                   self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_attention_heads,
                   self.head_dim).transpose(1, 2)

        # Apply rotary position embedding.
        kv_seq_len = K.shape[-2]
        if past_key_values is not None:
            kv_seq_len += past_key_values[0].shape[-2]
        Q, K = self.rotary_emb(Q, K, position_ids, seq_len=kv_seq_len)

        # Update KV Cache.
        if past_key_values is not None:
            K = torch.cat([past_key_values[0], K], dim=2)
            V = torch.cat([past_key_values[1], V], dim=2)
        past_key_values = (K, V)

        # (Q * K) / sqrt
        attn_weights = torch.matmul(Q, K.transpose(2, 3)) / \
            math.sqrt(self.head_dim)

        # Apply attention mask.
        attn_weights += attention_mask.to(attn_weights.device)

        # Upcast to float32 to keep accuracy.
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_weights = attn_weights.to(Q.dtype)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len,
                                          self.hidden_size // self.tp_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, past_key_values


class LlamaRMSNorm(torch.nn.Module):

    def __init__(self, hidden_size: int, rms_norm_eps: float) -> None:
        super(LlamaRMSNorm, self).__init__()
        self.rms_norm_eps = rms_norm_eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


class LlamaDecoderLayer(torch.nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        rms_norm_eps: float,
        process_group: torch.distributed.ProcessGroup,
    ) -> None:
        super(LlamaDecoderLayer, self).__init__()
        self.input_layernorm = LlamaRMSNorm(hidden_size=hidden_size,
                                            rms_norm_eps=rms_norm_eps)
        self.self_attn = LlamaAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            process_group=process_group)
        self.post_attention_layernorm = LlamaRMSNorm(hidden_size=hidden_size,
                                                     rms_norm_eps=rms_norm_eps)
        self.mlp = LlamaMLP(hidden_size=hidden_size,
                            intermediate_size=intermediate_size,
                            process_group=process_group)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values: List[torch.Tensor],
    ) -> torch.Tensor:
        residual = x

        hidden_states = self.input_layernorm(x)
        hidden_states, present_key_value = self.self_attn(
            x=hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values)
        hidden_states += residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)
        hidden_states += residual

        return hidden_states, present_key_value


class LlamaModel(torch.nn.Module):

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        intermediate_size: int,
        num_attention_heads: int,
        rms_norm_eps: float,
        process_group: torch.distributed.ProcessGroup = None,
    ) -> None:
        super(LlamaModel, self).__init__()
        self.embed_tokens = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList()
        self.num_hidden_layers = num_hidden_layers
        for _ in range(self.num_hidden_layers):
            self.layers.append(
                LlamaDecoderLayer(hidden_size=hidden_size,
                                  intermediate_size=intermediate_size,
                                  num_attention_heads=num_attention_heads,
                                  rms_norm_eps=rms_norm_eps,
                                  process_group=process_group))
        self.norm = LlamaRMSNorm(hidden_size=hidden_size,
                                 rms_norm_eps=rms_norm_eps)

    @classmethod
    def _generate_square_subsequent_mask(self, size: int) -> torch.Tensor:
        # Make an upper triangular matrix with elements being 1 (including the diagonal).
        mask = torch.triu(torch.ones(size, size))
        # Transpose to a lower triangular.
        mask = mask.transpose(0, 1)
        # Set the positions that are zero to the maximum value.
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(
            mask == 1, float(0.0))

        return mask

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: torch.Tensor,
    ) -> torch.Tensor:
        # The length of KV Cache.
        past_key_value_length = 0
        if past_key_values is not None:
            past_key_value_length = past_key_values[0][0].shape[2]

        # Prepare position_ids.
        seq_length = input_ids.size(1)
        position_ids = torch.arange(past_key_value_length,
                                    seq_length + past_key_value_length,
                                    dtype=torch.long,
                                    device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        # (batch_size, seq_len, hidden_size)
        hidden_states = self.embed_tokens(input_ids)

        # Prepare attention_mask.
        attention_mask = self._generate_square_subsequent_mask(seq_length)

        # To store KV Cache, (num_layers, 2, batch_size, num_heads, seq_length, hidden_size / num_heads)
        present_key_values = []
        for i in range(self.num_hidden_layers):
            past_key_value = past_key_values[i] \
                if past_key_values is not None else None
            hidden_states, present_key_value = self.layers[i](hidden_states,
                                                              position_ids,
                                                              attention_mask,
                                                              past_key_value)
            present_key_values.append(present_key_value)

        hidden_states = self.norm(hidden_states)

        return hidden_states, present_key_values


class LlamaForCausalLM(Model):

    def __init__(self, model_id: str, dtype: torch.dtype) -> None:
        super(LlamaForCausalLM, self).__init__(model_id, dtype)
        self.hidden_size = self.model_cfg["hidden_size"]
        self.vocab_size = self.model_cfg["vocab_size"]
        self.num_hidden_layers = self.model_cfg["num_hidden_layers"]
        self.intermediate_size = self.model_cfg["intermediate_size"]
        self.num_attention_heads = self.model_cfg["num_attention_heads"]
        self.rms_norm_eps = self.model_cfg["rms_norm_eps"]
        self.model = LlamaModel(vocab_size=self.vocab_size,
                                hidden_size=self.hidden_size,
                                num_hidden_layers=self.num_hidden_layers,
                                intermediate_size=self.intermediate_size,
                                num_attention_heads=self.num_attention_heads,
                                rms_norm_eps=self.rms_norm_eps,
                                process_group=self.process_group)
        self.lm_head = FastLinear(self.hidden_size, self.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        past_key_values: List[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states, present_key_values = self.model(input_ids,
                                                       past_key_values)
        logits = self.lm_head(hidden_states).to(torch.float32)

        return logits, present_key_values

    @classmethod
    @calculate_time
    def init_model(self, model_id):
        with init_empty_weights():
            llama = LlamaForCausalLM(model_id, dtype=torch.float16).eval()

        return llama

    @calculate_time
    def load_weights(self):
        weight_files = get_weight_files(self.model_id)

        def load_file(file_name, model):
            dict_tensor: Dict[str, torch.Tensor] = torch.load(file_name)
            for k, v in dict_tensor.items():
                # Get extension.
                layer_name, extension = k.rsplit('.', maxsplit=1)
                # Get module.
                if 'lm_head' == layer_name:
                    module_name = layer_name
                else:
                    module_name = layer_name.split('.', maxsplit=1)[1]
                module = model.get_submodule(module_name)
                # Set value.
                v = v.to("cuda").to(self.dtype)
                module._parameters[extension] = v

        def load_file_tp(file_name, model):
            dict_tensor: Dict[str, torch.Tensor] = torch.load(file_name)
            for k, v in dict_tensor.items():
                # Get extension.
                layer_name, extension = k.rsplit('.', maxsplit=1)
                # Get module.
                if 'lm_head' == layer_name:
                    module_name = layer_name
                    module = model.get_submodule(module_name)
                else:
                    module_name = layer_name.split('.', maxsplit=1)[1]
                    module = model.model.get_submodule(module_name)
                if isinstance(module, TensorParallelColLinear):
                    size = v.shape[0]
                    block_size = size // self.world_size
                    start = self.rank * block_size
                    stop = (self.rank + 1) * block_size
                    v = v[start:stop]
                elif isinstance(module, TensorParallelRowLinear):
                    size = v.shape[1]
                    block_size = size // self.world_size
                    start = self.rank * block_size
                    stop = (self.rank + 1) * block_size
                    v = v[:, start:stop]
                else:
                    pass
                # Set value.
                v = v.contiguous().to(self.dtype).to(f"cuda:{self.rank}")
                module._parameters[extension] = v

        for i in tqdm(range(len(weight_files))):
            if self.process_group is not None:
                load_file_tp(weight_files[i], self)
            else:
                load_file(weight_files[i], self)
