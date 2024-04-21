import os
import abc
import json
import torch

from llm_infer.utils.model_utils import initialize_torch_distributed
from llm_infer.utils.time import calculate_time


class Model(torch.nn.Module, metaclass=abc.ABCMeta):

    def __init__(self, model_id: str, dtype=torch.float16) -> None:
        super(Model, self).__init__()
        self.model_id = model_id
        self.process_group, self.rank, self.world_size = \
            initialize_torch_distributed()
        self.dtype = dtype
        self.model_cfg = self.parse_config()

    def parse_config(self):
        config_path = os.path.join(self.model_id, "config.json")
        assert os.path.exists(
            config_path), f"{self.model_id} does not contain a config.json."
        with open(config_path, "r") as f:
            model_cfg = json.load(f)

        return model_cfg

    @abc.abstractmethod
    def init_model(self, *args, **kwargs):
        raise NotImplementedError("init_model must have an implementation.")

    @abc.abstractmethod
    def load_weights(self, *args, **kwargs):
        raise NotImplementedError("load_weights must have an implementation.")

    @abc.abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("forward must have an implementation.")

    @calculate_time
    def inference(self, input_ids, max_new_tokens) -> list:
        past_key_values = None
        output_ids = list()
        for _ in range(max_new_tokens):
            inner_list = []
            with torch.no_grad():
                logits, past_key_values = self.forward(
                    input_ids=input_ids, past_key_values=past_key_values)

            # The next input of inference.
            input_ids = torch.argmax(logits, dim=-1)[:, -1].unsqueeze(1)

            # Append output.
            for j in range(input_ids.size(0)):
                inner_list.append(input_ids[j][0].cpu().item())
            output_ids.append(inner_list)

        return output_ids
