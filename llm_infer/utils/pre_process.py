import os
import torch


def prepare_inputs(tokenizer, input_texts, input_file=None) -> torch.Tensor:
    if input_file is not None:
        assert os.path.exists(input_file), f"{input_file} doesn't exist."
        with open(input_file, "r") as f:
            input_texts = [line.strip() for line in f.readlines()]
    else:
        assert isinstance(input_texts, list), "Input text must be a list."

    inputs = tokenizer(input_texts, padding=True, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")

    return input_ids, input_texts