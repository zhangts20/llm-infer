import argparse

from llm_infer.utils.model_utils import load_tokenizer
from llm_infer.models.llama import LlamaForCausalLM
from llm_infer.utils.pre_process import prepare_inputs


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id",
                        required=True,
                        help="The root directory of model.")
    parser.add_argument("--trust-remote-code",
                        action="store_true",
                        help="Whether to use self tokenizer.")
    parser.add_argument("--input-text",
                        default="What is AI?",
                        help="The single input text.")
    parser.add_argument("--input-file",
                        help="The multi input texts in a txt file.")
    parser.add_argument("--max-new-tokens",
                        default=17,
                        type=int,
                        help="The max new tokens of inference.")

    return parser.parse_args()


def main(args):
    tokenizer = load_tokenizer(args.model_id)
    model = LlamaForCausalLM.init_model(args.model_id)
    model.load_weights()

    input_texts = args.input_text
    if isinstance(input_texts, str):
        input_texts = [input_texts]
    input_ids, input_texts = prepare_inputs(tokenizer, input_texts,
                                            args.input_file)

    output_ids = model.inference(input_ids, args.max_new_tokens)

    trans_output_ids = list(map(list, zip(*output_ids)))
    for i in range(len(input_texts)):
        print("[ In_{}]: {}\n[Out_{}]: {}".format(
            i, input_texts[i], i, tokenizer.decode(trans_output_ids[i])))


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
