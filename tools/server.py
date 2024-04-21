import argparse
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, JSONResponse

from llm_infer.utils.model_utils import load_tokenizer
from llm_infer.models.llama import LlamaForCausalLM
from llm_infer.utils.pre_process import prepare_inputs

app = FastAPI()

model = None
tokenizer = None
device = "cuda"


@app.get("/health")
async def health() -> Response:
    return JSONResponse({"text": "Health!"})


@app.post("/generate")
async def generate(request: Request) -> Response:
    request_dict = await request.json()
    input_texts = request_dict.pop("input_text", "What is AI?")
    if not isinstance(input_texts, list):
        input_texts = [input_texts]
    input_ids, input_texts = prepare_inputs(tokenizer, input_texts)
    max_new_tokens = request_dict.pop("max_new_tokens", 17)

    output_ids = model.inference(input_ids, max_new_tokens)

    # Decode output_ids.
    output_texts = []
    trans_output_ids = list(map(list, zip(*output_ids)))
    for i in range(len(input_texts)):
        output_texts.append(tokenizer.decode(trans_output_ids[i]))

    # Return output texts.
    ret = {"text": output_texts}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id",
                        required=True,
                        help="The root directory of model.")
    parser.add_argument("--host",
                        default="127.0.0.1",
                        help="The host of serving.")
    parser.add_argument("--port",
                        default=8000,
                        type=int,
                        help="The port of serving.")
    args = parser.parse_args()

    # Init tokenizer and model.
    tokenizer = load_tokenizer(args.model_id)
    model = LlamaForCausalLM.init_model(args.model_id)
    model.load_weights()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
