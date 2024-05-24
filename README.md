# llm-infer
A tool for inference of LLM.

# TODO List.
- Paged KV Cache

# Getting Started.
Run offline inference using one GPU.
```shell
torchrun --nproc-per-node 1 tools/offline_inference.py \
    --model-id /data/models/llama2-7b \
    --input-text "What is AI?" \
    --input-file "./data.txt" \
    --max-new-tokens 17
```
Run offline inference using multi GPUs.
```shell
torchrun --nproc-per-node 2 tools/offline_inference.py \
    --model-id /data/models/llama2-7b \
    --input-text "What is AI?" \
    --input-file "./data.txt" \
    --max-new-tokens 17
```
Run a serving.
```shell
# Start a server.
torchrun --nproc-per-node 1 \
    tools/server.py \
    --model-id /data/models/llama2-7b

# Client a server.
curl 127.0.0.1:8000/generate \
    -X POST \
    -d '{"input_text":"What is your name?","parameters":{"max_new_tokens":17}}' \
    -H 'Content-Type: application/json'
```