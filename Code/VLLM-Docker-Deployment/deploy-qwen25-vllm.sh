#!/bin/bash

# =========================
# User-configurable settings
# =========================

CONTAINER_NAME="vllm-qwen25-72b-instruct"

# Port used to access the vLLM server from the host machine.
HOST_PORT="8000"

# Path to the Hugging Face cache directory on the host machine.
# Change this to the directory where your downloaded models are stored.
HOST_MODEL_CACHE="/path/to/your/huggingface/cache"

# Model to load.
#
# Option 1: Hugging Face model ID
MODEL="Qwen/Qwen2.5-72B-Instruct"
#
# Option 2: Local model path inside the Docker container
# MODEL="/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/<snapshot-id>"

# Model name exposed through the OpenAI-compatible API.
# Client code must use this value in the `model` field.
SERVED_MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"


docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus '"device=0,1,2,3"' \
  --restart unless-stopped \
  -p "${HOST_PORT}:8000" \
  -v "${HOST_MODEL_CACHE}:/root/.cache/huggingface" \
  --ipc=host \
  -e VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
  vllm/vllm-openai:latest \
  --model "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --seed 42 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --gpu-memory-utilization 0.9 \
  --max-num-batched-tokens 8192