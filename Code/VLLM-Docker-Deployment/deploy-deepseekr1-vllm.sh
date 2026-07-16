#!/bin/bash

# =========================
# User-configurable settings
# =========================

CONTAINER_NAME="vllm-deepseek-r1-70b"

# Port used to access the vLLM server from the host machine.
HOST_PORT="8000"

# Path to the Hugging Face cache directory on the host machine.
# Change this to the directory where your downloaded models are stored.
HOST_MODEL_CACHE="/path/to/your/huggingface/cache"

# Model to load.
#
# Option 1: Hugging Face model ID
MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
#
# Option 2: Local model path inside the Docker container
# MODEL="/root/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/snapshots/<snapshot-id>"

# Model name exposed through the OpenAI-compatible API.
# Client code must use this value in the `model` field.
SERVED_MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-70B"


docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus '"device=0,1,2,3"' \
  --restart unless-stopped \
  -p "${HOST_PORT}:8000" \
  -v "${HOST_MODEL_CACHE}:/root/.cache/huggingface" \
  --ipc=host \
  -e HF_HOME=/root/.cache/huggingface \
  -e VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
  vllm/vllm-openai:v0.10.0 \
  --model "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9 \
  --tokenizer-mode auto \
  --reasoning-parser deepseek_r1 \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json