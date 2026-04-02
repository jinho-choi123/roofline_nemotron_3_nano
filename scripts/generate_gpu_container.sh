#!/usr/bin/env bash

set -euo pipefail

GPU_ID="${1:-0}"
IMAGE_NAME="${IMAGE_NAME:-hybrid-model-benchmark:cuda12.4}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/nsys-reps" "${HOME}/.cache/huggingface"

docker run --rm -it \
  --gpus "device=${GPU_ID}" \
  --cap-add=SYS_ADMIN \
  -e NVIDIA_VISIBLE_DEVICES="${GPU_ID}" \
  -v "${PROJECT_ROOT}:/workspace" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace \
  "${IMAGE_NAME}" \
  bash