#!/usr/bin/env bash
# W7: GLM-5.3-Flash-4layer BF16 TP1 on the patched NoPE sparse-MLA image.
set -uo pipefail
ARM="$1"            # a | b
IMG="glm53-nope-sm121:w7"
NAME="w7-glm-${ARM}"
docker rm -f "$NAME" >/dev/null 2>&1
ARGS=(vllm serve /mnt/shared/models/GLM-5.3-Flash-4layer
  --enforce-eager --gpu-memory-utilization 0.55 --host 0.0.0.0 --port 8000
  --max-model-len 4096 --max-num-seqs 8 --max-logprobs 1024 --trust-remote-code)
if [ "$ARM" = "a" ]; then
  ARGS+=(--hf-overrides '{"index_topk": null}')
fi
exec docker run --rm --name "$NAME" --gpus all --ipc=host --network=host \
  -e HF_HUB_OFFLINE=1 -e VLLM_LOGGING_LEVEL=INFO \
  -v /mnt/shared:/mnt/shared \
  "$IMG" "${ARGS[@]}"
