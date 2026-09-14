#!/usr/bin/env bash
# The serve command AS RUN on 2026-09-13 for the GLM-5.3-Flash 4-layer stub.
# This is a record of one measurement, not a sanctioned default. Three things
# it is NOT, so nobody runs it as one:
#   * IMG resolves by TAG, on the one box that holds it. The image identity is
#     MANIFEST.json's derived_image_local_id; see the manifest, not this line.
#   * --gpu-memory-utilization 0.55 is what the stub needed (its weights alone
#     are 45.35 GiB). It is above the 0.30 ceiling an agent serve may use, and
#     the full 45-layer body has no sizing rule at all yet.
#   * The model path and --enforce-eager are the stub's, and the stub emits
#     gibberish by construction: 4 of 45 layers.
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
