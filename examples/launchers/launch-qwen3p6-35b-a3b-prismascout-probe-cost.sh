#!/usr/bin/env bash
# Probe + cost seed stage for a BF16-source Qwen3.6-35B-A3B PrismaSCOUT run.
# This intentionally stops before allocation/export so the measured cost
# artifact can feed the kneedle/L3 PrismaSCOUT search.
#
# Visual weights are quantized in the downstream allocation/export phase via
# the allocator's source-discovery override. Match the 27B PrismaSCOUT artifact:
# stamp visual Linears uniformly as NVFP4, while the text-only probe/cost below
# measures body + MTP + lm_head.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B/snapshots/53c43178507d69762986fbfa314f6e8d4d859409}"
IMAGE="${IMAGE:-vllm-fresh-b12x:latest}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_ROOT="${RUN_ROOT:-/home/rob/dq-runs/qwen3p6-35b-a3b-prismascout-bf16-${TS}}"
CNAME="${CNAME:-pq-qwen36-35b-a3b-pqscout-$(printf '%s' "${TS}" | tr '[:upper:]' '[:lower:]')}"

NSAMPLES="${NSAMPLES:-32}"
SEQLEN="${SEQLEN:-1024}"
DATASET="${DATASET:-ultrachat_200k}"
FORMATS="${FORMATS:-NVFP4,MXFP8_E4M3,BF16}"
VISUAL_FORMAT="${VISUAL_FORMAT:-NVFP4}"
VISUAL_SENSITIVITY="${VISUAL_SENSITIVITY:-uniform}"
LAYERS_PER_SHARD="${LAYERS_PER_SHARD:-auto}"
PREFETCH_LOOKAHEAD="${PREFETCH_LOOKAHEAD:-auto}"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-auto}"
PREFETCH_MIN_AVAILABLE_GB="${PREFETCH_MIN_AVAILABLE_GB:-auto}"
ACTIVATION_ROWS_LIMIT="${ACTIVATION_ROWS_LIMIT:-256}"

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[launch] missing MODEL_PATH: ${MODEL_PATH}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}"/{artifacts,act,work,logs,home,hf_modules,hf_datasets,tf_cache}

docker run -d --gpus all --ipc=host --shm-size=8g \
  --user "$(id -u):$(id -g)" \
  --name "${CNAME}" \
  -v /home/rob/prismaquant:/prismaquant \
  -v /home/rob/.cache/huggingface:/home/rob/.cache/huggingface:ro \
  -v "${RUN_ROOT}":/work \
  -e HOME=/work/home \
  -e HF_MODULES_CACHE=/work/hf_modules \
  -e HF_DATASETS_CACHE=/work/hf_datasets \
  -e TRANSFORMERS_CACHE=/work/tf_cache \
  -e PYTHONPATH=/prismaquant \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e MODEL_PATH="${MODEL_PATH}" \
  -e NSAMPLES="${NSAMPLES}" \
  -e SEQLEN="${SEQLEN}" \
  -e DATASET="${DATASET}" \
  -e FORMATS="${FORMATS}" \
  -e VISUAL_FORMAT="${VISUAL_FORMAT}" \
  -e VISUAL_SENSITIVITY="${VISUAL_SENSITIVITY}" \
  -e LAYERS_PER_SHARD="${LAYERS_PER_SHARD}" \
  -e PREFETCH_LOOKAHEAD="${PREFETCH_LOOKAHEAD}" \
  -e PREFETCH_WORKERS="${PREFETCH_WORKERS}" \
  -e PREFETCH_MIN_AVAILABLE_GB="${PREFETCH_MIN_AVAILABLE_GB}" \
  -e ACTIVATION_ROWS_LIMIT="${ACTIVATION_ROWS_LIMIT}" \
  -w /prismaquant \
  --entrypoint bash "${IMAGE}" \
  -lc '
    set -euo pipefail
    PROBE=/work/artifacts/probe.pkl
    COST=/work/artifacts/cost.pkl

    echo "[launch] model=${MODEL_PATH}"
    echo "[launch] work=/work"
    echo "[launch] formats=${FORMATS}"
    echo "[launch] downstream visual=${VISUAL_FORMAT} sensitivity=${VISUAL_SENSITIVITY}"
    echo "[launch] nsamples=${NSAMPLES} seqlen=${SEQLEN}"
    echo "[launch] probe=${PROBE}"
    echo "[launch] cost=${COST}"

    python3 -m pip install --user --quiet accelerate datasets \
      2>&1 | tee /work/logs/pip_install.log

    python3 - <<PY
from pathlib import Path
from safetensors import safe_open
p = Path("${MODEL_PATH}")
with safe_open(sorted(p.glob("*.safetensors"))[0], framework="pt", device="cpu") as sf:
    for name in sf.keys():
        tensor = sf.get_tensor(name)
        print(f"[launch] source dtype check: {name} {tensor.dtype} {tuple(tensor.shape)}")
        break
PY

    if [[ ! -f "${PROBE}" ]]; then
      echo "[1/2] probe ..."
      python3 -m prismaquant.incremental_probe \
        --model "${MODEL_PATH}" \
        --dataset "${DATASET}" \
        --nsamples "${NSAMPLES}" --seqlen "${SEQLEN}" \
        --device cuda --dtype bf16 \
        --output "${PROBE}" \
        --activation-cache-dir /work/act \
        --work-dir /work/work \
        --layers-per-shard "${LAYERS_PER_SHARD}" \
        --prefetch-lookahead "${PREFETCH_LOOKAHEAD}" \
        --prefetch-workers "${PREFETCH_WORKERS}" \
        --prefetch-min-available-gb "${PREFETCH_MIN_AVAILABLE_GB}" \
        --activation-rows-limit "${ACTIVATION_ROWS_LIMIT}" \
        --calibration-modality text-only \
        --include-mtp \
        --no-include-visual \
        --include-lm-head \
        2>&1 | tee /work/logs/probe.log
    else
      echo "[1/2] probe.pkl exists, skipping"
    fi

    if [[ ! -f "${COST}" ]]; then
      echo "[2/2] cost ..."
      python3 -m prismaquant.incremental_measure_quant_cost \
        --model "${MODEL_PATH}" \
        --probe "${PROBE}" \
        --activation-cache-dir /work/act \
        --formats "${FORMATS}" \
        --output "${COST}" \
        --work-dir /work/work \
        --device cuda --dtype bf16 \
        --mode batched --chunk-size 256 \
        --layers-per-shard "${LAYERS_PER_SHARD}" \
        --skip-missing-activations \
        --include-mtp \
        --no-include-visual \
        --include-lm-head \
        2>&1 | tee /work/logs/cost.log
    else
      echo "[2/2] cost.pkl exists, skipping"
    fi

    echo "[done] probe=${PROBE} cost=${COST}"
  '

echo "[launch] container: ${CNAME}"
echo "[launch] workdir:   ${RUN_ROOT}"
echo "[launch] logs:      docker logs -f ${CNAME}"
echo "[launch] probe:     tail -f ${RUN_ROOT}/logs/probe.log"
echo "[launch] cost:      tail -f ${RUN_ROOT}/logs/cost.log"
