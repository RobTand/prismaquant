#!/usr/bin/env bash
# Historical cross-layer experiment on Qwen3-4B.
# This launcher references modules now archived or removed from the live tree.
# Retained as research context; it is not a supported quantization launcher.
#   1. Build prod cache (NVFP4 GPTQ + scale_sweep)
#   2. Re-measure four-term with prod cache active (production δw)
#   3. Build dense cone from the prod-faithful payload
#   4. Real-KL validate cone with prod cache
#   5. Polish from the new knee
#
# The historical run estimate was ~30-40 min on 4B.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
IMAGE="${IMAGE:-vllm-fresh-b12x:latest}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"

RUN_ROOT="${RUN_ROOT:-/home/rob/dq-runs/qwen3-4b-prodfaithful-${TS}}"
CNAME="${CNAME:-pq-pf-4b-$(printf '%s' "${TS}" | tr '[:upper:]' '[:lower:]')}"

PROD_FORMATS="${PROD_FORMATS:-NVFP4}"
PROD_LEVERS="${PROD_LEVERS:-gptq,scale_sweep}"
PROD_N_CALIB="${PROD_N_CALIB:-8}"
PROD_CALIB_SEQ="${PROD_CALIB_SEQ:-256}"
PROD_MAX_ACT_ROWS="${PROD_MAX_ACT_ROWS:-512}"

FORMATS="${FORMATS:-NVFP4,MXFP8_E4M3,BF16}"
N_CALIB_SAMPLES="${N_CALIB_SAMPLES:-2}"
CALIB_SEQLEN="${CALIB_SEQLEN:-128}"

POLISH_BUDGET_CREEP="${POLISH_BUDGET_CREEP:-0.05}"
POLISH_MAX_PASSES="${POLISH_MAX_PASSES:-12}"

BPPS="${BPPS:-4.50,4.55,4.60,4.65,4.70,4.75,4.80,4.85,4.90,4.95,5.00,5.10,5.20,5.30,5.40,5.50,5.75,6.00,6.50,7.00,8.00}"

mkdir -p "${RUN_ROOT}"/{logs,home,hf_modules,hf_datasets,tf_cache,cone}

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
  -e PROD_FORMATS="${PROD_FORMATS}" \
  -e PROD_LEVERS="${PROD_LEVERS}" \
  -e PROD_N_CALIB="${PROD_N_CALIB}" \
  -e PROD_CALIB_SEQ="${PROD_CALIB_SEQ}" \
  -e PROD_MAX_ACT_ROWS="${PROD_MAX_ACT_ROWS}" \
  -e FORMATS="${FORMATS}" \
  -e N_CALIB_SAMPLES="${N_CALIB_SAMPLES}" \
  -e CALIB_SEQLEN="${CALIB_SEQLEN}" \
  -e BPPS="${BPPS}" \
  -e POLISH_BUDGET_CREEP="${POLISH_BUDGET_CREEP}" \
  -e POLISH_MAX_PASSES="${POLISH_MAX_PASSES}" \
  -w /prismaquant \
  --entrypoint bash "${IMAGE}" \
  -lc '
    set -euo pipefail
    python3 -m pip install --user --quiet accelerate datasets 2>&1 \
      | tee "/work/logs/pip.log"

    echo "[1/4] build production cache (NVFP4 + GPTQ + scale_sweep)"
    python3 -m prismaquant.build_production_cache \
      --model "${MODEL_PATH}" \
      --output /work/production_cache.pkl \
      --formats "${PROD_FORMATS}" \
      --enable "${PROD_LEVERS}" \
      --n-calib-samples "${PROD_N_CALIB}" \
      --calib-seqlen "${PROD_CALIB_SEQ}" \
      --max-act-rows "${PROD_MAX_ACT_ROWS}" \
      --dtype bf16 \
      2>&1 | tee "/work/logs/build_cache.log"

    echo "[2/4] iterate (prod cache active for measurement + polish)"
    python3 -m prismaquant.iterate_block_clado \
      --model "${MODEL_PATH}" \
      --output-root /work \
      --max-iterations 1 \
      --formats "${FORMATS}" \
      --n-calib-samples "${N_CALIB_SAMPLES}" \
      --calib-seqlen "${CALIB_SEQLEN}" \
      --calib-split train \
      --calib-seed 42 \
      --dtype bf16 \
      --device cuda \
      --polish-max-passes "${POLISH_MAX_PASSES}" \
      --polish-noise-floor 1e-5 \
      --polish-budget-creep "${POLISH_BUDGET_CREEP}" \
      --polish-steepest-first \
      --use-frozen-weight-cache \
      --measure-method four_term \
      --n-neighbors-validate 4 \
      --production-weight-cache /work/production_cache.pkl \
      2>&1 | tee "/work/logs/iterate.log"

    echo "[3/4] dense exact-budget cone from new prod-faithful payload"
    python3 -m prismaquant.dense_cone \
      --payload /work/iter_0/block_clado.json \
      --output-dir /work/cone \
      --bpps "${BPPS}" \
      2>&1 | tee "/work/logs/dense_cone.log"

    echo "[4/4] real-KL validate dense cone WITH prod cache"
    python3 -m prismaquant.validate_block_clado \
      --model "${MODEL_PATH}" \
      --kneedle-dir /work/cone \
      --output /work/dense_validation_prod.json \
      --n-calib-samples "${N_CALIB_SAMPLES}" \
      --calib-seqlen "${CALIB_SEQLEN}" \
      --calib-split train \
      --calib-seed 42 \
      --dtype bf16 \
      --production-weight-cache /work/production_cache.pkl \
      2>&1 | tee "/work/logs/validate_prod.log"

    echo
    echo "[done] iterate best, then dense cone Pareto:"
    python3 - <<PYEOF
import json
s = json.load(open("/work/summary.json"))
b = s["best_overall"]
print(f"  iterate best: bpp={b[\"best_validated_bpp\"]:.4f} polished_kl={b[\"polished_kl\"]:.6f}")

v = json.load(open("/work/dense_validation_prod.json"))["results"]
v = sorted(v, key=lambda r: r["bpp"])
best_kl = float("inf")
print(f"  cone Pareto (bpp asc, KL strictly decreasing):")
for r in v:
    if r["real_kl"] < best_kl - 1e-9:
        best_kl = r["real_kl"]
        print(f"    {r[\"bpp\"]:.4f}  KL={r[\"real_kl\"]:.4f}  ({r[\"label\"]})")
PYEOF
  '

echo "[launch] container: ${CNAME}"
echo "[launch] workdir:   ${RUN_ROOT}"
echo "[launch] tail:      docker logs -f ${CNAME}"
