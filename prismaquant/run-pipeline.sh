#!/usr/bin/env bash
# run-pipeline.sh — end-to-end PrismaQuant pipeline: probe → cost →
# allocator → native compressed-tensors export → vLLM validate.
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3.6-35B-A3B \
#   WORK_DIR=./dq-runs/qwen36 \
#   FORMATS=NVFP4,MXFP8_E4M3,BF16 \
#   TARGET_BITS=4.75 \
#   VISUAL_FORMAT=BF16 \
#   CALIBRATION_MODALITY=text-only \
#   ./prismaquant/run-pipeline.sh
#
# VISUAL_FORMAT (BF16 | NVFP4 | MXFP8) applies to visual-encoder Linears
# on multimodal models. In text-only calibration mode it's the Phase 1
# uniform override for every visual Linear. In multimodal mode it's the
# fallback applied to un-probed visual Linears the allocator's Fisher-
# driven DP didn't touch (plus the graceful OOM fallback on 122B-scale
# VLMs that can't fit the whole model in RAM for the multimodal pass).
#
# CALIBRATION_MODALITY (text-only | multimodal):
#   - text-only (default): body-only streaming probe + cost. Visual
#     shards emit empty pickles; allocator stamps all visual Linears
#     with --visual-format uniformly.
#   - multimodal: also runs a non-streaming second probe pass with
#     image+text calibration (synthetic stub by default; set MM_DATASET
#     to a HuggingFace dataset id to use real images). The allocator
#     treats visual Linears as regular DP candidates when real Fisher
#     stats are present. Requires ~full-model RAM; falls back to
#     text-only behavior automatically on OOM.
#
# Memory note: probe + cost peak around 90 GB on a 35B model under
# BF16 calibration. The watchdog in incremental_measure_quant_cost
# aborts cleanly on swap pressure rather than OOM-killing the host.
#
# MTP is folded into the incremental probe + cost as a built-in shard;
# mtp.* tensors are measured in the same pass as the body and land in
# the same probe/cost pickles. No separate MTP stages.

set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the source HF model directory}"
: "${WORK_DIR:?Set WORK_DIR to a writable directory for artifacts}"
: "${FORMATS:=NVFP4,MXFP8_E4M3,BF16}"
: "${TARGET_BITS:=4.75}"
: "${PARETO_TARGETS:=4.5,4.6,4.7,4.75,4.85,5.0,5.25,5.5,6.0,7.0,8.25}"
# Calibration defaults. 4x256 was the historical minimum for correctness
# validation; 32x1024 (N=32, T=1024 = 32768 tokens/sample, 32 samples)
# produces ~7% lower PPL on the resulting quantized artifact at a
# linear time cost in probe wall-time. Override for faster iteration.
: "${NSAMPLES:=32}"
: "${SEQLEN:=1024}"
# LAYERS_PER_SHARD controls how many decoder layers share one reverse
# sweep for Fisher accumulation. Larger = fewer sweeps = faster probe,
# but each shard needs more gradient + retained-activation memory.
# Default `auto` asks prismaquant.autoscale to pick from available RAM +
# model size. Set to an int (e.g. `2`) to override.
: "${LAYERS_PER_SHARD:=auto}"
# CACHE_HEADROOM_GB controls the streaming layer-cache budget
# (cache = free_RAM - headroom). Default `auto` picks from model +
# LAYERS_PER_SHARD. Set to a float to override.
: "${CACHE_HEADROOM_GB:=auto}"
: "${DATASET:=ultrachat_200k}"
: "${DEVICE:=cuda}"
: "${EXPORT_DEVICE:=cuda}"   # CUDA ~10× faster than CPU on NVFP4 packing
: "${TARGET_PROFILE:=vllm_qwen3_5_packed_moe}"
# Visual encoder format: fallback for visual Linears. See header docstring
# for the full text-only vs multimodal semantics. BF16 (default) is
# passthrough; NVFP4 / MXFP8 uniformly quantize non-Fisher-allocated
# visual Linears via the existing RTN math at export time.
: "${VISUAL_FORMAT:=BF16}"
# Calibration modality. `text-only` (default) is the streaming body probe
# alone; `multimodal` adds a second non-streaming pass over the full
# model with image+text calibration so visual Linears get real Fisher
# stats. See header docstring.
: "${CALIBRATION_MODALITY:=text-only}"
# Multimodal dataset. `synthetic` is the offline stub that exercises the
# code path without network. Set to an HF dataset id (e.g.
# `HuggingFaceM4/COCO`) for real image calibration when
# CALIBRATION_MODALITY=multimodal.
: "${MM_DATASET:=synthetic}"

PROBE_PATH="${WORK_DIR}/artifacts/probe.pkl"
COST_PATH="${WORK_DIR}/artifacts/cost.pkl"

mkdir -p "${WORK_DIR}"/{artifacts,act,work,logs,exported}

echo "[pipeline] config:"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  WORK_DIR=$WORK_DIR"
echo "  FORMATS=$FORMATS  TARGET_BITS=$TARGET_BITS"
echo "  NSAMPLES=$NSAMPLES SEQLEN=$SEQLEN LAYERS_PER_SHARD=$LAYERS_PER_SHARD"
echo "  VISUAL_FORMAT=$VISUAL_FORMAT"
echo "  CALIBRATION_MODALITY=$CALIBRATION_MODALITY  MM_DATASET=$MM_DATASET"
echo

# -----------------------------------------------------------------------
# 1. Sensitivity probe (per-Linear empirical Fisher diagonal trace,
#    body + MTP in one pass)
# -----------------------------------------------------------------------
if [[ ! -f "${PROBE_PATH}" ]]; then
  echo "[pipeline] [1/4] running sensitivity probe ..."
  python3 -m prismaquant.incremental_probe \
    --model "$MODEL_PATH" \
    --dataset "$DATASET" \
    --nsamples "$NSAMPLES" --seqlen "$SEQLEN" \
    --device "$DEVICE" --dtype bf16 \
    --output "${PROBE_PATH}" \
    --activation-cache-dir "${WORK_DIR}/act" \
    --work-dir "${WORK_DIR}/work" \
    --layers-per-shard "$LAYERS_PER_SHARD" \
    --calibration-modality "$CALIBRATION_MODALITY" \
    --mm-dataset "$MM_DATASET" \
    --mm-nsamples 8 --mm-max-text-len 128 \
    2>&1 | tee "${WORK_DIR}/logs/probe.log"
else
  # Reuse guard: make sure the pre-existing probe.pkl matches the
  # currently-requested calibration modality. Silently reusing a
  # text-only probe under multimodal (or vice versa) would produce
  # an assignment calibrated for the wrong activation distribution
  # — visible later as bad PPL that's hard to root-cause. Fail loud
  # and point the user at the file to delete.
  probe_modality=$(python3 -c "
import pickle, sys
try:
    with open(sys.argv[1], 'rb') as f:
        blob = pickle.load(f)
    meta = blob.get('meta', {}) if isinstance(blob, dict) else {}
    m = meta.get('calibration_modality') or meta.get('modality') or 'text-only'
    print(m)
except Exception as e:
    print(f'__error__:{e}', file=sys.stderr)
    sys.exit(2)
" "${PROBE_PATH}" 2>/dev/null || echo "__unknown__")
  if [[ "${probe_modality}" == "__unknown__" ]]; then
    echo "[pipeline] [1/4] probe.pkl exists but its calibration_modality"
    echo "             could not be read. Aborting to avoid mixing probes."
    echo "             Delete it explicitly to regenerate:"
    echo "               rm ${PROBE_PATH}"
    exit 2
  fi
  if [[ "${probe_modality}" != "${CALIBRATION_MODALITY}" ]]; then
    echo "[pipeline] [1/4] ABORT: probe.pkl was calibrated for"
    echo "             modality='${probe_modality}' but this run requests"
    echo "             CALIBRATION_MODALITY='${CALIBRATION_MODALITY}'."
    echo "             Reusing the probe would silently produce an"
    echo "             assignment calibrated on the wrong activations."
    echo ""
    echo "             Delete the stale probe to regenerate:"
    echo "               rm ${PROBE_PATH}"
    echo "             Or unset CALIBRATION_MODALITY to match the probe."
    exit 2
  fi
  echo "[pipeline] [1/4] probe.pkl exists (modality=${probe_modality}), skipping"
fi

# -----------------------------------------------------------------------
# 2. Cost measurement (per-(Linear, format) measured RTN error,
#    body + MTP in one pass)
# -----------------------------------------------------------------------
if [[ ! -f "${COST_PATH}" ]]; then
  echo "[pipeline] [2/4] measuring per-(layer, format) cost ..."
  python3 -m prismaquant.incremental_measure_quant_cost \
    --model "$MODEL_PATH" \
    --probe "${PROBE_PATH}" \
    --activation-cache-dir "${WORK_DIR}/act" \
    --formats "$FORMATS" \
    --output "${COST_PATH}" \
    --work-dir "${WORK_DIR}/work" \
    --device "$DEVICE" --dtype bf16 \
    --mode batched --chunk-size 256 \
    --layers-per-shard "$LAYERS_PER_SHARD" \
    --skip-missing-activations \
    --swap-grow-limit-mb "${SWAP_GROW_LIMIT_MB:-2048}" \
    2>&1 | tee "${WORK_DIR}/logs/cost.log"
else
  echo "[pipeline] [2/4] cost.pkl exists, skipping"
fi

# -----------------------------------------------------------------------
# 3. Allocator (multi-choice knapsack over per-layer formats)
# -----------------------------------------------------------------------
echo "[pipeline] [3/4] running allocator at target=${TARGET_BITS} bpp ..."
# Choose visual-sensitivity mode from calibration modality:
#   text-only → uniform (Phase 1 --visual-format path, as before)
#   multimodal → fisher (Phase 2: DP places visual Linears from real
#                        multimodal Fisher; --visual-format acts as a
#                        fallback for un-probed visual Linears only)
if [[ "$CALIBRATION_MODALITY" == "multimodal" ]]; then
  VISUAL_SENSITIVITY=fisher
else
  VISUAL_SENSITIVITY=uniform
fi
python3 -m prismaquant.allocator \
  --probe "${PROBE_PATH}" \
  --costs "${COST_PATH}" \
  --target-bits "$TARGET_BITS" \
  --formats "$FORMATS" \
  --target-profile "$TARGET_PROFILE" \
  --pareto-targets "$PARETO_TARGETS" \
  --visual-format "$VISUAL_FORMAT" \
  --visual-sensitivity "$VISUAL_SENSITIVITY" \
  --layer-config "${WORK_DIR}/artifacts/layer_config.json" \
  --pareto-csv "${WORK_DIR}/artifacts/pareto.csv" \
  2>&1 | tee "${WORK_DIR}/logs/allocator.log"

# -----------------------------------------------------------------------
# 4. Native compressed-tensors export
# -----------------------------------------------------------------------
echo "[pipeline] [4/4] exporting to compressed-tensors ..."
python3 -m prismaquant.export_native_compressed \
  --model "$MODEL_PATH" \
  --layer-config "${WORK_DIR}/artifacts/layer_config.json" \
  --output "${WORK_DIR}/exported" \
  --device "$EXPORT_DEVICE" \
  --activation-cache-dir "${WORK_DIR}/act" \
  2>&1 | tee "${WORK_DIR}/logs/export.log"

echo
echo "[pipeline] done."
echo "  Artifact: ${WORK_DIR}/exported"
echo "  Validate: python3 -m prismaquant.validate_native_export \\"
echo "              --model ${WORK_DIR}/exported"
echo "  Serve:    vllm serve ${WORK_DIR}/exported \\"
echo "              --quantization compressed-tensors --trust-remote-code"
