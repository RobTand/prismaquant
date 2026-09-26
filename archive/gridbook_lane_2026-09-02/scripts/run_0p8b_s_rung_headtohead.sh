#!/usr/bin/env bash
# ============================================================================
# run_0p8b_s_rung_headtohead.sh — K-vs-S rung head-to-head on Qwen3.5-0.8B
# ============================================================================
# Robert 2026-07-22: "try the signed-bit formats on a very small model — I
# want to finalize their propriety asap." Menu carries BOTH families at the
# SAME rates (K13-K16 product vs S13-S16 sign-magnitude, two-tier coding) +
# FP8_DYNAMIC + BF16 escape hatches; the allocator's per-Linear measured
# choices at a low-bit target ARE the verdict. Cost is measured directly for
# all 8 CB rungs (no ladder interp — S-rungs are outside the K-family law).
# Export/serve/KL of the winner proves the full serving chain.
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/qwen35-0p8b-bf16}"
export WORK_DIR="${WORK_DIR:-/home/rob/dq-runs/s-rung-headtohead}"
export TMPDIR="${WORK_DIR}/tmp"
mkdir -p "$WORK_DIR" "$TMPDIR" "$WORK_DIR/logs"

export EXPORT_CONTAINER=nvfp4_cb
export TARGET_PROFILE=nvfp4_cb
export COST_MODE=local
export PRODUCTION_CACHE=0
export PRODUCTION_RECACHE=0
export CB_LADDER_INTERP=0
export EXPORT_STREAMING=auto

export FORMATS="NVFP4_CB_K13,NVFP4_CB_K14,NVFP4_CB_K15,NVFP4_CB_K16,NVFP4_CB_S13,NVFP4_CB_S14,NVFP4_CB_S15,NVFP4_CB_S16,FP8_DYNAMIC,BF16"
export CB_SCALE_CODING=two_tier
export TARGET_BITS="${TARGET_BITS:-2.6}"
export PARETO_TARGETS="${PARETO_TARGETS:-2.4,2.6,2.8}"

export NSAMPLES=8
export SEQLEN=512
export CB_CODEBOOK_SOURCE=lattice
export PRISMAQUANT_CB_ENCODE_TIER=balanced
export CALIBRATION_MODALITY=text-only
export DEVICE=cuda
export EXPORT_DEVICE=cuda
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "==== K-vs-S head-to-head @ ${TARGET_BITS} bpp on $(basename $MODEL_PATH) ===="
echo "  FORMATS=$FORMATS"
PATH="/home/rob/dq-runs/venvs/prismaquant-cu130/bin:$PATH" \
  bash "${REPO}/prismaquant/run-pipeline.sh"
echo "## HEADTOHEAD EXPORT DONE"
