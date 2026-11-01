#!/usr/bin/env bash
# Reproduce the bounded fresh-text activation capture and LDLQ evaluation.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${REPO}/ldlq-fresh"
RUN_ROOT="${RUN_ROOT:-/home/rob/dq-runs/dsv4-flash-0731}"
MODEL="${MODEL:-${RUN_ROOT}/source}"
CALIB="${CALIB:-/home/rob/dq-runs/calibration/diverse-v1.jsonl}"
PYTHON="${PYTHON:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}"
EXT_DIR="${EXT_DIR:-${RUN_ROOT}/ext}"
CAPTURE_ROOT="${OUT}/.capture-tmp"

export PYTHONPATH="${REPO}"
export TMPDIR="${OUT}/tmp"
mkdir -p "${TMPDIR}" "${CAPTURE_ROOT}" "${OUT}/act"

"${PYTHON}" "${OUT}/fresh_validation.py" prepare-corpus \
  --source "${CALIB}" --out "${OUT}" --model "${MODEL}"

run_layer() {
  local layer="$1"
  local layer_out="${CAPTURE_ROOT}/l${layer}"
  mkdir -p "${layer_out}/act" "${layer_out}/work" "${layer_out}/logs"
  "${PYTHON}" -m prismaquant.incremental_probe \
    --model "${MODEL}" \
    --dataset "${OUT}/fresh-text.jsonl" \
    --nsamples 16 --seqlen 512 --calib-seed 42 \
    --device cuda --dtype bf16 \
    --output "${layer_out}/probe.pkl" \
    --activation-cache-dir "${layer_out}/act" \
    --work-dir "${layer_out}/work" \
    --layers-per-shard 1 --start-layer "${layer}" --end-layer "$((layer + 1))" \
    --prefetch-lookahead 2 --prefetch-workers 2 \
    --prefetch-min-available-gb 40 \
    --activation-rows-limit 64 \
    --calibration-modality text-only \
    --no-include-mtp --no-include-visual --no-include-lm-head \
    >"${layer_out}/logs/probe.log" 2>&1
}

run_layer 20
run_layer 40

"${PYTHON}" "${OUT}/fresh_validation.py" collect \
  --capture-root "${CAPTURE_ROOT}" --act-out "${OUT}/act" \
  --text-manifest "${OUT}/text_manifest.json" \
  --manifest-out "${OUT}/capture_manifest.json"

"${PYTHON}" "${OUT}/fresh_validation.py" evaluate \
  --out "${OUT}" \
  --sample-root "${RUN_ROOT}/tier3-sample" \
  --act-dir "${OUT}/act" \
  --heldout-report "${REPO}/rotpilot-out/holdout/REPORT.md" \
  --ext-dir "${EXT_DIR}" \
  --block-size 64 --damping-fraction 0.01

echo "Capture intermediates remain at ${CAPTURE_ROOT}; remove them after auditing logs."
