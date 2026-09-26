#!/usr/bin/env bash
# CPU-only, explicitly user-accepted study allocation grid for DSv4-Flash.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-/home/rob/dq-runs/dsv4-flash-0731}"
WORK="${WORK:-${RUN_ROOT}/prod-cal-0p6-v2}"
ART="${ART:-${WORK}/artifacts-mxfp4}"
SEGMENTS="${SEGMENTS:-${ART}/probe-k12k18/by-layer}"
OUT="${OUT:-${REPO}/research-alloc}"

BASE="${WORK}/artifacts/cost_full.pkl"
PROBE="${ART}/probe.pkl"
COL_WEIGHTS="${ART}/cb_col_weights.pkl"
ACCEPTED="${OUT}/cost_accepted.pkl"
MENU="NVFP4_CB_K12,NVFP4_CB_K13,NVFP4_CB_K14,NVFP4_CB_K15,NVFP4_CB_K16,NVFP4_CB_K17,NVFP4_CB_K18,FP8_CB_K36,MXFP4_SOURCE,FP8_BLOCK_UE8M0_SOURCE"
PARETO="${PARETO:-1.85,1.90,1.95,2.00,2.02,2.03,2.04,2.05,2.06,2.07,2.08,2.09,2.10,2.11,2.12,2.13,2.14,2.15,2.16,2.17,2.18,2.19,2.20,2.21,2.22,2.25,2.30,2.35,2.40,2.50,2.65,2.80,3.00,3.25,3.50,4.00,4.25,4.50}"
MTP_BYTES="${MTP_BYTES:-10862838300}"

mkdir -p "${OUT}/logs"

run_cell() {
  local variant="$1" budget_bytes="$2" extra_bytes="$3"
  local nominal_gb=$((budget_bytes / 1000000000))
  local effective_bytes=$((budget_bytes + extra_bytes))
  local effective_gb
  effective_gb="$(python3 -c "print(${effective_bytes}/1e9)")"
  local cell="${OUT}/${variant}-${nominal_gb}"
  mkdir -p "$cell"
  if [[ -s "$cell/selection.json" && -s "$cell/layer_config.json" ]]; then
    echo "[research-alloc] ${variant}-${nominal_gb}: complete outputs exist; reusing"
    return
  fi

  local assembly_args=()
  if [[ ! -f "$ACCEPTED" ]]; then
    assembly_args=(
      --research-cost-base "$BASE"
      --research-cost-segments-dir "$SEGMENTS"
    )
  fi
  echo "[research-alloc] ${variant}-${nominal_gb}: effective card ${effective_gb} GB"
  PYTHONPATH="$REPO" PRISMAQUANT_ACTIVATION_FAIR_PRICING=1 \
    python3 -m prismaquant.allocator \
      --probe "$PROBE" \
      --costs "$ACCEPTED" \
      --accept-research-cost-table \
      "${assembly_args[@]}" \
      --model-override "${RUN_ROOT}/source" \
      --formats "$MENU" \
      --target-bits 2.17 --pareto-targets "$PARETO" \
      --target-disk-gb "$effective_gb" \
      --artifact-overhead-reserve-bytes 268435456 \
      --target-profile nvfp4_cb \
      --cb-scale-coding two_tier --cb-codebook-source lattice \
      --cb-scale-sweep 1 --cb-ldlq 0 --cb-encode-tier balanced \
      --cb-col-weights "$COL_WEIGHTS" \
      --layer-config "$cell/layer_config.json" \
      --bit-attribution-json "$cell/bit_attribution.json" \
      --pareto-csv "$cell/pareto.csv" \
      >"${OUT}/logs/${variant}-${nominal_gb}.log" 2>&1
}

# (b): MTP remains in the immutable floor.
run_cell b 92000000000 0
run_cell b 88000000000 0
# (c): release the exact MTP bytes into the allocatable-equivalent budget.
run_cell c 92000000000 "$MTP_BYTES"
run_cell c 88000000000 "$MTP_BYTES"

PYTHONPATH="$REPO" python3 "$REPO/scripts/summarise_dual_alloc.py" "$OUT"
PYTHONPATH="$REPO" python3 "$REPO/scripts/knee_analysis.py" \
  --pareto "$OUT/b-92/pareto.csv" \
  --selection "$OUT/b-92/selection.json" \
  --out "$OUT/KNEE.json"
