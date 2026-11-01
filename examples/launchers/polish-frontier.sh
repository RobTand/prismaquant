#!/usr/bin/env bash
# Historical launcher for the archived cross-layer polish experiment.
# Its live module entry points were removed; retain these commands as research
# context, not as a supported quantization workflow.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/home/rob/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca}"
IMAGE="${IMAGE:-vllm-fresh-b12x:latest}"
RUN_ROOT="${RUN_ROOT:?set RUN_ROOT}"
TS="${TS:-$(date -u +%Y%m%dT%H%M%SZ)}"
CNAME="${CNAME:-pq-bc-polish-frontier-$(printf '%s' "${TS}" | tr '[:upper:]' '[:lower:]')}"

PAYLOAD="${PAYLOAD:?set PAYLOAD path}"
KNEEDLE_DIR="${KNEEDLE_DIR:?set KNEEDLE_DIR path}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/polish_frontier}"
LOG_NAME="${LOG_NAME:-polish_frontier.log}"

N_CALIB_SAMPLES="${N_CALIB_SAMPLES:-2}"
CALIB_SEQLEN="${CALIB_SEQLEN:-128}"
CALIB_SPLIT="${CALIB_SPLIT:-train}"
CALIB_SEED="${CALIB_SEED:-42}"
MAX_PASSES="${MAX_PASSES:-8}"
NOISE_FLOOR="${NOISE_FLOOR:-1e-5}"
BUDGET_CREEP="${BUDGET_CREEP:-0.05}"
USE_FROZEN_WEIGHT_CACHE="${USE_FROZEN_WEIGHT_CACHE:-1}"
STEEPEST_FIRST="${STEEPEST_FIRST:-1}"

mkdir -p "${OUTPUT_DIR}" "${RUN_ROOT}/logs"

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
  -e PAYLOAD="${PAYLOAD/$RUN_ROOT/\/work}" \
  -e KNEEDLE_DIR="${KNEEDLE_DIR/$RUN_ROOT/\/work}" \
  -e OUTPUT_DIR="${OUTPUT_DIR/$RUN_ROOT/\/work}" \
  -e N_CALIB_SAMPLES="${N_CALIB_SAMPLES}" \
  -e CALIB_SEQLEN="${CALIB_SEQLEN}" \
  -e CALIB_SPLIT="${CALIB_SPLIT}" \
  -e CALIB_SEED="${CALIB_SEED}" \
  -e MAX_PASSES="${MAX_PASSES}" \
  -e NOISE_FLOOR="${NOISE_FLOOR}" \
  -e BUDGET_CREEP="${BUDGET_CREEP}" \
  -e USE_FROZEN_WEIGHT_CACHE="${USE_FROZEN_WEIGHT_CACHE}" \
  -e STEEPEST_FIRST="${STEEPEST_FIRST}" \
  -e LOG_NAME="${LOG_NAME}" \
  -w /prismaquant \
  --entrypoint bash "${IMAGE}" \
  -lc '
    set -euo pipefail
    python3 -m pip install --user --quiet accelerate datasets 2>&1 \
      | tee "/work/logs/polish_frontier_pip.log"

    python3 - <<PYEOF 2>&1 | tee "/work/logs/${LOG_NAME}"
import json, os, tempfile
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from prismaquant import block_clado as bc
from prismaquant import coord_descent_polish as cdp
from prismaquant.build_rtn_cache import cache_reference_log_probs, stage_multimodal
from prismaquant.measure_adjoint_l3 import _dtype_from_name, load_wikitext_calibration_windowed
from prismaquant.model_profiles import DefaultProfile, detect_profile

model_path = os.environ["MODEL_PATH"]
payload_path = Path(os.environ["PAYLOAD"])
kneedle_dir = Path(os.environ["KNEEDLE_DIR"])
output_dir = Path(os.environ["OUTPUT_DIR"])
output_dir.mkdir(parents=True, exist_ok=True)

n_calib = int(os.environ["N_CALIB_SAMPLES"])
seqlen = int(os.environ["CALIB_SEQLEN"])
split = os.environ["CALIB_SPLIT"]
seed = int(os.environ["CALIB_SEED"])
max_passes = int(os.environ["MAX_PASSES"])
noise_floor = float(os.environ["NOISE_FLOOR"])
budget_creep = float(os.environ["BUDGET_CREEP"])
use_frozen = bool(int(os.environ["USE_FROZEN_WEIGHT_CACHE"]))
steepest = bool(int(os.environ["STEEPEST_FIRST"]))

print(f"[polish-frontier] payload={payload_path}")
print(f"[polish-frontier] kneedle_dir={kneedle_dir}")
candidates = sorted(p for p in kneedle_dir.glob("*.json") if p.name != "summary.json")
print(f"[polish-frontier] {len(candidates)} candidates")

# Load decision units once
units, pairs_by_block = cdp.units_and_pairs_from_payload(payload_path)
print(f"[polish-frontier] loaded {len(units)} units, "
      f"{sum(len(p) for p in pairs_by_block.values())} pairs")

# Load model + tokenizer + calib once
dtype = _dtype_from_name("bf16")
staged, cleanup = stage_multimodal(model_path)
work_root = Path(tempfile.mkdtemp(prefix="prismaquant_pf_"))
results = []
try:
    local_only = Path(staged).exists()
    tokenizer = AutoTokenizer.from_pretrained(staged, trust_remote_code=True, local_files_only=local_only)
    calib_ids = load_wikitext_calibration_windowed(tokenizer, n_calib, seqlen, split=split, seed=seed)
    model = AutoModelForCausalLM.from_pretrained(
        staged, torch_dtype=dtype, trust_remote_code=True, local_files_only=local_only,
        device_map="cuda",
    )
    model.eval()
    try:
        profile = detect_profile(model_path)
    except Exception:
        profile = DefaultProfile()
    device = next(model.parameters()).device
    ref_log_probs = cache_reference_log_probs(model, calib_ids, device)

    for path in candidates:
        payload = json.loads(path.read_text())
        starting = payload.get("assignment") or payload
        bpp = float(payload.get("bpp", 0.0))
        starting_bits = cdp._assignment_bits(units, starting)
        budget = starting_bits * (1.0 + budget_creep)
        result = cdp.coord_descent_polish(
            model, calib_ids, ref_log_probs,
            units=units, starting_assignment=starting,
            profile=profile, work_root=work_root,
            noise_floor=noise_floor, max_passes=max_passes,
            bits_budget=budget, bits_tolerance=0.0,
            pairs_by_block=pairs_by_block, steepest_first=steepest,
            use_frozen_weight_cache=use_frozen,
        )
        counts = dict(Counter(result.final_assignment.values()))
        row = {
            "label": path.stem,
            "starting_bpp": bpp,
            "starting_kl": result.initial_kl,
            "final_kl": result.final_kl,
            "improvement": result.initial_kl - result.final_kl,
            "n_steps_accepted": len(result.steps),
            "n_kl_measurements": result.n_kl_measurements,
            "elapsed_seconds": result.elapsed_seconds,
            "format_counts": counts,
            "final_assignment": result.final_assignment,
        }
        out_path = output_dir / f"{path.stem}.json"
        out_path.write_text(json.dumps(row, indent=2) + "\n")
        results.append(row)
        print(f"[polish-frontier] {path.stem:40s}  start_bpp={bpp:.4f}  "
              f"start_kl={result.initial_kl:.4f}  final_kl={result.final_kl:.4f}  "
              f"steps={len(result.steps)}  elapsed={result.elapsed_seconds:.1f}s",
              flush=True)
finally:
    import shutil
    if cleanup is not None:
        shutil.rmtree(cleanup, ignore_errors=True)
    shutil.rmtree(work_root, ignore_errors=True)

# Summary CSV-style
summary_path = output_dir / "summary.json"
summary_path.write_text(json.dumps({
    "schema": "prismaquant.polish_frontier.v1",
    "n_candidates": len(results),
    "results": results,
}, indent=2) + "\n")
print(f"[polish-frontier] wrote {summary_path}")

# Pareto curve
print("\\n=== polished Pareto ===")
print(f"{'label':>40s}  {'start_bpp':>9s}  {'start_kl':>9s}  {'final_kl':>9s}  {'steps':>6s}")
for r in sorted(results, key=lambda x: x["starting_bpp"]):
    print(f"{r['label']:>40s}  {r['starting_bpp']:9.4f}  {r['starting_kl']:9.4f}  "
          f"{r['final_kl']:9.4f}  {r['n_steps_accepted']:6d}")
PYEOF
  '

echo "[launch] container: ${CNAME}"
echo "[launch] output:    ${OUTPUT_DIR}/"
echo "[launch] tail:      tail -f ${RUN_ROOT}/logs/${LOG_NAME}"
