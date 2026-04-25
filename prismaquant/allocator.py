#!/usr/bin/env python3
"""allocator.py — multi-choice knapsack mixed-precision assignment.

Given:
  - per-Linear empirical Fisher diagonal trace (from sensitivity_probe.py)
  - per-(Linear, format) measured quantization cost (from measure_quant_cost.py)
  - a bit budget (target average bits per parameter)
  - a format registry (any subset of registered formats)

Solve for a per-Linear format assignment that minimizes total predicted
loss increase subject to the bit budget.

Derivation of the per-(layer, format) predicted loss term
---------------------------------------------------------
Let L be the per-token loss (negative log-likelihood). Quantizing layer
ℓ's weight tensor W by ΔW = W_q - W produces a perturbed loss whose
expectation under the calibration distribution admits the standard
second-order expansion:

    E[ΔL] ≈ 0.5 · ΔW · F · ΔWᵀ                         (1)

where F is the Fisher information matrix of L w.r.t. W. Replacing F by
its diagonal (the standard HAWQ-V1 simplification) and approximating
F_ww by the empirical Fisher diagonal F̂_ww = E_token[(∂L/∂W_w)²]:

    E[ΔL] ≈ 0.5 · Σ_w F̂_ww · (ΔW_w)²                   (2)

Under the further assumption that the per-weight quantization error
(ΔW_w)² and the per-weight Fisher diagonal F̂_ww are uncorrelated across
w (which is the same assumption HAWQ already makes when it summarizes a
layer by a single scalar), this collapses to the product of two
per-layer scalars:

    E[ΔL] ≈ 0.5 · H_trace · MSE_W                       (3)

where
    H_trace = Σ_w F̂_ww            (per-token Fisher diagonal trace)
    MSE_W   = (1/n_w) · Σ_w (ΔW_w)²

Both quantities are produced by upstream stages:
    H_trace ← sensitivity_probe.py / FisherAccumulator (`h_trace`)
    MSE_W   ← measure_quant_cost.py (per-(layer, format) `weight_mse`)

So we use eq. (3) directly. There is no `* d_out` factor; the previous
implementation carried one but it does not appear in the derivation —
it was a holdover from an earlier output-side formulation that mixed
units and was off by a per-layer multiplicative constant that varies
with d_out.

For MoE experts an additional route-probability normalization is folded
into H_trace inside the probe so that sparsely-routed experts' Fisher
contributions are on the same per-token footing as dense layers'.

Solver:
  Multi-choice knapsack via DP with bit-budget discretization (we round
  bit costs to 0.001-bit bins). For 35B with ~300 Linears × 8 formats ×
  ~5000 budget bins, runtime is under 1s.

Fused-projection siblings (q/k/v/o, gate/up, ...) are post-processed:
  all siblings promoted to the highest format chosen for any of them,
  to match vLLM's fused-tensor loader constraints. Since promotion can
  push achieved bits past the requested budget, the DP is re-run with a
  tightened target until achieved is within tolerance.

Optional empirical calibration:
  If `--calibration` points at a JSON produced by calibrate_allocator.py
  containing `calibrated_gains[fmt] = α_fmt`, the predicted Δloss for
  format f is multiplied by α_f before the DP runs. This corrects for
  systematic over- or under-prediction per format observed against
  measured KL on the bake-off frontier.

Auto-Pareto knee via Kneedle (Satopää et al.). Reports the knee target
plus a few flanking points so you can eyeball.
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

from . import format_registry as fr
from .allocator_solver import (
    Candidate,
    _candidate_for_assignment,
    _group_by_profile,
    _shape_from_stats,
    compute_achieved,
    compute_assignment_predicted_dloss,
    fused_siblings,
    predicted_dloss,
    promote_fused,
    promote_moe_pair,
    solve_allocation,
    solve_with_promotion,
)
from .allocator_candidates import (
    PASSTHROUGH_SOURCE_REQUIREMENTS,
    _FUSED_SIBLING_MARKER,
    _flashinfer_kernel_accepts,
    _format_kernel_supports_shape,
    _is_passthrough_format,
    _passthrough_source_ok,
    _scan_source_dtype_manifest,
    aggregate_fused_siblings,
    build_candidates,
    expand_fused_sibling_assignment,
)
from .allocator_prune import (
    _aggregate_candidate_memory_bits,
    _expert_ids_in_group,
    _moe_group_and_projection,
    _packed_entry_router_qname,
    _prune_cost_per_expert,
    _saliency_complete_for_eids,
    _saliency_has_eid,
    _saliency_lookup,
    aggregate_moe_candidates,
    apply_consensus_prune,
    apply_global_prune_ratio,
    apply_nested_global_prune_ratio,
    build_prune_manifest,
    compute_max_prune_ratio,
    expand_moe_assignment,
)
from .schemas import validate_cost_payload, validate_probe_payload



# ---------------------------------------------------------------------------
# Kneedle knee detection
# ---------------------------------------------------------------------------
def kneedle(x: list[float], y: list[float]) -> int:
    """Return index of the knee in a convex-decreasing curve."""
    if len(x) < 3:
        return 0
    xs = [xi for xi in x]
    ys = [yi for yi in y]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin or ymax == ymin:
        return 0
    x_norm = [(xi - xmin) / (xmax - xmin) for xi in xs]
    y_norm = [(yi - ymin) / (ymax - ymin) for yi in ys]
    # For a convex-decreasing curve, the knee is the point with max
    # distance below the chord from (0,1) to (1,0).
    diffs = [yn - (1.0 - xn) for xn, yn in zip(x_norm, y_norm)]
    # Convex-decreasing, so we want the most-negative diff (max dip).
    return min(range(len(diffs)), key=lambda i: diffs[i])




def _allowed_format(target_profile: str, name: str, fmt: str) -> bool:
    if target_profile == "research":
        return True
    if target_profile == "vllm_qwen3_5_packed_moe":
        if ".mlp.experts" in name:
            return fmt in {"NVFP4", "FP8_E4M3", "FP8_E5M2", "BF16", "MXFP4"}
        return True
    raise ValueError(f"Unknown target profile: {target_profile}")


def filter_candidates_for_profile(
    candidates: dict[str, list[Candidate]],
    target_profile: str,
) -> dict[str, list[Candidate]]:
    out = {}
    for name, cands in candidates.items():
        kept = [c for c in cands if _allowed_format(target_profile, name, c.fmt)]
        if kept:
            out[name] = kept
    return out


# ---------------------------------------------------------------------------
# Visual encoder override
# ---------------------------------------------------------------------------
# Phase 1 visual-encoder support: the probe's text-only calibration does not
# exercise the visual tower, so per-Linear Fisher gradients for
# `model.visual.blocks.*` Linears are zero — the knapsack DP has no
# sensitivity signal to allocate on. Rather than let every visual Linear
# default to the cheapest format or go through stale passthrough, we accept
# a single uniform target format (`BF16`, `NVFP4`, or `MXFP8`) and assign
# every visual Linear to it. BF16 (the default) reproduces the previous
# passthrough behavior; NVFP4/MXFP8 shrink the tower to quantized storage
# using the same RTN math the body gets.
#
# Phase 2 (tracked separately) will replace this override with a real
# multimodal Fisher: load images + text, run full forward through the
# visual encoder → projector → body LM, capture per-Linear empirical Fisher
# gradients, and feed those into the allocator's closed-form Δloss. That
# requires a multimodal dataset loader, multimodal tokenizer wiring, and a
# probe path that doesn't strip the visual tower — none of which ship in
# Phase 1.
_VISUAL_PREFIX_RE = re.compile(r"^(?:model\.)?visual\.")


def _is_visual_linear(name: str) -> bool:
    """True when `name` refers to a Linear inside the visual encoder.

    Matches both the raw HF checkpoint form (`model.visual.blocks.*`) and
    the post-remap form (`visual.blocks.*`) so the override behaves the
    same regardless of which side of `profile.live_to_recipe_name` the
    allocator's stats dictionary landed on.
    """
    return bool(_VISUAL_PREFIX_RE.match(name))


def apply_visual_format_override(
    assignment: dict[str, str],
    visual_format: str,
) -> dict[str, str]:
    """Force every visual-encoder Linear in `assignment` to `visual_format`.

    Called after the knapsack DP + fused-sibling promotion so the override
    wins even if the solver would have picked a different format per
    per-Linear sensitivity noise (which is meaningless for visual Linears
    under text-only calibration — see module comment above).

    `visual_format="BF16"` is a no-op when a visual Linear already has no
    allocator entry (the export's existing passthrough keeps it at BF16);
    we still write `BF16` into the returned assignment so the layer_config
    round-trip is explicit and downstream tooling (export, validate) has a
    uniform record of the decision.
    """
    out = dict(assignment)
    for name in list(out.keys()):
        if _is_visual_linear(name):
            out[name] = visual_format
    return out


def discover_visual_linears_from_source(model_path: str) -> list[str]:
    """Scan the source safetensors index for `model.visual.blocks.*.weight`
    entries with rank-2 shapes — these are the Linear modules the visual
    encoder exposes.

    Returned names are the basename (`.weight` stripped) so they slot
    directly into the allocator's assignment dictionary and the exporter's
    quantize-by-recipe dispatch.

    The probe's text-only staging strips the visual tower, so visual
    Linears never appear in the probe or cost pickles. This helper lets
    the allocator emit a layer_config entry for them anyway when
    `--visual-format` is non-BF16 — the exporter can then quantize each
    of them uniformly under the requested format. Without this scan, the
    allocator has no way to enumerate visual Linear names (there is no
    in-memory visual module at allocation time).
    """
    src = Path(model_path)
    idx_path = src / "model.safetensors.index.json"
    candidates: list[tuple[str, tuple[int, ...]]] = []
    if idx_path.exists():
        with open(idx_path) as f:
            wm = json.load(f).get("weight_map", {})
        # Index file carries only names, not shapes. We need to open each
        # referenced shard once to read rank.
        from collections import defaultdict as _dd
        by_shard: dict[str, list[str]] = _dd(list)
        for key, shard in wm.items():
            if not key.endswith(".weight"):
                continue
            if not _VISUAL_PREFIX_RE.match(key):
                continue
            by_shard[shard].append(key)
        try:
            from safetensors import safe_open
        except ImportError:
            return []
        for shard, keys in by_shard.items():
            shard_path = src / shard
            if not shard_path.exists():
                continue
            with safe_open(str(shard_path), framework="pt") as sf:
                for k in keys:
                    try:
                        shape = tuple(sf.get_slice(k).get_shape())
                    except Exception:
                        continue
                    candidates.append((k, shape))
    else:
        # No index file — scan every safetensors shard directly. Used for
        # small, single-file checkpoints.
        try:
            from safetensors import safe_open
        except ImportError:
            return []
        import os as _os
        if not src.exists():
            return []
        for f in sorted(_os.listdir(src)):
            if not f.endswith(".safetensors"):
                continue
            with safe_open(str(src / f), framework="pt") as sf:
                for k in sf.keys():
                    if not k.endswith(".weight"):
                        continue
                    if not _VISUAL_PREFIX_RE.match(k):
                        continue
                    try:
                        shape = tuple(sf.get_slice(k).get_shape())
                    except Exception:
                        continue
                    candidates.append((k, shape))

    # Only rank-2 weights are Linear-like; conv1d / norms / biases are
    # kept at BF16 passthrough regardless of --visual-format.
    # Additionally, blacklist known rank-2 tensors that live in
    # `nn.Parameter` / `nn.Embedding` modules (NOT `nn.Linear`), which
    # the compressed-tensors loader in vLLM cannot consume. Example:
    # `model.visual.pos_embed.weight` is an Embedding-like learned
    # parameter with shape (num_pos, hidden) — rank-2 but NOT a Linear.
    # Quantizing it produces `pos_embed.input_global_scale` etc. which
    # vLLM's VL runtime rejects with `KeyError: pos_embed.input_global_scale`
    # because its `model.visual.pos_embed` is a bare Parameter, not a
    # quantizable Linear module.
    _NON_LINEAR_RE = re.compile(
        r"(?:^|\.)("
        r"pos_embed"            # positional embedding (nn.Parameter/Embedding)
        r"|rotary_emb"          # rotary pos embed cache
        r")(?:\.|$)"
    )
    out: list[str] = []
    for name, shape in candidates:
        if len(shape) != 2:
            continue
        if _NON_LINEAR_RE.search(name):
            continue
        out.append(name[:-len(".weight")] if name.endswith(".weight") else name)
    return sorted(set(out))



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True, help="sensitivity_probe pickle")
    ap.add_argument("--costs", required=True, help="measure_quant_cost pickle")
    ap.add_argument("--model-override", default=None,
                    help="Override the model path stored in probe.pkl's meta. "
                         "Useful when re-running allocator against a probe "
                         "whose container-side paths no longer exist (e.g., "
                         "the original source was at /src/qwen36 in a prior "
                         "container run but is now only accessible via a "
                         "different mount). Overrides both profile detection "
                         "and visual-Linear source discovery.")
    ap.add_argument("--target-bits", type=float, default=4.75)
    ap.add_argument("--formats", default="",
                    help="Comma-separated format names to consider; empty=all")
    ap.add_argument("--pareto-targets",
                    default="4.5,4.6,4.7,4.75,4.85,5.0,5.25,5.5,6.0,7.0,8.25",
                    help="Comma-separated budgets to sweep for Pareto curve")
    ap.add_argument("--layer-config", required=True,
                    help="Output AutoRound layer_config JSON")
    ap.add_argument("--pareto-csv", required=True, help="Output Pareto CSV")
    ap.add_argument("--no-fused-promote", action="store_true",
                    help="Skip fused-projection sibling promotion")
    ap.add_argument("--no-fused-aggregation", action="store_true",
                    help="Disable pre-DP aggregation of fused siblings "
                         "(qkv_proj / gate_up_proj). Falls back to the "
                         "legacy promote_fused post-pass with tightening "
                         "retries. Pre-aggregation is strictly better "
                         "for hitting the target bit budget exactly on "
                         "dense models; use this flag only for "
                         "back-compat experiments.")
    ap.add_argument("--enforce-family-coherence", action="store_true",
                    help="Error (instead of warn) if the format set contains "
                         "multiple candidates for the same bit tier (e.g. "
                         "NVFP4 and MXFP4 both at 4 bits)")
    ap.add_argument("--bit-precision", type=float, default=0.0001,
                    help="Knapsack bit-bin granularity in avg-bits/param "
                         "(smaller = slower; default 0.0001 → ~50000 bins). "
                         "Measured on MiniMax-M2.7: going from 0.001 to 0.0001 "
                         "cuts predicted Δloss ~10% at the same bit budget. "
                         "Coarser values (0.01) leave 40% on the table.")
    ap.add_argument("--threads", type=int, default=0,
                    help="OMP/numpy threads for DP (0 = default)")
    ap.add_argument("--expert-granularity", choices=["layer", "expert"],
                    default="layer",
                    help="MoE experts allocation granularity. 'layer' (default) "
                         "assigns one format to all experts in a layer's fused "
                         "tensor — required for full-speed fused-MoE serving "
                         "on every major stack (vLLM FlashInfer/Marlin, SGLang, "
                         "TensorRT-LLM). 'expert' allows per-expert mixing but "
                         "forces slower sequential serving and is noise-floor "
                         "limited at typical calibration budgets.")
    ap.add_argument("--enable-expert-prune", action="store_true",
                    help="Joint REAP prune + quant optimization. When set, "
                         "the allocator generates DROP-variant candidates per "
                         "MoE super-Linear at each ratio in --prune-ratios, "
                         "and the DP picks the best (format, prune_ratio) "
                         "combination per layer under the bpp budget. Requires "
                         "expert saliency in the probe pickle (emitted "
                         "automatically since 2026-04-24; older probes have "
                         "no-op empty saliency). The exporter drops the "
                         "selected expert ids + shrinks the router output dim.")
    ap.add_argument("--prune-ratios",
                    default="0.0,0.125,0.25,0.375,0.5",
                    help="Comma-separated candidate prune ratios per MoE "
                         "super-Linear. Each ratio R triggers dropping the "
                         "floor(R · num_experts) lowest-saliency experts. "
                         "0.0 always included implicitly (no-prune candidate).")
    ap.add_argument("--prune-alpha", type=float, default=0.5,
                    help="Scalar calibration on the prune Δloss formula "
                         "α · S_j, where S_j is the probe's REAP dropout-"
                         "loss estimate for the expert. Smaller α makes "
                         "pruning cheaper (the DP prunes more aggressively); "
                         "larger α protects experts.")
    ap.add_argument("--target-profile",
                    choices=["research", "vllm_qwen3_5_packed_moe"],
                    default="research",
                    help="Serving/backend constraint profile. "
                         "'vllm_qwen3_5_packed_moe' collapses Qwen3.5/3.6 MoE "
                         "to legal packed serving units and restricts MoE "
                         "formats to the existing vLLM path.")
    ap.add_argument("--calibration", default=None,
                    help="Optional path to a calibrate_allocator.py JSON "
                         "containing 'calibrated_gains[fmt] = α_fmt'. When "
                         "present, the per-(layer, format) predicted Δloss "
                         "is multiplied by α_fmt before the DP runs.")
    ap.add_argument("--overshoot-tolerance", type=float, default=0.01,
                    help="Maximum allowed overshoot (bits/param) of the "
                         "achieved budget over the requested target after "
                         "fused-sibling promotion. The DP is re-run with a "
                         "tightened target until overshoot is within tol.")
    ap.add_argument("--visual-format",
                    choices=["BF16", "NVFP4", "MXFP8"],
                    default="BF16",
                    help="Uniform format for all visual-encoder Linears "
                         "(`model.visual.blocks.*`). Phase 1 fallback: "
                         "assigned to every visual Linear when "
                         "--visual-sensitivity=uniform OR when --visual-"
                         "sensitivity=fisher but the probe / cost pickles "
                         "don't carry real visual Fisher data. BF16 (default) "
                         "reproduces passthrough behavior; NVFP4 / MXFP8 "
                         "shrink the tower to quantized storage via the "
                         "existing RTN math at export time.")
    ap.add_argument("--visual-sensitivity",
                    choices=["fisher", "uniform"],
                    default="fisher",
                    help="How visual-encoder Linears enter the allocator. "
                         "'fisher' (default) treats them as regular DP "
                         "candidates when the probe pickle carries real "
                         "multimodal Fisher stats (produced by "
                         "`incremental_probe --calibration-modality="
                         "multimodal`). If those stats are missing, falls "
                         "back to uniform --visual-format. 'uniform' forces "
                         "the Phase 1 path: every visual Linear gets "
                         "--visual-format regardless of what's in the probe.")
    args = ap.parse_args()

    if args.threads > 0:
        import os
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)

    # Detect the model profile from the probe's metadata. The probe
    # writes `meta.model` when it runs, so we can look up the HF
    # config at that path and map it to a registered ModelProfile.
    # Profile governs fused-sibling promotion (allocator's
    # `promote_fused`) and the vLLM-internal name remap
    # (`build_quantization_config` via export_native_compressed).
    from .model_profiles import detect_profile, DefaultProfile
    model_profile = DefaultProfile()
    with open(args.probe, "rb") as f:
        _probe_peek = pickle.load(f)
    validate_probe_payload(_probe_peek, args.probe)
    probe_model_path = _probe_peek.get("meta", {}).get("model")
    del _probe_peek
    if args.model_override:
        probe_model_path = args.model_override
        print(f"[alloc] model-override: {probe_model_path}", flush=True)
    if probe_model_path:
        model_profile = detect_profile(probe_model_path)
        print(f"[alloc] model profile: {model_profile.name} "
              f"(derived from {probe_model_path})", flush=True)

    with open(args.probe, "rb") as f:
        probe = pickle.load(f)
    with open(args.costs, "rb") as f:
        cost_data = pickle.load(f)
    validate_probe_payload(probe, args.probe)
    validate_cost_payload(cost_data, args.costs)
    stats = probe["stats"]
    costs = cost_data["costs"]
    print(f"[alloc] stats: {len(stats)} Linears, costs: {len(costs)} Linears")

    if args.formats:
        fmt_names = [s.strip() for s in args.formats.split(",") if s.strip()]
    else:
        fmt_names = cost_data["formats"]
    specs = [fr.get_format(n) for n in fmt_names]
    specs_sorted = sorted(specs, key=lambda s: s.effective_bits)

    # --- Format-family coherence check -----------------------------------
    # A sensible format ladder has at most ONE format per bit tier. Having
    # both NVFP4 and MXFP4 (or MXFP6_E3M2 and MXFP6_E2M3) means the allocator
    # picks between them based on tiny measurement noise per-layer, which
    # produces a serving mess: two separate kernel paths for the same tier.
    #
    # We bucket formats by effective_bits rounded to 0.25 and warn when a
    # bucket has more than one member. If --enforce-family-coherence is
    # set we error instead.
    from collections import Counter as _Counter
    buckets: dict[float, list[str]] = {}
    for s in specs_sorted:
        key = round(s.effective_bits * 4) / 4
        buckets.setdefault(key, []).append(s.name)
    collisions = {k: v for k, v in buckets.items() if len(v) > 1}
    if collisions:
        msg = ("format set has multiple candidates at the same bit tier; "
               "the allocator will pick among them based on per-layer RTN "
               "noise, which is usually not what you want:\n"
               + "\n".join(f"  {k} bits: {v}" for k, v in collisions.items())
               + "\nRecommended bundles (vLLM serving, today):\n"
               "  Ship-ready     : NVFP4,MXFP8       (validated)\n"
               "  MX-pure        : MXFP4,MXFP8\n"
               "  Experimental   : NVFP4,MXFP6_E3M2,MXFP8   "
               "(MXFP6 hardware-supported on Blackwell, vLLM kernels not yet landed)")
        if args.enforce_family_coherence:
            raise SystemExit(f"[alloc] ERROR: {msg}")
        else:
            print(f"[alloc] WARNING: {msg}", flush=True)
    format_rank = {s.name: i for i, s in enumerate(specs_sorted)}
    format_specs = {s.name: s for s in specs}
    print(f"[alloc] formats (low→high bits): "
          f"{[f'{s.name}({s.effective_bits:.2f}b)' for s in specs_sorted]}")

    # Optional empirical calibration: per-format scalar gain α_f produced
    # by calibrate_allocator.py. When absent, all gains default to 1.0.
    calibrated_gains: dict[str, float] = {}
    if args.calibration:
        with open(args.calibration) as f:
            cal_payload = json.load(f)
        cal_raw = cal_payload.get("calibrated_gains") or {}
        for fmt_name, gain_val in cal_raw.items():
            try:
                calibrated_gains[fmt_name] = float(gain_val)
            except (TypeError, ValueError):
                continue
        if calibrated_gains:
            print(f"[alloc] calibration loaded from {args.calibration}: "
                  f"{ {k: round(v, 4) for k, v in calibrated_gains.items()} }",
                  flush=True)
        else:
            print(f"[alloc] WARNING: {args.calibration} has no usable "
                  f"calibrated_gains; running uncalibrated", flush=True)

    # Source-dtype manifest drives passthrough-integrity filtering in
    # build_candidates. None when model path is unknown — candidates
    # fall back to cost-pickle-only gating (pre-passthrough behavior).
    source_manifest: dict[str, str] | None = None
    if probe_model_path:
        source_manifest = _scan_source_dtype_manifest(
            probe_model_path, model_profile)
        if source_manifest:
            n_fp8 = sum(1 for v in source_manifest.values() if v == "fp8")
            n_bf16 = sum(1 for v in source_manifest.values() if v == "bf16")
            print(f"[alloc] source-dtype manifest: {n_fp8} fp8, "
                  f"{n_bf16} bf16 (gates FP8_SOURCE/BF16 per source)",
                  flush=True)

    candidates = build_candidates(
        stats, costs, specs_sorted, calibrated_gains,
        source_manifest=source_manifest,
    )
    print(f"[alloc] candidates built for {len(candidates)} Linears")

    # Joint REAP-prune + quant: ingest the observer-collected saliency
    # and expert_info from the probe pickle. Both are empty dicts for
    # dense models or legacy probes; the aggregator falls through to
    # format-only candidates when either is empty.
    probe_expert_saliency = probe.get("expert_saliency", {}) if args.enable_expert_prune else {}
    probe_expert_info = probe.get("expert_info", {}) if args.enable_expert_prune else {}
    # Parse the user's ratio list as a SWEEP. For packed-3D models we
    # couple every layer to the same ratio (below) to match vLLM's
    # uniform-num_experts constraint — so "multi-ratio" means "try each
    # of these values globally and pick the best per target_bits."
    # For nested-MoE models we still hand the list down to
    # `aggregate_moe_candidates`'s per-layer prune path.
    user_prune_ratios = tuple(
        float(x) for x in args.prune_ratios.split(",")
        if x.strip() and float(x) > 0.0
    ) if args.enable_expert_prune else ()
    # Top-k safety floor: we must never drop below `top_k` experts
    # kept per layer, else routing has fewer experts than it tries to
    # select. Read top_k from the probe meta; default 8 for Qwen3.5.
    top_k = 8
    probe_meta = probe.get("meta", {}) if isinstance(probe.get("meta"), dict) else {}
    if "top_k" in probe_meta:
        top_k = int(probe_meta["top_k"])
    # Filter ratios to those satisfying min-kept >= top_k.
    global_ratio_sweep: list[float] = [0.0]
    if args.enable_expert_prune and probe_expert_saliency:
        try:
            max_r = compute_max_prune_ratio(stats, top_k)
        except ValueError as e:
            raise SystemExit(f"[alloc] {e}")
        above_floor = [r for r in user_prune_ratios if r > max_r]
        valid = sorted({r for r in user_prune_ratios if 0 < r <= max_r})
        if above_floor:
            print(
                f"[alloc] dropped prune ratios above kept>=top_k "
                f"(top_k={top_k}, max_R={max_r:.3f}): "
                f"{sorted(above_floor)}", flush=True,
            )
        global_ratio_sweep = [0.0] + valid
        n_routers_with_saliency = sum(
            1 for r in probe_expert_saliency.values() if r
        )
        print(
            f"[alloc] expert-prune ENABLED: saliency routers="
            f"{n_routers_with_saliency}, sweep ratios={global_ratio_sweep}, "
            f"top_k={top_k}, alpha={args.prune_alpha}", flush=True,
        )

    # Nested-MoE aggregation with per-layer prune variants (research
    # path). For packed-3D models `aggregate_moe_candidates` is a
    # no-op here because no super-Linears form.
    if args.target_profile == "vllm_qwen3_5_packed_moe":
        stats, costs, candidates = aggregate_moe_candidates(
            stats, costs, specs_sorted, candidates, granularity="layer",
            calibrated_gains=calibrated_gains,
            expert_saliency=probe_expert_saliency,
            expert_info=probe_expert_info,
            prune_ratios=(),  # packed-3D path handles prune below
            prune_alpha=args.prune_alpha,
            source_manifest=source_manifest)
        moe_groups = sum(1 for n in candidates if ".__fused__." in n)
        print(f"[alloc] packed-MoE serving aggregation: {moe_groups} fused MoE blocks")
    elif args.expert_granularity == "layer":
        stats, costs, candidates = aggregate_moe_candidates(
            stats, costs, specs_sorted, candidates, granularity="projection",
            calibrated_gains=calibrated_gains,
            expert_saliency=probe_expert_saliency,
            expert_info=probe_expert_info,
            prune_ratios=user_prune_ratios,
            prune_alpha=args.prune_alpha,
            source_manifest=source_manifest)
        moe_groups = sum(1 for n in candidates if ".__fused__." in n)
        print(f"[alloc] MoE aggregation: {moe_groups} fused-expert super-Linears")

    # MiniMax-style nested MoE exports also carry one scalar expert-count
    # field in config.json. They therefore need the same global-ratio
    # discipline as packed-3D MoE: the DP may choose quant formats per
    # group, but not independently choose each layer's prune ratio.
    nested_global_prune = (
        args.enable_expert_prune
        and args.expert_granularity == "layer"
        and args.target_profile == "research"
        and getattr(model_profile, "name", "") in {"minimax_m2"}
    )
    if nested_global_prune:
        print("[alloc] nested-MoE global prune sweep enabled "
              f"(profile={model_profile.name})", flush=True)

    # Pre-aggregate fused siblings (qkv_proj, gate_up_proj, ...) into
    # single DP items. The DP can't pick mixed-sibling solutions because
    # there's only one item per group — so promote_fused becomes a no-op
    # on aggregated items and the overshoot-tightening loop collapses to
    # a single pass on well-behaved models. Must run AFTER the MoE
    # aggregation (it skips `.__fused__.` entries explicitly).
    if not args.no_fused_aggregation:
        stats, costs, candidates = aggregate_fused_siblings(
            stats, costs, specs_sorted, candidates, profile=model_profile,
            calibrated_gains=calibrated_gains)
        sib_groups = sum(1 for n in candidates if _FUSED_SIBLING_MARKER in n)
        print(f"[alloc] fused-sibling aggregation: {sib_groups} groups "
              f"(qkv_proj / gate_up_proj / ...)")

    candidates = filter_candidates_for_profile(candidates, args.target_profile)

    # `candidates` is now fully aggregated (MoE, fused siblings) and
    # filtered. For the packed-3D prune sweep we clone it per ratio
    # and rewrite the packed entries in place via
    # `apply_global_prune_ratio`. The original is preserved as the
    # R=0 baseline.
    import copy as _copy

    def _solve_for_ratio(R: float, target_bits: float):
        """Solve the DP at a single (ratio, target_bits) combination.
        Returns (assignment, pruned_map, achieved, total_dloss) or
        (None, {}, nan, inf) if infeasible."""
        # Clear any lingering packed-entry memory maps from a previous
        # R's apply_global_prune_ratio — otherwise R=0 would pick up
        # the last non-zero ratio's shrunk bytes from stats. Safe to
        # pop unconditionally; apply_global_prune_ratio repopulates
        # when R > 0.
        for name, s in stats.items():
            if (_packed_entry_router_qname(name) is not None
                    and isinstance(s, dict)):
                s.pop("_memory_bytes_by_format", None)
        if R > 0.0 and probe_expert_saliency:
            c = _copy.deepcopy(candidates)
            apply_global_prune_ratio(
                c, stats, probe_expert_saliency,
                global_ratio=R, prune_alpha=args.prune_alpha,
            )
        else:
            c = candidates
        if nested_global_prune:
            filtered, filter_warnings = apply_nested_global_prune_ratio(
                c, stats, global_ratio=R,
            )
            if filtered is None:
                if filter_warnings:
                    print("[alloc] nested-MoE global prune ratio "
                          f"R={R:.6g} infeasible: {filter_warnings[0]}",
                          flush=True)
                return None, {}, float("nan"), float("inf")
            c = filtered
        assign, pruned_map_r, achieved_r = solve_with_promotion(
            stats, c, target_bits, format_specs, format_rank,
            args.bit_precision,
            no_fused_promote=args.no_fused_promote,
            overshoot_tolerance=args.overshoot_tolerance,
            profile=model_profile,
        )
        if assign is None:
            return None, {}, float("nan"), float("inf")
        total = compute_assignment_predicted_dloss(assign, c, pruned_map_r)
        return assign, pruned_map_r, achieved_r, total

    # Pareto sweep: for each target_bits, pick the (R, assignment)
    # with the lowest predicted Δloss among all ratios in the sweep.
    targets = [float(x) for x in args.pareto_targets.split(",")]
    curve = []
    for t in targets:
        best = None
        for R in global_ratio_sweep:
            assign, pruned_r, achieved, total = _solve_for_ratio(R, t)
            if assign is None:
                continue
            if best is None or total < best["predicted_dloss"]:
                best = {
                    "ratio": R,
                    "assignment": assign,
                    "pruned_map": pruned_r,
                    "achieved_bits": achieved,
                    "predicted_dloss": total,
                }
        if best is None:
            curve.append({"target_bits": t, "feasible": False})
            continue
        format_counts = defaultdict(int)
        format_params = defaultdict(int)
        for name, fmt in best["assignment"].items():
            format_counts[fmt] += 1
            format_params[fmt] += stats[name]["n_params"]
        n_layers_pruned = len(best["pruned_map"])
        n_experts_dropped = sum(len(v) for v in best["pruned_map"].values())
        curve.append({
            "target_bits": t,
            "feasible": True,
            "achieved_bits": best["achieved_bits"],
            "predicted_dloss": best["predicted_dloss"],
            "global_prune_ratio": best["ratio"],
            "n_layers_pruned": n_layers_pruned,
            "n_experts_dropped": n_experts_dropped,
            **{f"layers_{k}": v for k, v in format_counts.items()},
            **{f"params_{k}": v for k, v in format_params.items()},
        })

    # Output Pareto CSV
    keys = sorted({k for row in curve for k in row.keys()})
    with open(args.pareto_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in curve:
            w.writerow(row)
    print(f"[alloc] Pareto curve → {args.pareto_csv}")

    # Kneedle
    feasible = [row for row in curve if row.get("feasible")]
    if len(feasible) >= 3:
        kidx = kneedle([r["achieved_bits"] for r in feasible],
                       [r["predicted_dloss"] for r in feasible])
        knee = feasible[kidx]
        print(f"[alloc] suggested knee: target={knee['target_bits']}, "
              f"achieved={knee['achieved_bits']:.3f}, "
              f"Δloss={knee['predicted_dloss']:.3e}")

    # Print table
    print("\n  target  achieved     Δloss (pred)   " + "   ".join(
        f"{s.name[:11]:>11}" for s in specs_sorted))
    for row in curve:
        if not row.get("feasible"):
            print(f"  {row['target_bits']:>6.3f}  INFEASIBLE")
            continue
        fmt_str = "   ".join(
            f"{row.get(f'layers_{s.name}', 0):>11,}" for s in specs_sorted)
        print(f"  {row['target_bits']:>6.3f}  {row['achieved_bits']:>7.3f}  "
              f"{row['predicted_dloss']:>14.4e}   {fmt_str}")

    # Emit chosen layer_config for target_bits. Same outer-loop-over-R
    # pattern as the Pareto sweep — pick the ratio that minimizes
    # predicted Δloss at the requested target.
    best_final = None
    for R in global_ratio_sweep:
        assign, pruned_r, achieved_r, total = _solve_for_ratio(
            R, args.target_bits,
        )
        if assign is None:
            continue
        if best_final is None or total < best_final["predicted_dloss"]:
            best_final = {
                "ratio": R,
                "assignment": assign,
                "pruned_map": pruned_r,
                "achieved_bits": achieved_r,
                "predicted_dloss": total,
            }
    if best_final is None:
        raise SystemExit(
            f"Infeasible at target_bits={args.target_bits}. "
            "Consider raising the target or widening the format set.")
    assignment = best_final["assignment"]
    pruned_map = best_final["pruned_map"]
    achieved = best_final["achieved_bits"]
    chosen_ratio = best_final["ratio"]
    if chosen_ratio > 0.0:
        print(
            f"[alloc] target_bits={args.target_bits}: picked global prune "
            f"ratio R={chosen_ratio:.4f} (achieved_bits={achieved:.3f}, "
            f"Δloss={best_final['predicted_dloss']:.3e})",
            flush=True,
        )

    # Build the prune manifest (router-keyed, consensus-intersected)
    # from the DP's winning candidates. Empty when prune is disabled or
    # saliency is absent. Coerce each super-Linear's dropped set to the
    # router consensus so expansion and the sidecar agree.
    prune_manifest: dict[str, dict] = {}
    prune_warnings: list[str] = []
    if pruned_map:
        # expert_info may be empty for packed-3D models (no per-expert
        # leaves to discover). build_prune_manifest handles that via
        # _packed_entry_router_qname; no gating on expert_info here.
        # expert_saliency feeds the uniform-kept coercion pass
        # (padding lighter-pruned layers to match the heaviest prune,
        # using lowest-saliency experts as the extras).
        uniform_kept = (
            args.target_profile == "vllm_qwen3_5_packed_moe"
            or nested_global_prune
        )
        prune_manifest, prune_warnings = build_prune_manifest(
            pruned_map, stats, probe_expert_info,
            expert_saliency=probe_expert_saliency,
            uniform_kept=uniform_kept,
        )
        if uniform_kept:
            post_dp_warnings = [
                w for w in prune_warnings
                if "padded drops" in w or "DP chose no prune" in w
            ]
            if post_dp_warnings:
                sample = "\n  ".join(post_dp_warnings[:5])
                raise SystemExit(
                    "[alloc] prune uniform-kept would add unscored drops "
                    "after the DP. This means the packed global-ratio "
                    "candidate set did not cover every saliency router. "
                    f"Refusing to emit a sidecar that differs from the "
                    f"scored plan. Sample:\n  {sample}"
                )
        pruned_map = apply_consensus_prune(
            pruned_map, prune_manifest, stats, probe_expert_info,
        )
        for w in prune_warnings:
            print(f"[alloc] prune-consensus: {w}", flush=True)
        if not uniform_kept and prune_manifest:
            kept_counts = {
                int(e["num_experts_kept"]) for e in prune_manifest.values()
            }
            if len(kept_counts) > 1:
                print(
                    "[alloc] WARNING: prune manifest has mixed "
                    f"num_experts_kept values {sorted(kept_counts)}. "
                    "The exporter will reject this for HF/vLLM configs "
                    "with a single scalar expert-count field; use a "
                    "global packed-MoE prune ratio for serveable uniform "
                    "expert counts.",
                    flush=True,
                )
        total_orig = sum(r["num_experts_orig"] for r in prune_manifest.values())
        total_kept = sum(r["num_experts_kept"] for r in prune_manifest.values())
        print(
            f"[alloc] prune: {len(prune_manifest)} MoE layers, "
            f"{total_orig - total_kept}/{total_orig} experts dropped "
            f"(kept {total_kept})",
            flush=True,
        )

    # Expand MoE super-Linears back to per-expert entries before writing
    # the AutoRound layer_config (which expects one entry per individual
    # nn.Linear module name). When prune is active, dropped experts are
    # omitted from the expanded assignment — their leaves won't appear
    # in layer_config.json and the exporter won't quantize/export them.
    if args.expert_granularity == "layer":
        assignment_expanded = expand_moe_assignment(
            assignment, stats,
            pruned_map=pruned_map,
            expert_info=probe_expert_info,
        )
    else:
        assignment_expanded = dict(assignment)

    # Expand fused-sibling super-Linears (qkv_proj / gate_up_proj).
    # Complementary to the MoE expansion above — the MoE path handles
    # `.__fused__.` markers; the sibling path handles `.__siblings__.`
    # markers. Running both is idempotent on any assignment that doesn't
    # contain the relevant marker.
    if not args.no_fused_aggregation:
        assignment_expanded = expand_fused_sibling_assignment(
            assignment_expanded, stats)

    # Post-expansion MoE unity promotion. The super-Linear solver picks
    # a format per (layer, projection); expansion propagates that to
    # all per-expert leaves with the same projection. But vLLM's
    # FusedMoE requires ALL projections (gate+up+down) of the SAME
    # expert to share one scheme — so when the solver picks e.g.
    # w1=MXFP8 and w2=NVFP4 for layer L, every (L, expert_i) triple
    # would disagree and the serve-time dispatch raises 'All MoE
    # projections need same scheme'. This pass groups expanded leaves
    # by (experts-prefix, expert-idx) and bumps the lower-rank
    # projections to match the highest. Costs a little bpp on layers
    # where gate/up/down really wanted different formats, but that
    # cost is the ticket for serveability.
    assignment_expanded = promote_moe_pair(assignment_expanded, format_rank)

    # Visual-encoder Linear handling. Two paths:
    #
    # 1. --visual-sensitivity=fisher (default) + probe/cost have real
    #    visual entries → visual Linears already participated in the
    #    knapsack DP above with their own per-Linear Fisher + per-format
    #    RTN cost. No override needed; just make sure every discoverable
    #    visual Linear has an assignment entry (fall back to --visual-
    #    format for any that the probe missed, e.g. patch_embed Linears
    #    that the probe's regex didn't hit).
    #
    # 2. --visual-sensitivity=uniform OR Fisher missing → Phase 1 path:
    #    scan source checkpoint for visual Linears and stamp them all
    #    with --visual-format.
    visual_format = args.visual_format
    visual_sensitivity = args.visual_sensitivity

    def _visual_fisher_available(stats_d: dict, costs_d: dict) -> bool:
        """True when both the probe and cost pickles carry real visual
        entries — the signal a multimodal calibration pass ran."""
        any_visual_stats = any(_is_visual_linear(n) for n in stats_d)
        any_visual_costs = any(_is_visual_linear(n) for n in costs_d)
        return any_visual_stats and any_visual_costs

    fisher_visual_ok = (visual_sensitivity == "fisher"
                        and _visual_fisher_available(stats, costs))
    if visual_sensitivity == "fisher" and not fisher_visual_ok:
        print("[alloc] --visual-sensitivity=fisher requested but probe / "
              "cost pickles have no visual Linear entries; falling back "
              f"to --visual-format={visual_format} (Phase 1 uniform).",
              flush=True)

    if probe_model_path:
        visual_names_src = discover_visual_linears_from_source(probe_model_path)
    else:
        visual_names_src = []

    if fisher_visual_ok:
        # Fisher path: DP already placed visual Linears. Fill in any
        # discoverable visual Linear that the DP missed (e.g. the probe
        # regex matched only `visual.blocks.*` but the source has
        # `visual.merger.*` or `visual.patch_embed.*` too) with the
        # uniform --visual-format as a safety net.
        dp_visual_count = sum(1 for n in assignment_expanded
                              if _is_visual_linear(n))
        filled = 0
        for vname in visual_names_src:
            if vname not in assignment_expanded:
                assignment_expanded[vname] = visual_format
                filled += 1
        print(f"[alloc] --visual-sensitivity=fisher: DP placed "
              f"{dp_visual_count} visual Linears via per-Linear Fisher; "
              f"{filled} additional visual Linears (un-probed) stamped "
              f"with --visual-format={visual_format}.", flush=True)
    else:
        # Uniform path (Phase 1): stamp every discoverable visual Linear.
        if visual_names_src:
            for vname in visual_names_src:
                assignment_expanded[vname] = visual_format
            print(f"[alloc] --visual-format={visual_format}: assigned "
                  f"{len(visual_names_src)} visual Linears uniformly "
                  f"(source={probe_model_path})", flush=True)
        elif visual_format != "BF16":
            print(f"[alloc] --visual-format={visual_format}: no visual "
                  f"Linears found in source checkpoint — override is a "
                  f"no-op", flush=True)

    # Passthrough-integrity belt-and-suspenders. The filter in
    # build_candidates drops mismatched FP8_SOURCE / BF16 per-Linear
    # candidate, but downstream aggregation + promotion (fused
    # siblings, MoE expert-unity) can in principle push a format onto
    # a group whose members have heterogeneous source dtypes. On
    # modern checkpoints this doesn't happen (siblings share source
    # dtype), but if it ever does we want a loud early failure rather
    # than a broken export artifact.
    if source_manifest:
        violations: list[tuple[str, str, str]] = []
        for name, fmt in assignment_expanded.items():
            if not _is_passthrough_format(fmt):
                continue
            kind = source_manifest.get(name)
            if kind is None:
                # Not in manifest — likely a visual Linear stamped via
                # --visual-format (bypasses the manifest by design) or
                # a name the profile rewrite didn't map. Skip.
                continue
            if not _passthrough_source_ok(fmt, kind):
                violations.append((name, fmt, kind))
        if violations:
            head = "\n  ".join(
                f"{n}: picked {f} but source is {k} "
                f"(requires {PASSTHROUGH_SOURCE_REQUIREMENTS[f]})"
                for n, f, k in violations[:10]
            )
            raise SystemExit(
                f"[alloc] passthrough-integrity violation: "
                f"{len(violations)} Linears have a passthrough format "
                f"picked over a mismatched source dtype. Sample:\n"
                f"  {head}\n"
                "The per-Linear filter should have excluded these — "
                "investigate fused-sibling / MoE-unity promotion."
            )

    layer_cfg = {}
    for name, fmt in assignment_expanded.items():
        if fmt in format_specs:
            layer_cfg[name] = format_specs[fmt].autoround_config()
        else:
            # Visual format outside the body's format set (e.g., user
            # passed --formats NVFP4,BF16 plus --visual-format MXFP8).
            # Resolve from the global registry.
            layer_cfg[name] = fr.get_format(fmt).autoround_config()

    out = Path(args.layer_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(layer_cfg, f, indent=2)

    # Prune sidecar: emitted alongside layer_config.json when any MoE
    # layer had experts dropped. Exporter reads this to drop the router
    # rows, reindex kept experts to dense 0..K-1, and update
    # config.json's num_experts. Absent sidecar => no pruning happened
    # and the exporter uses its pre-prune code path.
    prune_sidecar_path = out.with_suffix(out.suffix + ".prune.json")
    if prune_manifest:
        with open(prune_sidecar_path, "w") as f:
            json.dump(prune_manifest, f, indent=2, sort_keys=True)
        print(f"Prune manifest → {prune_sidecar_path}")
    elif prune_sidecar_path.exists():
        # Stale sidecar from an earlier prune run would silently taint
        # a subsequent non-prune run. Remove it.
        prune_sidecar_path.unlink()

    counts = defaultdict(int)
    for fmt in assignment.values():
        counts[fmt] += 1
    print(f"\n[alloc] target={args.target_bits} achieved={achieved:.3f}")
    for fmt, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {fmt:>14}: {n:>5} layers")
    print(f"\nLayer config → {out}")
    print(f"Feed to AutoRound via --layer_config {out}")


if __name__ == "__main__":
    main()
