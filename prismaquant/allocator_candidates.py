"""Candidate construction and coupled-candidate aggregation."""
from __future__ import annotations

import json
from pathlib import Path

from . import format_registry as fr
from .allocator_solver import Candidate, _shape_from_stats, predicted_dloss


PASSTHROUGH_SOURCE_REQUIREMENTS: dict[str, str] = {
    "FP8_SOURCE": "fp8",
    "BF16": "bf16",
}


def _is_passthrough_format(format_name: str) -> bool:
    return format_name in PASSTHROUGH_SOURCE_REQUIREMENTS


def _passthrough_source_ok(
    format_name: str,
    source_kind: str | None,
) -> bool:
    required = PASSTHROUGH_SOURCE_REQUIREMENTS.get(format_name)
    if required is None:
        return True
    if source_kind is None:
        return True
    return source_kind == required


def _format_kernel_supports_shape(fmt_name: str, in_features: int,
                                  out_features: int) -> bool:
    """Return True if the runtime kernel can handle this Linear shape."""
    flashinfer_verdict = _flashinfer_kernel_accepts(
        fmt_name, in_features, out_features)
    if flashinfer_verdict is False:
        return False

    if fmt_name.startswith("MXFP8"):
        if out_features < 128 or in_features < 128:
            return False
        if in_features % 32 != 0:
            return False
        if out_features % 128 != 0:
            return False
        return True
    if fmt_name in ("INT2", "INT3", "NVINT2", "NVINT3"):
        return in_features % 16 == 0
    if fmt_name.startswith("NVFP4"):
        return in_features % 16 == 0
    return True


def _flashinfer_kernel_accepts(fmt_name: str, in_features: int,
                               out_features: int) -> bool | None:
    """Ask FlashInfer's own problem-size validator when available."""
    try:
        if fmt_name.startswith("MXFP8"):
            from flashinfer.gemm.gemm_base import _check_mm_mxfp8_problem_size
            import torch
            a = torch.empty((1, in_features), dtype=torch.float8_e4m3fn)
            b = torch.empty((in_features, out_features),
                            dtype=torch.float8_e4m3fn)
            from flashinfer.gemm.gemm_base import _mxfp8_swizzled_scale_len
            from flashinfer.gemm.gemm_base import SfLayout
            a_desc_len = _mxfp8_swizzled_scale_len(
                a.shape[0], a.shape[1], SfLayout.layout_8x4)
            b_desc_len = _mxfp8_swizzled_scale_len(
                b.shape[1], b.shape[0], SfLayout.layout_8x4)
            a_desc = torch.empty((a_desc_len,), dtype=torch.uint8)
            b_desc = torch.empty((b_desc_len,), dtype=torch.uint8)
            try:
                return _check_mm_mxfp8_problem_size(a, b, a_desc, b_desc) is True
            except Exception:
                return False
        return None
    except Exception:
        return None


def build_candidates(stats: dict, costs: dict, formats: list[fr.FormatSpec],
                     calibrated_gains: dict[str, float] | None = None,
                     source_manifest: dict[str, str] | None = None,
                     ) -> dict[str, list[Candidate]]:
    """Build legal format candidates for every measured Linear."""
    gains = calibrated_gains or {}
    out: dict[str, list[Candidate]] = {}
    masked_by_shape: dict[str, list[str]] = {}
    masked_by_passthrough: dict[str, list[str]] = {}
    for name, s in stats.items():
        if name not in costs:
            continue
        h_trace = s["h_trace"]
        shape = _shape_from_stats(s)
        in_features = int(s.get("in_features", 0) or 0)
        out_features = int(s.get("out_features", 0) or 0)
        source_kind = (source_manifest or {}).get(name)
        cands = []
        for spec in formats:
            entry = None
            entry_fmt = spec.name
            for candidate_name in fr.aliases_for(spec.name):
                if candidate_name in costs[name]:
                    entry = costs[name][candidate_name]
                    entry_fmt = candidate_name
                    break
            if entry is None or "error" in entry:
                continue
            if (source_kind is not None
                    and _is_passthrough_format(spec.name)
                    and not _passthrough_source_ok(spec.name, source_kind)):
                masked_by_passthrough.setdefault(spec.name, []).append(name)
                continue
            if in_features and out_features and not _format_kernel_supports_shape(
                spec.name, in_features, out_features
            ):
                masked_by_shape.setdefault(spec.name, []).append(name)
                continue
            gain = float(gains.get(spec.name, gains.get(entry_fmt, 1.0)))
            if "predicted_dloss" in entry:
                predicted = float(entry["predicted_dloss"]) * gain
            else:
                weight_mse = float(entry.get("weight_mse", 0.0))
                predicted = predicted_dloss(h_trace, weight_mse, gain=gain)
            cands.append(Candidate(
                fmt=spec.name,
                bits_per_param=spec.effective_bits_for_shape(shape),
                memory_bytes=spec.memory_bytes_for_shape(shape),
                predicted_dloss=max(predicted, 0.0),
            ))
        if cands:
            out[name] = cands
    if masked_by_shape:
        for fmt, names in masked_by_shape.items():
            print(f"[alloc] kernel shape-mask: {len(names)} Linear(s) "
                  f"dropped {fmt} (sample: {names[:3]})", flush=True)
    if masked_by_passthrough:
        for fmt, names in masked_by_passthrough.items():
            print(f"[alloc] passthrough-integrity: {len(names)} Linear(s) "
                  f"dropped {fmt} (source dtype mismatch; "
                  f"sample: {names[:3]})", flush=True)
    return out


_FUSED_SIBLING_MARKER = ".__siblings__."


def aggregate_fused_siblings(
    stats: dict,
    costs: dict,
    formats: list[fr.FormatSpec],
    candidates: dict[str, list[Candidate]],
    profile,
    calibrated_gains: dict[str, float] | None = None,
) -> tuple[dict, dict, dict]:
    """Aggregate fused siblings into single DP items."""
    if profile is None:
        return stats, costs, candidates

    gains = calibrated_gains or {}
    grouped: dict[str, list[str]] = {}
    ungrouped: list[str] = []
    for name in candidates:
        if ".__fused__." in name:
            ungrouped.append(name)
            continue
        try:
            key = profile.fused_sibling_group(name)
        except Exception:
            key = None
        if key is None:
            ungrouped.append(name)
            continue
        grouped.setdefault(key, []).append(name)

    for key in list(grouped.keys()):
        if len(grouped[key]) < 2:
            ungrouped.extend(grouped.pop(key))

    if not grouped:
        return stats, costs, candidates

    stats_ext = {n: stats[n] for n in ungrouped}
    costs_ext = {n: costs.get(n, {}) for n in ungrouped}
    candidates_ext = {n: candidates[n] for n in ungrouped}

    for key, members in grouped.items():
        members = sorted(members)
        safe_key = key.replace(".", "__")
        super_name = f"{members[0].rsplit('.', 1)[0]}{_FUSED_SIBLING_MARKER}{safe_key}"

        n_params = sum(stats[m]["n_params"] for m in members)
        sum_h = sum(stats[m]["h_trace"] for m in members)
        d_out = int(stats[members[0]].get("out_features", 0) or 0)
        d_in = int(stats[members[0]].get("in_features", 0) or 0)

        stats_ext[super_name] = {
            "h_trace": sum_h,
            "h_trace_raw": sum(stats[m].get("h_trace_raw", 0.0) for m in members),
            "h_w2_sum": sum(stats[m].get("h_w2_sum", 0.0) for m in members),
            "w_max_abs": max(stats[m].get("w_max_abs", 0.0) for m in members),
            "w_norm_sq": sum(stats[m].get("w_norm_sq", 0.0) for m in members),
            "n_params": n_params,
            "in_features": d_in,
            "out_features": d_out,
            "n_tokens_seen": sum(stats[m].get("n_tokens_seen", 0) for m in members),
            "_fused_siblings": members,
            "_memory_bytes_by_format": {},
        }

        super_cost = {}
        for spec in formats:
            missing = [m for m in members
                       if spec.name not in costs.get(m, {})
                       or "error" in costs.get(m, {}).get(spec.name, {})]
            if missing:
                super_cost[spec.name] = {"error": "partial"}
                continue
            sum_pred = 0.0
            for m in members:
                c = costs[m][spec.name]
                if "predicted_dloss" in c:
                    sum_pred += float(c["predicted_dloss"])
                else:
                    h_i = stats[m]["h_trace"]
                    sum_pred += 0.5 * h_i * float(c.get("weight_mse", 0.0))
            effective_mse = sum_pred / (0.5 * sum_h) if sum_h > 0 else 0.0
            super_cost[spec.name] = {
                "weight_mse": effective_mse,
                "predicted_dloss": sum_pred,
            }
        costs_ext[super_name] = super_cost

        member_format_sets = [
            {c.fmt for c in candidates.get(m, [])}
            for m in members
        ]
        if member_format_sets:
            member_format_intersection = set.intersection(*member_format_sets)
        else:
            member_format_intersection = set()

        cands = []
        for spec in formats:
            if spec.name not in member_format_intersection:
                continue
            entry = super_cost.get(spec.name)
            if entry is None or "error" in entry:
                continue
            total_bytes = 0
            for m in members:
                shape = _shape_from_stats(stats[m])
                total_bytes += spec.memory_bytes_for_shape(shape)
            bits_per_param = 8.0 * total_bytes / max(n_params, 1)
            stats_ext[super_name]["_memory_bytes_by_format"][spec.name] = total_bytes
            gain = float(gains.get(spec.name, 1.0))
            predicted = entry["predicted_dloss"] * gain
            cands.append(Candidate(
                fmt=spec.name,
                bits_per_param=bits_per_param,
                memory_bytes=total_bytes,
                predicted_dloss=max(predicted, 0.0),
            ))
        if cands:
            candidates_ext[super_name] = cands

    return stats_ext, costs_ext, candidates_ext


def expand_fused_sibling_assignment(assignment: dict[str, str],
                                    stats_ext: dict) -> dict[str, str]:
    """Broadcast a fused-sibling super-item assignment back to members."""
    out = {}
    for name, fmt in assignment.items():
        if _FUSED_SIBLING_MARKER in name:
            members = stats_ext[name].get("_fused_siblings", [])
            for m in members:
                out[m] = fmt
        else:
            out[name] = fmt
    return out


def _scan_source_dtype_manifest(
    model_path: str,
    profile=None,
) -> dict[str, str]:
    """Classify source Linear weights as ``fp8`` or ``bf16`` for passthrough gating."""
    del profile
    src = Path(model_path)
    idx_path = src / "model.safetensors.index.json"
    if not idx_path.exists():
        return {}
    try:
        with open(idx_path) as f:
            weight_map = json.load(f).get("weight_map", {})
    except Exception:
        return {}
    bases: dict[str, set[str]] = {}
    for key in weight_map:
        for suffix in (".weight_scale_inv", ".weight"):
            if key.endswith(suffix):
                base = key[: -len(suffix)]
                bases.setdefault(base, set()).add(suffix[1:])
                break

    def _to_live_name(ck_base: str) -> str:
        if (ck_base.startswith("model.visual.")
                or ck_base.startswith("model.audio_tower.")
                or ck_base.startswith("model.vision_tower.")
                or ck_base.startswith("model.embed_vision.")
                or ck_base.startswith("model.embed_audio.")
                or ck_base.startswith("mtp.")):
            return ""
        if ck_base.startswith("model.language_model."):
            return "model." + ck_base[len("model.language_model."):]
        return ck_base

    manifest: dict[str, str] = {}
    for base, suffixes in bases.items():
        if "weight" not in suffixes:
            continue
        source_kind = "fp8" if "weight_scale_inv" in suffixes else "bf16"
        live_name = _to_live_name(base)
        if not live_name:
            continue
        manifest[live_name] = source_kind
    return manifest
