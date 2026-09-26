"""Empirical routed-MoE expert costs for the AURA hybrid recipe.

AURA's smooth per-Linear cost is route-flip-blind on routed experts (Step A,
2026-06-29: Spearman drops 0.45->0.35 under faithful dW; predicted NVFP4/FP8
ratios 2-49x vs measured 1.1-1.5x), so expert costs are MEASURED, not
modeled: per MoE layer the serving unit = all profile-coupled routed expert
tensors (packed Parameters or per-expert Linears; they must share one format —
vLLM FusedMoE constraint), and the unit cost of a format is the end-to-end
mean-token KL(BF16 || unit-quantized) with everything else left at source
precision. The unit KL is split across the member tensors proportionally to
n_params so the allocator's per-member aggregation charges it exactly once.

The quantizer is plain RTN ``quantize_dequantize`` from the format registry —
the same estimator contract as the AURA non-expert cost (RTN-vs-GPTQ dW is a
wash at fp4 and RTN is *better* at fp8 on the served 27B A/B); the deliberate
GPTQ render happens later in the production cache, and real-KL frontier
selection (M4) judges the actual rendered bytes.

FP8 stays IN the expert menu (standing decision 2026-06-29): it is
Pareto-dominated on routed experts (~1.3x lower KL for 2x bits), and the
right place for that fact to act is the allocator's DP + the real-KL
frontier — not a hardcoded ban here.

This module also performs the hybrid merge that previously lived as a
one-off in /home/rob/dq-runs/aura-35b/: ``--merge-base`` unions these expert
rows into an AURA (non-expert) cost payload, and ``--backfill-base`` copies
rows for any name the merged payload still lacks (MTP / visual sidecars the
AURA pass never sees) from the baseline incremental cost.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import io_spans
from prismaquant import format_registry as fr
from prismaquant.tensor_digests import tensor_value_stamp as _tensor_value_stamp
from prismaquant.routed_experts import (
    UnpackedExpertLinear,
    _profile_call,
    profile_declared_routed_expert_targets,
    profile_declared_unpacked_expert_linears,
    resolve_routed_expert_profile,
)

SCHEMA = "prismaquant.expert_empirical_cost.v1"
PASSTHROUGH_FORMATS = {"BF16", "FP8_SOURCE"}


def _log(msg: str) -> None:
    print(f"[expert-cost {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _canon_formats(formats: Sequence[str]) -> list[str]:
    seen: list[str] = []
    for raw in formats:
        name = fr.canonical_format_name(str(raw).strip())
        if name and name not in seen:
            seen.append(name)
    return seen


_CALIB_BATCH_ENV = "PRISMAQUANT_EXPERT_CALIB_BATCH"


def _calib_batch() -> int:
    """Calibration sequences per forward. Default 1 preserves the historical
    per-sequence numerics exactly; >1 batches independent windows (semantics
    identical, per-position arithmetic may differ at reassociation level —
    baseline and quantized arms always use the SAME batching, so the KL
    comparison stays internally consistent). Becomes the dominant-wall knob
    once expert sampling shrinks the encode side."""
    return max(1, int(os.environ.get(_CALIB_BATCH_ENV, "1") or 1))


@torch.no_grad()
def _baseline_logprobs(
    model, calib_ids: torch.Tensor,
    *,
    forward_model=None,
) -> list[torch.Tensor]:
    """Baseline log-probs over the calibration windows."""
    out = []
    bs = _calib_batch()
    n_total = calib_ids.shape[0]
    t0 = time.time()
    for i in range(0, n_total, bs):
        logits = (forward_model or model)(
            calib_ids[i:i + bs]
        ).logits.float()
        out.append(F.log_softmax(logits, dim=-1).cpu())
        done = min(i + bs, n_total)
        dt = time.time() - t0
        tps = done * calib_ids.shape[1] / max(dt, 1e-9)
        _log(f"baseline forward {done}/{n_total} windows "
             f"(batch={bs}, {dt:.0f}s elapsed, {tps:.0f} tok/s)")
    return out


@torch.no_grad()
def _expert_sample_idx(num_experts: int, sample: int) -> torch.Tensor | None:
    """Deterministic stratified expert subsample (the local cost path's
    linspace pattern — even coverage of the expert index range)."""
    if sample <= 0 or num_experts <= sample:
        return None
    return torch.linspace(0, num_experts - 1, sample).round().long().unique()


def _quantize_unit_inplace(
    mod,
    param_names: Sequence[str],
    fmt: str,
    *,
    expert_chunk: int = 16,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    unit_qname: str = "",
    sample_idx: torch.Tensor | None = None,
) -> None:
    """Render every member of one expert serving unit in-place in ``fmt``.

    ``sample_idx`` (expert subsampling): quantize ONLY those expert slices,
    leaving the rest BF16 — the caller extrapolates the partial unit KL.
    Each sampled expert's render is identical to its full-stack render, so
    sampling changes COVERAGE, never the bytes measured for a covered expert.
    """
    spec = fr.get_format(fmt)
    qdq = spec.quantize_dequantize
    for pn in param_names:
        w = getattr(mod, pn).data
        if spec.family == "nv":
            # NV formats derive one per-TENSOR global scale from
            # whatever slice they are given, while export ships one
            # global PER EXPERT. Chunk-batching would share a global
            # across the chunk and make the measured KL depend on the
            # --expert-chunk knob; quantize per expert slice instead
            # (mirrors measure_quant_cost._batched_quantize, which does
            # the per-slice loop for exactly this reason).
            experts = (sample_idx.tolist() if sample_idx is not None
                       else range(w.shape[0]))
            for e in experts:
                w[e] = qdq(w[e].float()).to(w.dtype)
        else:
            # Scale-local formats are chunk-invariant, so batching is
            # safe: FP8_E4M3/FP8_E5M2 reshape to (-1, in) and scale each
            # output row independently (fp8_dynamic_weight_qdq), and
            # group/block-scaled formats (MX) never cross the expert
            # boundary within a row.
            if sample_idx is not None:
                idx = sample_idx.to(w.device)
                w[idx] = qdq(w[idx].float()).to(w.dtype)
            else:
                for e in range(0, w.shape[0], expert_chunk):
                    w[e:e + expert_chunk] = qdq(
                        w[e:e + expert_chunk].float()).to(w.dtype)


@torch.no_grad()
def _unit_kl(
    model,
    calib_ids: torch.Tensor,
    baseline: list[torch.Tensor],
    mod,
    param_names: Sequence[str],
    fmt: str,
    *,
    expert_chunk: int = 16,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    unit_qname: str = "",
    sample_idx: torch.Tensor | None = None,
    forward_model=None,
) -> float:
    """Mean-token KL(BF16 || model-with-this-unit-quantized).

    With ``sample_idx``, only those expert slices are quantized (and cloned
    for restore) — the caller owns the extrapolation to the full unit."""
    if sample_idx is None:
        originals = {pn: getattr(mod, pn).data.clone() for pn in param_names}
    else:
        originals = {pn: getattr(mod, pn).data[
            sample_idx.to(getattr(mod, pn).device)].clone()
            for pn in param_names}
    try:
        _quantize_unit_inplace(
            mod, param_names, fmt, expert_chunk=expert_chunk,
            col_weights=col_weights, unit_qname=unit_qname,
            sample_idx=sample_idx)
        total = 0.0
        n_tok = 0
        bs = _calib_batch()
        for bi, i in enumerate(range(0, calib_ids.shape[0], bs)):
            lp = F.log_softmax(
                (forward_model or model)(
                    calib_ids[i:i + bs]
                ).logits.float(), -1)
            bl = baseline[bi].to(lp.device)
            kl = (bl.exp() * (bl - lp)).sum(-1)
            total += float(kl.sum().item())
            n_tok += kl.numel()
        return total / max(n_tok, 1)
    finally:
        for pn in param_names:
            w = getattr(mod, pn).data
            if sample_idx is None:
                w.copy_(originals[pn])
            else:
                w[sample_idx.to(w.device)] = originals[pn]


class _UnpackedExpertUnit(NamedTuple):
    """One routed serving unit backed by per-expert ``nn.Linear`` rows."""

    qname: str
    group_key: str
    members: tuple[UnpackedExpertLinear, ...]
    roles: tuple[tuple[str, tuple[str, ...]], ...]
    num_experts: int
    members_by_target: dict[str, dict[tuple[str, int], str]]


_expert_profile_call = partial(_profile_call, purpose="render", subject="layout")


def _unpacked_expert_units(model, profile) -> list[_UnpackedExpertUnit]:
    """Validate and group profile-declared per-expert Linear units.

    Classification already happened through ``packed_expert_format_group``.
    The remaining profile accessors declare how those physical rows form the
    packed serving tensors used by the quantizer/exporter.  Missing projection
    coverage, non-contiguous expert ids, or an accessor that cannot answer is
    a hard error; none of those states is equivalent to a dense model.
    """
    discovered = profile_declared_unpacked_expert_linears(model, profile)
    grouped: dict[str, list[UnpackedExpertLinear]] = {}
    for member in discovered:
        grouped.setdefault(member.unit_qname, []).append(member)

    units: list[_UnpackedExpertUnit] = []
    for qname in sorted(grouped):
        members = sorted(grouped[qname], key=lambda member: member.qname)
        group_keys = {member.group_key for member in members}
        if len(group_keys) != 1:
            raise RuntimeError(
                f"{qname}: profile {type(profile).__name__} assigned the "
                f"unpacked expert rows to conflicting serving-format groups: "
                f"{sorted(group_keys)!r}"
            )

        by_projection: dict[str, dict[int, UnpackedExpertLinear]] = {}
        for member in members:
            per_expert = by_projection.setdefault(member.projection_name, {})
            if member.expert_id in per_expert:
                raise RuntimeError(
                    f"{qname}: duplicate routed expert projection "
                    f"{member.projection_name!r} for expert {member.expert_id}"
                )
            per_expert[member.expert_id] = member
        expected_ids: list[int] | None = None
        for projection, per_expert in sorted(by_projection.items()):
            ids = sorted(per_expert)
            if ids != list(range(len(ids))):
                raise RuntimeError(
                    f"{qname}: non-contiguous expert ids for {projection}: "
                    f"{ids[:16]!r}"
                )
            if expected_ids is None:
                expected_ids = ids
            elif ids != expected_ids:
                raise RuntimeError(
                    f"{qname}: expert ids differ across projections; "
                    f"{projection} has {ids[:16]!r}, expected "
                    f"{expected_ids[:16]!r}"
                )
        if not expected_ids:
            raise RuntimeError(f"{qname}: routed expert group has no members")

        projections_by_parent: dict[str, set[str]] = {}
        for projection in by_projection:
            parent = _expert_profile_call(
                profile, "packed_expert_parent_for_projection", projection
            )
            if not isinstance(parent, str) or not parent:
                raise RuntimeError(
                    f"{qname}: profile {type(profile).__name__} grouped "
                    f"projection {projection!r} as routed but did not declare "
                    "its packed serving parent"
                )
            projections_by_parent.setdefault(parent, set()).add(projection)

        roles: list[tuple[str, tuple[str, ...]]] = []
        members_by_target: dict[str, dict[tuple[str, int], str]] = {}
        consumed: set[str] = set()
        for parent in sorted(projections_by_parent):
            declared = _expert_profile_call(
                profile, "packed_expert_projection_names", parent
            )
            try:
                projections = tuple(str(name) for name in declared)
            except TypeError as exc:
                raise RuntimeError(
                    f"{qname}: profile {type(profile).__name__} returned a "
                    f"malformed projection order for packed parent {parent!r}"
                ) from exc
            actual = projections_by_parent[parent]
            if not projections or set(projections) != actual:
                raise RuntimeError(
                    f"{qname}: packed parent {parent!r} needs exactly profile-"
                    f"declared projections {projections!r}, found "
                    f"{sorted(actual)!r}"
                )
            if len(set(projections)) != len(projections):
                raise RuntimeError(
                    f"{qname}: packed parent {parent!r} repeats a projection "
                    f"in {projections!r}"
                )
            consumed.update(projections)
            target = f"{qname}.{parent}"
            members_by_target[target] = {
                (projection, expert_id):
                    by_projection[projection][expert_id].qname
                for projection in projections
                for expert_id in expected_ids
            }
            roles.append((parent, projections))

            # Packing concatenates projections along the output dimension.
            # Every row in one role must therefore share an input width, and
            # every expert must expose the same physical shape per projection.
            input_widths: set[int] = set()
            for projection in projections:
                shapes = {
                    tuple(int(dim) for dim in member.module.weight.shape)
                    for member in by_projection[projection].values()
                }
                if len(shapes) != 1:
                    raise RuntimeError(
                        f"{qname}: {projection} shapes differ across experts: "
                        f"{sorted(shapes)!r}"
                    )
                shape = next(iter(shapes))
                if len(shape) != 2:
                    raise RuntimeError(
                        f"{qname}: profile-declared unpacked Linear "
                        f"{projection} has non-matrix shape {shape!r}"
                    )
                input_widths.add(shape[1])
            if len(input_widths) != 1:
                raise RuntimeError(
                    f"{qname}: packed parent {parent!r} projections disagree "
                    f"on input width: {sorted(input_widths)!r}"
                )
        if consumed != set(by_projection):
            raise RuntimeError(
                f"{qname}: routed expert layout did not consume projections "
                f"{sorted(set(by_projection) - consumed)!r}"
            )
        units.append(_UnpackedExpertUnit(
            qname=qname,
            group_key=next(iter(group_keys)),
            members=tuple(members),
            roles=tuple(roles),
            num_experts=len(expected_ids),
            members_by_target=members_by_target,
        ))
    return units


def _virtual_packed_module(unit: _UnpackedExpertUnit) -> nn.Module:
    """Materialize the export-equivalent packed stacks for one live unit."""
    by_key = {
        (member.projection_name, member.expert_id): member
        for member in unit.members
    }
    packed = nn.Module()
    for parent, projections in unit.roles:
        expert_rows = []
        for expert_id in range(unit.num_experts):
            tensors = [
                by_key[(projection, expert_id)].module.weight.detach()
                for projection in projections
            ]
            expert_rows.append(
                tensors[0] if len(tensors) == 1
                else torch.cat(tensors, dim=0)
            )
        setattr(
            packed,
            parent,
            nn.Parameter(torch.stack(expert_rows), requires_grad=False),
        )
    return packed


def _scatter_virtual_packed_module(
    unit: _UnpackedExpertUnit,
    packed: nn.Module,
    expert_ids: Sequence[int],
) -> None:
    by_key = {
        (member.projection_name, member.expert_id): member
        for member in unit.members
    }
    for parent, projections in unit.roles:
        tensor = getattr(packed, parent).data
        offset = 0
        for projection in projections:
            width = int(by_key[(projection, 0)].module.weight.shape[0])
            for expert_id in expert_ids:
                by_key[(projection, int(expert_id))].module.weight.data.copy_(
                    tensor[int(expert_id), offset:offset + width]
                )
            offset += width


@torch.no_grad()
def _unpacked_unit_kl(
    model,
    calib_ids: torch.Tensor,
    baseline: list[torch.Tensor],
    unit: _UnpackedExpertUnit,
    fmt: str,
    *,
    expert_chunk: int = 16,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    sample_idx: torch.Tensor | None = None,
    forward_model=None,
) -> float:
    """Unit KL for per-expert Linears via the shipped packed-stack render."""
    if sample_idx is None:
        expert_ids = list(range(unit.num_experts))
    else:
        expert_ids = [int(value) for value in sample_idx.tolist()]
    selected = set(expert_ids)
    originals = {
        member.qname: member.module.weight.data.clone()
        for member in unit.members
        if member.expert_id in selected
    }
    packed = _virtual_packed_module(unit)
    param_names = [parent for parent, _projections in unit.roles]
    try:
        _quantize_unit_inplace(
            packed,
            param_names,
            fmt,
            expert_chunk=expert_chunk,
            col_weights=col_weights,
            unit_qname=unit.qname,
            sample_idx=sample_idx,
        )
        _scatter_virtual_packed_module(unit, packed, expert_ids)
        # The packed stacks are a full copy of the unit's expert mass
        # (~25 GB on a GLM-5.3 layer) and are redundant once scattered
        # back into the live members. Holding them through the forward
        # streams stacked a third expert-mass copy on top of `originals`
        # + resident weights and wedged the box twice; free the blocks
        # back to the shared pool before the eval forwards.
        for parent, _projections in unit.roles:
            delattr(packed, parent)
        del packed
        torch.cuda.empty_cache()
        total = 0.0
        n_tok = 0
        bs = _calib_batch()
        for bi, i in enumerate(range(0, calib_ids.shape[0], bs)):
            lp = F.log_softmax(
                (forward_model or model)(
                    calib_ids[i:i + bs]
                ).logits.float(), -1
            )
            bl = baseline[bi].to(lp.device)
            kl = (bl.exp() * (bl - lp)).sum(-1)
            total += float(kl.sum().item())
            n_tok += kl.numel()
        return total / max(n_tok, 1)
    finally:
        for member in unit.members:
            original = originals.get(member.qname)
            if original is not None:
                member.module.weight.data.copy_(original)
        # Return the clone's ~25 GB to the shared pool before the next
        # format's eval clones again (unified memory: allocator-cached
        # blocks are invisible to `avail` and to other processes).
        originals.clear()
        torch.cuda.empty_cache()


def measure_expert_unit_costs(
    model,
    profile,
    calib_ids: torch.Tensor,
    formats: Sequence[str],
    *,
    expert_chunk: int = 16,
    progress: bool = True,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    expert_sample: int = 0,
    max_units: int = 0,
    unit_filter: str | None = None,
    forward_model=None,
    baseline_logprobs: list[torch.Tensor] | None = None,
) -> tuple[dict, dict, dict]:
    """Measure empirical KL costs for profile-declared routed experts.

    Returns ``(stats, costs, unit_kls)`` where stats/costs are
    allocator-payload row dicts keyed by full member names and ``unit_kls``
    maps ``experts_qname -> {fmt: unit_kl}``.
    """
    from prismaquant.sensitivity_probe import (
        _is_packed_experts_module,
        _packed_experts_param_names,
    )

    profile = resolve_routed_expert_profile(model, profile)
    menu = _canon_formats(formats)
    measured_fmts = [f for f in menu if f not in PASSTHROUGH_FORMATS]
    # First classify every routed target through the profile-owned, rank-free
    # predicate.  Physical-layout discovery below is allowed to inspect rank
    # only to choose a renderer; it must account for every classified target.
    routed_targets = set(profile_declared_routed_expert_targets(model, profile))
    packed_units = [
        (qn, m) for qn, m in model.named_modules()
        if _is_packed_experts_module(m, profile)
    ]
    unpacked_units = _unpacked_expert_units(model, profile)
    packed_qnames = {qn for qn, _mod in packed_units}
    unpacked_qnames = {unit.qname for unit in unpacked_units}
    overlap = sorted(packed_qnames & unpacked_qnames)
    if overlap:
        raise RuntimeError(
            "profile-declared routed expert units expose both packed "
            f"Parameters and unpacked Linears at {overlap[:8]!r}; refusing "
            "to measure overlapping state twice"
        )

    accounted: set[str] = {
        member.qname
        for unit in unpacked_units
        for member in unit.members
    }
    for qn, mod in packed_units:
        for param_name in _packed_experts_param_names(mod, profile):
            accounted.add(f"{qn}.{param_name}" if qn else str(param_name))
    if accounted != routed_targets:
        missing = sorted(routed_targets - accounted)
        unexpected = sorted(accounted - routed_targets)
        raise RuntimeError(
            "profile-declared routed expert discovery could not form an "
            "empirical serving unit for every target; "
            f"unaccounted={missing[:8]!r}, non-profile targets="
            f"{unexpected[:8]!r}"
        )

    units: list[tuple[str, str, object]] = [
        ("packed", qn, mod) for qn, mod in packed_units
    ] + [
        ("unpacked", unit.qname, unit) for unit in unpacked_units
    ]
    if unit_filter:
        pat = re.compile(unit_filter)
        units = [record for record in units if pat.search(record[1])]
    if max_units > 0:
        units = units[:max_units]
    if progress:
        _log(f"{len(units)} expert serving units; measured formats: "
             f"{measured_fmts} (menu {menu})"
             + (f"; expert_sample={expert_sample}" if expert_sample else ""))
    stats: dict = {}
    costs: dict = {}
    unit_kls: dict = {}
    if not units or not measured_fmts:
        return stats, costs, unit_kls

    baseline = (
        baseline_logprobs
        if baseline_logprobs is not None
        else _baseline_logprobs(
            model, calib_ids, forward_model=forward_model
        )
    )
    for unit_kind, qn, storage in units:
        if unit_kind == "packed":
            mod = storage
            pnames = list(_packed_experts_param_names(mod, profile))
            n_params_unit = sum(
                int(getattr(mod, pn).numel()) for pn in pnames
            )
            num_experts = int(getattr(mod, pnames[0]).shape[0])
            row_members = [
                (
                    f"{qn}.{pn}" if qn else pn,
                    getattr(mod, pn),
                    {
                        "num_experts": num_experts,
                        "_packed_experts_module": qn,
                        "_packed_param": pn,
                    },
                )
                for pn in pnames
            ]
            for pn in pnames:
                t = getattr(mod, pn)
                if not bool((t != 0).any()):
                    raise RuntimeError(
                        f"{qn}.{pn}: packed-expert stack is ALL ZERO — the "
                        f"checkpoint's per-expert weights were never mapped "
                        f"into the packed class param (the zero-expert "
                        f"calibration bug). A unit KL measured now would read "
                        f"exactly 0 for every format. Fill via "
                        f"layer_streaming.fill_packed_experts_from_source "
                        f"before measuring."
                    )
        else:
            unit = storage
            num_experts = unit.num_experts
            n_params_unit = sum(
                int(member.module.weight.numel()) for member in unit.members
            )
            row_members = [
                (
                    member.qname,
                    member.module.weight,
                    {"_unpacked_expert_unit": qn},
                )
                for member in unit.members
            ]
            if not any(
                bool((member.module.weight != 0).any())
                for member in unit.members
            ):
                raise RuntimeError(
                    f"{qn}: profile-declared unpacked expert unit is ALL "
                    "ZERO; an empirical unit KL would silently read zero for "
                    "every format"
                )
        # One stratified subsample SHARED across every format of the unit, so
        # inter-format comparability (what the allocator consumes) is exact
        # even under sampling; the extrapolation to the full unit rides on
        # cross-expert additivity (validated fp32-additive in this repo) and
        # is scaled by expert count (uniform stacks).
        sample_idx = _expert_sample_idx(num_experts, expert_sample)
        kl_scale = (float(num_experts) / float(sample_idx.numel())
                    if sample_idx is not None else 1.0)

        def kl_of(fmt):
            if unit_kind == "packed":
                out = _unit_kl(
                    model, calib_ids, baseline, mod, pnames, fmt,
                    expert_chunk=expert_chunk, col_weights=col_weights,
                    unit_qname=qn, sample_idx=sample_idx,
                    forward_model=forward_model)
            else:
                out = _unpacked_unit_kl(
                    model, calib_ids, baseline, unit, fmt,
                    expert_chunk=expert_chunk, col_weights=col_weights,
                    sample_idx=sample_idx,
                    forward_model=forward_model)
            return kl_scale * out

        kls = {fmt: kl_of(fmt) for fmt in measured_fmts}
        unit_kls[qn] = dict(kls)
        if sample_idx is not None:
            unit_kls[qn]["_sampling"] = {
                "num_experts": num_experts,
                "sampled": int(sample_idx.numel()),
                "scale": round(kl_scale, 4),
            }
        for full, tensor, expert_metadata in row_members:
            npm = int(tensor.numel())
            shape = list(tensor.shape)
            if unit_kind == "packed":
                in_features = int(shape[2])
                out_features = int(shape[1])
            else:
                in_features = int(shape[1])
                out_features = int(shape[0])
            stats[full] = {
                # h_trace is meaningless for an empirically-costed unit; the
                # allocator consumes predicted_dloss directly. 0.0 marks
                # "do not fall back to h_trace x weight_mse" for this row.
                "h_trace": 0.0,
                "n_params": npm,
                "in_features": in_features,
                "out_features": out_features,
                **expert_metadata,
                "n_probes": 0,
            }
            row: dict = {}
            for fmt in measured_fmts:
                # Split the UNIT cost across members by n_params so the
                # per-member sum re-assembles exactly one unit KL.
                row[fmt] = {
                    "predicted_dloss": kls[fmt] * npm / n_params_unit,
                    "cost_source": "empirical_unit_kl",
                    "output_mse_measured": False,
                }
            for fmt in menu:
                if fmt in PASSTHROUGH_FORMATS:
                    row[fmt] = {
                        "predicted_dloss": 0.0,
                        "cost_source": "passthrough_zero",
                        "output_mse_measured": False,
                    }
            costs[full] = row
        if progress:
            _log(f"  {qn}: " + "  ".join(
                f"{fmt} unit KL = {kls[fmt]:.4e}" for fmt in measured_fmts)
                + f"  (n_params={n_params_unit / 1e6:.0f}M, "
                  f"experts={num_experts})")
    return stats, costs, unit_kls


EXPERT_CHECKPOINT_IDENTITY_SCHEMA = (
    "prismaquant.expert_empirical_checkpoint.identity.v1"
)
EXPERT_CHECKPOINT_STAGE = "expert empirical cost"


def _streamed_expert_unit_records(model, profile, *, unit_filter, max_units):
    """Return serving units in the resident measurer's exact order."""
    from prismaquant.sensitivity_probe import (
        _is_packed_experts_module,
        _packed_experts_param_names,
    )

    packed = [
        ("packed", qname, mod)
        for qname, mod in model.named_modules()
        if _is_packed_experts_module(mod, profile)
    ]
    unpacked = [
        ("unpacked", unit.qname, unit)
        for unit in _unpacked_expert_units(model, profile)
    ]
    records = packed + unpacked
    if unit_filter:
        pattern = re.compile(str(unit_filter))
        records = [record for record in records if pattern.search(record[1])]
    if max_units > 0:
        records = records[:max_units]
    identities: list[dict[str, object]] = []
    for kind, qname, storage in records:
        if kind == "packed":
            members = []
            for param_name in _packed_experts_param_names(storage, profile):
                tensor = getattr(storage, param_name)
                members.append({
                    "qname": f"{qname}.{param_name}" if qname else param_name,
                    "shape": [int(dim) for dim in tensor.shape],
                    "dtype": str(tensor.dtype),
                })
        else:
            members = [
                {
                    "qname": member.qname,
                    "shape": [
                        int(dim) for dim in member.module.weight.shape
                    ],
                    "dtype": str(member.module.weight.dtype),
                }
                for member in storage.members
            ]
        identities.append({
            "qname": str(qname),
            "storage": str(kind),
            "members": members,
        })
    return records, identities


def _expert_checkpoint_identity(
    *,
    runner,
    profile,
    calib_ids: torch.Tensor,
    formats: Sequence[str],
    col_weights: Mapping[str, torch.Tensor],
    unit_identities: Sequence[Mapping[str, object]],
    model_identity: Mapping[str, object],
    expert_chunk: int,
    expert_sample: int,
    max_units: int,
    unit_filter: str | None,
    identity_extra: Mapping[str, object] | None,
) -> dict[str, object]:
    from prismaquant.cost_stage_checkpoint import canonical_json
    from prismaquant.perturbed_x_cache import calibration_data_hash
    from prismaquant.production_weight_cache import (
        _production_cache_source_sha256,
    )

    calib = calib_ids.detach().to("cpu").contiguous()
    from prismaquant.cost_streaming import validate_streamed_model_identity

    exact_model_identity = validate_streamed_model_identity(
        model_identity, where="expert empirical checkpointing"
    )
    identity = {
        "schema": EXPERT_CHECKPOINT_IDENTITY_SCHEMA,
        "git_commit": _git_commit(),
        "producer_source_sha256": _production_cache_source_sha256(),
        "model": exact_model_identity,
        "profile": {
            "module": type(profile).__module__,
            "class": type(profile).__qualname__,
        },
        "calibration": {
            "shape": [int(dim) for dim in calib.shape],
            "dtype": str(calib.dtype),
            "sha256": hashlib.sha256(
                calib.view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
            "calib_hash": calibration_data_hash(calib_ids),
        },
        "formats": [str(fmt) for fmt in formats],
        "units": [dict(record) for record in unit_identities],
        "imatrix": {
            str(name): _tensor_value_stamp(value)
            for name, value in sorted(col_weights.items())
        },
        "measurement_dtype": str(runner.dtype),
        "expert_chunk": int(expert_chunk),
        "calib_batch": int(_calib_batch()),
        "expert_sample": int(expert_sample),
        "max_units": int(max_units),
        "unit_filter": unit_filter,
        "extra": dict(identity_extra or {}),
    }
    return canonical_json(identity, where="expert empirical checkpoint identity")


def _write_expert_unit_checkpoint(
    root: Path,
    *,
    qname: str,
    identity_sha256: str,
    state: Mapping[str, object],
) -> None:
    """Patchable publication seam used by interruption tests."""
    from prismaquant.cost_stage_checkpoint import write_unit

    write_unit(
        root,
        stage=EXPERT_CHECKPOINT_STAGE,
        qname=qname,
        identity_sha256=identity_sha256,
        state=state,
    )


def _validate_expert_checkpoint_state(
    qname: str, state: Mapping[str, object]
) -> dict[str, object]:
    expected = {"stats", "costs", "unit_kls"}
    if set(state) != expected:
        raise RuntimeError(
            f"expert unit checkpoint for {qname} has fields "
            f"{sorted(state)!r}, expected {sorted(expected)!r}; refusing "
            "reuse or recompute"
        )
    for field in expected:
        if not isinstance(state[field], Mapping):
            raise RuntimeError(
                f"expert unit checkpoint for {qname} has invalid {field}; "
                "refusing reuse or recompute"
            )
    unit_kls = state["unit_kls"]
    if set(unit_kls) != {qname}:
        raise RuntimeError(
            f"expert unit checkpoint for {qname} carries unit_kls keys "
            f"{sorted(unit_kls)!r}; refusing reuse or recompute"
        )
    return dict(state)


def measure_expert_unit_costs_streamed(
    runner,
    profile,
    calib_ids: torch.Tensor,
    formats: Sequence[str],
    *,
    expert_chunk: int = 16,
    progress: bool = True,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    expert_sample: int = 0,
    max_units: int = 0,
    unit_filter: str | None = None,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    model_identity: Mapping[str, object] | None = None,
    checkpoint_identity_extra: Mapping[str, object] | None = None,
) -> tuple[dict, dict, dict]:
    """Measure routed serving units with one decoder layer resident at once.

    The numerical core is the resident ``measure_expert_unit_costs`` function:
    identical qdq, fp32 log-softmax/KL, window order, and row builder.
    Streaming changes only model residency.  A target layer stays pinned for
    the complete serving unit so its temporary qdq is restored before the
    context can cache/unload it.
    """
    if resume and checkpoint_dir is None:
        raise ValueError("resume=True requires checkpoint_dir")
    profile = resolve_routed_expert_profile(runner.model, profile)
    menu = _canon_formats(formats)
    weights = {
        str(name): torch.as_tensor(value)
        for name, value in dict(col_weights or {}).items()
    }
    records, unit_identities = _streamed_expert_unit_records(
        runner.model,
        profile,
        unit_filter=unit_filter,
        max_units=max_units,
    )
    qnames = [str(record[1]) for record in records]
    if len(qnames) != len(set(qnames)):
        raise RuntimeError(
            "streamed expert discovery produced duplicate serving-unit qnames"
        )
    # Every serving unit is layer-local by serving contract. Resolve this
    # before opening the journal so malformed topology cannot acquire state.
    for qname in qnames:
        runner.layer_index_for_qname(qname)

    menu_by_unit = {qname: menu for qname in qnames}

    completed: dict[str, dict[str, object]] = {}
    journal_root: Path | None = None
    journal_identity_sha256: str | None = None
    if checkpoint_dir is not None:
        if model_identity is None:
            raise RuntimeError(
                "streamed expert checkpointing requires exact model_identity; "
                "refusing model-name-gated resume"
            )
        extra = dict(checkpoint_identity_extra or {})
        identity = _expert_checkpoint_identity(
            runner=runner,
            profile=profile,
            calib_ids=calib_ids,
            formats=menu,
            col_weights=weights,
            unit_identities=unit_identities,
            model_identity=model_identity,
            expert_chunk=expert_chunk,
            expert_sample=expert_sample,
            max_units=max_units,
            unit_filter=unit_filter,
            identity_extra=extra,
        )
        from prismaquant.cost_stage_checkpoint import prepare_journal

        journal_root, journal_identity_sha256, raw_completed = prepare_journal(
            checkpoint_dir,
            stage=EXPERT_CHECKPOINT_STAGE,
            resume=resume,
            identity=identity,
            qnames=qnames,
        )
        completed = {
            qname: _validate_expert_checkpoint_state(qname, state)
            for qname, state in raw_completed.items()
        }

    pending = [qname for qname in qnames if qname not in completed]
    baseline = None
    if pending:
        # Resume identity was validated above, before this first forward.
        baseline = _baseline_logprobs(
            runner.model,
            calib_ids,
            forward_model=runner,
        )

    results: dict[str, dict[str, object]] = dict(completed)
    t_units0 = time.time()
    for u_i, qname in enumerate(pending):
        if progress:
            _log(f"streamed unit {qname} ({u_i + 1}/{len(pending)})")
        t_unit0 = time.time()
        exact_filter = rf"\A{re.escape(qname)}\Z"
        unit_menu = menu_by_unit[qname]
        with runner.pin_layer_for_qname(qname):
            unit_stats, unit_costs, unit_kls = measure_expert_unit_costs(
                runner.model,
                profile,
                calib_ids,
                unit_menu,
                expert_chunk=expert_chunk,
                progress=False,
                col_weights=weights,
                expert_sample=expert_sample,
                max_units=0,
                unit_filter=exact_filter,
                forward_model=runner,
                baseline_logprobs=baseline,
            )
        if set(unit_kls) != {qname}:
            raise RuntimeError(
                f"streamed expert unit {qname} produced unit_kls keys "
                f"{sorted(unit_kls)!r}"
            )
        state = {
            "stats": unit_stats,
            "costs": unit_costs,
            "unit_kls": unit_kls,
        }
        results[qname] = state
        if journal_root is not None:
            assert journal_identity_sha256 is not None
            _write_expert_unit_checkpoint(
                journal_root,
                qname=qname,
                identity_sha256=journal_identity_sha256,
                state=state,
            )
        if progress:
            dt = time.time() - t_unit0
            rate = (time.time() - t_units0) / (u_i + 1)
            eta_min = rate * (len(pending) - u_i - 1) / 60.0
            _log(f"unit {qname} done in {dt:.0f}s "
                 f"({sorted(unit_kls[qname])}); "
                 f"~{eta_min:.0f} min left for {len(pending) - u_i - 1} units")

    stats: dict = {}
    costs: dict = {}
    unit_kls: dict = {}
    for qname in qnames:
        state = results[qname]
        stats.update(state["stats"])
        costs.update(state["costs"])
        unit_kls.update(state["unit_kls"])
    return stats, costs, unit_kls


@torch.no_grad()
def _fork_quantized_unit(
    runner,
    batch,
    layer: int,
    unit_kind: str,
    storage,
    qn: str,
    fmt: str,
    *,
    profile,
    expert_chunk: int,
    col_weights: Mapping[str, torch.Tensor],
    rows_meta: dict,
) -> torch.Tensor:
    """One fork step: run this unit's layer with an export-equivalent
    quantized COPY of its expert weights on the baseline boundary input.

    The live module is never mutated — each expert Parameter attribute is
    swapped for the quantized copy for exactly one layer call and the
    ORIGINAL Parameter object is reattached in ``finally`` (so the
    streaming cache's install/unload bookkeeping sees the same objects).
    The only transient is the quantized copy itself (~one unit of expert
    mass); there is no restore clone.

    Also captures ``rows_meta[qn]`` (member names/shapes/metadata) on the
    first call, while the layer is resident.
    """
    from prismaquant.sensitivity_probe import _packed_experts_param_names

    device = runner.device
    swaps: list[tuple] = []
    holder = nn.Module()
    if unit_kind == "packed":
        mod = storage
        pnames = list(_packed_experts_param_names(mod, profile))
        for pn in pnames:
            live = getattr(mod, pn)
            if not bool((live != 0).any()):
                raise RuntimeError(
                    f"{qn}.{pn}: packed-expert stack is ALL ZERO at fork "
                    "time — the zero-expert calibration bug; a unit KL "
                    "measured now would read exactly 0 for every format"
                )
            setattr(holder, pn, nn.Parameter(
                live.data.clone(), requires_grad=False))
        _quantize_unit_inplace(
            holder, pnames, fmt,
            expert_chunk=expert_chunk, col_weights=col_weights,
            unit_qname=qn,
        )
        if qn not in rows_meta:
            num_experts = int(getattr(mod, pnames[0]).shape[0])
            members = []
            for pn in pnames:
                shape = list(getattr(mod, pn).shape)
                members.append((
                    f"{qn}.{pn}" if qn else pn,
                    int(getattr(mod, pn).numel()),
                    int(shape[2]),
                    int(shape[1]),
                    {
                        "num_experts": num_experts,
                        "_packed_experts_module": qn,
                        "_packed_param": pn,
                    },
                ))
            rows_meta[qn] = {
                "n_params_unit": sum(m[1] for m in members),
                "members": members,
            }
        for pn in pnames:
            swaps.append((mod, pn, getattr(mod, pn),
                          getattr(holder, pn)))
    else:
        unit = storage
        packed = _virtual_packed_module(unit)
        param_names = [parent for parent, _projections in unit.roles]
        _quantize_unit_inplace(
            packed, param_names, fmt,
            expert_chunk=expert_chunk, col_weights=col_weights,
            unit_qname=qn,
        )
        for parent, _projections in unit.roles:
            setattr(holder, parent, getattr(packed, parent))
            delattr(packed, parent)
        del packed
        if qn not in rows_meta:
            members = [(
                member.qname,
                int(member.module.weight.numel()),
                int(member.module.weight.shape[1]),
                int(member.module.weight.shape[0]),
                {"_unpacked_expert_unit": qn},
            ) for member in unit.members]
            rows_meta[qn] = {
                "n_params_unit": sum(m[1] for m in members),
                "members": members,
            }
        by_key = {
            (member.projection_name, member.expert_id): member
            for member in unit.members
        }
        for parent, projections in unit.roles:
            stack = getattr(holder, parent)
            offset = 0
            for projection in projections:
                width = int(by_key[(projection, 0)].module.weight.shape[0])
                for expert_id in range(unit.num_experts):
                    mod = by_key[(projection, expert_id)].module
                    swaps.append((
                        mod, "weight", mod.weight,
                        nn.Parameter(
                            stack[expert_id, offset:offset + width],
                            requires_grad=False),
                    ))
                offset += width
    try:
        for mod, attr, _orig, q in swaps:
            setattr(mod, attr, q if isinstance(q, nn.Parameter)
                    else nn.Parameter(q, requires_grad=False))
        fork_h = runner.isolated_layer(
            batch, layer,
            batch.activations_cpu[layer].to(device),
            pass_state=None,
        )
    finally:
        for mod, attr, orig, _q in swaps:
            setattr(mod, attr, orig)
    del swaps, holder
    return fork_h


@torch.no_grad()
def measure_expert_unit_costs_forked(
    runner,
    profile,
    calib_ids: torch.Tensor,
    formats: Sequence[str],
    *,
    expert_chunk: int = 16,
    progress: bool = True,
    col_weights: Mapping[str, torch.Tensor] | None = None,
    max_units: int = 0,
    unit_filter: str | None = None,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    model_identity: Mapping[str, object] | None = None,
    checkpoint_identity_extra: Mapping[str, object] | None = None,
) -> tuple[dict, dict, dict]:
    """Forked-stream expert unit KLs: O(1) body streams instead of O(units).

    The window-major streamed driver re-streams the whole body for EVERY
    forward (a 306 GB source at 16 windows = ~1.1 TB of disk per unit,
    ~47 TB for a 42-unit run — measured 2026-08-26 on GLM-5.3-Flash, GPU
    idling at ~25% while the nvme pinned at 2 GB/s). This driver streams
    the body THREE times total: one boundary-capture pass (baseline
    activations at every decoder boundary + baseline logits), then one
    pass per measured format in which every unit's quantized stream is
    forked from its layer's baseline boundary and advanced through the
    live layers together with all other forks.

    Semantics per unit are identical to the window-major path: exactly one
    unit quantized (export-equivalent packed render), everything else at
    source precision, KL(BF16 || quantized) from fp32 log-softmax at the
    head. Downstream route-flips propagate through the fork's own suffix,
    so the empirical route-flip floor is preserved. Differences are
    reassociation-class only (all windows forward in one batch instead of
    ``_calib_batch()`` chunks).

    The live model is NEVER mutated: the fork's layer call swaps each
    member Linear's ``weight`` Parameter for a view into the quantized
    packed stacks and reattaches the ORIGINAL Parameter objects in
    ``finally`` — no restore clone exists, so the triple-expert-mass peak
    that wedged the window-major runs (resident + clone + packed) cannot
    recur; the transient is the packed render alone.

    Fail-closed scope (v1): profiles with per-pass layer state and expert
    subsampling are refused — use the window-major driver for those. Packed
    and unpacked units are both handled; the packed all-zero-stack guard
    fires at fork time (the unpacked variant relies on the window driver's
    discovery-time check if you re-enable it there).
    Checkpoint granularity is the whole run: per-unit rows are journaled
    only after every format pass completes, so a mid-pass crash re-pays
    the passes (~1 h at GLM scale), never the journal identity.
    """
    if profile.new_forward_pass_state() != {}:
        raise RuntimeError(
            "forked expert eval requires stateless layer passes; profile "
            f"{type(profile).__name__} declares per-pass state — use the "
            "window-major streamed driver"
        )
    profile = resolve_routed_expert_profile(runner.model, profile)
    menu = _canon_formats(formats)
    measured_fmts = [f for f in menu if f not in PASSTHROUGH_FORMATS]
    weights = {
        str(name): torch.as_tensor(value)
        for name, value in dict(col_weights or {}).items()
    }
    records, unit_identities = _streamed_expert_unit_records(
        runner.model,
        profile,
        unit_filter=unit_filter,
        max_units=max_units,
    )
    for kind, qname, _storage in records:
        if kind not in ("packed", "unpacked"):
            raise RuntimeError(
                f"forked expert eval got unknown unit kind {kind!r} for "
                f"{qname}"
            )
    qnames = [str(record[1]) for record in records]
    if len(qnames) != len(set(qnames)):
        raise RuntimeError(
            "forked expert discovery produced duplicate serving-unit qnames"
        )
    unit_by_qname = {str(qn): unit for _kind, qn, unit in records}
    kind_by_qname = {str(qn): kind for kind, qn, _unit in records}
    layer_of = {qn: runner.layer_index_for_qname(qn) for qn in qnames}

    completed: dict[str, dict[str, object]] = {}
    journal_root: Path | None = None
    journal_identity_sha256: str | None = None
    if checkpoint_dir is not None:
        if model_identity is None:
            raise RuntimeError(
                "forked expert checkpointing requires exact model_identity; "
                "refusing model-name-gated resume"
            )
        extra = dict(checkpoint_identity_extra or {})
        if "eval_driver" in extra:
            raise ValueError(
                "checkpoint_identity_extra cannot override eval_driver"
            )
        extra["eval_driver"] = "forked-stream/1"
        identity = _expert_checkpoint_identity(
            runner=runner,
            profile=profile,
            calib_ids=calib_ids,
            formats=menu,
            col_weights=weights,
            unit_identities=unit_identities,
            model_identity=model_identity,
            expert_chunk=expert_chunk,
            expert_sample=0,
            max_units=max_units,
            unit_filter=unit_filter,
            identity_extra=extra,
        )
        from prismaquant.cost_stage_checkpoint import prepare_journal

        journal_root, journal_identity_sha256, raw_completed = prepare_journal(
            checkpoint_dir,
            stage=EXPERT_CHECKPOINT_STAGE,
            resume=resume,
            identity=identity,
            qnames=qnames,
        )
        completed = {
            qname: _validate_expert_checkpoint_state(qname, state)
            for qname, state in raw_completed.items()
        }
    pending = [qn for qn in qnames if qn not in completed]

    device = runner.device
    num_layers = runner.num_layers

    def _avail_swap_gb() -> tuple[float, float]:
        vals = io_spans.read_meminfo()
        swap = (vals.get("SwapTotal", 0) - vals.get("SwapFree", 0)) / 1024 ** 3
        return vals.get("MemAvailable", 0) / 1024 ** 3, swap

    unit_kl_means: dict[str, dict[str, float]] = {qn: {} for qn in pending}
    unit_kl_windows: dict[str, dict[str, list[float]]] = {
        qn: {} for qn in pending
    }
    # Row-shape metadata captured at fork time (first format pass), while
    # the unit's layer is resident — by row-building time it is unloaded.
    rows_meta: dict[str, dict[str, object]] = {}
    baseline_lp: list[torch.Tensor] | None = None
    batch = None

    if pending:
        n_windows = int(calib_ids.shape[0])
        _log(f"forked driver: {len(pending)}/{len(qnames)} units pending, "
             f"formats {measured_fmts}, {n_windows} windows in one batch")
        t0 = time.time()
        batch = runner.capture_boundaries(calib_ids)
        avail, swap = _avail_swap_gb()
        _log(f"boundary pass done in {time.time() - t0:.0f}s "
             f"({len(batch.activations_cpu)} boundaries, "
             f"avail {avail:.0f}G, swap {swap:.1f}G)")
        logits = runner.tail_logits(
            batch, batch.activations_cpu[-1].to(device)
        )
        baseline_lp = [
            F.log_softmax(logits[w].float(), dim=-1).cpu()
            for w in range(n_windows)
        ]
        del logits
        torch.cuda.empty_cache()

        for f_i, fmt in enumerate(measured_fmts):
            t_pass0 = time.time()
            forks: dict[str, torch.Tensor] = {}
            for depth in range(runner.prefetch_lookahead):
                runner.context.schedule_prefetch(depth)
            for layer in range(num_layers):
                runner.context.install(
                    layer,
                    require_prefetched=runner.require_prefetched_residency,
                )
                runner.context.schedule_prefetch(
                    layer + runner.prefetch_lookahead
                )
                try:
                    # Advance every existing fork through this live layer
                    # BEFORE forking at it, so a new fork's hidden is not
                    # double-advanced.
                    for qn in list(forks):
                        forks[qn] = runner.isolated_layer(
                            batch, layer, forks[qn], pass_state=None
                        )
                    for qn in pending:
                        if layer_of[qn] != layer:
                            continue
                        forks[qn] = _fork_quantized_unit(
                            runner,
                            batch,
                            layer,
                            kind_by_qname[qn],
                            unit_by_qname[qn],
                            qn,
                            fmt,
                            profile=profile,
                            expert_chunk=expert_chunk,
                            col_weights=weights,
                            rows_meta=rows_meta,
                        )
                        torch.cuda.empty_cache()
                finally:
                    runner.context.unload(layer)
                if progress and (
                    layer % 5 == 4 or layer == num_layers - 1
                ):
                    avail, swap = _avail_swap_gb()
                    rate = (time.time() - t_pass0) / (layer + 1)
                    eta_min = rate * (num_layers - layer - 1) / 60.0
                    _log(f"fork pass {fmt} ({f_i + 1}/"
                         f"{len(measured_fmts)}): layer "
                         f"{layer + 1}/{num_layers}, {len(forks)} forks "
                         f"live, avail {avail:.0f}G swap {swap:.1f}G, "
                         f"~{eta_min:.0f} min left in pass")
            for qn in pending:
                logits = runner.tail_logits(batch, forks.pop(qn))
                total = 0.0
                n_tok = 0
                windows: list[float] = []
                for w in range(n_windows):
                    lp = F.log_softmax(logits[w].float(), dim=-1)
                    bl = baseline_lp[w].to(lp.device)
                    kl = (bl.exp() * (bl - lp)).sum(-1)
                    wsum = float(kl.sum().item())
                    total += wsum
                    n_tok += kl.numel()
                    windows.append(wsum / max(kl.numel(), 1))
                    del lp, bl, kl
                del logits
                unit_kl_means[qn][fmt] = total / max(n_tok, 1)
                unit_kl_windows[qn][fmt] = windows
            torch.cuda.empty_cache()
            _log(f"fork pass {fmt} done in "
                 f"{(time.time() - t_pass0) / 60.0:.1f} min")

    stats: dict = {}
    costs: dict = {}
    unit_kls: dict = {}
    results: dict[str, dict[str, object]] = dict(completed)
    for qn in pending:
        kls = unit_kl_means[qn]
        # Row construction mirrors measure_expert_unit_costs field for
        # field — the merge and the allocator must not be able to tell the
        # drivers apart. Shapes come from rows_meta (captured at fork time,
        # while the layer was resident).
        meta = rows_meta[qn]
        n_params_unit = int(meta["n_params_unit"])
        unit_stats: dict = {}
        unit_costs: dict = {}
        for full, npm, in_features, out_features, member_meta in (
            meta["members"]
        ):
            unit_stats[full] = {
                "h_trace": 0.0,
                "n_params": npm,
                "in_features": in_features,
                "out_features": out_features,
                **member_meta,
                "n_probes": 0,
            }
            row: dict = {}
            for fmt in measured_fmts:
                row[fmt] = {
                    "predicted_dloss": kls[fmt] * npm / n_params_unit,
                    "cost_source": "empirical_unit_kl",
                    "output_mse_measured": False,
                }
            for fmt in menu:
                if fmt in PASSTHROUGH_FORMATS:
                    row[fmt] = {
                        "predicted_dloss": 0.0,
                        "cost_source": "passthrough_zero",
                        "output_mse_measured": False,
                    }
            unit_costs[full] = row
        state = {
            "stats": unit_stats,
            "costs": unit_costs,
            "unit_kls": {qn: dict(kls)},
            "kl_windows": {
                fmt: list(vals)
                for fmt, vals in unit_kl_windows[qn].items()
            },
        }
        results[qn] = state
        if journal_root is not None:
            assert journal_identity_sha256 is not None
            _write_expert_unit_checkpoint(
                journal_root,
                qname=qn,
                identity_sha256=journal_identity_sha256,
                state=state,
            )
        if progress:
            _log(f"  {qn}: " + "  ".join(
                f"{fmt} unit KL = {kls[fmt]:.4e}"
                for fmt in measured_fmts))
    for qn in qnames:
        state = results[qn]
        stats.update(state["stats"])
        costs.update(state["costs"])
        unit_kls.update(state["unit_kls"])
    return stats, costs, unit_kls


def merge_cost_payloads(
    base: Mapping[str, object],
    expert_stats: Mapping[str, object],
    expert_costs: Mapping[str, object],
    *,
    formats: Sequence[str],
    replace_experts: bool = False,
) -> dict:
    """Union base non-expert rows with empirical expert rows.

    AURA lane (``replace_experts=False``): collisions are an error —
    aura_cost must have been run with ``--allow-packed-expert-omission``
    (its guard fail-fasts otherwise), so no name may be costed by both
    estimators.

    Replace semantics (``replace_experts=True``): a COST_MODE=local base
    payload DOES cost the expert stacks (smoothly — route-flip-blind); those
    rows are REPLACED by the empirical ones and recorded in provenance,
    non-expert rows stay untouched.
    """
    merged = dict(base)
    base_stats = dict(base.get("stats", {}) or {})
    base_costs = dict(base.get("costs", {}) or {})
    overlap = set(base_costs) & set(expert_costs)
    if overlap and not replace_experts:
        raise RuntimeError(
            f"hybrid merge collision: {len(overlap)} names costed by BOTH "
            f"the base payload and the expert empirical pass (e.g. "
            f"{sorted(overlap)[:3]}). The base run must omit routed experts "
            f"(or pass replace_experts for the replace semantics).")
    canonical_formats = _canon_formats(formats)
    if overlap:
        for name in overlap:
            base_costs.pop(name)
            base_stats.pop(name, None)
        prov = dict(merged.get("provenance", {}) or {})
        prov["replaced_smooth_expert_rows"] = sorted(overlap)
        merged["provenance"] = prov
    base_stats.update(expert_stats)
    base_costs.update(expert_costs)
    merged["stats"] = base_stats
    merged["costs"] = base_costs
    merged["schema"] = SCHEMA
    merged["formats"] = canonical_formats
    return merged


def backfill_missing_from_base(
    payload: dict,
    base_cost: Mapping[str, object],
) -> list[str]:
    """Copy rows for names the payload lacks from the baseline cost pkl.

    Covers MTP / visual sidecars the AURA pass never sees (the synthesized
    MTP module lives outside the CausalLM the cost harness loads). Returns
    the backfilled names, and records them in provenance for honesty: these
    rows carry the baseline estimator, not the AURA adjoint.
    """
    base_costs = dict(base_cost.get("costs", {}) or {})
    base_stats = dict(base_cost.get("stats", {}) or {})
    added: list[str] = []
    for name, row in base_costs.items():
        if name in payload["costs"]:
            continue
        payload["costs"][name] = row
        if name in base_stats and name not in payload["stats"]:
            payload["stats"][name] = base_stats[name]
        added.append(name)
    return sorted(added)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Empirical routed-MoE expert cost (+ hybrid merge)")
    p.add_argument("--model", required=True)
    p.add_argument("--cost-mode", default="",
                   help="Pipeline COST_MODE stamped into "
                        "provenance['cost_mode'] (re-vet R2).")
    p.add_argument("--output", required=True)
    p.add_argument(
        "--formats", default="NVFP4,FP8_DYNAMIC,BF16",
        help="Expert format menu. Non-passthrough formats are measured; "
        "BF16/FP8_SOURCE rows are passthrough-zero.")
    p.add_argument("--n-calib-samples", type=int, default=16)
    p.add_argument("--calib-seqlen", type=int, default=512)
    p.add_argument("--calib-split", default="train")
    p.add_argument("--calib-seed", type=int, default=42)
    p.add_argument(
        "--dataset", default=None,
        help="Optional calibration source (HF id, .jsonl, .txt) via "
        "sensitivity_probe.load_calibration; default is the WikiText "
        "windowed loader (matches aura_cost).")
    p.add_argument("--expert-chunk", type=int, default=16,
                   help="Experts quantized per in-place RTN chunk.")
    p.add_argument(
        "--merge-base", default=None,
        help="AURA non-expert cost pkl to union the expert rows into "
        "(the hybrid recipe). Output = merged payload.")
    p.add_argument(
        "--backfill-base", default=None,
        help="Baseline incremental cost pkl; rows for names still missing "
        "after the merge (MTP/visual sidecars) are copied from it.")
    p.add_argument(
        "--replace-experts", action="store_true",
        help="Replace merge semantics: a COST_MODE=local base payload "
        "costs expert stacks smoothly (route-flip-blind); REPLACE those "
        "rows with the empirical ones (recorded in provenance) instead of "
        "treating the collision as an error.")
    p.add_argument(
        "--col-weights", default=None,
        help="Pickle {qname: per-input-column importance} (the exporter's "
        "imatrix), bound into the checkpoint identity.")
    p.add_argument(
        "--expert-sample", type=int, default=0,
        help="Quantize only a stratified subsample of N experts per unit and "
        "extrapolate the unit KL by expert count. The bf16 unit KL is a "
        "perturbation floor (measured on the retired codebook lane, archived "
        "2026-09-25, #1304: S=1 of 256 already read ~90%% of the full-stack "
        "KL, so count-scaling over-predicted ~10x), so this suits only "
        "floor-regime probing and coarse-format menus where unit KLs sit far "
        "above the floor. 0 = full stack (default). The LOCAL cost's "
        "PRISMAQUANT_EXPERT_COST_SAMPLE is the unbiased alternative (MSE "
        "sampling is unbiased; KL sampling is not).")
    p.add_argument(
        "--max-units", type=int, default=0,
        help="Measure only the first N units (0 = all). Validation/"
        "sharding aid.")
    p.add_argument(
        "--unit-filter", default=None,
        help="Regex on the experts qname; only matching units are "
             "measured. Validation/sharding aid.")
    p.add_argument(
        "--checkpoint-dir", default=None,
        help="Durable identity-bound per-serving-unit checkpoint directory.")
    p.add_argument(
        "--resume", action="store_true",
        help="Validate and reuse completed qname-keyed unit shards; any "
             "identity mismatch refuses reuse and recomputation.")
    p.add_argument(
        "--streaming", action="store_true",
        help="Use the existing decoder-layer prefetch/cache context instead "
             "of loading the expanded source model resident.")
    p.add_argument(
        "--streaming-offload-dir", default=None,
        help="Streaming model work directory. Defaults below the checkpoint "
             "or output directory and is never placed in /tmp.")
    p.add_argument("--device", default="cuda")
    return p


def _streaming_model_identity(
    runner,
    source_model: str,
    *,
    identity_cache_path: str | Path | None = None,
) -> dict[str, object]:
    """Stable source manifest used to bind expert end-to-end KL resumes."""
    from prismaquant.cost_streaming import build_streamed_model_identity

    return build_streamed_model_identity(
        runner,
        source_model,
        identity_cache_path=identity_cache_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.resume and not args.checkpoint_dir:
        raise SystemExit("--resume requires --checkpoint-dir")

    from prismaquant.gpu_guard import require_cuda_hot_path
    require_cuda_hot_path("expert_empirical_cost", args.device)

    import os
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prismaquant.build_rtn_cache import stage_multimodal
    from prismaquant.model_profiles import detect_profile

    staged, _cleanup = stage_multimodal(args.model)
    # Detection is the authority for routed-expert membership and also
    # installs any architecture-owned vendored modelling override. Resolve it
    # before constructing the model; a fallback-after-load would recreate the
    # silent wrong-topology failure this empirical path exists to prevent.
    profile = detect_profile(staged)
    local_only = Path(staged).exists()
    tok = AutoTokenizer.from_pretrained(
        staged, trust_remote_code=True, local_files_only=local_only)
    _log(f"loading {args.model} (staged={staged}) bf16 ...")
    model = None
    streamed_runner = None
    if args.streaming:
        from prismaquant.cost_streaming import build_streamed_causal_lm

        offload_dir = args.streaming_offload_dir
        if not offload_dir:
            anchor = Path(args.checkpoint_dir or args.output).parent
            offload_dir = str(anchor / "expert-cost-streaming-offload")
        streamed_runner = build_streamed_causal_lm(
            staged,
            device=torch.device(args.device),
            dtype=torch.bfloat16,
            offload_folder=offload_dir,
            profile=profile,
        )
        model = streamed_runner.model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            staged, dtype=torch.bfloat16, trust_remote_code=True,
            local_files_only=local_only, attn_implementation="eager",
            device_map=args.device,
        ).eval()
        for prm in model.parameters():
            prm.requires_grad_(False)
    # Per-expert-on-disk -> packed-in-class checkpoints (Qwen3.5-MoE /
    # Ornith): the text class lacks the per-expert->packed mapper, so the
    # packed params load ZERO-INITIALIZED — the zero-expert calibration bug
    # (quantizing zeros is a no-op; every unit KL reads exactly 0). Fill from
    # the source shards; measure_expert_unit_costs also hard-fails on
    # all-zero stacks so this class of silent garbage can never recur.
    from prismaquant.layer_streaming import fill_packed_experts_from_source
    # NOTE: pass the ORIGINAL model dir, not the staged text-only view — the
    # staging rewrites index keys to `model.layers.*` while the profile's
    # source_tensor_name keeps the checkpoint's own nesting
    # (`model.language_model.*`), so against the staged index the fill's
    # prefix filter silently matches nothing (how production_recache calls it).
    fill_src = args.model if Path(args.model).exists() else staged
    filled = (
        0
        if streamed_runner is not None
        else fill_packed_experts_from_source(
            model, fill_src, profile, progress=True
        )
    )
    if filled:
        _log(f"filled {filled} packed-expert params from source shards")

    if args.dataset:
        from prismaquant.sensitivity_probe import load_calibration
        calib = load_calibration(
            tok, args.dataset, args.n_calib_samples, args.calib_seqlen,
            calib_seed=args.calib_seed)
    else:
        from prismaquant.calibration_data import (
            load_wikitext_calibration_windowed,
        )
        calib = load_wikitext_calibration_windowed(
            tok, args.n_calib_samples, args.calib_seqlen,
            split=args.calib_split, seed=args.calib_seed)
    calib = calib.to(args.device)

    formats = _canon_formats(
        [f for f in args.formats.split(",") if f.strip()])
    col_weights = None
    if args.col_weights:
        with open(args.col_weights, "rb") as fh:
            col_weights = {k: torch.as_tensor(v)
                           for k, v in pickle.load(fh).items()}
    if col_weights is None:
        col_weights = {}
    if streamed_runner is not None:
        try:
            streamed_model_identity = None
            if args.checkpoint_dir:
                streamed_model_identity = _streaming_model_identity(
                    streamed_runner,
                    args.model,
                    identity_cache_path=(
                        Path(args.checkpoint_dir)
                        / "streamed_model_identity.json"
                    ),
                )
            eval_driver = os.environ.get(
                "PRISMAQUANT_EXPERT_EVAL_DRIVER", "window"
            ).strip().lower()
            if eval_driver not in ("window", "forked"):
                raise SystemExit(
                    f"PRISMAQUANT_EXPERT_EVAL_DRIVER={eval_driver!r} is not "
                    "one of: window, forked"
                )
            if eval_driver == "forked":
                # O(1) body streams instead of O(units): see
                # measure_expert_unit_costs_forked. Scope guards live in the
                # driver (stateless pass, unpacked units); the CLI features
                # it does not take are refused here, loudly.
                refused = {
                    "--expert-sample": args.expert_sample,
                }
                on = sorted(k for k, v in refused.items() if v)
                if on:
                    raise SystemExit(
                        "PRISMAQUANT_EXPERT_EVAL_DRIVER=forked does not "
                        f"support {on}; unset them or use the window driver"
                    )
                stats, costs, unit_kls = measure_expert_unit_costs_forked(
                    streamed_runner,
                    profile,
                    calib,
                    formats,
                    expert_chunk=args.expert_chunk,
                    col_weights=col_weights,
                    max_units=args.max_units,
                    unit_filter=args.unit_filter,
                    checkpoint_dir=args.checkpoint_dir,
                    resume=args.resume,
                    model_identity=streamed_model_identity,
                )
            else:
                stats, costs, unit_kls = measure_expert_unit_costs_streamed(
                    streamed_runner,
                    profile,
                    calib,
                    formats,
                    expert_chunk=args.expert_chunk,
                    col_weights=col_weights,
                    expert_sample=args.expert_sample,
                    max_units=args.max_units,
                    unit_filter=args.unit_filter,
                    checkpoint_dir=args.checkpoint_dir,
                    resume=args.resume,
                    model_identity=streamed_model_identity,
                )
        finally:
            streamed_runner.shutdown()
    else:
        stats, costs, unit_kls = measure_expert_unit_costs(
            model, profile, calib, formats, expert_chunk=args.expert_chunk,
            col_weights=col_weights,
            expert_sample=args.expert_sample, max_units=args.max_units,
            unit_filter=args.unit_filter)
    provenance = {
        "schema": SCHEMA,
        "git_commit": _git_commit(),
        "model": args.model,
        "dataset": args.dataset or f"wikitext:{args.calib_split}",
        "n_calib_samples": int(calib.shape[0]),
        "calib_seqlen": int(calib.shape[1]),
        "calib_seed": args.calib_seed,
        "calib_sha256": hashlib.sha256(
            calib.cpu().numpy().tobytes()).hexdigest(),
        "expert_units": len(unit_kls),
        "unit_kls": unit_kls,
        "formats_measured": [
            f for f in formats if f not in PASSTHROUGH_FORMATS],
        "col_weights": args.col_weights,
        "expert_sample": int(args.expert_sample),
        "max_units": int(args.max_units),
        "unit_filter": args.unit_filter,
    }

    if args.merge_base:
        with open(args.merge_base, "rb") as fh:
            base = pickle.load(fh)
        payload = merge_cost_payloads(
            base, stats, costs, formats=formats,
            replace_experts=bool(args.replace_experts))
        prov = dict(payload.get("provenance", {}) or {})
        prov["expert_empirical_cost"] = provenance
        prov["merge_base"] = args.merge_base
        payload["provenance"] = prov
        _log(f"merged {len(costs)} expert member rows into "
             f"{args.merge_base} ({len(payload['costs'])} total)")
    else:
        payload = {
            "schema": SCHEMA,
            "formats": formats,
            "stats": stats,
            "costs": costs,
            "provenance": provenance,
        }

    if args.backfill_base:
        with open(args.backfill_base, "rb") as fh:
            base_cost = pickle.load(fh)
        added = backfill_missing_from_base(payload, base_cost)
        prov = dict(payload.get("provenance", {}) or {})
        prov["backfilled_from_base"] = added
        prov["backfill_base"] = args.backfill_base
        payload["provenance"] = prov
        if added:
            _log(f"backfilled {len(added)} sidecar rows from "
                 f"{args.backfill_base}: {added[:5]}"
                 f"{' ...' if len(added) > 5 else ''}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    _log(f"wrote {out}: {len(payload['costs'])} cost rows "
         f"({len(unit_kls)} expert units)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
