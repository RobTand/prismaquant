"""Compute streamed KL-adjoint costs for each Linear and format.

    predicted_dloss[i, f] = 0.5 * mean_k (<gW_i^(k), dW_i,f>)**2

Each gW is the gradient of a KL-Fisher probe scalar. Each dW is the
production-rendered weight minus the source weight. The CLI renders anchors
from keyed act/ rows through the existing production renderer.

The adjoint keeps one layer's graph and anchors at a time. Per-Linear
checkpoints bind the source, activation rows, render settings, and format plan.
Research callers can supply a ProductionWeightCache or registry RTN deltas.
The output uses the allocator cost schema with output_mse_measured=False.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, MutableMapping
import math
import os
import pickle
import re
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

import prismaquant.format_registry as fr
from prismaquant.kl_fisher import fisher_probe_scalar
from prismaquant.io_spans import mem_available_bytes
from prismaquant.perturbed_x_cache import calibration_data_hash
from prismaquant.routed_experts import (
    PackedExpertProjection,
    ProfileRoutedExpertClassifier,
    profile_declared_packed_expert_projections,
    profile_declared_routed_expert_targets,
    profile_declared_unpacked_expert_linears,
    refresh_packed_expert_projections,
    resolve_routed_expert_profile,
)
from . import aura_checkpoint_journal, aura_provenance
from .digests import bytes_sha256hex
from .fla_config_pin import fla_config_identity
from .joint_price_step import (AuraUnitMeasurements, JointPriceStep, JointResumeContract,
                               ProbeResumePolicy, RequiredAttribution)

SCHEMA = "prismaquant.aura_cost.v1"
AURA_PRODUCTION_ANCHOR_DELTA_CONSUMER_IDENTITY = {
    "schema": "prismaquant.aura.production_anchor_delta_consumer.v1",
    "canonical_input": "production_weight_cache_canonical_cpu_tensor",
    "operation": "fp32_subtract_then_store",
    "subtraction_dtype": "torch.float32",
    "storage_dtype": "torch.bfloat16",
    "output_residency": "source_weight_device",
}


# Source passthrough formats have exact zero weight error.
_ZERO_COST_FORMATS = {
    "BF16",
    "FP8_SOURCE",
    "MXFP4_SOURCE",
}


def _log(msg: str) -> None:
    print(f"[aura {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _free_gib() -> float:
    """Reclaimable-inclusive free memory in GiB.

    On the GB10/DGX Spark unified-memory box, CUDA and host share one physical
    pool, and clean page cache (model safetensors, cache shards) counts as
    'used' in ``torch.cuda.mem_get_info()`` even though the kernel reclaims it
    on demand. ``/proc/meminfo`` ``MemAvailable`` is the true 'can still
    allocate' headroom and is what should gate the watchdog -- gating on CUDA
    free aborts spuriously whenever a large file was just read. Fall back to
    the CUDA figure off-Linux / if /proc is unreadable."""
    try:
        return mem_available_bytes() / (1024 ** 3)
    except Exception:
        pass
    try:
        return torch.cuda.mem_get_info()[0] / (1024 ** 3)
    except Exception:
        return float("inf")


def _release_streamed_anchor_allocator_cache(device: object) -> None:
    """Return retired streamed CUDA allocations to Spark unified memory.

    Dropping the final tensor reference only moves its CUDA allocation into
    PyTorch's reusable cache.  On the single-GPU GB10 path that memory still
    overlaps the much larger FP32 adjoint deltas unless the cache is returned
    to the unified host/device pool before the next allocation phase. Anchor
    execution calls this once per streamed layer; bounded operator replay also
    calls it before reserving each phase. Live tensor owners remain intact.
    Non-CUDA test/research paths stay a no-op.
    """
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return
    torch.cuda.synchronize(resolved)
    torch.cuda.empty_cache()


def _stored_production_anchor_delta(
    rendered: torch.Tensor,
    source: torch.Tensor,
    *,
    storage_dtype: torch.dtype,
) -> torch.Tensor:
    """Subtract production anchors in FP32, then store the configured dW.

    The projection consumer always upcasts dW to FP32. Keeping every routed
    layer's two anchor deltas in FP32 therefore doubles residency without
    changing the accumulator. Subtract in FP32 to preserve cancellation, then
    use the same validated BF16 storage contract as ordinary cached-menu AURA.
    ``copy=True`` prevents an FP32 injected/test render from being mutated by
    the in-place subtraction.
    """
    delta_fp32 = rendered.to(
        device=source.device,
        dtype=torch.float32,
        copy=True,
    )
    delta_fp32.sub_(source)
    if storage_dtype == torch.float32:
        return delta_fp32
    return delta_fp32.to(dtype=storage_dtype)


def aura_unit_topology(
    model: nn.Module,
    linears: Mapping[str, object],
    *,
    profile,
    packed_members: Sequence[PackedExpertProjection] = (),
) -> dict[str, dict]:
    """Per-unit topology for AURA stats rows, from the probe's own sources.

    ``tessera_serving_scope.unit_structure_from_stats`` reads these facts to
    classify a unit under an explicit serving target (PQ #1278). They are the
    fields the incremental probe writes, taken from the same places:

    * an nn.Linear gets ``router_path``/``expert_id`` from
      ``sensitivity_probe.discover_moe_structure`` -- ``(None, None)`` when the
      walk places it under no router;
    * a packed expert's per-expert view gets ``_packed_experts_module`` and
      ``num_experts``, as ``tessera_campaign`` records for the same members.
      The count is the number of experts the profile split out of that packed
      parameter (``packed_members`` is the full, unfiltered split), never the
      subset a format plan or unit filter kept.

    A Linear the profile declares routed but the walk cannot place gets no
    keys at all: writing ``None``/``None`` would record a dense fact nobody
    observed, and the scope reader then refuses it as missing, not as a
    conflict.
    """
    from prismaquant.sensitivity_probe import discover_moe_structure

    routed = discover_moe_structure(model, profile=profile)
    classifier = ProfileRoutedExpertClassifier(profile)
    experts_by_packed: dict[str, set[int]] = {}
    for member in packed_members:
        experts_by_packed.setdefault(member.packed_qname, set()).add(int(member.expert_id))
    counts: dict[str, int] = {}
    for packed_qname, experts in experts_by_packed.items():
        if experts != set(range(len(experts))):
            raise RuntimeError(
                f"packed expert split of {packed_qname} is not experts 0..N-1")
        counts[packed_qname] = len(experts)
    out: dict[str, dict] = {}
    for name, module in linears.items():
        if isinstance(module, PackedExpertProjection):
            if module.packed_qname not in counts:
                raise RuntimeError(
                    f"{name}: packed expert view without its full split; "
                    "pass the unfiltered packed_members")
            out[name] = {"_packed_experts_module": module.module_qname,
                         "num_experts": counts[module.packed_qname]}
        elif name in routed:
            router_path, expert_id = routed[name]
            out[name] = {"router_path": router_path, "expert_id": expert_id}
        elif classifier.classify(name) is not None:
            out[name] = {}
        else:
            out[name] = {"router_path": None, "expert_id": None}
    return out


def _target_linears(
    model: nn.Module,
    *,
    include_lm_head: bool = False,
    include_routed_experts: bool = False,
    profile=None,
) -> dict[str, nn.Linear]:
    """Quantizable nn.Linear targets. lm_head is EXCLUDED by default (the
    profile pins it BF16). include_lm_head adds it so Aura can MEASURE its
    KL-sensitivity and let the allocator choose its format as a budget
    decision rather than a hardcoded pin -- the KL probe gradient flows
    directly into lm_head (it produces the logits), so its cost is
    measured the same way as any body Linear."""
    # Routed-expert membership is a serving/model-profile property, not a
    # tensor-rank property.  In particular DSv4's probe model exposes every
    # expert projection as a 2-D Linear, which otherwise looks exactly like a
    # smooth AURA target.  Resolve the same profile-owned predicate used by
    # the coverage guard below so exclusion and enforcement cannot diverge.
    profile = resolve_routed_expert_profile(model, profile)
    unpacked_expert_names = {
        member.qname
        for member in profile_declared_unpacked_expert_linears(model, profile)
    }
    out: dict[str, nn.Linear] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if name in unpacked_expert_names and not include_routed_experts:
            continue
        if "lm_head" in name and not include_lm_head:
            continue
        if mod.weight.dim() == 2 and min(mod.weight.shape) >= 16:
            out[name] = mod
    return out


def _delta_w(
    name: str,
    fmt: str,
    weight: torch.Tensor,
    cache: object | None,
    *,
    strict: bool = False,
) -> tuple[torch.Tensor, str] | None:
    """Q_f(W)-W plus its provenance: ``(delta, "rendered"|"rtn")``.

    "rendered" = production-rendered error from the cache (the bytes export
    ships); "rtn" = format-registry RTN fallback. The distinction is recorded
    per cost row because it is result-changing: RTN-vs-rendered dW moved FP8
    allocations by +36% served KL (2026-06 A/B). ``strict``
    (require_production_cache): when a cache is supplied but lacks the rendered
    (name, fmt), fail fast with a clear coverage error instead of silently
    falling back to RTN -- so a 'production-faithful' run cannot quietly mix
    RTN deltas into the cost. Default off preserves the RTN fallback used by
    non-production ablations."""
    if cache is not None:
        try:
            rendered = cache.get(name, fr.canonical_format_name(fmt))
        except Exception:
            rendered = None
        if rendered is not None:
            delta = rendered.to(weight.device, torch.float32) - weight.float()
            return delta, "rendered"
        if strict:
            raise RuntimeError(
                f"require_production_cache: production-rendered weight missing "
                f"for ({name!r}, {fmt!r}); refusing silent RTN fallback. Build the "
                f"cache for this (Linear, format) or drop --require-production-cache.")
    # Ahead of ``get_format``, which imports the owning lane's package to
    # synthesize its spec: the refusal is about the name, not the spec.
    if fr.requires_production_render(fmt):
        raise RuntimeError(
            f"{name}={fmt}: AURA {fr.format_owner_label(fmt)} delta "
            "requires a production-cache "
            "render. The registry fallback is a weights-only reconstruction, "
            "not the decoded wire and not the H-aware encode that ships, so "
            "it would price a different dW under the same format name -- and "
            "``strict`` defaults off, so nothing else would say so."
        )
    spec = fr.get_format(fmt)
    qdq = getattr(spec, "quantize_dequantize", None)
    if qdq is None:
        return None
    try:
        rendered = qdq(weight.float())
        return rendered - weight.float(), "rtn"
    except Exception:
        return None


def _packed_expert_targets(model: nn.Module, profile=None) -> list[str]:
    """Profile-declared routed expert targets in either physical layout.

    The historical name is retained for artifact/provenance compatibility,
    but rank is intentionally absent: packed 3-D Parameters and unpacked 2-D
    expert Linears are the same route-sensitive serving class.
    """
    return profile_declared_routed_expert_targets(model, profile)


def _guard_packed_expert_coverage(
    model: nn.Module,
    profile=None,
    *,
    allow_omission: bool = False,
) -> list[str]:
    routed = _packed_expert_targets(model, profile)
    if routed and not allow_omission:
        sample = ", ".join(routed[:6])
        raise RuntimeError(
            "Aura cost does not yet implement packed-MoE expert costs in its "
            "smooth sweep; "
            f"found {len(routed)} profile-declared routed expert target(s), "
            f"sample={sample}. Use the empirical routed-expert cost path or pass "
            "--allow-packed-expert-omission only for an explicit research/debug "
            "run that accepts experts being omitted from the AURA cost payload."
        )
    if routed:
        print(
            "[aura-cost] WARNING: omitting profile-declared routed experts "
            "from Aura cost by explicit request: "
            f"{len(routed)} targets, sample={routed[:6]}",
            flush=True,
        )
    return routed


def _assemble_streamed_aura_payload(
    *,
    linears: Mapping[str, nn.Linear],
    names: Sequence[str],
    formats: Sequence[str],
    formats_by_qname: Mapping[str, Sequence[str]],
    unmeasured_formats_by_qname: Mapping[str, Sequence[str]],
    n_probes: int,
    token_scope: str,
    seed_base: int,
    temperature: float,
    dw_dtype: str,
    measurement_dtype: torch.dtype,
    n_linear_chunks: int,
    calib_ids: torch.Tensor,
    omitted_packed_experts: Sequence[str],
    checkpoint_git_commit: str | None,
    collect_col_energy: bool,
    s2: Mapping[tuple[str, str], float],
    s4: Mapping[tuple[str, str], float],
    x2_probe: Mapping[tuple[str, str], list[float]],
    dw_src: Mapping[tuple[str, str], str],
    g_trace: Mapping[str, float],
    col_energy: Mapping[str, torch.Tensor],
    weight_mse_diagnostic: Mapping[tuple[str, str], float],
    unit_topology: Mapping[str, Mapping[str, object]],
) -> dict:
    inv = 1.0 / float(n_probes)
    stats: dict[str, dict] = {}
    costs: dict[str, dict] = {}
    for name in names:
        mod = linears[name]
        stats[name] = {
            "h_trace": g_trace[name] * inv,
            "n_params": int(mod.weight.numel()),
            "in_features": int(getattr(mod, "in_features", mod.weight.shape[1])),
            "out_features": int(getattr(mod, "out_features", mod.weight.shape[0])),
            "n_probes": int(n_probes),
            **unit_topology[name],
        }
        if collect_col_energy and name in col_energy:
            stats[name]["fisher_col"] = (
                col_energy[name].float() * inv
            ).detach().cpu()
        costs[name] = {}
        for fmt in formats_by_qname[name]:
            if fmt in _ZERO_COST_FORMATS:
                costs[name][fmt] = {
                    "predicted_dloss": 0.0,
                    "output_mse_measured": False,
                    "cost_source": "aura_passthrough_zero",
                }
                continue
            if fmt in unmeasured_formats_by_qname.get(name, ()):
                # Identity-bearing terminals can be semantically real yet
                # outside this measurement currency.  Retain them in the
                # exact streamed plan, but never invent a scalar row.
                continue
            key = (name, fmt)
            if key not in s2:
                continue
            mean_x2 = inv * s2[key]
            variance = (
                max(
                    (s4[key] - n_probes * mean_x2 * mean_x2)
                    / (n_probes - 1),
                    0.0,
                )
                if n_probes >= 2
                else 0.0
            )
            row = {
                "predicted_dloss": 0.5 * mean_x2,
                "predicted_dloss_stderr": 0.5 * math.sqrt(variance * inv),
                "x2_per_probe": x2_probe[key],
                "dw_source": dw_src[key],
                "output_mse_measured": False,
                "cost_source": "aura",
            }
            if dw_src[key] == "production_render":
                row.update({
                    "production_anchor_measured": True,
                    "production_anchor_zero": mean_x2 == 0.0,
                })
            if key in weight_mse_diagnostic:
                row.update({
                    # Diagnostic only: AURA remains the allocator's sole cost
                    # currency. This same-render value tests within-basis
                    # currency-ratio invariance on fitting-panel cells.
                    "weight_mse_diagnostic": float(
                        weight_mse_diagnostic[key]
                    ),
                    "weight_mse_diagnostic_normalization": "mean_per_weight",
                    "weight_mse_is_cost_input": False,
                })
            costs[name][fmt] = row

    calib_cpu = calib_ids.detach().cpu().contiguous()
    return {
        "schema": SCHEMA,
        "n_probes": n_probes,
        "formats": list(formats),
        "token_scope": token_scope,
        "stats": stats,
        "costs": costs,
        "provenance": {
            "seed_base": int(seed_base),
            "temperature": float(temperature),
            "dw_dtype": str(dw_dtype),
            "measurement_dtype": str(measurement_dtype),
            "include_lm_head": False,
            "n_linear_chunks": int(n_linear_chunks),
            "calib_shape": list(calib_ids.shape),
            "calib_sha256": bytes_sha256hex(
                calib_cpu.numpy().tobytes()
            ),
            "calib_hash": calibration_data_hash(calib_ids),
            "calib_hashes": [calibration_data_hash(calib_ids)],
            "omitted_packed_experts": list(omitted_packed_experts),
            "dw_rendered_rows": int(sum(
                value == "rendered" for value in dw_src.values()
            )),
            "dw_production_anchor_rows": int(sum(
                value == "production_render" for value in dw_src.values()
            )),
            "dw_rtn_fallback_rows": int(sum(
                value == "rtn" for value in dw_src.values()
            )),
            "git_commit": checkpoint_git_commit or aura_provenance.producer_git_commit(),
            "streaming": True,
            "streamed_gradient_harvest": (
                "post_accumulate_per_parameter"
            ),
            "streamed_cotangent_rollover": "in_place_per_probe",
            "streamed_boundary_release": "progressive_reverse",
        },
    }


def _with_boundary_artifacts(function):
    """Keep temporary activation generations inside the public call lifetime."""
    from functools import wraps

    @wraps(function)
    def run(*args, **kwargs):
        config = kwargs.pop("boundary_storage", None)
        if config is None:
            return function(*args, **kwargs)
        from prismaquant.streamed_boundary_artifacts import StreamedBoundaryArtifacts
        with StreamedBoundaryArtifacts(config) as storage:
            result = function(*args, **kwargs, boundary_storage=storage)
        result["provenance"]["streamed_boundary_storage"] = storage.receipt()
        return result
    return run


@_with_boundary_artifacts
def compute_aura_cost_streamed(
    runner,
    calib_ids: torch.Tensor,
    formats: Sequence[str],
    *,
    n_probes: int = 16,
    probe_microbatch: int = 0,
    token_scope: str = "all",
    temperature: float = 1.0,
    production_cache: object | None = None,
    min_free_gib: float = 20.0,
    seed_base: int = 7000,
    assert_bf16_passthrough: bool = False,
    require_production_cache: bool = False,
    dw_dtype: str = "bfloat16",
    include_lm_head: bool = False,
    allow_packed_expert_omission: bool = False,
    collect_col_energy: bool = False,
    checkpoint_dir: str | Path | None = None,
    resume: bool = False,
    unit_filter: str | None = None,
    model_identity: Mapping[str, object] | None = None,
    checkpoint_identity_extra: Mapping[str, object] | None = None,
    formats_by_qname: Mapping[str, Sequence[str]] | None = None,
    unmeasured_formats_by_qname: Mapping[str, Sequence[str]] | None = None,
    anchor_renderer: object | None = None,
    include_routed_experts: bool = False,
    diagnostic_weight_mse_pairs: Sequence[tuple[str, str]] | None = None,
    joint_activation: bool = False,
    joint_projection_backend=None,
    source_transition=None,
    boundary_storage=None,
    operator_windows=None,
    retained_operator_windows=None,
    device_envelope_bytes=None,
    prepared_render_identities=None,
    cost_read_schedule=None,
    progress_base: int = 0,
    profile=None,
    sequence_attribution: Mapping[str, object] | None = None,
) -> dict:
    """Layer-streamed KL-adjoint with identity-bound per-Linear shards.

    The source forward stores only decoder-boundary activations.  Its reverse
    pass then installs one decoder layer, recomputes that layer for every KL
    probe, projects each weight gradient onto production ``dW`` in fp32, and
    unloads the layer.  Thus source weights, gradients, and rendered deltas
    are bounded by one decoder layer; no autograd graph can retain the whole
    model. ``probe_microbatch > 0`` opts joint AURA into complete-sequence
    partitions with versioned global-row probes. Local signed terms and weight
    gradients sum across all partitions before squaring; full-vocabulary GPU
    tensors are bounded by the partition. By default CPU boundaries and shared
    pass state cover the full calibration. Explicit ``boundary_storage`` v1/v2
    stores exact snapshots and rolling cotangents through the existing
    activation artifact owner, leases bounded resident input windows, and caps
    metadata/shared-state residency. V2 uses layer-major baseline capture with
    the original per-batch source sequence; v1 preserves batch-major capture.
    Both preserve scalar projection arithmetic; candidate/gradient bounds
    remain separate admission requirements. Checkpoints bind the execution partition
    because floating-point kernels may round differently with batch shape.
    """
    from prismaquant.joint_statistics_replay import (
        normalize_operator_windows, operator_window_guard, resident_candidates,
        check_operator_allocation, preflight_joint_operator_admission,
        observe_and_project_windows, observe_and_project_retained_windows,
        statistics_arithmetic_identity,
    )
    operator_windows = normalize_operator_windows(operator_windows)
    from .joint_retained_window_plan import (
        OBSERVED_BASELINE_KEY, normalize_retained_execution, RetainedWindowBudget)
    if retained_operator_windows is not None:
        retained_operator_windows = normalize_retained_execution(retained_operator_windows,
            operator_windows=operator_windows,
            boundary_storage=None if boundary_storage is None else boundary_storage.config)
    retained_budget = (None if retained_operator_windows is None else
                       RetainedWindowBudget.from_dict(retained_operator_windows['budget']))
    operator_guard = None
    operator_window_receipts = []
    if operator_windows is not None:
        if not joint_activation or production_cache is None or anchor_renderer is not None:
            raise ValueError('joint operator windows require joint AURA with a materialized PWC')
        if any(module.training for module in runner.model.modules()):
            raise ValueError('joint operator replay requires an eval source')
        operator_guard = operator_window_guard(runner.device, device_bytes=device_envelope_bytes)
    if retained_budget is not None:
        if checkpoint_dir is None:
            raise ValueError('retained COST requires existing durable unit checkpoints')
        if production_cache._window_resident_storages():
            raise RuntimeError('retained COST starts with a nonempty PWC resident owner')
        # This is a COST lifetime, not a change to the serialized PREPARE
        # policy or its source/render qualification evidence.
        production_cache.enable_lru(retained_budget.retained_render_cap_bytes)
        if operator_guard is not None:
            retained_budget.require_physical_guard(operator_guard)

    def retained_source_phase(stage, layer, actual_auxiliary_bytes):
        if retained_budget is None:
            return
        snapshot = runner.context.source_residency_snapshot(range(runner.num_layers), include_head=True)
        if any(owner.get('state') in ('pending', 'failed') for owner in snapshot['owners']):
            raise RuntimeError('retained COST source phase has an unsettled or failed source owner')
        source_bytes = snapshot['unique_storage_bytes']
        source_cap = retained_operator_windows['source_reserve_bytes']
        if source_bytes > source_cap:
            raise RuntimeError('actual source/head owners exceed the sealed retained COST source cap')
        if operator_guard is not None:
            observed = check_operator_allocation(operator_guard, f'before_retained_{stage}:{layer}', reserve_bytes=0)
            retained_budget.require_observed_baseline(
                observed_bytes=observed[OBSERVED_BASELINE_KEY],
                source_bytes=source_bytes, actual_auxiliary_bytes=actual_auxiliary_bytes,
                label=f'before_retained_{stage}:{layer}')
            auxiliary_growth = max(0, retained_budget.auxiliary_reserve_bytes - actual_auxiliary_bytes)
            future = retained_budget.boundary_reserve_bytes + auxiliary_growth
            if stage == 'source_loading':
                future += source_cap - source_bytes + retained_operator_windows['source_loading_reserve_bytes']
            else:
                future += retained_budget.workspace_reserve_bytes
            check_operator_allocation(operator_guard, f'admit_retained_{stage}:{layer}', reserve_bytes=future)
    if source_transition is not None:
        from prismaquant.joint_aura_transitions import require_verified_transition
        source_transition = require_verified_transition(
            source_transition, checkpoint_dir=checkpoint_dir,
            resume=resume, joint_activation=joint_activation,
        )
    if type(probe_microbatch) is not int or probe_microbatch < 0:
        raise ValueError("probe_microbatch must be a nonnegative integer")
    if calib_ids.ndim != 2 or min(calib_ids.shape) < 1:
        raise ValueError("streamed AURA needs nonempty [rows, sequence] calibration")
    if boundary_storage is not None and model_identity is None:
        raise ValueError("exact boundary storage requires exact model_identity")
    if joint_projection_backend is not None and not joint_activation:
        raise ValueError("joint_projection_backend requires joint_activation")
    if probe_microbatch and not joint_activation:
        raise ValueError("streamed probe_microbatch currently requires joint_activation")
    from prismaquant.joint_aura import (
        normalize_sequence_attribution, sequence_attribution_candidates)
    attribution_config = normalize_sequence_attribution(sequence_attribution)
    if attribution_config is not None:
        if operator_windows is not None or retained_budget is not None:
            raise ValueError(
                "sequence attribution requires the streamed dense joint path; "
                "the operator-window replay retains no captured rows to re-read")
        if not joint_activation:
            raise ValueError("sequence attribution requires joint_activation")
    batch_rows = min(probe_microbatch or len(calib_ids), len(calib_ids))
    row_offsets = list(range(0, len(calib_ids), batch_rows))
    probe_layout = None
    if probe_microbatch:
        from prismaquant.kl_fisher import ROW_PROBE_LAYOUT
        sequence_length = int(calib_ids.shape[1])
        selected_tokens = {"all": sequence_length, "last": 1,
                           "causal": sequence_length - 1}.get(token_scope, 0)
        if selected_tokens < 1:
            raise ValueError("invalid token scope or sequence length for streamed probes")
        probe_layout = {
            "schema": ROW_PROBE_LAYOUT,
            "global_rows": len(calib_ids),
            "sequence_length": sequence_length,
            "selected_tokens_per_row": selected_tokens,
            "vocab_size": int(runner._head().weight.shape[0]),
            "token_scope": token_scope,
            "global_token_count": len(calib_ids) * selected_tokens,
        }
    execution_partition = {
        "schema": "prismaquant.aura.streamed_microbatch.v1",
        "requested_rows": probe_microbatch,
        "effective_rows": batch_rows,
        "partition_count": len(row_offsets),
        "row_order": "contiguous_complete_sequences",
        "gradient_diagnostics": "sum_fp32_gradients_before_norm",
    } if probe_microbatch else None
    if n_probes < 1:
        raise ValueError(f"n_probes must be >= 1, got {n_probes!r}")
    if resume and checkpoint_dir is None:
        raise ValueError("resume=True requires checkpoint_dir")
    if joint_activation:
        if production_cache is None and anchor_renderer is None:
            raise ValueError("joint AURA requires production-rendered weights")
        if type(n_probes) is not int or n_probes < 2:
            raise ValueError("joint AURA requires at least two probes for sampling uncertainty")
        if model_identity is None:
            raise ValueError("joint AURA requires exact model_identity for its cotangents")
        # Rounding dW to BF16 drops part of the requested local residual.
        dw_dtype = "float32"
    profile = resolve_routed_expert_profile(
        runner.model, profile or runner.profile
    )
    packed_projections = []
    if include_routed_experts:
        routed_targets = set(_packed_expert_targets(runner.model, profile))
        unpacked_targets = {
            member.qname
            for member in profile_declared_unpacked_expert_linears(
                runner.model, profile
            )
        }
        if joint_activation:
            packed_projections = profile_declared_packed_expert_projections(runner.model, profile)
        packed_targets = {member.packed_qname for member in packed_projections}
        unsupported = sorted(routed_targets - unpacked_targets - packed_targets)
        if unsupported:
            raise RuntimeError(
                "streamed AURA can include routed experts only when the "
                "profile exposes them as per-expert nn.Linear modules; "
                f"unsupported profile-declared targets={unsupported[:8]}"
            )
        omitted_packed_experts: list[str] = []
    else:
        omitted_packed_experts = _guard_packed_expert_coverage(
            runner.model,
            profile,
            allow_omission=allow_packed_expert_omission,
        )
    if include_lm_head:
        raise RuntimeError(
            "streamed AURA currently requires the profile-pinned lm_head to "
            "remain resident BF16; --include-lm-head is unsupported"
        )
    linears = _target_linears(
        runner.model,
        include_lm_head=False,
        include_routed_experts=include_routed_experts,
        profile=profile,
    )
    if packed_projections:
        if anchor_renderer is not None:
            raise RuntimeError(
                "packed joint AURA requires decoded ProductionWeightCache entries; "
                "the on-demand production anchor renderer supports only nn.Linear modules"
            )
        linears.update({member.qname: member for member in packed_projections})
    fmts = list(dict.fromkeys(
        fr.canonical_format_name(fmt) for fmt in formats
    ))
    raw_plan: dict[str, tuple[str, ...]] | None = None
    if formats_by_qname is not None:
        raw_plan = {}
        for raw_name, raw_formats in formats_by_qname.items():
            name = str(raw_name)
            if isinstance(raw_formats, (str, bytes)):
                raise TypeError(
                    f"streamed AURA format plan {name!r} must be a sequence"
                )
            planned = tuple(dict.fromkeys(
                fr.canonical_format_name(fmt) for fmt in raw_formats
            ))
            if not planned:
                raise ValueError(
                    f"streamed AURA format plan has no formats for {name}"
                )
            unknown_formats = sorted(set(planned) - set(fmts))
            if unknown_formats:
                raise ValueError(
                    f"streamed AURA format plan for {name} contains formats "
                    f"outside the requested union: {unknown_formats}"
                )
            raw_plan[name] = planned
        unknown_names = sorted(set(raw_plan) - set(linears))
        if unknown_names:
            raise ValueError(
                "streamed AURA format plan contains qnames that are not "
                f"eligible live Linears; sample={unknown_names[:8]}"
            )
        linears = {
            name: linears[name] for name in raw_plan
        }
    raw_unmeasured: dict[str, tuple[str, ...]] = {}
    if unmeasured_formats_by_qname is not None:
        if raw_plan is None:
            raise ValueError(
                "unmeasured streamed formats require formats_by_qname"
            )
        for raw_name, raw_formats in unmeasured_formats_by_qname.items():
            name = str(raw_name)
            if name not in raw_plan:
                raise ValueError(
                    f"unmeasured streamed format plan has unknown qname {name!r}"
                )
            if isinstance(raw_formats, (str, bytes)):
                raise TypeError(
                    f"unmeasured streamed formats for {name!r} must be a sequence"
                )
            values = tuple(dict.fromkeys(
                fr.canonical_format_name(fmt) for fmt in raw_formats
            ))
            unexpected = sorted(set(values) - set(raw_plan[name]))
            if unexpected:
                raise ValueError(
                    f"unmeasured streamed formats for {name} are outside "
                    f"its exact plan: {unexpected}"
                )
            raw_unmeasured[name] = values
    if unit_filter:
        try:
            pattern = re.compile(str(unit_filter))
        except re.error as exc:
            raise ValueError(f"invalid unit_filter regex {unit_filter!r}") from exc
        linears = {
            name: mod for name, mod in linears.items() if pattern.search(name)
        }
        if not linears:
            raise ValueError(
                f"unit_filter {unit_filter!r} matched no AURA Linear qnames"
            )
    names = list(linears)
    if not names:
        raise RuntimeError("streamed AURA found no smooth Linear targets")
    unit_topology = aura_unit_topology(
        runner.model, linears, profile=profile, packed_members=packed_projections)
    names_by_layer: dict[int, list[str]] = {}
    for name in names:
        layer = runner.layer_index_for_qname(name)
        names_by_layer.setdefault(layer, []).append(name)

    def _refresh_packed_layer_views(layer):
        members = [linears[name] for name in names_by_layer.get(layer, ())
                   if isinstance(linears[name], PackedExpertProjection)]
        if members:
            linears.update({member.qname: member for member in
                            refresh_packed_expert_projections(members, profile)})

    def _source_parameter(target):
        return target.parameter if isinstance(target, PackedExpertProjection) else target.weight

    def _require_installed_render_sources(layer):
        # Prepare compared each render with the installed source tensor. The
        # installed dtype is the loader's decision (``_read_layer_to_device``
        # casts to its dtype policy, ``_fast_install`` keeps the loaded dtype),
        # not a fact the meta skeleton carries, so the dtype/byte half of the
        # prepared proof is checked here on the tensor the layer actually
        # holds, before any of its renders is consumed.
        if prepared_render_identities is None:
            return
        for name in names_by_layer.get(layer, ()):
            renders = joint_cache_renders.get(name)
            if not renders:
                continue
            source = linears[name].weight
            if source.is_meta:
                raise RuntimeError(f'installed source is still a meta parameter for {name}')
            for fmt, value in renders.items():
                if (value['shape'] != list(source.shape)
                        or value['dtype'] != str(source.dtype)
                        or value['logical_bytes'] != source.numel() * source.element_size()):
                    raise RuntimeError(
                        f'prepared render tensor proof differs from the installed source for {name}@{fmt}')

    def _source_gradient(target, gradient):
        return target.gradient_view(gradient) if isinstance(target, PackedExpertProjection) else gradient

    unit_formats: dict[str, tuple[str, ...]] = {}
    render_formats: dict[str, tuple[str, ...]] = {}
    for name in names:
        planned = raw_plan[name] if raw_plan is not None else tuple(fmts)
        unmeasured = set(raw_unmeasured.get(name, ()))
        measured = tuple(
            fmt for fmt in planned
            if fmt not in _ZERO_COST_FORMATS and fmt not in unmeasured
        )
        unit_formats[name] = tuple(planned)
        render_formats[name] = measured
    attribution_keys = sequence_attribution_candidates(attribution_config, render_formats)
    if operator_windows is not None and any(not render_formats[name] for name in names):
        raise ValueError('joint operator windows require a measured candidate for every target')
    joint_probe_identity = None
    joint_run_identity = None
    attribution_components: dict[tuple[str, str], list[dict]] = {}
    attribution_blocks: list[dict] | None = None
    joint_source_tensors: dict[str, dict] = {}
    joint_cache_renders: dict[str, dict[str, dict]] = {}
    joint_prefetch_stats: list[dict] = []
    preflight_retained_windows = None
    if joint_activation:
        from prismaquant.joint_projection_backend import prewarm_projection_backend
        joint_projection_backend = prewarm_projection_backend(joint_projection_backend, device=runner.device)
        # Combined operator/loader/delta admission reads declared bytes only --
        # the roster, each matrix's geometry, the PWC candidate file sizes and
        # the sealed budget -- so it is answerable now, before the first
        # boundary capture, for every layer at once. It used to run inside the
        # reverse loop, where an inadmissible plan cost a whole capture before
        # it refused (#743). The per-layer call still re-derives its own plan
        # and is compared with what was admitted here through ``sealed_windows``.
        if operator_windows is not None:
            preflight_retained_windows = preflight_joint_operator_admission(
                {layer: [name for name in layer_names if render_formats[name]]
                 for layer, layer_names in names_by_layer.items()},
                linears, render_formats, production_cache,
                policy=operator_windows, retained_budget=retained_budget,
                source_bytes=(None if retained_budget is None
                              else retained_operator_windows['source_reserve_bytes']))
        from prismaquant.streamed_model_identity import validate_streamed_model_identity
        from prismaquant.joint_aura import SignedJointProjectionLease, JointOperatorStatisticsLease, activation_identity, arithmetic_identity, make_joint_probe_identity, prefetch_joint_cache, source_execution_identity, sequence_attribution_run_identity
        from prismaquant.joint_aura_rows import joint_identity_sha256, squared_signed, validate_joint_aura_entry, sequence_attribution_sidecar
        from prismaquant.production_weight_cache import _cb_cache_tensor_identity

        joint_probe_identity = make_joint_probe_identity(
            calibration_ids=calib_ids,
            source_model=validate_streamed_model_identity(model_identity, where="joint AURA"),
            n_probes=n_probes, seed_base=seed_base,
            token_scope=token_scope, temperature=temperature,
            producer_source_sha256=(aura_provenance.producer_package_sha256() if source_transition is None
                                    else source_transition.measurement_source_sha256),
            source_execution=source_execution_identity(runner.model),
            arithmetic=(statistics_arithmetic_identity(runner.dtype, joint_projection_backend)
                        if operator_windows is not None else arithmetic_identity(runner.dtype, joint_projection_backend)),
        )
        if operator_windows is not None:
            joint_probe_identity['arithmetic']['operator_windows'] = operator_windows
            joint_probe_identity['arithmetic']['gradient_diagnostics'] = 'sum_output_operators_fp32_before_norm'
            if execution_partition is not None:
                execution_partition['gradient_diagnostics'] = 'sum_output_operators_fp32_before_norm'
        if probe_layout is not None:
            joint_probe_identity["noise_layout"] = probe_layout
            # RNG row coordinates are partition independent; the source
            # arithmetic is not. Existing paired/assignment consumers compare
            # this complete probe identity and must refuse mixed batch shapes.
            joint_probe_identity["arithmetic"]["execution_partition"] = execution_partition
        # Hash actual decoded production outputs before checkpoint admission,
        # in layer-bounded prefetch windows. This is identity preparation,
        # outside the cotangent/projection hot path; no tensor copy is retained.
        # Before install, a streamed decoder Linear is the meta skeleton's
        # parameter: its shape is the checkpoint's, its dtype is torch's
        # default (``build_streaming_skeleton`` passes no dtype), so only the
        # shape is compared here. Prepare verified every render against the
        # INSTALLED tensor (``verify_anchor_render``); the run repeats that
        # comparison per layer, at install, before the layer's first render is
        # consumed (``_require_installed_render_sources``).
        if prepared_render_identities is not None:
            expected_pairs = {(name, fmt) for name in names for fmt in render_formats[name]}
            if (production_cache is None or not isinstance(prepared_render_identities, dict)
                    or set(prepared_render_identities) != expected_pairs
                    or set(production_cache._expected_file_sha256 or {}) != expected_pairs):
                raise RuntimeError('prepared render identity/file SHA coverage differs from joint roster')
            verified = (production_cache.metadata or {}).get('verified_cells', {})
            for name, fmt in sorted(expected_pairs):
                value = prepared_render_identities[name, fmt]
                source = linears[name].weight
                if (not isinstance(value, dict) or set(value) != {
                        'shape', 'dtype', 'logical_bytes', 'content_sha256'}
                        or value['shape'] != list(source.shape)
                        or not isinstance(value['dtype'], str)
                        or type(value['logical_bytes']) is not int
                        or not isinstance(value['content_sha256'], str)
                        or len(value['content_sha256']) != 64
                        or any(c not in '0123456789abcdef' for c in value['content_sha256'])
                        or verified.get((name, fmt), {}).get('rendered_weight') != value
                        or verified.get((name, fmt), {}).get('render_file_sha256') !=
                           production_cache._expected_file_sha256[name, fmt]):
                    raise RuntimeError(f'prepared render tensor proof differs for {name}@{fmt}')
                joint_cache_renders.setdefault(name, {})[fmt] = dict(value)
        elif production_cache is not None:
            for layer_names in names_by_layer.values():
                if operator_windows is not None:
                    keys = [(name, fmt) for name in layer_names for fmt in render_formats[name]]
                    for name in layer_names:
                        joint_cache_renders[name] = {}
                    with resident_candidates(production_cache, keys, operator_windows,
                                             guard=operator_guard) as windows:
                        for quantum, receipt in windows:
                            for name, fmt in quantum:
                                joint_cache_renders[name][fmt] = _cb_cache_tensor_identity(
                                    production_cache.get_resident(name, fmt))
                            joint_prefetch_stats.append(dict(receipt))
                else:
                    joint_prefetch_stats.append(prefetch_joint_cache(
                        production_cache, layer_names, render_formats,
                        max_resident_bytes=max(0, int((_free_gib() - min_free_gib) * 1024**3)),
                    ))
                    for name in layer_names:
                        joint_cache_renders[name] = {
                            fmt: _cb_cache_tensor_identity(production_cache.get(name, fmt))
                            for fmt in render_formats[name]
                        }
                    production_cache.compact_for_pickle()
        joint_run_identity = {
            "schema": "prismaquant.joint_aura.run.v2",
            "probe_identity": joint_probe_identity,
            "cached_rendered_weights": joint_cache_renders,
            "activation_contracts": ({
                name: {fmt: activation_identity(fr.get_format(fmt), production_cache.activation_max_abs or {}, name)
                       for fmt in unit_formats[name]}
                for name in names
            } if production_cache is not None else None),
        }
        attribution_run = sequence_attribution_run_identity(attribution_config)
        if attribution_run is not None:
            # The opt-in selector, geometry and collector scope bind the RUN
            # identity, never the priced probe identity: a resume across the
            # attribution boundary refuses at the rows it reloads.
            joint_run_identity["sequence_attribution"] = attribution_run

    anchor_identity: Mapping[str, object] | None = None
    if anchor_renderer is not None:
        if production_cache is not None:
            raise ValueError(
                "streamed production anchors cannot also consume a "
                "materialized production_cache"
            )
        if formats_by_qname is None:
            raise ValueError(
                "streamed production anchors require an exact "
                "formats_by_qname anchor plan"
            )
        if str(dw_dtype) not in {"bfloat16", "float32"}:
            raise ValueError(
                "streamed production-anchor AURA requires dw_dtype to be "
                "'bfloat16' or 'float32'"
            )
        empty = sorted(name for name in names if not render_formats[name])
        if empty:
            raise ValueError(
                "every streamed production-anchor unit needs a real rendered "
                f"format; passthrough-only sample={empty[:8]}"
            )
        renderer_plan = getattr(anchor_renderer, "formats_by_qname", None)
        expected_plan = {
            name: tuple(render_formats[name]) for name in names
        }
        observed_plan = (
            {
                str(name): tuple(
                    fr.canonical_format_name(fmt) for fmt in values
                )
                for name, values in renderer_plan.items()
            }
            if isinstance(renderer_plan, Mapping)
            else None
        )
        if observed_plan != expected_plan:
            raise ValueError(
                "streamed production anchor renderer plan differs from the "
                "AURA qname->format plan"
            )
        anchor_identity = getattr(anchor_renderer, "identity", None)
        if not isinstance(anchor_identity, Mapping):
            raise ValueError(
                "streamed production anchor renderer has no exact identity"
            )
    diagnostic_pairs = {
        (str(name), fr.canonical_format_name(fmt))
        for name, fmt in (diagnostic_weight_mse_pairs or ())
    }
    if diagnostic_pairs:
        if anchor_renderer is None:
            raise ValueError(
                "diagnostic weight-MSE is available only from streamed "
                "production-anchor renders"
            )
        expected_anchor_pairs = {
            (name, fmt)
            for name in names
            for fmt in render_formats[name]
        }
        unexpected = sorted(diagnostic_pairs - expected_anchor_pairs)
        if unexpected:
            raise ValueError(
                "diagnostic weight-MSE requests unrendered anchor cells; "
                f"sample={unexpected[:8]}"
            )
    if checkpoint_dir is not None:
        # The checkpoint identity must embed a value-bearing render identity:
        # the production-anchor renderer's exact identity (bound below as
        # extra["production_anchor_renderer"], with the qname->format plan
        # asserted equal above), or a joint-activation run's own identity.
        if anchor_identity is None and not joint_activation:
            raise RuntimeError(
                "streamed AURA durable checkpointing requires a value-bearing "
                "render identity: a production-anchor renderer with exact "
                "identity"
            )
    if assert_bf16_passthrough and "BF16" in fmts:
        if runner.dtype not in (torch.bfloat16, torch.float16):
            raise RuntimeError(
                "assert_bf16_passthrough: BF16 zero-cost requires a bf16/fp16 "
                f"streamed source, got {runner.dtype}"
            )
    dw_torch_dtype = (
        torch.float32 if str(dw_dtype) == "float32" else torch.bfloat16
    )

    for parameter in runner.model.parameters():
        parameter.requires_grad_(False)

    measurements = AuraUnitMeasurements(
        n_probes=n_probes, render_formats=render_formats,
        collect_col_energy=collect_col_energy,
        diagnostic_weight_mse_pairs=diagnostic_pairs,
        require_source_weight_identity=anchor_renderer is not None)
    s2, s4 = measurements.s2, measurements.s4
    x2_probe, dw_src = measurements.x2_probe, measurements.dw_src
    g_trace, col_energy = measurements.g_trace, measurements.col_energy
    weight_mse_diagnostic = measurements.weight_mse_diagnostic
    source_weight_identity = measurements.source_weight_identity
    pilot_panel = (checkpoint_identity_extra or {}).get('joint_eval') if joint_activation else None
    if pilot_panel is not None and operator_windows is None:
        raise ValueError('joint diagnostic panel requires observer-backed operator windows')
    from .joint_eval_observation import new_observation_counts, observe_probe, stamp_observations
    observation_counts = new_observation_counts(names, n_probes) if pilot_panel is not None else None
    measurements.observation_counts = observation_counts
    price = None
    if joint_activation:
        required_attribution = (None if attribution_config is None else RequiredAttribution(
            attribution_keys, lambda name, fmt: ValueError(
                f"sequence_attribution cannot resume committed no-attribution rows for {name}@{fmt}")))
        price = JointPriceStep(
            measurements, probe_identity=joint_probe_identity,
            formats_by_qname={name: tuple(fmt for fmt in unit_formats[name]
                                         if fmt not in raw_unmeasured.get(name, ())) for name in names},
            zero_cost_formats=_ZERO_COST_FORMATS,
            rendered_identities=None if production_cache is None else joint_cache_renders,
            resume_contract=JointResumeContract(
                ProbeResumePolicy.EXACT,
                activation_identities=(None if production_cache is None else
                                       joint_run_identity["activation_contracts"]),
                attribution=required_attribution))
    joint_rows = {} if price is None else price.rows
    joint_components = {} if price is None else price.components
    joint_operators = {} if price is None else price.operators
    completed_checkpoint_units: set[str] = set()
    checkpoint_journal = None
    checkpoint_git_commit: str | None = None

    ordered_layer_chunks = [
        list(names_by_layer[layer]) for layer in sorted(names_by_layer)
    ]
    if checkpoint_dir is not None:
        checkpoint_git_commit = aura_provenance.checkpoint_git_commit()
        if model_identity is None:
            raise RuntimeError(
                "streamed AURA checkpointing requires exact model_identity; "
                "refusing model-name-gated resume"
            )
        from prismaquant.streamed_model_identity import validate_streamed_model_identity

        exact_model_identity = validate_streamed_model_identity(
            model_identity, where="streamed AURA checkpointing"
        )
        extra = dict(checkpoint_identity_extra or {})
        reserved = {
            "joint_aura",
            "streamed_formats_by_qname",
            "unmeasured_streamed_formats_by_qname",
            "production_anchor_renderer",
            "include_routed_experts",
            "diagnostic_weight_mse_pairs",
            "streamed_gradient_harvest",
            "streamed_cotangent_rollover",
            "streamed_boundary_release",
            "streamed_microbatch",
            "streamed_boundary_storage",
            "retained_operator_windows",
        } & set(extra)
        if reserved:
            raise ValueError(
                "checkpoint_identity_extra attempts to override streamed "
                f"AURA identity fields: {sorted(reserved)}"
            )
        extra["streaming"] = True
        if retained_operator_windows is not None:
            extra['retained_operator_windows'] = retained_operator_windows
        if execution_partition is not None:
            extra["streamed_microbatch"] = execution_partition
        if boundary_storage is not None:
            extra["streamed_boundary_storage"] = boundary_storage.identity
        if joint_activation:
            extra["joint_aura"] = joint_run_identity
        extra["streamed_model_identity"] = exact_model_identity
        extra["streamed_formats_by_qname"] = {
            name: list(unit_formats[name]) for name in names
        }
        extra["unmeasured_streamed_formats_by_qname"] = {
            name: list(raw_unmeasured.get(name, ())) for name in names
        }
        extra["include_routed_experts"] = bool(include_routed_experts)
        extra["diagnostic_weight_mse_pairs"] = [
            [name, fmt] for name, fmt in sorted(diagnostic_pairs)
        ]
        extra["streamed_gradient_harvest"] = (
            "post_accumulate_per_parameter"
        )
        extra["streamed_cotangent_rollover"] = "in_place_per_probe"
        extra["streamed_boundary_release"] = "progressive_reverse"
        if anchor_identity is not None:
            extra["production_anchor_renderer"] = dict(anchor_identity)
        identity = aura_checkpoint_journal.build_checkpoint_identity(
            model=runner.model,
            calib_ids=calib_ids,
            names=names,
            linears=linears,
            formats=fmts,
            chunks=ordered_layer_chunks,
            n_probes=n_probes,
            token_scope=token_scope,
            temperature=temperature,
            seed_base=seed_base,
            dw_dtype=dw_dtype,
            include_lm_head=False,
            hook_harvest=True,
            allow_packed_expert_omission=allow_packed_expert_omission,
            probe_microbatch=probe_microbatch,
            collect_col_energy=collect_col_energy,
            require_production_cache=require_production_cache,
            production_cache=production_cache,
            git_commit=checkpoint_git_commit,
            extra_identity=extra,
        )
        if source_transition is not None:
            identity = source_transition.measurement_identity(identity)
        checkpoint_journal = aura_checkpoint_journal.AuraCheckpointJournal.open(
            checkpoint_dir, resume=resume, identity=identity, names=names)
        completed_checkpoint_units = checkpoint_journal.resume_units(
            measurements if price is None else price)

    if cost_read_schedule is not None:
        if retained_budget is None or prepared_render_identities is None:
            raise RuntimeError('COST read schedule requires retained replay and prepared tensor identities')
        cost_read_schedule.bind_runtime(
            target_names_by_layer={layer: [name for name in names_by_layer.get(layer, [])
                                         if render_formats[name]]
                                   for layer in range(runner.num_layers)},
            validated_completed_units=completed_checkpoint_units)

    def completed_sidecar(name, fmt, components):
        key = (name, fmt)
        if attribution_config is None or key not in attribution_components:
            return None
        return sequence_attribution_sidecar(
            blocks=attribution_blocks, components_per_probe=attribution_components[key],
            authoritative_totals=[value["total"] for value in components],
            gate_relative=attribution_config["gate_relative"],
            arithmetic_scope="streamed_dense_lease_per_invocation_contractions",
            sequence_length=int(calib_ids.shape[1]),
            selected_tokens_per_row={"all": int(calib_ids.shape[1]), "last": 1,
                                     "causal": int(calib_ids.shape[1]) - 1}[token_scope],
            n_sequences=len(calib_ids),
            calibration_sha256=joint_probe_identity["calibration_sha256"])

    def commit_streamed_units(targets):
        targets = [name for name in targets if name not in completed_checkpoint_units]
        if not targets:
            return
        if joint_activation:
            if source_execution_identity(runner.model) != joint_probe_identity["source_execution"]:
                raise RuntimeError("joint AURA source execution backend changed during measurement")
            price.prepare_rows(targets, sidecar=completed_sidecar)
        if checkpoint_journal is not None:
            checkpoint_journal.publish_group(measurements if price is None else price, targets,
                extra_state=({"execution_provenance": source_transition.execution_provenance}
                             if source_transition is not None else None))
            if cost_read_schedule is not None:
                cost_read_schedule.enter_phase(cost_read_schedule.current_phase,
                                                len(completed_checkpoint_units))

    def _finish_streamed_payload() -> dict:
        payload = _assemble_streamed_aura_payload(
            unit_topology=unit_topology,
            linears=linears,
            names=names,
            formats=fmts,
            formats_by_qname=unit_formats,
            unmeasured_formats_by_qname=raw_unmeasured,
            n_probes=n_probes,
            token_scope=token_scope,
            seed_base=seed_base,
            temperature=temperature,
            dw_dtype=dw_dtype,
            measurement_dtype=runner.dtype,
            n_linear_chunks=len(ordered_layer_chunks),
            calib_ids=calib_ids,
            omitted_packed_experts=omitted_packed_experts,
            checkpoint_git_commit=checkpoint_git_commit,
            collect_col_energy=collect_col_energy,
            s2=s2,
            s4=s4,
            x2_probe=x2_probe,
            dw_src=dw_src,
            g_trace=g_trace,
            col_energy=col_energy,
            weight_mse_diagnostic=weight_mse_diagnostic,
        )
        source_binding = (checkpoint_identity_extra or {}).get("source_binding")
        if source_binding is not None and not joint_activation and anchor_renderer is None:
            for rows in payload["costs"].values():
                for row in rows.values():
                    row["source_binding"] = source_binding
        if cost_read_schedule is not None:
            payload['provenance']['cost_read_schedule'] = cost_read_schedule.identity
        if retained_budget is not None:
            payload['provenance']['retained_operator_windows'] = retained_operator_windows
        if source_transition is not None:
            payload["provenance"]["source_transition"] = source_transition.final_provenance()
        if execution_partition is not None:
            payload["provenance"]["streamed_microbatch"] = execution_partition
        if joint_activation:
            if set(joint_rows) != set(names):
                raise RuntimeError("joint AURA incomplete unit coverage")
            payload["costs"] = joint_rows
            if observation_counts is not None:
                stamp_observations(payload, observation_counts, pilot_panel)
            payload["provenance"].update({
                "cost_mode": "aura", "joint_activation": True,
                "cost_currency": "joint_aura_predicted_dloss",
                "joint_aura_identity": joint_run_identity,
                "joint_aura_identity_sha256": joint_identity_sha256(joint_run_identity),
                "probe_identity": joint_probe_identity,
                "probe_identity_sha256": joint_identity_sha256(joint_probe_identity),
                "joint_prefetch": joint_prefetch_stats,
                **({'joint_operator_windows': operator_window_receipts,
                    'joint_operator_memory': operator_guard.snapshot() if operator_guard is not None else None}
                   if operator_windows is not None else {}),
                "measurement_status": "diagnostic_pilot" if pilot_panel is not None else "research",
                "uncertainty_scope": "probe_sampling_conditional_on_fixed_calibration",
            })
        if anchor_renderer is not None:
            missing_source_identity = sorted(
                set(names) - set(source_weight_identity)
            )
            if missing_source_identity:
                raise RuntimeError(
                    "streamed production-anchor source identity is "
                    f"incomplete; sample={missing_source_identity[:8]}"
                )
            bind_source = getattr(
                anchor_renderer,
                "bind_completed_source_weight_identities",
                None,
            )
            completed_renderer_identity = (
                bind_source(source_weight_identity)
                if callable(bind_source)
                else {
                    **dict(anchor_identity or {}),
                    "source_weights": {
                        "complete": True,
                        "records": {
                            name: dict(source_weight_identity[name])
                            for name in sorted(source_weight_identity)
                        },
                    },
                }
            )
            payload["provenance"]["production_anchor_renderer"] = (
                completed_renderer_identity
            )
            expected_renders = sum(
                len(render_formats[name]) for name in names
            )
            rendered_now = int(getattr(anchor_renderer, "render_count", 0))
            payload["provenance"].update({
                "production_anchor_expected_renders": expected_renders,
                "production_anchor_rendered_this_invocation": rendered_now,
                "production_anchor_restored_renders": (
                    expected_renders - rendered_now
                ),
                "production_anchor_max_live_rendered": int(getattr(
                    anchor_renderer, "max_live_rendered", 0
                )),
                "production_anchor_no_full_menu_materialization": True,
                "production_anchor_cost_currency": "aura_only",
                "weight_mse_diagnostic_rows": len(
                    weight_mse_diagnostic
                ),
                "weight_mse_diagnostic_is_cost_input": False,
            })
        return payload

    if completed_checkpoint_units == set(names):
        _log(f"checkpoint resume: validated all {len(names)} streamed units; "
             "skip forward/reverse")
        return _finish_streamed_payload()

    def _record_joint_operator(name, fmt, source, rendered):
        if not joint_activation:
            return
        cache_owner = production_cache if production_cache is not None else getattr(anchor_renderer, "cache", None)
        scales = getattr(cache_owner, "activation_max_abs", None) or {}
        activation = activation_identity(fr.get_format(fmt), scales, name)
        rendered_identity = _cb_cache_tensor_identity(rendered)
        if production_cache is not None and fmt in render_formats[name]:
            if rendered_identity != joint_cache_renders[name][fmt] or activation != joint_run_identity["activation_contracts"][name][fmt]:
                raise RuntimeError(f"joint AURA actual render/activation identity changed for {name}@{fmt}")
        if name not in joint_source_tensors:
            joint_source_tensors[name] = _cb_cache_tensor_identity(source)
        joint_operators[(name, fmt)] = {
            "schema": "prismaquant.joint_aura.operator.v2",
            "qname": name, "format": fmt,
            "source_weight": joint_source_tensors[name],
            "rendered_weight": rendered_identity,
            "activation": activation,
            "arithmetic": joint_probe_identity["arithmetic"],
            "probe_identity_sha256": joint_identity_sha256(joint_probe_identity),
        }
        joint_components.setdefault((name, fmt), [])

    # Identity validation above is intentionally before the first model
    # forward: a mismatched resume is a refusal, never a recomputation.
    from prismaquant.streamed_boundary_artifacts import prefetched_boundary_batches
    if boundary_storage is not None:
        from prismaquant.streamed_model_identity import validate_streamed_model_identity

        def check_boundary_memory(label):
            available = _free_gib()
            if available < min_free_gib:
                raise RuntimeError(f"free UMA {available:.1f} < floor {min_free_gib:.1f}; {label}")

        boundary_storage.bind({
            "source_model": validate_streamed_model_identity(model_identity, where="exact boundary storage"),
            "producer_source_sha256": aura_provenance.producer_package_sha256(),
            "calibration_sha256": bytes_sha256hex(calib_ids.detach().cpu().contiguous().numpy().tobytes()),
            "calibration_shape": list(calib_ids.shape), "calibration_dtype": str(calib_ids.dtype),
            "n_probes": n_probes, "seed_base": seed_base, "token_scope": token_scope,
            "temperature": temperature, "execution_partition": execution_partition,
            "joint_probe_identity": joint_probe_identity,
        }, n_probes=n_probes, check_memory=check_boundary_memory)
        # Every published entry file is a durable unit and the only thing the
        # window can advance on. The reporter owns the clock; the writer only
        # tells it a file landed. With no channel in the environment this is a
        # log line and nothing else, which is what every run submitted without
        # the transport keeps getting.
        from prismaquant.joint_run_progress import JointRunProgress
        run_progress = JointRunProgress(layers=runner.num_layers,
                                        partitions=len(row_offsets),
                                        base_units=progress_base, log=_log)
        run_progress.priced_units(len(completed_checkpoint_units))
        boundary_storage.watch_progress(run_progress)
    else:
        run_progress = None

    def _report_units(phase_hint=None):
        """Fold the cost stage's journalled units into the run's one counter."""
        if run_progress is None:
            return
        run_progress.priced_units(len(completed_checkpoint_units))
        if phase_hint is not None:
            run_progress.enter(phase_hint)
        run_progress.flush(force=True)

    def capture_source_phase(stage, layer, actual_auxiliary_bytes):
        if cost_read_schedule is not None and stage == 'source_loading':
            cost_read_schedule.enter_phase(f'cost_capture_{layer:03d}', len(completed_checkpoint_units))
        retained_source_phase(stage, layer, actual_auxiliary_bytes)

    _log(f"boundary capture: calib {tuple(calib_ids.shape)} in "
         f"{len(row_offsets)} partition(s) across {runner.num_layers} layers ...")
    capture_started = time.time()
    batches = []
    if boundary_storage is not None:
        boundary_storage.watch_auxiliary(batches, [])
    if boundary_storage is not None and boundary_storage.config.get("capture_order") == "layer_major":
        retained_source_phase('source_loading', -1, 0)
        batches = runner.capture_layer_major_boundaries(
            [calib_ids[offset:offset + batch_rows] for offset in row_offsets],
            storage=boundary_storage,
            **({'source_phase': capture_source_phase} if retained_budget is not None else {}))
        retries = runner.layer_major_prefetch_retries
        if retries:
            _log(f"layer-major capture re-read {len(retries)} layer(s) whose speculative "
                 f"prefetch the runner no longer held at install: {list(retries)}")
    else:
        for batch_index, offset in enumerate(row_offsets):
            available_gib = _free_gib()
            if available_gib < min_free_gib:
                raise RuntimeError(f"free UMA {available_gib:.1f} < floor {min_free_gib:.1f}; "
                                   f"abort before calibration row {offset}")
            if boundary_storage is None:
                batches.append(runner.capture_boundaries(calib_ids[offset:offset + batch_rows]))
            else:
                def write_boundary(depth, tensor):
                    return boundary_storage.write(tensor, batch_index=batch_index, boundary_index=depth)

                def check_capture_state(current, state):
                    boundary_storage.check_auxiliary([*batches, current], extra=(state,),
                        shared_extra=state if current.shared_pass_state is None else ())

                batches.append(runner.capture_boundaries(calib_ids[offset:offset + batch_rows],
                    boundary_writer=write_boundary, resource_check=check_capture_state))
                boundary_storage.check_auxiliary(batches)
    _report_units()
    _log(f"boundary capture done in {(time.time() - capture_started) / 60:.1f} "
         f"min; starting {n_probes}-probe tail cotangents")
    device = runner.device
    dtype = runner.dtype

    # One existing shared-state instance per complete sequence/probe partition.
    # Exact-artifact mode additionally caps these retained auxiliary tensors;
    # the legacy mode keeps its original full-calibration host ownership.
    from prismaquant.sensitivity_probe import (
        SharedStateCotangents,
        kv_cotangent_path_enabled,
    )
    cotangents = [[SharedStateCotangents(enabled=kv_cotangent_path_enabled())
                  for _ in batches] for _ in range(n_probes)]
    grad_outs = [[] for _ in range(n_probes)]
    if boundary_storage is not None:
        boundary_storage.watch_auxiliary(batches, cotangents)
        boundary_storage.check_auxiliary(batches, cotangents=cotangents)
    if cost_read_schedule is not None:
        cost_read_schedule.enter_phase('cost_tail', len(completed_checkpoint_units))
    _report_units()
    if retained_budget is not None:
        retained_source_phase('tail', runner.num_layers,
            boundary_storage.actual_auxiliary_bytes(batches, cotangents=cotangents))
    with prefetched_boundary_batches(boundary_storage, batches, runner.num_layers) as tail_batches:
        for batch_index, batch, tail_cpu, _unused in tail_batches:
            try:
                for probe_index in range(n_probes):
                    tail = tail_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
                    logits = runner.tail_logits(batch, tail)
                    if probe_layout is not None and list(logits.shape) != [
                        len(batch.input_ids), int(calib_ids.shape[1]), probe_layout["vocab_size"]
                    ]:
                        raise RuntimeError("streamed AURA tail differs from bound probe geometry")
                    probe = fisher_probe_scalar(
                        logits, seed=seed_base + probe_index, token_scope=token_scope,
                        temperature=temperature, distribution="rademacher",
                        **({"token_count_override": probe_layout["global_token_count"],
                            "global_row_offset": row_offsets[batch_index]}
                           if probe_layout is not None else {}),
                    )
                    probe.backward()
                    if tail.grad is None:
                        raise RuntimeError("streamed AURA tail produced no cotangent")
                    grad_outs[probe_index].append(tail.grad.detach().to("cpu") if boundary_storage is None
                        else boundary_storage.write(tail.grad, batch_index=batch_index,
                            boundary_index=runner.num_layers, probe_index=probe_index))
                    del logits, probe, tail
                if boundary_storage is not None:
                    boundary_storage.retire(batch.activations_cpu[-1])
                batch.activations_cpu[-1] = torch.empty(0)
            finally:
                tail_cpu = logits = probe = tail = None

    _report_units()
    reverse_started = time.time()
    reverse_layers_done = 0
    for layer in reversed(range(runner.num_layers)):
        if cost_read_schedule is not None:
            cost_read_schedule.enter_phase(f'cost_reverse_{layer:03d}_source', len(completed_checkpoint_units))
        _report_units()
        if retained_budget is not None:
            retained_source_phase('source_loading', layer,
                boundary_storage.actual_auxiliary_bytes(batches, cotangents=cotangents))
        runner.context.install(
            layer,
            require_prefetched=runner.require_prefetched_residency,
            **({'prefetch_following': False} if operator_windows is not None else {}),
        )
        _refresh_packed_layer_views(layer)
        _require_installed_render_sources(layer)
        # Forward boundary capture leaves the final lookahead window hot.
        # Keep that pipeline moving in the direction of this traversal: the
        # existing StreamingContext/LayerCache loads the next reverse layer
        # while the current layer performs its expensive anchor render and
        # adjoint probes.  Without this call, every layer after the retained
        # tail window falls through ensure_loaded()'s synchronous cold path.
        if operator_windows is None:
            runner.schedule_reverse_prefetch(layer)
        else:
            # The operator reservation covers exactly the explicit lookahead.
            # Adaptive cache top-up can otherwise enqueue additional owners.
            successors = range(max(0, layer-runner.prefetch_lookahead), layer)
            for successor in reversed(successors):
                runner.context.schedule_prefetch(successor)
            settle = getattr(runner.context, 'settle_prefetched_layers', None)
            if callable(settle):
                settle(successors)
            elif torch.device(runner.device).type == 'cuda':
                raise RuntimeError('joint operator replay requires source prefetch settlement')
        if retained_budget is not None:
            retained_source_phase('reverse', layer,
                boundary_storage.actual_auxiliary_bytes(batches, cotangents=cotangents))
        pending = [
            name for name in names_by_layer.get(layer, [])
            if name not in completed_checkpoint_units
        ]
        d_weights: dict[tuple[str, str], torch.Tensor] = {}
        parameter_members = {}
        parameters = {}
        measured = {}
        try:
            for name in pending:
                parameter = _source_parameter(linears[name])
                parameter.requires_grad_(operator_windows is None)
                parameters[id(parameter)] = parameter
                parameter_members.setdefault(id(parameter), []).append(name)
                g_trace[name] = 0.0
            with torch.no_grad():
                if operator_windows is not None:
                    pass
                elif anchor_renderer is not None and pending:
                    render_layer = getattr(
                        anchor_renderer, "render_layer", None
                    )
                    if not callable(render_layer):
                        raise TypeError(
                            "streamed production anchor renderer has no "
                            "callable render_layer()"
                        )
                    layer_modules = {
                        name: linears[name]
                        for name in names_by_layer.get(layer, [])
                    }
                    expected_pairs = {
                        (name, fmt)
                        for name in pending
                        for fmt in render_formats[name]
                    }
                    transient_render = getattr(
                        anchor_renderer, "render_layer_transient", None
                    )

                    def _consume_anchor(
                        *, qname, fmt, reference_weight, rendered_weight,
                        render_score,
                    ):
                        del render_score
                        name = str(qname)
                        canonical_fmt = fr.canonical_format_name(fmt)
                        key = (name, canonical_fmt)
                        if key not in expected_pairs:
                            raise RuntimeError(
                                "streamed production anchor consumer received "
                                f"unexpected pair {key}"
                            )
                        if key in d_weights:
                            raise RuntimeError(
                                "streamed production anchor consumer received "
                                f"duplicate pair {key}"
                            )
                        if not isinstance(rendered_weight, torch.Tensor):
                            raise TypeError(
                                "streamed production anchor renderer returned "
                                f"{type(rendered_weight).__name__} for "
                                f"{name}@{canonical_fmt}, expected Tensor"
                            )
                        if tuple(rendered_weight.shape) != tuple(
                            reference_weight.shape
                        ):
                            raise RuntimeError(
                                "streamed production anchor shape differs for "
                                f"{name}@{canonical_fmt}: rendered="
                                f"{tuple(rendered_weight.shape)} source="
                                f"{tuple(reference_weight.shape)}"
                            )
                        _record_joint_operator(name, canonical_fmt, reference_weight, rendered_weight)
                        d_weights[key] = _stored_production_anchor_delta(
                            rendered_weight,
                            reference_weight,
                            storage_dtype=dw_torch_dtype,
                        )
                        if key in diagnostic_pairs:
                            weight_mse_diagnostic[key] = float(
                                d_weights[key].float().pow(2).mean().item()
                            )
                        dw_src[key] = "production_render"
                        s2[key] = 0.0
                        s4[key] = 0.0
                        x2_probe[key] = []
                        return {
                            "operation": "fp32_subtract_then_store",
                            "storage_dtype": str(dw_torch_dtype),
                        }

                    if callable(transient_render):
                        observed_pairs = {
                            (str(name), fr.canonical_format_name(fmt))
                            for name, fmt in transient_render(
                                layer=layer,
                                modules=layer_modules,
                                formats_by_qname={
                                    name: render_formats[name]
                                    for name in pending
                                },
                                consume_render=_consume_anchor,
                                consumer_identity=(
                                    {**AURA_PRODUCTION_ANCHOR_DELTA_CONSUMER_IDENTITY,
                                     "storage_dtype": "torch.float32"}
                                    if joint_activation else AURA_PRODUCTION_ANCHOR_DELTA_CONSUMER_IDENTITY
                                ),
                            )
                        }
                    else:
                        rendered_anchors = render_layer(
                            layer=layer,
                            modules=layer_modules,
                            formats_by_qname={
                                name: render_formats[name]
                                for name in pending
                            },
                        )
                        if not isinstance(rendered_anchors, Mapping):
                            raise TypeError(
                                "streamed production anchor render_layer() "
                                "must return a Mapping"
                            )
                        observed_pairs = {
                            (str(name), fr.canonical_format_name(fmt))
                            for name, fmt in rendered_anchors
                        }
                        # Compatibility for injected/research renderers. The
                        # production renderer above consumes each pair before
                        # the next pair is materialized.
                        rendered_anchor_pool = (
                            rendered_anchors
                            if isinstance(rendered_anchors, MutableMapping)
                            else dict(rendered_anchors)
                        )
                        del rendered_anchors
                        for name in pending:
                            weight = linears[name].weight.data
                            for fmt in render_formats[name]:
                                key = (name, fmt)
                                rendered = rendered_anchor_pool.pop(key)
                                _consume_anchor(
                                    qname=name,
                                    fmt=fmt,
                                    reference_weight=weight,
                                    rendered_weight=rendered,
                                    render_score={},
                                )
                                del rendered
                        del rendered_anchor_pool
                    if observed_pairs != expected_pairs:
                        raise RuntimeError(
                            "streamed production anchor renderer returned a "
                            "different pair set: "
                            f"missing={sorted(expected_pairs - observed_pairs)[:8]} "
                            f"unexpected={sorted(observed_pairs - expected_pairs)[:8]}"
                        )
                    for name in pending:
                        weight = linears[name].weight.data
                        source_lookup = getattr(
                            anchor_renderer,
                            "source_weight_identity_for",
                            None,
                        )
                        if callable(source_lookup):
                            source_weight_identity[name] = dict(
                                source_lookup(name)
                            )
                        else:
                            # Test/injected renderers may not own a CB cache.
                            # Production uses the renderer's already-computed
                            # source binding and therefore never hashes twice.
                            from prismaquant.production_weight_cache import (
                                _source_weight_value_identity,
                            )

                            source_shape, source_sha256 = (
                                _source_weight_value_identity(weight)
                            )
                            source_weight_identity[name] = {
                                "shape": source_shape,
                                "sha256": source_sha256,
                            }
                    _release_streamed_anchor_allocator_cache(device)
                else:
                    if joint_activation and pending:
                        joint_prefetch_stats.append(prefetch_joint_cache(
                            production_cache, pending, render_formats,
                            max_resident_bytes=max(0, int((_free_gib() - min_free_gib) * 1024**3)),
                        ))
                    for name in pending:
                        for fmt in render_formats[name]:
                            if joint_activation:
                                weight = linears[name].weight.data
                                rendered = production_cache.get(name, fmt)
                                _record_joint_operator(name, fmt, weight, rendered)
                                result = (rendered.to(weight.device, torch.float32) - weight.float(), "rendered")
                                del rendered
                            else:
                                result = _delta_w(
                                    name, fmt, linears[name].weight.data,
                                    production_cache, strict=require_production_cache,
                                )
                            if result is None:
                                continue
                            delta, source = result
                            key = (name, fmt)
                            d_weights[key] = delta.to(dw_torch_dtype)
                            dw_src[key] = source
                            s2[key] = 0.0
                            s4[key] = 0.0
                            x2_probe[key] = []
                if joint_activation:
                    for name in pending:
                        for fmt in unit_formats[name]:
                            if fmt in _ZERO_COST_FORMATS:
                                weight = linears[name].weight.data
                                _record_joint_operator(name, fmt, weight, weight)
            if anchor_renderer is None:
                compact = getattr(production_cache, "compact_for_pickle", None)
                if callable(compact):
                    compact()

            if operator_windows is not None:
                for name in pending:
                    for fmt in render_formats[name]:
                        key = (name, fmt)
                        dw_src[key] = 'rendered'
                        s2[key] = s4[key] = 0.0
                        x2_probe[key] = []

                def replay_backward(*, final, lease, probe=None):
                    active_probe = probe_index if probe is None else probe
                    with prefetched_boundary_batches(boundary_storage, batches, layer,
                            grad_outs[active_probe]) as reverse_batches:
                        for batch_index, batch, boundary_cpu, incoming_cpu in reverse_batches:
                            owner = cotangents[active_probe][batch_index]
                            replay_owner = None
                            try:
                                if not final:
                                    replay_owner = owner.fork_for_replay(
                                        max_resident_bytes=operator_windows['max_replay_cotangent_bytes'])
                                    owner = replay_owner
                                if boundary_storage is not None:
                                    boundary_storage.check_auxiliary(batches, cotangents=cotangents,
                                        extra=() if final else owner.resident_tensors())
                                if operator_guard is not None:
                                    check_operator_allocation(operator_guard, 'before_joint_window_backward', reserve_bytes=(
                                        operator_windows['workspace_reserve_bytes'] +
                                        (0 if lease is None else lease.statistics_capacity_bytes - lease.resident_statistics_bytes)))
                                if _free_gib() < min_free_gib:
                                    raise RuntimeError('joint window replay crossed free UMA floor')
                                cpu_rng = torch.get_rng_state()
                                cuda_rng = (torch.cuda.get_rng_state(device)
                                            if torch.device(device).type == 'cuda' else None)
                                incoming_grad = incoming_cpu.to(device)
                                x_in = boundary_cpu.to(device=device, dtype=dtype).detach().requires_grad_(True)
                                isolated = profile.isolated_layer_pass_state(
                                    batch.shared_pass_state, runner.layers[layer])
                                isolated = owner.graft(isolated)
                                out = runner.isolated_layer(batch, layer, x_in, pass_state=isolated)
                                roots, root_grads = owner.produced_roots()
                                torch.autograd.backward([out, *roots], [incoming_grad, *root_grads])
                                owner.harvest()
                                if not final:
                                    from prismaquant.streamed_boundary_artifacts import _state_storage_bytes
                                    if _state_storage_bytes(owner.resident_tensors()) > operator_windows['max_replay_cotangent_bytes']:
                                        raise RuntimeError('joint replay cotangents exceed their resident budget')
                                    if boundary_storage is not None:
                                        boundary_storage.check_auxiliary(batches, cotangents=cotangents,
                                            extra=owner.resident_tensors())
                                if not torch.equal(cpu_rng, torch.get_rng_state()) or (
                                        cuda_rng is not None and not torch.equal(cuda_rng, torch.cuda.get_rng_state(device))):
                                    raise RuntimeError('joint operator replay source consumed Torch RNG')
                                if x_in.grad is None:
                                    raise RuntimeError('joint operator replay produced no input cotangent')
                                if final:
                                    grad_outs[active_probe][batch_index] = (x_in.grad.detach().to('cpu')
                                        if boundary_storage is None else boundary_storage.write(x_in.grad,
                                            batch_index=batch_index, boundary_index=layer, probe_index=active_probe,
                                            previous=grad_outs[active_probe][batch_index]))
                                    if boundary_storage is not None:
                                        boundary_storage.check_auxiliary(batches, cotangents=cotangents)
                            finally:
                                boundary_cpu = incoming_cpu = None
                                out = x_in = incoming_grad = isolated = roots = root_grads = None
                                replay_owner = owner = None

                measured = {name: linears[name] for name in (
                    names_by_layer.get(layer, []) if retained_budget is not None else pending)
                    if render_formats[name]}
                source_seal = {name: JointOperatorStatisticsLease._source_fingerprint(module.weight)
                               for name, module in measured.items()}
                def consume_window_probe(probe_index, terms, diagnostics, receipt):
                    operator_window_receipts.append(dict(layer=layer, probe_index=probe_index, **receipt))
                    price.consume_probe(probe_index, terms, diagnostics)
                if retained_budget is not None and measured:
                    observe_and_project_retained_windows(
                        measured, {name: {fmt: fr.get_format(fmt) for fmt in render_formats[name]}
                                   for name in measured}, production_cache, operator_windows,
                        retained_budget=retained_budget, n_probes=n_probes,
                        source_bytes=retained_operator_windows['source_reserve_bytes'],
                        backward=lambda *, probe_index, final, lease: replay_backward(
                            final=final, lease=lease, probe=probe_index),
                        record_operator=_record_joint_operator,
                        consume_probe=consume_window_probe,
                        collect_col_energy=collect_col_energy, backend=joint_projection_backend,
                        guard=operator_guard, source_fingerprints=source_seal,
                        completed_names=set(measured) & completed_checkpoint_units,
                        sealed_windows=(cost_read_schedule.windows_for_layer(layer)
                                        if cost_read_schedule is not None else
                                        (preflight_retained_windows or {}).get(layer)),
                        before_window=(None if cost_read_schedule is None else
                            lambda index, names: cost_read_schedule.enter_phase(
                                f'cost_reverse_{layer:03d}_window_{index:03d}', len(completed_checkpoint_units))),
                        after_window=lambda index, names: commit_streamed_units(names))
                else:
                    for probe_index in range(n_probes):
                        if not measured:
                            replay_backward(final=True, lease=None)
                            continue
                        terms, diagnostics, receipt = observe_and_project_windows(
                            measured, {name: {fmt: fr.get_format(fmt) for fmt in render_formats[name]}
                                       for name in measured}, production_cache, operator_windows,
                            backward=replay_backward, record_operator=_record_joint_operator,
                            collect_col_energy=collect_col_energy, backend=joint_projection_backend,
                            guard=operator_guard, source_fingerprints=source_seal)
                        consume_window_probe(probe_index, terms, diagnostics, receipt)
            else:
                # Project each fully accumulated parameter gradient from its
                # post-accumulate hook, then clear ``param.grad`` immediately.
                # A routed layer can otherwise retain another complete 12-GiB
                # BF16 parameter plane on top of its source weights and dW menu.
                # Each reduction below is per parameter, so performing it when
                # that parameter's AccumulateGrad node completes is numerically
                # identical to the old post-backward qname loop.
                harvested: set[str] = set()
                accumulated_gradients: dict[str, torch.Tensor] = {}

                def _consume_streamed_gradient(name, gradient):
                    if probe_microbatch:
                        with torch.no_grad():
                            if name in accumulated_gradients:
                                accumulated_gradients[name].add_(gradient)
                            else:
                                accumulated_gradients[name] = gradient.to(torch.float32, copy=True)
                    else:
                        _harvest_streamed_gradient(name, gradient)


                def _harvest_streamed_gradient(
                    name: str, gradient: torch.Tensor
                ) -> None:
                    if name in harvested:
                        raise RuntimeError(
                            "streamed AURA harvested a parameter twice in one "
                            f"probe: {name}"
                        )
                    with torch.no_grad():
                        gradient_fp32 = gradient.float()
                        g_trace[name] += float(
                            (gradient_fp32 * gradient_fp32).sum().item()
                        )
                        if collect_col_energy:
                            energy = (
                                gradient_fp32 * gradient_fp32
                            ).sum(dim=0)
                            previous = col_energy.get(name)
                            col_energy[name] = (
                                energy if previous is None else previous + energy
                            )
                        for fmt in render_formats[name]:
                            key = (name, fmt)
                            if key not in d_weights:
                                continue
                            if joint_activation:
                                # Output hooks retain all signed local components;
                                # this parameter hook owns diagnostics only.
                                continue
                            projection = float(
                                (
                                    gradient_fp32
                                    * d_weights[key].float()
                                ).sum().item()
                            )
                            value = projection ** 2
                            s2[key] += value
                            s4[key] += value * value
                            x2_probe[key].append(value)
                    harvested.add(name)

                def _make_streamed_gradient_hook(member_names):
                    def _hook(parameter: torch.Tensor) -> None:
                        gradient = parameter.grad
                        if gradient is None:
                            return
                        for name in member_names:
                            if name not in harvested:
                                _consume_streamed_gradient(name, _source_gradient(linears[name], gradient))
                        parameter.grad = None

                    return _hook

                hook_handles = []
                joint_lease = None
                # Completed layers still propagate cotangents to pending earlier
                # layers, but have no target device or projections to lease.
                if joint_activation and pending:
                    cache_owner = production_cache if production_cache is not None else getattr(anchor_renderer, "cache", None)
                    joint_lease = SignedJointProjectionLease(
                        {name: linears[name] for name in pending},
                        {name: {fmt: fr.get_format(fmt) for fmt in render_formats[name]} for name in pending},
                        d_weights, activation_max_abs=getattr(cache_owner, "activation_max_abs", None),
                        projection_backend=joint_projection_backend,
                        attribution=attribution_config is not None,
                        attribution_keys=attribution_keys,
                    )
                try:
                    if joint_lease is not None:
                        joint_lease.__enter__()
                    for parameter_id, parameter in parameters.items():
                        hook_handles.append(
                            parameter.register_post_accumulate_grad_hook(
                                _make_streamed_gradient_hook(parameter_members[parameter_id])
                            )
                        )
                    for probe_index in range(n_probes):
                        harvested.clear()
                        accumulated_gradients.clear()
                        if joint_lease is not None:
                            joint_lease.begin_probe()
                        with prefetched_boundary_batches(boundary_storage, batches, layer,
                                grad_outs[probe_index]) as reverse_batches:
                            for batch_index, batch, boundary_cpu, incoming_cpu in reverse_batches:
                                if attribution_config is not None:
                                    # Whole caller-owned batch blocks from the
                                    # streamed partition's own offsets; never a
                                    # split of flattened rows.
                                    joint_lease.note_block(
                                        batch_index,
                                        first_sequence=row_offsets[batch_index],
                                        sequences=len(batch.input_ids))
                                try:
                                    available_gib = _free_gib()
                                    if available_gib < min_free_gib:
                                        raise RuntimeError(
                                            f"free UMA {available_gib:.1f} < floor "
                                            f"{min_free_gib:.1f}; abort before streamed "
                                            f"layer {layer} probe {probe_index + 1}"
                                        )
                                    incoming_grad = incoming_cpu.to(device)
                                    x_in = boundary_cpu.to(
                                        device=device, dtype=dtype
                                    ).detach().requires_grad_(True)
                                    isolated = profile.isolated_layer_pass_state(
                                        batch.shared_pass_state, runner.layers[layer]
                                    )
                                    isolated = cotangents[probe_index][batch_index].graft(isolated)
                                    out = runner.isolated_layer(
                                        batch, layer, x_in, pass_state=isolated
                                    )
                                    roots, root_grads = cotangents[probe_index][batch_index].produced_roots()
                                    if roots:
                                        torch.autograd.backward(
                                            [out, *roots],
                                            [incoming_grad, *root_grads],
                                        )
                                    else:
                                        out.backward(incoming_grad)
                                    cotangents[probe_index][batch_index].harvest()
                                    if x_in.grad is None:
                                        raise RuntimeError(
                                            f"streamed AURA layer {layer} produced no input "
                                            "cotangent"
                                        )
                                    # Replace this probe's consumed incoming cotangent now.
                                    # The former next_grad_outs list retained all 32 incoming
                                    # CPU tensors while growing a second complete outgoing
                                    # plane. In-place rollover bounds the CPU plane to 32
                                    # tensors plus the one result currently being copied.
                                    grad_outs[probe_index][batch_index] = (x_in.grad.detach().to("cpu")
                                        if boundary_storage is None else boundary_storage.write(x_in.grad,
                                            batch_index=batch_index, boundary_index=layer, probe_index=probe_index,
                                            previous=grad_outs[probe_index][batch_index]))
                                    if boundary_storage is not None:
                                        boundary_storage.check_auxiliary(batches, cotangents=cotangents)
                                    for parameter_id, parameter in parameters.items():
                                        gradient = parameter.grad
                                        if gradient is not None:
                                            # Defensive straggler path for a backend that did
                                            # not invoke the post-accumulate hook. It performs
                                            # the identical reduction and still frees the
                                            # gradient before the next probe.
                                            for name in parameter_members[parameter_id]:
                                                if name not in harvested:
                                                    _consume_streamed_gradient(name, _source_gradient(linears[name], gradient))
                                            parameter.grad = None
                                    del (out, x_in, incoming_grad, isolated, roots, root_grads)
                                finally:
                                    boundary_cpu = incoming_cpu = None
                                    out = x_in = incoming_grad = isolated = roots = root_grads = None
                        for name in list(accumulated_gradients):
                            _harvest_streamed_gradient(name, accumulated_gradients.pop(name))
                        for name in pending:
                            if name in harvested:
                                continue
                            # A routed expert not selected by this probe has an
                            # exact zero projection. Record the sample explicitly
                            # so route-sparse and never-routed experts retain the
                            # same K-probe rows as the legacy post-backward loop.
                            for fmt in render_formats[name]:
                                key = (name, fmt)
                                if key in d_weights and not joint_activation:
                                    x2_probe[key].append(0.0)
                        if joint_lease is not None:
                            for key, terms in joint_lease.finish_probe().items():
                                joint_components[key].append(terms)
                                # Same squaring as make_joint_aura_entry: the
                                # checkpoint reload compares both lists exactly.
                                value = squared_signed(terms["total"])
                                s2[key] += value
                                s4[key] += value * value
                                x2_probe[key].append(value)
                            if attribution_config is not None:
                                # The descriptive sidecar probe: per-invocation
                                # contractions, a separate arithmetic from the
                                # authoritative totals above.
                                captured = joint_lease.finish_attribution()
                                attribution_blocks = captured["blocks"]
                                for key, per_block in captured["components"].items():
                                    attribution_components.setdefault(key, []).append(per_block)
                finally:
                    accumulated_gradients.clear()
                    if joint_lease is not None:
                        joint_lease.__exit__(None, None, None)
                        joint_lease = None
                    for handle in hook_handles:
                        handle.remove()

            commit_streamed_units(pending)

            # This layer's input boundary will not be read again in the reverse
            # sweep (the next iteration consumes boundary ``layer - 1``).
            # Release it progressively instead of retaining all 44 DSv4
            # hc_mult=4 boundary snapshots to the end.
            for batch in batches:
                if boundary_storage is not None:
                    boundary_storage.retire(batch.activations_cpu[layer])
                batch.activations_cpu[layer] = torch.empty(0)

            # Report measured progress without extrapolating a layer rate:
            # resumed layers and dense/MoE layers can have very different costs.
            reverse_layers_done += 1
            if pending:
                elapsed = time.time() - reverse_started
                remaining = runner.num_layers - reverse_layers_done
                _log(
                    f"reverse layer {layer} done "
                    f"({reverse_layers_done}/{runner.num_layers}, "
                    f"{len(pending)} unit(s), {elapsed / 60:.1f} min elapsed, "
                    f"{remaining} layer(s) remaining)"
                )
        finally:
            for parameter in parameters.values():
                parameter.grad = None
                parameter.requires_grad_(False)
            # Loop locals and the lease otherwise retain the last layer's
            # delta plane through the next source install/render window.
            d_weights.clear()
            result = delta = weight = None
            del d_weights
            measured.clear()
            parameters.clear()
            parameter = None
            runner.context.unload(layer)
            _refresh_packed_layer_views(layer)

    return _finish_streamed_payload()


def run_streamed_production_anchor_aura(
    runner,
    calib_ids: torch.Tensor,
    *,
    formats_by_qname: Mapping[str, Sequence[str]],
    render_purposes_by_qname: Mapping[
        str, Mapping[str, str | Sequence[str]]
    ] | None = None,
    unmeasured_formats_by_qname: Mapping[str, Sequence[str]] | None = None,
    activation_index,
    render_levers: Mapping[str, object],
    col_weights: Mapping[str, torch.Tensor],
    calibration_hash: str,
    arm_identity: Mapping[str, object],
    model_identity: Mapping[str, object],
    checkpoint_dir: str | Path,
    resume: bool,
    n_probes: int,
    min_free_gib: float = 20.0,
    probe_microbatch: int = 0,
    token_scope: str = "all",
    temperature: float = 1.0,
    seed_base: int = 7000,
    cold_expert_provenance: Mapping[str, object] | None = None,
    max_act_rows: int = 512,
    checkpoint_identity_extra: Mapping[str, object] | None = None,
    include_routed_experts: bool = True,
    allow_packed_expert_omission: bool = False,
    collect_col_energy: bool = False,
    joint_activation: bool = False,
    joint_projection_backend=None,
    boundary_storage=None,
    profile=None,
) -> dict:
    """Run one streamed KL adjoint over an exact production-anchor plan.

    This is the concrete campaign API. ``formats_by_qname`` may contain
    several measured CB anchors for a unit (anchor, fitting-panel, and held-out
    validation cells) plus an exact identity-bearing terminal.  A terminal
    declared in ``unmeasured_formats_by_qname`` is never sent to the renderer
    and receives no synthetic scalar unless it is independently covered by
    the exact zero-cost passthrough contract.  Every other listed cell is
    rendered in the fixed production arm exactly once unless its
    identity-bound per-unit AURA shard is resumed.

    The returned ordinary AURA payload carries scalar ``predicted_dloss`` rows
    and exact renderer/plan provenance.  Rendered weights and ``dW`` live only
    for the current reverse layer and are discarded before it is unloaded.

    On a routed-expert model whose packed experts are priced by the empirical
    unit-KL path, pass ``include_routed_experts=False`` together with
    ``allow_packed_expert_omission=True``; the payload then carries no
    routed-expert rows and the caller must merge the empirical rows before
    allocation (the packed-expert coverage guard raises otherwise).
    """
    if checkpoint_dir is None:
        raise ValueError(
            "production-anchor AURA requires a durable checkpoint_dir"
        )
    profile = resolve_routed_expert_profile(
        runner.model, profile or runner.profile
    )
    canonical_plan: dict[str, tuple[str, ...]] = {}
    render_plan: dict[str, tuple[str, ...]] = {}
    canonical_unmeasured: dict[str, tuple[str, ...]] = {}
    format_union: list[str] = []
    for raw_name, raw_formats in formats_by_qname.items():
        name = str(raw_name)
        if isinstance(raw_formats, (str, bytes)):
            raise TypeError(
                f"production-anchor formats for {name} must be a sequence"
            )
        formats = tuple(dict.fromkeys(
            fr.canonical_format_name(fmt) for fmt in raw_formats
        ))
        if not formats:
            raise ValueError(
                f"production-anchor formats are empty for {name}"
            )
        raw_unmeasured = (
            unmeasured_formats_by_qname.get(name, ())
            if isinstance(unmeasured_formats_by_qname, Mapping) else ()
        )
        if isinstance(raw_unmeasured, (str, bytes)):
            raise TypeError(
                f"unmeasured production-anchor formats for {name} must be a sequence"
            )
        retained = tuple(dict.fromkeys(
            fr.canonical_format_name(fmt) for fmt in raw_unmeasured
        ))
        unexpected_retained = sorted(set(retained) - set(formats))
        if unexpected_retained:
            raise ValueError(
                f"unmeasured production-anchor formats for {name} are "
                f"outside its exact plan: {unexpected_retained}"
            )
        measured = tuple(
            fmt for fmt in formats
            if fmt not in _ZERO_COST_FORMATS and fmt not in set(retained)
        )
        if not measured:
            raise ValueError(
                f"production-anchor unit {name} has no real rendered anchor"
            )
        canonical_plan[name] = formats
        render_plan[name] = measured
        canonical_unmeasured[name] = retained
        for fmt in formats:
            if fmt not in format_union:
                format_union.append(fmt)
    if isinstance(unmeasured_formats_by_qname, Mapping):
        unexpected_names = sorted(
            set(unmeasured_formats_by_qname) - set(canonical_plan)
        )
        if unexpected_names:
            raise ValueError(
                "unmeasured production-anchor formats contain unplanned "
                f"qnames; sample={unexpected_names[:8]}"
            )

    allowed_purposes = frozenset({"anchor", "panel", "validation"})
    canonical_purposes: dict[str, dict[str, list[str]]] = {}
    raw_purposes = render_purposes_by_qname
    for name, formats in render_plan.items():
        rows: dict[str, list[str]] = {}
        supplied = (
            raw_purposes.get(name, {})
            if isinstance(raw_purposes, Mapping) else {}
        )
        if not isinstance(supplied, Mapping):
            raise TypeError(
                f"production-anchor purposes for {name} must be a mapping"
            )
        for fmt in formats:
            raw = supplied.get(fmt, "anchor")
            values = [raw] if isinstance(raw, str) else list(raw)
            purposes = list(dict.fromkeys(str(value) for value in values))
            invalid = sorted(set(purposes) - allowed_purposes)
            if not purposes or invalid:
                raise ValueError(
                    f"production-anchor purposes for {name}@{fmt} are "
                    f"invalid: {purposes}; allowed={sorted(allowed_purposes)}"
                )
            rows[fmt] = purposes
        unexpected = sorted(set(str(fmt) for fmt in supplied) - set(formats))
        if unexpected:
            raise ValueError(
                f"production-anchor purposes contain unrendered cells for "
                f"{name}: {unexpected}"
            )
        canonical_purposes[name] = rows
    if isinstance(raw_purposes, Mapping):
        unexpected_names = sorted(set(raw_purposes) - set(render_plan))
        if unexpected_names:
            raise ValueError(
                "production-anchor purposes contain unplanned qnames; "
                f"sample={unexpected_names[:8]}"
            )

    from prismaquant.streaming_production_cache import (
        StreamedProductionAnchorRenderer,
    )

    renderer = StreamedProductionAnchorRenderer(
        runner.model,
        act_index=activation_index,
        formats_by_qname=render_plan,
        levers=render_levers,
        profile=profile,
        device=runner.device,
        col_weights=col_weights,
        calibration_hash=calibration_hash,
        arm_identity=arm_identity,
        model_identity=model_identity,
        cold_expert_provenance=cold_expert_provenance,
        max_act_rows=max_act_rows,
        transient_consumer_identity=(
            {**AURA_PRODUCTION_ANCHOR_DELTA_CONSUMER_IDENTITY,
             "storage_dtype": "torch.float32"}
            if joint_activation else AURA_PRODUCTION_ANCHOR_DELTA_CONSUMER_IDENTITY
        ),
    )
    extra = dict(checkpoint_identity_extra or {})
    reserved_extra = {
        "production_anchor_render_purposes",
        "production_anchor_unmeasured_formats_by_qname",
    } & set(extra)
    if reserved_extra:
        raise ValueError(
            "checkpoint_identity_extra cannot override production-anchor "
            f"plan identity fields: {sorted(reserved_extra)}"
        )
    extra["production_anchor_render_purposes"] = canonical_purposes
    extra["production_anchor_unmeasured_formats_by_qname"] = {
        name: list(values)
        for name, values in sorted(canonical_unmeasured.items())
    }
    payload = compute_aura_cost_streamed(
        runner,
        calib_ids,
        format_union,
        n_probes=n_probes,
        min_free_gib=min_free_gib,
        probe_microbatch=probe_microbatch,
        token_scope=token_scope,
        temperature=temperature,
        production_cache=None,
        seed_base=seed_base,
        require_production_cache=True,
        # DSv4's worst routed layer carries two full anchor planes. FP32
        # subtraction + BF16 storage retains the validated AURA contract
        # (the projection below upcasts) while keeping Spark above the host's
        # 3-GiB guardian floor.
        dw_dtype="bfloat16",
        include_lm_head=False,
        allow_packed_expert_omission=allow_packed_expert_omission,
        collect_col_energy=collect_col_energy,
        joint_activation=joint_activation,
        joint_projection_backend=joint_projection_backend,
        boundary_storage=boundary_storage,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        model_identity=model_identity,
        checkpoint_identity_extra=extra,
        formats_by_qname=canonical_plan,
        unmeasured_formats_by_qname=canonical_unmeasured,
        anchor_renderer=renderer,
        include_routed_experts=include_routed_experts,
        diagnostic_weight_mse_pairs=[
            (name, fmt)
            for name, rows in canonical_purposes.items()
            for fmt, purposes in rows.items()
            if "panel" in purposes
        ],
        profile=profile,
    )
    counts = {
        purpose: sum(
            purpose in purposes
            for rows in canonical_purposes.values()
            for purposes in rows.values()
        )
        for purpose in sorted(allowed_purposes)
    }
    payload["provenance"].update({
        "production_anchor_render_purposes": canonical_purposes,
        "production_anchor_unmeasured_formats_by_qname": {
            name: list(values)
            for name, values in sorted(canonical_unmeasured.items())
        },
        "production_anchor_purpose_counts": counts,
        "production_anchor_union_render_count": sum(
            len(rows) for rows in canonical_purposes.values()
        ),
    })
    return payload


def _stage_aura_model(model_path: str) -> str:
    """Return AURA's canonical text-only execution view of a checkpoint.

    Wrapper checkpoints must execute the nested ``text_config`` values, while
    already-flattened sources stay on their original path.  The shared stager
    also owns process-exit cleanup of any temporary staged tree, keeping it
    alive for the complete streamed AURA run.
    """
    from prismaquant.sensitivity_probe import stage_text_only

    return stage_text_only(model_path)


def main(argv: Sequence[str] | None = None) -> int:
    from prismaquant.model_runtime import initialize_model_runtime
    initialize_model_runtime()
    p = argparse.ArgumentParser(description="Streamed production-anchor AURA cost")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--streaming", action="store_true", required=True)
    p.add_argument("--production-anchors", action="store_true", required=True)
    p.add_argument("--probe", required=True, help="Probe with the render calibration hash.")
    p.add_argument("--activation-cache-dir", required=True, help="Probe act/ rows.")
    p.add_argument("--formats", default="NVFP4,FP8_DYNAMIC,BF16")
    p.add_argument("--enable", default="gptq,static_act_order,joint_scale_opt")
    p.add_argument("--disable", default="")
    p.add_argument("--col-weights", default=None)
    p.add_argument("--max-act-rows", type=int, default=512)
    p.add_argument("--skip-qnames", nargs="*", default=[])
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--unit-filter", default="")
    p.add_argument("--streaming-offload-dir", default=None)
    p.add_argument("--n-probes", type=int, default=16)
    p.add_argument("--n-calib-samples", type=int, default=4)
    p.add_argument("--calib-seqlen", type=int, default=256)
    p.add_argument("--calib-split", default="train")
    p.add_argument("--calib-seed", type=int, default=42)
    p.add_argument("--dataset", default=None)
    p.add_argument("--calibration-input", default=None)
    p.add_argument("--calibration-input-sha256", default=None)
    p.add_argument("--token-scope", default="all")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "auto"],
                   help="Streamed source dtype. Auto selects bfloat16.")
    p.add_argument("--min-free-gib", type=float, default=20.0)
    p.add_argument("--seed-base", type=int, default=7000)
    p.add_argument("--include-lm-head", action="store_true")
    p.add_argument("--allow-packed-expert-omission", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cost-mode", default="")
    args = p.parse_args(argv)
    if bool(args.calibration_input) != bool(args.calibration_input_sha256):
        p.error("--calibration-input and --calibration-input-sha256 are required together")
    if args.calibration_input and args.dataset:
        p.error("--calibration-input and --dataset are mutually exclusive")
    if args.include_lm_head:
        p.error("streamed AURA does not support --include-lm-head")
    pin_fla_configs()
    from prismaquant.gpu_guard import require_cuda_hot_path
    require_cuda_hot_path("aura_cost", args.device)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    from transformers import AutoTokenizer
    from prismaquant.build_production_cache import _load_col_weights
    from prismaquant.calibration_data import load_calibration, load_calibration_input, load_wikitext_calibration_windowed
    from prismaquant.cost_streaming import build_streamed_causal_lm
    from prismaquant.measure_quant_cost import ActivationIndex
    from prismaquant.model_profiles import detect_profile
    from prismaquant.streamed_model_identity import (
        build_source_model_binding, build_streamed_model_identity,
        resolve_source_checkpoint_path, source_model_decode_contract,
    )

    source_root = resolve_source_checkpoint_path(args.model)
    native_source = source_model_decode_contract(source_root) is not None
    staged = _stage_aura_model(str(source_root))
    profile = detect_profile(staged)
    if args.dtype == "auto":
        args.dtype = "bfloat16"
    dt = torch.float32 if args.dtype == "float32" else torch.bfloat16
    if native_source and dt == torch.float32:
        p.error("native FP8 streaming requires --dtype bfloat16. Set AURA_COST_DTYPE=bfloat16.")
    tok = AutoTokenizer.from_pretrained(
        staged, trust_remote_code=True, local_files_only=Path(staged).exists())
    calibration_input_receipt = None
    if args.calibration_input:
        calib, calibration_input_receipt = load_calibration_input(
            args.calibration_input, expected_sha256=args.calibration_input_sha256,
            n_samples=args.n_calib_samples, seqlen=args.calib_seqlen)
    elif args.dataset:
        calib = load_calibration(tok, args.dataset, args.n_calib_samples, args.calib_seqlen,
                                 calib_seed=args.calib_seed)
    else:
        calib = load_wikitext_calibration_windowed(
            tok, args.n_calib_samples, args.calib_seqlen,
            split=args.calib_split, seed=args.calib_seed)
    calib = calib.to(args.device)
    with open(args.probe, "rb") as fh:
        probe = pickle.load(fh)
    render_calibration_hash = probe["meta"]["calib_hash"]
    requested_formats = list(dict.fromkeys(
        fr.canonical_format_name(f.strip()) for f in args.formats.split(",") if f.strip()))
    if not set(requested_formats) - _ZERO_COST_FORMATS:
        p.error("production anchors require a non-passthrough menu format")
    levers = {name.strip(): True for name in args.enable.split(",") if name.strip()}
    levers.update({name.strip(): False for name in args.disable.split(",") if name.strip()})
    checkpoint_dir = Path(args.checkpoint_dir)
    runner = build_streamed_causal_lm(
        str(source_root) if native_source else staged, device=torch.device(args.device),
        dtype=dt, offload_folder=args.streaming_offload_dir or str(checkpoint_dir / "offload"),
        profile=profile, attn_implementation="eager")
    try:
        targets = _target_linears(runner.model, profile=profile)
        plan = {name: requested_formats for name in sorted(targets)
                if name not in args.skip_qnames
                and (not args.unit_filter or re.search(args.unit_filter, name))}
        model_identity = build_streamed_model_identity(
            runner, str(source_root),
            identity_cache_path=checkpoint_dir / "streamed_model_identity.json")
        source_extra = ({"source_binding": build_source_model_binding(
            source_root, dtype=dt,
            streamed_identity_cache_path=checkpoint_dir / "streamed_model_identity.json")}
            if native_source else {})
        payload = run_streamed_production_anchor_aura(
            runner, calib, formats_by_qname=plan,
            activation_index=ActivationIndex(Path(args.activation_cache_dir), sorted(plan)),
            render_levers=levers,
            col_weights=_load_col_weights(args.col_weights, requested_formats) or {},
            calibration_hash=render_calibration_hash,
            arm_identity={"cost_mode": args.cost_mode or "aura",
                          "probe_calibration_hash": render_calibration_hash},
            model_identity=model_identity, checkpoint_dir=checkpoint_dir,
            resume=args.resume, n_probes=args.n_probes,
            token_scope=args.token_scope, temperature=args.temperature,
            seed_base=args.seed_base, max_act_rows=args.max_act_rows,
            min_free_gib=args.min_free_gib,
            include_routed_experts=False,
            allow_packed_expert_omission=args.allow_packed_expert_omission,
            checkpoint_identity_extra=source_extra, profile=profile)
    finally:
        runner.shutdown()
    payload["provenance"].update({
        "model": str(args.model), "dtype": args.dtype,
        "calib_source": args.calibration_input or args.dataset or f"wikitext:{args.calib_split}",
        "n_calib_samples": args.n_calib_samples, "calib_seqlen": args.calib_seqlen,
        "calib_seed": (calibration_input_receipt["provenance"].get("seed")
                       if calibration_input_receipt is not None else args.calib_seed),
        "cost_mode": args.cost_mode, "fla_configs": fla_config_identity(),
    })
    if calibration_input_receipt is not None:
        payload["provenance"]["calibration_input"] = calibration_input_receipt
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "wb") as fh:
        pickle.dump(payload, fh)
    _log(f"wrote {args.output}: {len(payload['costs'])} Linear rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
