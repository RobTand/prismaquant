#!/usr/bin/env python3
"""Content-keyed 43-layer DSV4 LDLQ ladder burn.

The method is selected mechanically from ``MINCHAIN_PILOT.json``.  This is a
campaign tool, not a production encoder entry point: incumbent and strict
min-chain measurement share the production free-fit renderer, warm-state,
per-slice dual-holdout law, and measured sliced fallback.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from prismaquant import format_registry as fr
from prismaquant.cb_minchain import (
    MINCHAIN_CONTEXT_VERSION,
    chain_identity_from_digest,
    embed_predecessor,
    recipe_solution_digest,
    select_arm,
)
from prismaquant.cb_warm_state import (
    CBWarmStateStore,
    build_warm_record,
    tensor_value_identity,
)
from prismaquant.expert_empirical_cost import _cb_ladder_law
from prismaquant.layer_streaming import (
    _build_fp8_scale_inv_map,
    _build_weight_map,
    _read_layer_to_device,
)
from prismaquant.nvfp4_cb_footprint import (
    cb_serialization_context_stamp,
    cb_fields_for_context,
)
from prismaquant.nvfp4_cb_formats import nvfp4_cb_reconstruct
from prismaquant.production_weight_cache import (
    canonical_cb_col_weights_sha256,
    validate_cb_render_source_weight,
)
from prismaquant.research_cost_acceptance import (
    RESEARCH_COST_MANIFEST_SCHEMA,
    RESEARCH_COST_PROVENANCE,
)
from tools.dsv4_ldlq_cost_campaign import (
    ACT_ROOT,
    ANCHORS,
    BY_LAYER,
    COL_WEIGHTS,
    CONTEXT,
    HOLDOUTS,
    PROJECTIONS,
    RUNGS,
    RUN_ROOT,
    SOURCE,
    TOLERANCE,
    atomic_json,
    atomic_pickle,
    atomic_text,
    content_sha256_float32,
    load_direct_activation,
    load_layer_identity,
    load_projection,
    per_slice_mse,
    percentile,
    sha256_file,
)
from tools.dsv4_minchain_pilot import _evaluate_all, _refine_all


LAYER_COUNT = 43
EXPERT_COUNT = 256
MEASURED_RUNGS = tuple(sorted(set(ANCHORS).union(HOLDOUTS)))
MISSING_RUNGS = tuple(k for k in RUNGS if k not in MEASURED_RUNGS)
BURN_ROOT = RUN_ROOT / "burn"
SHARD_ROOT = BURN_ROOT / "by-layer"
WARM_ROOT = BURN_ROOT / "warm-state"
PILOT_JSON = RUN_ROOT / "minchain-pilot/MINCHAIN_PILOT.json"
BASE_COST = Path(
    "/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2/"
    "artifacts-mxfp4/probe-k12k18/cost_probe_only.pkl"
)
OLD_FULL_COST = Path(
    "/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2/"
    "artifacts/cost_full.pkl"
)
PROBE = Path(
    "/home/rob/dq-runs/dsv4-flash-0731/prod-cal-0p6-v2/"
    "artifacts-mxfp4/probe.pkl"
)
REPORT_SCHEMA = "prismaquant.dsv4_ldlq_burn.v1"
SHARD_SCHEMA = "prismaquant.dsv4_ldlq_layer_shard.v1"
MANIFEST_SCHEMA = "prismaquant.dsv4_ldlq_burn_manifest.v1"
OVERHEAD_RESERVE_BYTES = 268_435_456


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _content_key(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _method_from_pilot() -> tuple[str, dict]:
    report = json.loads(PILOT_JSON.read_text())
    method = str(report["burn_method"])
    if method not in {"MIN-CHAIN STRICT", "INCUMBENT"}:
        raise AssertionError(f"unknown pilot burn decision {method!r}")
    mandatory = all(report["gates"][gate]["pass"] for gate in ("G1", "G2", "G3", "G4"))
    if (method == "MIN-CHAIN STRICT") != mandatory:
        raise AssertionError("pilot method does not match its mandatory gates")
    return method, report


def _split_fields(fields: Mapping[str, Any], count: int, rows: int) -> list[dict]:
    out = []
    for expert in range(count):
        start, stop = expert * rows, (expert + 1) * rows
        local = dict(fields)
        for key in ("indices", "scales", "signs", "scale_super", "scale_sub"):
            if isinstance(local.get(key), torch.Tensor):
                local[key] = local[key][start:stop].contiguous()
        local["shape"] = (rows, int(fields["shape"][-1]))
        out.append(local)
    return out


def _compact(fields: Mapping[str, Any]) -> dict:
    out = dict(fields)
    for key in ("indices", "scales", "signs", "scale_super", "scale_sub"):
        if isinstance(out.get(key), torch.Tensor):
            out[key] = out[key].clone().contiguous()
    codebook = out.get("codebook")
    if isinstance(codebook, torch.Tensor):
        out["codebook"] = codebook.clone().contiguous()
    elif codebook is not None:
        out["codebook"] = tuple(table.clone().contiguous() for table in codebook)
    return out


def _pack_solution(fields: Mapping[str, Any]) -> dict:
    """Resident checkpoint form: int16 indices, otherwise value-verbatim."""
    out = dict(fields)
    indices = out["indices"]
    # FP8-CB K28..K48 product subtables top out at 12 index bits; int16 is
    # therefore lossless by the registered format law (no device sync scan).
    out["indices"] = indices.to(torch.int16).contiguous()
    for key in ("scales", "signs", "scale_super", "scale_sub"):
        if isinstance(out.get(key), torch.Tensor):
            out[key] = out[key].clone().contiguous()
    codebook = out.get("codebook")
    if isinstance(codebook, torch.Tensor):
        out["codebook"] = codebook.clone().contiguous()
    elif codebook is not None:
        out["codebook"] = tuple(table.clone().contiguous() for table in codebook)
    return out


def _unpack_solution(fields: Mapping[str, Any]) -> dict:
    out = dict(fields)
    out["indices"] = out["indices"].to(torch.int64).contiguous()
    return out


def _projection_subset(data: Mapping[str, Any], experts: Sequence[int]) -> dict:
    index = torch.tensor(list(experts), device=data["weight"].device)
    return {
        "weight": data["weight"].index_select(0, index).contiguous(),
        "col_weights": data["col_weights"].index_select(0, index).contiguous(),
        "activation_rows": tuple(data["activation_rows"][i] for i in experts),
        "qnames": [data["qnames"][i] for i in experts],
    }


@torch.inference_mode()
def _per_slice_output_metrics(
    weight: torch.Tensor,
    reconstruction: torch.Tensor,
    activation_rows: Sequence[torch.Tensor],
    spec: Any,
) -> dict[str, list[float] | list[int]]:
    """Measure the allocator's activation-inclusive contract per slice.

    Activations stay in the existing CPU cache and are staged one slice at a
    time.  The GEMMs and activation QDQ remain GPU work; scalar synchronization
    is consolidated at the end so this audit does not turn the encoder hot
    path into a CPU-bound loop.
    """
    output_mse: list[torch.Tensor] = []
    relative_mse: list[torch.Tensor] = []
    row_counts: list[int] = []
    device = weight.device
    for index, cached in enumerate(activation_rows):
        if cached.ndim != 2 or not int(cached.shape[0]):
            raise AssertionError(f"slice {index}: activation rows absent")
        x = cached.to(device=device, dtype=torch.float32, non_blocking=True)
        x_hat = spec.activation_quantize_dequantize(x.clone())
        w = weight[index].float()
        w_hat = reconstruction[index].float()
        y_ref = x @ w.T
        y_q = x_hat @ w_hat.T
        mse = (y_ref - y_q).square().mean()
        ref_energy = y_ref.square().mean()
        output_mse.append(mse)
        relative_mse.append(mse / ref_energy.clamp_min(1e-12))
        row_counts.append(int(cached.shape[0]))
        del x, x_hat, w, w_hat, y_ref, y_q, mse, ref_energy
    columns = torch.stack(
        [torch.stack(output_mse), torch.stack(relative_mse)], dim=1
    ).cpu().tolist()
    return {
        "output_mse": [float(row[0]) for row in columns],
        "relative_mse": [float(row[1]) for row in columns],
        "n_activation_rows": row_counts,
    }


@torch.inference_mode()
def _selected_output_metrics(
    *, data: Mapping[str, Any], expert_ids: Sequence[int],
    solutions: Sequence[Mapping[str, Any]], rung: int,
) -> dict[str, list[float] | list[int]]:
    """Score selected min-chain solutions without retaining reconstructions."""
    subset = _projection_subset(data, expert_ids)
    spec = fr.get_format(f"FP8_CB_K{rung}")
    output_mse: list[torch.Tensor] = []
    relative_mse: list[torch.Tensor] = []
    row_counts: list[int] = []
    for index, fields in enumerate(solutions):
        reconstruction = nvfp4_cb_reconstruct(
            fields, rung, grid="fp8", mode="product"
        ).float()
        cached = subset["activation_rows"][index]
        if cached.ndim != 2 or not int(cached.shape[0]):
            raise AssertionError(f"slice {index}: activation rows absent")
        x = cached.to(
            device=subset["weight"].device, dtype=torch.float32,
            non_blocking=True,
        )
        x_hat = spec.activation_quantize_dequantize(x.clone())
        y_ref = x @ subset["weight"][index].float().T
        y_q = x_hat @ reconstruction.T
        mse = (y_ref - y_q).square().mean()
        output_mse.append(mse)
        relative_mse.append(mse / y_ref.square().mean().clamp_min(1e-12))
        row_counts.append(int(cached.shape[0]))
        del reconstruction, x, x_hat, y_ref, y_q, mse
    columns = torch.stack(
        [torch.stack(output_mse), torch.stack(relative_mse)], dim=1
    ).cpu().tolist()
    del subset
    return {
        "output_mse": [float(row[0]) for row in columns],
        "relative_mse": [float(row[1]) for row in columns],
        "n_activation_rows": row_counts,
    }


def _load_ordinary(
    qname: str, *, device: torch.device, identity: Mapping[str, Any],
    all_col_weights: Mapping[str, Any], model_to_shard: Mapping[str, str],
    model_to_ckpt: Mapping[str, str], scale_map: Mapping[str, Any],
) -> dict[str, Any]:
    weight_name = qname + ".weight"
    loaded = _read_layer_to_device(
        weight_name, model_to_shard, model_to_ckpt, torch.bfloat16, device,
        fp8_scale_inv_map=scale_map,
    )
    if set(loaded) != {weight_name}:
        raise AssertionError(f"{qname}: source resolved {sorted(loaded)}")
    weight = loaded[weight_name].to(torch.bfloat16).contiguous()
    validate_cb_render_source_weight(
        identity, qname, weight, where="DSV4 burn ordinary source"
    )
    col_weights = torch.as_tensor(all_col_weights[qname]).float().contiguous()
    if list(col_weights.shape) != list(identity["col_weights_shapes"][qname]):
        raise AssertionError(f"{qname}: col-weight shape mismatch")
    if content_sha256_float32(col_weights) != identity["col_weights_content_sha256"][qname]:
        raise AssertionError(f"{qname}: col-weight digest mismatch")
    activation = load_direct_activation(qname, int(weight.shape[1]))
    if not activation.shape[0]:
        raise AssertionError(f"{qname}: ordinary LDLQ activation rows absent")
    return {
        "weight": weight.unsqueeze(0),
        "col_weights": col_weights.reshape(1, 1, -1).to(device).contiguous(),
        "activation_rows": (activation,),
        "qnames": [qname],
    }


def _encode_free(
    *, layer: int, projection: str, rung: int, data: Mapping[str, Any],
    expert_ids: Sequence[int], keep_fields: bool, measure_output: bool,
    warm_identity_cache: dict[tuple[int, ...], tuple[Any, Any, str]],
) -> dict[str, Any]:
    subset = _projection_subset(data, expert_ids)
    spec = fr.get_format(f"FP8_CB_K{rung}")
    torch.cuda.synchronize()
    started = time.perf_counter()
    fields = cb_fields_for_context(
        spec, subset["weight"], context=CONTEXT,
        col_weights=subset["col_weights"],
        activation_rows=subset["activation_rows"],
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    reconstruction = nvfp4_cb_reconstruct(
        fields, rung, grid="fp8", mode="product"
    ).to(subset["weight"].dtype)
    errors = per_slice_mse(subset["weight"], reconstruction)
    output_metrics = (
        _per_slice_output_metrics(
            subset["weight"], reconstruction, subset["activation_rows"], spec
        ) if measure_output else None
    )
    # Warm records remain pure free-fit state. The chain dimension is carried
    # by layer-shard identities; a future chain-aware warm loader must match
    # that dimension before accepting these scales as a chain predecessor.
    ids_digest = hashlib.sha256(
        ",".join(str(i) for i in expert_ids).encode()
    ).hexdigest()[:16]
    logical_qname = (
        f"model.layers.{layer}.mlp.experts.{projection}.subset-{ids_digest}"
    )
    identity_key = tuple(int(value) for value in expert_ids)
    if identity_key not in warm_identity_cache:
        activation_hasher = hashlib.sha256()
        for qname, activation in zip(
            subset["qnames"], subset["activation_rows"]
        ):
            activation_hasher.update(qname.encode())
            activation_hasher.update(
                tensor_value_identity(torch.as_tensor(activation))[1].encode()
            )
        warm_identity_cache[identity_key] = (
            tensor_value_identity(subset["weight"]),
            tensor_value_identity(subset["col_weights"]),
            activation_hasher.hexdigest(),
        )
    source_identity, col_weights_identity, activation_digest = (
        warm_identity_cache[identity_key]
    )
    warm_path = CBWarmStateStore(WARM_ROOT).write(build_warm_record(
        qname=logical_qname,
        format_name=spec.name,
        source_weight=subset["weight"],
        col_weights=subset["col_weights"],
        context=CONTEXT,
        fields=fields,
        source_identity=source_identity,
        col_weights_identity=col_weights_identity,
    ))
    locals_ = (
        _split_fields(fields, len(expert_ids), int(subset["weight"].shape[1]))
        if keep_fields else None
    )
    context_stamp = cb_serialization_context_stamp(CONTEXT, formats=[spec.name])
    recipe_digests = [
        recipe_solution_digest({
            "kind": "free_fit",
            "qname": qname,
            "format": spec.name,
            "subset_source_digest": source_identity[1],
            "subset_col_weights_digest": col_weights_identity[1],
            "subset_activation_digest": activation_digest,
            "serialization_context": context_stamp,
        }) for qname in subset["qnames"]
    ]
    del reconstruction, fields, subset
    torch.cuda.empty_cache()
    return {
        "errors": errors,
        "fields": locals_,
        "elapsed_seconds": elapsed,
        "warm_state_path": str(warm_path),
        "recipe_digests": recipe_digests,
        "output_metrics": output_metrics,
    }


def _select_chain_rung(
    *, rung: int, data: Mapping[str, Any], expert_ids: Sequence[int],
    free: Mapping[str, Any], predecessors: Sequence[Mapping[str, Any]],
    predecessor_errors: Sequence[float], predecessor_ids: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    subset = _projection_subset(data, expert_ids)
    embedded = [embed_predecessor(value, rung) for value in predecessors]
    embed_errors, embed_seconds = _evaluate_all(subset["weight"], embedded, rung)
    for expert, observed, prior in zip(
        expert_ids, embed_errors, predecessor_errors
    ):
        rel = abs(observed - float(prior)) / max(abs(float(prior)), 1e-30)
        if rel > 2e-6:
            raise AssertionError(
                f"K{rung} expert {expert}: predecessor embed changed "
                f"reported metric by {rel:.3e}"
            )
    refined, refine_errors, refine_seconds = _refine_all(
        weight=subset["weight"], col_weights=subset["col_weights"],
        activation_rows=subset["activation_rows"],
        predecessors=predecessors, rung=rung,
    )
    selected, errors, identities, arms = [], [], [], []
    violations = tax = 0
    for local, expert in enumerate(expert_ids):
        arm, error = select_arm({
            "free": float(free["errors"][local]),
            "embed": float(embed_errors[local]),
            "refine": float(refine_errors[local]),
        })
        candidates = {
            "free": free["fields"][local],
            "embed": embedded[local],
            "refine": refined[local],
        }
        chosen = _compact(candidates[arm])
        selected.append(chosen)
        errors.append(error)
        arms.append(arm)
        pred_digest = (
            predecessor_ids[local]["solution_digest"]
            if arm in {"embed", "refine"} else None
        )
        if arm == "free":
            solution_digest_value = free["recipe_digests"][local]
        else:
            solution_digest_value = recipe_solution_digest({
                "kind": f"{arm}_arm",
                "target_rung": rung,
                "predecessor_digest": pred_digest,
                "free_context_digest": free["recipe_digests"][local],
                "refinement": (
                    "add_one_frozen_prefix_shared_scale_lloyd3_ldlq"
                    if arm == "refine" else None
                ),
            })
        identities.append(chain_identity_from_digest(
            winning_arm=arm,
            solution_digest_value=solution_digest_value,
            predecessor_digest=pred_digest,
        ))
        violations += int(
            error > float(predecessor_errors[local])
            + max(abs(float(predecessor_errors[local])), 1e-30) * 1e-12
        )
        tax += int(
            error > float(free["errors"][local])
            + max(abs(float(free["errors"][local])), 1e-30) * 1e-12
        )
    del subset, embedded, refined
    if violations or tax:
        raise AssertionError(
            f"K{rung} chain construction failed monotone={violations} tax={tax}"
        )
    return {
        "errors": errors,
        "fields": selected,
        "identity": identities,
        "arms": arms,
        "embed_seconds": embed_seconds,
        "refine_seconds": refine_seconds,
    }


def _fit_slices(errors: Mapping[int, Sequence[float]]) -> tuple[list[int], dict[int, Any], dict]:
    names = {k: f"FP8_CB_K{k}" for k in RUNGS}
    kmap = {name: k for k, name in names.items()}
    anchors = [names[k] for k in ANCHORS]
    accepted: list[int] = []
    laws: dict[int, Any] = {}
    holdout_rel = {k: [] for k in HOLDOUTS}
    for expert in range(EXPERT_COUNT):
        values = {names[k]: float(errors[k][expert]) for k in MEASURED_RUNGS}
        law = _cb_ladder_law(kmap, anchors, values)
        if law is None:
            continue
        rels = {}
        for k in HOLDOUTS:
            measured = values[names[k]]
            rel = abs(law.predict(names[k]) - measured) / max(abs(measured), 1e-30)
            rels[k] = rel
            holdout_rel[k].append(rel)
        if all(rels[k] <= TOLERANCE for k in HOLDOUTS):
            accepted.append(expert)
            laws[expert] = law
    return accepted, laws, {
        "accepted": len(accepted),
        "rejected": EXPERT_COUNT - len(accepted),
        "acceptance_rate": len(accepted) / EXPERT_COUNT,
        "holdouts": {
            f"K{k}": {
                "median": statistics.median(values) if values else math.nan,
                "p95": percentile(values, 0.95),
                "max": max(values) if values else math.nan,
            } for k, values in holdout_rel.items()
        },
    }


def _measure_projection(
    *, layer: int, projection: str, data: Mapping[str, Any], method: str,
) -> tuple[dict[int, list[float]], dict[str, Any]]:
    strict = method == "MIN-CHAIN STRICT"
    errors: dict[int, list[float]] = {}
    free_bank: dict[int, dict] = {}
    chain_bank: dict[int, dict] = {}
    measured_output: dict[int, dict[str, list[float] | list[int]]] = {}
    timing = {"free_seconds": 0.0, "embed_seconds": 0.0, "refine_seconds": 0.0}
    arm_counts = {"free": 0, "embed": 0, "refine": 0}
    warm_identity_cache: dict[tuple[int, ...], tuple[Any, Any, str]] = {}
    identities_by_rung: dict[int, list[dict | None]] = {
        k: [None] * EXPERT_COUNT for k in RUNGS
    }
    all_experts = tuple(range(EXPERT_COUNT))
    predecessors = predecessor_errors = predecessor_ids = None

    for rung in MEASURED_RUNGS:
        print(f"[burn] L{layer:02d} {projection} free K{rung}", flush=True)
        free = _encode_free(
            layer=layer, projection=projection, rung=rung, data=data,
            expert_ids=all_experts, keep_fields=strict, measure_output=True,
            warm_identity_cache=warm_identity_cache,
        )
        timing["free_seconds"] += free["elapsed_seconds"]
        free_bank[rung] = free
        if not strict or rung == MEASURED_RUNGS[0]:
            errors[rung] = list(free["errors"])
            if strict:
                selected = [_compact(value) for value in free["fields"]]
                identities = [
                    chain_identity_from_digest(
                        winning_arm="free",
                        solution_digest_value=free["recipe_digests"][index],
                        predecessor_digest=None,
                    ) for index, value in enumerate(selected)
                ]
                chain = {
                    "errors": list(free["errors"]), "fields": selected,
                    "identity": identities, "arms": ["free"] * EXPERT_COUNT,
                }
                chain_bank[rung] = chain
                identities_by_rung[rung] = list(identities)
                predecessors, predecessor_errors, predecessor_ids = (
                    selected, chain["errors"], identities
                )
                arm_counts["free"] += EXPERT_COUNT
        else:
            chain = _select_chain_rung(
                rung=rung, data=data, expert_ids=all_experts, free=free,
                predecessors=predecessors,
                predecessor_errors=predecessor_errors,
                predecessor_ids=predecessor_ids,
            )
            timing["embed_seconds"] += chain["embed_seconds"]
            timing["refine_seconds"] += chain["refine_seconds"]
            for arm in chain["arms"]:
                arm_counts[arm] += 1
            errors[rung] = list(chain["errors"])
            chain_bank[rung] = chain
            identities_by_rung[rung] = list(chain["identity"])
            predecessors, predecessor_errors, predecessor_ids = (
                chain["fields"], chain["errors"], chain["identity"]
            )
        measured_output[rung] = (
            _selected_output_metrics(
                data=data, expert_ids=all_experts,
                solutions=chain_bank[rung]["fields"], rung=rung,
            )
            if strict and rung != MEASURED_RUNGS[0]
            else copy.deepcopy(free["output_metrics"])
        )
        if strict:
            free["packed_fields"] = [
                _pack_solution(value) for value in free["fields"]
            ]
            free["fields"] = None
            if rung == MEASURED_RUNGS[0]:
                chain_bank[rung]["packed_fields"] = [
                    _pack_solution(value) for value in chain_bank[rung]["fields"]
                ]
                chain_bank[rung]["fields"] = None
            elif rung in chain_bank:
                # Only K28 is needed to seed rejected sequential fallback.
                chain_bank[rung]["fields"] = None

    accepted, laws, fit = _fit_slices(errors)
    rejected = sorted(set(all_experts).difference(accepted))
    for expert in accepted:
        for rung in MISSING_RUNGS:
            errors.setdefault(rung, [math.nan] * EXPERT_COUNT)
            errors[rung][expert] = float(laws[expert].predict(f"FP8_CB_K{rung}"))
    predecessors = predecessor_errors = predecessor_ids = None
    torch.cuda.empty_cache()

    if rejected and not strict:
        for rung in MISSING_RUNGS:
            print(
                f"[burn] L{layer:02d} {projection} incumbent fallback K{rung} "
                f"n={len(rejected)}", flush=True,
            )
            free = _encode_free(
                layer=layer, projection=projection, rung=rung, data=data,
                expert_ids=rejected, keep_fields=False, measure_output=False,
                warm_identity_cache=warm_identity_cache,
            )
            timing["free_seconds"] += free["elapsed_seconds"]
            errors.setdefault(rung, [math.nan] * EXPERT_COUNT)
            for local, expert in enumerate(rejected):
                errors[rung][expert] = float(free["errors"][local])
    elif rejected:
        # Rebuild only rejected curves consecutively. At a previously measured
        # rung its banked free fields are reused; the free scale sweep is never
        # repeated. This lets the immediate predecessor arm participate.
        first = MEASURED_RUNGS[0]
        pred_fields = [
            _unpack_solution(chain_bank[first]["packed_fields"][expert])
            for expert in rejected
        ]
        pred_errors = [errors[first][expert] for expert in rejected]
        pred_ids = [chain_bank[first]["identity"][expert] for expert in rejected]
        for rung in RUNGS[1:]:
            if rung in MEASURED_RUNGS:
                free = {
                    "errors": [free_bank[rung]["errors"][expert] for expert in rejected],
                    "fields": [
                        _unpack_solution(free_bank[rung]["packed_fields"][expert])
                        for expert in rejected
                    ],
                    "recipe_digests": [
                        free_bank[rung]["recipe_digests"][expert]
                        for expert in rejected
                    ],
                }
            else:
                print(
                    f"[burn] L{layer:02d} {projection} min-chain fallback K{rung} "
                    f"n={len(rejected)}", flush=True,
                )
                free = _encode_free(
                    layer=layer, projection=projection, rung=rung, data=data,
                    expert_ids=rejected, keep_fields=True, measure_output=False,
                    warm_identity_cache=warm_identity_cache,
                )
                timing["free_seconds"] += free["elapsed_seconds"]
            chain = _select_chain_rung(
                rung=rung, data=data, expert_ids=rejected, free=free,
                predecessors=pred_fields, predecessor_errors=pred_errors,
                predecessor_ids=pred_ids,
            )
            timing["embed_seconds"] += chain["embed_seconds"]
            timing["refine_seconds"] += chain["refine_seconds"]
            for arm in chain["arms"]:
                arm_counts[arm] += 1
            errors.setdefault(rung, [math.nan] * EXPERT_COUNT)
            for local, expert in enumerate(rejected):
                errors[rung][expert] = float(chain["errors"][local])
                identities_by_rung[rung][expert] = chain["identity"][local]
            if rung in MEASURED_RUNGS:
                selected_output = _selected_output_metrics(
                    data=data, expert_ids=rejected,
                    solutions=chain["fields"], rung=rung,
                )
                for key in (
                    "output_mse", "relative_mse", "n_activation_rows"
                ):
                    for local, expert in enumerate(rejected):
                        measured_output[rung][key][expert] = (
                            selected_output[key][local]
                        )
            pred_fields, pred_errors, pred_ids = (
                chain["fields"], chain["errors"], chain["identity"]
            )

    for rung in RUNGS:
        if rung not in errors or len(errors[rung]) != EXPERT_COUNT:
            raise AssertionError(f"L{layer} {projection} K{rung}: incomplete curve")
        if any(not math.isfinite(value) or value < 0 for value in errors[rung]):
            raise AssertionError(f"L{layer} {projection} K{rung}: invalid errors")
    if strict:
        violations = sum(
            errors[k][expert] > errors[k - 1][expert]
            + max(abs(errors[k - 1][expert]), 1e-30) * 1e-12
            for expert in range(EXPERT_COUNT) for k in RUNGS[1:]
            if expert in rejected
        )
        if violations:
            raise AssertionError(
                f"L{layer} {projection}: rejected chain has {violations} violations"
            )
    meta = {
        "fit": fit,
        "accepted_expert_ids": accepted,
        "rejected_expert_ids": rejected,
        "timing": timing,
        "arm_counts": arm_counts if strict else None,
        "warm_state_paths": {
            f"K{k}": free_bank[k]["warm_state_path"] for k in MEASURED_RUNGS
        },
        "predicted_rungs": list(MISSING_RUNGS),
        "fallback_measured_slices": len(rejected) * len(MISSING_RUNGS),
        "measured_output_metrics": {
            f"K{k}": measured_output[k] for k in MEASURED_RUNGS
        },
        "chain_serialization_identity": (
            {f"K{k}": identities_by_rung[k] for k in RUNGS}
            if strict else None
        ),
    }
    return errors, meta


def _measure_ordinary(
    *, layer: int, qname: str, data: Mapping[str, Any], method: str,
) -> tuple[dict[int, float], dict[str, Any]]:
    """Dual-holdout ladder for one non-routed body Linear."""
    strict = method == "MIN-CHAIN STRICT"
    errors: dict[int, float] = {}
    free_bank: dict[int, dict] = {}
    chain_bank: dict[int, dict] = {}
    measured_output: dict[int, dict[str, list[float] | list[int]]] = {}
    identities: dict[int, dict | None] = {k: None for k in RUNGS}
    timing = {"free_seconds": 0.0, "embed_seconds": 0.0, "refine_seconds": 0.0}
    warm_identity_cache: dict[tuple[int, ...], tuple[Any, Any, str]] = {}
    predecessors = predecessor_errors = predecessor_ids = None
    label = "ordinary-" + hashlib.sha256(qname.encode()).hexdigest()[:16]
    for rung in MEASURED_RUNGS:
        print(f"[burn] L{layer:02d} {qname.rsplit('.', 1)[-1]} free K{rung}", flush=True)
        free = _encode_free(
            layer=layer, projection=label, rung=rung, data=data,
            expert_ids=(0,), keep_fields=strict, measure_output=True,
            warm_identity_cache=warm_identity_cache,
        )
        timing["free_seconds"] += free["elapsed_seconds"]
        free_bank[rung] = free
        if not strict or rung == MEASURED_RUNGS[0]:
            errors[rung] = float(free["errors"][0])
            if strict:
                chosen = _compact(free["fields"][0])
                identity = chain_identity_from_digest(
                    winning_arm="free",
                    solution_digest_value=free["recipe_digests"][0],
                    predecessor_digest=None,
                )
                chain_bank[rung] = {
                    "fields": [chosen], "errors": [errors[rung]],
                    "identity": [identity], "arms": ["free"],
                }
                identities[rung] = identity
                predecessors, predecessor_errors, predecessor_ids = (
                    [chosen], [errors[rung]], [identity]
                )
        else:
            chain = _select_chain_rung(
                rung=rung, data=data, expert_ids=(0,), free=free,
                predecessors=predecessors,
                predecessor_errors=predecessor_errors,
                predecessor_ids=predecessor_ids,
            )
            timing["embed_seconds"] += chain["embed_seconds"]
            timing["refine_seconds"] += chain["refine_seconds"]
            errors[rung] = float(chain["errors"][0])
            identities[rung] = chain["identity"][0]
            chain_bank[rung] = chain
            predecessors, predecessor_errors, predecessor_ids = (
                chain["fields"], chain["errors"], chain["identity"]
            )
        measured_output[rung] = (
            _selected_output_metrics(
                data=data, expert_ids=(0,), solutions=chain_bank[rung]["fields"],
                rung=rung,
            )
            if strict and rung != MEASURED_RUNGS[0]
            else copy.deepcopy(free["output_metrics"])
        )

    names = {k: f"FP8_CB_K{k}" for k in RUNGS}
    kmap = {name: k for k, name in names.items()}
    values = {names[k]: errors[k] for k in MEASURED_RUNGS}
    law = _cb_ladder_law(kmap, [names[k] for k in ANCHORS], values)
    holdout_rel = {}
    if law is not None:
        for k in HOLDOUTS:
            holdout_rel[k] = abs(law.predict(names[k]) - errors[k]) / max(
                abs(errors[k]), 1e-30
            )
    accepted = bool(
        law is not None
        and all(holdout_rel[k] <= TOLERANCE for k in HOLDOUTS)
    )
    if accepted:
        for rung in MISSING_RUNGS:
            errors[rung] = float(law.predict(names[rung]))
    elif not strict:
        for rung in MISSING_RUNGS:
            free = _encode_free(
                layer=layer, projection=label, rung=rung, data=data,
                expert_ids=(0,), keep_fields=False, measure_output=False,
                warm_identity_cache=warm_identity_cache,
            )
            timing["free_seconds"] += free["elapsed_seconds"]
            errors[rung] = float(free["errors"][0])
    else:
        first = MEASURED_RUNGS[0]
        pred_fields = chain_bank[first]["fields"]
        pred_errors = chain_bank[first]["errors"]
        pred_ids = chain_bank[first]["identity"]
        for rung in RUNGS[1:]:
            if rung in MEASURED_RUNGS:
                free = {
                    "errors": free_bank[rung]["errors"],
                    "fields": free_bank[rung]["fields"],
                    "recipe_digests": free_bank[rung]["recipe_digests"],
                }
            else:
                free = _encode_free(
                    layer=layer, projection=label, rung=rung, data=data,
                    expert_ids=(0,), keep_fields=True, measure_output=False,
                    warm_identity_cache=warm_identity_cache,
                )
                timing["free_seconds"] += free["elapsed_seconds"]
            chain = _select_chain_rung(
                rung=rung, data=data, expert_ids=(0,), free=free,
                predecessors=pred_fields, predecessor_errors=pred_errors,
                predecessor_ids=pred_ids,
            )
            timing["embed_seconds"] += chain["embed_seconds"]
            timing["refine_seconds"] += chain["refine_seconds"]
            errors[rung] = float(chain["errors"][0])
            identities[rung] = chain["identity"][0]
            if rung in MEASURED_RUNGS:
                measured_output[rung] = _selected_output_metrics(
                    data=data, expert_ids=(0,), solutions=chain["fields"],
                    rung=rung,
                )
            pred_fields, pred_errors, pred_ids = (
                chain["fields"], chain["errors"], chain["identity"]
            )
    if set(errors) != set(RUNGS) or any(
        not math.isfinite(value) or value < 0 for value in errors.values()
    ):
        raise AssertionError(f"{qname}: incomplete/invalid ordinary ladder")
    return errors, {
        "accepted": accepted,
        "holdout_relative_error": {f"K{k}": holdout_rel.get(k, math.inf) for k in HOLDOUTS},
        "timing": timing,
        "chain_serialization_identity": identities if strict else None,
        "measured_output_metrics": {
            f"K{k}": measured_output[k] for k in MEASURED_RUNGS
        },
        "method": method,
    }


def _write_ordinary_costs(
    row: dict, errors: Mapping[int, float], meta: Mapping[str, Any]
) -> None:
    for rung in RUNGS:
        measured = rung in MEASURED_RUNGS or not meta["accepted"]
        entry = {
            "weight_mse": float(errors[rung]),
            "cost_source": (
                "production_render_weight_mse" if measured else "band_interpolated"
            ),
            "minchain_method": meta["method"],
            "ladder": {
                "anchors": list(ANCHORS), "holdouts": list(HOLDOUTS),
                "tolerance": TOLERANCE, "accepted": meta["accepted"],
            },
        }
        output = meta["measured_output_metrics"].get(f"K{rung}")
        if output is not None:
            entry.update({
                "output_mse": float(output["output_mse"][0]),
                "relative_mse": float(output["relative_mse"][0]),
                "n_activation_rows": int(output["n_activation_rows"][0]),
                "output_mse_measured": True,
            })
        else:
            entry["output_mse_measured"] = False
        if meta["method"] == "MIN-CHAIN STRICT":
            entry["cb_minchain_identity"] = (
                meta["chain_serialization_identity"][rung]
                if measured else {
                    "chain_version": MINCHAIN_CONTEXT_VERSION,
                    "status": "law_predicted__encode_identity_deferred",
                    "winning_arm": None, "predecessor_digest": None,
                }
            )
            entry["cb_serialization_context_extension"] = {
                "chain_version": MINCHAIN_CONTEXT_VERSION,
                "slice_group_identity": entry["cb_minchain_identity"],
            }
        row[f"FP8_CB_K{rung}"] = entry


def _layer_key(
    *, layer: int, method: str, layer_sha: str, pilot_sha: str,
    col_weights_sha: str,
) -> tuple[str, dict]:
    identity = {
        "schema": SHARD_SCHEMA,
        "layer": layer,
        "method": method,
        "source_index_sha256": sha256_file(SOURCE / "model.safetensors.index.json"),
        "base_layer_sha256": layer_sha,
        "pilot_report_sha256": pilot_sha,
        "col_weights_sha256": col_weights_sha,
        "formats": [f"FP8_CB_K{k}" for k in RUNGS],
        "anchors": list(ANCHORS),
        "holdouts": list(HOLDOUTS),
        "tolerance": TOLERANCE,
        "cb_serialized_payload": cb_serialization_context_stamp(
            CONTEXT, formats=[f"FP8_CB_K{k}" for k in RUNGS]
        ),
        "chain_version": (
            MINCHAIN_CONTEXT_VERSION if method == "MIN-CHAIN STRICT" else None
        ),
        "implementation_sha256": {
            "burn_tool": sha256_file(Path(__file__).resolve()),
            "minchain_module": sha256_file(
                Path(__file__).resolve().parents[1] / "prismaquant/cb_minchain.py"
            ),
        },
    }
    return _content_key(identity), identity


def _write_projection_costs(
    costs: dict, projection: str, errors: Mapping[int, Sequence[float]],
    meta: Mapping[str, Any],
) -> None:
    accepted = set(meta["accepted_expert_ids"])
    for expert in range(EXPERT_COUNT):
        qname = f"model.layers.{meta['layer']}.mlp.experts.{expert}.{projection}"
        row = costs[qname]
        for rung in RUNGS:
            measured = rung in MEASURED_RUNGS or expert not in accepted
            entry = {
                "weight_mse": float(errors[rung][expert]),
                "cost_source": (
                    "production_render_weight_mse"
                    if measured else "band_interpolated"
                ),
                "minchain_method": meta["method"],
                "ladder": {
                    "anchors": list(ANCHORS),
                    "holdouts": list(HOLDOUTS),
                    "tolerance": TOLERANCE,
                    "accepted": expert in accepted,
                },
                **(
                    {
                        "cb_minchain_identity": (
                            meta["chain_serialization_identity"][f"K{rung}"][expert]
                            if measured else {
                                "chain_version": MINCHAIN_CONTEXT_VERSION,
                                "status": "law_predicted__encode_identity_deferred",
                                "winning_arm": None,
                                "predecessor_digest": None,
                            }
                        ),
                        "cb_serialization_context_extension": {
                            "chain_version": MINCHAIN_CONTEXT_VERSION,
                            "slice_group_identity": (
                                meta["chain_serialization_identity"][f"K{rung}"][expert]
                                if measured else {
                                    "status": "law_predicted__encode_identity_deferred"
                                }
                            ),
                        },
                    }
                    if meta["method"] == "MIN-CHAIN STRICT" else {}
                ),
            }
            output = meta["measured_output_metrics"].get(f"K{rung}")
            if output is not None:
                entry.update({
                    "output_mse": float(output["output_mse"][expert]),
                    "relative_mse": float(output["relative_mse"][expert]),
                    "n_activation_rows": int(
                        output["n_activation_rows"][expert]
                    ),
                    "output_mse_measured": True,
                })
            else:
                entry["output_mse_measured"] = False
            row[f"FP8_CB_K{rung}"] = entry


def _run_layer(
    layer: int, *, method: str, pilot_sha: str,
    all_col_weights: Mapping[str, Any], col_weights_sha: str,
    model_to_shard: Mapping[str, str], model_to_ckpt: Mapping[str, str],
    scale_map: Mapping[str, Any], old_full: Mapping[str, Any],
) -> dict:
    base_layer, layer_record = load_layer_identity(layer)
    content_key, identity = _layer_key(
        layer=layer, method=method, layer_sha=layer_record["sha256"],
        pilot_sha=pilot_sha, col_weights_sha=col_weights_sha,
    )
    shard = SHARD_ROOT / f"layer_{layer:03d}.pkl"
    if shard.is_file():
        payload = _load(shard)
        if payload.get("content_key") != content_key:
            raise AssertionError(f"stale burn shard {shard}")
        if len(payload.get("costs", {})) != 775:
            raise AssertionError(f"burn shard row count mismatch {shard}")
        print(f"[burn] resume layer {layer:02d}", flush=True)
        return payload

    started = time.time()
    costs = copy.deepcopy(base_layer["costs"])
    # BF16 is immutable truth from the original complete production run.
    for qname, row in costs.items():
        old = old_full["costs"][qname]
        row["BF16"] = copy.deepcopy(old["BF16"])

    projection_meta = {}
    device = torch.device("cuda:0")
    ordinary_meta = {}
    ordinary_qnames = sorted(
        qname for qname in costs if ".mlp.experts." not in qname
    )
    if len(ordinary_qnames) != 7:
        raise AssertionError(
            f"layer {layer}: expected 7 ordinary body Linears, got "
            f"{len(ordinary_qnames)}"
        )
    for qname in ordinary_qnames:
        data = _load_ordinary(
            qname, device=device, identity=layer_record["identity"],
            all_col_weights=all_col_weights, model_to_shard=model_to_shard,
            model_to_ckpt=model_to_ckpt, scale_map=scale_map,
        )
        ordinary_errors, meta = _measure_ordinary(
            layer=layer, qname=qname, data=data, method=method
        )
        _write_ordinary_costs(costs[qname], ordinary_errors, meta)
        ordinary_meta[qname] = meta
        del data, ordinary_errors
        torch.cuda.empty_cache()
    for projection in PROJECTIONS:
        data = load_projection(
            layer, projection, device=device, identity=layer_record["identity"],
            all_col_weights=all_col_weights, model_to_shard=model_to_shard,
            model_to_ckpt=model_to_ckpt, scale_map=scale_map,
        )
        errors, meta = _measure_projection(
            layer=layer, projection=projection, data=data, method=method
        )
        meta["layer"] = layer
        meta["method"] = method
        _write_projection_costs(costs, projection, errors, meta)
        projection_meta[projection] = meta
        del data, errors
        torch.cuda.empty_cache()

    if len(costs) != 775:
        raise AssertionError(f"layer {layer}: expected 775 cost rows")
    payload = {
        "schema": SHARD_SCHEMA,
        "created_at": utc_now(),
        "content_key": content_key,
        "identity": identity,
        "costs": costs,
        "formats": sorted({fmt for row in costs.values() for fmt in row}),
        "meta": {
            "layer": layer,
            "method": method,
            "elapsed_seconds": time.time() - started,
            "projection": projection_meta,
            "ordinary": ordinary_meta,
            "row_count": len(costs),
        },
    }
    atomic_pickle(shard, payload)
    print(
        f"[burn] wrote layer {layer:02d} elapsed={payload['meta']['elapsed_seconds']/60:.1f}m",
        flush=True,
    )
    return payload


def _projection_abort(manifest: Mapping[str, Any], shards: Sequence[Mapping[str, Any]]) -> str:
    elapsed = time.time() - float(manifest["burn_started_epoch"])
    completed = len(shards)
    projected = elapsed / completed * LAYER_COUNT
    timebox = float(manifest["timebox_seconds"])
    lines = [
        "# DSV4 Burn Projection Abort",
        "",
        f"- Method: {manifest['method']}",
        f"- Completed evidence-grade shards: {completed}/43",
        f"- Observed foreground wall: {elapsed/3600:.3f} h",
        f"- Linear projection: {projected/3600:.3f} h",
        f"- Registered timebox: {timebox/3600:.1f} h",
        f"- Excess: {(projected-timebox)/3600:.3f} h ({projected/timebox:.3f}x)",
        "",
        "The pre-registered three-shard projection exceeded the method's timebox. The campaign stopped before starting shard 4; completed content-keyed shards remain resumable evidence.",
        "",
    ]
    return "\n".join(lines)


def run_burn() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("burn requires CUDA")
    method, pilot = _method_from_pilot()
    os.environ["PRISMAQUANT_CB_LDLQ"] = "1"
    os.environ["PRISMAQUANT_CB_LDLQ_BATCH_EXPERTS"] = "1"
    os.environ["PRISMAQUANT_CB_LDLQ_FEEDER_THREADS"] = "16"
    os.environ["PRISMAQUANT_CB_LDLQ_EXPERT_BATCH"] = "16"
    os.environ["PRISMAQUANT_CB_ENCODE_TIER"] = "balanced"
    if method == "MIN-CHAIN STRICT":
        os.environ["PRISMAQUANT_CB_MINCHAIN_PILOT"] = "1"
    pilot_sha = sha256_file(PILOT_JSON)
    with COL_WEIGHTS.open("rb") as handle:
        all_col_weights = pickle.load(handle)
    _, layer_zero = load_layer_identity(0)
    col_weights_sha = canonical_cb_col_weights_sha256(
        all_col_weights, layer_zero["identity"]["col_weights_qnames"]
    )
    model_to_shard, model_to_ckpt = _build_weight_map(str(SOURCE))
    scale_map = _build_fp8_scale_inv_map(str(SOURCE))
    old_full = _load(OLD_FULL_COST)
    manifest_path = BURN_ROOT / "BURN_MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest["method"] != method or manifest["pilot_report_sha256"] != pilot_sha:
            raise AssertionError("burn manifest differs from pilot decision")
    else:
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "created_at": utc_now(),
            "burn_started_epoch": time.time(),
            "method": method,
            "pilot_report": str(PILOT_JSON),
            "pilot_report_sha256": pilot_sha,
            "timebox_seconds": (16 if method == "MIN-CHAIN STRICT" else 14) * 3600,
            "thread_count": 16,
            "layer_count": LAYER_COUNT,
            "full_menu": [
                *[f"NVFP4_CB_K{k}" for k in range(12, 19)],
                *[f"FP8_CB_K{k}" for k in RUNGS], "BF16",
            ],
            "mtp_policy": "untouched fixed BF16 auxiliary",
        }
        atomic_json(manifest_path, manifest)
    completed = []
    for layer in range(LAYER_COUNT):
        payload = _run_layer(
            layer, method=method, pilot_sha=pilot_sha,
            all_col_weights=all_col_weights, col_weights_sha=col_weights_sha,
            model_to_shard=model_to_shard, model_to_ckpt=model_to_ckpt,
            scale_map=scale_map, old_full=old_full,
        )
        completed.append(payload)
        if len(completed) == 3:
            elapsed = time.time() - float(manifest["burn_started_epoch"])
            projected = elapsed / 3 * LAYER_COUNT
            if projected > float(manifest["timebox_seconds"]):
                atomic_text(
                    RUN_ROOT / "PROJECTION_ABORT.md",
                    _projection_abort(manifest, completed),
                )
                return 3
    return merge_burn()


def merge_burn() -> int:
    method, pilot = _method_from_pilot()
    shards = []
    sources = []
    all_costs: dict[str, dict] = {}
    for layer in range(LAYER_COUNT):
        path = SHARD_ROOT / f"layer_{layer:03d}.pkl"
        if not path.is_file():
            raise SystemExit(f"merge requires 43/43 shards; missing {path}")
        payload = _load(path)
        if payload.get("schema") != SHARD_SCHEMA or payload["meta"]["layer"] != layer:
            raise AssertionError(f"invalid layer shard {path}")
        if payload["meta"]["method"] != method or len(payload["costs"]) != 775:
            raise AssertionError(f"layer shard contract mismatch {path}")
        overlap = set(all_costs).intersection(payload["costs"])
        if overlap:
            raise AssertionError(f"duplicate merged rows {sorted(overlap)[:3]}")
        all_costs.update(copy.deepcopy(payload["costs"]))
        shards.append(payload)
        sources.append({
            "layer": layer, "path": str(path), "sha256": sha256_file(path),
            "content_key": payload["content_key"], "row_count": 775,
        })
    expected = LAYER_COUNT * 775
    if len(all_costs) != expected:
        raise AssertionError(f"merge row count {len(all_costs)} != {expected}")
    base = _load(BASE_COST)
    if set(all_costs) != set(base["costs"]):
        raise AssertionError("merged/base qname sets differ")
    manifest = {
        "schema": RESEARCH_COST_MANIFEST_SCHEMA,
        "cost_provenance": RESEARCH_COST_PROVENANCE,
        "acceptance": "explicit_user_decision_for_learning_experiment",
        "base": {
            "path": str(BASE_COST), "sha256": sha256_file(BASE_COST),
            "row_count": len(base["costs"]),
        },
        "segments_directory": str(SHARD_ROOT),
        "layers": sources,
        "layer_count": LAYER_COUNT,
        "rows_per_layer": 775,
        "assembled_row_count": len(all_costs),
        "segment_formats": sorted({fmt for row in all_costs.values() for fmt in row}),
        "formats": sorted({fmt for row in all_costs.values() for fmt in row}),
        "precedence": "campaign shards contain verified base columns plus new ladder",
        "method": method,
        "pilot_report_sha256": sha256_file(PILOT_JSON),
    }
    provenance = copy.deepcopy(base.get("provenance") or {})
    provenance["cost_provenance"] = RESEARCH_COST_PROVENANCE
    provenance["research_cost_manifest"] = manifest
    provenance["cb_serialized_payload"] = cb_serialization_context_stamp(
        CONTEXT,
        formats=[fmt for fmt in manifest["formats"] if "_CB_" in fmt],
    )
    provenance["campaign_mixed_cb_context"] = {
        "fp8_ladder": "LDLQ=true, balanced, warm-state, method per pilot",
        "retained_nvfp4_k12_k18": (
            "verified historical by-layer rows; their original render identity "
            "is retained inside each layer source and the research manifest"
        ),
    }
    merged = {
        "costs": all_costs,
        "formats": manifest["formats"],
        "provenance": provenance,
        "meta": {
            **copy.deepcopy(base.get("meta") or {}),
            "research_assembly": {
                "row_count": len(all_costs), "layer_count": LAYER_COUNT,
                "rows_per_layer": 775, "method": method,
                "mtp_policy": "fixed BF16 auxiliary; not body rows",
            },
        },
    }
    output = BURN_ROOT / "cost_merged.pkl"
    atomic_pickle(output, merged)
    summary = {
        "schema": REPORT_SCHEMA,
        "created_at": utc_now(),
        "method": method,
        "merge": "PASS",
        "layers": "43/43",
        "rows": len(all_costs),
        "cost_path": str(output),
        "cost_sha256": sha256_file(output),
        "pilot_gates": pilot["gates"],
        "projection_fit": {
            projection: {
                "accepted": sum(
                    shard["meta"]["projection"][projection]["fit"]["accepted"]
                    for shard in shards
                ),
                "total": LAYER_COUNT * EXPERT_COUNT,
            } for projection in PROJECTIONS
        },
        "elapsed_seconds_sum": sum(shard["meta"]["elapsed_seconds"] for shard in shards),
    }
    atomic_json(BURN_ROOT / "BURN_REPORT.json", summary)
    print(f"[burn] merge PASS 43/43 rows={len(all_costs)} -> {output}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "merge"))
    args = parser.parse_args()
    return run_burn() if args.command == "run" else merge_burn()


if __name__ == "__main__":
    raise SystemExit(main())
