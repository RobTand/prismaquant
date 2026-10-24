"""Explicit-input O5 adapter, not a W2 producer or a qualification authority.

See docs/evidence/qualified_w2_driver_2905.md for the input/handoff contract.
Run as ``python -m experiments.qualified_w2_replay_2905 --inputs INPUT.json``.
No artifact discovery, synthetic costs, research fallback, or serving launch.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path

from prismaquant.digests import DIRECT_ASCII_STRICT, bytes_sha256hex, is_sha256hex


class HandoffError(ValueError):
    """An explicit input cannot support this replay."""


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise HandoffError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read(path):
    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=_object,
                       parse_constant=lambda x: (_ for _ in ()).throw(
                           HandoffError(f"non-finite JSON constant: {x}")))
    # Also reject finite-looking JSON numbers that overflow Python's float.
    try:
        DIRECT_ASCII_STRICT.encoded(value)
    except ValueError as exc:
        raise HandoffError(f"nonfinite JSON number: {path}") from exc
    return value, bytes_sha256hex(raw)


def _qualification(roster, hashes, stats, costs, names, config):
    """Freeze a complete producer decision grid; never infer absent refusals."""
    if roster.get("schema") != "prismaquant.w2_qualification_roster.v1":
        raise HandoffError("qualification roster schema required")
    if roster.get("replay_config_sha256") != DIRECT_ASCII_STRICT.sha256(
            {k: v for k, v in config.items() if k != "qualification"}):
        raise HandoffError("qualification replay configuration differs")
    expected = {k: v for k, v in hashes.items() if k not in ("inputs", "qualification")}
    if roster.get("artifact_sha256") != expected:
        raise HandoffError("qualification artifact bindings differ")
    identities = roster.get("identities", {})
    if set(identities) != {"source", "probe", "calibration", "teacher", "scope"} or not all(
            is_sha256hex(v) for v in identities.values()):
        raise HandoffError("source/probe/calibration/teacher/scope SHA-256 identities required")
    if any(identities[k] != hashes[k + "_identity"]
           for k in ("source", "probe", "calibration", "teacher")):
        raise HandoffError("lineage identity differs from hashed artifact")
    if identities["scope"] != DIRECT_ASCII_STRICT.sha256(sorted(stats)):
        raise HandoffError("mutable scope identity differs")
    cells, refused, qualified = {}, set(), set()
    for cell in roster["cells"]:
        key = (cell["unit"], cell["format"])
        if key in cells:
            raise HandoffError("duplicate qualification cell")
        cells[key] = cell
        row = costs.get(key[0], {}).get(key[1])
        if cell["status"] == "qualified":
            if row is None or cell.get("row_sha256") != DIRECT_ASCII_STRICT.sha256(row):
                raise HandoffError("qualified row missing or changed")
            qualified.add(key)
        elif cell["status"] == "refused":
            if row is not None or not isinstance(cell.get("reason"), str) or not cell["reason"].strip():
                raise HandoffError("refused cell must be absent with a reason")
            refused.add(key)
        else:
            raise HandoffError("unknown qualification status")
    if set(cells) != {(u, f) for u in stats for f in names}:
        raise HandoffError("qualification roster must cover the complete declared grid")
    if qualified != {(u, f) for u, rows in costs.items() for f in rows}:
        raise HandoffError("cost coverage differs from qualification roster")
    return frozenset(refused)


def _scientific_rows(stats, costs, parsed, currency, identities):
    """Bind actual price coordinates, not merely the identity documents' bytes."""
    try:
        _require_scientific_rows(stats, costs, parsed, currency, identities)
    except (KeyError, TypeError, AttributeError, RuntimeError) as exc:
        raise HandoffError(f"malformed scientific row or identity document: {exc}") from exc


def _require_scientific_rows(stats, costs, parsed, currency, identities):
    from prismaquant.allocator_solver import _shape_from_stats
    from prismaquant.joint_aura_rows import joint_identity_sha256, require_probe_alignment
    from prismaquant.rung_allowability import (
        complete_scientific_price, qualified_cost_scope, require_quality_scope_alignment)
    from prismaquant.streamed_model_identity import validate_streamed_model_identity

    probe_scope = identities["probe"]
    probe = probe_scope["probe_identity"]
    if probe_scope["probe_identity_sha256"] != joint_identity_sha256(probe):
        raise HandoffError("handoff probe identity digest differs")
    teacher = identities["teacher"]
    validate_streamed_model_identity(teacher, where="handoff teacher_identity")
    validate_streamed_model_identity(probe["source_model"], where="handoff probe teacher")
    weights = identities["source"]["source_weights"]
    if set(weights) != set(stats):
        raise HandoffError("source_identity source_weights must cover the mutable scope")
    first = None
    for unit, rows in costs.items():
        for name, row in rows.items():
            where = f"{unit}/{name}"
            if complete_scientific_price(row, format_name=name) is None:
                raise HandoffError(f"{where}: producer-qualified quality required")
            scope = qualified_cost_scope(row, family=parsed[name][0].name,
                                         unit=unit, format_name=name)
            if scope is None or scope["currency"] != currency:
                raise HandoffError(f"{where}: common-reference scope required")
            if tuple(scope["shape"]) != tuple(_shape_from_stats(stats[unit])):
                raise HandoffError(f"{where}: scientific source shape differs from stats")
            if not math.isfinite(float(row["predicted_dloss"])):
                raise HandoffError(f"{where}: finite canonical quality required")
            # Production owns mathematical alignment and D32 provenance policy.
            if first is None:
                first = scope
            else:
                require_quality_scope_alignment(
                    first, scope, where=where, same_unit=first["unit"] == unit)
            require_probe_alignment(probe_scope, scope, where=where,
                                    message="handoff probe coordinates differ")
            window = identities["calibration"]
            if (window.get("calibration_sha256") != scope["calibration"]
                    or window.get("calibration_sha256") != probe["calibration_sha256"]
                    or any(window.get(key) != value for key, value in scope["window"].items())):
                raise HandoffError(f"{where}: handoff calibration coordinates differ")
            if (scope["teacher"] != teacher["content_sha256"]
                    or scope["probe_identity"]["source_model"]["content_sha256"]
                    != teacher["content_sha256"]):
                raise HandoffError(f"{where}: handoff teacher coordinate differs")
            if weights[unit] != scope["source_weight"]:
                raise HandoffError(f"{where}: handoff source weight coordinate differs")


def _row_provenance(supplied, effective):
    """Two stages: roster-qualified inputs, then production-owned chord rebuilds.

    Never transfer the supplied row's qualification digest onto a different row.
    A rebuild must be a validated canonical chord of the *supplied* anchors.
    Scientific validation is performed separately on both complete tables.
    """
    from prismaquant.rung_allowability import CANONICAL_CHORD_SOURCE

    cells = []
    if set(effective) != set(supplied):
        raise HandoffError("production effective row coverage differs")
    for unit, rows in sorted(supplied.items()):
        if set(effective[unit]) != set(rows):
            raise HandoffError(f"{unit}: production effective grid differs")
        for name, row in sorted(rows.items()):
            result = effective[unit][name]
            before, after = (DIRECT_ASCII_STRICT.sha256(value) for value in (row, result))
            chord = result.get("cost_source") == CANONICAL_CHORD_SOURCE
            anchors = []
            if chord:
                for anchor in result["canonical_anchors"]:
                    fmt = anchor["format"]
                    if (fmt not in rows or DIRECT_ASCII_STRICT.sha256(anchor["row"])
                            != DIRECT_ASCII_STRICT.sha256(rows[fmt])):
                        raise HandoffError(f"{unit}/{name}: effective anchor differs from supplied roster")
                    anchors.append({"format": fmt,
                                    "supplied_row_sha256": DIRECT_ASCII_STRICT.sha256(rows[fmt])})
            elif before != after:
                raise HandoffError(f"{unit}/{name}: production replaced a non-chord row")
            cells.append({"unit": unit, "format": name,
                          "supplied_row_sha256": before, "effective_row_sha256": after,
                          "effective_origin": "production_canonical_chord" if chord else "supplied",
                          "replaced": before != after, "anchors": anchors})
    return {"schema": "prismaquant.o5_effective_rows.v1",
            "supplied_costs_sha256": DIRECT_ASCII_STRICT.sha256(supplied),
            "effective_costs_sha256": DIRECT_ASCII_STRICT.sha256(effective),
            "cells": cells}


def _measured_only(contexts, extra):
    # Campaign wire-backed units are projected routed experts (#183). Without
    # reopening source tensors here, conservatively restrict ALL routed units,
    # using the production, profile-checked topology reader rather than a list.
    from prismaquant.lane_eligibility import STRUCTURE_ROUTED_MOE
    if (not isinstance(extra, list) or any(not isinstance(n, str) for n in extra)
            or len(set(extra)) != len(extra) or set(extra) - set(contexts)):
        raise HandoffError("invalid additional measured-only units")
    return frozenset(extra) | frozenset(
        n for n, context in contexts.items() if context.structure == STRUCTURE_ROUTED_MOE)


def _measured_overhead(overhead):
    if overhead.get("schema") != "prismaquant.measured_overhead.v1":
        raise HandoffError("measured overhead artifact required (explicit zero allowed)")
    components = overhead["components"]
    if not isinstance(components, dict) or any(
            not isinstance(k, str) or not k or type(v) is not int or v < 0
            for k, v in components.items()):
        raise HandoffError("measured overhead components must be named nonnegative integer bytes")
    if not isinstance(overhead.get("measurement_reference"), str) or not overhead["measurement_reference"].strip():
        raise HandoffError("measured overhead measurement reference required")
    return sum(components.values())


def _layer_configs(report):
    from prismaquant import format_registry as fr
    configs = {}
    for target in report["targets"]:
        for arm, result in target["arms"].items():
            if result["status"] != "ok":
                continue
            configs[f"{target['target_bpp']}/{arm}"] = {
                n: fr.get_format(fmt).autoround_config()
                for n, fmt in sorted(result["assignment"].items())}
    return configs


def run(input_path):
    """Validate producer-qualified rows and delegate optimization to production."""
    cfg, config_digest = _read(input_path)
    if cfg.get("schema") != "prismaquant.o5_explicit_w2_inputs.v2":
        raise HandoffError("explicit W2 input schema required")
    hashes = {"inputs": config_digest}

    def artifact(key):
        record = cfg[key]
        path = Path(record["path"])
        if not path.is_absolute():
            path = input_path.parent / path
        value, digest = _read(path)
        if digest != record["sha256"]:
            raise HandoffError(f"{key}: SHA-256 mismatch")
        hashes[key] = digest
        return value

    # Keep imports lazy: missing-input refusals need no serving/GPU dependencies.
    stats = artifact("stats")
    costs = artifact("costs")
    model_config = artifact("model_config")
    source = artifact("source_manifest")
    builds = artifact("kernel_builds")
    identities = {identity: artifact(identity + "_identity")
                  for identity in ("source", "probe", "calibration", "teacher")}
    overhead = artifact("measured_overhead")
    roster = artifact("qualification")
    from prismaquant import footprint as fp, format_registry as fr
    from prismaquant.allocator_candidates import build_candidates
    from prismaquant.model_profiles import profile_from_config
    from prismaquant.priced_cell import CostPricingPolicy
    from prismaquant.rung_allowability import load_rung_allowability
    from prismaquant.tessera_formats import parse_tessera_format_name
    from prismaquant.tessera_menu import tessera_runtime_contract
    from prismaquant.tessera_serving_scope import (ServingTarget,
                                                    context_by_unit_from_stats)

    currency = cfg["currency"]
    if not isinstance(currency, str) or not currency:
        raise HandoffError("explicit common-reference currency required")
    if not stats or set(stats) != set(costs):
        raise HandoffError("stats/cost units must be nonempty and identical")
    names = cfg["formats"]
    if not names or len(set(names)) != len(names):
        raise HandoffError("explicit unique dense-grid format names required")
    refused = _qualification(roster, hashes, stats, costs, names, cfg)
    parsed = {name: parse_tessera_format_name(name) for name in names}
    if any(value is None for value in parsed.values()):
        raise HandoffError("W2 driver accepts only Tessera rung formats")
    formats = [fr.get_format(name) for name in names]
    profile = profile_from_config(model_config)
    if profile.name != cfg["profile"]:
        raise HandoffError("detected model profile differs from explicit profile")
    for unit, rows in costs.items():
        if not rows or set(rows) - set(names):
            raise HandoffError(f"{unit}: empty costs or undeclared format")
    _scientific_rows(stats, costs, parsed, currency, identities)
    # Immutable snapshots are the two provenance boundaries. Production receives
    # a fresh table; neither aliasing nor later replay mutation can rewrite the
    # roster-qualified inputs or the effective-row hash evidence.
    supplied_cost_bytes = DIRECT_ASCII_STRICT.encoded(costs)

    tp = cfg["tensor_parallel"]
    if type(tp) is not int or tp < 1:
        raise HandoffError("positive integer tensor_parallel required")
    # Same explicit context for contract lookup, member admission and grouping.
    contexts = context_by_unit_from_stats(
        ServingTarget(**cfg["serving_context"]), stats, profile)
    contract = tessera_runtime_contract()
    if contract is None:
        raise HandoffError("pinned Tessera runtime contract unavailable")
    allowability = {}
    publication = Path(cfg["allowability_root"])
    if not publication.is_absolute():
        publication = input_path.parent / publication
    for family in sorted({value[0].name for value in parsed.values()}):
        if family not in contract.allowable_rungs:
            raise HandoffError(f"{family}: pinned runtime has no allowable_rungs rule")
        allowability[family] = load_rung_allowability(
            publication,
            format_entry={"family": family,
                          "allowable_rungs": contract.allowable_rungs[family]},
            expected_kernel_build=builds[family])
        if not allowability[family].scoped:
            raise HandoffError(f"{family}: canonical scoped publication required")
    built = build_candidates(
        stats, json.loads(supplied_cost_bytes), formats, target_profile=cfg["target_profile"],
        bit_precision=cfg["bit_precision"], preserve_runtime_frontier=True,
        defer_menu_reduction=set(stats), context_by_unit=contexts,
        rung_allowability=allowability, allowability_tensor_parallel=tp,
        pricing_policy=CostPricingPolicy(ucb_z=0.0, fisher_output_mse=False))
    candidates = built.candidates
    for unit, rows in costs.items():
        if {c.fmt for c in candidates.get(unit, [])} != set(rows):
            raise HandoffError(f"{unit}: production admission changed the supplied grid")
    if set(candidates) != set(stats):
        raise HandoffError("production candidate coverage differs from mutable stats")
    effective_cost_bytes = DIRECT_ASCII_STRICT.encoded(built.cost_rows)
    effective_costs = json.loads(effective_cost_bytes)
    _scientific_rows(stats, effective_costs, parsed, currency, identities)
    provenance = _row_provenance(json.loads(supplied_cost_bytes), effective_costs)
    manifest = fp.SourceByteManifest(
        source["bytes"], spans={name: frozenset(spans)
                                for name, spans in source["spans"].items()})
    if set(manifest) != set(manifest.spans):
        raise HandoffError("source byte and span coverage differs")
    if any(type(n) is not int or n < 0 for n in manifest.values()):
        raise HandoffError("source bytes must be nonnegative integers")
    if sum(manifest.values()) != source["total_bytes"]:
        raise HandoffError("source manifest total differs")
    reserve = cfg["overhead_reserve_bytes"]
    if type(reserve) is not int or reserve < 0:
        raise HandoffError("explicit nonnegative integer reserve required")
    measured = _measured_overhead(overhead)
    measured_only = _measured_only(contexts, cfg["measured_only_units"])
    # The portable seam owns three-arm filtering and re-aggregation, including
    # tessera_group_composites under the pinned fused_module licence. Do not
    # pre-fold here: that would leak full-arm options into measured-only arms.
    replay = importlib.import_module("prismaquant.interpolation_regret")
    report = replay.replay_three_arms(
        stats=stats, costs=json.loads(effective_cost_bytes), formats=formats, candidates=candidates,
        profile=profile, currency=currency, targets=("3.5", "4.5", "6.0"),
        bit_precision=cfg["bit_precision"],
        source_total_bytes=source["total_bytes"], source_manifest=manifest,
        measured_overhead_bytes=measured,
        overhead_reserve_bytes=reserve, refused_cells=refused,
        measured_only_units=measured_only)
    return {"schema": "prismaquant.o5_explicit_w2_result.v2",
            "qualification_claim": False,
            "input_sha256": hashes,
            "code_sha256": {
                module.__name__: bytes_sha256hex(Path(module.__file__).read_bytes())
                for module in (importlib.import_module(__name__), replay,
                               importlib.import_module("prismaquant.allocator_candidates"),
                               importlib.import_module("prismaquant.allocator_solver"), fp)},
            "runtime": contract.identity(),
            "allowability": {k: v.provenance() for k, v in allowability.items()},
            "qualification_roster": roster,
            "effective_row_provenance": provenance,
            "measured_only_units": sorted(measured_only),
            "measured_only_policy": "all production-profile-checked routed units plus explicit additions",
            "measured_overhead": overhead,
            "layer_config_status": "assignment-only; export preflight still required",
            "layer_configs": _layer_configs(report),
            "replay": report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Write the complete report and assignment configs as JSON")
    args = parser.parse_args(argv)
    try:
        result = run(args.inputs)
        encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(encoded + "\n")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError,
            AssertionError) as exc:
        parser.exit(2, f"W2 replay refused: {exc}\n")
    print(encoded)


if __name__ == "__main__":
    main()
