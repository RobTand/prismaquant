"""Select the GLM MTP layer's rungs under a declared byte sub-budget (PQ #1346).

The MTP layer is priced on its own objective, the MTP head's self-KL against
the BF16 MTP head (``glm_mtp.MTP_OBJECTIVE``). Its rows cannot join the body's
table: they are a different currency, and the body join refuses them. A draft
never changes the target's outputs, so its bytes buy throughput, not quality,
and summing its KL into the body knapsack would invent an exchange rate. The
two problems decompose instead: the body is allocated to its own budget, and
this module chooses the draft under a declared sub-budget with the canon
selector, ``mtp_rung_selection.select_rung``.

The payload (``prismaquant.glm_mtp_cost.v1``) carries:

- ``costs[unit][rung]``: complete joint-AURA rows on one MTP probe identity;
- ``wire_bytes[unit][rung]``: the priced rung's serialized bytes;
- ``params[unit]`` and ``source_dtype[unit]``;
- ``groups``: unit lists that each take one uniform rung (the routed stack,
  the shared expert).

A group whose every unit has a BF16 source is also offered BF16 passthrough,
at zero cost and two bytes per parameter (principle 11: never synthesized).
"""
from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .schemas import Contract
from .digests import bytes_sha256hex

SCHEMA = "prismaquant.glm_mtp_cost.v1"
RECORD_SCHEMA = "prismaquant.glm_mtp_selection.v1"
_BF16 = "BF16"
_STORAGE = Contract(ValueError, "MTP storage: ")


def load_mtp_cost(path) -> dict:
    """The MTP cost payload at ``path`` (pickle or JSON)."""
    raw = Path(path).read_bytes()
    payload = json.loads(raw) if str(path).endswith(".json") else pickle.loads(raw)
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
        raise ValueError(f"MTP cost payload must be {SCHEMA}")
    return dict(payload)


MERGE_SCHEMA = "prismaquant.glm_mtp_cost.merge.v1"
WIRE_BINDING_SCHEMA = "prismaquant.glm_mtp_priced_wires.v1"


def merge_mtp_costs(payloads, *, sources=()) -> dict:
    """One payload from several priced on the same MTP probe (PQ #1409).

    A rung is priced where it is measured, and one quantum prices one Tessera
    rate: GLM-5.3's routed E4M3 is attested at R896 and its routed BF16 at
    R1024, so the layer's menu comes from two runs. The allocator reads one
    ``--mtp-joint-cost``. The parts must describe the same layer on the same
    probe identity (same units, groups, parameter counts and source dtypes),
    and no rung may be priced twice; anything else refuses, by field. Rows are
    carried unchanged. ``sources`` (for example path and sha256 per part) is
    recorded beside each part's own provenance.
    """
    payloads = [dict(payload) for payload in payloads]
    sources = list(sources)
    if len(payloads) < 2:
        raise ValueError("merging MTP cost payloads needs at least two parts")
    if sources and len(sources) != len(payloads):
        raise ValueError(f"{len(sources)} sources for {len(payloads)} MTP cost parts")
    for index, payload in enumerate(payloads):
        if payload.get("schema") != SCHEMA:
            raise ValueError(f"MTP cost part {index} is not {SCHEMA}")
    first = payloads[0]
    for field in ("mtp_layer", "groups", "params", "source_dtype"):
        for index, payload in enumerate(payloads[1:], start=1):
            if payload[field] != first[field]:
                raise ValueError(f"MTP cost part {index} disagrees with part 0 on {field!r}")
    probes = [payload.get("provenance", {}).get("probe_identity_sha256") for payload in payloads]
    if None in probes or len(set(probes)) != 1:
        raise ValueError(f"MTP cost parts must name one probe identity, got {probes}")
    costs = {unit: {} for unit in first["costs"]}
    wire = {unit: {} for unit in first["costs"]}
    for index, payload in enumerate(payloads):
        if set(payload["costs"]) != set(costs):
            raise ValueError(f"MTP cost part {index} prices a different unit set")
        if set(payload["wire_bytes"]) != set(costs):
            raise ValueError(f"MTP cost part {index} serializes a different unit set")
        for unit, by_rung in payload["costs"].items():
            twice = sorted(set(by_rung) & set(costs[unit]))
            if twice:
                raise ValueError(f"MTP unit {unit}: rung(s) {twice} priced by more than one part")
            costs[unit].update(by_rung)
            by_wire = payload["wire_bytes"][unit]
            repeated_wires = sorted(set(by_wire) & set(wire[unit]))
            if repeated_wires:
                raise ValueError(f"MTP unit {unit}: wire rung(s) {repeated_wires} serialized by more than one part")
            wire[unit].update({rung: _STORAGE.integer(value, where=f"wire_bytes[{unit}][{rung}]", minimum=1)
                               for rung, value in by_wire.items()})
    return {
        "schema": SCHEMA,
        "mtp_layer": first["mtp_layer"],
        "costs": costs,
        "wire_bytes": wire,
        "params": dict(first["params"]),
        "source_dtype": dict(first["source_dtype"]),
        "groups": {name: list(members) for name, members in first["groups"].items()},
        "provenance": {
            "schema": MERGE_SCHEMA,
            "probe_identity_sha256": probes[0],
            "parts": [{**({"source": sources[index]} if sources else {}),
                       "rungs": sorted({rung for rows in payload["costs"].values() for rung in rows}),
                       "wire_rungs": sorted({rung for rows in payload["wire_bytes"].values() for rung in rows}),
                       "provenance": payload.get("provenance", {})}
                      for index, payload in enumerate(payloads)],
        },
    }


def _bound_payload(reference: Mapping, *, label: str) -> dict:
    """Read an already published cost artifact only under its exact byte anchor."""
    from .stage_inputs import read_bound as _read_bound

    if (not isinstance(reference, Mapping) or set(reference) != {"path", "sha256"}
            or not isinstance(reference["path"], str)
            or not isinstance(reference["sha256"], str)
            or len(reference["sha256"]) != 64):
        raise ValueError(f"{label} needs an exact path and SHA-256 source")
    payload = pickle.loads(_read_bound(dict(reference), label))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} is not a cost payload")
    return dict(payload)


def enrich_mtp_cost_wires(payload: Mapping) -> dict:
    """Join measured M3 wires through exact M4 anchors into an M6 cost copy.

    This is offline metadata work. The historical M3/M4/M6 files and blob
    directories stay unchanged. Every source is checked before its receipt is
    used; the export intake remains responsible for hashing each selected blob.
    """
    from . import tessera_expert_projection as tep
    from .tessera_formats import parse_tessera_format_name

    if payload.get("schema") != SCHEMA:
        raise ValueError(f"MTP cost payload must be {SCHEMA}")
    provenance = payload.get("provenance", {})
    if provenance.get("schema") != MERGE_SCHEMA or not isinstance(provenance.get("parts"), list):
        raise ValueError("MTP priced-wire join needs the bound merged cost provenance")
    parts = provenance["parts"]
    if len(parts) < 2:
        raise ValueError("MTP priced-wire join needs at least two bound parts")
    receipts = {name: {} for name in payload["costs"]}
    wire_roots = {name: {} for name in payload["costs"]}
    bindings = {name: {} for name in payload["costs"]}
    projection = None
    seen = set()
    seen_wires = set()
    for index, part in enumerate(parts):
        m4_ref = part.get("source")
        m4 = _bound_payload(m4_ref, label=f"MTP M4 part {index}")
        if (m4.get("schema") != SCHEMA or m4.get("mtp_layer") != payload["mtp_layer"]
                or m4.get("groups") != payload["groups"]
                or m4.get("params") != payload["params"]
                or m4.get("source_dtype") != payload["source_dtype"]
                or m4.get("provenance") != part.get("provenance")
                or set(m4.get("costs", {})) != set(payload["costs"])):
            raise ValueError(f"MTP M4 part {index} differs from merged cost provenance")
        rungs = sorted({fmt for by_fmt in m4["costs"].values() for fmt in by_fmt})
        if rungs != part.get("rungs"):
            raise ValueError(f"MTP M4 part {index} rung roster differs from merged cost")
        wire_rungs = sorted({fmt for by_fmt in m4.get("wire_bytes", {}).values() for fmt in by_fmt})
        if wire_rungs != part.get("wire_rungs"):
            raise ValueError(f"MTP M4 part {index} wire roster differs from merged cost")
        anchors = m4["provenance"].get("tessera_joint_anchors", {})
        m3 = _bound_payload(anchors.get("inputs", {}).get("merged_cost"),
                            label=f"MTP M3 price {index}")
        if m3.get("schema") != "prismaquant.tessera_campaign_cost.v1":
            raise ValueError(f"MTP M3 price {index} has no campaign cost schema")
        m3_projection = m3.get("provenance", {}).get(tep.PROJECTION_KEY)
        try:
            _source, units, _stacks = tep.carried_units(m3_projection)
        except tep.ExpertProjectionError as exc:
            raise ValueError(f"MTP M3 price {index} has no valid expert projection: {exc}") from exc
        if projection is None:
            projection = m3_projection
        elif m3_projection != projection:
            raise ValueError(f"MTP M3 price {index} changes the producer projection")
        root = m3.get("provenance", {}).get("wire_dir")
        if not isinstance(root, str) or not root:
            raise ValueError(f"MTP M3 price {index} has no wire directory")
        for name, by_fmt in m4["costs"].items():
            for fmt, row in by_fmt.items():
                cell = (name, fmt)
                if cell in seen:
                    raise ValueError(f"MTP {name}@{fmt} is priced by multiple parts")
                seen.add(cell)
                if payload["costs"].get(name, {}).get(fmt) != row:
                    raise ValueError(f"MTP {name}@{fmt} differs from bound M4 price")
        for name, by_fmt in m4["wire_bytes"].items():
            if name not in payload["costs"]:
                raise ValueError(f"MTP M4 part {index} serializes an unknown unit")
            for fmt, wire_bytes in by_fmt.items():
                cell = (name, fmt)
                if cell in seen_wires:
                    raise ValueError(f"MTP {name}@{fmt} wire is bound by multiple parts")
                seen_wires.add(cell)
                wire_bytes = _STORAGE.integer(wire_bytes, where=f"wire_bytes[{name}][{fmt}]", minimum=1)
                if payload["wire_bytes"].get(name, {}).get(fmt) != wire_bytes:
                    raise ValueError(f"MTP {name}@{fmt} differs from bound M4 wire")
                if m3.get("costs", {}).get(name, {}).get(fmt, {}).get("wire_bytes") != wire_bytes:
                    raise ValueError(f"MTP {name}@{fmt} wire bytes differ from the M3 price")
                if name not in units:
                    # A selected dense BF16 passthrough needs no expert wire.
                    continue
                parsed = parse_tessera_format_name(fmt)
                if parsed is None:
                    raise ValueError(f"MTP {name}@{fmt} has no projected Tessera wire")
                family, q256 = parsed
                record = m3.get(tep.EXPERT_WIRES_KEY, {}).get(name, {}).get(fmt)
                try:
                    checked = tep.check_expert_wire_receipt(
                        record, name=name, unit=units[name], q256=int(q256),
                        grid=family.payload_grid().name)
                    tep.locate_expert_wire(checked, name=name, wire_dir=Path(root))
                except tep.ExpertProjectionError as exc:
                    raise ValueError(f"MTP {name}@{fmt} lacks its priced wire: {exc}") from exc
                if wire_bytes != checked["blob_bytes"]:
                    raise ValueError(f"MTP {name}@{fmt} wire bytes differ from the M3 price")
                receipts[name][fmt] = checked
                wire_roots[name][fmt] = root
                bindings[name][fmt] = {"m4": dict(m4_ref),
                                       "m3": dict(anchors["inputs"]["merged_cost"])}
    if seen != {(name, fmt) for name, by_fmt in payload["costs"].items() for fmt in by_fmt}:
        raise ValueError("MTP bound parts do not cover the merged priced cells")
    if seen_wires != {(name, fmt) for name, by_fmt in payload["wire_bytes"].items() for fmt in by_fmt}:
        raise ValueError("MTP bound parts do not cover the merged wire cells")
    return {**payload, "mtp_expert_projection": projection,
            "mtp_expert_wires": receipts, "mtp_expert_wire_roots": wire_roots,
            "mtp_expert_source_bindings": bindings,
            "mtp_expert_wire_binding_schema": WIRE_BINDING_SCHEMA}


def _mtp_probe(payload) -> tuple[str, dict]:
    """The one MTP probe identity every row carries, validated."""
    from .glm_mtp import MTP_OBJECTIVE, MTP_OBJECTIVE_SCHEMA
    from .joint_aura import (prepare_joint_aura_identities, release_joint_aura_identities,
                             validate_joint_aura_entry)

    digests, last_row = set(), None
    try:
        # Pickle retains shared probe objects. The joint validator already
        # owns a content-bound immutable wrapper that validates each distinct
        # source model once while continuing to validate every row/operator.
        prepare_joint_aura_identities(payload)
        for unit, by_rung in payload["costs"].items():
            if not by_rung:
                raise ValueError(f"MTP unit {unit}: no priced source operator evidence")
            params = _STORAGE.integer(payload["params"][unit], where=f"params[{unit}]", minimum=1)
            for rung, row in by_rung.items():
                if not validate_joint_aura_entry(row):
                    raise ValueError(f"MTP row {unit} @ {rung} is not a joint-AURA entry")
                operator = row["joint_operator_identity"]
                if operator["qname"] != unit or operator["format"] != rung:
                    raise ValueError(f"MTP row {unit} @ {rung} names {operator['qname']} @ {operator['format']}")
                source_shape = operator["source_weight"]["shape"]
                source_params = math.prod(source_shape)
                if params != source_params:
                    raise ValueError(
                        f"MTP params[{unit}]={params} differ from source shape "
                        f"{source_shape} ({source_params}) for {rung}")
                objective = row["probe_identity"].get("objective")
                if (not isinstance(objective, Mapping) or objective.get("schema") != MTP_OBJECTIVE_SCHEMA
                        or objective.get("objective") != MTP_OBJECTIVE):
                    raise ValueError(f"MTP row {unit} @ {rung} was not priced on the MTP objective")
                if objective.get("mtp_layer") != payload["mtp_layer"]:
                    raise ValueError(f"MTP row {unit} @ {rung} names MTP layer {objective.get('mtp_layer')}")
                digests.add(row["probe_identity_sha256"])
                last_row = row
    finally:
        release_joint_aura_identities(payload)
    if len(digests) != 1:
        raise ValueError(f"MTP rows must share one probe identity, got {len(digests)}")
    return digests.pop(), last_row["probe_identity"]


def _unit_storage(payload, unit):
    """Admit exact storage counts before eligibility can hide a malformed row."""
    wire = {
        rung: _STORAGE.integer(value, where=f"wire_bytes[{unit}][{rung}]", minimum=1)
        for rung, value in payload["wire_bytes"].get(unit, {}).items()
    }
    params = _STORAGE.integer(payload["params"][unit], where=f"params[{unit}]", minimum=1)
    return wire, params


def add_mtp_scope_arguments(parser) -> None:
    """Declare the MTP workload independently of the body's workload."""
    parser.add_argument("--mtp-regime", type=int, default=None,
                        help="Actual MTP token-row regime M for canonical admission")
    parser.add_argument("--mtp-tensor-parallel", type=int, default=1,
                        help="Serving tensor parallel size for the MTP rank-local shape")
    parser.add_argument("--mtp-routing", default=None,
                        help="Actual routing coordinate for scoped MTP experts")


def _mtp_allowability_scope(unit, rung, measured, stats, context, owners, *, m, tensor_parallel):
    """Derive scope from validated source operators through the shared owner."""
    from .allocator_candidates import unit_allowability_scope
    row = measured.get(rung)
    if row is not None:
        shape = row["joint_operator_identity"]["source_weight"]["shape"]
    else:
        shapes = {tuple(anchor["joint_operator_identity"]["source_weight"]["shape"])
                  for anchor in measured.values()}
        shape = next(iter(shapes)) if len(shapes) == 1 else ()
    actual = dict(stats or {})
    # Caller metadata can supply topology, but not replace the measured shape.
    actual.pop("out_features", None)
    actual.pop("in_features", None)
    if len(shape) >= 2:
        actual.update(out_features=shape[-2], in_features=shape[-1])
    return unit_allowability_scope(rung, unit, actual, context, owners,
                                  m=m, tensor_parallel=tensor_parallel)


def _unit_rows(payload, eligible=None, *, rung_allowability=None, quality_prices=None,
               quality_provenance=None, scope_provenance=None, stats=None, context_by_unit=None,
               target_profile=None, allowability_m=None, allowability_tensor_parallel=1) -> tuple[dict, dict]:
    """Price the exact wire menu from measured anchors or bound proposals."""
    from .rung_allowability import owner_for_format
    from .allocator_candidates import candidate_rung_admission
    from .lane_spec import family_hook
    from .serving_profiles import load_serving_profile
    from . import format_registry as fr
    production = not load_serving_profile(target_profile).emulation_only
    groups = payload["groups"]
    units = [unit for members in groups.values() for unit in members]
    if set(units) != set(payload["costs"]) or len(units) != len(set(units)):
        raise ValueError("MTP groups must partition the priced units")
    rows, unattested = {}, {}
    for unit in units:
        wire, params = _unit_storage(payload, unit)
        measured = payload["costs"][unit]
        proposals = (quality_prices or {}).get(unit, {})
        if set(wire) != set(measured) and rung_allowability is None and not proposals:
            raise ValueError(f"MTP unit {unit}: wire bytes and costs name different rungs")
        if _BF16 in measured or _BF16 in wire:
            raise ValueError(f"MTP unit {unit}: BF16 is passthrough, not a priced row")
        rows[unit] = {}
        for rung in wire:
            if eligible is not None and not eligible(unit, rung):
                unattested.setdefault(rung, []).append(unit)
                continue
            owner = owner_for_format(rung_allowability, rung)
            if rung_allowability is not None or production:
                context = None if context_by_unit is None else context_by_unit.get(unit)
                unit_scope = _mtp_allowability_scope(unit, rung, measured,
                    (stats or {}).get(unit), context, rung_allowability,
                    m=allowability_m, tensor_parallel=allowability_tensor_parallel)
                if owner is not None and owner.scoped:
                    required = ("kernel_kind", "rows", "columns", "m")
                    if (unit_scope is None or any(unit_scope.get(axis) is None for axis in required)
                            or (unit_scope["kernel_kind"] in ("routed_moe", "routed")
                                and unit_scope.get("routing") is None)):
                        unattested.setdefault(rung, []).append(unit)
                        continue
                admission = candidate_rung_admission(rung, target_profile=target_profile,
                    serving_context=context, rung_allowability=rung_allowability,
                    allowability_scope=unit_scope)
                family = fr.format_family_of(fr.canonical_format_name(rung))
                if admission is not None and (
                        (rung_allowability is not None and owner is None)
                        or not admission.admits(family_hook(family, "menu_mode_in_force")(None))):
                    unattested.setdefault(rung, []).append(unit)
                    continue
                if scope_provenance is not None and unit_scope is not None:
                    scope_provenance[(unit, rung)] = unit_scope
            price = (owner.chord_cost(rung, unit=unit, costs=measured)
                     if owner is not None else proposals.get(rung, measured.get(rung)))
            if price is None:
                unattested.setdefault(rung, []).append(unit)
                continue
            if quality_provenance is not None and price.get("canonical_quality") is not None:
                quality_provenance.setdefault(unit, {})[rung] = price["canonical_quality"]
            rows[unit][rung] = (float(price["predicted_dloss"]), wire[rung])
        if payload["source_dtype"][unit] == "bfloat16":
            rows[unit][_BF16] = (0.0, 2 * params)
    return rows, {rung: sorted(units) for rung, units in sorted(unattested.items())}

def _recompute_recorded_quality(payload, recorded):
    """Recompute proposal prices from their actual bound anchors before export."""
    from .rung_allowability import (_producer_api, qualified_cost_scope,
                                   qualified_rung_quality, require_quality_result_matches)
    from .tessera_formats import parse_tessera_format_name
    prices = {}
    producer = None
    for unit, by_rung in recorded.items():
        for name, expected in by_rung.items():
            family, rung = parse_tessera_format_name(name)
            lower, upper = expected["anchors"]
            rows = payload["costs"][unit]
            left, right = rows[family.format_name(lower)], rows[family.format_name(upper)]
            if producer is None:
                producer = _producer_api()
            actual = qualified_rung_quality(producer, family.name, rung,
                lower_rung=lower, upper_rung=upper,
                lower_value=left["predicted_dloss"], upper_value=right["predicted_dloss"],
                lower_scope=qualified_cost_scope(left, family=family.name, unit=unit,
                                                format_name=family.format_name(lower)),
                upper_scope=qualified_cost_scope(right, family=family.name, unit=unit,
                                                format_name=family.format_name(upper)))
            require_quality_result_matches(actual, expected, where="MTP quality price")
            if actual["provenance"]["unit"] != unit:
                raise ValueError("MTP canonical quality differs from its actual bound unit")
            prices.setdefault(unit, {})[name] = {"predicted_dloss": actual["value"]}
    return prices


class MtpMenuRefused(ValueError):
    """A declared MTP menu (``formats``) that the priced, attested rows cannot honour."""


def _declared_menu(formats) -> list[str]:
    """The registry spellings of a declared MTP menu; an empty or unknown name refuses."""
    from . import format_registry as fr

    names = [str(name).strip() for name in formats]
    if not names or not all(names):
        raise MtpMenuRefused("MTP formats must name at least one format, and no empty name")
    declared = set()
    for name in names:
        try:
            fr.get_format(name)
        except (KeyError, ValueError) as exc:
            raise MtpMenuRefused(f"MTP formats name an unknown format: {name!r}") from exc
        declared.add(fr.canonical_format_name(name))
    return sorted(declared)


def _restrict_to_declared(rows: dict, declared) -> tuple[dict, dict, list]:
    """Intersect each unit's offered rungs with the declared menu.

    BF16 passthrough is a rung like any other here: a declaration that omits
    it removes it. Returns the kept rows, ``{rung: units removed}`` and the
    declared names offered to no unit. A unit left with no rung refuses: the
    declaration is never widened back to the attested menu.
    """
    from . import format_registry as fr

    wanted = set(declared)
    kept, removed, offered = {}, {}, set()
    for unit, menu in rows.items():
        kept[unit] = {}
        for rung, value in menu.items():
            canonical = fr.canonical_format_name(rung)
            if canonical in wanted:
                kept[unit][rung] = value
                offered.add(canonical)
            else:
                removed[rung] = removed.get(rung, 0) + 1
    empty = sorted(unit for unit, menu in kept.items() if not menu)
    if empty:
        raise MtpMenuRefused(
            f"MTP formats {sorted(wanted)} leave {len(empty)} unit(s) with an empty "
            f"menu (not priced, not attested, or not a BF16 source): {empty[:3]}")
    return kept, dict(sorted(removed.items())), sorted(wanted - offered)


@dataclass
class _MtpMenu:
    payload: dict
    byte_budget: int
    probe_sha256: str
    probe: dict
    groups: dict
    rows: dict
    menu: list
    incomplete: dict
    unattested: dict
    declared_record: dict
    fixed_formats: Mapping | None
    quality_provenance: dict
    scope_provenance: dict
    rung_allowability: Mapping | None


def _prepare_mtp_menu(payload: Mapping, *, byte_budget: int, eligible=None,
                      fixed_formats=None, formats=None, rung_allowability=None,
                      stats=None, context_by_unit=None, target_profile=None,
                      allowability_m=None, allowability_tensor_parallel=1) -> _MtpMenu:
    """Admit the priced group menu without a winner decision."""
    from . import mtp_rung_selection as canon

    payload = dict(payload)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"MTP cost payload must be {SCHEMA}")
    byte_budget = _STORAGE.integer(byte_budget, where="byte_budget", minimum=0)
    probe_sha256, probe = _mtp_probe(payload)
    quality_provenance, scope_provenance = {}, {}
    rows, unattested = _unit_rows(payload, eligible, rung_allowability=rung_allowability,
        quality_provenance=quality_provenance, scope_provenance=scope_provenance,
        stats=stats, context_by_unit=context_by_unit, target_profile=target_profile,
        allowability_m=allowability_m, allowability_tensor_parallel=allowability_tensor_parallel)
    groups = {name: tuple(members) for name, members in payload["groups"].items()}
    declared_record = {}
    if formats is not None:
        declared = _declared_menu(formats)
        rows, restricted, unoffered = _restrict_to_declared(rows, declared)
        declared_record = {"mtp_formats": declared, "menu_restricted_rungs": restricted,
                           "mtp_formats_unoffered": unoffered}
    if fixed_formats is not None:
        if not isinstance(fixed_formats, Mapping) or any(
                not isinstance(name, str) or not isinstance(fmt, str) or not fmt
                for name, fmt in fixed_formats.items()):
            raise ValueError("MTP fixed_formats must map group names to format names")
        unknown = set(fixed_formats) - set(groups)
        if unknown:
            raise ValueError(f"MTP fixed_formats name unknown groups: {sorted(unknown)}")
        for group, fmt in fixed_formats.items():
            absent = [unit for unit in groups[group] if fmt not in rows[unit]]
            if absent:
                raise ValueError(
                    f"MTP fixed {group}={fmt} is missing or ineligible for "
                    f"{len(absent)} member(s): {absent[:3]}")
            for unit in groups[group]:
                rows[unit] = {fmt: rows[unit][fmt]}
    if formats is not None:
        bare = {group: sorted(set().union(*(rows[unit] for unit in members)))
                for group, members in groups.items()
                if not set.intersection(*(set(rows[unit]) for unit in members))}
        if bare:
            raise MtpMenuRefused(
                f"MTP formats {declared_record['mtp_formats']} leave group(s) "
                f"{sorted(bare)} with no complete rung; each member keeps a rung, but no "
                f"rung is priced and attested for every member: {bare}")
    menu, incomplete = canon.group_product_menu(groups, rows, params=payload["params"])
    return _MtpMenu(payload, byte_budget, probe_sha256, probe, groups, rows, menu,
                    incomplete, unattested, declared_record, fixed_formats,
                    quality_provenance, scope_provenance, rung_allowability)


def _mtp_record(prepared: _MtpMenu, rung, *, constants_source: str, selection: dict) -> dict:
    """Retain one admitted choice and its exact selected wire receipts."""
    payload = prepared.payload
    groups = prepared.groups
    chosen = dict(part.split("=", 1) for part in rung.name.split("|"))
    assignment = {unit: chosen[group] for group, members in groups.items() for unit in members}
    selected_wires = {}
    enriched = payload
    parts = enriched.get("provenance", {}).get("parts", [])
    if any(isinstance(part.get("source"), Mapping) and
           "sha256" in part["source"] for part in parts):
        # The original cost remains a valid historical input. For bound M4
        # parts, however, a selected Tessera cell must retain its exact M3
        # receipt and root or export would have to encode unpriced bytes.
        enriched = enrich_mtp_cost_wires(enriched)
        prepared.payload = enriched
        priced = {key: {} for key in ("mtp_expert_wires", "mtp_expert_wire_roots",
                                      "mtp_expert_source_bindings")}
        for name, fmt in assignment.items():
            if fmt == _BF16:
                continue
            for key in priced:
                try:
                    priced[key][name] = enriched[key][name][fmt]
                except KeyError as exc:
                    raise ValueError(f"MTP {name}@{fmt} has no {key}") from exc
        selected_wires = {"mtp_expert_projection": enriched["mtp_expert_projection"],
                          **priced, "mtp_expert_wire_binding_schema": WIRE_BINDING_SCHEMA}
    return {
        "schema": RECORD_SCHEMA,
        "objective": prepared.probe["objective"]["objective"],
        "mtp_layer": int(payload["mtp_layer"]),
        "probe_identity_sha256": prepared.probe_sha256,
        "byte_budget": prepared.byte_budget,
        "rung": rung.name,
        "rung_by_group": chosen,
        "resident_bytes": int(rung.resident_bytes),
        "bits": float(rung.bits),
        "E": float(rung.E),
        "constants_source": constants_source,
        "incomplete_rungs": prepared.incomplete,
        "unattested_rungs": {rung: len(units) for rung, units in prepared.unattested.items()},
        **prepared.declared_record,
        "selection": selection,
        **({"fixed_formats": dict(sorted(prepared.fixed_formats.items()))}
           if prepared.fixed_formats is not None else {}),
        **({"rung_allowability_scopes": {unit: prepared.scope_provenance[(unit, fmt)]
               for unit, fmt in assignment.items() if (unit, fmt) in prepared.scope_provenance},
            "rung_allowability": {family: owner.provenance()
                for family, owner in prepared.rung_allowability.items()}}
           if prepared.rung_allowability is not None else {}),
        "assignment": assignment,
        **({"canonical_quality": prepared.quality_provenance} if prepared.quality_provenance else {}),
        **selected_wires,
    }


def select_mtp_rungs(payload: Mapping, *, byte_budget: int, constants: Mapping,
                     acceptance_points=(), k: int = 1, eligible=None,
                     fixed_formats: Mapping[str, str] | None = None,
                     formats=None, rung_allowability=None, stats=None, context_by_unit=None,
                     target_profile=None, allowability_m=None, allowability_tensor_parallel=1) -> dict:
    """Select the independent MTP winner under the declared sub-budget.

    Canonical admission and the native callback both restrict the menu.
    A declared menu or group pin cannot add an unpriced or ineligible rung.
    Without acceptance points, the selector returns the lowest-E feasible rung.
    """
    from . import mtp_rung_selection as canon

    prepared = _prepare_mtp_menu(payload, byte_budget=byte_budget, eligible=eligible,
        fixed_formats=fixed_formats, formats=formats, rung_allowability=rung_allowability,
        stats=stats, context_by_unit=context_by_unit, target_profile=target_profile,
        allowability_m=allowability_m, allowability_tensor_parallel=allowability_tensor_parallel)
    serve = canon.ServeConstants(t_ms=float(constants["t_ms"]), d0_ms=float(constants["d0_ms"]),
                                 c_ms_per_bit=float(constants["c_ms_per_bit"]))
    points = [canon.AcceptancePoint(**point) for point in acceptance_points]
    result = canon.select_rung(prepared.menu, serve, points,
        mem_budget_bytes=prepared.byte_budget, k=k, h_source="joint_aura_mtp_head_self_kl")
    return _mtp_record(prepared, result.rung,
        constants_source=str(constants.get("source", "undeclared")), selection=result.provenance)


def enumerate_mtp_rungs(payload: Mapping, *, byte_budget: int, constants: Mapping,
                        eligible=None, fixed_formats=None, formats=None, rung_allowability=None,
                        stats=None, context_by_unit=None, target_profile=None,
                        allowability_m=None, allowability_tensor_parallel=1) -> list[dict]:
    """Enumerate the feasible declared group choices without a winner decision.

    The stable order uses group names, member names, and rung names.
    MTP self-KL remains a separate price, not a term in the body objective.
    """
    ordered = {**payload, "groups": {group: sorted(members)
                                    for group, members in sorted(payload["groups"].items())}}
    prepared = _prepare_mtp_menu(ordered, byte_budget=byte_budget, eligible=eligible,
        fixed_formats=fixed_formats, formats=formats, rung_allowability=rung_allowability,
        stats=stats, context_by_unit=context_by_unit, target_profile=target_profile,
        allowability_m=allowability_m, allowability_tensor_parallel=allowability_tensor_parallel)
    records = []
    for rung in prepared.menu:
        if rung.resident_bytes > prepared.byte_budget:
            continue
        record = _mtp_record(prepared, rung,
            constants_source=str(constants.get("source", "undeclared")),
            selection={"regime": "option_b_enumeration", "winner_selected": False})
        record["unit_prices"] = {
            unit: {"E": prepared.rows[unit][fmt][0],
                   "resident_bytes": prepared.rows[unit][fmt][1],
                   "joint_operator_identity_sha256":
                       payload["costs"][unit].get(fmt, {}).get("joint_operator_identity_sha256")}
            for unit, fmt in sorted(record["assignment"].items())}
        records.append(record)
    return records


def backfill_mtp_selection_wires(layer_config: Mapping, cost_path) -> dict:
    """Return a new allocation config with bound wires for its existing MTP choice.

    This checks the recorded choice against the measured cost and every unit's
    config. It does not invoke an allocator or change a body/MTP assignment.
    """
    from . import format_registry as fr
    from . import mtp_rung_selection as canon

    cost_raw = Path(cost_path).read_bytes()
    cost_sha256 = bytes_sha256hex(cost_raw)
    cost = pickle.loads(cost_raw) if not str(cost_path).endswith(".json") else json.loads(cost_raw)
    payload = enrich_mtp_cost_wires(cost)
    meta = layer_config.get("__prismaquant__", {})
    record = meta.get("mtp_selection", {})
    if (record.get("schema") != RECORD_SCHEMA or
            record.get("cost_path") != str(cost_path) or
            record.get("mtp_layer") != payload["mtp_layer"] or
            record.get("units") != len(payload["costs"])):
        raise ValueError("MTP selection does not name the bound cost and unit roster")
    probe_sha256, probe = _mtp_probe(payload)
    if (record.get("probe_identity_sha256") != probe_sha256 or
            record.get("objective") != probe["objective"]["objective"]):
        raise ValueError("MTP selection probe differs from the bound cost")
    chosen = record.get("rung_by_group")
    if (not isinstance(chosen, Mapping) or set(chosen) != set(payload["groups"]) or
            record.get("rung") != "|".join(f"{group}={chosen[group]}"
                                            for group in sorted(chosen))):
        raise ValueError("MTP selection rung differs from the bound groups")
    fixed = record.get("fixed_formats")
    if fixed is not None:
        if (not isinstance(fixed, Mapping) or not set(fixed) <= set(chosen)
                or any(chosen[group] != fmt for group, fmt in fixed.items())):
            raise ValueError("MTP selected rung differs from fixed group formats")
    assignment = {unit: chosen[group] for group, members in payload["groups"].items()
                  for unit in members}
    if len(assignment) != len(payload["costs"]) or set(assignment) != set(payload["costs"]):
        raise ValueError("MTP selection groups do not partition the bound cost")
    quality_prices = _recompute_recorded_quality(payload, record.get("canonical_quality", {}))
    rows, _unattested = _unit_rows(payload, quality_prices=quality_prices)
    menu, _incomplete = canon.group_product_menu(
        {group: tuple(members) for group, members in payload["groups"].items()},
        rows, params=payload["params"])
    priced = next((entry for entry in menu if entry.name == record["rung"]), None)
    if (priced is None or priced.resident_bytes != record.get("resident_bytes") or
            priced.bits != record.get("bits") or priced.E != record.get("E") or
            priced.resident_bytes > record.get("byte_budget", -1)):
        raise ValueError("MTP selection price differs from the bound measured menu")
    for name, fmt in assignment.items():
        if layer_config.get(name) != fr.get_format(fmt).autoround_config():
            raise ValueError(f"MTP {name} config differs from its recorded selection")
    selected = {key: {} for key in ("mtp_expert_wires", "mtp_expert_wire_roots",
                                    "mtp_expert_source_bindings")}
    for name, fmt in assignment.items():
        if fmt == _BF16:
            continue
        for key in selected:
            try:
                selected[key][name] = payload[key][name][fmt]
            except KeyError as exc:
                raise ValueError(f"MTP {name}@{fmt} has no bound {key}") from exc
    enriched_record = {**record, "mtp_joint_cost_sha256": cost_sha256,
                       "mtp_expert_projection": payload["mtp_expert_projection"],
                       **selected, "mtp_expert_wire_binding_schema": WIRE_BINDING_SCHEMA}
    return {**layer_config, "__prismaquant__": {**meta, "mtp_selection": enriched_record}}


def _main() -> None:
    """Backfill a completed allocation to a new path without rerunning M6."""
    import argparse
    from .cost_stage_checkpoint import publish_new_bytes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-config", required=True)
    parser.add_argument("--mtp-joint-cost", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).resolve() == Path(args.layer_config).resolve():
        parser.error("output must be a new path, not the historical layer config")
    source = json.loads(Path(args.layer_config).read_text())
    result = backfill_mtp_selection_wires(source, args.mtp_joint_cost)
    raw = (json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if not publish_new_bytes(Path(args.output), raw):
        parser.error("output already exists; refusing overwrite")
    print(json.dumps({"output": args.output, "sha256": bytes_sha256hex(raw),
                      "selected_expert_wires": len(result["__prismaquant__"]["mtp_selection"][
                          "mtp_expert_wires"])}))


if __name__ == "__main__":
    _main()


__all__ = ["SCHEMA", "RECORD_SCHEMA", "MERGE_SCHEMA", "WIRE_BINDING_SCHEMA",
           "load_mtp_cost", "merge_mtp_costs", "enrich_mtp_cost_wires", "select_mtp_rungs",
           "enumerate_mtp_rungs",
           "backfill_mtp_selection_wires"]
