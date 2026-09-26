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
import hashlib
import pickle
from pathlib import Path
from typing import Mapping

SCHEMA = "prismaquant.glm_mtp_cost.v1"
RECORD_SCHEMA = "prismaquant.glm_mtp_selection.v1"
_BF16 = "BF16"


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
        for unit, by_rung in payload["costs"].items():
            if set(payload["wire_bytes"].get(unit, {})) != set(by_rung):
                raise ValueError(f"MTP cost part {index}, unit {unit}: wire bytes and costs "
                                 "name different rungs")
            twice = sorted(set(by_rung) & set(costs[unit]))
            if twice:
                raise ValueError(f"MTP unit {unit}: rung(s) {twice} priced by more than one part")
            costs[unit].update(by_rung)
            wire[unit].update(payload["wire_bytes"][unit])
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
                       "provenance": payload.get("provenance", {})}
                      for index, payload in enumerate(payloads)],
        },
    }


def _bound_payload(reference: Mapping, *, label: str) -> dict:
    """Read an already published cost artifact only under its exact byte anchor."""
    from .tessera_joint_allocation import _read_bound

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
                if (payload["costs"].get(name, {}).get(fmt) != row
                        or payload["wire_bytes"].get(name, {}).get(fmt) !=
                        m4.get("wire_bytes", {}).get(name, {}).get(fmt)):
                    raise ValueError(f"MTP {name}@{fmt} differs from bound M4 price")
                if (m3.get("costs", {}).get(name, {}).get(fmt, {}).get("wire_bytes") !=
                        m4["wire_bytes"][name][fmt]):
                    raise ValueError(f"MTP {name}@{fmt} wire bytes differ from the M3 price")
                if name not in units:
                    # Dense/shared rows are priced by the same M3 source but
                    # are not members of its routed expert projection. Their
                    # selected BF16 passthrough needs no producer wire.
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
                if (m4["wire_bytes"][name][fmt] !=
                        checked["blob_bytes"]):
                    raise ValueError(f"MTP {name}@{fmt} wire bytes differ from the M3 price")
                receipts[name][fmt] = checked
                wire_roots[name][fmt] = root
                bindings[name][fmt] = {"m4": dict(m4_ref),
                                       "m3": dict(anchors["inputs"]["merged_cost"])}
    if seen != {(name, fmt) for name, by_fmt in payload["costs"].items()
                for fmt in by_fmt}:
        raise ValueError("MTP bound parts do not cover the merged priced cells")
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
            for rung, row in by_rung.items():
                if not validate_joint_aura_entry(row):
                    raise ValueError(f"MTP row {unit} @ {rung} is not a joint-AURA entry")
                operator = row["joint_operator_identity"]
                if operator["qname"] != unit or operator["format"] != rung:
                    raise ValueError(f"MTP row {unit} @ {rung} names {operator['qname']} @ {operator['format']}")
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


def _unit_rows(payload, eligible=None) -> tuple[dict, dict]:
    """``{unit: {rung: (E, bytes)}}`` and the priced rungs the runtime does not attest.

    BF16 passthrough is added where the source is BF16. A priced rung that
    ``eligible(unit, rung)`` refuses is not offered; it is returned as
    ``{rung: [units]}`` so the narrowing is recorded, not inferred.
    """
    groups = payload["groups"]
    units = [unit for members in groups.values() for unit in members]
    if set(units) != set(payload["costs"]) or len(units) != len(set(units)):
        raise ValueError("MTP groups must partition the priced units")
    rows, unattested = {}, {}
    for unit in units:
        wire = payload["wire_bytes"].get(unit, {})
        if set(wire) != set(payload["costs"][unit]):
            raise ValueError(f"MTP unit {unit}: wire bytes and costs name different rungs")
        if _BF16 in payload["costs"][unit]:
            raise ValueError(f"MTP unit {unit}: BF16 is passthrough, not a priced row")
        rows[unit] = {}
        for rung, row in payload["costs"][unit].items():
            if eligible is not None and not eligible(unit, rung):
                unattested.setdefault(rung, []).append(unit)
                continue
            rows[unit][rung] = (float(row["predicted_dloss"]), int(wire[rung]))
        if payload["source_dtype"][unit] == "bfloat16":
            rows[unit][_BF16] = (0.0, 2 * int(payload["params"][unit]))
    return rows, {rung: sorted(units) for rung, units in sorted(unattested.items())}


def select_mtp_rungs(payload: Mapping, *, byte_budget: int, constants: Mapping,
                     acceptance_points=(), k: int = 1, eligible=None) -> dict:
    """The MTP assignment and its selection record under ``byte_budget``.

    ``constants`` are the caller's declared serve constants
    (``t_ms``, ``d0_ms``, ``c_ms_per_bit`` and a ``source``); they are
    recorded, and with no ``acceptance_points`` they cannot move the choice:
    the selector is degenerate and returns the lowest-E rung within the budget.
    ``eligible(unit, rung)``, when given, is the pinned runtime's attestation
    (principle 14); a priced rung it refuses is left off the menu and recorded.
    """
    from . import mtp_rung_selection as canon

    payload = dict(payload)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"MTP cost payload must be {SCHEMA}")
    probe_sha256, probe = _mtp_probe(payload)
    rows, unattested = _unit_rows(payload, eligible)
    groups = {name: tuple(members) for name, members in payload["groups"].items()}
    menu, incomplete = canon.group_product_menu(groups, rows, params=payload["params"])
    serve = canon.ServeConstants(t_ms=float(constants["t_ms"]), d0_ms=float(constants["d0_ms"]),
                                 c_ms_per_bit=float(constants["c_ms_per_bit"]))
    points = [canon.AcceptancePoint(**point) for point in acceptance_points]
    result = canon.select_rung(menu, serve, points, mem_budget_bytes=int(byte_budget), k=k,
                               h_source="joint_aura_mtp_head_self_kl")
    chosen = dict(part.split("=", 1) for part in result.rung.name.split("|"))
    assignment = {unit: chosen[group] for group, members in groups.items() for unit in members}
    selected_wires = {}
    parts = payload.get("provenance", {}).get("parts", [])
    if any(isinstance(part.get("source"), Mapping) and
           "sha256" in part["source"] for part in parts):
        # The original cost remains a valid historical input. For bound M4
        # parts, however, a selected Tessera cell must retain its exact M3
        # receipt and root or export would have to encode unpriced bytes.
        payload = enrich_mtp_cost_wires(payload)
        priced = {key: {} for key in ("mtp_expert_wires", "mtp_expert_wire_roots",
                                      "mtp_expert_source_bindings")}
        for name, fmt in assignment.items():
            if fmt == _BF16:
                continue
            for key in priced:
                try:
                    priced[key][name] = payload[key][name][fmt]
                except KeyError as exc:
                    raise ValueError(f"MTP {name}@{fmt} has no {key}") from exc
        selected_wires = {"mtp_expert_projection": payload["mtp_expert_projection"],
                          **priced, "mtp_expert_wire_binding_schema": WIRE_BINDING_SCHEMA}
    return {
        "schema": RECORD_SCHEMA,
        "objective": probe["objective"]["objective"],
        "mtp_layer": int(payload["mtp_layer"]),
        "probe_identity_sha256": probe_sha256,
        "byte_budget": int(byte_budget),
        "rung": result.rung.name,
        "rung_by_group": chosen,
        "resident_bytes": int(result.rung.resident_bytes),
        "bits": float(result.rung.bits),
        "E": float(result.rung.E),
        "constants_source": str(constants.get("source", "undeclared")),
        "incomplete_rungs": incomplete,
        "unattested_rungs": {rung: len(units) for rung, units in unattested.items()},
        "selection": result.provenance,
        "assignment": assignment,
        **selected_wires,
    }


def backfill_mtp_selection_wires(layer_config: Mapping, cost_path) -> dict:
    """Return a new allocation config with bound wires for its existing MTP choice.

    This checks the recorded choice against the measured cost and every unit's
    config. It does not invoke an allocator or change a body/MTP assignment.
    """
    from . import format_registry as fr
    from . import mtp_rung_selection as canon

    cost_raw = Path(cost_path).read_bytes()
    cost_sha256 = hashlib.sha256(cost_raw).hexdigest()
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
    assignment = {unit: chosen[group] for group, members in payload["groups"].items()
                  for unit in members}
    if len(assignment) != len(payload["costs"]) or set(assignment) != set(payload["costs"]):
        raise ValueError("MTP selection groups do not partition the bound cost")
    rows, _unattested = _unit_rows(payload)
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
    print(json.dumps({"output": args.output, "sha256": hashlib.sha256(raw).hexdigest(),
                      "selected_expert_wires": len(result["__prismaquant__"]["mtp_selection"][
                          "mtp_expert_wires"])}))


if __name__ == "__main__":
    _main()


__all__ = ["SCHEMA", "RECORD_SCHEMA", "MERGE_SCHEMA", "WIRE_BINDING_SCHEMA",
           "load_mtp_cost", "merge_mtp_costs", "enrich_mtp_cost_wires", "select_mtp_rungs",
           "backfill_mtp_selection_wires"]
