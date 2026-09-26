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


def _mtp_probe(payload) -> tuple[str, dict]:
    """The one MTP probe identity every row carries, validated."""
    from .glm_mtp import MTP_OBJECTIVE, MTP_OBJECTIVE_SCHEMA
    from .joint_aura import validate_joint_aura_entry

    digests, probe = set(), None
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
            probe = row["probe_identity"]
    if len(digests) != 1:
        raise ValueError(f"MTP rows must share one probe identity, got {len(digests)}")
    return digests.pop(), probe


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
    }


__all__ = ["SCHEMA", "RECORD_SCHEMA", "MERGE_SCHEMA", "load_mtp_cost", "merge_mtp_costs",
           "select_mtp_rungs"]
