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


def _unit_rows(payload) -> dict:
    """``{unit: {rung: (E, bytes)}}``, with BF16 passthrough where the source is BF16."""
    groups = payload["groups"]
    units = [unit for members in groups.values() for unit in members]
    if set(units) != set(payload["costs"]) or len(units) != len(set(units)):
        raise ValueError("MTP groups must partition the priced units")
    rows = {}
    for unit in units:
        wire = payload["wire_bytes"].get(unit, {})
        if set(wire) != set(payload["costs"][unit]):
            raise ValueError(f"MTP unit {unit}: wire bytes and costs name different rungs")
        if _BF16 in payload["costs"][unit]:
            raise ValueError(f"MTP unit {unit}: BF16 is passthrough, not a priced row")
        rows[unit] = {rung: (float(row["predicted_dloss"]), int(wire[rung]))
                      for rung, row in payload["costs"][unit].items()}
        if payload["source_dtype"][unit] == "bfloat16":
            rows[unit][_BF16] = (0.0, 2 * int(payload["params"][unit]))
    return rows


def select_mtp_rungs(payload: Mapping, *, byte_budget: int, constants: Mapping,
                     acceptance_points=(), k: int = 1) -> dict:
    """The MTP assignment and its selection record under ``byte_budget``.

    ``constants`` are the caller's declared serve constants
    (``t_ms``, ``d0_ms``, ``c_ms_per_bit`` and a ``source``); they are
    recorded, and with no ``acceptance_points`` they cannot move the choice:
    the selector is degenerate and returns the lowest-E rung within the budget.
    """
    from . import mtp_rung_selection as canon

    payload = dict(payload)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"MTP cost payload must be {SCHEMA}")
    probe_sha256, probe = _mtp_probe(payload)
    rows = _unit_rows(payload)
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
        "selection": result.provenance,
        "assignment": assignment,
    }


__all__ = ["SCHEMA", "RECORD_SCHEMA", "load_mtp_cost", "select_mtp_rungs"]
