"""Research-only offline transition from stack evidence to missing-cell data.

No encoding, fleet dispatch, wire admission, or production allocation occurs here.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import TypedDict

from prismaquant.digests import canonical_json_bytes, canonical_json_sha256
from prismaquant.tessera_formats import TesseraFormatError
from prismaquant.tessera_rate_surface import (
    StackRateSample, selective_encode_plan, stack_transfer_regret_gate,
)


class _PlannerInputs(TypedDict):
    records: Mapping[str, StackRateSample]
    unit_bytes: Mapping[str, Mapping[int, int]]
    byte_budget: int
    winners: Mapping[str, int]
    measured_cells: Sequence[tuple[str, int, int]]
    draws: int
    seed: int
    max_regret_pct: float
    budget_sensitivity: Sequence[float]


def _require_positive_builtin_int(value: object, where: str) -> int:
    if type(value) is not int or value <= 0:
        raise TesseraFormatError(f"{where}: expected a positive integer")
    return value


def _validate_records(records):
    if not isinstance(records, Mapping) or len(records) < 2:
        raise TesseraFormatError("records must contain at least two stack samples")
    required = set()
    bands, references, currencies = set(), set(), set()
    for name, sample in records.items():
        if (not isinstance(name, str) or not name or not isinstance(sample, StackRateSample)
                or sample.stack != name):
            raise TesseraFormatError("record key must match a typed stack sample's name")
        frame = set(sample.experts)
        if (len(frame) != len(sample.experts)
                or any(type(e) is not int or e < 0 for e in frame)):
            raise TesseraFormatError(f"{name}: invalid expert frame")
        if (not sample.currency or not isinstance(sample.currency, str)
                or any(not isinstance(p, str) or not p for p in sample.projections)):
            raise TesseraFormatError(f"{name}: explicit currency and projection names required")
        if (len(set(sample.sampled_experts)) != len(sample.sampled_experts)
                or any(type(e) is not int or e not in frame for e in sample.sampled_experts)):
            raise TesseraFormatError(f"{name}: sampled experts must be unique frame members")
        reference = _require_positive_builtin_int(sample.reference_q256, f"{name} reference rung")
        targets = tuple(sorted(_require_positive_builtin_int(q, f"{name} target rung")
                               for q in sample.sampled_mse))
        references.add(reference)
        bands.add(targets)
        currencies.add(sample.currency)
        if set(sample.reference_mse) != frame:
            raise TesseraFormatError(f"{name}: reference evidence does not match expert frame")
        for rate, rows in {reference: sample.reference_mse, **sample.sampled_mse}.items():
            if not isinstance(rows, Mapping) or not set(rows) <= frame:
                raise TesseraFormatError(f"{name}: evidence names an unknown expert")
            for expert, row in rows.items():
                if (type(expert) is not int or not isinstance(row, Mapping)
                        or set(row) != set(sample.projections)):
                    raise TesseraFormatError(f"{name}: evidence projections do not match")
                for value in row.values():
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value) or value <= 0):
                        raise TesseraFormatError(f"{name}: evidence must be finite positive MSE")
                required.add((name, expert, rate))
        if sample.weights is not None:
            if set(sample.weights) != frame or any(
                isinstance(w, bool) or not isinstance(w, (int, float))
                or not math.isfinite(w) or w < 0 for w in sample.weights.values()
            ) or not any(w > 0 for w in sample.weights.values()):
                raise TesseraFormatError(f"{name}: invalid explicit expert weights")
    if len(references) != 1 or len(bands) != 1 or len(currencies) != 1:
        raise TesseraFormatError("stacks must share reference rung, target band and currency")
    return required


def plan_reduced_schedule(
    records: Mapping[str, StackRateSample], *,
    unit_bytes: Mapping[str, Mapping[int, int]], byte_budget: int,
    winners: Mapping[str, int], measured_cells: Sequence[tuple[str, int, int]],
    draws: int = 64, seed: int = 0, max_regret_pct: float = 0.1,
    budget_sensitivity: Sequence[float] = (0.5, 0.75, 1.0, 1.25),
) -> dict:
    """Request missing cells at explicit current winners or the full global band.

    The output binds supplied scalar evidence, not verified encoded-wire receipts.
    The reused gate's greedy diagnostic is not production group-knapsack regret.
    """
    required = _validate_records(records)
    if (not isinstance(winners, Mapping) or set(winners) != set(records)
            or not isinstance(unit_bytes, Mapping) or set(unit_bytes) != set(records)):
        raise TesseraFormatError("winners and byte costs must exactly match the stack roster")
    _require_positive_builtin_int(byte_budget, "byte_budget")
    if type(seed) is not int:
        raise TesseraFormatError("seed must be an integer")
    frames = {n: tuple(sorted(s.experts)) for n, s in sorted(records.items())}
    menus = {n: {s.reference_q256, *s.target_q256} for n, s in records.items()}
    for name in records:
        costs = unit_bytes[name]
        if not isinstance(costs, Mapping) or set(costs) != menus[name]:
            raise TesseraFormatError(f"{name}: byte costs must exactly cover the declared band")
        for rate, cost in costs.items():
            _require_positive_builtin_int(rate, f"{name} byte-cost rung")
            _require_positive_builtin_int(cost, f"{name} byte cost")
        if type(winners[name]) is not int or winners[name] not in menus[name]:
            raise TesseraFormatError(f"{name}: winner is not a declared rung")
    if sum(unit_bytes[n][winners[n]] for n in records) > byte_budget:
        raise TesseraFormatError("current winners exceed the declared byte budget")
    if not isinstance(measured_cells, Sequence) or isinstance(measured_cells, (str, bytes)):
        raise TesseraFormatError("measured_cells must be a sequence of triples")
    measured = set()
    for cell in measured_cells:
        if (not isinstance(cell, (list, tuple)) or len(cell) != 3
                or not isinstance(cell[0], str) or type(cell[1]) is not int
                or type(cell[2]) is not int):
            raise TesseraFormatError("measured cell must be a (stack, expert, rate) triple")
        name, expert, rate = cell
        if name not in frames or expert not in frames[name] or rate not in menus[name]:
            raise TesseraFormatError(f"unknown measured cell: {cell!r}")
        measured.add((name, expert, rate))
    if not required <= measured:
        raise TesseraFormatError("measured-cell ledger omits evidence used by the stack records")
    ordered_records = dict(sorted(records.items()))
    config = dict(draws=draws, seed=seed, max_regret_pct=max_regret_pct,
                  budget_sensitivity=tuple(budget_sensitivity))
    gate = stack_transfer_regret_gate(
        ordered_records, unit_bytes=unit_bytes, byte_budget=byte_budget,
        draws=draws, seed=seed, max_regret_pct=max_regret_pct,
        budget_sensitivity=tuple(budget_sensitivity))
    if gate["passes"]:
        selected = selective_encode_plan(winners, stack_experts=frames,
                                         measured_cells=sorted(measured))
        pending = [{k: row[k] for k in ("stack", "expert", "rate_q256")}
                   for row in selected["rows"]]
        state = "awaiting_selective_measurements" if pending else "winner_cells_covered"
        fallback = "none"
    else:
        pending = [{"stack": n, "expert": e, "rate_q256": q}
                   for n in sorted(frames) for e in frames[n] for q in sorted(menus[n])
                   if (n, e, q) not in measured]
        state = "awaiting_full_measurements" if pending else "full_band_cells_covered"
        fallback = "all_stacks"
    binding = dict(records={n: asdict(s) for n, s in ordered_records.items()},
                   winners=dict(winners), unit_bytes=unit_bytes, byte_budget=byte_budget,
                   measured_cells=sorted(measured), config=config)
    return dict(
        schema="prismaquant.stack_reduced_schedule_plan.v1",
        input_sha256=canonical_json_sha256(binding, where="reduced schedule input"),
        state=state, fallback_scope=fallback, pending_cells=pending,
        winners=dict(sorted(winners.items())), gate=gate, wire_ready=False,
        gate_allocator="greedy_diagnostic_not_production_dp",
        evidence_scope="caller_supplied_scalar_cells_not_wire_receipts",
    )


def _integer_map(value, where):
    if not isinstance(value, dict):
        raise TesseraFormatError(f"{where}: expected an object")
    result = {}
    for key, item in value.items():
        try:
            integer = int(key)
        except (TypeError, ValueError) as exc:
            raise TesseraFormatError(f"{where}: expected decimal integer keys") from exc
        if str(integer) != key:
            raise TesseraFormatError(f"{where}: noncanonical integer key {key!r}")
        result[integer] = item
    return result


def _decode_input(payload) -> _PlannerInputs:
    if not isinstance(payload, dict) or payload.get("schema") != "prismaquant.stack_reduced_schedule_input.v1":
        raise TesseraFormatError("unsupported reduced schedule input schema")
    required = {"schema", "records", "unit_bytes", "byte_budget", "winners", "measured_cells"}
    optional = {"draws", "seed", "max_regret_pct", "budget_sensitivity"}
    if not required <= set(payload) or set(payload) - required - optional:
        raise TesseraFormatError("missing or unknown reduced schedule input fields")
    records = {}
    for name, raw in payload["records"].items():
        fields = dict(raw)
        for key in ("projections", "experts", "sampled_experts"):
            fields[key] = tuple(fields[key])
        fields["reference_mse"] = _integer_map(fields["reference_mse"], "reference_mse")
        fields["sampled_mse"] = {q: _integer_map(rows, "sampled_mse experts")
                                 for q, rows in _integer_map(fields["sampled_mse"], "sampled_mse rates").items()}
        if fields.get("weights") is not None:
            fields["weights"] = _integer_map(fields["weights"], "weights")
        records[name] = StackRateSample(**fields)
    return {
        "records": records,
        "unit_bytes": {n: _integer_map(rows, "unit_bytes")
                       for n, rows in payload["unit_bytes"].items()},
        "byte_budget": payload["byte_budget"], "winners": payload["winners"],
        "measured_cells": payload["measured_cells"],
        "draws": payload.get("draws", 64), "seed": payload.get("seed", 0),
        "max_regret_pct": payload.get("max_regret_pct", 0.1),
        "budget_sensitivity": payload.get("budget_sensitivity", (0.5, 0.75, 1.0, 1.25)),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        plan = plan_reduced_schedule(**_decode_input(payload))
        encoded = canonical_json_bytes(plan, where="reduced schedule output") + b"\n"
        with args.output.open("xb") as out:
            out.write(encoded)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
