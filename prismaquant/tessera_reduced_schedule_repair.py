"""Bounded offline history replay, not a solver or measured-wire admission."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Any

from .digests import canonical_json_bytes, canonical_json_sha256, is_sha256hex
from .tessera_formats import TesseraFormatError
from .tessera_reduced_schedule import _decode_input, plan_reduced_schedule


def _reduced_repair_fixed_inputs(inputs: Mapping[str, Any]) -> dict:
    """The declared experiment fields that observation completion cannot alter."""
    return {
        key: value for key, value in inputs.items()
        if key not in {"records", "winners", "measured_cells"}
    } | {
        "records": {
            name: {f.name: getattr(sample, f.name) for f in fields(sample)
                   if f.name not in {"reference_mse", "sampled_mse"}}
            | {"target_q256": list(sample.target_q256)}
            for name, sample in inputs["records"].items()
        }
    }


def _reduced_repair_observations(inputs: Mapping[str, Any]) -> dict:
    rows = {}
    for name, sample in inputs["records"].items():
        for rate, experts in {sample.reference_q256: sample.reference_mse,
                              **sample.sampled_mse}.items():
            for expert, row in experts.items():
                rows[(name, expert, rate)] = row
    return rows


def _reduced_repair_receipt(plans: list[dict], steps: list[dict], cap: int) -> dict:
    current = plans[-1]
    used = len(plans) - 1
    if used == cap and current["pending_cells"]:
        raise TesseraFormatError("repair round cap exhausted with cells outstanding")
    out = {
        "schema": "prismaquant.stack_reduced_schedule_repair.v1",
        "root_input_sha256": plans[0]["input_sha256"],
        "max_rounds": cap,
        "rounds_used": used,
        "steps": steps,
        "current_plan": current,
        "state": "awaiting_measurements" if current["pending_cells"] else "scalar_coverage_only",
        "wire_ready": False,
        "production_solver_run": False,
    }
    out["binding_sha256"] = canonical_json_sha256(out, where="reduced repair receipt")
    return out


def plan_reduced_schedule_repair(
    history: Sequence[Mapping[str, Any]], *, max_rounds: int,
    expected_previous_sha256: str | None = None,
) -> dict:
    """Replay supplied assignments/observations under one fixed, bounded contract.

    Histories contain closed reduced-schedule JSON input objects. Appending a
    snapshot requires the preceding receipt's digest, complete prior requests,
    monotonic coverage and immutable prior scalar observations. Fixed inputs
    and prior rows are compared as canonical JSON bytes, not Python numeric
    equality. Neither a checksum nor scalar coverage authenticates a measurement
    or a wire.
    """
    if type(max_rounds) is not int or max_rounds <= 0:
        raise TesseraFormatError("max_rounds must be an explicit positive integer")
    if (not isinstance(history, Sequence) or isinstance(history, (str, bytes))
            or not history):
        raise TesseraFormatError("history must be a nonempty sequence of input objects")
    if len(history) - 1 > max_rounds:
        raise TesseraFormatError("repair history exceeds the declared round cap")
    if len(history) == 1:
        if expected_previous_sha256 is not None:
            raise TesseraFormatError("a root history must not supply a previous receipt")
    elif not is_sha256hex(expected_previous_sha256):
        raise TesseraFormatError("appended history needs a previous receipt SHA256")

    plans: list[dict] = []
    steps: list[dict] = []
    prior_cells: set[tuple[str, int, int]] = set()
    prior_rows: dict = {}
    fixed = None
    for index, snapshot in enumerate(history):
        try:
            inputs = _decode_input(dict(snapshot))
            plan = plan_reduced_schedule(**inputs)
        except (TypeError, ValueError, KeyError, AttributeError) as exc:
            raise TesseraFormatError(f"history[{index}]: invalid input: {exc}") from exc
        cells = {(c[0], c[1], c[2]) for c in inputs["measured_cells"]}
        rows = _reduced_repair_observations(inputs)
        declared = canonical_json_bytes(
            _reduced_repair_fixed_inputs(inputs), where="repair fixed inputs")
        if index == 0:
            fixed = declared
        else:
            if declared != fixed:
                raise TesseraFormatError("repair history changed the declared experiment")
            if not prior_cells <= cells:
                raise TesseraFormatError("repair history lost previously covered cells")
            requested = {(c["stack"], c["expert"], c["rate_q256"])
                         for c in plans[-1]["pending_cells"]}
            if not requested <= cells:
                raise TesseraFormatError("prior pending requests are not completed")
            if not (cells - prior_cells) <= set(rows):
                raise TesseraFormatError("new completion cells have no scalar observations")
            for cell, row in prior_rows.items():
                if cell not in rows or canonical_json_bytes(
                        row, where="prior scalar row") != canonical_json_bytes(
                        rows[cell], where="current scalar row"):
                    raise TesseraFormatError("repair history rewrote a prior scalar observation")
        if index == len(history) - 1 and index > 0:
            prefix = _reduced_repair_receipt(plans, steps, max_rounds)
            if prefix["binding_sha256"] != expected_previous_sha256:
                raise TesseraFormatError("previous repair receipt binding does not match history")
        steps.append({
            "input_sha256": plan["input_sha256"],
            "plan_sha256": canonical_json_sha256(plan, where="repair step plan"),
            "winners_changed": bool(plans and plan["winners"] != plans[-1]["winners"]),
            "added_measured_cell_count": len(cells - prior_cells),
        })
        plans.append(plan)
        prior_cells, prior_rows = cells, rows
    return _reduced_repair_receipt(plans, steps, max_rounds)
