"""Opt-in driver for the Tessera quality-prefill experiment.

The contract lives in ``prismaquant/quality_prefill_contract.py`` and performs
no I/O.  This file is the I/O shell around it: it reads manifests, seals plan
bytes, walks the section 4 phase state machine and prints reviewable JSON.

Commands
--------
``plan``     Count the phase DAG and report what is still unresolved.
``freeze``   Seal an immutable plan.  Refuses while any required value is open,
             naming every one of them.
``submit``   Hand a ready phase's task roster to PrismaBuild.  The adapter is a
             separate work package; this driver holds one named seam,
             :func:`submit_phase_to_prismabuild`, and nothing else.
``status``   Print each phase's state and which phases are ready.
``collect``  Adopt child receipts and advance a phase to ``complete``.
``solve``    Validate the joint price table and refuse a mixed currency.
``report``   Emit the frontier report skeleton and the run's coverage.

``plan``, ``status`` and ``report`` accept an unfrozen draft; ``submit``,
``collect`` and ``solve`` require a frozen manifest, because a numerical
execution phase may not silently supply a value nobody chose.

Nothing here touches a GPU.  The legal-domain inventory, population selection,
transfer forms and the allocator are other work packages; this driver consumes
their artifacts by reference and never re-implements them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Run directly (`python experiments/tessera_quality_prefill.py`) the script's own
# directory is sys.path[0], so put the repository root ahead of it.  Under
# `PYTHONPATH=. python -m ...` this is already true and the insert is a no-op.
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from prismaquant.cost_stage_checkpoint import atomic_write_bytes  # noqa: E402
from prismaquant.quality_prefill_contract import (  # noqa: E402
    JOINT_CURRENCY,
    PHASE_DAG,
    PhaseState,
    QualityPrefillContractError,
    canonical_json_bytes,
    canonical_sha256,
    decode_strict_json,
    freeze_manifest,
    initial_phase_states,
    parse_manifest,
    ready_phases,
    transition,
    unresolved_inputs,
    validate_frontier_report,
    validate_price_table,
)


COMMANDS = ("plan", "freeze", "submit", "status", "collect", "solve", "report")


# ---------------------------------------------------------------------------
# The one PrismaBuild seam
# ---------------------------------------------------------------------------


def submit_phase_to_prismabuild(
    *,
    manifest: dict,
    phase: str,
    task_roster: dict,
    priority: int,
) -> dict:
    """Submit one phase's task roster to PrismaBuild and return its receipt.

    This is the only seam the PQ -> PB adapter work package fills.  It is
    deliberately unimplemented here so that a half-built adapter cannot burn a
    real queue slot.

    Arguments
        manifest: the frozen, sealed manifest (``identity_sha256`` present).
        phase: a phase name from ``PHASE_DAG``.
        task_roster: the decoded, strictly parsed roster artifact named by
            ``manifest["execution"]["phases"][phase]["task_roster"]``.
        priority: ``manifest["execution"]["pb_policy"]["priority"]``.

    Returns a submission receipt::

        {
          "schema": "prismaquant.quality_prefill_experiment.submission.v1",
          "phase": <phase>,
          "plan_identity_sha256": <manifest identity>,
          "action_keys": [<one PB action key per submitted task>],
          "task_ids": [<the task ids those action keys cover, in order>],
          "priority": <int>,
        }

    ``collect`` consumes exactly that shape: ``task_ids`` must be an exact
    cover of the phase's expected task set, and each ``action_keys[i]`` names
    the action that produced ``task_ids[i]``.
    """

    raise NotImplementedError(
        "the PQ -> PB submission adapter is a separate work package; "
        "fill experiments/tessera_quality_prefill.submit_phase_to_prismabuild"
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise QualityPrefillContractError(f"cannot read {path}: {exc}") from exc


def _load_manifest(path: Path, *, mode: str) -> dict:
    return parse_manifest(_read(path), mode=mode)


def _publish(path: Path, value: object) -> None:
    """Publish canonical bytes exclusively: never overwrite a frozen plan."""

    if path.exists():
        raise QualityPrefillContractError(
            f"refusing to overwrite {path}: frozen bytes are published once"
        )
    atomic_write_bytes(path, canonical_json_bytes(value) + b"\n")


def _emit(payload: object) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2))


def _load_states(path: Path | None) -> dict[str, PhaseState]:
    if path is None:
        return initial_phase_states()
    decoded = decode_strict_json(_read(path), where="phase state")
    if not isinstance(decoded, dict):
        raise QualityPrefillContractError("phase state root must be an object")
    states = initial_phase_states()
    for name, raw in decoded.items():
        if name not in states:
            raise QualityPrefillContractError(f"unknown phase {name!r} in state file")
        if not isinstance(raw, dict) or set(raw) != {"state", "attempt", "reason"}:
            raise QualityPrefillContractError(f"phase {name} state fields differ")
        states[name] = PhaseState(
            phase=name,
            state=str(raw["state"]),
            attempt=int(raw["attempt"]),
            reason=str(raw["reason"]),
        )
    return states


def _dump_states(states: dict[str, PhaseState]) -> dict:
    return {
        name: {"state": item.state, "attempt": item.attempt, "reason": item.reason}
        for name, item in states.items()
    }


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def command_plan(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest, mode="draft")
    open_inputs = unresolved_inputs(manifest)
    _emit(
        {
            "command": "plan",
            "experiment_id": manifest["experiment_id"],
            "schema": manifest["schema"],
            "draft_body_sha256": canonical_sha256(manifest),
            "phase_count": len(PHASE_DAG),
            "phases": [
                {"phase": spec.phase, "title": spec.title, "dependencies": list(spec.dependencies)}
                for spec in PHASE_DAG
            ],
            "unresolved_inputs": open_inputs,
            "freezable": not open_inputs,
        }
    )
    return 0


def command_freeze(args: argparse.Namespace) -> int:
    draft = _load_manifest(args.manifest, mode="draft")
    frozen = freeze_manifest(draft)
    _publish(args.out, frozen)
    _emit(
        {
            "command": "freeze",
            "experiment_id": frozen["experiment_id"],
            "identity_sha256": frozen["identity_sha256"],
            "out": str(args.out),
        }
    )
    return 0


def command_submit(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest, mode="frozen")
    states = _load_states(args.state)
    if args.phase not in states:
        raise QualityPrefillContractError(f"unknown phase {args.phase!r}")
    budget = manifest["execution"]["retry_policy"]["max_attempts"]
    if states[args.phase].state != "ready":
        if args.phase not in ready_phases(states):
            raise QualityPrefillContractError(
                f"{args.phase} is {states[args.phase].state} and its dependencies are "
                "not all complete: only a ready phase is submitted"
            )
        states[args.phase] = transition(states[args.phase], "ready", max_attempts=budget)
    entry = manifest["execution"]["phases"][args.phase]
    roster = decode_strict_json(
        _read(Path(entry["task_roster"]["path"])), where=f"{args.phase} task roster"
    )
    if not isinstance(roster, dict):
        raise QualityPrefillContractError("task roster root must be an object")
    receipt = submit_phase_to_prismabuild(
        manifest=manifest,
        phase=args.phase,
        task_roster=roster,
        priority=manifest["execution"]["pb_policy"]["priority"],
    )
    states[args.phase] = transition(states[args.phase], "submitted", max_attempts=budget)
    _emit({"command": "submit", "phase": args.phase, "receipt": receipt,
           "states": _dump_states(states)})
    return 0


def command_status(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest, mode=args.mode)
    states = _load_states(args.state)
    _emit(
        {
            "command": "status",
            "experiment_id": manifest["experiment_id"],
            "frozen": "identity_sha256" in manifest,
            "unresolved_inputs": unresolved_inputs(manifest),
            "states": _dump_states(states),
            "ready": list(ready_phases(states)),
        }
    )
    return 0


def command_collect(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest, mode="frozen")
    states = _load_states(args.state)
    decoded = decode_strict_json(_read(args.receipts), where="child receipts")
    if not isinstance(decoded, dict) or set(decoded) != {"task_ids", "receipts"}:
        raise QualityPrefillContractError(
            "child receipts must be {'task_ids': [...], 'receipts': [...]}"
        )
    states[args.phase] = transition(
        states[args.phase],
        "complete",
        max_attempts=manifest["execution"]["retry_policy"]["max_attempts"],
        receipts=decoded["receipts"],
        expected_task_ids=decoded["task_ids"],
    )
    _emit({"command": "collect", "phase": args.phase, "states": _dump_states(states)})
    return 0


def command_solve(args: argparse.Namespace) -> int:
    _load_manifest(args.manifest, mode="frozen")
    decoded = decode_strict_json(_read(args.price_table), where="price table")
    table = validate_price_table(decoded, where="price_table")
    if table["currency"] != JOINT_CURRENCY:
        raise QualityPrefillContractError(
            f"the solve gate reads a {JOINT_CURRENCY} table; "
            f"{table['currency']} is a screen, not a shipping currency"
        )
    _emit(
        {
            "command": "solve",
            "currency": table["currency"],
            "rows": len(table["rows"]),
            "context_sha256": table["context_sha256"],
            "table_sha256": canonical_sha256(table),
            "frontier": "the allocator work package owns the sweep; this driver "
            "validates its inputs only",
        }
    )
    return 0


def command_report(args: argparse.Namespace) -> int:
    manifest = _load_manifest(args.manifest, mode=args.mode)
    states = _load_states(args.state)
    payload = {
        "command": "report",
        "experiment_id": manifest["experiment_id"],
        "frozen": "identity_sha256" in manifest,
        "unresolved_inputs": unresolved_inputs(manifest),
        "phase_states": _dump_states(states),
        "complete_phases": sorted(
            name for name, item in states.items() if item.state == "complete"
        ),
    }
    if args.frontier is not None:
        decoded = decode_strict_json(_read(args.frontier), where="frontier report")
        report = validate_frontier_report(decoded)
        payload["frontier_report_id"] = report["report_id"]
        payload["proposed_points"] = len(report["proposed_points"])
        payload["served_points"] = len(report["served_points"])
        payload["omissions"] = list(report["omissions"])
    _emit(payload)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tessera_quality_prefill.py",
        description=(
            "Opt-in driver for the Tessera quality-prefill experiment "
            "(prismaquant.quality_prefill_experiment.v1)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    planned = sub.add_parser("plan", help="count the phase DAG and list unresolved inputs")
    planned.add_argument("--manifest", required=True, type=Path)

    frozen = sub.add_parser("freeze", help="seal immutable plan bytes")
    frozen.add_argument("--manifest", required=True, type=Path)
    frozen.add_argument("--out", required=True, type=Path)

    submitted = sub.add_parser("submit", help="hand a ready phase to PrismaBuild")
    submitted.add_argument("--manifest", required=True, type=Path)
    submitted.add_argument("--phase", required=True)
    submitted.add_argument("--state", type=Path)

    status = sub.add_parser("status", help="print phase states and ready phases")
    status.add_argument("--manifest", required=True, type=Path)
    status.add_argument("--state", type=Path)
    status.add_argument("--mode", choices=("draft", "frozen"), default="frozen")

    collected = sub.add_parser("collect", help="adopt child receipts for a phase")
    collected.add_argument("--manifest", required=True, type=Path)
    collected.add_argument("--phase", required=True)
    collected.add_argument("--receipts", required=True, type=Path)
    collected.add_argument("--state", type=Path)

    solved = sub.add_parser("solve", help="validate the joint price table")
    solved.add_argument("--manifest", required=True, type=Path)
    solved.add_argument("--price-table", required=True, type=Path)

    reported = sub.add_parser("report", help="emit coverage and the frontier report")
    reported.add_argument("--manifest", required=True, type=Path)
    reported.add_argument("--state", type=Path)
    reported.add_argument("--frontier", type=Path)
    reported.add_argument("--mode", choices=("draft", "frozen"), default="frozen")
    return parser


_DISPATCH = {
    "plan": command_plan,
    "freeze": command_freeze,
    "submit": command_submit,
    "status": command_status,
    "collect": command_collect,
    "solve": command_solve,
    "report": command_report,
}
assert set(_DISPATCH) == set(COMMANDS)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _DISPATCH[args.command](args)
    except QualityPrefillContractError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except NotImplementedError as exc:
        print(f"UNIMPLEMENTED SEAM: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
