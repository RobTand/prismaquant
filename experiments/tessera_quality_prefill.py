"""Opt-in driver for the Tessera quality-prefill experiment.

The contract lives in ``prismaquant/quality_prefill_contract.py`` and performs
no I/O.  This file is the I/O shell around it: it reads manifests, seals plan
bytes, walks the section 4 phase state machine and prints reviewable JSON.

Commands
--------
``plan``     Count the phase DAG and report what is still unresolved.
``freeze``   Seal an immutable plan.  Refuses while any required value is open,
             naming every one of them.
``submit``   Submit a frozen finite phase through native Flux records.
``status``   Print each phase's state and which phases are ready.
``collect``  Authenticate saved native children and exact scientific coverage.
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
import hashlib
import json
from pathlib import Path
import sys

# Run directly (`python experiments/tessera_quality_prefill.py`) the script's own
# directory is sys.path[0], so put the repository root ahead of it.  Under
# `PYTHONPATH=. python -m ...` this is already true and the insert is a no-op.
_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from prismaquant.cost_stage_checkpoint import publish_new_bytes  # noqa: E402
from prismaquant import flux_jobs, quality_prefill_flux  # noqa: E402
from prismaquant.staged_manifest import read_document  # noqa: E402
from prismaquant.quality_prefill_contract import (  # noqa: E402
    JOINT_CURRENCY,
    PHASE_DAG,
    PHASE_STATES,
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
# I/O helpers
# ---------------------------------------------------------------------------


def _read(path: Path) -> str:
    try:
        return read_document(path.absolute(), 16 << 20)[0].decode('utf-8')
    except (OSError, ValueError, UnicodeError) as exc:
        raise QualityPrefillContractError(f"cannot read {path}: {exc}") from exc


def _read_checked(path: Path, expected_sha256: str, *, where: str) -> str:
    """Read a referenced artifact and refuse it unless its bytes match.

    A path string is not an identity: every artifact reference in this
    contract is ``{path, sha256}`` and the driver reads the pair, never
    the path alone.
    """

    text = _read(path)
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if actual != expected_sha256:
        raise QualityPrefillContractError(
            f"{where} at {path} has sha256 {actual}, "
            f"but the reference declares {expected_sha256}"
        )
    return text


def _load_manifest(path: Path, *, mode: str) -> dict:
    return parse_manifest(_read(path), mode=mode)


def _publish(path: Path, value: object) -> None:
    """Publish canonical bytes exclusively: never overwrite a frozen plan."""

    destination = path.absolute()
    encoded = canonical_json_bytes(value) + b"\n"
    if not publish_new_bytes(destination, encoded, nofollow=True):
        raise QualityPrefillContractError(
            f"refusing to overwrite {path}: frozen bytes are published once"
        )


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
        if raw["state"] not in PHASE_STATES:
            raise QualityPrefillContractError(
                f"phase {name} carries unknown state {raw['state']!r}"
            )
        if type(raw["attempt"]) is not int or raw["attempt"] < 0:
            raise QualityPrefillContractError(
                f"phase {name} attempt must be a non-negative int"
            )
        if not isinstance(raw["reason"], str):
            raise QualityPrefillContractError(f"phase {name} reason must be a string")
        states[name] = PhaseState(
            phase=name,
            state=raw["state"],
            attempt=raw["attempt"],
            reason=raw["reason"],
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
    parents = tuple(flux_jobs.load_submission(path.absolute()) for path in args.afterok_record)
    collection = quality_prefill_flux.submit_phase(_manifest_binding(args.manifest),
        phase=args.phase, checkout=Path(_ROOT),
        record_dir=args.record_dir.absolute(), afterok=parents)
    _project_native_state(states, args.phase, collection)
    if args.state_out is not None:
        _publish(args.state_out, _dump_states(states))
    _emit({"command": "submit", "phase": args.phase, "collection": collection,
           "states": _dump_states(states)})
    acknowledged = all(row['execution'] is not None for row in collection['children'])
    return 0 if acknowledged and collection['state'] in {'running', 'complete'} else 1


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
    _load_manifest(args.manifest, mode="frozen")
    states = _load_states(args.state)
    collection, phase_result = quality_prefill_flux.collect_phase_result(
        _manifest_binding(args.manifest), phase=args.phase,
        record_dir=args.record_dir.absolute(), wait=args.wait)
    _project_native_state(states, args.phase, collection, phase_result)
    if args.state_out is not None:
        _publish(args.state_out, _dump_states(states))
    _emit({"command": "collect", "phase": args.phase, "collection": collection,
           "phase_result": phase_result, "states": _dump_states(states)})
    return 0 if collection['state'] == 'complete' and phase_result is not None else 1


def _manifest_binding(path: Path) -> dict:
    raw, _ = read_document(path.absolute(), 16 << 20)
    return dict(path=str(path.absolute()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _project_native_state(states: dict, phase: str, collection: dict,
                          phase_result: dict | None = None) -> None:
    """Rebuild this phase from native facts; saved presentation cannot grant success."""
    current = initial_phase_states()[phase]
    for step in ('ready', 'submitted'):
        current = transition(current, step, max_attempts=1)
    if collection['state'] == 'failed':
        current = transition(current, 'failed', max_attempts=1,
                             reason='conclusive native child or ancestor failure')
    elif collection['state'] in {'running', 'complete'}:
        current = transition(current, 'running', max_attempts=1)
        if phase_result is not None and phase_result['terminal_state'] == 'complete':
            current = transition(current, 'collecting', max_attempts=1)
            current = transition(current, 'complete', max_attempts=1,
                receipts=phase_result['child_receipts'],
                expected_task_ids=phase_result['expected_task_ids'])
    states[phase] = current


def command_solve(args: argparse.Namespace) -> int:
    _load_manifest(args.manifest, mode="frozen")
    table_text = (
        _read(args.price_table)
        if args.price_table_sha256 is None
        else _read_checked(
            args.price_table, args.price_table_sha256, where="price table"
        )
    )
    decoded = decode_strict_json(table_text, where="price table")
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
            "(prismaquant.quality_prefill_experiment.v2)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    planned = sub.add_parser("plan", help="count the phase DAG and list unresolved inputs")
    planned.add_argument("--manifest", required=True, type=Path)

    frozen = sub.add_parser("freeze", help="seal immutable plan bytes")
    frozen.add_argument("--manifest", required=True, type=Path)
    frozen.add_argument("--out", required=True, type=Path)

    submitted = sub.add_parser("submit", help="submit one frozen finite native phase")
    submitted.add_argument("--manifest", required=True, type=Path)
    submitted.add_argument("--phase", required=True)
    submitted.add_argument("--record-dir", required=True, type=Path)
    submitted.add_argument("--afterok-record", action='append', default=[], type=Path)
    submitted.add_argument("--state", type=Path)
    submitted.add_argument(
        "--state-out",
        default=None,
        type=Path,
        help="write the advanced phase state here (refuses to overwrite)",
    )

    status = sub.add_parser("status", help="print phase states and ready phases")
    status.add_argument("--manifest", required=True, type=Path)
    status.add_argument("--state", type=Path)
    status.add_argument("--mode", choices=("draft", "frozen"), default="frozen")

    collected = sub.add_parser("collect", help="authenticate saved native phase results")
    collected.add_argument("--manifest", required=True, type=Path)
    collected.add_argument("--phase", required=True)
    collected.add_argument("--record-dir", required=True, type=Path)
    collected.add_argument("--wait", action=argparse.BooleanOptionalAction, default=True)
    collected.add_argument(
        "--state-out",
        default=None,
        type=Path,
        help="write the advanced phase state here (refuses to overwrite)",
    )
    collected.add_argument("--state", type=Path)

    solved = sub.add_parser("solve", help="validate the joint price table")
    solved.add_argument("--manifest", required=True, type=Path)
    solved.add_argument("--price-table", required=True, type=Path)
    solved.add_argument(
        "--price-table-sha256",
        default=None,
        help="declared sha256 of the price table; the read refuses on mismatch",
    )

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
    except (QualityPrefillContractError, OSError, ValueError, RuntimeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
