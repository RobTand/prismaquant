"""Prefill-vs-accuracy frontier: one measured-runtime allocator solve per SLO.

The allocator's measured-runtime path (``allocator.py`` with
``--measured-runtime-table``) answers one question: the lowest predicted
Δloss assignment that fits the byte budget *and* one declared prefill p95-TTFT
budget. This module asks that question across a grid of prefill budgets and
records the whole answer -- the curve, not a knee -- as a
``prismaquant.prefill_frontier.v1`` document.

Every grid point is an ordinary single solve. ``allocator.main`` loads, checks
and prices the table exactly once (``MeasuredRuntimeSweep``), then this module
calls its ``solve`` per point; each call runs the same ``_solve_for_target``
path, the same ``solve_runtime_frontier`` search and the same exact payload
and serving checks a lone ``--slo-prefill-p95-ttft-ms`` run takes. Nothing
about the DP is re-implemented here, and a point in the document is what the
single-solve CLI would have produced at that SLO.

What the document says, and what it does not
--------------------------------------------
* ``points``: per SLO, feasibility (or the solver's refusal reason), the
  objective (``predicted_dloss``), exact payload bytes, the attained
  operator-sum prefill (fixed work included), decode when the table prices it,
  the assignment digest and the file it was written to, and the dominance flag.
* ``nondominated``: the lower envelope in (attained prefill, predicted Δloss),
  the same rule ``select_validated_frontier`` applies to its measured (bpp, KL)
  rows, with the same noise floor semantics. The predicted objective is
  deterministic, so the floor defaults to exactly zero.
* ``saturation``: the SLO beyond which relaxing the budget no longer changes
  the assignment. It is *measured*, not chosen: the attained prefill of the
  solve at the table's own upper bound (fixed work plus every unit's slowest
  priced option). The document then checks that every grid point at or above
  it returned the same assignment and says so.
* ``slo_axis``: the lower bound the table implies (fixed work plus every
  unit's fastest priced option) and that upper bound. Points below the lower
  bound are recorded as infeasible; they are not pruned.

The curve is a *proposal* under the table's sequential operator-sum model.
Operator sums do not certify p95 TTFT (``measured_runtime_prices``); the
served, fixed-teacher gates in ``docs/design/joint_aura_runtime_allocation.md``
still decide what ships. No knee is reported: the measured operating-point
frontier is log-linear with no interior optimum, so any single "best" point
would be a heuristic, and CLAUDE.md demotes kneedle to a diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Sequence

from .layer_config import LAYER_CONFIG_META_KEY
from .measured_runtime_prices import identity_sha256

SCHEMA = "prismaquant.prefill_frontier.v1"
ASSIGNMENT_SCHEMA = "prismaquant.prefill_frontier.assignment.v1"


class PrefillFrontierError(ValueError):
    """A sweep request this module refuses to run."""


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #

def parse_grid_spec(slo_ms: str | None, slo_grid: str | None) -> tuple[str, object]:
    """Return ``("explicit", [ms...])``, ``("auto", None)`` or ``("linear", n)``."""
    if (slo_ms is None) == (slo_grid is None):
        raise PrefillFrontierError("pass exactly one of --slo-ms or --slo-grid")
    if slo_ms is not None:
        values = []
        for token in slo_ms.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                value = float(token)
            except ValueError:
                raise PrefillFrontierError(f"--slo-ms: {token!r} is not a number") from None
            if not math.isfinite(value) or value <= 0:
                raise PrefillFrontierError(f"--slo-ms: {token!r} must be positive and finite")
            values.append(value)
        if not values:
            raise PrefillFrontierError("--slo-ms is empty")
        return "explicit", sorted(set(values))
    if slo_grid.strip().lower() == "auto":
        return "auto", None
    try:
        count = int(slo_grid)
    except ValueError:
        raise PrefillFrontierError("--slo-grid must be 'auto' or an integer >= 2") from None
    if count < 2:
        raise PrefillFrontierError("--slo-grid must be 'auto' or an integer >= 2")
    return "linear", count


def linear_grid(lower_ms: float, upper_ms: float, count: int) -> list[float]:
    """``count`` SLOs from the table lower bound to saturation, ends included."""
    if upper_ms < lower_ms:
        raise PrefillFrontierError(
            f"grid upper bound {upper_ms} is below lower bound {lower_ms}")
    if upper_ms == lower_ms:
        return [lower_ms]
    step = (upper_ms - lower_ms) / (count - 1)
    values = [lower_ms + step * i for i in range(count)]
    values[-1] = upper_ms
    return values


# --------------------------------------------------------------------------- #
# Document
# --------------------------------------------------------------------------- #

def _attained(record: dict, key: str):
    predicted = record.get("serve_constraints", {}).get("predicted", {})
    return predicted.get(key)


def nondominated_flags(points: Sequence[dict], *, loss_noise_floor: float) -> list[bool]:
    """The validated frontier's lower-envelope rule on (attained prefill, Δloss).

    ``select_validated_frontier.measured_frontier`` is written for measured
    (bpp, KL) rows: sort by the x axis, admit a point only when it lowers the
    running best y by more than the noise floor. The rule is generic in its
    two axes, so it is applied here under its own column names rather than
    copied: ``bpp`` carries attained prefill, ``kl`` carries predicted Δloss.
    Infeasible points are never on the envelope.
    """
    from .select_validated_frontier import measured_frontier

    rows = []
    for index, point in enumerate(points):
        if not point["feasible"]:
            continue
        rows.append({"label": str(index), "path": point["assignment_path"],
                     "bpp": point["attained_prefill_ms"],
                     "kl": point["predicted_dloss"]})
    envelope = measured_frontier(rows, kl_noise_floor=loss_noise_floor, tail_veto=None)
    kept = {row["label"] for row in envelope}
    return [str(index) in kept for index in range(len(points))]


def build_frontier_document(points: Sequence[dict], *, saturation: dict | None,
                            slo_axis: dict, loss_noise_floor: float,
                            provenance: dict) -> dict:
    """Assemble the v1 document from per-point records (pure; no solving)."""
    points = [dict(point) for point in points]
    flags = nondominated_flags(points, loss_noise_floor=loss_noise_floor)
    for point, flag in zip(points, flags):
        point["nondominated"] = flag
    feasible = [point for point in points if point["feasible"]]
    monotone_violations = []
    previous = None
    for point in feasible:
        if previous is not None and point["predicted_dloss"] > previous["predicted_dloss"]:
            monotone_violations.append({"slo_ms": point["slo_ms"],
                                        "predicted_dloss": point["predicted_dloss"],
                                        "previous_slo_ms": previous["slo_ms"],
                                        "previous_predicted_dloss": previous["predicted_dloss"]})
        previous = point
    if saturation is not None:
        above = [point for point in points if point["slo_ms"] >= saturation["slo_ms"]]
        differing = [point["slo_ms"] for point in above
                     if not point["feasible"]
                     or point["assignment_sha256"] != saturation["assignment_sha256"]]
        saturation = {**saturation, "points_at_or_above": len(above),
                      "verified": bool(above) and not differing,
                      "differing_slo_ms": differing}
    return {
        "schema": SCHEMA,
        "status": "proposal_data",
        "composition": "sequential_operator_sum",
        "certifies_p95": False,
        "certifies_end_to_end_slo": False,
        "objective": "predicted_dloss",
        "loss_noise_floor": float(loss_noise_floor),
        "slo_axis": dict(slo_axis),
        "saturation": saturation,
        "n_points": len(points),
        "n_feasible": len(feasible),
        "n_nondominated": sum(flags),
        "distinct_assignments": len({point["assignment_sha256"] for point in feasible}),
        "monotone_loss": not monotone_violations,
        "monotone_loss_violations": monotone_violations,
        "points": points,
        "provenance": dict(provenance),
    }


def _point_record(record: dict, assignments_dir: Path, *, provenance_stub: dict) -> dict:
    """One grid point from the allocator's solve record; writes its assignment."""
    diag = record.get("diagnostics", {})
    point = {
        "slo_ms": float(record["slo_ms"]),
        "target_bits": float(record["target_bits"]),
        "feasible": bool(record["feasible"]),
        "refusal_reason": None if record["feasible"] else record.get("reason"),
        "predicted_dloss": None,
        "achieved_bits": None,
        "payload_bytes": None,
        "attained_prefill_ms": None,
        "attained_decode_ms": None,
        "device_memory_bytes": None,
        "assignment_sha256": None,
        "assignment_path": None,
        "solver": {key: diag.get(key) for key in (
            "frontier_size", "transitions", "solver_seconds", "refusal",
            "refusal_detail", "prefill_slo_breakpoints_ms")},
    }
    if not record["feasible"]:
        return point
    assignment = dict(record["assignment"])
    digest = identity_sha256(assignment)
    path = assignments_dir / f"{digest}.json"
    if not path.exists():
        assignments_dir.mkdir(parents=True, exist_ok=True)
        payload = {**assignment, LAYER_CONFIG_META_KEY: {
            "schema": ASSIGNMENT_SCHEMA, "assignment_sha256": digest,
            "research_only": True,
            "note": ("prefill frontier sweep point; re-run the allocator at "
                     "this SLO for a shippable layer config with full metadata"),
            **provenance_stub}}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    point.update({
        "predicted_dloss": float(record["predicted_dloss"]),
        "achieved_bits": float(record["achieved_bits"]),
        "payload_bytes": int(record["payload_bytes"]),
        "attained_prefill_ms": _attained(record, "operator_sum_prefill_ms"),
        "attained_decode_ms": _attained(record, "operator_sum_decode_ms"),
        "device_memory_bytes": _attained(record, "device_memory_bytes"),
        "assignment_sha256": digest,
        "assignment_path": str(path),
        "serve_constraints": record["serve_constraints"],
    })
    if point["attained_prefill_ms"] is None:
        raise PrefillFrontierError(
            f"SLO {record['slo_ms']}: feasible solve carries no operator_sum_prefill_ms")
    return point


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #

def run_sweep(ctx, *, grid: tuple[str, object], assignments_dir: Path,
              loss_noise_floor: float, allocator_argv: Sequence[str]) -> dict:
    """Drive ``ctx.solve`` over the grid and return the v1 document."""
    fixed_prefill = float(ctx.fixed_resources["prefill_ms"])
    lower_bound = fixed_prefill + ctx.unit_min_prefill_sum_ms
    upper_bound = fixed_prefill + ctx.unit_max_prefill_sum_ms
    if not (upper_bound > 0 and math.isfinite(upper_bound)):
        raise PrefillFrontierError(
            "the table prices every option at zero prefill; there is no SLO axis")
    provenance_stub = {"table_id": ctx.table_identity["table_id"],
                       "table_sha256": ctx.table_identity["sha256"]}

    # The unconstrained point: no assignment needs more than the table's own
    # maximum, so this solve is what "no prefill SLO" would return, and its
    # attained prefill is the saturation SLO.
    top_record = ctx.solve(upper_bound, ctx.target_bits)
    top = _point_record(top_record, assignments_dir, provenance_stub=provenance_stub)
    saturation = None
    if top["feasible"]:
        saturation = {"slo_ms": float(top["attained_prefill_ms"]),
                      "assignment_sha256": top["assignment_sha256"],
                      "predicted_dloss": top["predicted_dloss"],
                      "derived_from": "attained operator-sum prefill of the solve at slo_axis.upper_bound_ms"}

    kind, spec = grid
    if kind == "explicit":
        requested = list(spec)
    elif saturation is None:
        requested = []
    elif kind == "auto":
        requested = list(top["solver"]["prefill_slo_breakpoints_ms"] or [])
    else:
        requested = linear_grid(lower_bound, saturation["slo_ms"], int(spec))
    grid_ms = sorted({float(value) for value in requested}
                     | ({saturation["slo_ms"]} if saturation else set()) | {upper_bound})
    points = []
    for slo_ms in grid_ms:
        if slo_ms == upper_bound:
            points.append(top)
            continue
        if saturation is None:
            # Nothing tighter can be feasible when the loosest budget is not.
            points.append({**_point_record({"slo_ms": slo_ms, "target_bits": ctx.target_bits,
                                            "feasible": False, "reason": top["refusal_reason"]},
                                           assignments_dir, provenance_stub=provenance_stub),
                           "refusal_reason": f"unconstrained_point_infeasible:{top['refusal_reason']}"})
            continue
        record = ctx.solve(slo_ms, ctx.target_bits)
        points.append(_point_record(record, assignments_dir, provenance_stub=provenance_stub))
        point = points[-1]
        print(f"[prefill-frontier] slo={slo_ms:.6g} ms: "
              + (f"dloss={point['predicted_dloss']:.6g} prefill={point['attained_prefill_ms']:.6g} ms "
                 f"bits={point['achieved_bits']:.4f} sha={point['assignment_sha256'][:12]}"
                 if point["feasible"] else f"INFEASIBLE ({point['refusal_reason']})"),
              flush=True)

    from .aura_cost import _git_commit
    cost_path = Path(ctx.cost_path)
    with cost_path.open("rb") as stream:
        cost_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = {
        "git_commit": _git_commit(),
        "table_identity": dict(ctx.table_identity),
        "runtime_context": dict(ctx.runtime_context),
        "fixed_resources": dict(ctx.fixed_resources),
        "cost_path": str(cost_path), "cost_sha256": cost_sha256,
        "probe_path": str(ctx.probe_path),
        "target_bits": float(ctx.target_bits),
        "serve_slos_other_axes": {key: value for key, value in ctx.slos.as_dict().items()
                                  if key != "p95_ttft_ms"},
        "grid": {"kind": kind, "spec": spec if kind != "auto" else "prefill_slo_breakpoints_ms"},
        "allocator_argv": list(allocator_argv),
        "assignments_dir": str(assignments_dir),
        "n_units": int(ctx.n_units),
    }
    slo_axis = {"lower_bound_ms": lower_bound, "upper_bound_ms": upper_bound,
                "fixed_prefill_ms": fixed_prefill,
                "lower_bound_derivation": "fixed prefill + sum over units of the fastest priced option",
                "upper_bound_derivation": "fixed prefill + sum over units of the slowest priced option"}
    return build_frontier_document(points, saturation=saturation, slo_axis=slo_axis,
                                   loss_noise_floor=loss_noise_floor, provenance=provenance)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _split_argv(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    argv = list(argv)
    if "--" not in argv:
        return argv, []
    cut = argv.index("--")
    return argv[:cut], argv[cut + 1:]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m prismaquant.prefill_frontier",
        description=("Sweep the prefill p95-TTFT SLO over a grid, solving each point "
                     "through the allocator's measured-runtime path, and write the "
                     "prismaquant.prefill_frontier.v1 curve. Allocator arguments follow "
                     "'--' and must include --measured-runtime-table and "
                     "--measured-runtime-context; the grid owns --slo-prefill-p95-ttft-ms."),
        epilog=("Example: python -m prismaquant.prefill_frontier --output frontier.json "
                "--slo-grid auto -- --probe probe.pkl --costs joint.pkl --formats F1,F2 "
                "--target-bits 4.75 --measured-runtime-table runtime.json "
                "--measured-runtime-context context.json --layer-config unused.json "
                "--pareto-csv unused.csv"))
    ap.add_argument("--output", required=True, help="Frontier JSON to write.")
    ap.add_argument("--assignments-dir", default=None,
                    help="Where each distinct assignment is written, one file per "
                         "digest (default: <output>.assignments).")
    ap.add_argument("--slo-ms", default=None,
                    help="Explicit comma-separated prefill p95-TTFT SLOs in ms.")
    ap.add_argument("--slo-grid", default=None,
                    help="'auto': the exact SLO breakpoints of the unconstrained solve's "
                         "proposal frontier; or an integer N >= 2: N linearly spaced SLOs "
                         "from the table lower bound to the saturation point.")
    ap.add_argument("--loss-noise-floor", type=float, default=0.0,
                    help="Nondominance noise floor on predicted_dloss, as "
                         "select_validated_frontier's --kl-noise-floor. The predicted "
                         "objective is deterministic, so the default is exactly 0.")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    own, allocator_argv = _split_argv(sys.argv[1:] if argv is None else argv)
    ap = build_parser()
    args = ap.parse_args(own)
    try:
        grid = parse_grid_spec(args.slo_ms, args.slo_grid)
    except PrefillFrontierError as exc:
        ap.error(str(exc))
    if not allocator_argv:
        ap.error("allocator arguments are required after '--'")
    if not math.isfinite(args.loss_noise_floor) or args.loss_noise_floor < 0:
        ap.error("--loss-noise-floor must be finite and nonnegative")
    output = Path(args.output)
    assignments_dir = (Path(args.assignments_dir) if args.assignments_dir
                       else output.with_name(output.name + ".assignments"))

    from . import allocator
    document: dict | None = None

    def sweep(ctx) -> None:
        nonlocal document
        document = run_sweep(ctx, grid=grid, assignments_dir=assignments_dir,
                             loss_noise_floor=args.loss_noise_floor,
                             allocator_argv=allocator_argv)

    # Loader and identity refusals surface as the allocator's own SystemExit;
    # they are not wrapped here.
    allocator.main(list(allocator_argv), measured_runtime_sweep=sweep)
    if document is None:
        raise SystemExit("[prefill-frontier] the allocator returned without running the sweep")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n")
    saturation = document["saturation"]
    print(f"[prefill-frontier] {document['n_feasible']}/{document['n_points']} feasible, "
          f"{document['n_nondominated']} nondominated, "
          f"{document['distinct_assignments']} distinct assignments -> {output}", flush=True)
    if saturation is None:
        print("[prefill-frontier] INFEASIBLE at the unconstrained point: "
              f"{document['points'][-1]['refusal_reason']}", file=sys.stderr, flush=True)
        return 2
    print(f"[prefill-frontier] saturation at slo={saturation['slo_ms']:.6g} ms "
          f"(verified={saturation['verified']}); lower bound "
          f"{document['slo_axis']['lower_bound_ms']:.6g} ms", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
