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
* ``attained_prefill_ms_bootstrap`` (and the decode twin where the table
  prices decode): the dispersion the measurement itself carries. Each priced
  row is resampled with replacement from its OWN samples and re-reduced by the
  same median the row was reduced by, and the sums are re-taken
  (``measured_runtime_prices.bootstrap_sum``, the same function the
  prefill-vs-accuracy curve uses). An attained prefill is a sum of medians; two
  points whose intervals overlap are not resolved by this table, and a reader
  who sees only the point estimates cannot tell. **No threshold is applied and
  no verdict is declared** -- nondominance, saturation and monotonicity are all
  computed from the point estimates exactly as before, and the intervals are
  published beside them.
* ``fixed_resource_scope``: ``null`` when the table's fixed whole-engine charge
  is admitted, and otherwise the scope the run read it under
  (``--measured-runtime-fixed-scope shape-only``), carrying the admission
  gate's refusal verbatim and naming every term this curve did not price. Under
  a scope every point's ``device_memory_bytes`` is ``null``: the fixed device
  terms are withheld, so a device number built from what is left would be a sum
  with a charge missing from it. **Two scopes, never mixed** -- the operator sum
  over the priced units is measured, the fixed whole-model charge is not
  carried, and the document says which is which rather than adding them.

The curve is a *proposal* under the table's sequential operator-sum model.
Operator sums do not certify p95 TTFT (``measured_runtime_prices``); the
served, fixed-teacher gates in ``docs/design/joint_aura_runtime_allocation.md``
still decide what ships. No knee is reported: the measured operating-point
frontier is log-linear with no interior optimum, so any single "best" point
would be a heuristic, and CLAUDE.md demotes kneedle to a diagnostic.

PACT shape-table mode defaults to its existing weighted hull. Explicit
``--pact-selection-mode constrained`` instead uses the existing finite runtime
frontier to minimize declared loss under bytes and a complete comparison
assignment's time, derived from the same admitted table/context/M/TP. It may
select a non-hull assignment. Neither lane qualifies served latency or quality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Sequence

from .cost_stage_checkpoint import atomic_write_bytes, publish_new_bytes
from .digests import bytes_sha256hex, file_digest_sha256hex, file_sha256hex
from .layer_config import LAYER_CONFIG_META_KEY
from .measured_runtime_prices import (
    BOOTSTRAP_CONFIDENCE, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, identity_sha256)

SCHEMA = "prismaquant.prefill_frontier.v1"
ASSIGNMENT_SCHEMA = "prismaquant.prefill_frontier.assignment.v1"
REPLAY_SCHEMA = "prismaquant.prefill_frontier.replay.v1"
#: The PACT hull document (PQ #1584): the exact lower convex hull of
#: (operator-sum time, predicted Δloss), not an SLO grid.
PACT_SCHEMA = "prismaquant.pact_frontier.v1"
CONSTRAINED_GENERATOR = "exact_runtime_frontier_constrained"
CONSTRAINED_NUMERIC_SEMANTICS = "exact integer bytes; sorted-unit binary64 loss/time sums"


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
                            provenance: dict, fixed_resource_scope: dict | None = None) -> dict:
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
        "fixed_resources_admitted": fixed_resource_scope is None,
        "fixed_resource_scope": None if fixed_resource_scope is None else dict(fixed_resource_scope),
        "certifies_placement": False,
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


def point_dispersion(verdict: dict, *, prefill_samples_ms: dict, decode_samples_ms: dict,
                     fixed_prefill_ms: float, fixed_decode_ms: float | None,
                     draws: int, seed: int, cache: dict | None = None) -> dict:
    """Bootstrap intervals for one solve's attained sums, from the rows it summed.

    The verdict says which ``(unit, format)`` rows it summed
    (``coverage.priced_rows``) and this resamples exactly those rows' own
    samples. A priced row with no samples is a refusal, not an omitted
    interval: a point published without its dispersion reads as a point whose
    dispersion is zero.

    The fixed whole-engine term enters as a constant offset so the interval is
    on the same axis as ``attained_prefill_ms``. It carries no width because
    the report schema observes no samples for it.
    """
    from .measured_runtime_prices import bootstrap_sum

    coverage = verdict.get("coverage", {})
    if "priced_rows" not in coverage:
        raise PrefillFrontierError(
            "the serve verdict does not say which rows it summed "
            "(coverage.priced_rows), so the sum's dispersion cannot be drawn "
            "from those rows' own samples")
    keys = [tuple(row) for row in coverage["priced_rows"]]
    predicted = verdict.get("predicted", {})
    out = {}
    for axis, samples_by_key, offset in (
        ("prefill", prefill_samples_ms, fixed_prefill_ms),
        ("decode", decode_samples_ms, fixed_decode_ms),
    ):
        if predicted.get(f"operator_sum_{axis}_ms") is None:
            # The table does not price this axis for this assignment; the
            # attained value is already null and there is nothing to disperse.
            out[axis] = None
            continue
        missing = [key for key in keys if key not in samples_by_key]
        if missing:
            raise PrefillFrontierError(
                f"the {axis} sum priced {len(keys)} rows but the table carries no "
                f"{axis} samples for {missing[:4]}; a sum of medians cannot be "
                "published without the dispersion those medians resolve")
        token = (axis, tuple(keys), float(offset), int(draws), int(seed))
        if cache is not None and token in cache:
            out[axis] = cache[token]
            continue
        interval = bootstrap_sum([samples_by_key[key] for key in keys],
                                 draws=draws, seed=seed, offset_ms=float(offset))
        interval["offset_term"] = (
            f"fixed whole-engine {axis}_ms, added to every draw; it contributes no "
            "width because the report schema carries no samples for it")
        interval["reduction"] = "median, per row, as the priced row itself was reduced"
        if cache is not None:
            cache[token] = interval
        out[axis] = interval
    return out


def _assignment_payload(assignment: dict, digest: str, provenance_stub: dict) -> dict:
    """The exact object this module publishes at ``<digest>.json``."""
    return {**assignment, LAYER_CONFIG_META_KEY: {
        "schema": ASSIGNMENT_SCHEMA, "assignment_sha256": digest,
        "research_only": True,
        "note": ("prefill frontier sweep point; use prefill_frontier replay "
                 "for a research-only layer config with full allocator metadata"),
        **provenance_stub}}


def _verify_reusable_assignment(path: Path, *, assignment: dict, digest: str,
                                provenance_stub: dict) -> dict:
    """Refuse an existing digest-named file that is not this solve's assignment.

    The path IS the assignment's identity, so a file found there is reusable
    only when it is that assignment: the bytes parse, the ``LAYER_CONFIG_META_KEY``
    block states this module's schema and digest and ``research_only`` true, the
    assignment recovered from the file equals this solve's and re-hashes to the
    digest.  Anything else is refused by name rather than reused, and is never
    overwritten -- the file may be another writer's work in flight.

    ``research_only`` is checked because it is the standing the artifact claims,
    and no point this module publishes may be carried by a file that does not
    claim it: a reused file whose block is missing the key or sets it false is
    refused, exactly as a wrong schema is, rather than adopted on the strength of
    the digest alone.

    THE PROVENANCE THE FILE RECORDS IS RETURNED, NOT DEMANDED.  The digest covers
    the assignment alone, so two sweeps over tables with different bytes can
    resolve to one path while their assignments agree -- a table re-measured
    without moving a median does it, and ``tests/test_prefill_frontier_dispersion.py``
    runs it.  The file then records whoever published it first, which is a true
    statement about the file and no statement about this run's solve.  So the
    obligation is that the file SAYS which table published it (a missing or
    non-string ``table_id``/``table_sha256`` is refused, because then nothing can
    tell the two apart), and the caller reports what it found beside the point's
    own numbers instead of adopting it as its own.  What made the old code wrong
    was never the difference in provenance; it was never reading the file at all.
    """
    where = f"assignment {digest}"
    try:
        if path.is_symlink() or not path.is_file():
            raise PrefillFrontierError(
                f"{where}: {path} is not a regular file; refusing to reuse it")
        raw = path.read_bytes()
    except OSError as exc:
        raise PrefillFrontierError(
            f"{where}: {path} is unreadable ({exc}); refusing to reuse it") from exc
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise PrefillFrontierError(
            f"{where}: {path} is not readable JSON ({exc}); refusing to reuse a corrupt "
            "or truncated assignment") from exc
    if not isinstance(payload, dict):
        raise PrefillFrontierError(
            f"{where}: {path} is not a JSON object; refusing to reuse it")
    meta = payload.get(LAYER_CONFIG_META_KEY)
    if not isinstance(meta, dict):
        raise PrefillFrontierError(
            f"{where}: {path} carries no {LAYER_CONFIG_META_KEY} block, so this module "
            "cannot bind it to the current solve; refusing to reuse it")
    problems = []
    if meta.get("schema") != ASSIGNMENT_SCHEMA:
        problems.append(f"schema {meta.get('schema')!r} != {ASSIGNMENT_SCHEMA!r}")
    if meta.get("assignment_sha256") != digest:
        problems.append(
            f"recorded assignment_sha256 {meta.get('assignment_sha256')!r} != {digest}")
    if meta.get("research_only") is not True:
        problems.append(
            f"research_only is {meta.get('research_only')!r}, not True, so the file does "
            "not claim the research-only standing every point this module publishes "
            "carries")
    stored = {key: value for key, value in payload.items() if key != LAYER_CONFIG_META_KEY}
    if stored != assignment:
        problems.append("the stored assignment differs from this solve's assignment")
    elif identity_sha256(stored) != digest:
        problems.append("the stored assignment does not hash to the name it is filed under")
    recorded = {}
    for key, value in sorted(provenance_stub.items()):
        if not isinstance(value, str) or not isinstance(meta.get(key), str):
            problems.append(
                f"provenance {key} is {meta.get(key)!r}, and a file whose published "
                "provenance cannot be read cannot be told from another table's")
        else:
            recorded[key] = meta[key]
    if problems:
        raise PrefillFrontierError(
            f"{where}: {path} exists but is not this solve's assignment: "
            + "; ".join(problems)
            + ". Refusing to reuse it and refusing to overwrite it; move it aside or "
            "point this sweep at another --assignments-dir.")
    return recorded


def _publish_assignment(assignments_dir: Path, *, assignment: dict, digest: str,
                        provenance_stub: dict) -> tuple[Path, dict]:
    """Publish this solve's assignment, or prove the file already there is it.

    ``publish_new_bytes`` is a hard-link creation, so the file is either absent
    or complete; when it reports that one was already there, the loser of that
    race reads what won and validates it rather than replacing it.  When this
    call created the file, the bytes behind the returned path are the payload it
    just built for exactly this digest; when somebody else did, the bytes are
    read back and must pass :func:`_verify_reusable_assignment` before the path
    is returned.  No path reaches the caller without bytes this call either
    wrote or read.

    Returns the path and the provenance the FILE records: this run's when this
    run published it, and the first publisher's when it was already there.
    """
    path = assignments_dir / f"{digest}.json"
    payload = _assignment_payload(assignment, digest, provenance_stub)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if publish_new_bytes(path, encoded):
        return path, {key: str(value) for key, value in provenance_stub.items()}
    return path, _verify_reusable_assignment(path, assignment=assignment, digest=digest,
                                             provenance_stub=provenance_stub)


def _point_record(record: dict, assignments_dir: Path, *, provenance_stub: dict,
                  fixed_resource_scope: dict | None = None, dispersion=None) -> dict:
    """One grid point from the allocator's solve record; writes its assignment.

    Under a fixed-resource scope the point must carry no device number. The
    withholding happens where the number would be built
    (``serve_constraints.evaluate_measured_assignment``); this refuses rather
    than trusts it, because the whole content of the scope is that no device
    term reaches a document.

    The assignment file is published through
    :func:`_publish_assignment`, so the ``assignment_sha256``/``assignment_path``
    pair this returns is one whose bytes this solve wrote (a fresh publication,
    whose payload is built from that digest's own assignment) or read back and
    bound to it (a reuse, which is refused unless the file IS the assignment):
    the name is the digest, and a name is not evidence.
    """
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
        "attained_prefill_ms_bootstrap": None,
        "attained_decode_ms": None,
        "attained_decode_ms_bootstrap": None,
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
    path, published_by = _publish_assignment(assignments_dir, assignment=assignment,
                                             digest=digest,
                                             provenance_stub=provenance_stub)
    point.update({
        "predicted_dloss": float(record["predicted_dloss"]),
        "achieved_bits": float(record["achieved_bits"]),
        "payload_bytes": int(record["payload_bytes"]),
        "attained_prefill_ms": _attained(record, "operator_sum_prefill_ms"),
        "attained_decode_ms": _attained(record, "operator_sum_decode_ms"),
        "device_memory_bytes": _attained(record, "device_memory_bytes"),
        "assignment_sha256": digest,
        "assignment_path": str(path),
        # The provenance the FILE records, which is this run's unless an earlier
        # sweep published this (identical) assignment first.  A reader can see
        # the difference here instead of inferring it from the file.
        "assignment_file_provenance": published_by,
        "serve_constraints": record["serve_constraints"],
    })
    if point["attained_prefill_ms"] is None:
        raise PrefillFrontierError(
            f"SLO {record['slo_ms']}: feasible solve carries no operator_sum_prefill_ms")
    if dispersion is not None:
        intervals = dispersion(record["serve_constraints"])
        point["attained_prefill_ms_bootstrap"] = intervals["prefill"]
        point["attained_decode_ms_bootstrap"] = intervals["decode"]
        if point["attained_prefill_ms_bootstrap"] is None:
            raise PrefillFrontierError(
                f"SLO {record['slo_ms']}: feasible solve carries an attained prefill "
                "with no interval over the samples it was reduced from")
    if fixed_resource_scope is not None and point["device_memory_bytes"] is not None:
        raise PrefillFrontierError(
            f"SLO {record['slo_ms']}: the {fixed_resource_scope['scope']} fixed-resource scope "
            f"withholds {', '.join(fixed_resource_scope['withheld_terms'])}, so this point may "
            f"carry no device_memory_bytes; it carries {point['device_memory_bytes']}")
    return point


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #

def _select(points, *, regime_m: int, table_identity, frontier_scope: str,
            draws: int, seed: int) -> dict:
    """The one PACT selection call, for the hull and the sweep alike (PQ #1659).

    Both paths hand ``select_pact`` points whose time intervals came from
    :func:`~prismaquant.measured_runtime_prices.bootstrap_sum` and say how they
    were drawn, so a record names its own level, resample count and seed.
    """
    from .pact_selection import select_pact

    return select_pact(
        points, regime_m=int(regime_m), table_identity=dict(table_identity),
        frontier_scope=frontier_scope,
        time_interval={"method": "per_shape_bootstrap_sum_v1",
                       "confidence": BOOTSTRAP_CONFIDENCE,
                       "draws": int(draws), "seed": int(seed)})


def sweep_selection_points(points) -> list[dict]:
    """The distinct feasible assignments of a sweep, as ``select_pact`` points.

    One assignment recurs at every SLO above its attained time, so the sweep's
    points are deduplicated by digest; the first (tightest-SLO) point stands
    for it. Every point must carry its bootstrap interval: a sweep built
    without dispersion has no interval to hand on.
    """
    seen: dict[str, dict] = {}
    for point in points:
        if point.get("feasible") is not True or point["assignment_sha256"] in seen:
            continue
        boot = point.get("attained_prefill_ms_bootstrap")
        if boot is None:
            raise PrefillFrontierError(
                f"sweep point {point['assignment_sha256']} carries no prefill bootstrap interval")
        seen[point["assignment_sha256"]] = {
            "point_id": point["assignment_sha256"][:16],
            "assignment_sha256": point["assignment_sha256"],
            "bytes": int(point["payload_bytes"]),
            "time_ms": float(point["attained_prefill_ms"]),
            "time_interval_ms": [float(boot["p2.5"]), float(boot["p97.5"])],
            "predicted_dloss": float(point["predicted_dloss"]),
            "kernel_lanes": None,
            "time_samples": {"samples_per_row": list(boot["samples_per_row"]),
                             "distinct_measurements": boot.get("distinct_measurements")},
        }
    return list(seen.values())


def run_sweep(ctx, *, grid: tuple[str, object], assignments_dir: Path,
              loss_noise_floor: float, allocator_argv: Sequence[str],
              bootstrap_draws: int = BOOTSTRAP_DRAWS,
              bootstrap_seed: int = BOOTSTRAP_SEED) -> dict:
    """Drive ``ctx.solve`` over the grid and return the v1 document."""
    scope = ctx.fixed_resource_scope
    fixed_prefill = float(ctx.fixed_resources["prefill_ms"])
    fixed_decode = ctx.fixed_resources["decode_ms"]
    # One assignment recurs across many SLOs (that is what saturation means),
    # so the draws are memoized on the exact row list they resample. Same
    # rows, same seed, same interval -- the cache changes no number.
    interval_cache: dict = {}

    def dispersion(verdict: dict) -> dict:
        return point_dispersion(
            verdict, prefill_samples_ms=ctx.prefill_samples_ms,
            decode_samples_ms=ctx.decode_samples_ms,
            fixed_prefill_ms=fixed_prefill,
            fixed_decode_ms=None if fixed_decode is None else float(fixed_decode),
            draws=bootstrap_draws, seed=bootstrap_seed, cache=interval_cache)
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
    top = _point_record(top_record, assignments_dir, provenance_stub=provenance_stub,
                    fixed_resource_scope=scope, dispersion=dispersion)
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
                                           assignments_dir, provenance_stub=provenance_stub,
                                           fixed_resource_scope=scope, dispersion=dispersion),
                           "refusal_reason": f"unconstrained_point_infeasible:{top['refusal_reason']}"})
            continue
        record = ctx.solve(slo_ms, ctx.target_bits)
        points.append(_point_record(record, assignments_dir, provenance_stub=provenance_stub,
                                    fixed_resource_scope=scope, dispersion=dispersion))
        point = points[-1]
        print(f"[prefill-frontier] slo={slo_ms:.6g} ms: "
              + (f"dloss={point['predicted_dloss']:.6g} prefill={point['attained_prefill_ms']:.6g} ms "
                 f"[{point['attained_prefill_ms_bootstrap']['p2.5']:.6g}, "
                 f"{point['attained_prefill_ms_bootstrap']['p97.5']:.6g}] "
                 f"bits={point['achieved_bits']:.4f} sha={point['assignment_sha256'][:12]}"
                 if point["feasible"] else f"INFEASIBLE ({point['refusal_reason']})"),
              flush=True)

    from .aura_cost import _git_commit
    cost_path = Path(ctx.cost_path)
    with cost_path.open("rb") as stream:
        cost_sha256 = file_digest_sha256hex(stream)
    provenance = {
        "git_commit": _git_commit(),
        "table_identity": dict(ctx.table_identity),
        "runtime_context": dict(ctx.runtime_context),
        "fixed_resources": dict(ctx.fixed_resources),
        "cost_path": str(cost_path), "cost_sha256": cost_sha256,
        "probe_path": str(ctx.probe_path),
        "probe_sha256": file_sha256hex(Path(ctx.probe_path)),
        "allocator_cwd": str(Path.cwd()),
        "target_bits": float(ctx.target_bits),
        "serve_slos_other_axes": {key: value for key, value in ctx.slos.as_dict().items()
                                  if key != "p95_ttft_ms"},
        "grid": {"kind": kind, "spec": spec if kind != "auto" else "prefill_slo_breakpoints_ms"},
        "bootstrap": {"draws": int(bootstrap_draws), "seed": int(bootstrap_seed),
                      "confidence": BOOTSTRAP_CONFIDENCE,
                      "function": "prismaquant.measured_runtime_prices.bootstrap_sum",
                      "resamples": "each priced row's own samples, with replacement, re-reduced by median",
                      "covers": ("the priced rows this solve summed; the fixed whole-engine term is a "
                                 "constant offset with no observed samples"),
                      "applied_as_a_threshold": False},
        "allocator_argv": list(allocator_argv),
        "assignments_dir": str(assignments_dir),
        "n_units": int(ctx.n_units),
    }
    slo_axis = {"lower_bound_ms": lower_bound, "upper_bound_ms": upper_bound,
                "fixed_prefill_ms": fixed_prefill,
                "fixed_prefill_ms_scope": ("admitted" if scope is None else
                                           f"{scope['scope']}: read, and refused if nonzero, "
                                           "because the report schema observes no timing term"),
                "lower_bound_derivation": "fixed prefill + sum over units of the fastest priced option",
                "upper_bound_derivation": "fixed prefill + sum over units of the slowest priced option"}
    document = build_frontier_document(points, saturation=saturation, slo_axis=slo_axis,
                                       loss_noise_floor=loss_noise_floor, provenance=provenance,
                                       fixed_resource_scope=scope)
    selection_points = sweep_selection_points(points)
    if selection_points:
        regime_m = int(ctx.runtime_context["prompt_tokens"])
        document["regime_m"] = regime_m
        document["pact_selection"] = _select(
            selection_points, regime_m=regime_m, table_identity=ctx.table_identity,
            frontier_scope="slo_grid_distinct_assignments",
            draws=bootstrap_draws, seed=bootstrap_seed)
    return document


# --------------------------------------------------------------------------- #
# PACT hull (PQ #1584)
# --------------------------------------------------------------------------- #

def is_pact_argv(allocator_argv: Sequence[str]) -> bool:
    """Does this allocator argv ask for the PACT shape-table hull?"""
    return any(arg == "--pact-shape-table" or arg.startswith("--pact-shape-table=")
               for arg in allocator_argv)


def _pact_provenance(ctx, allocator_argv, assignments_dir, draws, seed) -> dict:
    """The shared input/context identity of either PACT solver and its replay."""
    from .aura_cost import _git_commit

    return {
        "git_commit": _git_commit(), "table_identity": dict(ctx.table_identity),
        "scope": dict(ctx.scope), "regime_m": int(ctx.regime_m),
        "tensor_parallel": int(ctx.tensor_parallel),
        "cost_path": str(ctx.cost_path), "cost_sha256": file_sha256hex(Path(ctx.cost_path)),
        "probe_path": str(ctx.probe_path), "probe_sha256": file_sha256hex(Path(ctx.probe_path)),
        "allocator_cwd": str(Path.cwd()), "target_bits": float(ctx.target_bits),
        "bootstrap": {"draws": int(draws), "seed": int(seed),
                      "confidence": BOOTSTRAP_CONFIDENCE,
                      "function": "prismaquant.measured_runtime_prices.bootstrap_sum",
                      "resamples": ("each distinct shape row's own samples, once per draw, "
                                    "weighted by how many units read it"),
                      "applied_as_a_threshold": ("only through pact_selection: the "
                                                 "endpoint interval overlap is the "
                                                 "materiality verdict and the top-two "
                                                 "half-widths set min_separation")},
        "allocator_argv": list(allocator_argv), "assignments_dir": str(assignments_dir),
        "n_units": int(ctx.n_units),
    }


def run_hull(ctx, *, assignments_dir: Path, allocator_argv: Sequence[str],
             bootstrap_draws: int = BOOTSTRAP_DRAWS,
             bootstrap_seed: int = BOOTSTRAP_SEED) -> dict:
    """Build the exact PACT hull through ``ctx`` and return its document.

    Every vertex carries its operator-sum time, the dispersion of that sum
    over the rows' own samples (``ShapePricing.operator_sum_bootstrap``: a row
    many units read is drawn once per draw), the probe that found it (its
    weights re-derive it at replay), and the exact checks the allocator ran on
    it. Nothing is selected here: every PACT selection rule maximises an affine
    function of (time, Δloss), so it picks among these vertices.
    """
    from .pact_hull import CANDIDATE_GENERATOR
    from .shape_runtime_prices import TIME_CLAIM

    built = ctx.build_hull()
    hull = built["hull"]
    stub = {"table_id": ctx.table_identity["table_id"],
            "table_sha256": ctx.table_identity["sha256"]}
    vertices = []
    selection_points = []
    for i, (vertex, record) in enumerate(zip(hull.vertices, built["vertices"])):
        probe = hull.finding_probe(vertex)
        entry = {
            "vertex": i, "feasible": bool(record["feasible"]),
            "refusal_reason": record.get("reason"),
            "predicted_dloss": vertex.predicted_dloss,
            "operator_sum_ms": vertex.time_ms,
            "operator_sum_ms_bootstrap": ctx.pricing.operator_sum_bootstrap(
                vertex.assignment, draws=bootstrap_draws, seed=bootstrap_seed),
            "within_time_ceiling": (None if ctx.time_ceiling_ms is None
                                    else vertex.time_ms <= ctx.time_ceiling_ms),
            "candidate_bytes": int(vertex.memory_bytes),
            "lambda_to_next_dloss_per_ms": (hull.edge_lambdas[i] if i < len(hull.edge_lambdas)
                                            else None),
            "finding_probe": {"index": probe.index, "weights": list(probe.weights)},
            "achieved_bits": record.get("achieved_bits"),
            "payload_bytes": record.get("payload_bytes"),
            "whole_artifact_upper_bound_bytes": record.get("whole_artifact_upper_bound_bytes"),
            "assignment_sha256": None, "assignment_path": None,
        }
        if record["feasible"]:
            assignment = dict(record["assignment"])
            digest = identity_sha256(assignment)
            path, published_by = _publish_assignment(
                assignments_dir, assignment=assignment, digest=digest, provenance_stub=stub)
            entry.update({"assignment_sha256": digest, "assignment_path": str(path),
                          "assignment_file_provenance": published_by})
        if record["feasible"]:
            selection_points.append({
                "point_id": str(i),
                "assignment_sha256": entry["assignment_sha256"],
                "bytes": int(entry["payload_bytes"] if entry["payload_bytes"] is not None
                             else entry["candidate_bytes"]),
                "time_ms": float(vertex.time_ms),
                "time_interval_ms": [float(entry["operator_sum_ms_bootstrap"]["p2.5"]),
                                     float(entry["operator_sum_ms_bootstrap"]["p97.5"])],
                "predicted_dloss": float(vertex.predicted_dloss),
                "kernel_lanes": ctx.pricing.kernel_lane_histogram(assignment),
                "time_samples": {
                    "samples_per_row": list(entry["operator_sum_ms_bootstrap"]["samples_per_row"]),
                    "distinct_measurements":
                        entry["operator_sum_ms_bootstrap"].get("distinct_measurements")},
            })
        vertices.append(entry)
        boot = entry["operator_sum_ms_bootstrap"]
        print(f"[pact-hull] vertex {i}: dloss={vertex.predicted_dloss:.6g} "
              f"ops={vertex.time_ms:.6g} ms [{boot['p2.5']:.6g}, {boot['p97.5']:.6g}] "
              + (f"sha={entry['assignment_sha256'][:12]}" if entry["feasible"]
                 else f"REFUSED ({entry['refusal_reason']})"), flush=True)
    selection = _select(
        selection_points, regime_m=int(ctx.regime_m), table_identity=ctx.table_identity,
        frontier_scope="lower_convex_hull_vertices",
        draws=bootstrap_draws, seed=bootstrap_seed)
    print(f"[pact-hull] selection: materiality={selection['materiality']['verdict']} "
          + " ".join(f"{role}={pick['point_id']}" for role, pick in selection["picks"].items()),
          flush=True)
    return {
        "schema": PACT_SCHEMA,
        "pact_selection": selection,
        "candidate_generator": CANDIDATE_GENERATOR,
        "time_claim": TIME_CLAIM,
        "research_only": True, "certifies_placement": False, "certifies_p95": False,
        "selection_scope": (
            "every PACT selection rule maximises an affine function of (time, Δloss) -- "
            "argmin Δloss, argmin time, and select_development_point's chord distance in "
            "endpoint-normalised coordinates -- so its pick is a vertex of this hull; λ "
            "generates the hull and never enters a selection objective"),
        "separation_scope": ("the materiality test and the derived min_separation read the "
                             "hull vertices' own bootstrap intervals; a separation test "
                             "between the top two candidates sees hull vertices, not every "
                             "point of the exact frontier, and pact_selection.frontier_scope "
                             "says so on the record (PQ #1585)"),
        "regime_m": int(ctx.regime_m), "tensor_parallel": int(ctx.tensor_parallel),
        "time_ceiling_ms": ctx.time_ceiling_ms,
        "time_ceiling_role": ("report bound: vertices above it are flagged, never removed; "
                              "the constrained set's own boundary vertex is not generated"),
        "fixed_prefill_ms": 0.0,
        "remainder": {
            "fixed_members": len(ctx.fixed_members),
            "fixed_members_sample": list(ctx.fixed_members[:16]),
            "reading": ("the time axis is the operator sum over the DP's serving units at "
                        "regime M; members with a fixed format and every operator outside "
                        "the table are neither priced nor added to it"),
        },
        "max_memory_bytes": int(ctx.max_memory_bytes),
        "max_states": int(ctx.max_states),
        "max_transitions": int(ctx.max_transitions),
        "whole_artifact_budget": (None if ctx.whole_artifact_budget is None
                                  else dict(ctx.whole_artifact_budget)),
        "table_identity": dict(ctx.table_identity),
        "scope": dict(ctx.scope),
        "gap_report": ctx.pricing.gap_report(),
        "hull_seconds": float(built["seconds"]),
        "hull_peak_rss_kib": dict(built.get("peak_rss_kib") or {}),
        "n_vertices": len(vertices),
        "probe_count": len(hull.probes),
        "vertices": vertices,
        "hull": hull.as_dict(),
        "provenance": _pact_provenance(ctx, allocator_argv, assignments_dir,
                                       bootstrap_draws, bootstrap_seed),
    }


def run_constrained(ctx, *, assignments_dir: Path, allocator_argv: Sequence[str],
                    bootstrap_draws: int, bootstrap_seed: int) -> dict:
    """Publish the minimum-loss feasible choice over admitted time-priced options."""
    from .shape_runtime_prices import TIME_CLAIM

    built = ctx.build_constrained()
    solution, record = built["solution"], built["record"]
    assignment = dict(record["assignment"])
    digest = identity_sha256(assignment)
    path, published = _publish_assignment(
        assignments_dir, assignment=assignment, digest=digest,
        provenance_stub={"table_id": ctx.table_identity["table_id"],
                         "table_sha256": ctx.table_identity["sha256"]})
    point = {"point": 0, "feasible": True, "assignment_sha256": digest,
             "assignment_path": str(path), "assignment_file_provenance": published,
             "predicted_dloss": solution.predicted_dloss,
             "operator_sum_ms": solution.prefill_ms,
             "operator_sum_ms_bootstrap": ctx.pricing.operator_sum_bootstrap(
                 solution.assignment, draws=bootstrap_draws, seed=bootstrap_seed),
             "within_time_ceiling": (None if ctx.time_ceiling_ms is None
                                     else solution.prefill_ms <= ctx.time_ceiling_ms),
             "candidate_bytes": solution.memory_bytes,
             **{key: record[key] for key in ("achieved_bits", "payload_bytes",
                                             "whole_artifact_upper_bound_bytes") if key in record}}
    provenance = _pact_provenance(ctx, allocator_argv, assignments_dir,
                                  bootstrap_draws, bootstrap_seed)
    provenance["bootstrap"]["applied_as_a_threshold"] = "none; intervals describe the proposal"
    constraints = {"max_memory_bytes": int(ctx.max_memory_bytes),
                   "max_prefill_ms": built["baseline"]["derived_operator_sum_ms"],
                   "max_states": int(ctx.max_states), "max_transitions": int(ctx.max_transitions)}
    return {
        "schema": PACT_SCHEMA, "candidate_generator": CONSTRAINED_GENERATOR,
        "selection_mode": "constrained", "selected_assignment_sha256": digest,
        "time_claim": TIME_CLAIM, "research_only": True,
        "certifies_placement": False, "certifies_p95": False,
        "selection_scope": ("minimum-loss discrete assignment over the current admitted "
                            "time-priced candidates under declared bytes/time"),
        "numeric_semantics": CONSTRAINED_NUMERIC_SEMANTICS,
        "baseline": built["baseline"], "constraints": constraints,
        "regime_m": int(ctx.regime_m), "tensor_parallel": int(ctx.tensor_parallel),
        "time_ceiling_ms": ctx.time_ceiling_ms, "time_ceiling_role": "report bound only",
        "max_memory_bytes": int(ctx.max_memory_bytes),
        "max_states": int(ctx.max_states), "max_transitions": int(ctx.max_transitions),
        "whole_artifact_budget": ctx.whole_artifact_budget,
        "table_identity": dict(ctx.table_identity), "scope": dict(ctx.scope),
        "remainder": {"fixed_members": len(ctx.fixed_members),
                      "fixed_members_sample": list(ctx.fixed_members[:16]),
                      "reading": "only current independent DP serving units are time-priced"},
        "gap_report": ctx.pricing.gap_report(), "search_diagnostics": built["diagnostics"],
        "search_seconds": built["seconds"], "search_peak_rss_kib": built["peak_rss_kib"],
        "points": [point], "provenance": provenance,
    }


def _as_json(value):
    """``value`` as the document stores it, so a tuple compares equal to its list."""
    return json.loads(json.dumps(value))


def load_pact_baseline(path: Path, expected_sha256: str, *, table_identity: dict,
                       scope: dict, regime_m: int, tensor_parallel: int,
                       serving_scope: dict) -> tuple[dict, dict]:
    """Consume one bound existing assignment using the layer-config reader core.

    A plain shorthand is a comparison assignment under this run's explicit
    context, never a served/native timing claim. Any context declared by a
    richer existing config must agree. Time is derived later from the same
    admitted resources as the candidates, not from this file's numbers.
    """
    from .layer_config import canonicalize_assignment, layer_config_metadata
    from .runtime_provenance import _equal
    from .schemas import validate_layer_config_payload

    raw = path.read_bytes()
    actual_sha256 = bytes_sha256hex(raw)
    if actual_sha256 != expected_sha256:
        raise PrefillFrontierError("PACT baseline file SHA-256 mismatch")
    payload = json.loads(raw)
    validate_layer_config_payload(payload, str(path))
    assignment = canonicalize_assignment(payload)
    names = [str(name).removesuffix(".weight") for name in payload if name != LAYER_CONFIG_META_KEY]
    if len(names) != len(set(names)):
        raise PrefillFrontierError("PACT baseline has duplicate canonical assignment names")
    meta = layer_config_metadata(payload)
    declared_serving = meta.get("tessera_serving_scope")
    if "tessera_serving_scope" in meta:
        _equal(declared_serving, _as_json(serving_scope), "PACT baseline declared serving scope")
    replay = meta.get("prefill_frontier_replay")
    if "prefill_frontier_replay" in meta:
        if not isinstance(replay, dict):
            raise PrefillFrontierError("PACT baseline replay must be an object")
        for key, value in (("table_identity", table_identity), ("scope", scope), ("regime_m", regime_m),
                           ("tensor_parallel", tensor_parallel)):
            if key in replay:
                _equal(replay[key], _as_json(value), f"PACT baseline declared replay {key}")
    if meta.get("schema") == ASSIGNMENT_SCHEMA:
        for key, value in (("table_id", table_identity["table_id"]),
                           ("table_sha256", table_identity["sha256"])):
            if meta.get(key) != value:
                raise PrefillFrontierError(f"PACT baseline declared {key} differs")
        if meta.get("assignment_sha256") != identity_sha256(assignment):
            raise PrefillFrontierError("PACT baseline declared assignment digest differs")
    return assignment, {
        "assignment_path": str(path), "file_sha256": actual_sha256,
        "assignment_sha256": identity_sha256(assignment),
        "table_identity": _as_json(table_identity), "scope": _as_json(scope),
        "regime_m": int(regime_m), "tensor_parallel": int(tensor_parallel),
        "standing": "comparison_assignment_input; no served or native baseline timing claim",
    }


def replay_hull(document: dict, raw: bytes, digest: str, output: Path) -> None:
    """Re-run a recorded hull probe or constrained solve through the allocator.

    The hull is not rebuilt: the vertex's recorded probe weights are handed back
    to the same exact solver over the same priced inputs, and the answer must be
    the published assignment. Cost and probe bytes, the admitted table, the
    scope, the regime, the world size and the target are all bound to the hull
    document before anything is written. The layer config carries
    ``research_only``/``certifies_placement`` and a replay.v1 block naming the
    candidate generator and the time claim.
    """
    if document.get("research_only") is not True or document.get("certifies_placement") is not False:
        raise PrefillFrontierError("replay requires a research-only, non-placement-certified PACT hull")
    from .pact_hull import CANDIDATE_GENERATOR
    from .allocator_solver import _runtime_float, _runtime_int
    from .runtime_provenance import _equal
    generator = document.get("candidate_generator")
    if generator not in (CANDIDATE_GENERATOR, CONSTRAINED_GENERATOR):
        raise PrefillFrontierError("replay refuses an unknown PACT candidate generator")
    constrained = generator == CONSTRAINED_GENERATOR
    provenance = document["provenance"]
    if constrained:
        if (document.get("selection_mode") != "constrained"
                or document.get("numeric_semantics") != CONSTRAINED_NUMERIC_SEMANTICS
                or document.get("certifies_p95") is not False
                or document.get("time_claim") != "operator_sum_proposal"
                or document.get("pact_selection") is not None):
            raise PrefillFrontierError("constrained replay requires its closed proposal semantics")
        points = document.get("points")
        if (not isinstance(points, list) or len(points) != 1
                or not isinstance(points[0], dict)):
            raise PrefillFrontierError("constrained replay requires exactly its selected point")
        if _runtime_int(points[0].get("point"), "PACT replay point") != 0:
            raise PrefillFrontierError("constrained replay requires exactly its selected point")
    if constrained and digest != document.get("selected_assignment_sha256"):
        raise PrefillFrontierError("constrained replay requires its selected assignment")
    found = [v for v in document["points" if constrained else "vertices"]
             if v.get("assignment_sha256") == digest and v.get("feasible") is True]
    if not found:
        raise PrefillFrontierError(f"no feasible hull vertex for assignment {digest}")
    vertex = found[0]
    assignment_path = Path(vertex["assignment_path"])
    payload = json.loads(assignment_path.read_bytes())
    expected = {k: v for k, v in payload.items() if k != LAYER_CONFIG_META_KEY}
    _verify_reusable_assignment(
        assignment_path, assignment=expected, digest=digest,
        provenance_stub={"table_id": provenance["table_identity"]["table_id"],
                         "table_sha256": provenance["table_identity"]["sha256"]})
    for key in ("cost", "probe"):
        if file_sha256hex(Path(provenance[f"{key}_path"])) != provenance[f"{key}_sha256"]:
            raise PrefillFrontierError(f"replay {key}_sha256 mismatch")
    allocator_argv = provenance["allocator_argv"]
    if not isinstance(allocator_argv, list) or not all(isinstance(a, str) for a in allocator_argv):
        raise PrefillFrontierError("replay requires recorded allocator_argv")
    if output.exists() or output.is_symlink():
        raise PrefillFrontierError(f"replay refuses to overwrite {output}")
    from . import allocator

    emitted = False

    def emit(ctx):
        nonlocal emitted
        if not isinstance(ctx, allocator.PactHullSweep):
            raise PrefillFrontierError("a PACT hull replay reached a non-PACT allocator run")
        if (ctx.selection_mode == "constrained") != constrained:
            raise PrefillFrontierError("replay selection mode differs from the recorded generator")
        _runtime_float(provenance["target_bits"], "PACT replay target_bits")
        for key, actual in (("table_identity", ctx.table_identity), ("scope", ctx.scope),
                            ("regime_m", ctx.regime_m), ("tensor_parallel", ctx.tensor_parallel),
                            ("target_bits", ctx.target_bits), ("cost_path", ctx.cost_path),
                            ("probe_path", ctx.probe_path)):
            _equal(provenance[key], _as_json(actual), f"replay {key} differs from the hull provenance")
        if constrained:
            if document.get("time_ceiling_ms") is not None:
                _runtime_float(document["time_ceiling_ms"], "PACT replay time_ceiling_ms")
            for key, actual in (("table_identity", ctx.table_identity), ("scope", ctx.scope),
                                ("regime_m", ctx.regime_m), ("tensor_parallel", ctx.tensor_parallel),
                                ("time_ceiling_ms", ctx.time_ceiling_ms),
                                ("max_memory_bytes", ctx.max_memory_bytes),
                                ("max_states", ctx.max_states), ("max_transitions", ctx.max_transitions),
                                ("whole_artifact_budget", ctx.whole_artifact_budget)):
                _equal(document.get(key), _as_json(actual), f"constrained replay {key} differs")
        stamp = {
            "schema": REPLAY_SCHEMA,
            "frontier_sha256": bytes_sha256hex(raw),
            "assignment_sha256": digest,
            "candidate_generator": document["candidate_generator"],
            "time_claim": document["time_claim"],
            "regime_m": int(ctx.regime_m), "tensor_parallel": int(ctx.tensor_parallel),
            "operator_sum_ms": vertex["operator_sum_ms"],
            "target_bits": float(ctx.target_bits),
            "table_identity": _as_json(ctx.table_identity),
            "cost_sha256": file_sha256hex(Path(ctx.cost_path)),
            "probe_sha256": file_sha256hex(Path(ctx.probe_path)),
            "probe_bound_by_sweep": True,
            "point_claims": {key: vertex[key] for key in (
                "predicted_dloss", "operator_sum_ms", "candidate_bytes", "achieved_bits",
                "payload_bytes", "whole_artifact_upper_bound_bytes") if vertex.get(key) is not None},
        }
        selection = document.get("pact_selection")
        if selection is not None:
            stamp["pact_selection_sha256"] = selection["identity_sha256"]
            stamp["pact_selection"] = selection
        if constrained:
            selection_input = {"baseline": document["baseline"],
                               "constraints": document["constraints"]}
            stamp.update(selection_input, point=vertex["point"],
                         numeric_semantics=document["numeric_semantics"])
        else:
            selection_input = vertex["finding_probe"]["weights"]
            stamp.update(vertex=_runtime_int(vertex["vertex"], "PACT replay vertex"),
                         probe_weights=list(selection_input))
        ctx.emit_replay(selection_input, expected, stamp)
        emitted = True

    allocator.main([*allocator_argv, "--layer-config", str(output)], measured_runtime_sweep=emit)
    if not emitted:
        raise PrefillFrontierError("allocator returned without emitting the hull replay")


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
                          "--measured-runtime-context; the grid owns --slo-prefill-p95-ttft-ms. "
                     "A table whose fixed whole-engine charge no gate admits needs "
                     "--measured-runtime-fixed-scope shape-only among those arguments; the "
                     "curve then prices no fixed charge and no device budget, and says so. "
                     "With --pact-shape-table among the allocator arguments it instead writes "
                     "a prismaquant.pact_frontier.v1 document with no grid: the exact "
                     "PACT hull by default, or one constrained selection with explicit "
                     "--pact-selection-mode constrained and a bound baseline assignment."),
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
    ap.add_argument("--bootstrap-draws", type=int, default=BOOTSTRAP_DRAWS,
                    help="Bootstrap draws for each point's attained-prefill interval, "
                         "resampling each priced row's own samples. Default matches "
                         "experiments/pq_prefill_accuracy_curve.py.")
    ap.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED,
                    help="Seed for those draws, so a curve is reproducible.")
    ap.add_argument("--loss-noise-floor", type=float, default=0.0,
                    help="Nondominance noise floor on predicted_dloss, as "
                         "select_validated_frontier's --kl-noise-floor. The predicted "
                         "objective is deterministic, so the default is exactly 0.")
    return ap


def replay(frontier: Path, digest: str, output: Path) -> None:
    """Re-solve a digest-bound point and use the allocator's only config writer.

    Recorded argv is replayed without input overrides. Paths must resolve to
    the bound inputs, but cwd is audit-only: PB may materialize a new checkout
    for each action. Old v1 sweeps have no probe digest; they still bind cost,
    table, context, scope and the exact re-solved assignment. New sweeps also
    bind probe bytes. No placement/latency certification follows from replay.
    """
    raw = frontier.read_bytes()
    document = json.loads(raw)
    if document.get("schema") == PACT_SCHEMA:
        return replay_hull(document, raw, digest, output)
    if document.get("schema") != SCHEMA or document.get("certifies_placement") is not False:
        raise PrefillFrontierError("replay requires a non-placement-certified prefill frontier")
    provenance = document["provenance"]
    points = [p for p in document["points"]
              if p.get("assignment_sha256") == digest and p.get("feasible") is True]
    if not points:
        raise PrefillFrontierError(f"no feasible sweep point for assignment {digest}")
    # A digest can occur at many SLOs. The tightest recorded one is deterministic.
    point = min(points, key=lambda p: float(p["slo_ms"]))
    assignment_path = Path(point["assignment_path"])
    payload = json.loads(assignment_path.read_bytes())
    expected = {k: v for k, v in payload.items() if k != LAYER_CONFIG_META_KEY}
    _verify_reusable_assignment(
        assignment_path, assignment=expected, digest=digest,
        provenance_stub={k: provenance["table_identity"][v]
                         for k, v in (("table_id", "table_id"), ("table_sha256", "sha256"))})
    for key in ("cost", "probe"):
        checksum = provenance.get(f"{key}_sha256")
        if key == "cost" and not checksum:
            raise PrefillFrontierError("replay requires cost_sha256")
        if checksum and file_sha256hex(Path(provenance[f"{key}_path"])) != checksum:
            raise PrefillFrontierError(f"replay {key}_sha256 mismatch")
    allocator_argv = provenance["allocator_argv"]
    if not isinstance(allocator_argv, list) or not all(isinstance(a, str) for a in allocator_argv):
        raise PrefillFrontierError("replay requires recorded allocator_argv")
    if output.exists() or output.is_symlink():
        raise PrefillFrontierError(f"replay refuses to overwrite {output}")
    from . import allocator

    emitted = False

    def emit(ctx):
        nonlocal emitted
        for key, actual in (("table_identity", ctx.table_identity),
                            ("runtime_context", ctx.runtime_context),
                            ("fixed_resources", ctx.fixed_resources),
                            ("target_bits", ctx.target_bits),
                            ("cost_path", ctx.cost_path), ("probe_path", ctx.probe_path)):
            if provenance[key] != actual:
                raise PrefillFrontierError(f"replay {key} differs from sweep provenance")
        if ctx.fixed_resource_scope != document["fixed_resource_scope"]:
            raise PrefillFrontierError("replay fixed_resource_scope differs from sweep")
        axes = {k: v for k, v in ctx.slos.as_dict().items() if k != "p95_ttft_ms"}
        if axes != provenance["serve_slos_other_axes"]:
            raise PrefillFrontierError("replay serve_slos_other_axes differs from sweep")
        if point["target_bits"] != ctx.target_bits:
            raise PrefillFrontierError("replay point target_bits differs from sweep")
        stamp = {
            "schema": REPLAY_SCHEMA,
            "frontier_sha256": hashlib.sha256(raw).hexdigest(),
            "assignment_sha256": digest,
            "slo_ms": point["slo_ms"], "target_bits": point["target_bits"],
            "table_identity": ctx.table_identity,
            "cost_sha256": file_sha256hex(Path(ctx.cost_path)),
            "probe_sha256": file_sha256hex(Path(ctx.probe_path)),
            "probe_bound_by_sweep": "probe_sha256" in provenance,
        }
        selection = document.get("pact_selection")
        if selection is not None:
            # Same claim the hull replay stamps; an old sweep has none and replays as before.
            stamp["regime_m"] = document["regime_m"]
            stamp["pact_selection_sha256"] = selection["identity_sha256"]
            stamp["pact_selection"] = selection
        ctx.emit_replay(point["slo_ms"], point["target_bits"], expected, stamp)
        emitted = True

    allocator.main([*allocator_argv, "--layer-config", str(output)], measured_runtime_sweep=emit)
    if not emitted:
        raise PrefillFrontierError("allocator returned without emitting replay")


def replay_main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m prismaquant.prefill_frontier replay")
    ap.add_argument("--frontier", type=Path, required=True)
    ap.add_argument("--assignment-sha256", required=True)
    ap.add_argument("--layer-config", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        replay(args.frontier, args.assignment_sha256, args.layer_config)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        ap.error(str(exc))
    return 0


def hull_document(ap: argparse.ArgumentParser, args, allocator_argv: list[str],
                  assignments_dir: Path) -> dict:
    """One PACT allocator load and document for the declared selection mode."""
    if args.slo_ms is not None or args.slo_grid is not None:
        ap.error("--slo-ms/--slo-grid are an SLO grid; PACT (--pact-shape-table) has no grid")
    if args.loss_noise_floor != 0.0:
        ap.error("--loss-noise-floor applies to the SLO grid's nondominance, not to PACT")
    from . import allocator
    document: dict | None = None

    def hull(ctx) -> None:
        nonlocal document
        if not isinstance(ctx, allocator.PactHullSweep):
            raise SystemExit("[pact-hull] the allocator did not enter PACT mode")
        run = run_constrained if ctx.selection_mode == "constrained" else run_hull
        document = run(ctx, assignments_dir=assignments_dir, allocator_argv=allocator_argv,
                       bootstrap_draws=args.bootstrap_draws, bootstrap_seed=args.bootstrap_seed)

    allocator.main(list(allocator_argv), measured_runtime_sweep=hull)
    if document is None:
        raise SystemExit("[pact-hull] the allocator returned without building the hull")
    return document


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["replay"]:
        return replay_main(argv[1:])
    own, allocator_argv = _split_argv(argv)
    ap = build_parser()
    args = ap.parse_args(own)
    if not allocator_argv:
        ap.error("allocator arguments are required after '--'")
    pact = is_pact_argv(allocator_argv)
    if not pact:
        try:
            grid = parse_grid_spec(args.slo_ms, args.slo_grid)
        except PrefillFrontierError as exc:
            ap.error(str(exc))
    if not math.isfinite(args.loss_noise_floor) or args.loss_noise_floor < 0:
        ap.error("--loss-noise-floor must be finite and nonnegative")
    if args.bootstrap_draws < 1:
        # Every point publishes its interval. There is no "skip the dispersion"
        # setting: a sum of medians without one reads as a resolved number.
        ap.error("--bootstrap-draws must be at least 1")
    output = Path(args.output)
    assignments_dir = (Path(args.assignments_dir) if args.assignments_dir
                       else output.with_name(output.name + ".assignments"))

    from . import allocator
    document: dict | None = None

    def sweep(ctx) -> None:
        nonlocal document
        document = run_sweep(ctx, grid=grid, assignments_dir=assignments_dir,
                             loss_noise_floor=args.loss_noise_floor,
                             allocator_argv=allocator_argv,
                             bootstrap_draws=args.bootstrap_draws,
                             bootstrap_seed=args.bootstrap_seed)

    # Loader and identity refusals surface as the allocator's own SystemExit;
    # they are not wrapped here.
    if pact:
        document = hull_document(ap, args, allocator_argv, assignments_dir)
    else:
        allocator.main(list(allocator_argv), measured_runtime_sweep=sweep)
    if document is None:
        raise SystemExit("[prefill-frontier] the allocator returned without running the sweep")
    output.parent.mkdir(parents=True, exist_ok=True)
    # The document is one name with one current value, so it is REPLACED, and
    # by the shared atomic writer rather than write_text: an interrupted sweep
    # otherwise leaves a truncated curve where a complete one was, and every
    # reader of the output path reads the half-file with nothing to tell it so.
    atomic_write_bytes(
        output, (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))
    if pact:
        if document["candidate_generator"] == CONSTRAINED_GENERATOR:
            print(f"[pact-constrained] selected {document['selected_assignment_sha256']} "
                  f"within {document['constraints']['max_prefill_ms']:.6g} operator ms "
                  f"-> {output}", flush=True)
            return 0
        print(f"[pact-hull] {document['n_vertices']} vertices from {document['probe_count']} "
              f"exact probes in {document['hull_seconds']:.1f} s at M={document['regime_m']} "
              f"TP{document['tensor_parallel']} -> {output}", flush=True)
        return 0
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
