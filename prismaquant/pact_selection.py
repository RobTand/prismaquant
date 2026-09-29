"""PACT selection record (PQ #1585).

PACT prices one exact frontier per regime M. This module turns that frontier
into three named picks and one record a shipcard can carry:

* ``high_accuracy``: the point with the lowest ``predicted_dloss`` that fits the
  byte budget.
* ``balanced``: the chord-perpendicular knee of
  :func:`prismaquant.quality_prefill_knee.select_development_point`.
* ``high_prefill``: the same chord rule recursed on the sub-frontier between the
  fastest point and the balanced pick. This rule is a proposal that awaits the
  owner's choice; the alternative is a served-KL ceiling. The rule id is
  recorded so a change of rule is visible in the record identity.

Nothing here carries a hand-picked threshold (CLAUDE.md principle 2). Two
decisions would otherwise need one, and both derive from the table's own
bootstrap intervals instead:

**Materiality.** The time axis is flat when the frontier's attained-time span
is no larger than the pooled resolution of its two endpoints, that is, when the
fastest endpoint's upper bootstrap bound reaches the best endpoint's lower
bound. No knee exists on a flat axis and every pick is the accuracy endpoint.

**Resolution of the top two.** The knee rule refuses when its two best
candidates are closer than ``min_separation``, expressed in normalised
perpendicular units. The normalisation maps the fastest point to (0, 1) and the
best point to (1, 0), so a candidate whose time moves by ``h`` milliseconds
moves ``h / span`` along the time axis and ``h / (span * sqrt(2))`` along the
unit perpendicular ``(1, 1) / sqrt(2)``. The resolution is the sum of the top
two candidates' interval half widths in those units. A first pass ranks the
candidates with no resolution; a second pass applies the derived one.

The quality axis (``predicted_dloss``) carries no interval today, so the record
states ``quality_axis.interval = None``. Noise on the endpoints' quality is not
part of the resolution; the record says so instead of hiding it.
"""

from __future__ import annotations

import math
from fractions import Fraction
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

from .quality_prefill_contract import (
    JOINT_CURRENCY,
    QualityPrefillContractError,
    canonical_sha256,
)
from .quality_prefill_knee import (
    KNEE_RULE,
    KneeSelection,
    KneeSelectionError,
    nondominated,
    read_points,
    select_development_point,
)
from .schemas import Contract

SCHEMA = "prismaquant.pact_selection.v1"

HIGH_PREFILL_RULE = MappingProxyType(
    {
        "name": "chord_perpendicular_recursed_v1",
        "version": "v1",
        "sub_frontier": "non_dominated_points_no_slower_than_balanced",
        "decision": "default_pending_owner_choice",
    }
)
_MATERIALITY_RULE = MappingProxyType(
    {"name": "endpoint_interval_overlap_v1", "version": "v1", "axis": "attained_time"}
)
_SEPARATION_RULE = MappingProxyType(
    {
        "name": "top_two_interval_half_widths_v1",
        "version": "v1",
        "unit_perpendicular": "(1, 1) / sqrt(2)",
    }
)
_ACCURACY_RULE = MappingProxyType({"name": "argmin_predicted_dloss_v1", "version": "v1"})

_VERDICTS = ("flat", "material", "unresolved", "single_point", "no_feasible_point")
_STATUSES = ("selected", "refused", "flat_time_axis", "single_point")
_PICK_NAMES = ("high_accuracy", "balanced", "high_prefill")
_ROLE_NAMES = frozenset(
    {
        *_PICK_NAMES,
        "balanced_neighbour",
        "high_prefill_neighbour",
        "endpoint_fastest",
        "endpoint_best",
    }
)
_UNIT = "x"
_NS_PER_MS = 1_000_000


class PactSelectionError(QualityPrefillContractError):
    """Raised when a PACT selection cannot be made or a record is not valid."""


_fail = Contract(PactSelectionError).fail


# ------------------------------------------------------------------- inputs


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _interval(raw: object, where: str) -> tuple[float, float] | None:
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        _fail(f"{where}.time_interval_ms must be null or a [lo, hi] pair")
    lo, hi = raw
    if not (_finite(lo) and _finite(hi)) or lo > hi:
        _fail(f"{where}.time_interval_ms must be finite with lo <= hi")
    return float(lo), float(hi)


def _lanes(raw: object, where: str) -> dict[str, int] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        _fail(f"{where}.kernel_lanes must be null or a label-to-count mapping")
    out: dict[str, int] = {}
    for label in sorted(raw):
        count = raw[label]
        if not isinstance(label, str) or not label or type(count) is not int or count < 0:
            _fail(f"{where}.kernel_lanes must map non-empty labels to counts")
        out[label] = count
    return out


def _clean_points(points: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """Validate the input points and return them sorted by point id."""
    cleaned: dict[str, dict[str, object]] = {}
    for index, raw in enumerate(points):
        where = f"points[{index}]"
        if not isinstance(raw, Mapping):
            _fail(f"{where} must be a mapping")
        point_id = raw.get("point_id")
        if not isinstance(point_id, str) or not point_id:
            _fail(f"{where}.point_id must be a non-empty string")
        if point_id in cleaned:
            _fail(f"{where}.point_id {point_id!r} is already present")
        sha = raw.get("assignment_sha256")
        if not isinstance(sha, str) or not sha:
            _fail(f"{where}.assignment_sha256 must be a non-empty string")
        nbytes = raw.get("bytes")
        if type(nbytes) is not int or nbytes < 1:
            _fail(f"{where}.bytes must be a positive int")
        time_ms = raw.get("time_ms")
        if not _finite(time_ms) or time_ms <= 0:
            _fail(f"{where}.time_ms must be a positive finite number")
        dloss = raw.get("predicted_dloss")
        if not _finite(dloss) or dloss < 0:
            _fail(f"{where}.predicted_dloss must be a finite non-negative number")
        cleaned[point_id] = {
            "point_id": point_id,
            "assignment_sha256": sha,
            "bytes": nbytes,
            "time_ms": float(time_ms),
            "time_interval_ms": _interval(raw.get("time_interval_ms"), where),
            "predicted_dloss": float(dloss),
            "kernel_lanes": _lanes(raw.get("kernel_lanes"), where),
        }
    if not cleaned:
        _fail("a PACT selection needs at least one frontier point")
    return [cleaned[key] for key in sorted(cleaned)]


def _ns(time_ms: float) -> int:
    """Exact integer nanoseconds for a millisecond float."""
    ns = round(Fraction(time_ms) * _NS_PER_MS)
    if ns < 1:
        _fail(f"time {time_ms} ms is below one nanosecond")
    return int(ns)


def _knee_row(point: Mapping[str, object]) -> dict[str, object]:
    return {
        "point_id": point["point_id"],
        "assignment_id": point["assignment_sha256"],
        "bytes": point["bytes"],
        "prefill_budget": _ns(point["time_ms"]),
        "quality_value": point["predicted_dloss"],
        "currency": JOINT_CURRENCY,
        "units": _UNIT,
        "measurement_status": "measured",
    }


# --------------------------------------------------------------- materiality


def materiality(
    fastest: Mapping[str, object],
    best: Mapping[str, object],
    *,
    allow_unresolved: bool = False,
) -> dict[str, object]:
    """Decide whether the frontier's time span exceeds its endpoint resolution.

    ``fastest`` and ``best`` are the frontier's two endpoints, each a mapping
    with ``point_id``, ``time_ms`` and ``time_interval_ms``. The axis is flat
    when the fastest endpoint's upper bound reaches the best endpoint's lower
    bound, which is the same as the span being no larger than the sum of the two
    one-sided widths that face each other.
    """
    span = float(best["time_ms"]) - float(fastest["time_ms"])
    blank: dict[str, object] = {
        "verdict": "",
        "span_ms": span,
        "fastest_upper_width_ms": None,
        "best_lower_width_ms": None,
        "resolution_ms": None,
        "pooled_interval_ms": None,
        "reason": "",
    }
    if fastest["point_id"] == best["point_id"]:
        blank.update(
            verdict="single_point",
            span_ms=0.0,
            reason="the fastest and the most accurate point are the same point",
        )
        return blank
    fast_iv, best_iv = fastest.get("time_interval_ms"), best.get("time_interval_ms")
    if fast_iv is None or best_iv is None:
        missing = [
            name
            for name, iv in (("fastest", fast_iv), ("most accurate", best_iv))
            if iv is None
        ]
        reason = "no interval on the " + " and ".join(missing) + " endpoint"
        if not allow_unresolved:
            _fail(
                f"materiality is unresolved: {reason}. A production selection needs "
                "the table's bootstrap interval on both endpoints"
            )
        blank.update(verdict="unresolved", reason=reason)
        return blank
    upper = float(fast_iv[1]) - float(fastest["time_ms"])
    lower = float(best["time_ms"]) - float(best_iv[0])
    flat = float(fast_iv[1]) >= float(best_iv[0])
    blank.update(
        verdict="flat" if flat else "material",
        fastest_upper_width_ms=upper,
        best_lower_width_ms=lower,
        resolution_ms=upper + lower,
        pooled_interval_ms=[
            min(float(fast_iv[0]), float(best_iv[0])),
            max(float(fast_iv[1]), float(best_iv[1])),
        ],
        reason=(
            "the span lies inside the pooled bootstrap resolution of its endpoints"
            if flat
            else "the span exceeds the pooled bootstrap resolution of its endpoints"
        ),
    )
    return blank


# ---------------------------------------------------------------- one knee


def _endpoints(frontier):
    fastest = min(frontier, key=lambda p: (p.prefill_budget, p.quality_value, p.assignment_id))
    best = min(frontier, key=lambda p: (p.quality_value, p.prefill_budget, p.assignment_id))
    return fastest, best


def _derive_resolution(
    first: KneeSelection,
    typed_by_id: Mapping[str, object],
    by_id: Mapping[str, Mapping[str, object]],
    mat: Mapping[str, object],
    *,
    allow_unresolved: bool,
) -> dict[str, object]:
    """Derive ``min_separation`` from the top two candidates' interval widths."""
    endpoint_ids = set(first.endpoint_point_ids)
    interior = [
        typed_by_id[pid] for pid in first.nondominated_point_ids if pid not in endpoint_ids
    ]
    ranked = sorted(
        interior,
        key=lambda p: (-first.improvements[p.point_id], p.prefill_budget, p.assignment_id),
    )[:2]
    if len(ranked) < 2:
        return {"value": 0.0, "source": "single_interior_candidate"}
    if mat["verdict"] == "unresolved":
        return {"value": 0.0, "source": "no_intervals"}
    halves = []
    for candidate in ranked:
        interval = by_id[candidate.point_id]["time_interval_ms"]
        if interval is None:
            if allow_unresolved:
                return {"value": 0.0, "source": "no_intervals"}
            _fail(
                f"point {candidate.point_id!r} is a top-two candidate and has no "
                "interval, so the resolution cannot be derived"
            )
        halves.append((interval[1] - interval[0]) / 2.0)
    span = float(mat["span_ms"])
    return {
        "value": float(sum(halves) / (span * math.sqrt(2.0))),
        "source": "top_two_interval_half_widths",
    }


def _derive_knee(
    points: Sequence[Mapping[str, object]],
    *,
    byte_budget: int,
    allow_unresolved: bool,
) -> dict[str, object]:
    """Materiality, derived resolution and the chord-rule nominee for ``points``.

    ``points`` are already inside the byte budget.
    """
    rows = [_knee_row(point) for point in points]
    try:
        typed = read_points(rows)
    except QualityPrefillContractError as exc:
        raise PactSelectionError(str(exc)) from exc
    typed_by_id = {p.point_id: p for p in typed}
    by_id = {p["point_id"]: p for p in points}
    frontier = nondominated(typed)
    fastest, best = _endpoints(frontier)
    mat = materiality(
        by_id[fastest.point_id], by_id[best.point_id], allow_unresolved=allow_unresolved
    )
    out: dict[str, object] = {
        "frontier_ids": [p.point_id for p in frontier],
        "fastest_id": fastest.point_id,
        "best_id": best.point_id,
        "materiality": mat,
        "min_separation": None,
        "selection": None,
        "refusal": None,
    }
    if mat["verdict"] in ("flat", "single_point"):
        return out
    try:
        first = select_development_point(rows, byte_budget=byte_budget, min_separation=0.0)
    except KneeSelectionError as exc:
        out["refusal"] = f"the chord rule refused: {exc}"
        return out
    resolution = _derive_resolution(
        first, typed_by_id, by_id, mat, allow_unresolved=allow_unresolved
    )
    out["min_separation"] = resolution
    try:
        out["selection"] = select_development_point(
            rows, byte_budget=byte_budget, min_separation=resolution["value"]
        )
    except KneeSelectionError as exc:
        out["refusal"] = (
            "the derived resolution does not separate the top two candidates: " f"{exc}"
        )
    return out


# --------------------------------------------------------------- the record


def _pick(point_id: str | None, status: str, **extra: object) -> dict[str, object]:
    return {"point_id": point_id, "status": status, **extra}


def select_pact(
    points: Iterable[Mapping[str, object]],
    *,
    regime_m: int,
    table_identity: Mapping[str, object],
    frontier_scope: str,
    byte_budget: int | None = None,
    allow_unresolved: bool = False,
) -> dict[str, object]:
    """Return the ``prismaquant.pact_selection.v1`` record for one regime.

    Each point is a mapping with ``point_id``, ``assignment_sha256``, ``bytes``,
    ``time_ms``, ``time_interval_ms`` (a ``[lo, hi]`` pair or ``None``),
    ``predicted_dloss`` and ``kernel_lanes`` (a label-to-count mapping or
    ``None``). ``byte_budget`` defaults to the largest point and is recorded.

    A production selection refuses when an endpoint has no interval, because the
    materiality test cannot run. ``allow_unresolved`` lets a scratch table with
    no bootstrap columns through; the record then says ``unresolved``.
    """
    if type(regime_m) is not int or regime_m < 1:
        _fail("regime_m must be a positive int")
    if not isinstance(table_identity, Mapping) or not table_identity:
        _fail("table_identity must be a non-empty mapping")
    if not isinstance(frontier_scope, str) or not frontier_scope:
        _fail("frontier_scope must be a non-empty string")
    cleaned = _clean_points(points)
    if byte_budget is None:
        byte_budget = max(p["bytes"] for p in cleaned)
    if type(byte_budget) is not int or byte_budget < 1:
        _fail("byte_budget must be a positive int")
    feasible = [p for p in cleaned if p["bytes"] <= byte_budget]
    by_id = {p["point_id"]: p for p in cleaned}

    picks: dict[str, dict[str, object]] = {}
    roles: dict[str, set[str]] = {}

    def add_role(point_id: str, role: str) -> None:
        roles.setdefault(point_id, set()).add(role)

    balanced_derivation: dict[str, object] | None = None
    if not feasible:
        reason = (
            f"no point fits the byte budget {byte_budget}; the smallest is "
            f"{min(p['bytes'] for p in cleaned)} bytes"
        )
        mat = {
            "verdict": "no_feasible_point",
            "span_ms": None,
            "fastest_upper_width_ms": None,
            "best_lower_width_ms": None,
            "resolution_ms": None,
            "pooled_interval_ms": None,
            "reason": reason,
        }
        separation = {"value": None, "source": "not_derived"}
        for name in _PICK_NAMES:
            picks[name] = _pick(None, "refused", reason=reason)
    else:
        accuracy = min(
            feasible,
            key=lambda p: (p["predicted_dloss"], p["time_ms"], p["point_id"]),
        )
        derived = _derive_knee(
            feasible, byte_budget=byte_budget, allow_unresolved=allow_unresolved
        )
        mat = derived["materiality"]
        picks["high_accuracy"] = _pick(accuracy["point_id"], "selected")
        add_role(accuracy["point_id"], "high_accuracy")
        add_role(derived["fastest_id"], "endpoint_fastest")
        add_role(derived["best_id"], "endpoint_best")
        separation = derived["min_separation"] or {
            "value": None,
            "source": "not_derived",
        }
        best_id = derived["best_id"]
        if mat["verdict"] in ("flat", "single_point"):
            # The time axis carries no material difference, so the accuracy
            # endpoint stands for all three picks.
            status = "flat_time_axis" if mat["verdict"] == "flat" else "single_point"
            for name in ("balanced", "high_prefill"):
                picks[name] = _pick(best_id, status, reason=mat["reason"])
                add_role(best_id, name)
            picks["high_accuracy"] = _pick(best_id, "selected")
            roles[accuracy["point_id"]].discard("high_accuracy")
            add_role(best_id, "high_accuracy")
        elif derived["selection"] is None:
            balanced_derivation = {"materiality": mat, "min_separation": separation}
            picks["balanced"] = _pick(None, "refused", reason=derived["refusal"])
            picks["high_prefill"] = _pick(
                None,
                "refused",
                reason="the balanced pick is refused, so there is no sub-frontier: "
                + str(derived["refusal"]),
            )
        else:
            knee: KneeSelection = derived["selection"]
            balanced_id = knee.knee_point_id
            picks["balanced"] = _pick(
                balanced_id,
                "selected",
                improvement=float(knee.improvements[balanced_id]),
                normalisation=knee.normalisation.as_dict(),
            )
            add_role(balanced_id, "balanced")
            for pid in knee.neighbour_point_ids:
                add_role(pid, "balanced_neighbour")
            balanced_derivation = {"materiality": mat, "min_separation": separation}
            balanced_ns = _ns(by_id[balanced_id]["time_ms"])
            sub = [p for p in feasible if _ns(p["time_ms"]) <= balanced_ns]
            sub_frontier = nondominated(
                read_points([_knee_row(p) for p in sub])
            )
            if len(sub_frontier) < 3:
                picks["high_prefill"] = _pick(
                    None,
                    "refused",
                    reason=(
                        f"the sub-frontier [fastest, balanced] has {len(sub_frontier)} "
                        "non-dominated point(s), fewer than 3, so the chord rule "
                        "has no interior"
                    ),
                )
            else:
                inner = _derive_knee(
                    sub, byte_budget=byte_budget, allow_unresolved=allow_unresolved
                )
                sub_note = {
                    "materiality": inner["materiality"],
                    "min_separation": inner["min_separation"],
                }
                if inner["materiality"]["verdict"] in ("flat", "single_point"):
                    picks["high_prefill"] = _pick(
                        None,
                        "refused",
                        reason="the sub-frontier's time span is inside its own "
                        "endpoint resolution: " + str(inner["materiality"]["reason"]),
                        derivation=sub_note,
                    )
                elif inner["selection"] is None:
                    picks["high_prefill"] = _pick(
                        None, "refused", reason=inner["refusal"], derivation=sub_note
                    )
                else:
                    sub_knee: KneeSelection = inner["selection"]
                    fast_id = sub_knee.knee_point_id
                    picks["high_prefill"] = _pick(
                        fast_id,
                        "selected",
                        improvement=float(sub_knee.improvements[fast_id]),
                        normalisation=sub_knee.normalisation.as_dict(),
                        derivation=sub_note,
                    )
                    add_role(fast_id, "high_prefill")
                    for pid in sub_knee.neighbour_point_ids:
                        add_role(pid, "high_prefill_neighbour")

    roster = []
    for point_id in sorted(roles, key=lambda pid: (by_id[pid]["time_ms"], pid)):
        point = by_id[point_id]
        interval = point["time_interval_ms"]
        roster.append(
            {
                "point_id": point_id,
                "assignment_sha256": point["assignment_sha256"],
                "bytes": point["bytes"],
                "time_ms": point["time_ms"],
                "time_interval_ms": None if interval is None else list(interval),
                "predicted_dloss": point["predicted_dloss"],
                "kernel_lanes": point["kernel_lanes"],
                "roles": sorted(roles[point_id]),
            }
        )

    body: dict[str, object] = {
        "schema": SCHEMA,
        "regime_m": regime_m,
        "table_identity": dict(table_identity),
        "frontier_scope": frontier_scope,
        "byte_budget": byte_budget,
        "measurement_status": "predicted_proposal",
        "currency": JOINT_CURRENCY,
        "quality_axis": {
            "name": "predicted_dloss",
            "interval": None,
            "note": "the quality axis carries no bootstrap interval, so endpoint "
            "quality noise is not part of the resolution",
        },
        "time_axis": {"name": "attained_prefill_ms", "unit": "ms"},
        "rule": {
            "high_accuracy": dict(_ACCURACY_RULE),
            "balanced": dict(KNEE_RULE),
            "high_prefill": dict(HIGH_PREFILL_RULE),
            "materiality": dict(_MATERIALITY_RULE),
            "min_separation": dict(_SEPARATION_RULE),
        },
        "materiality": mat,
        "min_separation": separation,
        "picks": picks,
        "roster": roster,
    }
    body["identity_sha256"] = canonical_sha256(body)
    return body


# ----------------------------------------------------------------- validate


def validate_pact_selection(
    record: Mapping[str, object], *, require_resolved: bool = False
) -> Mapping[str, object]:
    """Return ``record`` when it is a consistent PACT selection, else raise.

    The check is the identity hash plus the structure the record implies: every
    pick is on the roster with its role, the endpoints on the roster reproduce
    the recorded materiality verdict, and a flat or single-point frontier picks
    one point. ``require_resolved`` also refuses an ``unresolved`` verdict.
    """
    if not isinstance(record, Mapping):
        _fail("a PACT selection record must be a mapping")
    if record.get("schema") != SCHEMA:
        _fail(f"PACT selection schema must be {SCHEMA!r}, got {record.get('schema')!r}")
    required = {
        "regime_m", "table_identity", "frontier_scope", "byte_budget",
        "measurement_status", "currency", "quality_axis", "time_axis", "rule",
        "materiality", "min_separation", "picks", "roster", "identity_sha256",
    }
    missing = required - set(record)
    if missing:
        _fail("PACT selection is missing key(s): " + ", ".join(sorted(missing)))
    body = {k: v for k, v in record.items() if k != "identity_sha256"}
    if record["identity_sha256"] != canonical_sha256(body):
        _fail("PACT selection identity_sha256 does not match its body")
    if type(record["regime_m"]) is not int or record["regime_m"] < 1:
        _fail("PACT selection regime_m must be a positive int")
    if record["measurement_status"] != "predicted_proposal":
        _fail("PACT selection measurement_status must be 'predicted_proposal'")
    rule = record["rule"]
    if not isinstance(rule, Mapping) or rule.get("high_prefill") != dict(HIGH_PREFILL_RULE):
        _fail("PACT selection rule.high_prefill is not the rule this build carries")
    if rule.get("balanced") != dict(KNEE_RULE):
        _fail("PACT selection rule.balanced is not the rule this build carries")

    mat = record["materiality"]
    verdict = mat.get("verdict") if isinstance(mat, Mapping) else None
    if verdict not in _VERDICTS:
        _fail(f"PACT selection materiality verdict {verdict!r} is not known")
    if require_resolved and verdict == "unresolved":
        _fail("PACT selection materiality is unresolved: the table had no interval")

    picks = record["picks"]
    if not isinstance(picks, Mapping) or set(picks) != set(_PICK_NAMES):
        _fail("PACT selection picks must be exactly " + ", ".join(_PICK_NAMES))
    roster = record["roster"]
    if not isinstance(roster, list):
        _fail("PACT selection roster must be a list")
    entries: dict[str, Mapping[str, object]] = {}
    for entry in roster:
        if not isinstance(entry, Mapping):
            _fail("PACT selection roster entries must be mappings")
        pid = entry.get("point_id")
        if not isinstance(pid, str) or not pid or pid in entries:
            _fail("PACT selection roster point ids must be unique non-empty strings")
        if not isinstance(entry.get("assignment_sha256"), str) or not entry["assignment_sha256"]:
            _fail(f"PACT selection roster entry {pid!r} needs an assignment_sha256")
        role_list = entry.get("roles")
        if (
            not isinstance(role_list, list)
            or not role_list
            or not set(role_list) <= _ROLE_NAMES
        ):
            _fail(f"PACT selection roster entry {pid!r} needs known roles")
        entries[pid] = entry

    for name in _PICK_NAMES:
        pick = picks[name]
        if not isinstance(pick, Mapping) or pick.get("status") not in _STATUSES:
            _fail(f"PACT selection pick {name!r} needs a known status")
        pid = pick.get("point_id")
        if pick["status"] == "refused":
            if pid is not None or not pick.get("reason"):
                _fail(f"PACT selection refused pick {name!r} needs a reason and no point")
            continue
        if pid not in entries:
            _fail(f"PACT selection pick {name!r} names {pid!r}, which is not on the roster")
        if name not in entries[pid]["roles"]:
            _fail(f"PACT selection roster entry {pid!r} does not carry the role {name!r}")

    if verdict == "no_feasible_point":
        if roster or any(p["status"] != "refused" for p in picks.values()):
            _fail("a frontier with no feasible point must refuse every pick and list no roster")
        return record

    def one_with(role: str) -> Mapping[str, object]:
        found = [e for e in roster if role in e["roles"]]
        if len(found) != 1:
            _fail(f"PACT selection roster must carry exactly one {role!r} entry")
        return found[0]

    fast_entry, best_entry = one_with("endpoint_fastest"), one_with("endpoint_best")
    try:
        recomputed = materiality(fast_entry, best_entry, allow_unresolved=True)
    except (KeyError, TypeError) as exc:
        raise PactSelectionError(f"PACT selection roster endpoints are malformed: {exc}") from exc
    if recomputed != dict(mat):
        _fail(
            "PACT selection materiality does not follow from its roster endpoints: "
            f"recorded {dict(mat)!r}, recomputed {recomputed!r}"
        )
    accuracy_id = picks["high_accuracy"]["point_id"]
    if verdict in ("flat", "single_point"):
        expected = "flat_time_axis" if verdict == "flat" else "single_point"
        for name in ("balanced", "high_prefill"):
            if picks[name]["status"] != expected or picks[name]["point_id"] != accuracy_id:
                _fail(f"a {verdict} frontier must pick the accuracy endpoint for {name!r}")
        if accuracy_id != best_entry["point_id"]:
            _fail("a flat or single-point frontier's accuracy pick must be the best endpoint")
    else:
        for name in ("balanced", "high_prefill"):
            if picks[name]["status"] in ("flat_time_axis", "single_point"):
                _fail(f"pick {name!r} claims a flat axis but the verdict is {verdict!r}")
        if picks["balanced"]["status"] == "selected" and picks["balanced"]["point_id"] in (
            fast_entry["point_id"],
            best_entry["point_id"],
        ):
            _fail("the balanced pick must be an interior point, not an endpoint")
    return record
