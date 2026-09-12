"""The ``coverage_first_v1`` screen: what the initial menu may and may not drop.

Specification section 5.4 of ``docs/design/tessera_quality_prefill_experiment.md``.

``coverage_first_v1`` is the experiment's *initial* screen policy, and its
defining property is negative: **it never prunes**. It retains the mandatory
boundary and control candidates, the frozen per-stratum interior draw, and at
least one representative of every distinct licensed activation/route class.
Everything else is deferred, which is a statement about acquisition order, not
about merit. :func:`apply_coverage_first_v1` emits no ``pruned`` disposition
at all, and ``tests/test_quality_prefill_screen.py`` asserts that.

That is not squeamishness. Section 5.4's reason is mechanical: before runtime
prices exist, a quality-or-byte dominance argument cannot see that the deleted
candidate was the faster one. :func:`refuse_dominance_screen` makes the
argument refuse out loud rather than quietly succeed on partial evidence.

Caps are the other half. ``max_retained_candidates``,
``max_composite_options`` and ``max_tasks_per_phase`` are applied *after* the
exact roster and its licensed composite expansion, and a cap that the
mandatory coverage already exceeds refuses the freeze, naming required versus
allowed. It never reduces a quota, drops a class, or retains a prefix: a
smaller plan is a different plan, and picking one silently is how a screen
becomes a result.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .quality_prefill_contract import (
    QualityPrefillContractError,
    SCREEN_DECISION_SCHEMA,
    canonical_sha256,
    validate_screen_decision,
)
from .quality_prefill_population import (
    PURPOSE_DISCARDED_CANDIDATE_AUDIT,
    SCREEN_POLICY_ID,
    stable_rank,
)

__all__ = [
    "ScreenError",
    "ScreenCaps",
    "ScreenCandidate",
    "ScreenResult",
    "SCREEN_RULE",
    "expand_licensed_composites",
    "refuse_dominance_screen",
    "apply_coverage_first_v1",
]

#: Published identity of the policy this module implements. A later hard
#: pruning policy is a different name and a different version (section 5.4);
#: it does not arrive as a new branch inside this function.
SCREEN_RULE = MappingProxyType({"name": SCREEN_POLICY_ID, "version": "v1"})

_RETAINED = "retained"
_DEFERRED = "deferred"

#: Coverage type for something retained but not yet measured, and for
#: something not yet acquired at all. Both are closed-set members of the
#: contract's ``COVERAGE_TYPES``, so neither can later be read as ``measured``.
_STATUS_RETAINED = "retained_pending_measurement"
_STATUS_DEFERRED = "not_yet_acquired"


class ScreenError(QualityPrefillContractError):
    """A screen plan cannot be applied as written.

    Every case is a fixable document -- a wider cap, a declared uncertainty, a
    named route class -- which is why this is a contract error and not the
    fleet-capability ``RuntimeError`` the PB adapter raises.
    """


def _fail(message: str) -> None:
    raise ScreenError(message)


@dataclass(frozen=True)
class ScreenCaps:
    """The manifest's integer work limits.

    All three are required integers. There is no ``None`` meaning "unlimited":
    an unstated cap is an unfrozen plan, and section 5.4 requires the manifest
    to carry them.
    """

    max_retained_candidates: int
    max_composite_options: int
    max_tasks_per_phase: int

    def __post_init__(self) -> None:
        for name in (
            "max_retained_candidates",
            "max_composite_options",
            "max_tasks_per_phase",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                _fail(f"{name} must be a positive int; got {value!r}")

    def as_dict(self) -> dict[str, int]:
        return {
            "max_retained_candidates": self.max_retained_candidates,
            "max_composite_options": self.max_composite_options,
            "max_tasks_per_phase": self.max_tasks_per_phase,
        }


@dataclass(frozen=True)
class ScreenCandidate:
    """One candidate as the screen sees it, before any evidence is opened.

    ``uncertainty`` has no default. Section 5.4's rule is that unknown quality,
    runtime or uncertainty is not zero and cannot prove dominance; writing
    ``0.0`` for "we have not looked" is precisely the lie that rule forbids, so
    a missing value refuses instead.
    """

    candidate_id: str
    rate_stratum_id: str
    route_class: str
    mandatory: bool
    interior_draw: bool
    composite_options: int
    tasks_per_phase: int
    uncertainty: float | None
    runtime_price_known: bool

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            _fail("candidate_id must be a non-empty string")
        for name in ("rate_stratum_id", "route_class"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                _fail(f"{self.candidate_id}: {name} must be a non-empty string")
        for name in ("mandatory", "interior_draw", "runtime_price_known"):
            if type(getattr(self, name)) is not bool:
                _fail(f"{self.candidate_id}: {name} must be a boolean")
        for name in ("composite_options", "tasks_per_phase"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                _fail(f"{self.candidate_id}: {name} must be a positive int")
        if self.uncertainty is not None:
            if type(self.uncertainty) not in (int, float):
                _fail(f"{self.candidate_id}: uncertainty must be a number or None")
            if self.uncertainty != self.uncertainty or self.uncertainty < 0.0:
                _fail(
                    f"{self.candidate_id}: uncertainty must be finite and "
                    "non-negative"
                )


@dataclass(frozen=True)
class ScreenResult:
    """The screen's complete disposition of every candidate it was shown."""

    retained: tuple[str, ...]
    deferred: tuple[str, ...]
    ranks: Mapping[str, int]
    audit: tuple[str, ...]
    required: Mapping[str, int]
    caps: ScreenCaps
    decision: Mapping[str, object]

    def identity_sha256(self) -> str:
        """The decision's seal, as ``_check_seal`` would recompute it.

        Returned rather than recomputed: the seal covers the body *without*
        its own field, so hashing the sealed mapping again would digest a
        different object and quietly stop matching the contract's.
        """
        return str(self.decision["identity_sha256"])


def expand_licensed_composites(candidates: Iterable[ScreenCandidate]) -> int:
    """Total licensed composite options across the given candidates.

    Caps are checked against this number, not against the candidate count,
    because one candidate may license several composite options and the work
    the fleet actually performs is per option.
    """
    return sum(candidate.composite_options for candidate in candidates)


def refuse_dominance_screen(candidates: Sequence[ScreenCandidate]) -> None:
    """Refuse a dominance argument while any runtime price is unknown.

    Section 5.4: before runtime prices exist, quality/byte dominance alone must
    not remove a possibly faster execution route. This function is the place
    that says so; call it before any hard pruning policy, and it raises with
    the exact candidates whose runtime is still unmeasured.
    """
    unknown = tuple(
        sorted(c.candidate_id for c in candidates if not c.runtime_price_known)
    )
    if unknown:
        _fail(
            "a quality/byte dominance screen cannot run while "
            f"{len(unknown)} candidate(s) have no measured runtime price: "
            + ", ".join(unknown[:8])
            + (" ..." if len(unknown) > 8 else "")
            + ". An unmeasured route is not a slow route"
        )


def _class_representatives(
    candidates: Sequence[ScreenCandidate],
    already: set[str],
    *,
    source_sha256: str,
    selection_seed: int,
) -> dict[str, str]:
    """One retained representative per licensed activation/route class.

    A class already covered by a mandatory or interior-draw candidate needs no
    extra representative. An uncovered class takes its first candidate by the
    population module's stable rank, so the choice is reproducible on any box.
    """
    covered = {
        candidate.route_class
        for candidate in candidates
        if candidate.candidate_id in already
    }
    chosen: dict[str, str] = {}
    by_class: dict[str, list[str]] = {}
    for candidate in candidates:
        if candidate.route_class in covered:
            continue
        by_class.setdefault(candidate.route_class, []).append(candidate.candidate_id)
    for route_class in sorted(by_class):
        ranked = stable_rank(
            sorted(by_class[route_class]),
            source_sha256=source_sha256,
            selection_seed=selection_seed,
            purpose=PURPOSE_DISCARDED_CANDIDATE_AUDIT,
            stratum_id=f"route_class:{route_class}",
        )
        chosen[route_class] = ranked[0]
    return chosen


def apply_coverage_first_v1(
    candidates: Iterable[ScreenCandidate],
    *,
    caps: ScreenCaps,
    source_sha256: str,
    selection_seed: int = 0,
    audit_count: int,
    decision_id: str,
) -> ScreenResult:
    """Apply the initial screen and return its complete disposition ledger.

    Retains, in this order and for these reasons:

    1. every mandatory boundary or control candidate;
    2. every member of the frozen per-stratum interior draw;
    3. one representative of each licensed activation/route class not already
       covered by 1 or 2.

    Everything else is ``deferred``. Nothing is ``pruned``: this policy makes
    no hard deletion, and a later one is a separate version.

    ``audit_count`` fixes how many deferred candidates are drawn, by stable
    rank, for the deterministic discarded-candidate audit. Section 5.4 requires
    that roster to be chosen *before* the deferred candidates' joint or runtime
    truth is opened, which is why this function -- which sees no prices --
    is where it is drawn.
    """
    roster = tuple(candidates)
    if not roster:
        _fail("coverage_first_v1 needs at least one candidate")
    ids = [candidate.candidate_id for candidate in roster]
    if len(set(ids)) != len(ids):
        duplicates = sorted({cid for cid in ids if ids.count(cid) > 1})
        _fail("duplicate candidate id(s): " + ", ".join(duplicates))
    if type(audit_count) is not int or audit_count < 0:
        _fail("audit_count must be a non-negative int")
    if not isinstance(decision_id, str) or not decision_id:
        _fail("decision_id must be a non-empty string")

    missing_uncertainty = tuple(
        sorted(c.candidate_id for c in roster if c.uncertainty is None)
    )
    if missing_uncertainty:
        _fail(
            f"{len(missing_uncertainty)} candidate(s) carry no declared "
            "uncertainty: "
            + ", ".join(missing_uncertainty[:8])
            + (" ..." if len(missing_uncertainty) > 8 else "")
            + ". Unknown uncertainty is not zero, so the ledger refuses rather "
            "than writing 0.0"
        )

    by_id = {candidate.candidate_id: candidate for candidate in roster}
    reasons: dict[str, str] = {}
    for candidate in roster:
        if candidate.mandatory:
            reasons[candidate.candidate_id] = (
                "mandatory boundary or control candidate; coverage_first_v1 "
                "retains every one"
            )
    for candidate in roster:
        if candidate.interior_draw and candidate.candidate_id not in reasons:
            reasons[candidate.candidate_id] = (
                "member of the frozen per-stratum interior draw for "
                f"{candidate.rate_stratum_id}"
            )
    representatives = _class_representatives(
        roster,
        set(reasons),
        source_sha256=source_sha256,
        selection_seed=selection_seed,
    )
    for route_class, candidate_id in representatives.items():
        reasons[candidate_id] = (
            f"sole retained representative of licensed route class {route_class}; "
            "a class is never dropped before runtime prices exist"
        )

    retained = tuple(sorted(reasons))
    deferred = tuple(sorted(set(ids) - set(retained)))

    required = {
        "retained_candidates": len(retained),
        "composite_options": expand_licensed_composites(
            by_id[cid] for cid in retained
        ),
        "tasks_per_phase": sum(by_id[cid].tasks_per_phase for cid in retained),
    }
    over = [
        (name, required[name], allowed)
        for name, allowed in (
            ("retained_candidates", caps.max_retained_candidates),
            ("composite_options", caps.max_composite_options),
            ("tasks_per_phase", caps.max_tasks_per_phase),
        )
        if required[name] > allowed
    ]
    if over:
        detail = "; ".join(
            f"{name} requires {need} but the manifest allows {allowed}"
            for name, need, allowed in over
        )
        _fail(
            "coverage_first_v1 refuses to freeze: "
            + detail
            + ". A cap never reduces a quota, drops a route class or retains a "
            "prefix; raising it is a new explicit plan"
        )

    if audit_count > len(deferred):
        _fail(
            f"the discarded-candidate audit asks for {audit_count} of "
            f"{len(deferred)} deferred candidate(s); it cannot draw more than "
            "the pool holds"
        )
    audit = tuple(
        stable_rank(
            list(deferred),
            source_sha256=source_sha256,
            selection_seed=selection_seed,
            purpose=PURPOSE_DISCARDED_CANDIDATE_AUDIT,
            stratum_id=f"screen:{decision_id}",
        )[:audit_count]
    ) if deferred else ()

    ranked_all = stable_rank(
        sorted(ids),
        source_sha256=source_sha256,
        selection_seed=selection_seed,
        purpose=PURPOSE_DISCARDED_CANDIDATE_AUDIT,
        stratum_id=f"screen:{decision_id}",
    )
    ranks = {candidate_id: index for index, candidate_id in enumerate(ranked_all)}

    dispositions = []
    for candidate_id in sorted(ids):
        candidate = by_id[candidate_id]
        is_retained = candidate_id in reasons
        reason = reasons.get(
            candidate_id,
            "deferred by coverage_first_v1: outside the mandatory set, the "
            "frozen interior draw and the route-class representatives. Deferred "
            "is an acquisition order, not a deletion"
            + (
                f"; drawn for the discarded-candidate audit at rank {ranks[candidate_id]}"
                if candidate_id in audit
                else ""
            ),
        )
        dispositions.append(
            {
                "candidate_id": candidate_id,
                "disposition": _RETAINED if is_retained else _DEFERRED,
                "measurement_status": _STATUS_RETAINED
                if is_retained
                else _STATUS_DEFERRED,
                "evidence_observation_ids": [],
                "reason": reason,
                "uncertainty": float(candidate.uncertainty),
            }
        )

    decision = {
        "schema": SCREEN_DECISION_SCHEMA,
        "decision_id": decision_id,
        "rule": dict(SCREEN_RULE),
        "dispositions": dispositions,
    }
    # The seal covers the body and never its own field, exactly as the
    # contract's `_seal` does. Computed before the key is added, on purpose.
    seal = canonical_sha256(decision)
    decision["identity_sha256"] = seal
    validate_screen_decision(decision)

    return ScreenResult(
        retained=retained,
        deferred=deferred,
        ranks=MappingProxyType(ranks),
        audit=audit,
        required=MappingProxyType(required),
        caps=caps,
        decision=MappingProxyType(decision),
    )
