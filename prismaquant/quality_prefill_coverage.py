"""When a proposal may call itself the whole-model frontier, and when it may not.

Specification section 5.2 of ``docs/design/tessera_quality_prefill_experiment.md``.

Section 5.2 states one equality and one consequence. The equality is between
four memberships that are produced by four different stages and drift apart
quietly:

* the full mutable serving-unit census,
* the quality table's coverage,
* the runtime binding's expansion, and
* the selected assignment's membership.

The consequence is that a proposal may be labelled
``whole_model_retained_menu_frontier`` only when all four are the *same set*
and every routed stack is complete down to its members. Anything less keeps
its ``restricted_pilot`` label. A pilot operator frontier is still a real
result; it is just a result about the units it actually covered.

:func:`frontier_label` therefore never returns a verdict without also
returning why, and :func:`require_census_equality` refuses by naming the exact
symmetric difference rather than a count. "17 units differ" sends you looking;
"layers.3.mlp.shared.down is in the census and not in the runtime binding"
tells you what happened.

A routed stack needs every required member's wire, scale, source/PWC and
aligned joint row. The high/low routing-count expert diagnostic of section 5.1
is two members out of hundreds, so :func:`routed_stack_gaps` treats a
diagnostic draw as a gap, not as coverage -- which is the whole reason the
function exists.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .quality_prefill_contract import QualityPrefillContractError, canonical_sha256

__all__ = [
    "CoverageError",
    "MEMBERSHIP_NAMES",
    "REQUIRED_MEMBER_EVIDENCE",
    "RoutedStack",
    "CoverageVerdict",
    "require_census_equality",
    "routed_stack_gaps",
    "frontier_label",
]

#: The four memberships section 5.2 requires to be equal, in the order the
#: pipeline produces them. The order is part of the published report.
MEMBERSHIP_NAMES = (
    "census_units",
    "quality_table_units",
    "runtime_binding_units",
    "assignment_units",
)

#: What makes a routed member's option complete. All four, or the stack is
#: incomplete; there is no majority rule.
REQUIRED_MEMBER_EVIDENCE = ("wire", "scale", "source_pwc", "joint_row")

LABEL_WHOLE_MODEL = "whole_model_retained_menu_frontier"
LABEL_RESTRICTED_PILOT = "restricted_pilot"


class CoverageError(QualityPrefillContractError):
    """Four memberships that should be one set are not one set."""


def _fail(message: str) -> None:
    raise CoverageError(message)


def _as_set(value: object, *, where: str) -> frozenset[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        _fail(f"{where} must be an iterable of unit ids")
    items = list(value)  # type: ignore[arg-type]
    for index, item in enumerate(items):
        if not isinstance(item, str) or not item:
            _fail(f"{where}[{index}] must be a non-empty unit id string")
    if len(set(items)) != len(items):
        duplicates = sorted({item for item in items if items.count(item) > 1})
        _fail(f"{where} repeats unit id(s): " + ", ".join(duplicates))
    return frozenset(items)


def _sample(names: Iterable[str], limit: int = 6) -> str:
    ordered = sorted(names)
    shown = ", ".join(ordered[:limit])
    return shown + (f" ... (+{len(ordered) - limit} more)" if len(ordered) > limit else "")


def require_census_equality(memberships: Mapping[str, Iterable[str]]) -> frozenset[str]:
    """Refuse unless all four memberships are exactly the same set.

    Returns that common set so a caller can use it directly rather than
    picking one of the four and hoping.

    The refusal names each membership that differs from the census, and on
    which side, because the two directions mean opposite things: a unit in the
    census but not in the runtime binding is missing evidence, while a unit in
    the binding but not in the census is evidence about something the model
    does not contain.
    """
    if not isinstance(memberships, Mapping):
        _fail("memberships must be a mapping of the four membership names")
    missing_names = set(MEMBERSHIP_NAMES) - set(memberships)
    extra_names = set(memberships) - set(MEMBERSHIP_NAMES)
    if missing_names or extra_names:
        parts = []
        if missing_names:
            parts.append("missing " + ", ".join(sorted(missing_names)))
        if extra_names:
            parts.append("unknown " + ", ".join(sorted(extra_names)))
        _fail(
            "section 5.2 compares exactly "
            + ", ".join(MEMBERSHIP_NAMES)
            + "; "
            + " and ".join(parts)
        )

    sets = {
        name: _as_set(memberships[name], where=name) for name in MEMBERSHIP_NAMES
    }
    census = sets["census_units"]
    if not census:
        _fail("the mutable serving-unit census is empty; there is nothing to cover")

    complaints: list[str] = []
    for name in MEMBERSHIP_NAMES[1:]:
        absent = census - sets[name]
        surplus = sets[name] - census
        if absent:
            complaints.append(
                f"{len(absent)} unit(s) in the census are absent from {name}: "
                + _sample(absent)
            )
        if surplus:
            complaints.append(
                f"{len(surplus)} unit(s) in {name} are not mutable census units: "
                + _sample(surplus)
            )
    if complaints:
        _fail(
            "the four memberships of section 5.2 are not one set, so this is a "
            "restricted pilot and not a whole-model frontier. "
            + "; ".join(complaints)
        )
    return census


@dataclass(frozen=True)
class RoutedStack:
    """One routed serving unit and the evidence each of its members carries.

    ``required_members`` is the full member roster the stack needs at serve
    time. ``member_evidence`` maps a member to the evidence actually held. A
    member the roster names and the evidence map omits is a gap, and so is a
    member holding three of the four required kinds.
    """

    unit_id: str
    required_members: tuple[str, ...]
    member_evidence: Mapping[str, frozenset[str]]
    diagnostic_members: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.unit_id, str) or not self.unit_id:
            _fail("RoutedStack.unit_id must be a non-empty string")
        if not self.required_members:
            _fail(f"{self.unit_id}: a routed stack has at least one required member")
        if len(set(self.required_members)) != len(self.required_members):
            _fail(f"{self.unit_id}: required_members repeats a member")
        unknown = set(self.member_evidence) - set(self.required_members)
        if unknown:
            _fail(
                f"{self.unit_id}: member_evidence names member(s) the roster does "
                "not require: " + _sample(unknown)
            )
        stray = set(self.diagnostic_members) - set(self.required_members)
        if stray:
            _fail(
                f"{self.unit_id}: diagnostic_members names member(s) outside the "
                "roster: " + _sample(stray)
            )


def routed_stack_gaps(stacks: Sequence[RoutedStack]) -> dict[str, dict[str, list[str]]]:
    """Report, per stack, each member missing any required evidence kind.

    An empty result means every routed stack is complete. A stack whose only
    covered members are its diagnostic draw reports every other member as a
    gap, which is section 5.2's rule that a high/low expert diagnostic cannot
    satisfy the completeness gate.
    """
    gaps: dict[str, dict[str, list[str]]] = {}
    for stack in stacks:
        per_member: dict[str, list[str]] = {}
        for member in stack.required_members:
            held = stack.member_evidence.get(member, frozenset())
            if not isinstance(held, (set, frozenset)):
                _fail(
                    f"{stack.unit_id}.{member}: evidence must be a set of "
                    + ", ".join(REQUIRED_MEMBER_EVIDENCE)
                )
            unknown = set(held) - set(REQUIRED_MEMBER_EVIDENCE)
            if unknown:
                _fail(
                    f"{stack.unit_id}.{member}: unknown evidence kind(s) "
                    + _sample(unknown)
                )
            absent = [kind for kind in REQUIRED_MEMBER_EVIDENCE if kind not in held]
            if absent:
                per_member[member] = absent
        if per_member:
            gaps[stack.unit_id] = per_member
    return gaps


@dataclass(frozen=True)
class CoverageVerdict:
    """The label a proposal has earned, and the evidence for it."""

    label: str
    covered_units: tuple[str, ...]
    routed_gaps: Mapping[str, Mapping[str, tuple[str, ...]]]
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "covered_units": list(self.covered_units),
            "routed_gaps": {
                unit: {member: list(kinds) for member, kinds in sorted(members.items())}
                for unit, members in sorted(self.routed_gaps.items())
            },
            "reasons": list(self.reasons),
        }

    def identity_sha256(self) -> str:
        return canonical_sha256(self.as_dict())


def frontier_label(
    memberships: Mapping[str, Iterable[str]],
    *,
    routed_stacks: Sequence[RoutedStack] = (),
) -> CoverageVerdict:
    """Decide which label section 5.2 permits, and say why.

    Unlike :func:`require_census_equality` this does not raise on an
    incomplete proposal: an honest restricted pilot is a valid outcome, and
    downgrading its label is the correct handling rather than an error. It
    raises only when the inputs themselves are malformed.
    """
    reasons: list[str] = []
    try:
        covered = require_census_equality(memberships)
        equal = True
    except CoverageError as exc:
        covered = _as_set(memberships.get("census_units", ()), where="census_units")
        reasons.append(str(exc))
        equal = False

    gaps = routed_stack_gaps(routed_stacks)
    if gaps:
        reasons.append(
            f"{len(gaps)} routed stack(s) lack complete member evidence: "
            + _sample(gaps)
        )

    label = LABEL_WHOLE_MODEL if equal and not gaps else LABEL_RESTRICTED_PILOT
    if label == LABEL_WHOLE_MODEL:
        reasons.append(
            "all four memberships are one set and every routed stack carries "
            + ", ".join(REQUIRED_MEMBER_EVIDENCE)
            + " for every required member"
        )
    return CoverageVerdict(
        label=label,
        covered_units=tuple(sorted(covered)),
        routed_gaps=MappingProxyType(
            {
                unit: MappingProxyType(
                    {member: tuple(kinds) for member, kinds in members.items()}
                )
                for unit, members in gaps.items()
            }
        ),
        reasons=tuple(reasons),
    )
