"""Deterministic population selection and the mandatory-rate set.

Work package C of the Tessera quality-prefill experiment
(``docs/design/tessera_quality_prefill_experiment.md`` §5.1 and §5.3).

Three facts govern every draw in this module.

*The mechanism is a hash, not a generator.* A candidate's rank is the SHA-256
of canonical JSON over exactly five fields — ``source_sha256``,
``selection_seed``, ``purpose``, ``stratum_id`` and either the canonical
``unit_id`` or the integer ``rate``. Candidates sort by ``(digest,
canonical_id)`` and the draw takes the prefix. There is no RNG, no shuffle and
no tie-break by list position, so re-running the selection reproduces the
frozen lists byte for byte.

*An empty pool refuses.* §5.1 makes an empty pool "an explicit coverage gap
requiring a new plan; there is no automatic substitution from a convenient
depth". A refusal names the stratum, the purpose and the third, and leaves the
exposure ledger and every already-built list untouched: the selection is
assembled in locals and committed only when all six strata succeed.

*Names come from the profile.* The six strata are derived from the model
profile's own structure spec (routed per-expert regex, shared-expert fused
group, unpacked projection names) and the roster from the census counts keyed
by those patterns. Nothing here hardcodes a GLM layer range or leaf spelling.

Canonical ``unit_id`` namespace: the **source/census spelling**
(``model.language_model.layers.N.mlp.…``), because that is the namespace both
the census ``counts`` map and its ``expert_projection.stacks`` block already
use. The same spelling is the routed block id (``…layers.N.mlp.experts``).

The canonical JSON hasher is
:func:`prismaquant.cost_stage_checkpoint.canonical_json_sha256`; this module
adds no second canonicalizer.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import re

from prismaquant.cost_stage_checkpoint import canonical_json, canonical_json_sha256


# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

#: §5.4 initial screen policy.
SCREEN_POLICY_ID = "coverage_first_v1"

#: §5.1 purpose domains. Every draw states one.
PURPOSE_PILOT = "pilot"
PURPOSE_CONFIRMATION = "confirmation"
PURPOSE_RATE_AUDIT = "rate_audit"
PURPOSE_ROUTED_EXPERT_HIGH = "routed_expert_selection_high"
PURPOSE_ROUTED_EXPERT_LOW = "routed_expert_selection_low"
PURPOSE_DISCARDED_CANDIDATE_AUDIT = "discarded_candidate_audit"

PURPOSES = (
    PURPOSE_PILOT,
    PURPOSE_CONFIRMATION,
    PURPOSE_RATE_AUDIT,
    PURPOSE_ROUTED_EXPERT_HIGH,
    PURPOSE_ROUTED_EXPERT_LOW,
    PURPOSE_DISCARDED_CANDIDATE_AUDIT,
)

#: §5.1 canonical stratum order. The index is the ``i`` of ``i % 3``.
CANONICAL_STRATA_IDS = (
    "shared_gate",
    "shared_up",
    "shared_down",
    "routed_gate",
    "routed_up",
    "routed_down",
)

#: §5.1: "Exclude L10 and L20 shared-down from confirmation." Spec-given
#: layer indices, not a heuristic; the exclusion is applied to the
#: ``shared_down`` stratum's confirmation pool only.
EXCLUDED_CONFIRMATION_SHARED_DOWN_LAYERS = (10, 20)

#: §5.3: research audit strata partition at integer schedule transitions,
#: "multiples of 256 and immediate neighbors where legal".
SCHEDULE_STRATUM_STRIDE = 256

#: §5.3 interior draw size.
INTERIOR_DRAW_TARGET = 16

_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


class PopulationSelectionError(RuntimeError):
    """Refusal. Selection never substitutes a stratum, third or draw size."""


# --------------------------------------------------------------------------
# The selection hash
# --------------------------------------------------------------------------


def selection_payload(
    *,
    source_sha256: str,
    selection_seed: int,
    purpose: str,
    stratum_id: str,
    unit_id: str | None = None,
    rate: int | None = None,
) -> dict[str, object]:
    """Return the exact five-field object the selection digest hashes.

    Exactly one of ``unit_id`` / ``rate`` is present. No other field may
    enter: a quartile, a third index or a screen policy changes the draw's
    *meaning*, and is recorded beside the draw, never inside its hash.
    """
    if not isinstance(source_sha256, str) or not _SHA256_RE.match(source_sha256):
        raise PopulationSelectionError(
            "selection source_sha256 must be a lowercase 64-hex SHA-256 string"
        )
    if type(selection_seed) is not int:
        raise PopulationSelectionError("selection_seed must be an int")
    if purpose not in PURPOSES:
        raise PopulationSelectionError(
            f"unknown selection purpose {purpose!r}; known purposes are {PURPOSES}"
        )
    if not isinstance(stratum_id, str) or not stratum_id:
        raise PopulationSelectionError("stratum_id must be a non-empty string")
    if (unit_id is None) == (rate is None):
        raise PopulationSelectionError(
            "a selection payload carries exactly one of unit_id or rate"
        )

    payload: dict[str, object] = {
        "source_sha256": source_sha256,
        "selection_seed": selection_seed,
        "purpose": purpose,
        "stratum_id": stratum_id,
    }
    if unit_id is not None:
        if not isinstance(unit_id, str) or not unit_id:
            raise PopulationSelectionError("unit_id must be a non-empty string")
        payload["unit_id"] = unit_id
    else:
        if type(rate) is not int:
            raise PopulationSelectionError("rate must be an int")
        payload["rate"] = rate
    return payload


def selection_digest(**fields: object) -> str:
    """SHA-256 over UTF-8 canonical JSON of :func:`selection_payload`."""
    payload = selection_payload(**fields)  # type: ignore[arg-type]
    return canonical_json_sha256(payload, where="quality-prefill selection")


def stable_rank(
    candidates: Iterable[str] | Iterable[int],
    *,
    source_sha256: str,
    selection_seed: int,
    purpose: str,
    stratum_id: str,
) -> tuple[str, ...] | tuple[int, ...]:
    """Order candidates by ``(digest, canonical_id)``.

    ``candidates`` are either canonical unit ids (``str``) or legal rates
    (``int``); the two never mix in one call.
    """
    items = list(candidates)
    if not items:
        return ()
    kinds = {type(item) for item in items}
    if kinds == {str}:
        key = "unit_id"
    elif kinds == {int}:
        key = "rate"
    else:
        raise PopulationSelectionError(
            "stable_rank takes either unit ids or integer rates, never both"
        )
    if len(set(items)) != len(items):
        raise PopulationSelectionError(
            f"duplicate candidate in stratum {stratum_id!r} for purpose {purpose!r}"
        )

    ranked = sorted(
        items,
        key=lambda item: (
            selection_digest(
                source_sha256=source_sha256,
                selection_seed=selection_seed,
                purpose=purpose,
                stratum_id=stratum_id,
                **{key: item},
            ),
            item,
        ),
    )
    return tuple(ranked)


def _draw(
    candidates: Sequence[str] | Sequence[int],
    count: int,
    *,
    source_sha256: str,
    selection_seed: int,
    purpose: str,
    stratum_id: str,
    where: str,
) -> tuple:
    if not candidates:
        raise PopulationSelectionError(
            f"quality-prefill selection refuses: stratum {stratum_id!r} has an "
            f"empty candidate pool for purpose {purpose!r} ({where}). §5.1 makes "
            "this an explicit coverage gap requiring a new plan; there is no "
            "automatic substitution from another stratum, third or depth."
        )
    if count > len(candidates):
        raise PopulationSelectionError(
            f"quality-prefill selection refuses: stratum {stratum_id!r} needs "
            f"{count} candidates for purpose {purpose!r} but the pool holds "
            f"{len(candidates)} ({where}); a draw is never silently shortened."
        )
    ranked = stable_rank(
        candidates,
        source_sha256=source_sha256,
        selection_seed=selection_seed,
        purpose=purpose,
        stratum_id=stratum_id,
    )
    return tuple(ranked[:count])


# --------------------------------------------------------------------------
# Thirds
# --------------------------------------------------------------------------


def thirds(roster: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Split an ordered roster into three contiguous thirds.

    §5.1 fixes the cuts at ``floor(N/3)`` and ``floor(2*N/3)``; for ``N`` not
    divisible by three the remainder lands in the last third(s), which is what
    integer floor division already does.
    """
    ordered = tuple(roster)
    n = len(ordered)
    first = n // 3
    second = (2 * n) // 3
    return ordered[:first], ordered[first:second], ordered[second:]


def stratum_thirds(index: int) -> tuple[int, int]:
    """Return ``(pilot_third, confirmation_third)`` for stratum ``index``."""
    if type(index) is not int or index < 0:
        raise PopulationSelectionError("stratum index must be a non-negative int")
    return index % 3, (index + 1) % 3


# --------------------------------------------------------------------------
# Strata and roster
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityStratum:
    index: int
    stratum_id: str
    structure: str  # "shared" | "routed"
    projection: str  # profile leaf spelling, e.g. "gate_proj"


@dataclass(frozen=True)
class PopulationRoster:
    """The eligible population, derived from a profile and census counts."""

    layers: tuple[int, ...]
    strata: tuple[QualityStratum, ...]
    #: (layer, projection) -> canonical unit id
    shared_units: Mapping[tuple[int, str], str]
    #: layer -> canonical routed stack (block) id
    routed_stacks: Mapping[int, str]
    #: (layer, expert, projection) -> canonical unit id
    routed_units: Mapping[tuple[int, int, str], str]
    #: canonical unit id -> routing count from the census
    counts: Mapping[str, int]

    def stratum(self, stratum_id: str) -> QualityStratum:
        for item in self.strata:
            if item.stratum_id == stratum_id:
                return item
        raise PopulationSelectionError(f"unknown stratum {stratum_id!r}")


def _spec_of(profile) -> object:
    spec = profile.structure_spec()
    if spec is None:
        raise PopulationSelectionError(
            "the model profile declares no structure spec; the six quality "
            "strata cannot be derived from it"
        )
    return spec


def _body_layer_prefix(spec) -> str:
    prefix = getattr(spec, "body_layer_prefix", None)
    if not prefix:
        raise PopulationSelectionError(
            "the structure spec declares no body_layer_prefix; the eligible "
            "layer roster cannot be derived"
        )
    return str(prefix)


def _shared_expert_parent(spec) -> str:
    """Derive the shared-expert module suffix from the spec's fused groups."""
    for group in getattr(spec, "fused_groups", ()) or ():
        target = str(group.target_suffix)
        if "." not in target:
            continue
        parent = target.rsplit(".", 1)[0]
        if parent.rsplit(".", 1)[-1].startswith("shared_expert"):
            return parent
    raise PopulationSelectionError(
        "the structure spec declares no shared-expert fused group; the three "
        "shared strata cannot be derived from it"
    )


def _routed_pattern(profile, spec) -> re.Pattern[str]:
    raw = profile.per_expert_moe_regex()
    if not raw:
        raise PopulationSelectionError(
            "the model profile declares no per-expert MoE regex; the three "
            "routed strata cannot be derived from it"
        )
    text = str(raw)
    if text.startswith("re:"):
        text = text[3:]
    return re.compile(text)


def build_strata(profile) -> tuple[QualityStratum, ...]:
    """The six strata in §5.1's canonical order, with profile leaf names."""
    projections = tuple(profile.unpacked_expert_projection_names())
    by_role = {}
    for leaf in projections:
        role = leaf.split("_", 1)[0]
        by_role[role] = leaf
    missing = [role for role in ("gate", "up", "down") if role not in by_role]
    if missing:
        raise PopulationSelectionError(
            "the model profile does not expose gate/up/down projection names "
            f"(missing {missing}); the six quality strata cannot be derived"
        )
    strata = []
    for index, stratum_id in enumerate(CANONICAL_STRATA_IDS):
        structure, role = stratum_id.split("_", 1)
        strata.append(
            QualityStratum(
                index=index,
                stratum_id=stratum_id,
                structure=structure,
                projection=by_role[role],
            )
        )
    return tuple(strata)


def build_roster(*, counts: Mapping[str, int], profile) -> PopulationRoster:
    """Derive the eligible population from census counts and the profile.

    A layer is eligible only when it carries all three shared projections and
    at least one routed expert per routed projection, so the six strata share
    one ordered layer roster (§5.1's "ordered eligible layer roster").
    """
    spec = _spec_of(profile)
    strata = build_strata(profile)
    prefix = _body_layer_prefix(spec)
    shared_parent = _shared_expert_parent(spec)
    routed_pattern = _routed_pattern(profile, spec)
    layer_re = re.compile(re.escape(prefix) + r"\.(\d+)\.")

    projections = tuple(sorted({item.projection for item in strata}))

    shared_units: dict[tuple[int, str], str] = {}
    routed_units: dict[tuple[int, int, str], str] = {}
    routed_stacks: dict[int, str] = {}
    kept_counts: dict[str, int] = {}

    expert_re = re.compile(r"\.experts\.(\d+)\.")

    for qname, count in counts.items():
        match = layer_re.match(qname)
        if match is None:
            continue
        layer = int(match.group(1))
        suffix = qname[len(prefix) + 1 + len(match.group(1)) + 1 :]
        if suffix.startswith(shared_parent + "."):
            leaf = suffix.rsplit(".", 1)[-1]
            if leaf not in projections:
                continue
            shared_units[(layer, leaf)] = qname
            kept_counts[qname] = int(count)
            continue
        if routed_pattern.match(qname):
            expert_match = expert_re.search(qname)
            if expert_match is None:
                continue
            expert = int(expert_match.group(1))
            leaf = qname.rsplit(".", 1)[-1]
            if leaf not in projections:
                continue
            routed_units[(layer, expert, leaf)] = qname
            stack = qname[: expert_match.start()] + ".experts"
            routed_stacks.setdefault(layer, stack)
            kept_counts[qname] = int(count)

    eligible: list[int] = []
    for layer in sorted(set(routed_stacks) | {key[0] for key in shared_units}):
        has_shared = all((layer, leaf) in shared_units for leaf in projections)
        has_routed = all(
            any(key[0] == layer and key[2] == leaf for key in routed_units)
            for leaf in projections
        )
        if has_shared and has_routed:
            eligible.append(layer)

    if not eligible:
        raise PopulationSelectionError(
            "no eligible layer carries both a complete shared-expert triple "
            "and routed experts; the population is empty"
        )

    layers = tuple(eligible)
    shared_units = {key: value for key, value in shared_units.items() if key[0] in set(layers)}
    routed_units = {key: value for key, value in routed_units.items() if key[0] in set(layers)}
    routed_stacks = {key: value for key, value in routed_stacks.items() if key in set(layers)}
    kept_counts = {
        qname: kept_counts[qname]
        for qname in set(shared_units.values()) | set(routed_units.values())
    }

    return PopulationRoster(
        layers=layers,
        strata=strata,
        shared_units=dict(sorted(shared_units.items())),
        routed_stacks=dict(sorted(routed_stacks.items())),
        routed_units=dict(sorted(routed_units.items())),
        counts=dict(sorted(kept_counts.items())),
    )


# --------------------------------------------------------------------------
# Exposure ledger
# --------------------------------------------------------------------------


class ExposureLedger:
    """Per-unit record of the purposes a unit has been drawn for.

    §5.1 asks for the *complete historical exposure inventory*, so the ledger
    is seeded with prior exposures and then extended by this run's draws. A
    unit drawn for a pilot is visible as such when a later purpose draws.
    """

    def __init__(self, prior: Mapping[str, Sequence[str]] | None = None) -> None:
        self._records: dict[str, list[str]] = {}
        for unit_id, purposes in (prior or {}).items():
            for purpose in purposes:
                self.record(str(unit_id), str(purpose))

    def record(self, unit_id: str, purpose: str) -> None:
        entries = self._records.setdefault(unit_id, [])
        if purpose not in entries:
            entries.append(purpose)

    def purposes_for(self, unit_id: str) -> tuple[str, ...]:
        return tuple(self._records.get(unit_id, ()))

    def is_exposed(self, unit_id: str) -> bool:
        return bool(self._records.get(unit_id))

    def as_mapping(self) -> dict[str, tuple[str, ...]]:
        return {unit: tuple(purposes) for unit, purposes in sorted(self._records.items())}

    def copy(self) -> "ExposureLedger":
        clone = ExposureLedger()
        clone._records = {unit: list(purposes) for unit, purposes in self._records.items()}
        return clone


# --------------------------------------------------------------------------
# Quartiles
# --------------------------------------------------------------------------


def _quartile_pools(
    ranked_by_count: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Low and high nonzero-routing-count quartiles.

    Explicit integer rule (§5.1 says "quartile" and does not define it): with
    ``n`` nonzero-count members ordered by ``(count, unit_id)``, the low pool
    is the first ``n // 4`` ranks and the high pool the last ``n // 4``. The
    pools are symmetric by construction and disjoint whenever ``n >= 2``.
    """
    n = len(ranked_by_count)
    if n < 2:
        # One member cannot supply two disjoint extremes; both pools are
        # empty so the draw refuses and names the stratum.
        return (), ()
    width = max(1, n // 4)
    low = tuple(ranked_by_count[:width])
    high = tuple(ranked_by_count[n - width :])
    return low, high


# --------------------------------------------------------------------------
# Population selection
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class UnitDraw:
    unit_id: str
    purpose: str
    stratum_id: str
    layer: int
    block_id: str
    routing_count: int
    quartile: str | None
    previously_exposed: bool


@dataclass(frozen=True)
class StratumSelection:
    stratum_id: str
    index: int
    pilot_third: int
    confirmation_third: int
    pilot: tuple[UnitDraw, ...]
    confirmation: tuple[UnitDraw, ...]

    @property
    def pilot_unit_ids(self) -> tuple[str, ...]:
        return tuple(draw.unit_id for draw in self.pilot)

    @property
    def confirmation_unit_ids(self) -> tuple[str, ...]:
        return tuple(draw.unit_id for draw in self.confirmation)


@dataclass(frozen=True)
class PopulationSelection:
    source_sha256: str
    selection_seed: int
    screen_policy_id: str
    layers: tuple[int, ...]
    third_layers: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    strata: tuple[StratumSelection, ...]
    exposure: Mapping[str, tuple[str, ...]]

    def stratum(self, stratum_id: str) -> StratumSelection:
        for item in self.strata:
            if item.stratum_id == stratum_id:
                return item
        raise PopulationSelectionError(f"unknown stratum {stratum_id!r}")


def _shared_candidates(
    roster: PopulationRoster,
    stratum: QualityStratum,
    layers: Sequence[int],
    *,
    excluded: frozenset[str],
) -> tuple[str, ...]:
    out = []
    for layer in layers:
        unit_id = roster.shared_units.get((layer, stratum.projection))
        if unit_id is None or unit_id in excluded:
            continue
        out.append(unit_id)
    return tuple(out)


def _prefer_unexposed(
    candidates: Sequence[str], ledger: ExposureLedger
) -> tuple[tuple[str, ...], bool]:
    """Prefer wholly unexposed units; fall back to the exposed pool.

    Returns ``(pool, overlap)``. ``overlap`` is True when every candidate was
    already exposed, which §5.1 requires reporting rather than hiding.
    """
    fresh = tuple(unit for unit in candidates if not ledger.is_exposed(unit))
    if fresh:
        return fresh, False
    return tuple(candidates), bool(candidates)


def _select_routed(
    roster: PopulationRoster,
    stratum: QualityStratum,
    layers: Sequence[int],
    *,
    purpose: str,
    source_sha256: str,
    selection_seed: int,
    ledger: ExposureLedger,
    where: str,
) -> tuple[UnitDraw, ...]:
    blocks = tuple(
        roster.routed_stacks[layer] for layer in layers if layer in roster.routed_stacks
    )
    (block_id,) = _draw(
        blocks,
        1,
        source_sha256=source_sha256,
        selection_seed=selection_seed,
        purpose=purpose,
        stratum_id=stratum.stratum_id,
        where=f"{where}; routed block pool",
    )
    layer = next(
        layer for layer, stack in roster.routed_stacks.items() if stack == block_id
    )

    members = [
        unit_id
        for (unit_layer, _expert, leaf), unit_id in roster.routed_units.items()
        if unit_layer == layer and leaf == stratum.projection
    ]
    nonzero = [unit for unit in members if roster.counts.get(unit, 0) > 0]
    ordered = sorted(nonzero, key=lambda unit: (roster.counts[unit], unit))
    low_pool, high_pool = _quartile_pools(ordered)

    draws: list[UnitDraw] = []
    for quartile, pool, member_purpose in (
        ("high", high_pool, PURPOSE_ROUTED_EXPERT_HIGH),
        ("low", low_pool, PURPOSE_ROUTED_EXPERT_LOW),
    ):
        eligible, overlap = _prefer_unexposed(pool, ledger)
        (unit_id,) = _draw(
            eligible,
            1,
            source_sha256=source_sha256,
            selection_seed=selection_seed,
            purpose=member_purpose,
            stratum_id=stratum.stratum_id,
            where=f"{where}; {quartile} routing-count quartile of {block_id}",
        )
        draws.append(
            UnitDraw(
                unit_id=unit_id,
                purpose=purpose,
                stratum_id=stratum.stratum_id,
                layer=layer,
                block_id=block_id,
                routing_count=int(roster.counts[unit_id]),
                quartile=quartile,
                previously_exposed=overlap or ledger.is_exposed(unit_id),
            )
        )
    return tuple(draws)


def _select_shared(
    roster: PopulationRoster,
    stratum: QualityStratum,
    layers: Sequence[int],
    *,
    purpose: str,
    source_sha256: str,
    selection_seed: int,
    ledger: ExposureLedger,
    excluded: frozenset[str],
    where: str,
) -> tuple[UnitDraw, ...]:
    candidates = _shared_candidates(roster, stratum, layers, excluded=excluded)
    eligible, overlap = _prefer_unexposed(candidates, ledger)
    (unit_id,) = _draw(
        eligible,
        1,
        source_sha256=source_sha256,
        selection_seed=selection_seed,
        purpose=purpose,
        stratum_id=stratum.stratum_id,
        where=where,
    )
    layer = next(
        key[0] for key, value in roster.shared_units.items() if value == unit_id
    )
    return (
        UnitDraw(
            unit_id=unit_id,
            purpose=purpose,
            stratum_id=stratum.stratum_id,
            layer=layer,
            block_id=unit_id.rsplit(".", 1)[0],
            routing_count=int(roster.counts[unit_id]),
            quartile=None,
            previously_exposed=overlap or ledger.is_exposed(unit_id),
        ),
    )


def confirmation_exclusions(roster: PopulationRoster) -> frozenset[str]:
    """The §5.1 named exclusions: L10 and L20 shared-down."""
    stratum = roster.stratum("shared_down")
    out = set()
    for layer in EXCLUDED_CONFIRMATION_SHARED_DOWN_LAYERS:
        unit_id = roster.shared_units.get((layer, stratum.projection))
        if unit_id is not None:
            out.add(unit_id)
    return frozenset(out)


def select_population(
    *,
    roster: PopulationRoster,
    source_sha256: str,
    selection_seed: int = 0,
    prior_exposure: Mapping[str, Sequence[str]] | None = None,
    extra_confirmation_exclusions: Iterable[str] = (),
) -> PopulationSelection:
    """Select the pilot and confirmation unit of each of the six strata.

    The whole selection is assembled in locals; the exposure ledger the caller
    sees is published only after all six strata have succeeded, so a refusal
    substitutes nothing and leaves no partial state.
    """
    working = ExposureLedger(prior_exposure)
    excluded = confirmation_exclusions(roster) | frozenset(
        str(unit) for unit in extra_confirmation_exclusions
    )
    pools = thirds(roster.layers)

    selections: list[StratumSelection] = []
    for stratum in roster.strata:
        pilot_third, confirmation_third = stratum_thirds(stratum.index)
        per_purpose: dict[str, tuple[UnitDraw, ...]] = {}
        for purpose, third_index in (
            (PURPOSE_PILOT, pilot_third),
            (PURPOSE_CONFIRMATION, confirmation_third),
        ):
            layers = pools[third_index]
            where = f"third {third_index} layers={list(layers)}"
            if stratum.structure == "routed":
                draws = _select_routed(
                    roster,
                    stratum,
                    layers,
                    purpose=purpose,
                    source_sha256=source_sha256,
                    selection_seed=selection_seed,
                    ledger=working,
                    where=where,
                )
            else:
                draws = _select_shared(
                    roster,
                    stratum,
                    layers,
                    purpose=purpose,
                    source_sha256=source_sha256,
                    selection_seed=selection_seed,
                    ledger=working,
                    excluded=excluded if purpose == PURPOSE_CONFIRMATION else frozenset(),
                    where=where,
                )
            for draw in draws:
                working.record(draw.unit_id, purpose)
                if draw.quartile is not None:
                    working.record(
                        draw.unit_id,
                        PURPOSE_ROUTED_EXPERT_HIGH
                        if draw.quartile == "high"
                        else PURPOSE_ROUTED_EXPERT_LOW,
                    )
            per_purpose[purpose] = draws

        pilot_ids = {draw.unit_id for draw in per_purpose[PURPOSE_PILOT]}
        confirmation_ids = {draw.unit_id for draw in per_purpose[PURPOSE_CONFIRMATION]}
        overlap = sorted(pilot_ids & confirmation_ids)
        if overlap:
            raise PopulationSelectionError(
                f"quality-prefill selection refuses: stratum "
                f"{stratum.stratum_id!r} drew the same unit for pilot and "
                f"confirmation ({overlap}); §5.1 requires the ids to differ."
            )
        selections.append(
            StratumSelection(
                stratum_id=stratum.stratum_id,
                index=stratum.index,
                pilot_third=pilot_third,
                confirmation_third=confirmation_third,
                pilot=per_purpose[PURPOSE_PILOT],
                confirmation=per_purpose[PURPOSE_CONFIRMATION],
            )
        )

    return PopulationSelection(
        source_sha256=source_sha256,
        selection_seed=int(selection_seed),
        screen_policy_id=SCREEN_POLICY_ID,
        layers=roster.layers,
        third_layers=pools,
        strata=tuple(selections),
        exposure=working.as_mapping(),
    )


def population_freeze_payload(selection: PopulationSelection) -> dict[str, object]:
    """The frozen, canonical-JSON-safe form of a selection."""
    return canonical_json(
        {
            "schema": "prismaquant.quality_prefill_population.v1",
            "screen_policy_id": selection.screen_policy_id,
            "source_sha256": selection.source_sha256,
            "selection_seed": selection.selection_seed,
            "layers": list(selection.layers),
            "thirds": [list(third) for third in selection.third_layers],
            "strata": [
                {
                    "stratum_id": stratum.stratum_id,
                    "index": stratum.index,
                    "pilot_third": stratum.pilot_third,
                    "confirmation_third": stratum.confirmation_third,
                    "pilot": [_draw_payload(draw) for draw in stratum.pilot],
                    "confirmation": [
                        _draw_payload(draw) for draw in stratum.confirmation
                    ],
                }
                for stratum in selection.strata
            ],
            "exposure": {
                unit: list(purposes) for unit, purposes in selection.exposure.items()
            },
        },
        where="quality-prefill population freeze",
    )


def _draw_payload(draw: UnitDraw) -> dict[str, object]:
    return {
        "unit_id": draw.unit_id,
        "purpose": draw.purpose,
        "stratum_id": draw.stratum_id,
        "layer": draw.layer,
        "block_id": draw.block_id,
        "routing_count": draw.routing_count,
        "quartile": draw.quartile,
        "previously_exposed": draw.previously_exposed,
    }


def population_freeze_digest(selection: PopulationSelection) -> str:
    return canonical_json_sha256(
        population_freeze_payload(selection),
        where="quality-prefill population freeze",
    )


# --------------------------------------------------------------------------
# §5.3 mandatory-rate set
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RateDomain:
    """The legal rate domain of one family: the single definition of it.

    Work package A owns the *derivation*: ``tessera_legal_domain.
    legal_rate_domain`` walks the frozen producer grammar and packaged reader
    contract (§2.1) and returns this class. Work package C consumes it
    through :data:`LegalRateDomainProvider` and never re-derives it. Owning the
    derivation is not owning the type, so the type lives here, in the package
    that has no Tessera dependency: ``tessera_formats`` refuses to import
    without the ``tessera`` package, and a duplicate class in A was the price
    of keeping C importable without it. A imports this one instead.

    ``rates`` and ``transition_rates`` are tuples, refused when they are not.
    A domain rebuilt from JSON arrives carrying lists, and a list-valued copy
    compares unequal to the derived domain while passing every other check —
    a difference that would surface as a silent mismatch rather than a
    refusal.
    """

    family: str
    rates: tuple[int, ...]
    transition_rates: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("rates", "transition_rates"):
            if type(getattr(self, name)) is not tuple:
                raise PopulationSelectionError(
                    f"rate domain for family {self.family!r} must carry "
                    f"{name} as a tuple, not a "
                    f"{type(getattr(self, name)).__name__}"
                )
        if not self.rates:
            raise PopulationSelectionError(
                f"rate domain for family {self.family!r} is empty"
            )
        if tuple(sorted(set(self.rates))) != tuple(self.rates):
            raise PopulationSelectionError(
                f"rate domain for family {self.family!r} must be sorted and unique"
            )
        legal = set(self.rates)
        stray = sorted(rate for rate in self.transition_rates if rate not in legal)
        if stray:
            raise PopulationSelectionError(
                f"family {self.family!r} declares transition rates outside its "
                f"legal domain: {stray}"
            )


#: The narrow interface to work package A.
LegalRateDomainProvider = Callable[[str], RateDomain]


def mandatory_rates(domain: RateDomain) -> tuple[int, ...]:
    """Resolver transitions, their legal immediate neighbours, domain ends.

    "Immediate neighbour" is the previous and next *legal* rate in the sorted
    domain, not ``rate ± 1``: a rate one integer away need not be legal.
    """
    rates = domain.rates
    index = {rate: position for position, rate in enumerate(rates)}
    out = {rates[0], rates[-1]}
    for rate in domain.transition_rates:
        position = index[rate]
        out.add(rate)
        if position > 0:
            out.add(rates[position - 1])
        if position + 1 < len(rates):
            out.add(rates[position + 1])
    return tuple(sorted(out))


def interior_boundaries(domain: RateDomain) -> tuple[int, ...]:
    """Audit-stratum boundaries: resolver transitions and 256-multiples.

    §5.3 partitions the research audit strata "at integer schedule transitions
    (multiples of 256 and immediate neighbors where legal)". The 256-multiple
    half is arithmetic over the legal domain, so work package C computes it;
    work package A supplies only the resolver transitions.
    """
    legal = set(domain.rates)
    boundaries = {domain.rates[0], domain.rates[-1]}
    boundaries.update(domain.transition_rates)
    for rate in domain.rates:
        if rate % SCHEDULE_STRATUM_STRIDE == 0 and rate in legal:
            boundaries.add(rate)
    return tuple(sorted(boundaries))


@dataclass(frozen=True)
class RateStratumDraw:
    stratum_id: str
    low: int
    high: int
    remaining_legal_rate_count: int
    drawn: tuple[int, ...]


@dataclass(frozen=True)
class MandatoryRateSet:
    family: str
    screen_policy_id: str
    source_sha256: str
    selection_seed: int
    mandatory: tuple[int, ...]
    strata: tuple[RateStratumDraw, ...]

    @property
    def roster(self) -> tuple[int, ...]:
        out = set(self.mandatory)
        for stratum in self.strata:
            out.update(stratum.drawn)
        return tuple(sorted(out))


def rate_stratum_id(family: str, low: int, high: int) -> str:
    return f"{family}:interior:{low}:{high}"


def build_mandatory_rate_set(
    *,
    domain: RateDomain,
    source_sha256: str,
    selection_seed: int = 0,
) -> MandatoryRateSet:
    """Construct the §5.3 mandatory set plus the per-stratum interior draw.

    Interior strata are the half-open segments between consecutive audit
    boundaries (the last one closed). Each draws exactly
    ``min(16, remaining_legal_rate_count)`` rates by the §5.1 stable rank after
    the mandatory points are removed.
    """
    mandatory = mandatory_rates(domain)
    mandatory_set = set(mandatory)
    boundaries = interior_boundaries(domain)

    strata: list[RateStratumDraw] = []
    for position in range(len(boundaries) - 1):
        low = boundaries[position]
        high = boundaries[position + 1]
        is_last = position == len(boundaries) - 2
        if is_last:
            segment = [rate for rate in domain.rates if low <= rate <= high]
        else:
            segment = [rate for rate in domain.rates if low <= rate < high]
        remaining = [rate for rate in segment if rate not in mandatory_set]
        stratum_id = rate_stratum_id(domain.family, low, high)
        count = min(INTERIOR_DRAW_TARGET, len(remaining))
        drawn = (
            _draw(
                remaining,
                count,
                source_sha256=source_sha256,
                selection_seed=selection_seed,
                purpose=PURPOSE_RATE_AUDIT,
                stratum_id=stratum_id,
                where=f"interior rate stratum [{low}, {high}]",
            )
            if count
            else ()
        )
        strata.append(
            RateStratumDraw(
                stratum_id=stratum_id,
                low=low,
                high=high,
                remaining_legal_rate_count=len(remaining),
                drawn=tuple(sorted(drawn)),
            )
        )

    return MandatoryRateSet(
        family=domain.family,
        screen_policy_id=SCREEN_POLICY_ID,
        source_sha256=source_sha256,
        selection_seed=int(selection_seed),
        mandatory=mandatory,
        strata=tuple(strata),
    )


def rate_set_freeze_payload(rate_set: MandatoryRateSet) -> dict[str, object]:
    return canonical_json(
        {
            "schema": "prismaquant.quality_prefill_rate_set.v1",
            "family": rate_set.family,
            "screen_policy_id": rate_set.screen_policy_id,
            "source_sha256": rate_set.source_sha256,
            "selection_seed": rate_set.selection_seed,
            "mandatory": list(rate_set.mandatory),
            "strata": [
                {
                    "stratum_id": stratum.stratum_id,
                    "low": stratum.low,
                    "high": stratum.high,
                    "remaining_legal_rate_count": stratum.remaining_legal_rate_count,
                    "drawn": list(stratum.drawn),
                }
                for stratum in rate_set.strata
            ],
            "roster": list(rate_set.roster),
        },
        where="quality-prefill rate set freeze",
    )


def rate_set_freeze_digest(rate_set: MandatoryRateSet) -> str:
    return canonical_json_sha256(
        rate_set_freeze_payload(rate_set), where="quality-prefill rate set freeze"
    )


# --------------------------------------------------------------------------
# Census intake
# --------------------------------------------------------------------------


def load_census_counts(path, *, expected_sha256: str) -> dict[str, int]:
    """Read the frozen census ``counts`` map after verifying its bytes.

    The census file is frozen campaign evidence; its SHA-256 is published in
    the run's census audit record. Bytes are hashed before the JSON is
    trusted, so a silently edited or truncated census refuses.
    """
    import hashlib
    import json
    from pathlib import Path

    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise PopulationSelectionError(
            f"census at {path} hashes to {digest}, not the expected "
            f"{expected_sha256}; refusing to derive a population from it"
        )
    document = json.loads(payload)
    counts = document.get("counts")
    if not isinstance(counts, Mapping) or not counts:
        raise PopulationSelectionError(f"census at {path} carries no counts map")
    return {str(key): int(value) for key, value in counts.items()}
