"""Acceptance tests for deterministic quality-prefill population selection.

Every test runs on a synthetic counts map so no test depends on the frozen
campaign census under ``/mnt/shared``. One optional test reads the real census
when it is mounted; its skip is visible in the PB log and certifies nothing on
its own.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prismaquant.quality_prefill_population import (
    CANONICAL_STRATA_IDS,
    EXCLUDED_CONFIRMATION_SHARED_DOWN_LAYERS,
    INTERIOR_DRAW_TARGET,
    PURPOSE_PILOT,
    PURPOSE_RATE_AUDIT,
    SCREEN_POLICY_ID,
    ExposureLedger,
    PopulationSelectionError,
    RateDomain,
    build_mandatory_rate_set,
    build_roster,
    confirmation_exclusions,
    interior_boundaries,
    load_census_counts,
    mandatory_rates,
    population_freeze_digest,
    population_freeze_payload,
    select_population,
    selection_digest,
    selection_payload,
    stable_rank,
    stratum_thirds,
    thirds,
)

SOURCE_SHA = "a" * 64
OTHER_SOURCE_SHA = "b" * 64

PREFIX = "model.language_model.layers"


def _profile():
    """The real GLM-5.3 profile object; no checkpoint or vLLM import needed."""
    from prismaquant.model_profiles.glm5_next import Glm5NextProfile

    return Glm5NextProfile()


def synthetic_counts(layers, *, experts: int = 8) -> dict[str, int]:
    """Census-shaped counts in the source spelling the real census uses."""
    counts: dict[str, int] = {}
    for layer in layers:
        for leaf in ("gate_proj", "up_proj", "down_proj"):
            counts[f"{PREFIX}.{layer}.mlp.shared_experts.{leaf}"] = 262144
        for expert in range(experts):
            # A spread of routing counts so the quartiles are well separated.
            count = 13 + (expert * 101 + layer * 7) % 4096
            for leaf in ("gate_proj", "up_proj", "down_proj"):
                counts[f"{PREFIX}.{layer}.mlp.experts.{expert}.{leaf}"] = count
    # Dense (non-MoE) MLP layers and the passthrough MTP block must not enter
    # the roster; include them so the filter is actually exercised.
    for layer in (0, 1, 2):
        for leaf in ("gate_proj", "up_proj", "down_proj"):
            counts[f"{PREFIX}.{layer}.mlp.{leaf}"] = 262144
    return counts


@pytest.fixture
def full_roster():
    layers = tuple(range(3, 45))
    return build_roster(counts=synthetic_counts(layers), profile=_profile())


# ---------------------------------------------------------------- roster ---


def test_roster_is_derived_from_the_profile_not_a_hardcoded_list(full_roster):
    assert full_roster.layers == tuple(range(3, 45))
    assert [item.stratum_id for item in full_roster.strata] == list(CANONICAL_STRATA_IDS)
    # Dense layers 0-2 carry mlp.{gate,up,down}_proj but no shared/routed
    # experts, so they are absent from the eligible roster.
    assert 0 not in full_roster.layers
    assert full_roster.shared_units[(10, "down_proj")] == (
        f"{PREFIX}.10.mlp.shared_experts.down_proj"
    )
    assert full_roster.routed_stacks[10] == f"{PREFIX}.10.mlp.experts"


# --------------------------------------------------- criterion 5: thirds ---


@pytest.mark.parametrize("n", [3, 4, 5, 9, 10, 11, 42, 100])
def test_thirds_cut_at_floor_n_over_3(n):
    roster = tuple(range(n))
    first, second, third = thirds(roster)
    assert len(first) == n // 3
    assert len(first) + len(second) == (2 * n) // 3
    assert first + second + third == roster
    # Contiguous, disjoint, complete.
    assert set(first) | set(second) | set(third) == set(roster)
    assert not (set(first) & set(second)) and not (set(second) & set(third))


def test_stratum_i_uses_thirds_i_mod_3_and_i_plus_1_mod_3():
    assert [stratum_thirds(i) for i in range(6)] == [
        (0, 1),
        (1, 2),
        (2, 0),
        (0, 1),
        (1, 2),
        (2, 0),
    ]


def test_selection_records_the_thirds_it_used(full_roster):
    selection = select_population(roster=full_roster, source_sha256=SOURCE_SHA)
    for stratum in selection.strata:
        assert (stratum.pilot_third, stratum.confirmation_third) == stratum_thirds(
            stratum.index
        )
        pilot_layers = selection.third_layers[stratum.pilot_third]
        confirmation_layers = selection.third_layers[stratum.confirmation_third]
        for draw in stratum.pilot:
            assert draw.layer in pilot_layers
        for draw in stratum.confirmation:
            assert draw.layer in confirmation_layers


# ------------------------------------------- criterion 1: reproducibility ---


def test_selection_reproduces_byte_for_byte(full_roster):
    one = select_population(roster=full_roster, source_sha256=SOURCE_SHA)
    two = select_population(roster=full_roster, source_sha256=SOURCE_SHA)
    encode = lambda value: json.dumps(  # noqa: E731
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    assert encode(population_freeze_payload(one)) == encode(
        population_freeze_payload(two)
    )
    assert population_freeze_digest(one) == population_freeze_digest(two)
    # And a roster rebuilt from the same counts reproduces it too.
    rebuilt = build_roster(
        counts=synthetic_counts(tuple(range(3, 45))), profile=_profile()
    )
    three = select_population(roster=rebuilt, source_sha256=SOURCE_SHA)
    assert encode(population_freeze_payload(three)) == encode(
        population_freeze_payload(one)
    )


def test_rate_set_reproduces_byte_for_byte():
    domain = RateDomain(
        family="E4M3_K1", rates=tuple(range(256, 1281)), transition_rates=(600, 601)
    )
    one = build_mandatory_rate_set(domain=domain, source_sha256=SOURCE_SHA)
    two = build_mandatory_rate_set(domain=domain, source_sha256=SOURCE_SHA)
    assert one == two
    assert one.screen_policy_id == SCREEN_POLICY_ID


# ---------------------------------------------- criterion 2: disjointness ---


def test_pilot_and_confirmation_are_disjoint_for_all_six_strata(full_roster):
    selection = select_population(roster=full_roster, source_sha256=SOURCE_SHA)
    assert len(selection.strata) == 6
    for stratum in selection.strata:
        pilot = set(stratum.pilot_unit_ids)
        confirmation = set(stratum.confirmation_unit_ids)
        assert pilot and confirmation
        assert not (pilot & confirmation), stratum.stratum_id


# --------------------------------------------- criterion 3: L10/L20 rule ---


def test_l10_and_l20_shared_down_never_appear_in_a_confirmation_set(full_roster):
    excluded = confirmation_exclusions(full_roster)
    assert excluded == {
        f"{PREFIX}.{layer}.mlp.shared_experts.down_proj"
        for layer in EXCLUDED_CONFIRMATION_SHARED_DOWN_LAYERS
    }
    for seed in range(50):
        selection = select_population(
            roster=full_roster, source_sha256=SOURCE_SHA, selection_seed=seed
        )
        for stratum in selection.strata:
            assert not (set(stratum.confirmation_unit_ids) & excluded), (
                stratum.stratum_id,
                seed,
            )


def test_the_l10_l20_exclusion_actually_bites():
    """Without the exclusion, an excluded unit would win some seed's draw."""
    roster = build_roster(
        counts=synthetic_counts(tuple(range(3, 45))), profile=_profile()
    )
    excluded = confirmation_exclusions(roster)
    stratum = roster.stratum("shared_down")
    _pilot_third, confirmation_third = stratum_thirds(stratum.index)
    pool = [
        roster.shared_units[(layer, stratum.projection)]
        for layer in thirds(roster.layers)[confirmation_third]
    ]
    assert excluded & set(pool), "the fixture must put an excluded unit in the pool"
    would_have_won = {
        stable_rank(
            pool,
            source_sha256=SOURCE_SHA,
            selection_seed=seed,
            purpose="confirmation",
            stratum_id=stratum.stratum_id,
        )[0]
        for seed in range(50)
    }
    assert would_have_won & excluded, (
        "no seed in 0..49 would have drawn an excluded unit; the exclusion test "
        "would pass vacuously"
    )


# ------------------------------------------------- criterion 4: refusal ---


def test_an_empty_pool_refuses_and_substitutes_nothing():
    # Two eligible layers: thirds are ((), (l0,), (l1,)), so stratum 0's
    # pilot third is empty.
    roster = build_roster(counts=synthetic_counts((3, 4)), profile=_profile())
    assert thirds(roster.layers)[0] == ()
    prior = {f"{PREFIX}.3.mlp.shared_experts.gate_proj": ["historical_exposure"]}
    prior_snapshot = json.dumps(prior, sort_keys=True)

    with pytest.raises(PopulationSelectionError) as excinfo:
        select_population(
            roster=roster,
            source_sha256=SOURCE_SHA,
            prior_exposure=prior,
        )

    message = str(excinfo.value)
    assert "shared_gate" in message
    assert PURPOSE_PILOT in message
    assert "third 0" in message
    assert "no automatic substitution" in message
    # Nothing was substituted and no caller state was mutated.
    assert json.dumps(prior, sort_keys=True) == prior_snapshot


def test_refusal_never_shortens_a_draw():
    roster = build_roster(counts=synthetic_counts((3, 4)), profile=_profile())
    with pytest.raises(PopulationSelectionError):
        select_population(roster=roster, source_sha256=SOURCE_SHA)


# --------------------------------------- criterion 6: min(16, remaining) ---


def test_interior_draw_is_min_16_when_more_than_16_remain():
    domain = RateDomain(
        family="E4M3_K1", rates=tuple(range(256, 1281)), transition_rates=(600, 601)
    )
    rate_set = build_mandatory_rate_set(domain=domain, source_sha256=SOURCE_SHA)
    big = [s for s in rate_set.strata if s.remaining_legal_rate_count > INTERIOR_DRAW_TARGET]
    assert big
    for stratum in big:
        assert len(stratum.drawn) == INTERIOR_DRAW_TARGET


def test_interior_draw_is_remaining_when_fewer_than_16_remain():
    domain = RateDomain(
        family="BF16_K1", rates=tuple(range(256, 272)), transition_rates=(260,)
    )
    rate_set = build_mandatory_rate_set(domain=domain, source_sha256=SOURCE_SHA)
    assert rate_set.mandatory == (256, 259, 260, 261, 271)
    small = [
        s for s in rate_set.strata if s.remaining_legal_rate_count < INTERIOR_DRAW_TARGET
    ]
    assert small
    for stratum in small:
        assert len(stratum.drawn) == stratum.remaining_legal_rate_count
        assert not set(stratum.drawn) & set(rate_set.mandatory)


def test_mandatory_neighbours_are_legal_not_plus_or_minus_one():
    # 600 and 602 are legal; 601 is not, so 600's next legal neighbour is 602.
    rates = tuple(r for r in range(256, 1025) if r != 601)
    domain = RateDomain(family="E4M3_K1", rates=rates, transition_rates=(600,))
    assert mandatory_rates(domain) == (256, 599, 600, 602, 1024)


def test_interior_boundaries_include_transitions_and_256_multiples():
    domain = RateDomain(
        family="E4M3_K1", rates=tuple(range(256, 1281)), transition_rates=(600, 601)
    )
    assert interior_boundaries(domain) == (256, 512, 600, 601, 768, 1024, 1280)


def test_rate_domain_refuses_a_transition_outside_the_legal_domain():
    with pytest.raises(PopulationSelectionError):
        RateDomain(family="E4M3_K1", rates=(256, 257), transition_rates=(999,))


@pytest.mark.parametrize("field", ["rates", "transition_rates"])
def test_rate_domain_refuses_a_list_where_a_tuple_belongs(field):
    """A domain rebuilt from JSON carries lists and would compare unequal.

    Every other check passes on a list-valued copy -- sorted, unique, no stray
    transition -- so without this refusal the mismatch shows up as a silent
    inequality against the derived domain rather than as an error.
    """
    fields = {
        "family": "E4M3_K1", "rates": (256, 257), "transition_rates": (257,),
    }
    fields[field] = list(fields[field])
    with pytest.raises(PopulationSelectionError, match="tuple"):
        RateDomain(**fields)


# ------------------------------------------- criterion 7: the hash input ---


def test_selection_payload_is_exactly_the_five_named_fields():
    unit_payload = selection_payload(
        source_sha256=SOURCE_SHA,
        selection_seed=0,
        purpose=PURPOSE_PILOT,
        stratum_id="shared_gate",
        unit_id="u",
    )
    assert set(unit_payload) == {
        "source_sha256",
        "selection_seed",
        "purpose",
        "stratum_id",
        "unit_id",
    }
    rate_payload = selection_payload(
        source_sha256=SOURCE_SHA,
        selection_seed=0,
        purpose=PURPOSE_RATE_AUDIT,
        stratum_id="E4M3_K1:interior:256:512",
        rate=300,
    )
    assert set(rate_payload) == {
        "source_sha256",
        "selection_seed",
        "purpose",
        "stratum_id",
        "rate",
    }
    assert type(rate_payload["rate"]) is int
    assert type(rate_payload["selection_seed"]) is int


def test_selection_payload_refuses_both_or_neither_of_unit_id_and_rate():
    for kwargs in ({}, {"unit_id": "u", "rate": 3}):
        with pytest.raises(PopulationSelectionError):
            selection_payload(
                source_sha256=SOURCE_SHA,
                selection_seed=0,
                purpose=PURPOSE_PILOT,
                stratum_id="shared_gate",
                **kwargs,
            )


def test_digest_is_sha256_of_the_canonical_encoding():
    import hashlib

    payload = selection_payload(
        source_sha256=SOURCE_SHA,
        selection_seed=0,
        purpose=PURPOSE_PILOT,
        stratum_id="shared_gate",
        unit_id="u",
    )
    expected = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert (
        selection_digest(
            source_sha256=SOURCE_SHA,
            selection_seed=0,
            purpose=PURPOSE_PILOT,
            stratum_id="shared_gate",
            unit_id="u",
        )
        == expected
    )


def test_changing_the_seed_changes_the_selection(full_roster):
    zero = select_population(roster=full_roster, source_sha256=SOURCE_SHA, selection_seed=0)
    one = select_population(roster=full_roster, source_sha256=SOURCE_SHA, selection_seed=1)
    assert population_freeze_digest(zero) != population_freeze_digest(one)
    picks_zero = [stratum.pilot_unit_ids for stratum in zero.strata]
    picks_one = [stratum.pilot_unit_ids for stratum in one.strata]
    assert picks_zero != picks_one


def test_changing_the_source_sha_changes_the_selection(full_roster):
    zero = select_population(roster=full_roster, source_sha256=SOURCE_SHA)
    other = select_population(roster=full_roster, source_sha256=OTHER_SOURCE_SHA)
    assert population_freeze_digest(zero) != population_freeze_digest(other)


def test_stable_rank_is_by_digest_then_canonical_id():
    pool = ["u1", "u2", "u3", "u4", "u5"]
    ranked = stable_rank(
        pool,
        source_sha256=SOURCE_SHA,
        selection_seed=0,
        purpose=PURPOSE_PILOT,
        stratum_id="shared_gate",
    )
    keys = [
        (
            selection_digest(
                source_sha256=SOURCE_SHA,
                selection_seed=0,
                purpose=PURPOSE_PILOT,
                stratum_id="shared_gate",
                unit_id=unit,
            ),
            unit,
        )
        for unit in ranked
    ]
    assert keys == sorted(keys)
    # List order does not enter the rank.
    assert (
        stable_rank(
            list(reversed(pool)),
            source_sha256=SOURCE_SHA,
            selection_seed=0,
            purpose=PURPOSE_PILOT,
            stratum_id="shared_gate",
        )
        == ranked
    )


# ------------------------------------------------------ exposure ledger ---


def test_exposure_ledger_shows_a_pilot_unit_to_a_later_purpose(full_roster):
    selection = select_population(roster=full_roster, source_sha256=SOURCE_SHA)
    for stratum in selection.strata:
        for draw in stratum.pilot:
            assert PURPOSE_PILOT in selection.exposure[draw.unit_id]
        for draw in stratum.confirmation:
            assert "confirmation" in selection.exposure[draw.unit_id]


def test_prior_exposure_is_visible_and_unexposed_units_are_preferred(full_roster):
    stratum = full_roster.stratum("shared_gate")
    pilot_third = thirds(full_roster.layers)[stratum_thirds(stratum.index)[0]]
    pool = [
        full_roster.shared_units[(layer, stratum.projection)] for layer in pilot_third
    ]
    baseline = select_population(roster=full_roster, source_sha256=SOURCE_SHA)
    chosen = baseline.stratum("shared_gate").pilot_unit_ids[0]
    ledger = ExposureLedger({chosen: ["historical_exposure"]})
    assert ledger.is_exposed(chosen)
    steered = select_population(
        roster=full_roster,
        source_sha256=SOURCE_SHA,
        prior_exposure={chosen: ["historical_exposure"]},
    )
    fresh = steered.stratum("shared_gate").pilot_unit_ids[0]
    assert fresh != chosen
    assert fresh in pool


def test_routed_draws_carry_a_high_and_a_low_quartile_member(full_roster):
    selection = select_population(roster=full_roster, source_sha256=SOURCE_SHA)
    for stratum_id in ("routed_gate", "routed_up", "routed_down"):
        stratum = selection.stratum(stratum_id)
        for draws in (stratum.pilot, stratum.confirmation):
            assert [draw.quartile for draw in draws] == ["high", "low"]
            high, low = draws
            assert high.routing_count > low.routing_count
            assert high.block_id == low.block_id
            assert high.routing_count > 0 and low.routing_count > 0


# --------------------------------------------------------- real census ---


REAL_CENSUS = Path(
    "/mnt/shared/tessera-measurements/glm-canonical-census-20260908/workspace/census.json"
)
REAL_CENSUS_SHA = "b63f7bf6c4320714b4ceb38fbd6996e032e0f0c9b82ac2a30a8337d846e358fd"


@pytest.mark.skipif(
    not REAL_CENSUS.exists(), reason="frozen GLM census is not mounted here"
)
def test_real_census_yields_the_42_moe_layers():
    counts = load_census_counts(REAL_CENSUS, expected_sha256=REAL_CENSUS_SHA)
    roster = build_roster(counts=counts, profile=_profile())
    assert roster.layers == tuple(range(3, 45))
    assert 45 not in roster.layers  # the passthrough MTP block
    assert len(roster.shared_units) == 42 * 3
    assert len(roster.routed_units) == 42 * 288 * 3
    selection = select_population(roster=roster, source_sha256=SOURCE_SHA)
    for stratum in selection.strata:
        assert not set(stratum.pilot_unit_ids) & set(stratum.confirmation_unit_ids)
