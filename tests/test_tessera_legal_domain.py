"""The legal Tessera rate domain, its bytes, and the five support facts.

Every test here checks a *derivation*, not a constant.  The audited counts
1,793 and 3,841, the table-width transitions at R3585 and R3841, and the
singleton native set are compared against what the pinned producer and reader
actually say; none of them is read back out of the module that would have to
be wrong for the check to matter.

The one place a literal is legitimate is the independent byte recomputation:
that test rebuilds a candidate's total from the grammar's own three
quantities -- the body rate, the grid's code width and the table width -- and
refuses to look at the stored total until it has its own.
"""
from __future__ import annotations

import dataclasses
import json
import os
from fractions import Fraction

import pytest

from prismaquant import tessera_legal_domain as domain
from prismaquant.tessera_formats import (
    family_q256_bounds,
    get_tessera_family,
    tessera_wire_recipe,
)


E4 = "TESSERA_E4M3_K1"
BF = "TESSERA_BF16_K1"
E2 = "TESSERA_E2M1_K2"


# ---------------------------------------------------------------------------
# Criterion 1: the counts
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rates():
    """``{family: (legal rates, holes)}`` at the real GLM shapes.

    Module-scoped because the walk asks Tessera's own Bresenham scheduler for
    every one of 5,634 rates at five shapes.
    """
    return {
        family: domain.legal_rates(family, domain.GLM53_LINEAR_SHAPES)
        for family in domain.PRIMARY_FAMILIES
    }


@pytest.mark.parametrize(
    "family, expected", [(E4, 1793), (BF, 3841)])
def test_legal_rate_count_is_the_audited_count(rates, family, expected):
    """The derived count equals the frozen audit's, and the audit agrees.

    Both directions matter.  The first assertion is the acceptance oracle.  The
    second is the guard against a module that reached the number by reading it:
    ``AUDITED_RATE_COUNTS`` must carry the same value the walk produced, and
    the walk must not consult it.
    """
    legal, holes = rates[family]
    assert len(legal) == expected
    assert domain.AUDITED_RATE_COUNTS[family] == expected
    assert holes == {}, (
        f"{family}: the shape walk refuses {len(holes)} rate(s) inside its own "
        f"bounds; that is a finding to report, not a rounding to smooth: "
        f"{sorted(holes)[:5]}"
    )


@pytest.mark.parametrize(
    "family, lo, hi", [(E4, 256, 2048), (BF, 256, 4096)])
def test_domain_is_contiguous_between_the_producer_bounds(rates, family, lo, hi):
    """No holes: the domain is every integer q256 the producer's bounds name."""
    legal, _holes = rates[family]
    assert family_q256_bounds(family) == (lo, hi)
    assert legal == tuple(range(lo, hi + 1))


def test_glm_shapes_are_derived_from_the_real_config():
    """The five frozen shapes are what the real GLM-5.3-Flash config generates.

    Skipped rather than faked when the model is not mounted; a skip certifies
    nothing and says so.
    """
    if not os.path.exists(domain.GLM53_CONFIG_PATH):
        pytest.skip(f"{domain.GLM53_CONFIG_PATH} is not mounted")
    with open(domain.GLM53_CONFIG_PATH, encoding="utf-8") as handle:
        config = json.load(handle)
    assert domain.glm_linear_shapes(config) == domain.GLM53_LINEAR_SHAPES


def test_every_glm_column_count_is_a_whole_number_of_superblocks():
    """Why the domain has no holes: the exact-quota grammar closes everywhere.

    This is the *reason* the previous test passes, stated as a property rather
    than left implicit, so a future shape whose columns are not divisible by
    256 fails here with the cause instead of failing the count with a symptom.
    """
    from prismaquant.tessera_formats import SUPERBLOCK_WEIGHTS

    for _rows, columns in domain.GLM53_LINEAR_SHAPES:
        assert columns % SUPERBLOCK_WEIGHTS == 0, columns


# ---------------------------------------------------------------------------
# Criterion 2: the table-width boundary witnesses
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "below, above, width_below, width_above",
    [(3584, 3585, 14, 15), (3840, 3841, 15, 16)],
)
def test_bf16_table_width_transitions_at_the_exact_boundary(
    below, above, width_below, width_above,
):
    """Both sides of each transition, read off the producer's own recipe.

    The widths come from ``tessera.export.wire_recipe`` per rung, so this
    witnesses the producer's ``max(default, ceil(q256/256))`` rule rather than
    restating three numbers: 3584/256 is exactly 14 so R3584 keeps the default,
    and ceil(3585/256) is 15.
    """
    assert domain.table_width_bits(BF, below) == width_below
    assert domain.table_width_bits(BF, above) == width_above
    # The arithmetic the producer performs, recomputed here from the rate.
    assert max(14, -(-below // 256)) == width_below
    assert max(14, -(-above // 256)) == width_above


def test_bf16_table_width_transitions_are_exactly_two():
    """The whole BF16 domain has these transitions and no others."""
    assert domain.table_width_transitions(BF) == ((256, 14), (3585, 15), (3841, 16))
    assert domain.table_width_transitions(BF) == domain.AUDITED_BF16_TABLE_TRANSITIONS


def test_e4m3_table_width_never_widens():
    """E4M3's schedule rate tops out at 8, so its L=14 recipe covers its domain."""
    assert domain.table_width_transitions(E4) == ((256, 14),)
    lo, hi = family_q256_bounds(E4)
    assert max(-(-rate // 256) for rate in (lo, hi)) <= 14


def test_boundary_witnesses_cover_both_sides_of_every_transition():
    witnesses = domain.boundary_witnesses(BF)
    for required in (256, 3584, 3585, 3840, 3841, 4096):
        assert required in witnesses, required


def test_every_candidate_carries_its_own_table_width():
    """The inventory row's width is the recipe's width, per rate."""
    inventory = _inventory()
    for entry in inventory["families"].values():
        for rows in entry.candidates_by_structure.values():
            for candidate in rows:
                recipe = tessera_wire_recipe(candidate.family, candidate.rate_q256)
                assert candidate.table_width_bits == int(recipe.window_bits)


# ---------------------------------------------------------------------------
# Criterion 3: R4096 is flagged, not admitted
# ---------------------------------------------------------------------------

def test_r4096_is_a_window_channel_artifact_and_not_source_bf16_passthrough():
    """The domain endpoint decodes to BF16 and is not a copy of the source.

    Three separate assertions because they are three separate claims, and
    conflating the first with the third is the exact error the flag exists to
    prevent: ``terminal_format == "BF16"`` says the decoded tile is BF16, and
    it does NOT say the wire carried the source bytes.
    """
    candidate = _candidate(BF, 4096, "dense")
    assert candidate.representation == "WINDOW body over a CHANNEL scale plane"
    assert candidate.terminal_format == "BF16"
    assert candidate.source_bf16_passthrough is False
    assert domain.FLAG_NOT_SOURCE_BF16_PASSTHROUGH in candidate.flags
    assert domain.FLAG_DOMAIN_ENDPOINT in candidate.flags
    assert "not a copy of the source bytes" in candidate.passthrough_note


def test_no_bf16_candidate_anywhere_claims_source_passthrough():
    """Not only the endpoint: every BF16 rung carries the flag."""
    inventory = _inventory()
    entry = inventory["families"][BF]
    seen = 0
    for rows in entry.candidates_by_structure.values():
        for candidate in rows:
            assert candidate.source_bf16_passthrough is False
            assert domain.FLAG_NOT_SOURCE_BF16_PASSTHROUGH in candidate.flags
            seen += 1
    assert seen > 0


def test_r4096_weighs_more_than_sixteen_bits_per_weight():
    """The endpoint is not free: its 64K-entry table is real bytes.

    R/256 would say 16.0 bpp.  The wire says more, because a 2-byte-per-entry
    table over 2**16 entries is 128 KiB the unit carries whatever its shape.
    """
    account = domain.byte_account(BF, 4096, (2048, 4096))
    assert account.table_bytes == 2 * (1 << 16)
    assert account.exact_bits_per_weight > Fraction(16)


# ---------------------------------------------------------------------------
# Criterion 4: five separate fields, and they disagree
# ---------------------------------------------------------------------------

def test_support_facts_has_exactly_the_five_named_fields():
    """Five fields, the five names, and no collapsing helper."""
    names = tuple(f.name for f in dataclasses.fields(domain.SupportFacts))
    assert names == domain.SUPPORT_FACT_NAMES
    assert len(names) == 5
    for forbidden in ("supported", "ok", "eligible", "admitted"):
        assert not hasattr(domain.SupportFacts, forbidden), forbidden


def test_the_five_facts_disagree_on_a_natively_qualified_candidate():
    """E4M3 R1024 dense: producer/reader/route/native yes, served-validation no.

    A candidate whose five facts all agreed would prove nothing.  This one is
    qualified natively by a cell and still has no complete assignment through
    export plus served validation in this artifact scope, which is the shape of
    disagreement the ledger exists to be able to record.
    """
    facts = _candidate(E4, 1024, "dense").support
    assert facts.disagree()
    assert facts.producer_legal.value is True
    assert facts.reader_supported.value is True
    assert facts.implemented_route.value is True
    assert facts.native_qualification.value is True
    assert facts.export_and_served_validation.value is False
    assert {f.name for f in facts.facts()} == set(domain.SUPPORT_FACT_NAMES)


def test_producer_legal_and_reader_supported_genuinely_diverge_on_the_control():
    """The E2M1 control is producer-legal below its reader range and refused.

    At this pin the two primary families' producer domains happen to equal
    their reader ranges, so the E4/BF16 roster cannot witness this divergence.
    The control can: the producer writes R128-R896 and the pinned reader
    declares R896-R896 only, so R256 is producer-legal and reader-unsupported.
    A ledger that collapsed the two would report this candidate as simply
    absent.
    """
    lo, hi = family_q256_bounds(E2)
    assert (lo, hi) == (128, 896)
    facts = domain.support_facts(E2, 256, "dense", shapes=domain.GLM53_LINEAR_SHAPES)
    assert facts.producer_legal.value is True
    assert facts.reader_supported.value is False
    assert facts.disagree()
    # And the reverse direction still holds at the one readable rung.
    attested = domain.support_facts(
        E2, 896, "dense", shapes=domain.GLM53_LINEAR_SHAPES)
    assert attested.producer_legal.value is True
    assert attested.reader_supported.value is True


def test_every_fact_names_the_table_that_answered_it():
    """Principle 14: a fact is derived from a table, and says which one."""
    for family, rate in ((E4, 1024), (BF, 4096)):
        for fact in _candidate(family, rate, "dense").support.facts():
            assert fact.source, fact.name
            assert "tessera" in fact.source.lower(), (fact.name, fact.source)
            assert fact.detail, fact.name


# ---------------------------------------------------------------------------
# Native qualification is exact membership, and does not shrink the domain
# ---------------------------------------------------------------------------

def test_native_qualification_is_exactly_three_cells_for_the_primary_families():
    """Exact membership at the pin: E4 R1024 dense and routed, BF R1792 dense.

    Nothing else, and in particular no neighbouring rate: attestation does not
    extrapolate from a singleton.
    """
    triples = domain.native_qualification_set()
    primary = {t for t in triples if t[0] in domain.PRIMARY_FAMILIES}
    assert primary == {
        (E4, 1024, "dense"),
        (E4, 1024, "routed_moe"),
        (BF, 1792, "dense"),
    }


def test_the_native_set_does_not_shrink_the_legal_domain(rates):
    """The count is computed without the attestation and is unaffected by it.

    The strong form: the rates that are natively qualified are a three-element
    subset of a 5,634-element domain, and the domain walk never reads the
    attestation.  A regression that let the menu's attested mode leak into the
    domain would collapse these counts to 2.
    """
    triples = domain.native_qualification_set()
    qualified_rates = {
        (family, rate) for (family, rate, _s) in triples
        if family in domain.PRIMARY_FAMILIES
    }
    assert len(qualified_rates) == 2
    total = sum(len(legal) for legal, _holes in rates.values())
    assert total == 1793 + 3841
    for family, rate in qualified_rates:
        assert rate in rates[family][0]


def test_an_unattested_neighbour_of_an_attested_rung_is_still_producer_legal():
    """R1023 has no cell and is still in the domain with a route."""
    facts = domain.support_facts(E4, 1023, "dense")
    assert facts.producer_legal.value is True
    assert facts.reader_supported.value is True
    assert facts.native_qualification.value is False
    assert 1023 in domain.legal_rates(E4)[0]


def test_routed_moe_attestation_uses_its_own_runtime_image():
    """The routed cells serve on a different image than the dense ones.

    Asking with the contract's ``default_serve_image`` reports the routed E4
    cell unattested, which would read as an absence.  The ledger derives each
    cell's serving context from the cell, so the image difference is recorded
    rather than lost.
    """
    cells = domain.attested_cells()
    routed = {c.runtime_image for c in cells if c.structure == "routed_moe"}
    dense = {c.runtime_image for c in cells if c.structure == "dense"}
    assert routed and dense
    assert routed.isdisjoint(dense)


# ---------------------------------------------------------------------------
# Criterion 5: bytes, recomputed independently from components
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "family, rate, shape",
    [
        (BF, 1792, (2048, 4096)),
        (BF, 4096, (2048, 4096)),
        (BF, 3585, (4096, 12288)),
        (E4, 1024, (12288, 4096)),
        (E4, 2048, (4096, 2048)),
    ],
)
def test_byte_total_is_reproduced_from_first_principles(family, rate, shape):
    """Rebuild the total from the grammar, then compare -- never the reverse.

    Three quantities, each from its own source and none from the accountant
    under test: the body rate ``rate/256`` bits per weight (exact because every
    GLM column count is a whole number of 256-weight superblocks), the grid's
    ``code_bytes`` per table entry, and the table width ``L`` from the rung's
    own recipe.  A CHANNEL plane adds one fp16 per output row.
    """
    from prismaquant.tessera_formats import SUPERBLOCK_WEIGHTS

    spec = get_tessera_family(family)
    rows, columns = shape
    assert columns % SUPERBLOCK_WEIGHTS == 0
    width = int(tessera_wire_recipe(spec, rate).window_bits)

    body = rows * columns * rate // (256 * 8)
    table = spec.code_bytes * (1 << width)
    scale = 2 * rows                       # one fp16 per output channel
    expected = body + table + scale

    account = domain.byte_account(family, rate, shape)
    assert account.scale_plane == "channel"
    assert account.body_bytes == body
    assert account.table_bytes == table
    assert account.scale_bytes == scale
    assert account.descendant_bytes == 0   # a WINDOW body has no forest
    assert account.total_bytes == expected


def test_components_sum_to_the_total_for_every_boundary_witness():
    """No component is silently dropped or double counted, at any witness."""
    for family in domain.PRIMARY_FAMILIES:
        for rate in domain.boundary_witnesses(family):
            for shape in domain.GLM53_LINEAR_SHAPES:
                account = domain.byte_account(family, rate, shape)
                assert sum(account.components().values()) == account.total_bytes


def test_rate_over_256_is_not_the_bits_per_parameter():
    """The headline trap, made a test.

    R1792/256 is 7.0.  The artifact is not 7.0 bits per weight, because the
    window table and the channel plane are real bytes the unit carries.
    """
    account = domain.byte_account(BF, 1792, (2048, 4096))
    assert account.exact_bits_per_weight > Fraction(1792, 256)
    assert account.table_bytes > 0 and account.scale_bytes > 0


def test_byte_account_agrees_with_the_shape_free_accountant():
    """Two PrismaQuant accountants, one answer.

    ``tessera_exact_bits_for_shape`` is the other consumer of the same
    breakdown; if this module's re-labelling of the planes had lost or
    duplicated a term, the two totals would part.
    """
    from prismaquant.tessera_footprint import tessera_exact_bits_for_shape

    for family, rate in ((BF, 1792), (BF, 4096), (E4, 1024)):
        for shape in domain.GLM53_LINEAR_SHAPES:
            account = domain.byte_account(family, rate, shape)
            other = tessera_exact_bits_for_shape(family, rate, shape)
            assert Fraction(account.total_bytes * 8) == other


# ---------------------------------------------------------------------------
# Criterion 6: pin drift
# ---------------------------------------------------------------------------

def test_live_pins_match_the_frozen_pins_today():
    report = domain.pin_drift()
    assert report["matches"] is True, report["differences"]
    assert report["differences"] == {}
    # The contract the producer actually packages is the reviewed one.
    live = domain.live_pins()
    assert (live.producer_installed_contract_sha256
            == live.serving_runtime_pinned_contract_sha256)


def test_the_spec_named_study_producer_is_documentary_only():
    """It is reported, and it is not one of the comparable fields.

    ``d403cc5a`` appears nowhere in PrismaQuant's source and is 25 commits
    after the reader pin, so putting it in the comparable block would make the
    drift report fire on every run against a value no code reads.
    """
    assert domain.SPEC_NAMED_STUDY_PRODUCER not in domain.FROZEN_PINS.as_dict().values()
    assert domain.pin_drift()["spec_named_study_producer"] == (
        domain.SPEC_NAMED_STUDY_PRODUCER)


def test_a_moved_reader_pin_produces_a_diff_report_not_a_silent_answer(monkeypatch):
    """Move the pin the code reads; the report names the field and both values."""
    moved = "0" * 40
    monkeypatch.setattr(
        "prismaquant.tessera_runtime_contract.TESSERA_DEV_PIN_COMMIT", moved)
    report = domain.pin_drift()
    assert report["matches"] is False
    assert "reader_dev_pin_commit" in report["differences"]
    assert report["differences"]["reader_dev_pin_commit"] == {
        "frozen": domain.FROZEN_PINS.reader_dev_pin_commit, "live": moved,
    }
    assert "PIN DRIFT" in report["verdict"]
    assert "reader_dev_pin_commit" in report["verdict"]


def test_a_moved_producer_digest_produces_a_diff_report(monkeypatch):
    """The producer leg drifts independently of the two tracked literals."""
    moved = "f" * 64
    monkeypatch.setattr(
        "prismaquant.tessera_serving_runtime_pin.installed_tessera_contract_sha256",
        lambda: moved)
    report = domain.pin_drift()
    assert report["matches"] is False
    assert set(report["differences"]) == {"producer_installed_contract_sha256"}
    assert report["differences"]["producer_installed_contract_sha256"]["live"] == moved


def test_the_inventory_carries_its_drift_report(monkeypatch):
    """A drifted build reports the drift inside the inventory it returns."""
    moved = "1" * 40
    monkeypatch.setattr(
        "prismaquant.tessera_runtime_contract.TESSERA_DEV_PIN_COMMIT", moved)
    inventory = domain.build_inventory(
        (E4,), ledger_rates=(), structures=("dense",))
    assert inventory["pin_drift"]["matches"] is False
    assert inventory["pins"]["reader_dev_pin_commit"] == moved
    # And the report puts the drift before the counts, where it cannot be missed.
    text = domain.format_report(inventory)
    assert text.index("PIN DRIFT") < text.index("Legal rate domain")


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_main_prints_the_counts_and_the_audit_verdict(capsys):
    assert domain.main() == 0
    text = capsys.readouterr().out
    assert "count 1793" in text
    assert "count 3841" in text
    assert "derived 1793 vs audited 1793 -- agrees" in text
    assert "derived 3841 vs audited 3841 -- agrees" in text
    assert text.index("Pin drift") < text.index("Legal rate domain")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _inventory():
    if "inventory" not in _CACHE:
        _CACHE["inventory"] = domain.build_inventory()
    return _CACHE["inventory"]


def _candidate(family, rate, structure):
    for candidate in _inventory()["families"][family].candidates_by_structure[structure]:
        if candidate.rate_q256 == rate:
            return candidate
    raise AssertionError(f"{family}_R{rate} on {structure} is not in the ledger")


# ---------------------------------------------------------------------------
# The provider work package C consumes
# ---------------------------------------------------------------------------

def test_rate_domain_has_the_pinned_three_fields_in_order():
    """The interface package C declares: family, rates, transition_rates."""
    names = tuple(f.name for f in dataclasses.fields(domain.RateDomain))
    assert names == ("family", "rates", "transition_rates")


@pytest.mark.parametrize("family, count", [(E4, 1793), (BF, 3841)])
def test_provider_returns_the_full_sorted_unique_domain(family, count):
    provided = domain.legal_rate_domain(family)
    assert provided.family == family
    assert len(provided.rates) == count
    assert tuple(sorted(set(provided.rates))) == provided.rates
    assert set(provided.transition_rates) <= set(provided.rates)


@pytest.mark.parametrize(
    "bad",
    [
        {"family": E4, "rates": (), "transition_rates": ()},
        {"family": E4, "rates": (512, 256), "transition_rates": ()},
        {"family": E4, "rates": (256, 256), "transition_rates": ()},
        {"family": E4, "rates": (256, 512), "transition_rates": (257,)},
    ],
)
def test_rate_domain_refuses_the_same_inputs_package_c_refuses(bad):
    """Empty, unsorted, duplicated, and a transition outside the domain."""
    with pytest.raises(Exception):
        domain.RateDomain(**bad)


def test_a_transition_names_the_first_rate_of_the_new_regime():
    """Pinned convention 1, checked against the resolver on both sides.

    R3585 is a transition and R3584 is not one *for the table width*: 3584/256
    is exactly 14 so it still belongs to the 14-bit regime.  R3584 is
    nonetheless a transition here, for the independent reason that the
    schedule becomes uniform at every 256-multiple -- two different regime
    changes that happen to be adjacent, and the ledger records both causes.
    """
    transitions = set(domain.resolver_transitions(BF))
    assert {3585, 3841} <= transitions
    assert domain.table_width_bits(BF, 3584) == 14
    assert domain.table_width_bits(BF, 3585) == 15
    width_steps = {rate for rate, _w in domain.table_width_transitions(BF)}
    assert 3584 not in width_steps and 3840 not in width_steps
    assert {3585, 3841} <= width_steps


@pytest.mark.parametrize("family", [E4, BF])
def test_every_256_multiple_above_the_endpoint_is_a_resolver_transition(family):
    """Pinned convention 2: the resolver DOES change at every 256-multiple.

    At ``R = 256k`` over a column count divisible by 256 the Bresenham
    schedule is uniform at per-column rate ``k``; at ``256k + 1`` it is the
    mixed ``{k, k+1}`` walk.  Those are different schedules, so both rates
    open a new regime and both are reported -- otherwise package C would draw
    them as ordinary interior rates and their neighbours would never become
    mandatory.
    """
    lo, hi = family_q256_bounds(family)
    transitions = set(domain.resolver_transitions(family))
    expected = {r for r in range(lo, hi + 1) if r % 256 == 0 and r != lo}
    expected |= {r for r in range(lo, hi + 1) if r % 256 == 1 and r != lo}
    assert transitions == expected
    assert lo not in transitions, "the domain's first rate has no predecessor"


@pytest.mark.parametrize(
    "rate, uniform", [(1792, True), (1793, False), (1024, True), (1025, False)])
def test_the_schedule_signature_is_what_makes_a_256_multiple_a_transition(
    rate, uniform,
):
    """The cause, not the symptom: one distinct column rate versus two."""
    signature = domain.schedule_signature(E4, rate)
    per_shape = signature[-1]
    for _columns, distinct in per_shape:
        assert (len(distinct) == 1) is uniform, (rate, distinct)


def test_boundary_witnesses_take_the_previous_legal_neighbour():
    """Not ``rate - 1``: the previous rate in the sorted legal domain."""
    provided = domain.legal_rate_domain(BF)
    witnesses = set(domain.boundary_witnesses(BF, provided.rates))
    assert {provided.rates[0], provided.rates[-1]} <= witnesses
    assert {3584, 3585, 3840, 3841} <= witnesses


def test_rate_domain_payload_constructs_the_dataclass():
    payload = domain.rate_domain_payload(E4)
    assert set(payload) == {"family", "rates", "transition_rates"}
    rebuilt = domain.RateDomain(**payload)
    assert rebuilt == domain.legal_rate_domain(E4)


def test_payload_satisfies_package_c_when_package_c_is_importable():
    """The real compatibility check, when both packages are on one branch.

    Skipped -- and therefore certifying nothing -- while work package C lives
    on its own branch.  It exists so the join is checked by a test at merge
    rather than by two matching docstrings.
    """
    population = pytest.importorskip(
        "prismaquant.quality_prefill_population",
        reason="work package C is not on this branch yet",
    )
    for family in domain.PRIMARY_FAMILIES:
        theirs = population.RateDomain(**domain.rate_domain_payload(family))
        assert theirs.rates == domain.legal_rate_domain(family).rates
        mandatory = population.mandatory_rates(theirs)
        assert {3584, 3585} <= set(mandatory) or family == E4


@pytest.mark.parametrize(
    "family, rate, body, plane",
    [(BF, 4096, "window", "channel"), (E4, 1024, "window", "channel"),
     (E2, 896, "tcq", "lut16")],
)
def test_recipe_labels_come_through_the_canonicalisers(family, rate, body, plane):
    """Regression: the wire's body and plane are enums, not printable strings.

    ``WireRecipe.body`` and ``.scale_plane`` are not guaranteed to arrive as
    enum members, so formatting them with ``str()`` read ``2`` for the CHANNEL
    plane and labelled a WINDOW rung "TCQ body over a 2 scale plane" -- a
    caught-by-PB failure of the R4096 flag test.  Both go through Tessera's
    ``BodyKind`` and the tree's ``scale_plane_name`` now, and a TCQ family is
    in the table so the WINDOW branch cannot be right by accident.
    """
    facts = domain.support_facts(family, rate, "dense")
    candidate = domain._candidate(
        get_tessera_family(family), rate, "dense", facts)
    assert candidate.body_kind == body
    assert candidate.scale_plane == plane
    assert candidate.representation == (
        f"{body.upper()} body over a {plane.upper()} scale plane")
    # The byte accountant resolves the same wire from the same recipe.
    account = domain.byte_account(family, rate, (2048, 4096))
    assert account.body_kind == body
    assert account.scale_plane == plane
