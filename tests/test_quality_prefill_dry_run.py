"""The milestone-1 dry-run table, as a test rather than a number in a report.

``docs/measurements/quality-prefill-milestone-1-2026-09-12.md`` §3 publishes an
acquisition size -- 442 rates over two families, 7.8% of the legal domain. That
table was first computed by hand-joining work package A's domain to work
package C's builder in a throwaway script, which is exactly the kind of number
this repository retracts: nobody could re-derive it.

This file is the join, committed. It also pins the seam between the two
structurally identical ``RateDomain`` dataclasses (debt item 1 of the report's
§5): A hands C plain data through ``rate_domain_payload``, C constructs its own
class from it, and if either side's field set moves this test fails rather than
the report going quietly stale.

``source_sha256`` is the grammar digest -- the bytes the legal domain is
actually derived from -- so the selection hash is anchored in the tree and the
test needs no fixture file.
"""

from __future__ import annotations

import pytest

domain = pytest.importorskip(
    "prismaquant.tessera_legal_domain",
    reason="requires a pinned Tessera on the path (see the report's §1)",
)
population = pytest.importorskip("prismaquant.quality_prefill_population")


# The report's §3 table, keyed by family. Legal counts are independently
# asserted against the source audit in test_tessera_legal_domain.py; they are
# repeated here so a change to either package shows up as a table diff.
REPORTED = {
    "TESSERA_E4M3_K1": {"legal": 1793, "mandatory": 29, "roster": 141, "strata": 14},
    "TESSERA_BF16_K1": {"legal": 3841, "mandatory": 61, "roster": 301, "strata": 30},
}
REPORTED_TOTAL_ROSTER = 442
REPORTED_TOTAL_LEGAL = 5634


def _mandatory_set(family: str):
    """A's domain, through the payload, into C's builder."""
    theirs = population.RateDomain(**domain.rate_domain_payload(family))
    return population.build_mandatory_rate_set(
        domain=theirs,
        source_sha256=domain.TESSERA_GRAMMAR_DIGEST,
        selection_seed=0,
    )


@pytest.mark.parametrize("family", sorted(REPORTED))
def test_the_reported_dry_run_row_is_what_the_code_produces(family):
    expected = REPORTED[family]
    built = _mandatory_set(family)

    assert built.family == family
    assert len(domain.legal_rate_domain(family).rates) == expected["legal"]
    assert len(built.mandatory) == expected["mandatory"]
    assert len(built.roster) == expected["roster"]
    assert len(built.strata) == expected["strata"]
    assert built.screen_policy_id == "coverage_first_v1"
    assert built.selection_seed == 0


def test_the_reported_total_and_its_percentage_add_up():
    """442 of 5,634 is the headline; both halves come from the same code."""
    rosters = {f: len(_mandatory_set(f).roster) for f in REPORTED}
    legal = {f: len(domain.legal_rate_domain(f).rates) for f in REPORTED}

    assert sum(rosters.values()) == REPORTED_TOTAL_ROSTER
    assert sum(legal.values()) == REPORTED_TOTAL_LEGAL
    # "7.8% of the legal domain", to the one decimal the report prints.
    share = 100.0 * REPORTED_TOTAL_ROSTER / REPORTED_TOTAL_LEGAL
    assert f"{share:.1f}" == "7.8"


def test_the_roster_is_a_superset_of_the_mandatory_set_and_stays_legal():
    """The interior draw adds to the mandatory set; it never replaces it."""
    for family in REPORTED:
        built = _mandatory_set(family)
        legal = set(domain.legal_rate_domain(family).rates)
        assert set(built.mandatory) <= set(built.roster)
        assert set(built.roster) <= legal


def test_the_two_rate_domain_classes_are_distinct_but_the_payload_joins_them():
    """Debt item 1, pinned: the duplication is real and the seam is the fix.

    If the classes are ever merged this test should be deleted along with the
    report's debt entry -- but until then, a field renamed on one side must
    fail here rather than in a hand-run script nobody kept.
    """
    assert domain.RateDomain is not population.RateDomain
    for family in domain.PRIMARY_FAMILIES:
        payload = domain.rate_domain_payload(family)
        assert set(payload) == {"family", "rates", "transition_rates"}
        theirs = population.RateDomain(**payload)
        mine = domain.legal_rate_domain(family)
        assert theirs.rates == mine.rates
        assert theirs.transition_rates == mine.transition_rates
