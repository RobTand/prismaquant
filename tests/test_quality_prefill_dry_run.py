"""The milestone-1 dry-run table, as a test rather than a number in a report.

``docs/measurements/quality-prefill-milestone-1-2026-09-12.md`` §3 publishes an
acquisition size -- 442 rates over two families, 7.8% of the legal domain. That
table was first computed by hand-joining work package A's domain to work
package C's builder in a throwaway script, which is exactly the kind of number
this repository retracts: nobody could re-derive it.

This file is the join, committed. It also pins the seam that used to run
between two structurally identical ``RateDomain`` dataclasses (debt item 1 of
the report's §5, paid 2026-09-12): there is now one class, defined in package C
and imported by package A, and A still hands C plain data through
``rate_domain_payload``. The payload is still the seam -- it is what a frozen
document carries -- so this test still builds the domain from the payload and
checks it against the grammar-derived one, and a field renamed on either side
still fails here rather than the report going quietly stale.

``source_sha256`` is the grammar digest -- the bytes the legal domain is
actually derived from -- so the selection hash is anchored in the tree and the
test needs no fixture file.
"""

from __future__ import annotations

import pytest

from prismaquant import quality_prefill_population as population
from prismaquant import tessera_legal_domain as domain


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


def test_one_rate_domain_class_and_the_payload_still_joins_through_it():
    """Debt item 1, paid: one class, and the payload seam still checked.

    The merge removes the duplicate, not the seam.  A frozen plan carries the
    domain as data, so the round trip that matters is grammar -> payload ->
    class -> equality with the grammar-derived domain, and that is what runs
    here.  A field renamed on either side still fails in this test rather than
    in a hand-run script nobody kept.
    """
    assert domain.RateDomain is population.RateDomain
    for family in domain.PRIMARY_FAMILIES:
        payload = domain.rate_domain_payload(family)
        assert set(payload) == {"family", "rates", "transition_rates"}
        rebuilt = population.RateDomain(**payload)
        derived = domain.legal_rate_domain(family)
        assert rebuilt == derived
        assert rebuilt.rates == derived.rates
        assert rebuilt.transition_rates == derived.transition_rates
        # The builder is the consumer; the payload has to satisfy it, not just
        # construct.
        assert set(population.mandatory_rates(rebuilt)) <= set(derived.rates)
