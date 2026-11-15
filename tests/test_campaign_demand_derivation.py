"""The dispatcher's memory demand, derived from the plan the row checks itself against.

A campaign row states a PrismaBuild memory demand; PrismaBuild turns that
number into a cgroup cap; the row then refuses unless its own phase plan fits
under that cap less the floor it measures for itself, and its guard refuses
again at ``cap - margin``. Three quantities, two of them outside the plan, and
until RobTand/prismaquant#522 the demand carried neither.

Everything here runs on an injected plan. No model files, no NFS, no GPU: what
is under test is the arithmetic that joins the demand to the predicate, not any
measurement of a real checkpoint.
"""
from __future__ import annotations


GIB = 1024 ** 3

# A plan whose rounding slack sits below the measured process floor, so the
# demand is only right if it charges the floor rather than inheriting it from
# ``ceil``. Both example rows inspected for #390 were on this side of the line.
PLAN_BYTES = 20 * GIB + 173_741_824

# The relaunch row this issue was filed on: ``extension-r1024-02`` row-0074,
# whose selected-anchor plan measured 108.63 GB while its manifest declared
# ``mem_gb: 102``.
GLM_ROW_0074_PLAN_BYTES = 108_630_000_000


# ---------------------------------------------------------------------------
# the derivation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the submit path
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the row's own refusal
# ---------------------------------------------------------------------------

def test_the_row_refusal_detail_carries_every_term_of_its_predicate():
    """The message the three lost rows did not have.

    The same plan and the same cap, twice, with only the floor moving inside
    the range the fleet measured: at the bottom of that range the row is
    admitted with about 11 MB to spare, and at the top it is refused. That is
    the coin flip #522 was filed on, and neither outcome was visible in the
    old message.
    """
    from prismaquant.tessera_campaign import memory_admission_detail
    admitted = memory_admission_detail(plan_bytes=108_630_000_000,
                                       cap_bytes=102 * GIB,
                                       baseline_bytes=880_000_000)
    assert admitted == dict(plan_bytes=108_630_000_000, cap_bytes=102 * GIB,
                            baseline_bytes=880_000_000,
                            slack_bytes=102 * GIB - 880_000_000 - 108_630_000_000)
    assert 0 < admitted['slack_bytes'] < 16 * 1024 ** 2

    refused = memory_admission_detail(plan_bytes=108_630_000_000,
                                      cap_bytes=102 * GIB,
                                      baseline_bytes=1_150_000_000)
    assert refused['slack_bytes'] < 0, 'the worst measured floor refuses'

    # The capture-path predicate has taken no reading yet, so it reports no
    # baseline rather than a measured zero.
    unmeasured = memory_admission_detail(plan_bytes=10, cap_bytes=4)
    assert 'baseline_bytes' not in unmeasured
    assert unmeasured['slack_bytes'] == -6
