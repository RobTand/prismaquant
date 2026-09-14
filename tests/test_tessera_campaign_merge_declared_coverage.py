"""A merge expects the scope, or exactly what the plan declares instead.

The rows stamp the whole census scope into ``campaign_scope`` whether or not
the plan ran every row, so a plan that deliberately dropped rows must say so
under ``dense_rows_excluded`` for the merge to expect less than the scope.
The expectation is read from the plan alone -- the union of its remaining
rows' groups -- never from which rows happen to exist, and the merged table
records what it left unpriced and why.
"""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dispatch_tessera_campaign as dispatch
from test_tessera_campaign_fanout import SCOPE, _shard

REASON = "the dense cells already exist in another table"


def _three_group_rows():
    """Two priced rows under a scope of three anchor groups (``u:c`` unpriced)."""
    scope = copy.deepcopy(SCOPE)
    scope["anchor_groups"]["u:c"] = ["c"]
    scope["dense_targets"] = scope["dense_all"] = ["a", "b", "c"]
    rows = {"row-0000": _shard("u:a", "a"), "row-0001": _shard("u:b", "b")}
    for payload in rows.values():
        payload["provenance"]["campaign_scope"] = copy.deepcopy(scope)
    return rows


def _plan(rows, excluded=None):
    plan = {"schema": dispatch.PLAN_SCHEMA,
            "rows": [{"row_id": row_id, "groups": list(groups)} for row_id, groups in rows]}
    if excluded is not None:
        plan[dispatch.DENSE_ROWS_EXCLUDED_KEY] = excluded
    return plan


def _merge(rows, coverage):
    return dispatch.merge_payloads(rows, census={"counts": {}}, capture_sha256="merged",
                                   plan_coverage=coverage)


def test_a_plan_without_the_declaration_keeps_the_scope_refusal():
    plan = _plan([("row-0000", ["u:a"]), ("row-0001", ["u:b"])])
    assert dispatch.declared_coverage(plan) is None
    with pytest.raises(dispatch.MergeRefused, match=r"do not cover 1 anchor group\(s\) of the scope: u:c"):
        _merge(_three_group_rows(), dispatch.declared_coverage(plan))


def test_a_declared_exclusion_merges_the_declared_groups_and_stamps_the_rest():
    plan = _plan([("row-0000", ["u:a"]), ("row-0001", ["u:b"])],
                 excluded={"rows": ["row-0002"], "reason": REASON})
    coverage = dispatch.declared_coverage(plan)
    assert coverage == {"expected_groups": ["u:a", "u:b"], "excluded_rows": ["row-0002"],
                        "reason": REASON}

    merged = _merge(_three_group_rows(), coverage)
    selection = merged["provenance"]["unit_selection"]
    # True is what a selection writes; False is the monolith's whole-scope claim.
    assert selection["selected"] is True
    assert [entry["key"] for entry in selection["groups"]] == ["u:a", "u:b"]
    assert merged["provenance"]["coverage"] == {
        "schema": dispatch.COVERAGE_SCHEMA,
        "scope_groups": 3, "priced_groups": 2, "unpriced_groups": ["u:c"],
        "excluded_rows": ["row-0002"], "reason": REASON,
    }
    assert sorted(merged["costs"]) == ["a", "b"]
    # The population block is still rebuilt over the scope, so the unpriced
    # dense unit is counted there too.
    from prismaquant.tessera_expert_projection import POPULATION_KEY
    population = merged["provenance"][POPULATION_KEY]
    assert population["priced"]["dense"] == ["a", "b"]
    assert "c" in population["unpriced"]["dense"]


def test_the_expectation_is_the_plan_and_not_the_rows_that_exist():
    # The plan declares three priced rows and excludes a fourth; only two rows
    # were merged.  The missing declared row is still missing.
    plan = _plan([("row-0000", ["u:a"]), ("row-0001", ["u:b"]), ("row-0002", ["u:c"])],
                 excluded={"rows": ["row-0003"], "reason": REASON})
    with pytest.raises(dispatch.MergeRefused, match=r"do not cover 1 anchor group\(s\) of the plan: u:c"):
        _merge(_three_group_rows(), dispatch.declared_coverage(plan))


def test_rows_pricing_a_group_the_plan_does_not_declare_refuse():
    plan = _plan([("row-0000", ["u:a"])], excluded={"rows": ["row-0002"], "reason": REASON})
    with pytest.raises(dispatch.MergeRefused, match="the plan does not declare: u:b"):
        _merge(_three_group_rows(), dispatch.declared_coverage(plan))


def test_a_declaration_that_still_covers_the_scope_is_a_whole_scope_table():
    rows = {"row-0000": _shard("u:a", "a"), "row-0001": _shard("u:b", "b")}
    plan = _plan([("row-0000", ["u:a"]), ("row-0001", ["u:b"])],
                 excluded={"rows": ["row-0009"], "reason": REASON})
    merged = _merge(rows, dispatch.declared_coverage(plan))
    assert merged["provenance"]["unit_selection"]["selected"] is False
    assert merged["provenance"]["coverage"]["unpriced_groups"] == []
    assert merged["provenance"]["coverage"]["priced_groups"] == 2


def test_a_declared_group_outside_the_scope_refuses():
    plan = _plan([("row-0000", ["u:a"]), ("row-0001", ["u:b", "u:z"])],
                 excluded={"rows": ["row-0002"], "reason": REASON})
    with pytest.raises(dispatch.MergeRefused, match="outside the campaign scope: u:z"):
        _merge(_three_group_rows(), dispatch.declared_coverage(plan))


@pytest.mark.parametrize("excluded, message", [
    ({"rows": ["row-0002"]}, "give a reason"),
    ({"rows": ["row-0002"], "reason": "  "}, "give a reason"),
    ({"rows": [], "reason": REASON}, "list the excluded row ids"),
    ({"reason": REASON}, "list the excluded row ids"),
    ("row-0002", "list the excluded row ids"),
    ({"rows": ["row-0001"], "reason": REASON}, "names planned row"),
])
def test_a_malformed_declaration_refuses(excluded, message):
    plan = _plan([("row-0000", ["u:a"]), ("row-0001", ["u:b"])], excluded=excluded)
    with pytest.raises(dispatch.MergeRefused, match=message):
        dispatch.declared_coverage(plan)


def test_a_whole_scope_merge_carries_no_coverage_block():
    rows = {"row-0000": _shard("u:a", "a"), "row-0001": _shard("u:b", "b")}
    merged = _merge(rows, None)
    assert "coverage" not in merged["provenance"]
    assert merged["provenance"]["unit_selection"]["selected"] is False
