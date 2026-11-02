"""Shared strict/reach transition policy, independent of repository language."""
from datetime import date
import json

import pytest

from tools.ci import graph_impact as core


def policy(graph, mode='reach'):
    return {**core.baseline_record(graph), 'test_to_test': mode,
            'test_to_test_strict_at_count': 0, 'test_to_test_strict_by': '2026-11-16'}


def sibling_graphs():
    before = core.measure({'pkg/a': {'x'}, 'pkg/b': {'x', 'y'}},
                          {('pkg/a', 'x'), ('pkg/b', 'x'), ('pkg/b', 'y')}, {'x': 1, 'y': 1})
    after = core.measure({'pkg/a': {'x', 'y'}, 'pkg/b': {'x', 'y'}},
                         set(map(tuple, before['edges'])) | {('pkg/a', 'pkg/b')}, before['loc'])
    return before, after


def test_strict_rejects_new_edge_within_package_reach():
    before, after = sibling_graphs()
    errors = core.compare(before, after, policy(before, 'strict'))
    assert any('new test_to_test edge: pkg/a -> pkg/b' in error for error in errors)


def test_reach_allows_new_edge_and_inherited_sibling_reach():
    before, after = sibling_graphs()
    assert core.compare(before, after, policy(before)) == []


def test_reach_rejects_edge_that_deepens_the_importer():
    before = core.measure({'pkg/a': {'x'}, 'pkg/b': {'y'}},
                          {('pkg/a', 'x'), ('pkg/b', 'y')}, {'x': 1, 'y': 1})
    after = core.measure({'pkg/a': {'x', 'y'}, 'pkg/b': {'y'}},
                         set(map(tuple, before['edges'])) | {('pkg/a', 'pkg/b')}, before['loc'])
    errors = core.compare(before, after, policy(before))
    assert any('pkg/a -> pkg/b' in error and 'package max 1' in error for error in errors)


@pytest.mark.parametrize('count, fails', [(5, False), (6, True)])
def test_reach_count_growth_is_bounded_by_five(count, fails):
    tests = {f'pkg/{i}': {'x'} for i in range(8)}
    edges = {(test, 'x') for test in tests}
    before = core.measure(tests, edges, {'x': 1})
    after = core.measure(tests, edges | {(f'pkg/{i}', f'pkg/{i+1}') for i in range(count)}, {'x': 1})
    errors = core.compare(before, after, policy(before))
    assert bool(errors) == fails
    if fails:
        assert any('test_to_test count' in error and '6' in error and '5' in error for error in errors)


def test_count_uses_adapter_edge_groups_and_deduplicates_aliases():
    graph = core.measure({'a': set(), 'b': set()}, {('a', 'b')})
    graph['edge_groups']['test_to_test_imports'] = ['a -> b', 'helper -> b']
    assert core.baseline_record(graph)['test_to_test_count'] == 2


def test_count_keeps_its_minimum_and_policy_metadata():
    before, after = sibling_graphs()
    original = policy(before)
    grown = core.baseline_record(after, original)
    assert grown['test_to_test_count'] == 0
    assert grown['test_to_test'] == 'reach'
    assert grown['test_to_test_strict_at_count'] == 0
    assert grown['test_to_test_strict_by'] == '2026-11-16'
    original['test_to_test_count'] = 7
    assert core.baseline_record(after, original)['test_to_test_count'] == 1
    assert any('test_to_test_count may only shrink' in e for e in core.baseline_errors(
        {**original, 'test_to_test_count': 8}, original))


def test_reach_uses_package_override_and_retains_strict_one_line_switch():
    before, after = sibling_graphs()
    assert core.compare(before, after, policy(before), test_to_test='reach') == []
    strict = {**policy(before), 'test_to_test': 'strict'}
    assert core.baseline_errors(strict, policy(before)) == []
    # Already accepted edges at base must survive the switch; new ones fail.
    assert core.compare(after, after, strict, record_gains=True) == []
    assert core.compare(before, after, strict)


def test_reach_does_not_relax_conftest_edges_or_cycles():
    before, after = sibling_graphs()
    after['edge_groups']['conftest_source'] = ['conftest -> y']
    assert any('conftest_source' in e for e in core.compare(before, after, policy(before)))
    cyclic = core.measure(after['reach'], set(map(tuple, after['edges'])) | {('pkg/b', 'pkg/a')}, after['loc'])
    assert any('SCC grew or merged' in e for e in core.compare(before, cyclic, policy(before)))


def test_reach_summary_reports_total_and_reminds_without_switching(tmp_path, capsys):
    before, after = sibling_graphs()
    baseline = policy(before)
    baseline['test_to_test_strict_at_count'] = 1
    path = tmp_path / 'baseline.json'
    path.write_text(json.dumps(baseline))
    assert core.check(before, after, path) == 0
    output = capsys.readouterr().out
    assert 'test→test edges 0 → 1 (+1, reach)' in output
    assert 'switch test_to_test to strict' in output
    assert json.loads(path.read_text())['test_to_test'] == 'reach'


def test_deadline_reminder_has_deterministic_boundary():
    before, after = sibling_graphs()
    baseline = policy(before)
    assert core.policy_reminders(after, baseline, today=date(2026, 11, 15)) == []
    assert any('2026-11-16' in line for line in core.policy_reminders(after, baseline, today=date(2026, 11, 16)))


def test_invalid_mode_is_rejected():
    before, after = sibling_graphs()
    with pytest.raises(ValueError, match='strict.*reach'):
        core.compare(before, after, policy(before, 'ignore'))
