"""Synthetic, language-neutral contracts and the live per-PR graph check."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.ci import graph_impact as impact

ROOT = Path(__file__).resolve().parents[1]


def measure(tests=None, edges=(), loc=None):
    return impact.measure(tests or {}, set(edges), loc)


def baseline(snapshot, **overrides):
    return {**impact.baseline_record(snapshot), **overrides}


def test_metrics_include_unreached_sources_and_line_weights():
    snapshot = measure({'t1': {'a', 'b'}, 't2': set()}, {('t1', 'a'), ('a', 'b')}, {'a': 1, 'b': 3, 'c': 2})
    assert snapshot['reach'] == {'t1': ['a', 'b'], 't2': []}
    assert snapshot['dependents'] == {'a': 1, 'b': 1, 'c': 0}
    assert snapshot['u_pct'] == pytest.approx(100 / 3)
    assert snapshot['loc_u_pct'] == pytest.approx(100 / 3)
    assert snapshot['reach_median'] == 1
    assert snapshot['reach_p90'] == snapshot['reach_max'] == 2
    weighted = measure({'t1': {'a', 'b'}, 't2': set()}, (), {'a': 1, 'b': 5, 'c': 0})
    assert weighted['loc_u_pct'] == 50


def test_empty_graph_and_zero_loc_are_defined():
    empty = measure()
    assert empty['u_pct'] == empty['loc_u_pct'] == empty['scc_max'] == empty['reach_max'] == 0
    assert measure({'t': {'empty'}}, loc={'empty': 0})['loc_u_pct'] == 0
    with pytest.raises(ValueError, match='LOC'):
        measure({'t': {'a'}}, loc={'a': -1})


def test_changed_test_budget_prints_shortest_path_to_farthest_hub():
    edges = {('test', 'fixture'), ('fixture', 'pkg/a'), ('pkg/a', 'pkg/hub')}
    snapshot = measure({'test': {'pkg/a', 'pkg/hub'}}, edges, {'pkg/a': 1, 'pkg/hub': 1})
    errors = impact.compare(snapshot, snapshot, baseline(snapshot, reach_budget=1), changed_tests={'test'})
    assert any('reach 2 exceeds budget 1' in error and 'test → fixture → pkg/a → pkg/hub' in error for error in errors)
    assert not impact.compare(snapshot, snapshot, baseline(snapshot, reach_budget=2))


def test_new_test_budget_and_no_history_absolute_check():
    before = measure({}, loc={'a': 1})
    after = measure({'new_test': {'a'}}, {('new_test', 'a')}, {'a': 1})
    for base in (None, before):
        errors = impact.compare(base, after, baseline(after, reach_budget=0))
        assert any('reach 1 exceeds budget 0' in error for error in errors)


@pytest.mark.parametrize('group,edge', [('test_to_test', 't1 -> t2'), ('conftest_source_imports', 'fixture -> a')])
def test_new_forbidden_edge_fails(group, edge):
    before = measure()
    before['edge_groups'][group] = []
    after = measure()
    after['edge_groups'][group] = [edge]
    assert any(f'new {group} edge: {edge}' in error for error in impact.compare(before, after, baseline(before)))


def test_test_to_test_edges_are_discovered_without_language_assumptions():
    after = measure({'//rust:a_test': set(), '//rust:b_test': set()}, {('//rust:a_test', '//rust:b_test')})
    assert after['edge_groups']['test_to_test'] == ['//rust:a_test -> //rust:b_test']


def test_uniform_regression_attributes_edges_and_fanout_growth():
    before = measure({'t': {'a'}}, {('t', 'a')}, {'a': 1, 'b': 1})
    after = measure({'t': {'a', 'b'}}, {('t', 'a'), ('a', 'b')}, {'a': 1, 'b': 1})
    errors = impact.compare(before, after, baseline(before, reach_budget=10))
    assert any('u_pct rose' in error and 'a -> b' in error and 'b +1' in error for error in errors)


def test_tolerated_noise_does_not_loosen_the_minimum():
    snapshot = measure({'t': {'a'}}, loc={'a': 1})
    previous = baseline(snapshot)
    previous['figures']['u_pct'] = 99.5
    assert not impact.regressions(snapshot, previous)
    assert impact.baseline_record(snapshot, previous)['figures']['u_pct'] == 99.5


@pytest.mark.parametrize('old_edges,added', [
    ({('a', 'b'), ('b', 'a'), ('b', 'c')}, ('c', 'a')),
    ({('a', 'b'), ('b', 'a'), ('b', 'c'), ('c', 'd'), ('d', 'c')}, ('d', 'a')),
    ({('a', 'b'), ('x', 'y'), ('y', 'z'), ('z', 'x')}, ('b', 'a')),
])
def test_every_scc_growth_or_merge_is_attributed(old_edges, added):
    before = measure(edges=old_edges)
    after = measure(edges=old_edges | {added})
    errors = impact.compare(before, after, baseline(before))
    assert any('SCC grew or merged' in error and f'{added[0]} -> {added[1]}' in error for error in errors)


def test_unrelated_added_edge_is_not_blamed_for_cycle_growth():
    before = measure(edges={('a', 'b')})
    after = measure(edges={('a', 'b'), ('b', 'a'), ('x', 'y')})
    error = next(e for e in impact.compare(before, after, baseline(before)) if 'SCC grew' in e)
    assert 'b -> a' in error and 'x -> y' not in error


def test_missing_gain_fails_and_recording_tightens_budget():
    before = measure({'t': {'a', 'b'}}, {('a', 'b'), ('b', 'a')}, {'a': 1, 'b': 1})
    after = measure({'t': {'a'}}, loc={'a': 1, 'b': 1})
    previous = baseline(before)
    assert any('u_pct fell' in error for error in impact.stale_entries(after, previous))
    record = impact.baseline_record(after, previous)
    assert record['figures']['u_pct'] == 50
    assert record['figures']['scc_max'] == 1
    assert record['reach_budget'] == 1


def test_baseline_and_allowlist_only_shrink():
    previous = baseline(measure())
    current = {**previous, 'reach_budget': 10}
    assert 'reach budget may only shrink' in impact.baseline_errors(current, previous)
    exception = {'reach:integration': 'integration contract'}
    assert impact.allowlist_errors(exception, {})
    assert impact.allowlist_errors({'reach:integration': ''}, exception)
    assert impact.allowlist_errors({'reach:integration': 'changed reason'}, exception)
    assert impact.allowlist_errors({}, exception) == []


def test_reasoned_exception_is_scoped_to_one_test():
    snapshot = measure({'a_test': {'a'}, 'b_test': {'a'}}, loc={'a': 1})
    errors = impact.compare(None, snapshot, baseline(snapshot, reach_budget=0),
                            allowlist={'reach:a_test': 'integration contract'})
    assert not any('a_test reach' in error for error in errors)
    assert any('b_test reach' in error for error in errors)


def test_summary_format_on_pass_and_failure():
    before = measure({'t1': {'a'}}, {('t1', 'a')}, {'a': 1})
    after = measure({'t1': {'a'}, 't2': set()}, {('t1', 'a')}, {'a': 1})
    assert impact.summary(before, after, 1) == (
        'U 100.0% → 50.0%, SCC max 1 → 1, new tests 1 (reach 0 ≤ budget 1), new edges 0, none to a hub')
    assert 'no base available' in impact.summary(None, after, 0)
    assert 'reach 1, 0 > budget 0' in impact.summary(None, after, 0)


def test_check_prints_summary_on_failure_and_preserves_baseline(tmp_path, capsys):
    snapshot = measure({'test': {'a'}}, {('test', 'a')}, {'a': 1})
    path = tmp_path / 'baseline.json'
    path.write_text(json.dumps(baseline(snapshot, reach_budget=0)))
    old = path.read_bytes()
    assert impact.check(None, snapshot, path, write_baseline=True) == 1
    assert path.read_bytes() == old
    output = capsys.readouterr()
    assert 'U no base available' in output.out
    assert 'refusing to record' in output.err and 'reach 1 exceeds budget 0' in output.err


def test_bazel_mermaid_blame_is_printed_on_failure(tmp_path, capsys):
    snapshot = measure({'test': {'a'}}, {('test', 'a')}, {'a': 1})
    path = tmp_path / 'baseline.json'
    path.write_text(json.dumps(baseline(snapshot, reach_budget=0)))
    assert impact.check(None, snapshot, path, explain=lambda test: 'graph TD; test-->a') == 1
    assert '```mermaid\ngraph TD; test-->a\n```' in capsys.readouterr().err


def test_vendored_core_hash():
    actual = hashlib.sha256((ROOT / 'tools/ci/graph_impact.py').read_bytes()).hexdigest()
    recorded = (ROOT / 'tools/ci/graph_impact.sha256').read_text().split()[0]
    assert actual == recorded, 'Update the recorded hash only when the core changes in all four repos'


def test_graph_impact_gate():
    from tools.ci import graph_impact_pq
    assert graph_impact_pq.main([]) == 0


def test_bazel_diff_reader_keeps_source_labels_and_excludes_manual(tmp_path):
    path = tmp_path / 'deps.json'
    path.write_text(json.dumps({'//t:fast': ['//lib:core'], '//t:manual': ['//other:x'],
                                '//lib:core': ['//lib:core.rs'], '//lib:core.rs': [], '//other:x': []}))
    tests, edges, loc = impact.read_deps(path, {'//t:fast', '//t:manual'},
                                        {'//lib:core.rs', '//unused:c.rs'}, manual_tests={'//t:manual'})
    assert tests == {'//t:fast': {'//lib:core.rs'}}
    assert ('//t:fast', '//lib:core') in edges
    assert loc == {'//lib:core.rs': 1, '//unused:c.rs': 1}
    assert impact.measure(tests, edges, loc)['u_pct'] == 50


def test_bazel_diff_reader_rejects_graph_missing_source_labels(tmp_path):
    path = tmp_path / 'deps.json'
    path.write_text(json.dumps({'//t:fast': ['//lib:core'], '//lib:core': []}))
    with pytest.raises(ValueError, match='source.file labels.*query'):
        impact.read_deps(path, {'//t:fast'}, {'//lib:core.rs'})


def test_bazel_query_fallback_reads_source_files_and_manual_tags(tmp_path):
    path = tmp_path / 'query.json'
    records = [{'type': 'RULE', 'rule': {'name': '//t:fast', 'ruleClass': 'rust_test',
                                        'ruleInput': ['//lib:core'], 'attribute': []}},
               {'type': 'RULE', 'rule': {'name': '//t:manual', 'ruleClass': 'py_test',
                                        'ruleInput': ['//lib:a.rs'],
                                        'attribute': [{'name': 'tags', 'stringListValue': ['manual']}]}},
               {'type': 'RULE', 'rule': {'name': '//lib:core', 'ruleClass': 'rust_library',
                                        'ruleInput': ['//lib:a.rs']}},
               {'type': 'SOURCE_FILE', 'sourceFile': {'name': '//lib:a.rs'}}]
    path.write_text('\n'.join(json.dumps(record) for record in records))
    tests, edges, loc = impact.read_query(path)
    assert tests == {'//t:fast': {'//lib:a.rs'}}
    assert ('//lib:core', '//lib:a.rs') in edges
    assert loc == {'//lib:a.rs': 1}


def test_pq_adapter_respects_generator_launch_and_conftest_rules(tmp_path):
    from tools.ci import graph_impact_pq as pq
    files = {'tests/test_a.py': 'import subprocess\nsubprocess.run(["python", "tools/run.py"])\n',
             'tests/sub/test_b.py': '', 'tests/sub/conftest.py': 'import prismaquant.hub\n',
             'tools/run.py': 'import prismaquant.hub\n', 'prismaquant/hub.py': 'x = 1\n'}
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    snapshot = pq.load_snapshot(tmp_path, population=['tests/test_a.py', 'tests/sub/test_b.py'], legacy=False)
    assert snapshot['reach'] == {'tests/sub/test_b.py': ['prismaquant/hub.py'],
                                  'tests/test_a.py': ['prismaquant/hub.py', 'tools/run.py']}


def test_pq_no_git_reports_no_base_and_enforces_absolute_budget(tmp_path, capsys):
    from tools.ci import graph_impact_pq as pq
    files = {'tests/test_a.py': 'import tools.hub\n', 'tools/hub.py': 'x = 1\n',
             'tools/ci/gate_durations.json': '{"files": {}}', 'tools/ci/gate_quarantine.json': '{}',
             'tests/fixtures/graph_impact_allowlist.json': '{}'}
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    snapshot = pq.load_snapshot(tmp_path)
    (tmp_path / 'tests/fixtures/graph_impact_baseline.json').write_text(json.dumps(pq.record(snapshot)))
    recorded = tmp_path / 'tests/fixtures/graph_impact_baseline.json'
    data = json.loads(recorded.read_text()); data['reach_budget'] = 0; recorded.write_text(json.dumps(data))
    assert pq.main(['--root', str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert 'no base available' in output.out and 'reach 1 exceeds budget 0' in output.err


def test_gate_always_selects_graph_impact():
    from tools.ci import gate_select
    plan = gate_select.create_plan(ROOT, ['prismaquant/allocator.py'],
                                  index={'tests/test_graph_impact.py': (set(), set()),
                                         'tests/test_unrelated.py': (set(), set())},
                                  durations={'files': {}}, quarantine={})
    assert 'tests/test_graph_impact.py' in {row['path'] for row in plan['selected']}
