"""History-free graph impact, including attribution and the minimum-only baseline."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

from tools.ci import graph_impact as impact


def graph(edges=None, roots=None, loc=None, contents=None):
    return impact.Snapshot(edges or {}, roots or {}, loc or {}, contents or {})


def base_record(snapshot, **overrides):
    return {**impact.baseline_record(impact.measure(snapshot)), **overrides}


def test_metrics_count_conftests_and_weight_source_lines():
    snapshot = graph({'t1': {'a'}, 't2': set(), 'conftest.py': {'b'}, 'a': {'b'}, 'b': set(), 'c': set()},
                     {'t1': {'t1', 'conftest.py'}, 't2': {'t2'}}, {'a': 1, 'b': 3, 'c': 2})
    live = impact.measure(snapshot)
    assert live['reach'] == {'t1': ['a', 'b'], 't2': []}
    assert live['dependents'] == {'a': 1, 'b': 1, 'c': 0}
    assert live['u_pct'] == pytest.approx(100 / 3)
    assert live['loc_u_pct'] == pytest.approx(100 / 3)
    assert live['reach_median'] == 1
    assert live['reach_p90'] == live['reach_max'] == 2
    weighted = impact.measure(graph(snapshot.edges, snapshot.roots, {'a': 1, 'b': 5, 'c': 0}))
    assert weighted['loc_u_pct'] == 50


def test_changed_test_budget_prints_shortest_path_to_farthest_hub():
    edges = {'tests/test_x.py': set(), 'tests/conftest.py': {'pkg/a.py'},
             'pkg/a.py': {'pkg/hub.py'}, 'pkg/hub.py': set()}
    roots = {'tests/test_x.py': {'tests/test_x.py', 'tests/conftest.py'}}
    before = graph(edges, roots, {'pkg/a.py': 1, 'pkg/hub.py': 1}, {'tests/test_x.py': 'old'})
    after = graph(edges, roots, before.loc, {'tests/test_x.py': 'changed'})
    errors = impact.compare(before, after, base_record(before, reach_budget=1))
    assert any('reach 2 exceeds budget 1' in error and
               'tests/test_x.py → tests/conftest.py → pkg/a.py → pkg/hub.py' in error for error in errors)
    assert not impact.compare(before, before, base_record(before, reach_budget=1))


def test_new_test_above_budget_fails_even_without_history():
    snapshot = graph({'tests/test_new.py': {'a'}, 'a': set()}, {'tests/test_new.py': {'tests/test_new.py'}}, {'a': 1})
    errors = impact.compare(None, snapshot, base_record(snapshot, reach_budget=0))
    assert any('reach 1 exceeds budget 0' in error for error in errors)


@pytest.mark.parametrize('edge, message', [
    (('tests/test_a.py', 'tests/test_b.py'), 'test_to_test_imports'),
    (('tests/conftest.py', 'pkg/a.py'), 'conftest_source_imports'),
])
def test_new_forbidden_edge_fails(edge, message):
    before = graph({edge[0]: set(), edge[1]: set()}, {}, {'pkg/a.py': 1})
    after = graph({edge[0]: {edge[1]}, edge[1]: set()}, {}, before.loc)
    errors = impact.compare(before, after, base_record(before))
    assert any(message in error and f'{edge[0]} -> {edge[1]}' in error for error in errors)


def test_uniform_regression_attributes_edges_and_fanout_growth():
    before = graph({'tests/test_a.py': {'a'}, 'a': set(), 'b': set()},
                   {'tests/test_a.py': {'tests/test_a.py'}}, {'a': 1, 'b': 1})
    after = graph({**before.edges, 'a': {'b'}}, before.roots, before.loc)
    errors = impact.compare(before, after, base_record(before, reach_budget=10))
    assert any('u_pct rose' in error and 'a -> b' in error and 'b +1' in error for error in errors)


def test_uniform_tolerance_does_not_loosen_recorded_minimum():
    live = {'u_pct': 20.5, 'loc_u_pct': 30.5, 'scc_max': 2, 'reach_p90': 4,
            'conftest_source_imports': [], 'test_to_test_imports': []}
    baseline = impact.baseline_record(live)
    baseline.update(u_pct=20, loc_u_pct=30)
    assert not impact.regressions(live, baseline)
    assert impact.baseline_record(live, baseline)['u_pct'] == 20


@pytest.mark.parametrize('old_edges,new_edge', [
    ({'a': {'b'}, 'b': {'a', 'c'}, 'c': set()}, ('c', 'a')),
    ({'a': {'b'}, 'b': {'a', 'c'}, 'c': {'d'}, 'd': {'c'}}, ('d', 'a')),
    # A smaller SCC grows while the largest component remains unchanged.
    ({'a': {'b'}, 'b': set(), 'x': {'y'}, 'y': {'z'}, 'z': {'x'}}, ('b', 'a')),
])
def test_every_scc_growth_or_merge_is_attributed(old_edges, new_edge):
    before = graph(old_edges)
    after = graph({**old_edges, new_edge[0]: old_edges[new_edge[0]] | {new_edge[1]}})
    errors = impact.compare(before, after, base_record(before))
    assert any('SCC grew or merged' in error and f'{new_edge[0]} -> {new_edge[1]}' in error for error in errors)


def test_an_unrelated_added_edge_is_not_blamed_for_cycle_growth():
    before = graph({'a': {'b'}, 'b': set(), 'x': set(), 'y': set()})
    after = graph({'a': {'b'}, 'b': {'a'}, 'x': {'y'}, 'y': set()})
    error = next(e for e in impact.compare(before, after, base_record(before)) if 'SCC grew' in e)
    assert 'b -> a' in error
    assert 'x -> y' not in error


def test_missing_recorded_gain_fails_and_recording_only_tightens():
    old = {'u_pct': 40, 'loc_u_pct': 50, 'scc_max': 4, 'reach_p90': 10,
           'conftest_source_imports': ['conftest.py -> a'], 'test_to_test_imports': []}
    baseline = impact.baseline_record(old)
    live = {**old, 'u_pct': 30, 'scc_max': 2, 'reach_p90': 8, 'conftest_source_imports': []}
    assert any('u_pct fell' in error for error in impact.stale_entries(live, baseline))
    record = impact.baseline_record(live, baseline)
    assert record['u_pct'] == 30
    assert record['reach_budget'] == 8
    assert record['conftest_source_imports'] == []


def test_allowlist_requires_reason_and_only_shrinks():
    assert impact.allowlist_errors({'reach:tests/test_a.py': 'integration contract'}, {})
    assert impact.allowlist_errors({'reach:tests/test_a.py': ''}, {'reach:tests/test_a.py': ''})
    assert impact.allowlist_errors({'reach:tests/test_a.py': 'new reason'}, {'reach:tests/test_a.py': 'old reason'})
    assert impact.allowlist_errors({}, {'reach:tests/test_a.py': 'integration contract'}) == []


def test_existing_reasoned_exception_applies_only_to_its_test():
    snapshot = graph({'tests/test_a.py': {'a'}, 'tests/test_b.py': {'a'}, 'a': set()},
                     {'tests/test_a.py': {'tests/test_a.py'}, 'tests/test_b.py': {'tests/test_b.py'}}, {'a': 1})
    errors = impact.compare(None, snapshot, base_record(snapshot, reach_budget=0),
                            allowlist={'reach:tests/test_a.py': 'integration contract'})
    assert not any('tests/test_a.py reach' in error for error in errors)
    assert any('tests/test_b.py reach' in error for error in errors)


def test_summary_format_on_pass_and_failure():
    before = graph({'tests/test_a.py': {'a'}, 'a': set()}, {'tests/test_a.py': {'tests/test_a.py'}}, {'a': 1})
    after = graph({**before.edges, 'tests/test_b.py': set()},
                  {**before.roots, 'tests/test_b.py': {'tests/test_b.py'}}, before.loc)
    assert impact.summary(before, after, 1) == (
        'U 100.0% → 50.0%, SCC max 1 → 1, new tests 1 (reach 0 ≤ budget 1), new edges 0, none to a hub')
    assert 'no base available' in impact.summary(None, after, 0)
    assert 'budget 0' in impact.summary(before, after, 0)


def write_tree(root, files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return root


def test_generator_adapter_follows_launches_and_scoped_conftests(tmp_path):
    write_tree(tmp_path, {'tests/test_a.py': 'import subprocess\nsubprocess.run(["python", "tools/run.py"])\n',
                          'tests/sub/test_b.py': '', 'tests/sub/conftest.py': 'import pkg.hub\n',
                          'tools/run.py': 'import pkg.hub\n', 'pkg/hub.py': 'x = 1\n'})
    snapshot = impact.load_snapshot(tmp_path, population=['tests/test_a.py', 'tests/sub/test_b.py'],
                                    source_roots=('pkg/', 'tools/'))
    assert impact.measure(snapshot)['reach'] == {
        'tests/sub/test_b.py': ['pkg/hub.py'], 'tests/test_a.py': ['pkg/hub.py', 'tools/run.py']}


def test_no_git_still_enforces_absolute_figures_and_prints_summary(tmp_path, capsys):
    write_tree(tmp_path, {'tests/test_a.py': 'import tools.hub\n', 'tools/hub.py': 'x = 1\n',
                          'tools/ci/gate_durations.json': '{"files": {}}',
                          'tools/ci/gate_quarantine.json': '{}', 'tests/fixtures/graph_impact_allowlist.json': '{}'})
    baseline = tmp_path / 'tests/fixtures/graph_impact_baseline.json'
    baseline.write_text(json.dumps(base_record(impact.load_snapshot(tmp_path), reach_budget=0)))
    assert impact.main(['--root', str(tmp_path)]) == 1
    output = capsys.readouterr()
    assert 'U ' in output.out and 'no base available' in output.out
    assert 'reach 1 exceeds budget 0' in output.err


def test_baseline_writer_refuses_losses(tmp_path, capsys):
    write_tree(tmp_path, {'tests/test_a.py': 'import tools.hub\n', 'tools/hub.py': '',
                          'tools/ci/gate_durations.json': '{"files": {}}', 'tools/ci/gate_quarantine.json': '{}',
                          'tests/fixtures/graph_impact_allowlist.json': '{}'})
    snapshot = impact.load_snapshot(tmp_path)
    baseline = tmp_path / 'tests/fixtures/graph_impact_baseline.json'
    baseline.write_text(json.dumps(base_record(snapshot, u_pct=0)))
    old = baseline.read_bytes()
    assert impact.main(['--root', str(tmp_path), '--write-baseline']) == 1
    assert baseline.read_bytes() == old
    assert 'refusing to record' in capsys.readouterr().err


def test_git_base_uses_merge_base_and_detects_changed_test(tmp_path):
    write_tree(tmp_path, {'tests/test_a.py': 'x = 1\n', 'tools/ci/gate_durations.json': '{"files": {}}',
                          'tools/ci/gate_quarantine.json': '{}'})
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=tmp_path, text=True).strip()
    git('init', '-q'); git('config', 'user.email', 'test@example.invalid'); git('config', 'user.name', 'Test')
    git('add', '.'); git('commit', '-qm', 'base')
    sha = git('rev-parse', 'HEAD')
    (tmp_path / 'tests/test_a.py').write_text('x = 2\n')
    baseline_snapshot = impact.snapshot_at(tmp_path, sha)
    current = impact.load_snapshot(tmp_path)
    assert baseline_snapshot.contents['tests/test_a.py'] != current.contents['tests/test_a.py']


def test_gate_always_selects_graph_impact_for_ordinary_python_change():
    from tools.ci import gate_select
    root = Path(__file__).resolve().parents[1]
    selected = gate_select.create_plan(root, ['prismaquant/allocator.py'],
                                       index={'tests/test_graph_impact.py': (set(), set()),
                                              'tests/test_unrelated.py': (set(), set())},
                                       durations={'files': {}}, quarantine={})
    assert 'tests/test_graph_impact.py' in {row['path'] for row in selected['selected']}


def test_graph_impact_gate(capsys):
    assert impact.main([]) == 0
    assert 'U ' in capsys.readouterr().out
