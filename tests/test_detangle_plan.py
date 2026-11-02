"""Runtime-weighted untangling contracts on bounded, synthetic graphs."""
from __future__ import annotations

import json

import pytest

from tools.bazel import detangle_plan as plan


def toy():
    return plan.Selection(
        edges={'fast': {'hub'}, 'slow': {'used'}, 'hub': {'used', 'unused'}, 'used': set(), 'unused': set()},
        roots={'fast': {'fast'}, 'slow': {'slow'}},
        data={'used': {'data.json'}},
    )


def test_cost_weights_files_and_keeps_fallback_separate():
    selection = toy()
    replay = [{'sha': 'a', 'changed': ['unused']}, {'sha': 'b', 'changed': ['used']},
              {'sha': 'c', 'changed': ['MODULE.bazel']}]
    result = plan.costs(selection.masks(), replay, {'fast': 1, 'slow': 60}, plan.Config())
    assert result['mean_seconds'] == pytest.approx(41)
    assert result['fallback_seconds'] == pytest.approx(61 / 3)
    assert result['graph_only_mean_seconds'] == 31
    assert result['floor_share'] == pytest.approx(61 / 123)
    assert result['fallback_merges'] == 1


def test_dynamic_selection_includes_test_itself_and_preserves_fallback():
    masks = plan.dynamic_masks({'fast': ['used'], 'slow': ['unused']}, ['fast', 'slow'])
    replay = [{'sha': 'a', 'changed': ['used']}, {'sha': 'b', 'changed': ['fast']},
              {'sha': 'c', 'changed': ['.github/workflows/gate.yml']}]
    result = plan.costs(masks, replay, {'fast': 1, 'slow': 60}, plan.Config())
    assert [row['seconds'] for row in result['merges']] == [1, 1, 61]


def test_bitset_delta_counts_shared_paths_once_and_keeps_data():
    selection = toy()
    assert selection.selected(['used', 'data.json']) == {'fast', 'slow'}
    cut = plan.Candidate('move these symbols down', (('hub', 'unused', 'import'),), 10,
                         ({'source': 'hub', 'target': 'unused', 'symbols': ['unused_name']},))
    replay = [{'sha': 'a', 'changed': ['unused']}, {'sha': 'b', 'changed': ['data.json']}]
    assert plan.projected_savings(selection, replay, {'fast': 1, 'slow': 60}, cut, plan.Config()) == .5
    assert selection.selected(['unused']) == {'fast'}


def test_greedy_rescores_overlapping_cuts_instead_of_counting_gain_twice():
    selection = plan.Selection(edges={'t': {'a'}, 'a': {'b'}, 'b': {'c'}, 'c': set()}, roots={'t': {'t'}})
    cuts = [plan.Candidate('move these symbols down', ((s, t, 'import'),), effort, ())
            for s, t, effort in [('a', 'b', 2), ('b', 'c', 1)]]
    ranked = plan.rank(selection, [{'sha': 'm', 'changed': ['c']}], {'t': 60}, cuts, plan.Config(), limit=20)
    assert len(ranked) == 1
    assert ranked[0]['edges'] == [['b', 'c', 'import']]
    assert ranked[0]['delta_seconds'] == 60
    assert ranked[0]['effort_lines'] == 1


def test_cycle_group_has_gain_when_each_edge_alone_has_none():
    selection = plan.Selection(edges={'t': {'a', 'b'}, 'a': {'b', 'x'}, 'b': {'a', 'x'}, 'x': set()},
                               roots={'t': {'t'}})
    replay = [{'sha': 'm', 'changed': ['x']}]
    cuts = [plan.Candidate('move these symbols down', ((s, 'x', 'import'),), 1, ()) for s in ('a', 'b')]
    assert all(plan.projected_savings(selection, replay, {'t': 9}, cut, plan.Config()) == 0 for cut in cuts)
    group = plan.Candidate('SCC sub-cut', tuple(edge for cut in cuts for edge in cut.edges), 2, ())
    assert plan.projected_savings(selection, replay, {'t': 9}, group, plan.Config()) == 9


def test_feedback_arc_set_breaks_a_cycle_deterministically():
    edges = {'a': {'b'}, 'b': {'c'}, 'c': {'a'}}
    cut = plan.feedback_cut(set(edges), edges, {('a', 'b'): 1, ('b', 'c'): 30, ('c', 'a'): 30})
    assert cut == {('a', 'b')}


def test_data_cut_does_not_remove_an_import_of_the_same_python_file():
    selection = plan.Selection(edges={'t': {'a'}, 'a': set()}, roots={'t': {'t'}}, data={'t': {'a'}})
    cut = selection.without({('t', 'a', 'data')})
    assert cut.selected(['a']) == {'t'}


def test_stopping_requires_measured_gate_and_both_second_conditions():
    assert plan.stopping(100, 95, 10, None, None)['gap_below_ten_percent']
    assert not plan.stopping(100, 70, 1, None, None)['best_below_two_percent_and_gate_near_floor']
    assert plan.stopping(100, 70, 1, 119, 100)['best_below_two_percent_and_gate_near_floor']
    assert not plan.stopping(100, 70, 3, 119, 100)['best_below_two_percent_and_gate_near_floor']


def test_timings_prefer_run_alone_then_hosted_and_report_missing(tmp_path):
    timings = tmp_path / 'alone.tsv'
    timings.write_text('path\talone_seconds\na\t60\n')
    weights, provenance = plan.runtime_weights(['a', 'b', 'c'], timings, {'a': 1, 'b': 2})
    assert weights == {'a': 60, 'b': 2, 'c': 2}
    assert provenance['missing'] == ['c']


def test_incomplete_coverage_cannot_be_reported_as_a_measured_floor(tmp_path):
    coverage = tmp_path / 'coverage.json'
    coverage.write_text(json.dumps({'tests': {'fast': {'files': ['used'], 'exit': 0}}}))
    with pytest.raises(ValueError, match='slow'):
        plan.load_dynamic(coverage, ['fast', 'slow'])


def test_symbol_evidence_tracks_alias_attributes_and_function_local_imports(tmp_path):
    source = tmp_path / 'a.py'
    source.write_text('import pkg.b as b\nfrom pkg.b import helper as h\ndef run():\n'
                      '    from pkg.b import local\n    return b.VALUE, h(), local()\n')
    result = plan.import_symbols(source.read_text(), 'pkg/a.py', 'pkg/b.py')
    assert result['symbols'] == ['VALUE', 'helper', 'local']
    assert result['functions'] == ['run']
    assert result['has_module_import']
