"""Observable replay costs for runtime fixes and the narrowed fallback policy."""
import pytest
from tools.bazel import detangle_plan as plan


def test_cheaper_test_ranks_against_cuts_and_uses_measured_runtime():
    selection=plan.Selection({'fast': {'unused'}, 'slow': {'used'}, 'unused': set(), 'used': set()},
                             {'fast': {'fast'}, 'slow': {'slow'}})
    replay=[{'sha':'a','changed':['unused']},{'sha':'b','changed':['used']},
            {'sha':'c','changed':['MODULE.bazel']}]
    weights={'fast':3,'slow':60}
    candidates=[plan.Candidate('move these symbols down',(('fast','unused','import'),),1,())]
    candidates+=plan.cheaper_tests(weights, {'fast':2,'slow':30}, effort={'slow':2}, measured={'slow':6})
    ranked=plan.rank(selection,replay,weights,candidates,plan.Config())
    assert ranked[0]['kind']=='make test cheaper'
    assert ranked[0]['runtime_after']==[['slow',6]]
    assert ranked[0]['delta_seconds']==pytest.approx(36)
    assert ranked[0]['effort_lines']==2
    assert ranked[0]['symbols'][0]['measurement']=='measured'
    assert ranked[1]['delta_seconds']==1
    assert ranked[-1]['remaining_seconds']==pytest.approx(17/3)


def test_runtime_savings_rescore_after_a_cut_removes_selections():
    selection=plan.Selection({'t': {'x'}, 'x':set()},{'t':{'t'}})
    replay=[{'sha':'a','changed':['x']},{'sha':'b','changed':['t']}]
    cut=plan.Candidate('move these symbols down',(('t','x','import'),),1,())
    ranked=plan.rank(selection,replay,{'t':100},[cut]+plan.cheaper_tests({'t':100},{'t':10},effort={'t':10}),plan.Config())
    assert [r['delta_seconds'] for r in ranked]==[50,45]
    assert ranked[-1]['remaining_seconds']==5


def test_heavy_tests_report_static_and_dynamic_cost_per_merge():
    replay=[{'sha':'a','changed':['x']},{'sha':'b','changed':['MODULE.bazel']}]
    rows=plan.test_costs({'x':3}, {'x':1}, replay, {'a':2,'b':60},plan.Config())
    assert [r['path'] for r in rows]==['b','a']
    assert rows[0]['static_seconds_per_merge']==60
    assert rows[0]['dynamic_seconds_per_merge']==30
    assert rows[0]['static_selection_frequency']==1


@pytest.mark.parametrize('path',['.github/workflows/gate.yml','BUILD.bazel','nested/BUILD','MODULE.bazel',
    '.bazelrc','.bazelversion','pkg/defs.bzl','tools/bazel/detangle_plan.py','tools/ci/gate_run.py'])
def test_narrowed_fallback_only_covers_gate_and_bazel(path):
    assert plan.Config.narrowed().fallback(path)


@pytest.mark.parametrize('path',['uv.lock','.github/workflows/release.yml','.github/scripts/check_dist.py',
    '.mergify.yml','docs/ARCHITECTURE.md','pyproject.toml','pytest.ini','tests/conftest.py','tools/ci/import_fanout.py'])
def test_other_inputs_do_not_force_fallback(path):
    assert not plan.Config.narrowed().fallback(path)


def test_lock_changes_select_only_changed_wheels_and_their_graph_dependents():
    before='[[package]]\nname="numpy"\nversion="1"\n[[package]]\nname="torch"\nversion="2"\n'
    after=before.replace('version="1"','version="3"')
    assert plan.changed_wheels(before,after)=={'numpy'}
    selection=plan.Selection({'a':{'wheel:torch'},'wheel:torch':{'wheel:numpy'},'b':{'wheel:requests'}},
                             {'a':{'a'},'b':{'b'}})
    replay=[{'sha':'a','changed':['uv.lock'],'changed_wheels':['numpy']},
            {'sha':'b','changed':['uv.lock'],'changed_wheels':[]}]
    result=plan.costs(selection.masks(),replay,{'a':60,'b':2},plan.Config.narrowed())
    assert [r['seconds'] for r in result['merges']]==[60,0]
    assert result['fallback_seconds']==0
    assert plan.Scorer(selection,replay,{'a':60,'b':2},plan.Config.narrowed()).cost()==30


def test_lock_mapping_requires_a_diff_instead_of_silently_ignoring_it():
    with pytest.raises(ValueError,match='changed_wheels'):
        plan.costs({},[{'sha':'a','changed':['uv.lock']}],{'a':1},plan.Config.narrowed())


@pytest.mark.parametrize('warm',[float('nan'),-1,101])
def test_invalid_runtime_estimates_fail_closed(warm):
    with pytest.raises(ValueError):plan.cheaper_tests({'a':100},{'a':warm})
