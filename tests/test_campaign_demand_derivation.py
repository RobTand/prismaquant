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

import json
import math

import pytest

GIB = 1024 ** 3

# A plan whose rounding slack sits below the measured process floor, so the
# demand is only right if it charges the floor rather than inheriting it from
# ``ceil``. Both example rows inspected for #390 were on this side of the line.
PLAN_BYTES = 20 * GIB + 173_741_824

# The relaunch row this issue was filed on: ``extension-r1024-02`` row-0074,
# whose selected-anchor plan measured 108.63 GB while its manifest declared
# ``mem_gb: 102``.
GLM_ROW_0074_PLAN_BYTES = 108_630_000_000


def _dispatch():
    from tools import dispatch_tessera_campaign as dispatch
    return dispatch


def _margin_bytes():
    from prismaquant.memory_management import CaptureMemoryGuard
    return CaptureMemoryGuard.MARGIN_BYTES


def _fixed_plan(monkeypatch, plan_bytes=PLAN_BYTES):
    """Inject the phase plan, so the derivation is the only variable."""
    dispatch = _dispatch()
    monkeypatch.setattr(dispatch, '_streamed_resource_plan',
        lambda spec, census, members, *, selected_source: dict(
            memory_bytes=plan_bytes, selected_layers=['0'],
            baseline_policy='declared-headroom-pre-run-measured-in-row'))
    return dispatch


def _census(tmp_path):
    census = dict(model='/source',
                  anchor_groups={'u:layers.0.proj': ['layers.0.proj']},
                  layer_stride=1, unit_shapes={'layers.0.proj': [3, 4]},
                  counts={'layers.0.proj': 9})
    (tmp_path / 'census.json').write_text(json.dumps(census))
    return census


def _units_file(tmp_path, name='row-0000.json'):
    path = tmp_path / name
    path.write_text(json.dumps({
        'schema': 'prismaquant.tessera_campaign_units.v1', 'model': '/source',
        'layer_stride': 1,
        'groups': [{'key': 'u:layers.0.proj', 'members': ['layers.0.proj']}]}))
    return path


def _manifest_row(tmp_path, *, mem_gb, extra_argv=()):
    units = _units_file(tmp_path)
    return {
        'argv': ['python3', '-u', '-m', 'prismaquant.tessera_campaign',
                 '--model', '/source', '--units', str(units), '--streaming',
                 *extra_argv],
        'cwd': str(tmp_path),
        'demand': {'gpu': 1, 'cpu': 4, 'mem_gb': int(mem_gb)},
        'env': {},
        'tags': ['gb10'],
    }


def _spec(tmp_path, **extra):
    spec = dict(model='/source', campaign_argv=['--streaming'], cwd=str(tmp_path),
                python='python3', env={}, cpus=1, **extra)
    (tmp_path / 'spec.json').write_text(json.dumps(spec))
    return spec


# ---------------------------------------------------------------------------
# the derivation
# ---------------------------------------------------------------------------

def test_the_demand_is_the_plan_plus_the_measured_floor_plus_the_guard_margin(
        monkeypatch, tmp_path):
    """Three terms, and the demand is short without any one of them.

    The plan is deltas; the floor is what the process already held before the
    first delta; the margin is what the guard holds back from the cap on every
    check. A demand covering only the first is admitted and then refused by the
    row itself.
    """
    dispatch = _fixed_plan(monkeypatch)
    spec = _spec(tmp_path)
    demand = dispatch._row_memory_demand(spec, ['layers.0.proj'], _census(tmp_path),
                                         selected_source=True)
    expected_bytes = (PLAN_BYTES + dispatch.DEFAULT_PROCESS_BASELINE_BYTES
                      + _margin_bytes())
    assert demand['plan_bytes'] == PLAN_BYTES
    assert demand['process_baseline_bytes'] == dispatch.DEFAULT_PROCESS_BASELINE_BYTES
    assert demand['guard_margin_bytes'] == _margin_bytes()
    assert demand['demand_bytes'] == expected_bytes
    assert demand['mem_gb'] == math.ceil(expected_bytes / GIB)
    assert dispatch._row_memory_gb(spec, ['layers.0.proj'], _census(tmp_path),
                                   selected_source=True) == demand['mem_gb']


def test_the_margin_is_read_from_the_guard_rather_than_restated(monkeypatch, tmp_path):
    """Move the guard's margin and the demand moves with it.

    A second copy of 2 GiB would pass every assertion above and still drift the
    day the guard's margin changes, so what is tested is the coupling: the
    constant is mutated on the guard, not in the fixture.
    """
    from prismaquant import memory_management
    dispatch = _fixed_plan(monkeypatch)
    spec = _spec(tmp_path)
    census = _census(tmp_path)
    before = dispatch._row_memory_demand(spec, ['layers.0.proj'], census,
                                         selected_source=True)
    monkeypatch.setattr(memory_management.CaptureMemoryGuard, 'MARGIN_BYTES',
                        memory_management.CaptureMemoryGuard.MARGIN_BYTES + 4 * GIB)
    after = dispatch._row_memory_demand(spec, ['layers.0.proj'], census,
                                        selected_source=True)
    assert after['guard_margin_bytes'] - before['guard_margin_bytes'] == 4 * GIB
    assert after['demand_bytes'] - before['demand_bytes'] == 4 * GIB
    assert after['mem_gb'] - before['mem_gb'] == 4


def test_a_spec_reservation_overrides_the_fleet_default_and_the_plan_says_which(
        monkeypatch, tmp_path):
    """A reader of a plan can tell an operator's number from this tool's.

    The default is a reading taken on other rows of another campaign. A spec
    that knows its own box overrides it, including with zero, and the
    ``baseline_policy`` names which of the two the plan carries.
    """
    from prismaquant import autoscale
    dispatch = _fixed_plan(monkeypatch)
    census = _census(tmp_path)

    default_bytes, default_policy = dispatch._process_baseline(_spec(tmp_path))
    assert default_bytes == dispatch.DEFAULT_PROCESS_BASELINE_BYTES
    assert default_policy == autoscale.BASELINE_POLICY_MEASURED_DEFAULT_RESERVATION

    declared_bytes, declared_policy = dispatch._process_baseline(
        _spec(tmp_path, process_baseline_bytes=3 * GIB))
    assert declared_bytes == 3 * GIB
    assert declared_policy == autoscale.BASELINE_POLICY_EXPLICIT_RESERVATION

    # A declared zero is a choice, not an absence: it reserves nothing and says
    # the spec asked for nothing.
    zero_bytes, zero_policy = dispatch._process_baseline(
        _spec(tmp_path, process_baseline_bytes=0))
    assert zero_bytes == 0
    assert zero_policy == autoscale.BASELINE_POLICY_EXPLICIT_RESERVATION
    assert dispatch._row_memory_demand(
        _spec(tmp_path, process_baseline_bytes=0), ['layers.0.proj'], census,
        selected_source=True)['demand_bytes'] == PLAN_BYTES + _margin_bytes()


def test_the_resident_source_branch_charges_the_same_two_terms(monkeypatch, tmp_path):
    """A floor and a margin exist whether or not the row streams."""
    dispatch = _dispatch()
    monkeypatch.setattr(dispatch, '_model_bytes', lambda model: 10 * GIB)
    census = dict(unit_shapes={'layers.0.proj': [4, 8]})
    spec = dict(model='/source', campaign_argv=[], headroom_gb=3, max_act_rows=2)
    body = 10 * GIB + 8 ** 2 * 4 + 8 * 2 * 4
    demand = dispatch._row_memory_demand(spec, ['layers.0.proj'], census)
    assert demand['plan_bytes'] == body
    assert demand['mem_gb'] == math.ceil(
        (body + dispatch.DEFAULT_PROCESS_BASELINE_BYTES + _margin_bytes()) / GIB) + 3


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------

def test_an_under_declared_row_is_refused_and_the_message_carries_the_numbers(
        monkeypatch, tmp_path):
    """The refusal names every term, because the operator has to fix one of them."""
    dispatch = _fixed_plan(monkeypatch)
    spec = _spec(tmp_path)
    census = _census(tmp_path)
    derived = dispatch._row_memory_demand(spec, ['layers.0.proj'], census,
                                          selected_source=True)['mem_gb']
    row = _manifest_row(tmp_path, mem_gb=derived - 1)
    with pytest.raises(dispatch.DemandRefused) as refusal:
        dispatch.verify_row_demand(spec, census, row, label='row-0000')
    message = str(refusal.value)
    for number in (PLAN_BYTES, dispatch.DEFAULT_PROCESS_BASELINE_BYTES,
                   _margin_bytes(),
                   PLAN_BYTES + dispatch.DEFAULT_PROCESS_BASELINE_BYTES + _margin_bytes(),
                   (derived - 1) * GIB):
        assert str(number) in message, f'{number} is missing from {message!r}'
    assert 'row-0000' in message
    assert str(derived) in message


def test_a_row_declaring_what_it_derives_passes(monkeypatch, tmp_path):
    dispatch = _fixed_plan(monkeypatch)
    spec, census = _spec(tmp_path), _census(tmp_path)
    derived = dispatch._row_memory_demand(spec, ['layers.0.proj'], census,
                                          selected_source=True)['mem_gb']
    record = dispatch.verify_row_demand(spec, census,
                                        _manifest_row(tmp_path, mem_gb=derived),
                                        label='row-0000')
    assert record['mem_gb'] == derived == record['declared_mem_gb']
    assert record['declared_bytes'] == derived * GIB


def test_the_check_reads_the_rows_own_argv_not_the_specs(monkeypatch, tmp_path):
    """The defect was a manifest argv the planning spec never had.

    ``--publication-overlap-bytes`` reached the relaunch rows through the
    manifest while the spec kept ``mem_gb: 102``. A check that re-derived from
    the spec would have agreed with the manifest and refused nothing.
    """
    dispatch = _dispatch()
    seen = {}

    def plan(spec, census, members, *, selected_source):
        seen['argv'] = list(spec['campaign_argv'])
        overlap = ('--publication-overlap-bytes' in spec['campaign_argv'])
        return dict(memory_bytes=PLAN_BYTES + (2 * GIB if overlap else 0),
                    selected_layers=['0'])

    monkeypatch.setattr(dispatch, '_streamed_resource_plan', plan)
    spec, census = _spec(tmp_path), _census(tmp_path)
    row = _manifest_row(tmp_path, mem_gb=999, extra_argv=(
        '--publication-overlap-bytes', '2147483648'))
    record = dispatch.verify_row_demand(spec, census, row, label='row-0000')
    assert '--publication-overlap-bytes' in seen['argv']
    assert record['plan_bytes'] == PLAN_BYTES + 2 * GIB


def test_a_row_wider_than_the_declared_box_is_refused(monkeypatch, tmp_path):
    """Capacity is a parameter, so the test never reads the fleet's files."""
    dispatch = _fixed_plan(monkeypatch)
    spec, census = _spec(tmp_path), _census(tmp_path)
    derived = dispatch._row_memory_demand(spec, ['layers.0.proj'], census,
                                          selected_source=True)['mem_gb']
    row = _manifest_row(tmp_path, mem_gb=derived)
    assert dispatch.verify_row_demand(spec, census, row, label='row-0000',
                                      box_memory_gb=derived)['mem_gb'] == derived
    with pytest.raises(dispatch.DemandRefused, match='no admission can come'):
        dispatch.verify_row_demand(spec, census, row, label='row-0000',
                                   box_memory_gb=derived - 1)


def test_the_relaunch_row_is_now_refused_at_submit_rather_than_after_admission(
        monkeypatch, tmp_path):
    """The consequence, stated as a test rather than left to be discovered.

    Row-0074's plan measured 108.63 GB. Charged the worst floor this fleet has
    measured and the guard's own margin, it derives 105 GiB, which is above the
    104 GiB a Spark declares. Under this rule the row is refused at submit
    instead of dying twenty seconds after admission. That is the intended
    outcome: the allowance is not shrunk to make the row fit.
    """
    dispatch = _fixed_plan(monkeypatch, plan_bytes=GLM_ROW_0074_PLAN_BYTES)
    spec, census = _spec(tmp_path), _census(tmp_path)
    demand = dispatch._row_memory_demand(spec, ['layers.0.proj'], census,
                                         selected_source=True)
    assert demand['mem_gb'] == 105
    with pytest.raises(dispatch.DemandRefused, match='no admission can come'):
        dispatch.verify_row_demand(spec, census,
                                   _manifest_row(tmp_path, mem_gb=102),
                                   label='row-0074', box_memory_gb=104)


def test_every_under_declared_row_is_named_at_once(monkeypatch, tmp_path):
    dispatch = _fixed_plan(monkeypatch)
    spec, census = _spec(tmp_path), _census(tmp_path)
    derived = dispatch._row_memory_demand(spec, ['layers.0.proj'], census,
                                          selected_source=True)['mem_gb']
    rows = [_manifest_row(tmp_path, mem_gb=derived),
            _manifest_row(tmp_path, mem_gb=derived - 1),
            _manifest_row(tmp_path, mem_gb=1)]
    with pytest.raises(dispatch.DemandRefused) as refusal:
        dispatch.verify_manifest_demands(spec, census, rows)
    assert '2 of 3 rows' in str(refusal.value)
    assert dispatch.verify_manifest_demands(
        spec, census, rows[:1])[0]["mem_gb"] == derived


# ---------------------------------------------------------------------------
# the submit path
# ---------------------------------------------------------------------------

def test_submit_refuses_an_under_declared_manifest_before_submitting_it(
        monkeypatch, tmp_path):
    """Nothing reaches pbcampaign that the row's own guard would decline."""
    dispatch = _fixed_plan(monkeypatch)
    spec, census = _spec(tmp_path), _census(tmp_path)
    submitted = []
    monkeypatch.setattr(dispatch, '_pbcampaign',
                        lambda *a, **k: submitted.append(a) or 0)
    derived = dispatch._row_memory_demand(spec, ['layers.0.proj'], census,
                                          selected_source=True)['mem_gb']
    (tmp_path / 'manifest.json').write_text(json.dumps(
        [_manifest_row(tmp_path, mem_gb=derived - 1)]))
    (tmp_path / 'plan.json').write_text(json.dumps(
        {'spec': str(tmp_path / 'spec.json'), 'census': str(tmp_path / 'census.json')}))
    with pytest.raises(dispatch.DemandRefused):
        dispatch.main(['submit', '--workspace', str(tmp_path)])
    assert submitted == []

    (tmp_path / 'manifest.json').write_text(json.dumps(
        [_manifest_row(tmp_path, mem_gb=derived)]))
    assert dispatch.main(['submit', '--workspace', str(tmp_path)]) == 0
    assert len(submitted) == 1


def test_submit_refuses_rather_than_skipping_the_check_it_cannot_run(tmp_path):
    """An unverifiable manifest is refused, not submitted unchecked."""
    dispatch = _dispatch()
    (tmp_path / 'manifest.json').write_text('[]')
    (tmp_path / 'plan.json').write_text(json.dumps({'model': '/source'}))
    with pytest.raises(dispatch.DemandRefused, match='neither --spec'):
        dispatch.main(['submit', '--workspace', str(tmp_path)])


def test_check_reports_every_row_it_verified(monkeypatch, tmp_path, capsys):
    dispatch = _fixed_plan(monkeypatch)
    spec, census = _spec(tmp_path), _census(tmp_path)
    derived = dispatch._row_memory_demand(spec, ['layers.0.proj'], census,
                                          selected_source=True)['mem_gb']
    (tmp_path / 'manifest.json').write_text(json.dumps(
        [_manifest_row(tmp_path, mem_gb=derived)]))
    assert dispatch.main(['check', '--workspace', str(tmp_path),
                          '--spec', str(tmp_path / 'spec.json'),
                          '--census', str(tmp_path / 'census.json')]) == 0
    printed = capsys.readouterr().out
    assert f'derives {derived} GiB' in printed
    assert str(PLAN_BYTES) in printed


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
