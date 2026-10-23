"""Admission charges the activation row cap the campaign will actually execute."""
import pytest

from prismaquant import tessera_campaign_rows as rows
from prismaquant.tessera_campaign_cli import campaign_argument_parser


def test_resident_demand_tracks_effective_cli_cap_and_execution_default(monkeypatch):
    monkeypatch.setattr(rows, '_model_bytes', lambda model: 1024)
    census = {'unit_shapes': {'one': [3, 4], 'two': [5, 8]}}
    spec = {'model': '/source', 'campaign_argv': [], 'headroom_gb': 0}
    execution_default = campaign_argument_parser().get_default('max_act_rows')
    before = rows.row_memory_demand(spec, ['one', 'two'], census)
    assert before['plan_bytes'] == 1024 + 4 * (4**2 + 8**2) + 4 * 12 * execution_default
    after = rows.row_memory_demand({**spec, 'campaign_argv': [
        '--max-act-rows=1536', '--max-act-rows', '2048']}, ['one', 'two'], census)
    assert after['demand_bytes'] - before['demand_bytes'] == 4 * 12 * (2048 - execution_default)


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('argv', [[], ['--max-act-rows', '1024']])
def test_shadow_activation_cap_cannot_change_admission_without_changing_execution(
        monkeypatch, streaming, argv):
    monkeypatch.setattr(rows, '_model_bytes', lambda model: 0)
    spec = {'model': '/source', 'campaign_argv': (['--streaming'] if streaming else []) + argv,
            'max_act_rows': 2}
    with pytest.raises(rows.RowClassRefused, match='max_act_rows differs from execution'):
        rows.row_memory_demand(spec, ['one'], {'unit_shapes': {'one': [3, 4]}})


def test_streamed_plan_uses_the_same_default_and_accepts_a_matching_shadow(monkeypatch):
    from prismaquant import autoscale
    seen = []
    monkeypatch.setattr(autoscale, 'streamed_calibration_resources',
                        lambda model, **options: seen.append(options) or {'memory_bytes': 0})
    census = {'unit_shapes': {'one': [3, 4]}, 'counts': {'one': 10000}}
    spec = {'model': '/source', 'campaign_argv': ['--streaming']}
    rows.streamed_resource_plan(spec, census, ['one'])
    rows.streamed_resource_plan({**spec, 'campaign_argv': ['--streaming', '--max-act-rows=2048'],
                                 'max_act_rows': 2048}, census, ['one'])
    assert [item['max_act_rows'] for item in seen] == [
        campaign_argument_parser().get_default('max_act_rows'), 2048]


def test_dispatch_refuses_old_reservation_for_a_larger_actual_row_cap(monkeypatch):
    from tools import dispatch_tessera_campaign as dispatch
    monkeypatch.setattr(rows, '_model_bytes', lambda model: 1024)
    census = {'unit_shapes': {'one': [3, 8]}, 'counts': {'one': 2**27}}
    spec = {'model': '/source', 'campaign_argv': [], 'headroom_gb': 0}
    old = rows.row_memory_demand(spec, ['one'], census)['mem_gb']
    row = {'argv': ['python3', '-m', 'prismaquant.tessera_campaign',
                    '--model', '/source', '--max-act-rows', str(2**27)],
           'demand': {'mem_gb': old}}
    with pytest.raises(dispatch.DemandRefused, match='own argv derives'):
        dispatch.verify_row_demand(spec, census, row, label='larger-activation-row')
