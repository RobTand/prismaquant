"""Workspace migration preserves the row ownership and ordinary seed gates."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def fixture(tmp_path):
    import tools.dispatch_tessera_campaign as d
    root = tmp_path/'seed'
    root.mkdir()
    census = {'model': '/model', 'anchor_groups': {'a': ['a'], 'b': ['b']}, 'layer_stride': 1,
              'unit_shapes': {'a': [8, 8], 'b': [8, 8]}}
    (root/'census.json').write_text(json.dumps(census))
    rows = []
    for name in ('a', 'b'):
        row = root/name
        row.mkdir()
        selection = {'schema': d.UNITS_SCHEMA, 'model': '/model', 'layer_stride': 1,
                     'groups': [{'key': name, 'members': [name]}]}
        (row/'units.json').write_text(json.dumps(selection))
        rows.append({'row_id': name, 'groups': [name], 'dir': str(row), 'units': str(row/'units.json')})
    # A partial checkpoint has a manifest and may have only some unit shards.
    (root/'a/cost.anchors.json').write_text(json.dumps({'identity_sha256': 'a'*64}))
    plan = {'schema': d.PLAN_SCHEMA, 'model': '/model', 'census': str(root/'census.json'),
            'calibration_cache': None, 'rows': rows}
    (root/'plan.json').write_text(json.dumps(plan))
    return d, root, census, plan


def test_plan_reuses_partial_seed_and_prices_unstarted_row(tmp_path, monkeypatch):
    d, root, census, _ = fixture(tmp_path)
    workspace = tmp_path/'new'
    workspace.mkdir()
    (workspace/'census.json').write_text(json.dumps(census))
    spec = {'model': '/model', 'campaign_argv': [], 'cwd': '/repo',
            'python': 'python', 'env': {}}
    monkeypatch.setattr(d, 'load_spec', lambda _: spec)
    monkeypatch.setattr(d, '_row_memory_gb', lambda *a, **k: 1)
    monkeypatch.setattr(d, '_row', lambda spec, argv, **kw: {'argv': argv, 'demand': {'mem_gb': 1}})
    args = SimpleNamespace(spec='unused', workspace=str(workspace), calibration_cache=None,
        stack_sample=None, seed_checkpoint=None, seed_wire_dir=None, seed_workspace=str(root),
        groups_per_row=1, rows_per_box=1, timeout_s=60, stack_sample_seed=0, audit_rate=10, probe=None)
    assert d.cmd_plan(args) == 0
    manifest = json.loads((workspace/'manifest.json').read_text())
    assert manifest[0]['argv'][manifest[0]['argv'].index('--seed-checkpoint')+1] == str(root/'a/cost.anchors.json')
    assert '--seed-checkpoint' not in manifest[1]['argv']
    plan = json.loads((workspace/'plan.json').read_text())
    assert plan['rows'][0]['seed']['identity_sha256'] == 'a'*64
    assert len(plan['seed_workspace']['plan_sha256']) == 64
    before = (workspace/'manifest.json').read_bytes()
    args.groups_per_row = 2
    with pytest.raises(RuntimeError, match='selection differs'):
        d.cmd_plan(args)
    assert (workspace/'manifest.json').read_bytes() == before


@pytest.mark.parametrize('field', ['model', 'census', 'capture', 'duplicate'])
def test_seed_workspace_refuses_incompatible_source(tmp_path, field):
    d, root, census, plan = fixture(tmp_path)
    if field == 'model': plan['model'] = '/other'
    elif field == 'census': census = {**census, 'layer_stride': 2}
    elif field == 'capture': plan['calibration_cache'] = {'path': '/other', 'sha256': 'b'*64}
    else: plan['rows'].append(copy.deepcopy(plan['rows'][0]))
    (root/'plan.json').write_text(json.dumps(plan))
    with pytest.raises(RuntimeError, match='seed workspace'):
        d._seed_workspace_rows(root, census=census, calibration_cache=None)


def test_seed_selection_refuses_changed_sample(tmp_path):
    d, root, census, _ = fixture(tmp_path)
    rows, _ = d._seed_workspace_rows(root, census=census, calibration_cache=None)
    selection = json.loads((root/'a/units.json').read_text())
    selection['groups'][0]['sampled'] = []
    with pytest.raises(RuntimeError, match='selection differs'):
        d._seed_for_selection(rows, ['a'], selection)
