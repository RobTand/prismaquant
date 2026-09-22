"""Actual bound policy → selected fp32 scale file → export admission."""
import copy
import hashlib
import json
import pickle
from pathlib import Path

import pytest

from prismaquant import tessera_export_lane as export
from prismaquant.joint_served_activation import derive_policy, policy_group, FORMAT
from test_routed_executed_scale_grouping import _layer_config, _scales_file


def write(path, value, binary=False):
    raw = pickle.dumps(value) if binary else json.dumps(value, sort_keys=True).encode()
    path.write_bytes(raw)
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def selected_policy(tmp_path, monkeypatch):
    from prismaquant.production_weight_cache import ProductionWeightCache
    from prismaquant import model_profiles
    monkeypatch.setattr(model_profiles, 'detect_profile', lambda _: None)
    names = [f'model.language_model.layers.8.mlp.experts.{expert}.{role}_proj'
             for expert in range(2) for role in ('gate', 'up', 'down')]
    census = write(tmp_path/'census.json', {'model': '/synthetic', 'unit_shapes': {n: [16,16] for n in names}})
    cache = ProductionWeightCache(weights={}, levers={}, activation_max_abs={name: float(i+1) for i,name in enumerate(names)},
                                  metadata={'inputs': {'census': census}})
    prepared = write(tmp_path/'prepared.json', {'status': 'complete',
        'calibration_input': {'shape': [512,512], 'calibration_sha256': '1'*64},
        'production_cache': write(tmp_path/'cache.pkl', cache, True), 'source_model_identity': {'synthetic': True},
        'formats_by_qname': {n: ['TESSERA_E4M3_K1_R1024','BF16'] for n in names}})
    policy = derive_policy(prepared); bound = write(tmp_path/'policy.json', policy)
    units = {name: policy_group(policy,name,FORMAT)[1]['input_global_scale'] for name in names}
    path = _layer_config(tmp_path, units=units, grouping=policy['executed_grouping'])
    payload = json.loads(path.read_text())
    payload['__prismaquant__']['tessera_activation_static_scales']['served_activation_policy'] = bound
    path.write_text(json.dumps(payload))
    return path, bound, policy, units


def test_canonical_executed_policy_admits_exact_priced_file(selected_policy, tmp_path):
    path, bound, policy, units = selected_policy
    scales = _scales_file(tmp_path, units)
    result = export.require_priced_export_inputs(path, input_scales_path=scales)
    assert result['activation_scale_grouping'] == 'executed_group.v1'
    assert result['activation_scale_grouping_qualified'] is False
    assert result['served_activation_policy'] == bound
    assert result['activation_scale_pricing_input_equality_verified'] is True


@pytest.mark.parametrize('change', ['policy_bytes','foreign_member','wrong_group','wrong_scale','missing_policy'])
def test_executed_declaration_cannot_authorize_unbound_values(selected_policy,tmp_path,change):
    path, bound, policy, units = selected_policy
    data=json.loads(path.read_text()); block=data['__prismaquant__']['tessera_activation_static_scales']
    if change=='policy_bytes': Path(bound['path']).write_text('{}')
    elif change=='foreign_member':
        old=next(iter(units)); new=old.replace('layers.8','layers.9')
        units[new]=units.pop(old); data[new]=data.pop(old); block['units']=units
    elif change=='wrong_group': block['activation_scale_grouping']['member_count']-=1
    elif change=='wrong_scale':
        name=next(iter(units)); units[name]*=2; block['units'][name]=units[name]
    else: del block['served_activation_policy']
    path.write_text(json.dumps(data)); scales=_scales_file(tmp_path,units)
    with pytest.raises((ValueError,RuntimeError)):
        export.require_priced_export_inputs(path,input_scales_path=scales)


def test_actual_selected_scale_writer_then_export_gate(selected_policy,tmp_path):
    from prismaquant.tessera_selected_scales import write_selected_scales
    from safetensors import safe_open
    path,bound,policy,units=selected_policy
    binding={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    out=tmp_path/'selected-inputs'
    receipt=write_selected_scales(binding,out)
    assert not (out/'hessian_capture.pt').exists()
    assert receipt['served_activation_policy']==bound
    scales=receipt['input_scales']['path']
    result=export.require_priced_export_inputs(path,input_scales_path=scales)
    assert result['activation_scale_pricing_input_equality_verified'] is True
    with safe_open(scales,framework='pt') as handle:
        assert set(handle.keys())=={name+'.input_global_scale' for name in units}
    before=Path(scales).read_bytes()
    with pytest.raises(FileExistsError): write_selected_scales(binding,out)
    assert Path(scales).read_bytes()==before


def test_writer_refuses_changed_assignment_before_creating_output(selected_policy,tmp_path):
    from prismaquant.tessera_selected_scales import write_selected_scales
    path,*_=selected_policy
    binding={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    path.write_text(path.read_text()+' ')
    out=tmp_path/'refused'
    with pytest.raises(ValueError): write_selected_scales(binding,out)
    assert not out.exists()


def test_writer_refuses_a_formula_label_not_bound_by_policy(selected_policy,tmp_path):
    from prismaquant.tessera_selected_scales import write_selected_scales
    from prismaquant.nvfp4_activation_contract import FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY
    path,*_=selected_policy
    data=json.loads(path.read_text())
    data['__prismaquant__']['tessera_activation_static_scales']['input_global_scale_policy']=FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY
    path.write_text(json.dumps(data))
    binding={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(export.TesseraExportLaneError,match='formula'):
        write_selected_scales(binding,tmp_path/'wrong-formula')
    assert not (tmp_path/'wrong-formula').exists()
