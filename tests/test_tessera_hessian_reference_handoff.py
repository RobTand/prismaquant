"""Canonical H handoffs union commitments and authenticate only consumed H."""
import copy
import sys
import json
from pathlib import Path

import pytest
import torch

from prismaquant import tessera_calibration_cache as cc
from prismaquant import tessera_campaign as campaign
from prismaquant import tessera_export_lane as export
from test_tessera_calibration_cache import canonical_fields, identity
from test_tessera_priced_export_inputs import TRIPLE, _assignment

POLICY=dict(schema='tessera.hessian_reference_load.v1', max_metadata_bytes=1024**2,
            max_file_bytes=1024**2,max_hessian_bytes=1024**2)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from tools import dispatch_tessera_campaign as dispatch


@pytest.fixture
def handoff(tmp_path):
    model=tmp_path/'source';model.mkdir()
    (model/'config.json').write_text('{}');(model/'model.safetensors').write_bytes(b'fixture')
    census=dict(model=str(model),counts={'a':5,'b':7},max_abs={'a':4.,'b':8.},
        unit_shapes={'a':[3,2],'b':[3,2]},layer_stride=1,anchor_groups={'u:a':['a'],'u:b':['b']},**canonical_fields())
    path=tmp_path/'census.json';path.write_text(json.dumps(census))
    calibration=dict(TRIPLE,model=str(model),seqlen=8,source='fixture')
    capture_id=identity(path,calibration=calibration,max_act_rows=2)
    H={'a':torch.eye(2)*13,'b':torch.eye(2)*25};X={n:torch.ones(2,2) for n in H}
    record=cc.publish_capture(tmp_path/'capture',census_path=path,identity=capture_id,acts=X,
        hessians=H,counts=census['counts'],maxima=census['max_abs'])
    return dict(H=H,census=census,census_path=path,calibration=calibration,record=record,tmp=tmp_path)


def write_row(f,name):
    root=f['tmp']/name;root.mkdir()
    return campaign.write_export_inputs(root,hessians={name:f['H'][name]},
        hessian_rows=f['census']['counts'],hessian_identity=f['calibration'],
        static_scales={},static_scale_policy='fixture',hessian_reference=dict(
            canonical_capture=f['record'],census_path=f['census_path'],load_policy=POLICY))


def test_row_handoff_has_exact_old_seal_and_no_h_sidecar(handoff,monkeypatch):
    f=handoff
    monkeypatch.setattr(torch,'save',lambda *a,**kw:pytest.fail('H sidecar materialization'))
    path,_,digest=write_row(f,'a')
    assert path.name=='hessian_capture.references.json'
    assert not list(path.parent.glob('*.pt*'))
    assert digest==export.hessian_capture_sha256({'a':f['H']['a']},{**f['calibration'],'hessian_role':'fit'})
    with cc.open_hessian_reference(path) as owner:
        assert owner.receipt()['verified_units']==[]
        torch.testing.assert_close(owner['a'],f['H']['a'],rtol=0,atol=0)


def rows(f):
    dirs={};payloads={}
    for name in f['H']:
        path,_,digest=write_row(f,name)
        binding=cc.hessian_reference_binding(f['record']['sha256'],cc.sha256(f['census_path']))
        dirs[name]=path.parent.parent/name
        cache=path.parent/'cache';cache.mkdir();path.replace(cache/path.name)
        hessian=dict(TRIPLE,supplied=True,capture_sha256=digest,reference_binding=binding)
        payloads[name]=dict(costs={name:{'format':{'hessian_identity':hessian}}},provenance=dict(
            calibration_cache=f['record'],hessian=hessian,
            unit_selection={'groups':[{'members':[name]}]}))
    return dirs,payloads


def test_dispatch_union_never_reads_h_and_matches_legacy_full_seal(handoff,monkeypatch):
    f=handoff;dirs,payloads=rows(f)
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('eager H load during metadata merge'))
    path,_,digest=dispatch.merge_export_inputs(dirs,payloads,out_cache=f['tmp']/'merged',
        identity=f['calibration'],policy='fixture',static_scales={},census=f['census'])
    assert digest==export.hessian_capture_sha256(f['H'],{**f['calibration'],'hessian_role':'fit'})
    with cc.open_hessian_reference(path) as owner:
        assert set(owner)==set(f['H']) and len(owner.descriptor['rows'])==2
        assert owner.receipt()['loaded_entries']==0


@pytest.mark.parametrize('defect',['row_seal','row_binding','canonical','roster','mixed'])
def test_dispatch_refuses_unbound_reference_rows(handoff,defect):
    f=handoff;dirs,payloads=rows(f);p=payloads['a']
    if defect=='row_seal':p['costs']['a']['format']['hessian_identity']['capture_sha256']='0'*64
    elif defect=='row_binding':p['provenance']['hessian']['reference_binding']['census_sha256']='0'*64
    elif defect=='canonical':p['provenance']['calibration_cache']={**f['record'],'sha256':'0'*64}
    elif defect=='roster':p['provenance']['unit_selection']['groups'][0]['members']=['b']
    else:p['provenance']['hessian'].pop('reference_binding')
    with pytest.raises(dispatch.MergeRefused):
        dispatch.merge_export_inputs(dirs,payloads,out_cache=f['tmp']/'merged',
            identity=f['calibration'],policy='fixture',static_scales={},census=f['census'])
    assert not (f['tmp']/'merged/hessian_capture.references.json').exists()


def test_export_gate_binds_reference_without_claiming_unconsumed_payloads(handoff,monkeypatch):
    f=handoff;path,_,digest=write_row(f,'a')
    assignment=_assignment(f['tmp'],formats={'a':'TESSERA_E4M3_K1_R1024'},capture_sha256=digest)
    payload=json.loads(assignment.read_text())
    payload['__prismaquant__']['tessera_hessian']['reference_binding']=cc.hessian_reference_binding(f['record']['sha256'],cc.sha256(f['census_path']))
    assignment.write_text(json.dumps(payload))
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('gate eagerly consumes H'))
    result=export.require_priced_export_inputs(assignment,hessian_path=path)
    assert result['hessian_payload_verification']=='deferred_to_consumption'
    assert result['hessian_verified_units']==[]
    payload['__prismaquant__']['tessera_hessian']['reference_binding']['census_sha256']='0'*64
    assignment.write_text(json.dumps(payload))
    with pytest.raises(export.TesseraExportLaneError,match='binding differs'):
        export.require_priced_export_inputs(assignment,hessian_path=path)


@pytest.mark.parametrize('mode',['inapplicable','invalid'])
def test_cli_reference_policy_refuses_before_any_source_intake(tmp_path,capsys,mode):
    args=['--model',str(tmp_path/'absent'),'--out',str(tmp_path/'cost.pkl'),
          '--cache-dir',str(tmp_path/'cache'),'--export-hessian-reference-policy']
    if mode=='invalid':
        args += [json.dumps({**POLICY,'max_file_bytes':0}),'--streaming','--units','a',
                 '--calibration-cache',str(tmp_path/'absent.json'),'--calibration-cache-sha256','a'*64]
    else:args += [json.dumps(POLICY)]
    with pytest.raises(SystemExit) as error:campaign.main(args)
    assert error.value.code==2
    assert ('requires selected reuse' if mode=='inapplicable' else 'invalid byte bounds') in capsys.readouterr().err
    assert not (tmp_path/'cache').exists()


def test_uniform_identity_carries_binding_and_refuses_mixed_modes():
    from prismaquant.tessera_menu import assert_uniform_hessian_identity
    binding=cc.hessian_reference_binding('a'*64,'b'*64)
    base=dict(TRIPLE,supplied=True,capture_sha256='c'*64,reference_binding=binding)
    costs={'a':{'TESSERA_E4M3_K1_R1024':{'hessian_identity':copy.deepcopy(base)},'TESSERA_E4M3_K1_R512':{'hessian_identity':copy.deepcopy(base)}}}
    assert assert_uniform_hessian_identity(costs)['reference_binding']==binding
    costs['a']['TESSERA_E4M3_K1_R512']['hessian_identity']['reference_binding']['census_sha256']='d'*64
    with pytest.raises(ValueError,match='mixes Hessian identities'):assert_uniform_hessian_identity(costs)
    costs['a']['TESSERA_E4M3_K1_R512']['hessian_identity'].pop('reference_binding')
    with pytest.raises(ValueError,match='mixes Hessian identities'):assert_uniform_hessian_identity(costs)


def test_sampled_materialization_refuses_reference_before_source_or_h_load(tmp_path,monkeypatch):
    import hashlib,pickle
    from prismaquant import tessera_materialization as materialization
    cost=tmp_path/'cost.pkl';cost.write_bytes(pickle.dumps({'provenance':{'hessian':{'reference_binding':cc.hessian_reference_binding('a'*64,'b'*64)}}}))
    request=tmp_path/'request.json';request.write_text(json.dumps(dict(schema=materialization.REQUEST_SCHEMA,
        status='selection_only_nonexportable',cost_path=str(cost),cost_sha256=hashlib.sha256(cost.read_bytes()).hexdigest())))
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('H load before reference materialization refusal'))
    with pytest.raises(RuntimeError,match='no bounded reference union yet'):materialization._request(request)


def test_selected_public_cli_publishes_references_without_changing_capture(monkeypatch,tmp_path):
    from test_selected_source_authentication import test_selected_campaign_hashes_only_consumed_shards_and_preserves_identity
    original=campaign.main
    def reference_main(argv):
        return original([*argv,'--export-hessian-reference-policy',json.dumps(POLICY)])
    monkeypatch.setattr(campaign,'main',reference_main)
    test_selected_campaign_hashes_only_consumed_shards_and_preserves_identity(monkeypatch,tmp_path)
    path=tmp_path/'cache/hessian_capture.references.json'
    assert path.is_file() and not (path.parent/'hessian_capture.pt').exists()
    with cc.open_hessian_reference(path) as owner:
        assert owner.receipt()['loaded_entries']==0
        assert owner.binding()['canonical_capture_sha256']==cc.sha256(tmp_path/'capture/capture_manifest.json')


def test_selected_cli_reuses_published_hessian_commitments(monkeypatch, tmp_path):
    """The source about to price must seal without another population scan."""
    from tessera import cached_unit
    sources = []
    original_memo = campaign._activation_kwargs_memo
    original_write = campaign.write_export_inputs
    original_label = campaign.contract_source_label
    published = False
    checked = []

    def memo(source, *args, **kwargs):
        sources.append(source)
        return original_memo(source, *args, **kwargs)

    def write(*args, **kwargs):
        nonlocal published
        result = original_write(*args, **kwargs)
        published = True
        return result

    def label():
        # The public fixture reaches the empty-menu gate after export inputs
        # are committed. Exercise the same source the anchor loop would use.
        if published:
            with monkeypatch.context() as patch:
                patch.setattr(cached_unit, 'tensor_identity', lambda *_a, **_k:
                    pytest.fail('resident capture was rehashed after its commitments were published'))
                checked.append(sources[-1].capture_sha256())
        return original_label()

    monkeypatch.setattr(campaign, '_activation_kwargs_memo', memo)
    monkeypatch.setattr(campaign, 'write_export_inputs', write)
    monkeypatch.setattr(campaign, 'contract_source_label', label)
    test_selected_public_cli_publishes_references_without_changing_capture(monkeypatch, tmp_path)
    assert checked
    descriptor = json.loads((tmp_path/'cache/hessian_capture.references.json').read_text())
    assert checked == [descriptor['capture_sha256']]
