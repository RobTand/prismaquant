"""Identity regressions only: these fixtures are not timing evidence."""
import copy
import pytest
from prismaquant.joint_aura import identity_sha256
from prismaquant.native_execution_binding import (
    BOUND_SCHEMA, RAW_RECEIPT_SCHEMA, bind_execution_receipt,
    execution_panel_from_joint, resolve_execution_binding)


def fixture():
    joint = {'qname':'q', 'format':'f', 'source_weight':{'sha':'source'},
             'rendered_weight':{'sha':'render'}, 'activation':{'scale':None},
             'probe_identity_sha256':'probe'}
    final = {'schema':'tessera.native_dense_panel.v1','unit':'q','format':'f',
             'joint_operator_identity':joint,'joint_operator_identity_sha256':identity_sha256(joint),
             'cost_sha256':'cost','probe_identity_sha256':'probe',
             'wire':{'sha':'wire'},'phases':{'prefill':{'m':512}},
             'runtime':{'image':'actual'},'calibration_sha256':'calibration'}
    panel = execution_panel_from_joint(final)
    raw = {'schema':RAW_RECEIPT_SCHEMA,'status':'timing_admissible',
           'panel':panel,'panel_sha256':identity_sha256(panel),
           'phases':{'prefill':{'measurement':{'samples_ms':[1.,2.,3.]}}}}
    return final,raw


def test_binding_preserves_original_evidence_and_timing():
    final,raw=fixture(); before=copy.deepcopy(raw)
    bound=bind_execution_receipt(raw,final)
    assert bound['schema']==BOUND_SCHEMA
    assert raw==before==bound['raw_receipt']
    assert 'cost_sha256' not in raw['panel']
    view=resolve_execution_binding(bound,final)
    assert view['panel']==final
    assert view['phases']==raw['phases']
    assert raw==before


@pytest.mark.parametrize('coordinate',['wire','phases','runtime','calibration_sha256'])
def test_late_binding_refuses_changed_measured_coordinate(coordinate):
    final,raw=fixture();final[coordinate]='changed'
    with pytest.raises(ValueError,match='changes measured'):
        bind_execution_receipt(raw,final)


def test_late_binding_refuses_changed_weight_even_with_rehashed_final_joint():
    final,raw=fixture();final['joint_operator_identity']['rendered_weight']={'sha':'different'}
    final['joint_operator_identity_sha256']=identity_sha256(final['joint_operator_identity'])
    with pytest.raises(ValueError,match='changes measured'):
        bind_execution_receipt(raw,final)


def test_consumer_refuses_modified_raw_evidence_and_inconsistent_binding():
    final,raw=fixture();bound=bind_execution_receipt(raw,final)
    bound['raw_receipt']['phases']['prefill']['measurement']['samples_ms'][0]=0.01
    with pytest.raises(ValueError,match='raw receipt digest'):
        resolve_execution_binding(bound,final)
    bound['raw_receipt_sha256']=identity_sha256(bound['raw_receipt'])
    bound['panel_sha256']='different'
    with pytest.raises(ValueError,match='binding differs'):
        resolve_execution_binding(bound,final)


def test_static_reference_refuses_missing_or_other_quality_quantizer_build():
    from prismaquant.native_execution_binding import require_reference_quantizer
    activation={'static_contract':{'measured_as_served':True}}
    with pytest.raises(ValueError,match='require their registered'):
        require_reference_quantizer({},activation)
    identity={'backend':'registered_scaled_fp4_quant','image_content_sha256':'a'*64}
    data={'reference_served_quantizer':identity}
    assert require_reference_quantizer(data,activation,{'arithmetic':{'served_quantizer':identity}})==identity
    with pytest.raises(ValueError,match='differs from actual joint quality arithmetic'):
        require_reference_quantizer(data,activation,{'arithmetic':{'served_quantizer':{**identity,'image_content_sha256':'b'*64}}})
