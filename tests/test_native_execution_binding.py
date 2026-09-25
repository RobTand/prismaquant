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


def _registered_record(**overrides):
    from prismaquant.nvfp4_activation_contract import SERVED_QUANTIZER_IDENTITY_SCHEMA
    record = {'schema': SERVED_QUANTIZER_IDENTITY_SCHEMA,
              'backend': 'registered_scaled_fp4_quant', 'op': 'scaled_fp4_quant',
              'platform': 'cuda:sm_121', 'torch': '2.13.0+cu130', 'torch_git': 'abc',
              'vllm': '0.20.0', 'image_content_sha256': 'a' * 64}
    record.update(overrides)
    return record


@pytest.mark.parametrize('reference_has_kernel', [False, True])
def test_static_reference_ignores_the_dequant_kernel_field(reference_has_kernel):
    """PQ #1211: ``dequant_kernel`` is recorded but is not a reuse axis.

    A native reference recorded before the field existed must still bind a
    joint row that carries it, and the other way round: the fused and the
    Torch dequantisation give bit-identical output, so they price one number.
    """
    from prismaquant.native_execution_binding import require_reference_quantizer
    from prismaquant.nvfp4_activation_contract import SERVED_QUANTIZER_DEQUANT_KERNEL
    activation = {'static_contract': {'measured_as_served': True}}
    with_kernel = _registered_record(dequant_kernel=SERVED_QUANTIZER_DEQUANT_KERNEL)
    without_kernel = _registered_record()
    reference, probe = ((with_kernel, without_kernel) if reference_has_kernel
                        else (without_kernel, with_kernel))
    data = {'reference_served_quantizer': reference}
    assert require_reference_quantizer(
        data, activation, {'arithmetic': {'served_quantizer': probe}}) == reference


@pytest.mark.parametrize('axis', ['schema', 'backend', 'op', 'platform', 'torch',
                                  'torch_git', 'vllm', 'image_content_sha256'])
def test_static_reference_still_refuses_a_reuse_axis_difference(axis):
    from prismaquant.native_execution_binding import require_reference_quantizer
    from prismaquant.nvfp4_activation_contract import SERVED_QUANTIZER_DEQUANT_KERNEL
    activation = {'static_contract': {'measured_as_served': True}}
    reference = _registered_record()
    probe = _registered_record(dequant_kernel=SERVED_QUANTIZER_DEQUANT_KERNEL,
                               **{axis: 'other'})
    with pytest.raises(ValueError, match='differs from actual joint quality arithmetic'):
        require_reference_quantizer({'reference_served_quantizer': reference}, activation,
                                    {'arithmetic': {'served_quantizer': probe}})


@pytest.mark.parametrize('probe', [{}, {'arithmetic': {}},
                                   {'arithmetic': {'served_quantizer': None}}])
def test_static_reference_refuses_a_probe_without_a_served_quantizer(probe):
    from prismaquant.native_execution_binding import require_reference_quantizer
    activation = {'static_contract': {'measured_as_served': True}}
    with pytest.raises(ValueError, match='differs from actual joint quality arithmetic'):
        require_reference_quantizer({'reference_served_quantizer': _registered_record()},
                                    activation, probe)
