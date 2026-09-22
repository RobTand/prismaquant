"""Extended metadata preserves historical layer extents and complete control head."""
import copy
import pytest
from tools.prepare_extended_joint_quanta import extend_parent


def test_extension_parent_only_augments_head_and_keeps_exact_layer_body():
    parent = {'schema': 'prismaquant.prismabuild.data_manifest.v1',
        'entries': [{'path': '/old-plan', 'offset': 0, 'bytes': 10, 'sha256': None},
                    {'path': '/source', 'offset': 0, 'bytes': 30, 'sha256': None}],
        'entry_count': 2, 'total_bytes': 40, 'annotations': {'plan_sha256': 'old',
            'campaign_scope': {'full512': True}, 'layers': [0],
            'phases': [{'name': 'head', 'bytes': 10, 'cumulative_bytes': 10},
                       {'name': 'layer-0', 'bytes': 30, 'cumulative_bytes': 40}]}}
    original = copy.deepcopy(parent)
    new = {'path': '/new-plan', 'sha256': 'new'}
    added = [{'path': '/new-plan', 'offset': 0, 'bytes': 20, 'sha256': None}]
    kwargs = dict(old_plan={'sha256': 'old'}, new_plan=new,
        prepared={'formats_by_qname': {'q': ['A8', 'A4', 'BF16']}}, extension={'path': '/proof'})
    result = extend_parent(parent, added + added, **kwargs)
    assert parent == original
    assert result['entries'][-1] == original['entries'][-1]
    assert result['entry_count'] == 3 and result['total_bytes'] == 60
    assert result['annotations']['phases'] == [
        {'name': 'head', 'bytes': 30, 'cumulative_bytes': 30},
        {'name': 'layer-0', 'bytes': 30, 'cumulative_bytes': 60}]
    assert result['annotations']['campaign_scope'] == {'full512': True}
    assert result['annotations']['executable_prepared_inputs_required'] is True
    assert result['annotations']['measured_cells'] == 2
    parent['annotations']['plan_sha256'] = 'foreign'
    with pytest.raises(ValueError, match='scientific plan'):
        extend_parent(parent, added, **kwargs)


def _stage_b_spec(tmp_path, wait):
    """The reviewed local-scratch spec's shape, with a chosen staged-range wait."""
    import json
    scratch = tmp_path / 'scratch'
    spec = {'container': {'content_sha256': 'd'*64, 'image': 'prismaquant-glm-derivative:test',
                          'mounts': [{'readonly': False, 'source': str(scratch), 'target': str(scratch)}]},
            'container_admission_reference': 'content:sha256:' + 'b'*64,
            'cpu_memory_gb': 28,
            'env': {'PRISMAQUANT_MAX_GPU_MEM_GB': '72', 'PRISMAQUANT_PROD_ACT_SCALES': '0',
                    'PRISMAQUANT_STAGED_RANGE_WAIT_S': str(wait),
                    'PRISMAQUANT_STAGE_B_COTANGENT_ROOT': str(scratch),
                    'PRISMAQUANT_STAGE_B_COTANGENT_MAX_BYTES': str(36 * 1024 ** 3)}}
    path = tmp_path / 'spec.json'
    path.write_text(json.dumps(spec))
    return path, spec


_POLICY = {'limits': {'host_bytes': 28 * 1024 ** 3, 'gpu_bytes': 72 * 1024 ** 3,
                      'physical_bytes': 100 * 1024 ** 3}}


def test_stage_b_spec_wait_must_sit_below_the_chunk_grace(tmp_path):
    """The reviewed spec a52b5359... seals a 900 s wait; every quantum row refuses it (#990)."""
    from tools.prepare_extended_joint_quanta import check_stage_b_spec
    path, spec = _stage_b_spec(tmp_path, 899)
    check_stage_b_spec(path, spec, _POLICY)
    path, spec = _stage_b_spec(tmp_path, 900)
    with pytest.raises(ValueError, match='progress grace'):
        check_stage_b_spec(path, spec, _POLICY)


def test_stage_b_spec_envelope_must_equal_the_resource_policy(tmp_path):
    from tools.prepare_extended_joint_quanta import check_stage_b_spec
    path, spec = _stage_b_spec(tmp_path, 600)
    with pytest.raises(ValueError, match='resource policy'):
        check_stage_b_spec(path, spec, {'limits': {**_POLICY['limits'], 'gpu_bytes': 64 * 1024 ** 3}})
