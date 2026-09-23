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



def _parent(plan_sha256):
    return {'schema': 'prismaquant.prismabuild.data_manifest.v1',
        'entries': [{'path': '/old-plan', 'offset': 0, 'bytes': 10, 'sha256': None},
                    {'path': '/source', 'offset': 0, 'bytes': 30, 'sha256': None}],
        'entry_count': 2, 'total_bytes': 40, 'annotations': {'plan_sha256': plan_sha256,
            'layers': [0],
            'phases': [{'name': 'head', 'bytes': 10, 'cumulative_bytes': 10},
                       {'name': 'layer-0', 'bytes': 30, 'cumulative_bytes': 40}]}}


_EXTEND = dict(old_plan={'sha256': 'original'}, new_plan={'path': '/new-plan', 'sha256': 'new'},
               prepared={'formats_by_qname': {'q': ['A8', 'BF16']}}, extension={'path': '/proof'})
_ADDED = [{'path': '/new-plan', 'offset': 0, 'bytes': 20, 'sha256': None}]


def test_a_parent_the_original_run_read_is_extended_though_it_names_an_older_plan():
    """R13's plan was re-derived from its parent's, which names the older plan (#1126)."""
    parent = _parent('older')
    with pytest.raises(ValueError, match='scientific plan'):
        extend_parent(parent, _ADDED, **_EXTEND)
    result = extend_parent(parent, _ADDED, read_by_original_run=True, **_EXTEND)
    assert result['annotations']['plan_sha256'] == 'new'
    assert result['entries'][-1] == parent['entries'][-1]
    assert result['annotations']['phases'][0] == {
        'name': 'head', 'bytes': 30, 'cumulative_bytes': 30}
    assert result['produced_by']['parent_admitted_by'] == {
        'rule': 'original_run_read_manifest', 'parent_plan_sha256': 'older'}


def test_a_parent_naming_the_original_plan_is_extended_as_before():
    """The sealed read changes nothing for a parent that names the original plan."""
    plain = extend_parent(_parent('original'), _ADDED, **_EXTEND)
    sealed = extend_parent(_parent('original'), _ADDED, read_by_original_run=True, **_EXTEND)
    assert plain == sealed
    assert 'parent_admitted_by' not in plain['produced_by']


_READ = 'a' * 64


def _proof(read, plan='original'):
    return {'status': 'band', 'run_identity': {'read_manifest_sha256': read, 'plan_sha256': plan}}


@pytest.mark.parametrize(('proofs', 'expected'), [
    ([_proof(_READ)], True),
    ([_proof(_READ), _proof(_READ)], True),
    ([_proof('b' * 64)], False),                   # the run read another manifest
    ([_proof(_READ), _proof('b' * 64)], False),    # proofs of two runs
    ([_proof(_READ, plan='older')], False),        # read under another plan
    ([{'status': 'band'}], False),                 # no run identity sealed
    ([], False),
])
def test_only_the_sealed_run_identity_admits_a_parent(proofs, expected):
    from tools.prepare_extended_joint_quanta import proofs_name_parent_as_read
    assert proofs_name_parent_as_read(
        proofs, parent_sha256=_READ, plan_sha256='original') is expected


def test_an_unbound_read_manifest_admits_nothing():
    """Stage A seals all zeros when the dispatcher bound no read manifest."""
    from tools.prepare_extended_joint_quanta import proofs_name_parent_as_read
    unbound = '0' * 64
    assert proofs_name_parent_as_read(
        [_proof(unbound)], parent_sha256=unbound, plan_sha256='original') is False

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
