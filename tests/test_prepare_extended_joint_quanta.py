"""Extended metadata preserves historical layer extents and complete control head."""
import copy
from pathlib import Path

import pytest

from tests.test_joint_quanta_join import campaign, probe  # noqa: F401
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


def test_prepare_reaches_the_generator_with_the_scope_the_proofs_seal(tmp_path, campaign, probe, monkeypatch):
    """The real ``prepare()`` flow on a band that seals ``campaign_scope: null``
    and names, as what its run read, a parent that annotates the campaign's
    scope under an older plan (R13, PQ #1126). The parent is admitted by the
    sealed read, the extension derives the scope from the original plan and
    the frozen identity, and the generator it launches produces records under
    the parent's scope (RV-1127 F1 and F2)."""
    import gzip
    import json
    from types import SimpleNamespace
    from prismaquant.joint_adjoint_slices import write_band_receipt
    from prismaquant.joint_catalog_extension import SCHEMA_V3
    from prismaquant.joint_layer_quanta import layer_quanta
    from tests.test_joint_catalog_extension import _campaign_identity, _expected_scope, _pair, _receipt, _write
    from tests.test_stage_b_band_binding import band_from_receipt
    import tools.prepare_extended_joint_quanta as pe

    inputs, _, _ = _pair(tmp_path, campaign, probe, scoped=True)
    identity = _campaign_identity(tmp_path, inputs)
    scope = _expected_scope(inputs, identity)
    parent = copy.deepcopy(campaign['parent_manifest'])
    parent['annotations']['campaign_scope'] = scope
    assert parent['annotations']['plan_sha256'] != inputs['original_plan']['sha256']
    parent_bound = _write(tmp_path, 'parent.json', parent)
    receipt = _receipt(inputs, campaign, probe, scope=None,
                       identity={'read_manifest_sha256': parent_bound['sha256']})
    band = band_from_receipt(receipt, 3)
    band_path = tmp_path / 'band-003.json'
    band_sha = write_band_receipt(band_path, band)
    stub = _write(tmp_path, 'stub.json', {})
    pair = _write(tmp_path, 'pair.json', inputs)
    root = tmp_path / 'meta'
    seen = {}

    def generator(argv):
        def flag(name):
            return argv[argv.index(name) + 1]
        from tools.regenerate_joint_quanta import _load_json
        published = _load_json(Path(flag('--parent-manifest')),
                               digest=flag('--parent-manifest-sha256'), where='published parent')
        extension = {'path': flag('--catalog-extension'), 'sha256': flag('--catalog-extension-sha256')}
        plan = _load_json(Path(flag('--plan')), digest=flag('--plan-sha256'), where='plan')
        prepared = _load_json(Path(flag('--prepared')), digest=flag('--prepared-sha256'), where='prepared')
        seen['parent'] = published
        seen['extension'] = extension
        seen['produced'] = layer_quanta(
            plan, prepared, published, parent_manifest_sha256=flag('--parent-manifest-sha256'),
            plan_path=flag('--plan'), plan_sha256=flag('--plan-sha256'),
            prepared_path=flag('--prepared'), prepared_sha256=flag('--prepared-sha256'),
            adjoint_receipts=[band], catalog_extension=extension, stride=8)
        return 0

    real_load = pe._load_json

    def load(path, *, digest, where):
        document = real_load(path, digest=digest, where=where)
        if where == 'extended plan':
            # The launch recipe embeds the extended plan's resource policy;
            # the fixture plan binds none (its budget check is stubbed below).
            document['stage_b_resource_policy'] = {'fixture': 'policy'}
        return document

    monkeypatch.setattr(pe, '_load_json', load)
    monkeypatch.setattr(pe, 'regenerate', generator)
    monkeypatch.setattr(pe, 'metadata_entries', lambda *a, **k: [])
    monkeypatch.setattr(pe, 'check_stage_b_spec', lambda *a, **k: None)
    monkeypatch.setattr(pe, 'stage_b_replay_mode', lambda spec: 'windowed')
    monkeypatch.setattr(pe, 'require_derived_budget',
                        lambda plan, *, plan_sha256: {'derivation': {'fixture': True}})
    args = SimpleNamespace(
        pair_inputs=Path(pair['path']), pair_inputs_sha256=pair['sha256'],
        parent_manifest=Path(parent_bound['path']), parent_manifest_sha256=parent_bound['sha256'],
        derivation=Path(stub['path']), derivation_sha256=stub['sha256'],
        spec=Path(stub['path']), spec_sha256=stub['sha256'],
        adjoint_receipt=None, adjoint_receipt_sha256=None,
        adjoint_band=[band_path], adjoint_band_sha256=[band_sha],
        metadata_root=root, data_manifest_sha256=None, allowed_tiers=None, produced_output=False,
        campaign_identity=Path(identity['path']), campaign_identity_sha256=identity['sha256'])
    pe.prepare(args)
    published = json.loads(gzip.decompress((root / 'parent.json.gz').read_bytes()))
    assert published['produced_by']['parent_admitted_by'] == {
        'rule': 'original_run_read_manifest', 'parent_plan_sha256': parent['annotations']['plan_sha256']}
    assert published['annotations']['campaign_scope'] == scope
    document = json.loads((root / 'catalog-extension.json').read_bytes())
    assert document['schema'] == SCHEMA_V3
    assert document['original_campaign_scope']['scope'] == scope
    assert document['original_campaign_scope']['derived_from']['campaign_identity'] == identity
    assert seen['extension']['path'] == str(root / 'catalog-extension.json')
    records = seen['produced']['records']
    assert len(records) == 3
    assert all(record['campaign']['campaign_scope'] == scope for record in records)
    assert all(record['catalog_extension'] == seen['extension'] for record in records)
    launch = json.loads((root / 'launch.bands-003.json').read_bytes())
    assert launch['catalog_extension'] == seen['extension']
    # A second prepare on the same root binds the extension it wrote and
    # republishes the same bytes.
    before = {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
    pe.prepare(args)
    assert before == {p.name: p.read_bytes() for p in root.iterdir() if p.is_file()}
