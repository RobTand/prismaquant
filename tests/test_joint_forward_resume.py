"""An explicit recovery authority, never reuse of an unfinished generation."""
import pytest
import copy
import json
from types import SimpleNamespace
from test_layer_major_boundary_capture import fixture, draw, owner, digest


def test_interrupted_forward_resumes_without_replaying_completed_layers(tmp_path):
    from prismaquant.joint_forward_resume import ForwardRecovery
    from prismaquant.joint_adjoint_checkpoints import exact_entry_record
    model, context, runner, _ = fixture()
    settings = owner(tmp_path / 'original')
    settings._published = True
    retained = {}
    original_write = settings.write
    def write(tensor, **kw):
        ref = original_write(tensor, **kw)
        retained[(kw['boundary_index'], kw['batch_index'])] = ref
        if kw['boundary_index'] == 1 and kw['batch_index'] == len(draw()) - 1:
            raise InterruptedError('simulated producer stop after full boundary')
        return ref
    settings.write = write
    with pytest.raises(InterruptedError), settings:
        runner.capture_layer_major_boundaries([r[None] for r in draw()], storage=settings)
    state = ForwardRecovery(1, len(draw()),
        {str(b): [exact_entry_record(retained[b, i]) for i in range(len(draw()))]
         for b in range(2)}, {'schema': 'fixture'})
    resumed = owner(tmp_path / 'resumed')
    with resumed:
        state.install(resumed)
        before = context.install_calls
        batches = runner.capture_layer_major_boundaries([r[None] for r in draw()],
            storage=resumed, forward_recovery=state)
        assert context.install_calls - before == runner.num_layers - 1
        with resumed.prefetch([b.activations_cpu[-1] for b in batches]) as window:
            actual = [digest(resumed.get(window, b.activations_cpu[-1])) for b in batches]
    with owner(tmp_path / 'uninterrupted', window=len(draw()), cap=10000) as baseline:
        batches = runner.capture_layer_major_boundaries([r[None] for r in draw()], storage=baseline)
        with baseline.prefetch([b.activations_cpu[-1] for b in batches]) as window:
            expected = [digest(baseline.get(window, b.activations_cpu[-1])) for b in batches]
    assert actual == expected
    assert all(__import__('pathlib').Path(r.path).exists() for r in retained.values())


def identity_document():
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256
    old = {'producer_source_sha256': 'a' * 64, 'source_model': {'content': 'b' * 64},
           'calibration_sha256': 'c' * 64, 'calibration_shape': [5, 4],
           'calibration_dtype': 'torch.int64', 'n_probes': 4, 'seed_base': 7000,
           'temperature': 1.0, 'token_scope': 'all',
           'execution_partition': {'partition_count': 5}}
    current = {**old, 'producer_source_sha256': 'd' * 64}
    doc = {'schema': 'prismaquant.joint_forward_recovery.v1',
        'original_bind_identity': old, 'campaign_identity': {'plan': 'e' * 64},
        'implementation_compatibility': {'original': 'a' * 64, 'recovery': 'd' * 64,
                                        'scope': 'forward-identical-memory-only'},
        'session': {'generation': 'old', 'run_identity_sha256':
                    canonical_json_sha256(old, where='fixture')}, 'frontier': 1, 'n_batches': 5}
    return doc, current


@pytest.mark.parametrize('field,value', [('source_model', {'content': 'x'}),
    ('calibration_sha256', 'f' * 64), ('seed_base', 7001), ('n_probes', 3),
    ('temperature', 0.5), ('producer_source_sha256', 'f' * 64)])
def test_changed_scientific_or_implementation_identity_refuses(field, value):
    from prismaquant.joint_forward_resume import validate_forward_state, ForwardRecoveryRefused
    doc, current = identity_document()
    assert validate_forward_state(doc, bind_identity=current,
                                  campaign_identity=doc['campaign_identity']) == (1, 5)
    current[field] = value
    with pytest.raises(ForwardRecoveryRefused):
        validate_forward_state(doc, bind_identity=current, campaign_identity=doc['campaign_identity'])


def test_live_or_uncontained_owner_refuses():
    from prismaquant.joint_forward_resume import require_contained, ForwardRecoveryRefused
    queue = SimpleNamespace(item_path=lambda *args: 'claim')
    instance = {'owner_action_key': 'a' * 64, 'owner_attempt': {'nonce': 'n', 'scope_id': 's'}}
    pool = SimpleNamespace(CLAIMED='claimed', _read_json=lambda path: {'live': True})
    sdk = {'pool': pool, 'reader_lease': SimpleNamespace(
        containment_certificate_ok=lambda *a: (False, 'scope-not-empty-retain'))}
    with pytest.raises(ForwardRecoveryRefused, match='still claimed'):
        require_contained(queue, instance, sdk)
    pool._read_json = lambda path: None
    with pytest.raises(ForwardRecoveryRefused, match='not contained'):
        require_contained(queue, instance, sdk)


def test_incomplete_boundary_and_duplicate_coordinates_refuse():
    from prismaquant.joint_forward_resume import _records, ForwardRecoveryRefused
    doc, _ = identity_document()
    doc.update(entry_shape=[1, 4, 8], entry_dtype='torch.float32')
    entries = [{'destination_path': f'/old/entries/boundary-{i}-{b}-at-{b}.pt',
                'sha256': 'f' * 64, 'bytes': 1024}
               for b in range(2) for i in range(5)]
    assert len(_records(doc, entries)['1']) == 5
    for bad in (entries[:-1], entries + entries[:1]):
        with pytest.raises(ForwardRecoveryRefused):
            _records(doc, bad)


def test_hyperconnection_four_dimensional_boundary_geometry():
    from prismaquant.joint_forward_resume import _records
    doc, _ = identity_document()
    doc.update(entry_shape=[1, 512, 4, 4096], entry_dtype='torch.bfloat16')
    entries = [{'destination_path': f'/old/entries/boundary-{i}-{b}-at-{b}.pt',
                'sha256': 'f' * 64, 'bytes': 16779369}
               for b in range(2) for i in range(5)]
    rows = _records(doc, entries)
    assert rows['1'][0]['shape'] == [1, 512, 4, 4096]
    assert rows['1'][0]['tensor_bytes'] == 16777216


def test_export_action_must_seal_the_exact_manifest_input():
    from prismaquant.joint_forward_resume import _checked_group, ForwardRecoveryRefused
    import hashlib
    instance = {'owner_action_key': 'a' * 64}
    manifest = {'batch_id': 'b', 'owner': instance['owner_action_key'],
                'instance': instance, 'template': {}}
    raw = json.dumps(manifest)
    sha = hashlib.sha256(raw.encode()).hexdigest()
    action = {'action_key': 'c'*64, 'params': {'produced_spool': {
        'owner': instance['owner_action_key'], 'batch_id': 'b', 'manifest_sha256': sha}},
        'inputs': [{'id': 'produced-spool-manifest', 'sha256': 'WRONG'}]}
    group = {'manifest': manifest, 'manifest_raw': raw, 'receipt': {}, 'record': {
        'export_key': 'c'*64, 'manifest_sha256': sha, 'batch_id': 'b', 'action': action}}
    sdk = {'core': SimpleNamespace(validate_action=lambda value: value)}
    with pytest.raises(ForwardRecoveryRefused, match='does not seal'):
        _checked_group(group, queue=None, instance=instance, template={}, commitments={}, sdk=sdk)


def test_recovered_tail_and_reverse_checkpoints_equal_uninterrupted(tmp_path, monkeypatch):
    from prismaquant import joint_cost_stage_a as stage_a
    from prismaquant import joint_forward_resume as recovery_mod
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts
    from prismaquant.joint_adjoint_checkpoints import exact_entry_record, load_adjoint_checkpoint, adjoint_space
    from test_joint_cost_quantum_runtime import _execution, _stage_a
    from test_streamed_cost_checkpoints import _model_identity
    import torch
    def capture(path, bound=None):
        runner, _ = _stage_a(path, monkeypatch)
        runner.context.settle_prefetched_layers = lambda *a, **kw: None
        return stage_a.run_adjoint_capture_core(runner, draw(), execution=_execution(path),
            output_root=path, stride=1, source_model_identity=_model_identity('joint-source'),
            unit_roster_sha256='a'*64, plan_sha256='b'*64, prepared_sha256='c'*64,
            read_manifest_sha256='d'*64, implementation_sha256='e'*64,
            forward_recovery=bound)
    retained = {}
    write = StreamedBoundaryArtifacts.write
    def interrupted(self, tensor, **kw):
        ref = write(self, tensor, **kw)
        if kw.get('probe_index') is None:
            retained[(kw['boundary_index'], kw['batch_index'])] = ref
            if kw['boundary_index'] == 1 and kw['batch_index'] == len(draw()) - 1:
                raise InterruptedError('complete forward prefix')
        return ref
    with monkeypatch.context() as patch:
        patch.setattr(StreamedBoundaryArtifacts, 'write', interrupted)
        with pytest.raises(InterruptedError):
            capture(tmp_path / 'interrupted')
    entries = [{'destination_path': r.path, 'sha256': r.sha256, 'bytes': r.file_bytes}
               for r in retained.values()]
    first = next(iter(retained.values()))
    capsule = {'schema': recovery_mod.SCHEMA, 'frontier': 1, 'n_batches': len(draw()),
        'session': json.loads(first.metadata_json)['identity']['session'],
        'entry_shape': list(first.shape), 'entry_dtype': first.dtype,
        'instance': {'owner_action_key': 'a'*64, 'owner_attempt': {'nonce': 'n', 'scope_id': 's'}},
        'groups': [{'manifest': {'entries': entries}}]}
    capsule_path = tmp_path / 'fixture-capsule.json'
    capsule_path.write_text(json.dumps(capsule))
    import hashlib
    binding = {'schema': recovery_mod.SCHEMA, 'frontier': 1,
        'original_session': capsule['session'], 'original_owner': 'a'*64,
        'original_attempt': capsule['instance']['owner_attempt'], 'capsule': {
            'path': str(capsule_path), 'sha256': hashlib.sha256(capsule_path.read_bytes()).hexdigest()}}
    recovery = recovery_mod.ForwardRecovery(1, len(draw()),
        {str(b): [exact_entry_record(retained[b, i]) for i in range(len(draw()))]
         for b in range(2)}, binding)
    real_load = recovery_mod.load_forward_recovery
    def fixture_authority(bound, **kwargs):
        if bound is None:
            return real_load(bound, **kwargs)
        recovery.install(kwargs['storage'])
        return recovery
    monkeypatch.setattr(recovery_mod, 'load_forward_recovery', fixture_authority)
    resumed = capture(tmp_path / 'resumed', {'fixture': 'authority-tested-separately'})
    baseline = capture(tmp_path / 'baseline')
    assert resumed['telemetry']['chain_backwards'] == baseline['telemetry']['chain_backwards']
    for new, old in zip(resumed['checkpoints'], baseline['checkpoints']):
        actual = load_adjoint_checkpoint(adjoint_space(tmp_path / 'resumed'), new)[0]
        expected = load_adjoint_checkpoint(adjoint_space(tmp_path / 'baseline'), old)[0]
        assert actual.keys() == expected.keys()
        assert all(torch.equal(actual[key], expected[key]) for key in actual)
    from prismaquant.joint_layer_quanta import bind_adjoint_receipt
    from prismaquant.joint_adjoint_checkpoints import reference_from_record
    assert bind_adjoint_receipt(resumed, plan_sha256='b'*64, prepared_sha256='c'*64,
        scope=None, checkpoints=[c['boundary'] for c in resumed['checkpoints']])
    settings = dict(resumed['boundary_storage']['policy'])
    settings['directory'] = resumed['boundary_storage']['directory']
    attached = StreamedBoundaryArtifacts(settings)
    attached.attach(resumed['boundary_storage']['session'], n_probes=4,
                    forward_recovery=resumed['boundary_storage']['forward_recovery'])
    imported = reference_from_record(resumed['boundary_entries']['1'][0])
    with attached, attached.prefetch([imported]) as window:
        assert attached.get(window, imported).shape == imported.shape
        from dataclasses import replace
        with pytest.raises(RuntimeError, match='stale generation'):
            attached._entry_identity(replace(imported, sha256='f'*64))


@pytest.mark.parametrize('frontier', [1, 2])
def test_recovery_manifest_only_stages_boundaries_in_consuming_phases(tmp_path, frontier):
    from tools.build_stagea_forward_recovery_package import recovery_manifest
    path = tmp_path / 'capsule.json'
    path.write_text('{}')
    names = ['head', 'forward-000', 'forward-001', 'chain-001', 'chain-000']
    original = {'entries': [{'path': '/input/meta', 'offset': 0, 'bytes': 1, 'sha256': None},
                           {'path': '/input/layer0', 'offset': 0, 'bytes': 3, 'sha256': None},
                           {'path': '/input/layer1', 'offset': 0, 'bytes': 5, 'sha256': None}],
                'annotations': {}, 'read_plan': {'phases': [
                    {'name': n, 'entry_indices': [i], 'bytes': 0, 'cumulative_bytes': 0}
                    for n, i in zip(names, [0, 1, 2, 2, 1])]}}
    capsule = {'frontier': frontier, 'n_batches': 2, 'groups': [{'manifest': {'entries': [
        {'destination_path': f'/input/old/entries/boundary-{i}-{b}-at-{b}.pt',
         'bytes': 7, 'sha256': 'a'*64} for b in range(frontier+1) for i in range(2)]}}]}
    result = recovery_manifest(original, capsule, {'path': str(path), 'sha256': 'b'*64})
    assert [p['name'] for p in result['read_plan']['phases']] == [
        'head', 'forward-001', 'chain-001', 'chain-000']
    for phase in result['read_plan']['phases']:
        paths = [result['entries'][i]['path'] for i in phase['entry_indices']]
        expected = (frontier if phase['name'].startswith('forward') else
                    int(phase['name'].split('-')[1]) if phase['name'].startswith('chain') else None)
        boundary_paths = [p for p in paths if 'boundary-' in p]
        assert len(boundary_paths) == (0 if expected is None else 2)
        assert all(f'-{expected}-at-{expected}.pt' in p for p in boundary_paths)
