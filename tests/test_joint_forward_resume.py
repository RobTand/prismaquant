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
            campaign_scope={'fixture': True}, forward_recovery=bound)
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
    from prismaquant.joint_adjoint_slices import stage_a_run_header, stage_a_slice
    from prismaquant.joint_layer_quanta import check_adjoint_run_header
    from prismaquant.joint_adjoint_checkpoints import reference_from_record
    # The resumed run's sealed receipt answers for the campaign and gives
    # every layer its slice (PQ #993). The run seals a campaign scope: an
    # unset scope is never compared with an unset scope (PQ #1126).
    sealed = json.loads(json.dumps(resumed))
    assert check_adjoint_run_header(
        stage_a_run_header(sealed), plan_sha256='b'*64, prepared_sha256='c'*64,
        scope={'fixture': True},
        checkpoints=[c['boundary'] for c in resumed['checkpoints']])
    assert all(stage_a_slice(sealed, layer)['boundary_storage']['forward_recovery']
               == resumed['boundary_storage']['forward_recovery']
               for layer in range(max(sealed['stride']['boundaries'])))
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


# --- Chained recovery: a resumed run that is itself contained mid-forward. ---

def _chain_sdk():
    """PB SDK seams the loader reads, answering from files the fixture wrote."""
    from pathlib import Path
    ns = SimpleNamespace
    return {
        'pool': ns(CLAIMED='claimed', _read_json=lambda path: None,
                   PoolQueue=lambda root: ns(root=Path(root),
                                             item_path=lambda state, key: Path(root) / state / key)),
        'reader_lease': ns(containment_certificate_ok=lambda queue, cert: (True, 'fixture')),
        'produced_output': ns(
            validate_instance=lambda value: value, validate_template=lambda value: value,
            instance_dir=lambda root, instance: Path(root) / 'instances' / instance['owner_action_key'],
            _load_batch_record=lambda root, instance, template, entry, batch: (entry, entry['descriptors'])),
        'produced_spool': ns(_check_receipt=lambda receipt, manifest, record: None),
        'core': ns(validate_action=lambda value: value)}


def _owner_spool(root, owner_key, refs):
    """File one contained owner's boundary groups the way PB exports them."""
    import hashlib
    instance = {'owner_action_key': owner_key,
                'owner_attempt': {'nonce': owner_key[:8], 'scope_id': 'fixture.slice'}}
    template = {'template_id': 'fixture-boundary-entries'}
    spool = root / 'spool' / owner_key[:12]
    batches = {}
    for boundary in sorted({b for b, _ in refs}):
        batch_id = f'stagea-boundary-b{boundary}-g0-fixture'
        entries = [{'artifact_class': 'payload', 'destination_path': refs[boundary, i].path,
                    'sha256': refs[boundary, i].sha256, 'bytes': refs[boundary, i].file_bytes}
                   for i in sorted(i for b, i in refs if b == boundary)]
        raw = json.dumps({'batch_id': batch_id, 'owner': owner_key, 'instance': instance,
                          'template': template, 'entries': entries})
        sha = hashlib.sha256(raw.encode()).hexdigest()
        export = hashlib.sha256((owner_key + batch_id).encode()).hexdigest()
        group = spool / batch_id
        group.mkdir(parents=True)
        (group / 'manifest.json').write_text(raw)
        (group / 'receipt.json').write_text('{}')
        (group / 'export.json').write_text(json.dumps({
            'export_key': export, 'manifest_sha256': sha, 'batch_id': batch_id, 'action': {
                'action_key': export, 'inputs': [{'id': 'produced-spool-manifest', 'sha256': sha}],
                'params': {'produced_spool': {'owner': owner_key, 'batch_id': batch_id,
                                              'manifest_sha256': sha}}}}))
        batches[batch_id] = {'descriptors': [
            {'path': e['destination_path'], 'artifact_class': 'payload',
             'sha256': e['sha256'], 'bytes': e['bytes']} for e in entries]}
    directory = root / 'queue' / 'instances' / owner_key
    directory.mkdir(parents=True)
    (directory / 'instance.json').write_text(json.dumps(instance))
    (directory / 'commitments.json').write_text(json.dumps({'batches': batches}))
    return spool, instance, template


def _spool_groups(spool):
    groups = []
    for path in sorted(spool.glob('*/manifest.json')):
        groups.append({'manifest': json.loads(path.read_text()), 'manifest_raw': path.read_text(),
                       'record': json.loads((path.parent / 'export.json').read_text()),
                       'receipt': json.loads((path.parent / 'receipt.json').read_text())})
    return groups


def _bound(path):
    import hashlib
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def test_two_hop_chained_recovery_equals_uninterrupted(tmp_path, monkeypatch):
    """R9 -> R10 -> R11: the second resume imports the first resume's prefix."""
    from prismaquant import joint_cost_stage_a as stage_a
    from prismaquant import joint_forward_resume as recovery_mod
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts
    from prismaquant.joint_adjoint_checkpoints import (
        adjoint_space, load_adjoint_checkpoint, reference_from_record)
    from test_joint_cost_quantum_runtime import _execution, _stage_a
    from test_streamed_cost_checkpoints import _model_identity
    from pathlib import Path
    import torch
    monkeypatch.setattr(recovery_mod, '_sdk', _chain_sdk)
    campaign = {'plan_sha256': 'b' * 64, 'prepared_sha256': 'c' * 64,
                'read_manifest_sha256': 'd' * 64, 'unit_roster_sha256': 'a' * 64,
                'campaign_scope': None}
    identities = []
    bind = StreamedBoundaryArtifacts.bind
    def recorded(self, identity, **kw):
        identities.append(copy.deepcopy(identity))
        return bind(self, identity, **kw)
    monkeypatch.setattr(StreamedBoundaryArtifacts, 'bind', recorded)

    def capture(path, implementation, bound=None, stop_after=None):
        retained = {}
        write = StreamedBoundaryArtifacts.write
        def interrupted(self, tensor, **kw):
            ref = write(self, tensor, **kw)
            if kw.get('probe_index') is None:
                retained[(kw['boundary_index'], kw['batch_index'])] = ref
                if kw['boundary_index'] == stop_after and kw['batch_index'] == len(draw()) - 1:
                    raise InterruptedError('contained after a whole boundary')
            return ref
        runner, _ = _stage_a(path, monkeypatch)
        runner.context.settle_prefetched_layers = lambda *a, **kw: None
        with monkeypatch.context() as patch:
            patch.setattr(StreamedBoundaryArtifacts, 'write', interrupted)
            run = lambda: stage_a.run_adjoint_capture_core(
                runner, draw(), execution=_execution(path), output_root=path, stride=1,
                source_model_identity=_model_identity('joint-source'),
                unit_roster_sha256='a' * 64, plan_sha256='b' * 64, prepared_sha256='c' * 64,
                read_manifest_sha256='d' * 64, implementation_sha256=implementation,
                forward_recovery=bound)
            if stop_after is None:
                return run(), retained
            with pytest.raises(InterruptedError):
                run()
        return None, retained

    # R9: contained after boundary 1; its prefix is recovered by an ordinary capsule.
    _, first = capture(tmp_path / 'r9', '1' * 64, stop_after=1)
    r9 = identities[-1]
    spool, instance, template = _owner_spool(tmp_path, 'e' * 64, first)
    ref = first[0, 0]
    capsule = tmp_path / 'r9-to-r10.json'
    recovery_mod.build_forward_recovery(specification={
        'schema': recovery_mod.SCHEMA, 'queue_root': str(tmp_path / 'queue'),
        'instance': instance, 'template': template,
        'session': json.loads(ref.metadata_json)['identity']['session'],
        'original_bind_identity': r9, 'campaign_identity': campaign,
        'implementation_compatibility': {'original': '1' * 64, 'recovery': '2' * 64,
                                         'scope': 'forward-identical-memory-only'},
        'n_batches': len(draw()), 'entry_shape': list(ref.shape), 'entry_dtype': ref.dtype},
        spool_directory=spool, output=capsule, frontier=1)
    r9_to_r10 = _bound(capsule)

    # R10: resumes from that capsule and is contained after boundary 2.
    _, second = capture(tmp_path / 'r10', '2' * 64, bound=r9_to_r10, stop_after=2)
    assert {b for b, _ in second} == {2}, 'R10 wrote only the boundary it replayed'
    r10 = identities[-1]
    spool, instance, template = _owner_spool(tmp_path, 'f' * 64, second)
    owner_action = {'action_key': 'f' * 64, 'params': {'command': [
        'python3', '-m', 'prismaquant.joint_adjoint_capture', '--forward-recovery',
        r9_to_r10['path'], '--forward-recovery-sha256', r9_to_r10['sha256']]}}
    chained = {
        'schema': recovery_mod.SCHEMA, 'queue_root': str(tmp_path / 'queue'),
        'instance': instance, 'template': template,
        'session': json.loads(second[2, 0].metadata_json)['identity']['session'],
        'original_bind_identity': r10, 'campaign_identity': campaign,
        'implementation_compatibility': {'original': '2' * 64, 'recovery': '3' * 64,
                                         'scope': 'forward-identical-memory-only'},
        'n_batches': len(draw()), 'entry_shape': list(ref.shape), 'entry_dtype': ref.dtype,
        'frontier': 2, 'first_boundary': 2, 'imported': r9_to_r10,
        'owner_action': owner_action, 'groups': _spool_groups(spool)}
    capsule = tmp_path / 'r10-to-r11.json'
    capsule.write_text(json.dumps(chained))

    # R11: resumes from the chained capsule, reading two contained generations.
    resumed, _ = capture(tmp_path / 'r11', '3' * 64, bound=_bound(capsule))
    baseline, _ = capture(tmp_path / 'baseline', '3' * 64)
    assert resumed['boundary_storage']['forward_recovery']['original_owner'] == 'f' * 64
    for new, old in zip(resumed['checkpoints'], baseline['checkpoints'], strict=True):
        actual = load_adjoint_checkpoint(adjoint_space(tmp_path / 'r11'), new)[0]
        expected = load_adjoint_checkpoint(adjoint_space(tmp_path / 'baseline'), old)[0]
        assert actual.keys() == expected.keys()
        assert all(torch.equal(actual[key], expected[key]) for key in actual)
    assert all(Path(r.path).exists() for r in [*first.values(), *second.values()])

    # Stage B attaches through the same chain: one reference from each owner.
    # In this two-layer fixture R10's boundary 2 is the final forward boundary,
    # which only the tail reads, so the receipt names R9 entries only. The
    # attached owner still authorizes every chain reference; at GLM scale
    # Stage B reads R10's boundary 44 this way.
    settings = dict(resumed['boundary_storage']['policy'])
    settings['directory'] = resumed['boundary_storage']['directory']
    attached = StreamedBoundaryArtifacts(settings)
    attached.attach(resumed['boundary_storage']['session'], n_probes=4,
                    forward_recovery=resumed['boundary_storage']['forward_recovery'])
    older = reference_from_record(resumed['boundary_entries']['1'][0])
    newer = reference_from_record(recovery_mod.chain_records(chained)['2'][0])
    assert older.path == first[1, 0].path and newer.path == second[2, 0].path
    with attached, attached.prefetch([older, newer]) as window:
        assert attached.get(window, older).shape == older.shape
        assert attached.get(window, newer).shape == newer.shape

    # A second owner attaching the same chain reads no capsule again: the
    # verified chain is cached by each capsule's (path, sha256).
    reads = []
    real_read = recovery_mod._read
    def counted(path, *args, **kwargs):
        reads.append(str(path))
        return real_read(path, *args, **kwargs)
    monkeypatch.setattr(recovery_mod, '_read', counted)
    again = StreamedBoundaryArtifacts(settings)
    again.attach(resumed['boundary_storage']['session'], n_probes=4,
                 forward_recovery=resumed['boundary_storage']['forward_recovery'])
    assert reads == []
    assert again._attached_forward_inputs == attached._attached_forward_inputs
    # The cache never skips the session check against the receipt's binding.
    changed = copy.deepcopy(resumed['boundary_storage']['forward_recovery'])
    changed['frontier'] += 1
    with pytest.raises(RuntimeError, match='session changed'):
        StreamedBoundaryArtifacts(settings).attach(
            resumed['boundary_storage']['session'], n_probes=4, forward_recovery=changed)
    monkeypatch.setattr(recovery_mod, '_read', real_read)

    # The freezer derives the same chained capsule from the spool and PB records.
    import prismaquant.aura_cost as aura_cost
    monkeypatch.setattr(aura_cost, '_aura_source_sha256', lambda: '3' * 64)
    request = tmp_path / 'r10-request.json'
    request.write_text(json.dumps(owner_action))
    derived = tmp_path / 'r10-to-r11-derived.json'
    summary = recovery_mod.build_forward_recovery(
        imported=r9_to_r10, owner_request=request, spool_directory=spool,
        output=derived, frontier=2, bind_current_implementation=True)
    assert summary['entries'] == 3 * len(draw()) and summary['segments'] == 2
    assert json.loads(derived.read_text()) == json.loads(capsule.read_text())

    # --inspect-live accepts the contained chained owner before any freeze.
    from tools.stagea_forward_recovery import inspect_owner
    report = inspect_owner(recovery_mod.chained_specification(
        r9_to_r10, spool_directory=spool, owner_request=request), spool, sdk=_chain_sdk())
    assert report == {'schema': recovery_mod.SCHEMA, 'inspect_only': True,
                      'authority_to_resume': False, 'contained': True,
                      'first_boundary': 2, 'frontier': 2, 'groups': 1,
                      'entries': len(draw()), 'segments': 2,
                      'payload_bytes': sum(r.file_bytes for r in second.values())}


def _chained_pair(tmp_path):
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256
    prior, _ = identity_document()
    prior.update(entry_shape=[1, 4, 8], entry_dtype='torch.float32', queue_root='/q', groups=[],
                 instance={'owner_action_key': 'a' * 64})
    path = tmp_path / 'prior.json'
    path.write_text(json.dumps(prior))
    bound = _bound(path)
    old = {**prior['original_bind_identity'], 'producer_source_sha256': 'd' * 64}
    top = {**copy.deepcopy(prior), 'original_bind_identity': old, 'frontier': 2,
           'first_boundary': 2, 'imported': bound,
           'implementation_compatibility': {'original': 'd' * 64, 'recovery': '9' * 64,
                                            'scope': 'forward-identical-memory-only'},
           'session': {'generation': 'new', 'run_identity_sha256':
                       canonical_json_sha256(old, where='fixture')},
           'instance': {'owner_action_key': 'b' * 64},
           'owner_action': {'action_key': 'b' * 64, 'params': {'command': [
               '--forward-recovery', bound['path'], '--forward-recovery-sha256', bound['sha256']]}}}
    return prior, top


def test_chain_links_each_segment_to_the_owner_that_imported_it(tmp_path):
    from prismaquant.joint_forward_resume import (
        chain_documents, _require_imported_by_owner)
    prior, top = _chained_pair(tmp_path)
    assert [d['frontier'] for d in chain_documents(top)] == [2, 1]
    _require_imported_by_owner(top, _chain_sdk())


@pytest.mark.parametrize('mutate', [
    lambda top: top.update(first_boundary=3),
    lambda top: top.update(frontier=1, first_boundary=2),
    lambda top: top['imported'].update(sha256='0' * 64),
    lambda top: top.update(n_batches=4),
    lambda top: top['original_bind_identity'].update(seed_base=7001),
    lambda top: top['original_bind_identity'].update(producer_source_sha256='7' * 64),
    lambda top: top.update(campaign_identity={'plan': '0' * 64}),
    lambda top: top.pop('imported'),
])
def test_broken_chain_links_refuse(tmp_path, mutate):
    from prismaquant.joint_forward_resume import chain_documents, ForwardRecoveryRefused
    _, top = _chained_pair(tmp_path)
    mutate(top)
    with pytest.raises(ForwardRecoveryRefused):
        chain_documents(top)


@pytest.mark.parametrize('mutate', [
    lambda action: action.update(action_key='c' * 64),
    lambda action: action['params']['command'].__setitem__(1, '/other/capsule.json'),
    lambda action: action['params']['command'].__setitem__(3, '0' * 64),
    lambda action: action['params']['command'].extend(action['params']['command'][:4]),
])
def test_owner_that_did_not_import_the_prior_refuses(tmp_path, mutate):
    from prismaquant.joint_forward_resume import (
        _require_imported_by_owner, ForwardRecoveryRefused)
    _, top = _chained_pair(tmp_path)
    mutate(top['owner_action'])
    with pytest.raises(ForwardRecoveryRefused):
        _require_imported_by_owner(top, _chain_sdk())


def test_recovery_manifest_stages_every_segment_and_imported_capsule(tmp_path):
    from tools.build_stagea_forward_recovery_package import recovery_manifest
    prior, top = _chained_pair(tmp_path)
    prior['groups'] = [{'manifest': {'entries': [
        {'destination_path': f'/old/entries/boundary-{i}-{b}-at-{b}.pt', 'bytes': 7, 'sha256': 'a' * 64}
        for b in range(2) for i in range(5)]}}]
    (tmp_path / 'prior.json').write_text(json.dumps(prior))
    top['imported'] = _bound(tmp_path / 'prior.json')
    top['groups'] = [{'manifest': {'entries': [
        {'destination_path': f'/new/entries/boundary-{i}-2-at-2.pt', 'bytes': 7, 'sha256': 'b' * 64}
        for i in range(5)]}}]
    path = tmp_path / 'top.json'
    path.write_text(json.dumps(top))
    names = ['head', 'forward-000', 'forward-001', 'forward-002',
             'chain-002', 'chain-001', 'chain-000']
    original = {'entries': [{'path': f'/input/{n}', 'offset': 0, 'bytes': 1, 'sha256': None}
                            for n in names], 'annotations': {}, 'read_plan': {'phases': [
        {'name': n, 'entry_indices': [i], 'bytes': 0, 'cumulative_bytes': 0}
        for i, n in enumerate(names)]}}
    result = recovery_manifest(original, top, _bound(path))
    phases = {p['name']: [result['entries'][i]['path'] for i in p['entry_indices']]
              for p in result['read_plan']['phases']}
    assert list(phases) == ['head', 'forward-002', 'chain-002', 'chain-001', 'chain-000']
    assert str(path) in phases['head'] and top['imported']['path'] in phases['head']
    assert sum('/new/' in p for p in phases['forward-002']) == 5
    assert sum('/old/entries/boundary-' in p and '-1-at-1' in p for p in phases['chain-001']) == 5
