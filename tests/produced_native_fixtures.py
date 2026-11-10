"""Application lifecycle fixtures; selected native execution is proved separately."""
from pathlib import Path

from prismaquant.stage_a_produced_output import BoundaryProducedPublication, build_boundary_template, publish_document


def publication(tmp_path, *, payload_max=1 << 24, group_size=2, tensor_bytes=1024,
                spool_bytes=1 << 24, write_only=False, window_groups=8,
                output_prefix=None):
    root = tmp_path / 'output' if output_prefix is None else Path(output_prefix)
    root.mkdir(parents=True, exist_ok=True)
    local = tmp_path / 'local'
    local.mkdir(parents=True)
    declaration = build_boundary_template(output_prefix=root,
        artifact_max_bytes=payload_max, group_size=group_size,
        max_entry_tensor_bytes=tensor_bytes, concurrent_groups=window_groups,
        write_only=write_only, spool=None if spool_bytes is None else
        {'root': str(local), 'max_bytes': spool_bytes})
    binding = publish_document(tmp_path / 'declaration.json', declaration)
    record = tmp_path / 'record'
    record.mkdir()
    # This constructor is a unit seam. No selected-result reader is replaced,
    # and none of these fixtures claims independently selected native authority.
    producer = {'schema': 'prismaquant.flux_result_producer.v1', 'job_id': '123',
        'record_dir': str(record), 'launch_sha256': 'a' * 64,
        'argv': ['fixture', '--produced-declaration', binding['path'],
                 '--produced-declaration-sha256', binding['sha256']]}
    return BoundaryProducedPublication(declaration=declaration, binding=binding, producer=producer)


def owner(tmp_path, **kwargs):
    from prismaquant.streamed_boundary_artifacts import StreamedBoundaryArtifacts, BOUNDARY_STORAGE_SCHEMA
    n_batches = kwargs.pop('n_batches', 2)
    probes = kwargs.pop('n_probes', 1)
    read_order = kwargs.pop('read_order', 'probe_major')
    batch_range = kwargs.pop('batch_range', None)
    origin_lifetime = kwargs.pop('origin_lifetime', 'retain')
    pub = publication(tmp_path, **kwargs)
    storage = StreamedBoundaryArtifacts({'schema': BOUNDARY_STORAGE_SCHEMA,
        'directory': str(Path(pub.output_prefix) / 'boundaries'),
        'max_resident_bytes': 1 << 20, 'max_auxiliary_bytes': 1 << 20,
        'max_artifact_bytes': 1 << 24, 'prefetch_batches': pub.declaration['group_size']})
    storage.bind({'fixture': 'native-produced'}, n_probes=probes, published=True)
    storage.bind_produced_output(pub, group_size=pub.declaration['group_size'],
        n_batches=n_batches, max_entry_tensor_bytes=pub.declaration['max_entry_tensor_bytes'],
        read_order=read_order, batch_range=batch_range,
        origin_lifetime=origin_lifetime if pub.write_only else None)
    return storage, pub


def selected_publications(monkeypatch):
    """Replace only terminal authority for offline scientific populations.

    Every declaration, descriptor, index, digest and tensor is real. Native
    terminal/launch/runtime authority is exercised by the admitted native proof,
    never granted by this fixture outside its private selection registry.
    """
    from types import SimpleNamespace
    from prismaquant import native_results
    from prismaquant.stage_a_produced_output import BoundaryProducedPublication, decode_produced_document
    registry = {}
    real_reader = native_results.read_selected_result
    finalize = BoundaryProducedPublication.finalize

    def read(selection, **kwargs):
        key = selection['record_dir']
        if key not in registry:
            return real_reader(selection, **kwargs)
        return registry[key]

    monkeypatch.setattr(native_results, 'read_selected_result', read)

    def publish(path, *, producer_id, inputs, artifacts, **kwargs):
        raw = Path(artifacts['produced_index']).read_bytes()
        producer = decode_produced_document(raw)['producer']
        selection = {'job_id': producer['job_id'], 'record_dir': producer['record_dir']}
        registry[selection['record_dir']] = SimpleNamespace(
            publication={'producer': producer, 'producer_id': producer_id, 'inputs': inputs},
            artifact_payloads={'produced_index': raw},
            launch={'argv': list(producer['argv'])})
        return selection, registry[selection['record_dir']]
    monkeypatch.setattr(native_results, 'publish_result', publish)

    def register(pub):
        return finalize(pub, Path(pub.producer['record_dir']) / 'result.json')
    return register


def recovery_publication(root, refs, specification, register, *, job_id='123'):
    """Publish exact existing tensor files using the application owner."""
    import json
    from prismaquant.stage_a_produced_output import BoundaryProducedPublication
    root.mkdir(parents=True, exist_ok=True)
    session = json.loads(next(iter(refs.values())).metadata_json)['identity']['session']
    prefix = Path(next(iter(refs.values())).path).parent.parent
    state = publish_document(root / 'forward-state.json', specification)
    declaration = build_boundary_template(output_prefix=prefix,
        artifact_max_bytes=1 << 26, group_size=32, max_entry_tensor_bytes=1 << 20,
        concurrent_groups=8, inputs={'forward_state': state})
    binding = publish_document(root / 'declaration.json', declaration)
    record = root / 'record'; record.mkdir()
    producer = {'schema': 'prismaquant.flux_result_producer.v1', 'job_id': job_id,
        'record_dir': str(record), 'launch_sha256': job_id.zfill(64),
        'argv': ['fixture', '--produced-declaration', binding['path'],
            '--produced-declaration-sha256', binding['sha256']]}
    if 'imported' in specification:
        producer['argv'] += ['--forward-recovery', specification['imported']['path'],
            '--forward-recovery-sha256', specification['imported']['sha256']]
    pub = BoundaryProducedPublication(declaration=declaration, binding=binding, producer=producer)
    pub.bind_session(session)
    for boundary in sorted({b for b, _ in refs}):
        selected = [refs[b, i] for b, i in sorted(refs) if b == boundary]
        batch = 'boundary-' + str(boundary)
        # The test's exact writer ran before transport admission. Seal the
        # actual prewrite request before attaching its independently written
        # files, rather than allowing an occupied path through require_prewrite.
        pub._prewrites[batch] = {'ceilings': {'payload_max_bytes': 1 << 24,
            'checkpoint_max_bytes': 0, 'temp_max_bytes': 0},
            'paths': [ref.path for ref in selected]}
        pub.publish(batch_id=batch, descriptors=[pub.descriptor_for(ref,
            producer_generation=batch) for ref in selected])
    selection, result = register(pub)
    return selection, pub, result


def bound_file(path):
    import hashlib
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
