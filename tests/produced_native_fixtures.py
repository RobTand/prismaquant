"""Application lifecycle fixtures; selected native execution is proved separately."""
from pathlib import Path

from prismaquant.stage_a_produced_output import BoundaryProducedPublication, build_boundary_template, publish_document


def publication(tmp_path, *, payload_max=1 << 24, group_size=2, tensor_bytes=1024,
                spool_bytes=1 << 24, write_only=False):
    root = tmp_path / 'output'
    root.mkdir(parents=True)
    local = tmp_path / 'local'
    local.mkdir()
    declaration = build_boundary_template(output_prefix=root,
        artifact_max_bytes=payload_max, group_size=group_size,
        max_entry_tensor_bytes=tensor_bytes, concurrent_groups=8,
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
    pub = publication(tmp_path, **kwargs)
    storage = StreamedBoundaryArtifacts({'schema': BOUNDARY_STORAGE_SCHEMA,
        'directory': str(Path(pub.output_prefix) / 'boundaries'),
        'max_resident_bytes': 1 << 20, 'max_auxiliary_bytes': 1 << 20,
        'max_artifact_bytes': 1 << 24, 'prefetch_batches': pub.declaration['group_size']})
    storage.bind({'fixture': 'native-produced'}, n_probes=probes, published=True)
    storage.bind_produced_output(pub, group_size=pub.declaration['group_size'],
        n_batches=n_batches, max_entry_tensor_bytes=pub.declaration['max_entry_tensor_bytes'],
        origin_lifetime='retain' if pub.write_only else None)
    return storage, pub
