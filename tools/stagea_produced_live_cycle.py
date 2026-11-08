"""Run an exact CPU tensor publication/readback cycle in an admitted native job.

This worker exercises the existing tensor writer, bounded spool, resident reader,
retirement and rematerialization owners. It does not qualify a numerical runtime.
The immutable declaration must be bound by the actual native launch arguments.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from prismaquant.stage_a_produced_output import add_declaration_arguments, BoundaryProducedPublication


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_declaration_arguments(parser)
    args = parser.parse_args(argv)
    if args.produced_declaration is None:
        parser.error('an immutable produced declaration is required')
    import torch
    from prismaquant.streamed_boundary_artifacts import StreamedBoundaryArtifacts, BOUNDARY_STORAGE_SCHEMA
    publication = BoundaryProducedPublication.bind_from_admitted_owner()
    if publication is None or publication.write_only:
        raise RuntimeError('this cycle requires a launch-bound readback declaration')
    count = publication.declaration['group_size']
    if count > 16:
        raise ValueError('CPU cycle is bounded to sixteen tensors per group')
    storage = StreamedBoundaryArtifacts({'schema': BOUNDARY_STORAGE_SCHEMA,
        'directory': str(Path(publication.output_prefix) / 'boundaries'),
        'max_resident_bytes': 1 << 20, 'max_auxiliary_bytes': 1 << 20,
        'max_artifact_bytes': publication.durable_maxima()['payload_max_bytes'],
        'prefetch_batches': count})
    storage.bind({'fixture': 'exact-native-cpu-cycle'}, n_probes=1, published=True)
    storage.bind_produced_output(publication, group_size=count, n_batches=count,
        max_entry_tensor_bytes=publication.declaration['max_entry_tensor_bytes'])
    expected = [torch.tensor([[1., 2., 3., 4.]]) + batch for batch in range(count)]
    with storage:
        references = [storage.write(value, batch_index=batch, boundary_index=0)
                      for batch, value in enumerate(expected)]
        storage.settle_local_output()
        # settle releases local copies, so this read also proves immutable
        # group rematerialization through the existing IO owner.
        with storage.prefetch(references) as window:
            for reference, value in zip(references, expected):
                if not torch.equal(storage.get(window, reference), value):
                    raise AssertionError('exact tensor readback differs')
        report = storage.produced_output_report()
    from prismaquant.digests import DIRECT_ASCII_STRICT
    print(DIRECT_ASCII_STRICT.encoded({'schema': 'prismaquant.produced_tensor_cycle.v1',
        'producer': publication.producer, 'tensors': count, 'report': report}).decode())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
