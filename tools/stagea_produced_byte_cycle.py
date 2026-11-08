"""Native CPU byte-fixture proof through publication, spool and recovery owners.

Files deliberately are simple bytes, not tensor archives or scientific output.
Use actual tensor readers separately for numerical/runtime qualification.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

from prismaquant.digests import bytes_sha256hex, canonical_json_sha256
from prismaquant.stage_a_produced_output import (BoundaryProducedPublication,
    add_declaration_arguments, read_bound_document)
from prismaquant.produced_output_spool import ProducedOutputSpool


@dataclass(frozen=True)
class FixtureReference:
    path: str
    name: str
    file_bytes: int
    sha256: str


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_declaration_arguments(parser)
    parser.add_argument('--payload-exit', type=int, default=0)
    parser.add_argument('--completion-ready', type=Path)
    parser.add_argument('--completion-release', type=Path)
    args = parser.parse_args(argv)
    publication = BoundaryProducedPublication.bind_from_admitted_owner()
    state = read_bound_document(publication.declaration['inputs']['forward_state'])
    session = {'generation': 'known-byte-fixture', 'run_identity_sha256':
        canonical_json_sha256(state['original_bind_identity'], where='byte fixture')}
    publication.bind_session(session)
    spool = ProducedOutputSpool.from_publication(publication)
    if spool is None:
        raise RuntimeError('byte fixture requires explicit same-box scratch spool')
    directory = Path(publication.output_prefix) / session['generation'] / 'entries'
    for boundary in range(2):
        batch = publication.batch_id_for(boundary_index=boundary, group_index=0)
        paths = [directory / f'boundary-{i}-{boundary}-at-{boundary}.pt' for i in range(2)]
        payloads = [bytes([0, boundary, i, 255]) + b'known-native-fixture' for i in range(2)]
        publication.require_prewrite(batch_id=batch,
            payload_ceiling_bytes=sum(map(len, payloads)), paths=list(map(str, paths)))
        local = spool.reserve(batch, sum(map(len, payloads)))
        references = []
        for path, raw in zip(paths, payloads):
            source = local / path.name
            with source.open('xb') as stream:
                stream.write(raw)
            ref = spool.record(batch, FixtureReference(str(source), path.name,
                len(raw), bytes_sha256hex(raw)), directory, read_back=True)
            references.append(ref)
        spool.submit(batch)
        spool.await_group(batch)
        with spool.local_reads(references) as held:
            assert [held[ref].read_bytes() for ref in references] == payloads
            for ref in references:
                spool.retire_entry(batch, ref)
            assert all(path.is_file() for path in held.values())
        assert spool.released(batch)
        spool.prepare_reads(references)
        with spool.local_reads(references) as held:
            assert [held[ref].read_bytes() for ref in references] == payloads
        assert [path.read_bytes() for path in paths] == payloads
    spool.join_copies()
    if (args.completion_ready is None) != (args.completion_release is None):
        raise ValueError('completion gate requires both explicit FIFO paths')
    if args.completion_ready is not None:
        import select
        ready = os.open(args.completion_ready, os.O_WRONLY | os.O_NOFOLLOW)
        try:
            os.write(ready, b'ready')
        finally:
            os.close(ready)
        release = os.open(args.completion_release, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        try:
            if not select.select([release], [], [], 60)[0] or os.read(release, 16) != b'release':
                raise RuntimeError('completion gate did not publish explicit release')
        finally:
            os.close(release)
    publication.finalize(Path(publication.producer['record_dir']) / 'publication.json',
                         checkpoint_exit=args.payload_exit or None)
    print('PASS byte fixture: actual native declaration, copies, local readback and rematerialization', flush=True)
    return args.payload_exit


if __name__ == '__main__':
    raise SystemExit(main())
