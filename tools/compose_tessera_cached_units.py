#!/usr/bin/env python3
"""Bind disjoint original Tessera cached-unit children without copying wires.

Each v1 child names its historical source package. Tessera's existing bundle
reader authenticates the child documents, source equality, exact producer
coverage and complete disjoint plan roster before this parent is published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.cost_stage_checkpoint import publish_new_bytes
from prismaquant.digests import file_sha256hex
from prismaquant.tessera_export_lane import read_cached_unit_bundle
from tessera.cached_unit import (CACHE_SCHEMA, COMPOSED_CACHE_SCHEMA,
                                 CachedUnitBundle, read_manifest)


def compose(children: list[dict], *, output: Path) -> tuple[dict, CachedUnitBundle]:
    if len(children) < 2 or output.exists() or output.is_symlink():
        raise ValueError('cached composition needs at least two children and a new output')
    documents = []
    for child in children:
        path = Path(child['manifest']['path'])
        if not path.is_absolute() or path.is_symlink() or path.resolve() != path:
            raise ValueError('cached child needs a canonical absolute manifest path')
        if file_sha256hex(path) != child['manifest']['sha256']:
            raise ValueError(f'cached child manifest checksum differs: {path}')
        document = read_manifest(path)
        if not isinstance(document, dict) or not isinstance(document.get('units'), dict):
            raise ValueError(f'cached child has no unit document: {path}')
        documents.append(document)
    source = documents[0]['source']
    units = set()
    for child, document in zip(children, documents):
        if document['source'] != source:
            raise ValueError('cached children name different source checkpoints')
        if units & set(document['units']):
            raise ValueError('cached children overlap on selected units')
        units.update(document['units'])
        if document['schema'] == CACHE_SCHEMA and child['producer_package'] is None:
            raise ValueError('original v1 child has no historical producer package')
    manifest = {'schema': COMPOSED_CACHE_SCHEMA, 'source': source,
                'children': sorted(children, key=lambda child: child['manifest']['path'])}
    bundle = read_cached_unit_bundle(manifest, output.parent, units, source)
    return manifest, bundle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child-v1', nargs=4, action='append', default=[],
                        metavar=('MANIFEST', 'SHA256', 'PACKAGE', 'PACKAGE_SHA256'))
    parser.add_argument('--child-v2', nargs=2, action='append', default=[],
                        metavar=('MANIFEST', 'SHA256'))
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    children = [{'manifest': {'path': path, 'sha256': sha},
                 'producer_package': {'path': package, 'sha256': seal}}
                for path, sha, package, seal in args.child_v1]
    children.extend({'manifest': {'path': path, 'sha256': sha},
                     'producer_package': None} for path, sha in args.child_v2)
    output = Path(args.out)
    manifest, bundle = compose(children, output=output)
    raw = (json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                      allow_nan=False) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not publish_new_bytes(output, raw):
        raise FileExistsError(f'cached composition output exists: {output}')
    print(json.dumps({'schema': 'prismaquant.cached_unit_composition_handoff.v1',
                      'manifest': str(output.resolve()),
                      'manifest_sha256': hashlib.sha256(raw).hexdigest(),
                      'children': bundle.child_manifests,
                      'units': len(bundle.units),
                      'encoder_source_proof_mode': bundle.encoder_source_proof_mode,
                      'warnings': bundle.warnings,
                      'export_qualified': False, 'serving_qualified': False},
                     sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
