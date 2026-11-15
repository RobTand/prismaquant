#!/usr/bin/env python3
"""Bind disjoint original Tessera cached-unit children without copying wires.

Each v1 child names its historical source package. Tessera's existing bundle
reader authenticates the child documents, source equality, exact producer
coverage and complete disjoint plan roster before this parent is published.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
