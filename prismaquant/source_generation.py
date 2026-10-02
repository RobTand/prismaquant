"""Interpret reviewed publisher authority using the existing bound readset.

This has no cache, delivery, network fetch, or persistent manifest of its own.
The caller's sealed control-input binding establishes publication authority;
this module checks that authority's native coordinates and their PB mapping.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .schemas import Contract, strict_json_loads
from .stage_inputs import read_bound, require_source_identity


_contract = Contract(RuntimeError, 'original generation: ')
_require = _contract.require


def _hex(value, length):
    return isinstance(value, str) and len(value) == length and all(c in '0123456789abcdef' for c in value)


def _control(record, label):
    # These are small authority/control inputs, not model payloads. A caller
    # must bind their digests independently in the enclosing reviewed action.
    _require(isinstance(record, dict) and set(record) == {'path', 'sha256'} and
             _hex(record.get('sha256'), 64), f'{label} needs independently bound control input')
    _require(Path(record['path']).stat().st_size <= 16 * 1024**2, f'{label} control input too large')
    return strict_json_loads(read_bound(record, label),
                            duplicate=lambda key: RuntimeError(f'{label}: duplicate key {key}'),
                            constant=lambda name: RuntimeError(f'{label}: invalid constant {name}'))


@dataclass(frozen=True)
class OriginalCoordinate:
    path: str
    size: int
    sha256: str
    git_blob: str | None


def original_generation_coordinates(*, publisher_input, publisher_id, publisher_revision,
                                    readset_input, source_paths, producer_source):
    """Closed native HF sibling roster projected into existing PB whole-file entries.

    A Git auxiliary's SHA256 comes from the separately bound readset; delivered
    bytes must ALSO authenticate to the publisher's native Git blob ID before
    bootstrap. LFS SHA256 and lengths are independently publisher-derived.
    """
    publisher = _control(publisher_input, 'publisher authority')
    _require(isinstance(publisher_id, str) and publisher_id and _hex(publisher_revision, 40),
             'explicit publisher and full revision required')
    _require(isinstance(publisher, dict) and publisher.get('id') == publisher_id and
             publisher.get('sha') == publisher_revision, 'publisher/revision authority mismatch')
    siblings = publisher.get('siblings')
    _require(isinstance(siblings, list) and siblings, 'closed publisher roster missing')
    native = {}
    for row in siblings:
        _require(isinstance(row, dict), 'invalid publisher coordinate')
        name, size = row.get('rfilename'), row.get('size')
        _contract.string(name, where='publisher coordinate')
        _require(isinstance(name, str) and name not in ('', '.', '..') and
                 Path(name).name == name and '/' not in name and '\\' not in name and
                 name not in native and type(size) is int and size > 0,
                 'unsupported or duplicate publisher coordinate')
        lfs = row.get('lfs')
        if lfs is not None:
            _require(isinstance(lfs, dict) and lfs.get('size') == size and
                     _hex(lfs.get('sha256'), 64), f'{name}: invalid publisher LFS authority')
        _require(_hex(row.get('blobId'), 40), f'{name}: native Git blob authority missing')
        native[name] = row
    _require(isinstance(source_paths, dict) and set(source_paths) == set(native),
             'source mapping must cover exactly the closed publisher roster')
    from prismabuild.core import validate_data_manifest
    readset = validate_data_manifest(_control(readset_input, 'original material readset'))
    entries = {(row['path'], row['offset']): row for row in readset['entries']}
    coordinates = {}
    for name, row in native.items():
        path = source_paths[name]
        _contract.absolute_posix_path(path, where=f'{name}: physical mapping')
        entry = entries.get((path, 0))
        _require(entry is not None and entry['bytes'] == row['size'] and
                 _hex(entry['sha256'], 64), f'{name}: complete whole-file readset binding required')
        lfs = row.get('lfs')
        _require(lfs is None or entry['sha256'] == lfs['sha256'],
                 f'{name}: readset differs from publisher LFS digest')
        coordinates[name] = OriginalCoordinate(path, row['size'], entry['sha256'],
                                               None if lfs is not None else row['blobId'])
    producer = require_source_identity(producer_source)
    weights = {name for name in native if name.endswith('.safetensors')}
    _require(weights and set(producer['files']) == weights,
             'complete producer weight roster differs from publisher')
    _require(not (set(producer['files']) & set(producer['auxiliary_sha256'])),
             'producer weight/auxiliary identity overlaps')
    declared = {**producer['files'], **producer['auxiliary_sha256'],
                'config.json': producer['config_sha256']}
    _require('config.json' in native and 'model.safetensors.index.json' in native,
             'whole-file lane requires config and complete checkpoint index')
    for name, digest in declared.items():
        _require(name in coordinates and coordinates[name].sha256 == digest,
                 f'{name}: publisher/readset differs from census producer')
    return MappingProxyType(coordinates), dict(producer['tensors'])
