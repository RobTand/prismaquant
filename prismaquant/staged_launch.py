"""Finite staged launch contract and external canonical run-set wrapper."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from .digests import bytes_sha256hex

from .staged_manifest import (PATH_ENV, SHA_ENV, READ_ENV, load_staged_manifest, publish_staged_json)

CACHE_HELPER = '/mnt/shared/flux-migration/caching/fluxcache.py'
CACHE_HELPER_SHA256 = 'a3d53765d2664a7c2c57980005ce3d1eb1d95088102ef992e8a6646cbe1d3e74'
SCHEMA = 'prismaquant.staged_launch.v1'


def validate_binding(binding, *, placement=None):
    if not isinstance(binding, dict) or set(binding) != {'schema', 'path', 'sha256', 'read_contract_path', 'read_contract_sha256', 'allowed_tiers'} or binding['schema'] != SCHEMA or binding['allowed_tiers'] != 'ssd':
        raise ValueError('explicit versioned staged launch binding required')
    manifest = load_staged_manifest(binding['path'], binding['sha256'])
    if manifest.read_contract_sha256 != binding['read_contract_sha256']:
        raise ValueError('launch read contract differs from staged manifest')
    if manifest.lifetime['mode'] == 'host_local_cache' and placement is not None:
        if placement != 'host:' + manifest.lifetime['owner_host']:
            raise ValueError('host-local staged input requires exact owner placement')
    validate_read_contract(binding, binding['read_contract_path'], manifest=manifest)
    return manifest


def validate_read_contract(binding, filename, *, manifest=None):
    """Former data_manifest intent needs exact scientific entries, not a label."""
    from .staged_manifest import read_document, MAX_BYTES, path, integer, digest
    if os.fspath(filename) != binding['read_contract_path']:
        raise ValueError('row scientific manifest path differs from staged launch')
    if manifest is None:
        manifest = load_staged_manifest(binding['path'], binding['sha256'])
    raw, _ = read_document(filename, MAX_BYTES)
    if bytes_sha256hex(raw) != manifest.read_contract_sha256:
        raise ValueError('data manifest digest differs from scientific read contract')
    data = read_contract_document(raw)
    expected = []
    for row in data['entries']:
        expected.append((row['path'], row['offset'], row['bytes'], row['sha256']))
    actual = [(e.source_path, e.source_offset, e.length, e.sha256) for e in manifest.extents]
    if sorted(expected) != sorted(actual):
        raise ValueError('staged manifest does not satisfy exact scientific coverage')
    return data


def read_contract_document(raw):
    import gzip
    import io
    from .staged_manifest import MAX_BYTES, path, integer, digest
    from .schemas import strict_json_loads
    if raw.startswith(b'\x1f\x8b'):
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            raw = stream.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError('scientific read contract exceeds decompressed bound')
    def invalid(key):
        return ValueError(f'invalid scientific read-contract JSON: {key}')
    value = strict_json_loads(raw, duplicate=invalid, constant=invalid)
    if not isinstance(value, dict) or not isinstance(value.get('entries'), list) or not value['entries']:
        raise ValueError('scientific read contract must declare nonempty entries')
    if value.get('schema') == 'prismaquant.prismabuild.data_manifest.v1':
        fields = {'schema', 'produced_by', 'mount_prefix', 'entries', 'entry_count', 'total_bytes', 'annotations'}
        if set(value) != fields or type(value['entry_count']) is not int or value['entry_count'] != len(value['entries']):
            raise ValueError('unsupported legacy scientific read requirements')
    elif set(value) != {'schema', 'entries'} or value['schema'] != 'prismaquant.read_contract.v1':
        raise ValueError('scientific read-contract schema is unqualified')
    extents = {}
    for row in value['entries']:
        if not isinstance(row, dict) or set(row) != {'path', 'offset', 'bytes', 'sha256'}:
            raise ValueError('unsupported scientific read requirements')
        path(row['path']); integer(row['offset']); integer(row['bytes'], positive=True); digest(row['sha256'])
        if value['schema'] == 'prismaquant.prismabuild.data_manifest.v1':
            if not Path(row['path']).is_relative_to(path(value['mount_prefix'])):
                raise ValueError('scientific source is outside its declared mount prefix')
        extents.setdefault(row['path'], []).append((row['offset'], row['offset'] + row['bytes']))
    for ranges in extents.values():
        ordered = sorted(ranges)
        if any(start < previous_end for (_, previous_end), (start, _) in zip(ordered, ordered[1:])):
            raise ValueError('scientific read contract repeats or overlaps ranges')
    if value['schema'] == 'prismaquant.prismabuild.data_manifest.v1' and (
            type(value['total_bytes']) is not int or value['total_bytes'] != sum(row['bytes'] for row in value['entries'])):
        raise ValueError('unsupported legacy scientific read requirements')
    return value


def launch(binding, command):
    manifest = validate_binding(binding)
    if not command:
        raise ValueError('staged launch needs a payload')
    os.environ.update({PATH_ENV: binding['path'], SHA_ENV: binding['sha256'],
                       READ_ENV: binding['read_contract_sha256'],
                       'PRISMAQUANT_READ_CONTRACT_PATH': binding['read_contract_path'],
                       'PRISMAQUANT_STAGED_ALLOWED_TIERS': binding['allowed_tiers']})
    if manifest.lifetime['mode'] == 'host_local_cache':
        from .staged_manifest import read_document
        helper, _ = read_document(CACHE_HELPER, 1 << 20)
        if bytes_sha256hex(helper) != CACHE_HELPER_SHA256:
            raise ValueError('staged cache helper sha256 differs from reviewed pin')
        members = Path(os.environ['TMPDIR']) / 'staged-members.json'
        publish_staged_json(members, manifest.members())
        os.execv('/usr/bin/python3', ['/usr/bin/python3', '-I', '-c', helper.decode('utf-8'),
            '--root', manifest.lifetime['root'], 'run-set', '--members', str(members), '--', *command])
    os.execvpe(command[0], command, os.environ)


if __name__ == '__main__':
    launch(json.loads(sys.argv[1]), sys.argv[2:])
