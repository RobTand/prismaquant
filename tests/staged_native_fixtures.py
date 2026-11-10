"""Real immutable extent fixtures, independent of executor SDKs."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

from prismaquant.staged_manifest import SCHEMA, publish_staged_json


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fixture(tmp_path, payload=b'known-fixture\x00\xff', *, ranges=None):
    root = tmp_path / 'phase'
    root.mkdir()
    source = tmp_path / 'canonical-unavailable.bin'
    source_id = {'path': str(source), 'bytes': len(payload), 'sha256': sha(payload)}
    rows = []
    for number, (start, end) in enumerate(ranges or [(0, len(payload))]):
        extent = payload[start:end]
        blob = b'NOT-A-HEADER:' + extent + b':TRAILER'
        obj = root / f'object-{number}'
        obj.write_bytes(blob)
        rows.append({'source': source_id, 'source_offset': start, 'length': end - start,
                     'object_offset': 13, 'sha256': sha(extent),
                     'object': {'path': str(obj), 'bytes': len(blob), 'sha256': sha(blob)}})
    contract = tmp_path / 'read-contract.json'
    contract_sha = publish_staged_json(contract, {'schema': 'prismaquant.read_contract.v1',
        'entries': [{'path': str(source), 'offset': r['source_offset'], 'bytes': r['length'],
                     'sha256': r['sha256']} for r in rows]})
    data = {'schema': SCHEMA, 'read_contract_sha256': contract_sha,
            'phase_id': 'unit2-fixture', 'producer': {'source_commit': '1' * 40, 'receipt_sha256': '2' * 64},
            'lifetime': {'mode': 'retained_shared_phase', 'namespace': str(root),
                         'reader_scopes': ['unit2-owned-reader'],
                         'retention': 'until_all_readers_stopped_or_fenced'}, 'entries': rows}
    filename = tmp_path / 'staged.json'
    digest = publish_staged_json(filename, data)
    binding = {'schema': 'prismaquant.staged_launch.v1', 'path': str(filename), 'sha256': digest,
               'read_contract_path': str(contract), 'read_contract_sha256': contract_sha, 'allowed_tiers': 'ssd'}
    return source, data, binding, contract


def bind(monkeypatch, binding):
    from prismaquant import residency_map
    # Native launch installs these explicit immutable bindings before imports.
    monkeypatch.setenv(residency_map.ENV_VAR, binding['path'])
    monkeypatch.setenv('PRISMAQUANT_STAGED_MANIFEST_SHA256', binding['sha256'])
    monkeypatch.setenv('PRISMAQUANT_READ_CONTRACT_SHA256', binding['read_contract_sha256'])
    monkeypatch.setenv('PRISMAQUANT_READ_CONTRACT_PATH', binding['read_contract_path'])
    residency_map.reset_residency_resolver_for_tests()
    return residency_map.residency_resolver()


def publish_inputs(tmp_path: Path, files, *, ranges=None, read_contract=None):
    """Materialize real fixture files through the existing immutable publisher."""
    from prismaquant.staged_manifest import materialize
    from prismaquant.staged_launch import validate_binding

    files = [Path(filename) for filename in files]
    assert files and len(files) == len(set(files)) <= 1024
    root = tmp_path / 'native-inputs'
    root.mkdir()
    entries, scientific = [], []
    if ranges is not None:
        assert set(ranges) == set(files)
    for filename in files:
        raw = filename.read_bytes()
        source = {'path': str(filename), 'bytes': len(raw), 'sha256': sha(raw)}
        for start, end in (ranges[filename] if ranges is not None else [(0, len(raw))]):
            assert type(start) is int and type(end) is int and 0 <= start < end <= len(raw)
            digest = sha(raw[start:end])
            entries.append({'source': source, 'source_offset': start,
                            'length': end - start, 'sha256': digest})
            scientific.append({'path': str(filename), 'offset': start,
                               'bytes': end - start, 'sha256': digest})
    contract = tmp_path / 'native-read-contract.json'
    if read_contract is None:
        digest = publish_staged_json(contract, {'schema': 'prismaquant.read_contract.v1',
                                               'entries': scientific})
    else:
        with contract.open('xb') as stream:
            stream.write(read_contract)
        digest = sha(read_contract)
    filename = tmp_path / 'native-staged.json'
    staged = materialize(entries, destination=root, manifest_path=filename,
        read_contract_sha256=digest, phase_id='4c-owned-fixture',
        producer={'source_commit': '1' * 40, 'receipt_sha256': '2' * 64},
        lifetime={'mode': 'retained_shared_phase', 'namespace': str(root),
                  'reader_scopes': ['4c-owned-reader'],
                  'retention': 'until_all_readers_stopped_or_fenced'})
    binding = {'schema': 'prismaquant.staged_launch.v1', 'path': str(filename),
               'sha256': staged, 'read_contract_path': str(contract),
               'read_contract_sha256': digest, 'allowed_tiers': 'ssd'}
    validate_binding(binding)
    return binding


@pytest.fixture
def native_staged_state(monkeypatch):
    """Own native staged globals and environment for one test, with no SDK."""
    from prismaquant import residency_map, stage_b_prep_io, staged_tier_policy

    monkeypatch.setenv('PRISMAQUANT_LAYER_READ_THREADS', '2')
    for name in ('PRISMAQUANT_STAGED_MANIFEST', 'PRISMAQUANT_STAGED_MANIFEST_SHA256',
                 'PRISMAQUANT_READ_CONTRACT_SHA256', 'PRISMAQUANT_READ_CONTRACT_PATH',
                 'PRISMAQUANT_STAGED_ALLOWED_TIERS', 'FLUXCACHE_MAP_FILE'):
        monkeypatch.delenv(name, raising=False)
    assert residency_map._RESOLVER is None, 'prior test left a staged resolver'
    monkeypatch.setattr(staged_tier_policy, '_ACTIVE', None)
    monkeypatch.setattr(staged_tier_policy, '_OWNER', None)
    monkeypatch.setattr(staged_tier_policy, '_STACK', [])
    stage_b_prep_io.reset_staged_reads_for_tests()
    try:
        yield
    finally:
        stage_b_prep_io.reset_staged_reads_for_tests()
        residency_map.reset_residency_resolver_for_tests()
        assert not any(name == 'prismabuild' or name.startswith('prismabuild.') for name in sys.modules)


@pytest.fixture
def empty_native_staged_inputs(tmp_path, monkeypatch, native_staged_state):
    """Bind a real manifest for CPU controls whose payload IO is stubbed."""
    _, _, binding, _ = fixture(tmp_path)
    bind(monkeypatch, binding)
    from prismaquant import tessera_joint_aura, residency_map
    binder = residency_map.bind_residency_manifest
    monkeypatch.setattr(tessera_joint_aura, 'bind_residency_manifest',
        lambda expected: binder(binding['read_contract_sha256'] if expected is None else expected))
    monkeypatch.setattr(residency_map, 'bind_residency_manifest',
        lambda expected: binder(binding['read_contract_sha256'] if expected is None else expected))
    return binding
