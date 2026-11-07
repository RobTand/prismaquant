"""Real immutable extent fixtures, independent of executor SDKs."""
import hashlib
import json
from pathlib import Path

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
