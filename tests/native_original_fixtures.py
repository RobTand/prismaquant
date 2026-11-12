"""Original-material fixtures through the real immutable native staged reader."""
from pathlib import Path

import pytest

from prismaquant import residency_map, stage_b_prep_io
from prismaquant.staged_manifest import materialize


@pytest.fixture(autouse=True)
def _forget_state(monkeypatch):
    residency_map.reset_residency_resolver_for_tests()
    stage_b_prep_io.reset_staged_reads_for_tests()
    yield
    residency_map.reset_residency_resolver_for_tests()
    stage_b_prep_io.reset_staged_reads_for_tests()


def _bind_without_policy(monkeypatch, binding):
    """Install a resolver with no staged-tier policy (test-only partial state).

    These fixtures serve original material from the stage, but read their
    calibration and campaign inputs from the pool. Native launch never builds
    this state, and residency_resolver() refuses it. Build it directly.
    """
    for name, value in (('PRISMAQUANT_STAGED_MANIFEST', binding['path']),
                        ('PRISMAQUANT_STAGED_MANIFEST_SHA256', binding['sha256']),
                        ('PRISMAQUANT_READ_CONTRACT_SHA256', binding['read_contract_sha256']),
                        ('PRISMAQUANT_READ_CONTRACT_PATH', binding['read_contract_path'])):
        monkeypatch.setenv(name, value)
    residency_map.reset_residency_resolver_for_tests()
    residency_map._RESOLVER = residency_map.ResidencyResolver(
        binding['path'], manifest_sha256=binding['sha256'],
        read_contract_sha256=binding['read_contract_sha256'])
    return residency_map._RESOLVER


def _stage_manifest(tmp_path, monkeypatch, manifest):
    from prismaquant.digests import bytes_sha256hex, DIRECT_ASCII_SPACED_STRICT
    root = tmp_path / 'native-original-phase'
    root.mkdir()
    contract = tmp_path / 'native-original-contract.json'
    raw = DIRECT_ASCII_SPACED_STRICT.encoded(manifest)
    contract.write_bytes(raw)
    identity = bytes_sha256hex(raw)
    rows = []
    for entry in manifest['entries']:
        source = Path(entry['path'])
        source_raw = source.read_bytes()
        rows.append({'source': {'path': str(source), 'bytes': len(source_raw),
                               'sha256': bytes_sha256hex(source_raw)},
                     'source_offset': entry['offset'], 'length': entry['bytes'],
                     'sha256': entry['sha256']})
    filename = tmp_path / 'native-original-staged.json'
    digest = materialize(rows, destination=root, manifest_path=filename,
        read_contract_sha256=identity, phase_id='original-cpu-scientific-fixture',
        producer={'source_commit': '1' * 40, 'receipt_sha256': '2' * 64},
        lifetime={'mode': 'retained_shared_phase', 'namespace': str(root),
                  'reader_scopes': ['original-cpu-fixture'],
                  'retention': 'until_all_readers_stopped_or_fenced'})
    # The negative controls deliberately rewrite these owned fixture objects.
    for artifact in root.iterdir():
        artifact.chmod(0o644)
    binding = {'schema': 'prismaquant.staged_launch.v1', 'path': str(filename), 'sha256': digest,
               'read_contract_path': str(contract), 'read_contract_sha256': identity, 'allowed_tiers': 'ssd'}
    resolver = _bind_without_policy(monkeypatch, binding)
    return {entry['path']: resolver.staged_read(Path(entry['path']))['stage_path']
            for entry in manifest['entries']}
