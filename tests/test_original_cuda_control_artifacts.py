"""Bound native sidecars, never a synthetic CUDA/source qualification."""
import hashlib
import json
from pathlib import Path

import pytest

from experiments.original_cuda_control import publish_control_artifacts
from prismaquant import native_results as nr, source_generation as sg
from native_runtime_fixtures import runtime_layout, nonqualifying_runtime  # noqa: F401


@pytest.fixture
def sidecars(tmp_path, monkeypatch, nonqualifying_runtime):
    # Unit fixture for the writer only. Real native provenance is exercised by
    # tools/native_result_cpu_proof.py through the admitted native proof.
    monkeypatch.setattr(nr, 'execution_context', lambda: {'unit_fixture': True})
    (tmp_path / 'controls').mkdir()
    roles = {'control': tmp_path / 'controls' / 'actual.json',
        'execution': tmp_path / 'test-result.json', 'action_result': tmp_path / 'action-result.json',
        'netdata_sparky': tmp_path / 'netdata-sparky.json',
        'netdata_sparklina': tmp_path / 'netdata-sparklina.json', 'torch_trace': tmp_path / 'trace.json'}
    for role, path in roles.items():
        path.write_bytes((role + '\n').encode())
    control = tmp_path / 'input.txt'; control.write_bytes(b'known explicit input\n')
    inputs = {'control': {'path': str(control), 'sha256': hashlib.sha256(control.read_bytes()).hexdigest(),
                           'bytes': len(b'known explicit input\n')}}
    return roles, inputs


def test_publication_binds_all_actual_node_sidecars(tmp_path, sidecars, capsys):
    roles, inputs = sidecars
    binding = publish_control_artifacts('actual-node', tmp_path,
        {'path': str(roles['torch_trace'])}, inputs=inputs, max_artifact_bytes=4096)
    raw = Path(binding['path']).read_bytes()
    assert len(raw) == binding['bytes'] and hashlib.sha256(raw).hexdigest() == binding['sha256']
    publication = json.loads(raw)
    assert publication['schema'] == nr.PUBLICATION_SCHEMA
    assert publication['producer_id'] == 'actual-node'
    assert set(publication['artifacts']) == set(roles)
    for role, path in roles.items():
        row = publication['artifacts'][role]
        assert row['path'] == str(path) and row['bytes'] == len((role + '\n').encode())
        assert nr._owned({k: row[k] for k in ('path', 'sha256', 'bytes')}, 4096,
                         identity=row['file_identity']) == (role + '\n').encode()
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    announcement = json.loads(lines[0].removeprefix(nr.ANNOUNCEMENT_PREFIX.decode()))
    assert announcement == {'schema': nr.ANNOUNCEMENT_SCHEMA, **binding}
    with pytest.raises(FileExistsError):
        publish_control_artifacts('actual-node', tmp_path, {'path': str(roles['torch_trace'])},
                                  inputs=inputs, max_artifact_bytes=4096)


@pytest.mark.parametrize('kind', ['empty', 'directory', 'symlink'])
def test_bind_result_artifact_refuses_nonregular_empty_or_indirect_bytes(tmp_path, kind):
    path = tmp_path / 'artifact'
    if kind == 'directory':
        path.mkdir()
    elif kind == 'symlink':
        target = tmp_path / 'target'; target.write_bytes(b'known bytes'); path.symlink_to(target)
    else:
        path.touch()
    with pytest.raises((OSError, ValueError)):
        nr.bind_result_artifact(path, max_bytes=4096)


@pytest.mark.parametrize('count', [0, 2])
def test_publication_refuses_missing_or_ambiguous_control(tmp_path, capsys, count):
    (tmp_path / 'controls').mkdir()
    for index in range(count):
        (tmp_path / 'controls' / f'{index}.json').write_bytes(b'control')
    with pytest.raises(RuntimeError, match='exactly one actual control'):
        publish_control_artifacts('actual-node', tmp_path, None, inputs={}, max_artifact_bytes=4096)
    assert capsys.readouterr().out == ''


def test_selected_original_reader_does_not_accept_legacy_or_cpu_dicts():
    for value in ({}, {'payload': b'old CAS bytes'}, {'producer_context': {}}):
        with pytest.raises(RuntimeError, match='verified native bytes'):
            sg._original_artifact_publication(value, node_id='actual-node', roles={'control'})
