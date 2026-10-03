"""Selected-result byte binding, not a synthetic CUDA/provider qualification."""
import hashlib
import json

import pytest

from experiments.original_cuda_control import artifact_identity, publish_control_artifacts


def test_publication_binds_all_actual_node_sidecars(tmp_path, capsys):
    (tmp_path / 'controls').mkdir()
    roles = {
        'control': tmp_path / 'controls' / 'actual.json',
        'execution': tmp_path / 'test-result.json',
        'action_result': tmp_path / 'action-result.json',
        'netdata_sparky': tmp_path / 'netdata-sparky.json',
        'netdata_sparklina': tmp_path / 'netdata-sparklina.json',
        'torch_trace': tmp_path / 'trace.json',
    }
    for role, path in roles.items():
        path.write_bytes((role + '\n').encode())
    profile = artifact_identity(roles['torch_trace'])
    publish_control_artifacts('actual-node', tmp_path, profile)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and lines[0].startswith('ORIGINAL_SOURCE_ARTIFACTS ')
    encoded = lines[0].removeprefix('ORIGINAL_SOURCE_ARTIFACTS ')
    value = json.loads(encoded)
    assert encoded == json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    assert set(value) == {'schema', 'node_id', 'artifacts'}
    assert value['schema'] == 'prismaquant.original_source_artifact_publication.v1'
    assert value['node_id'] == 'actual-node'
    assert set(value['artifacts']) == set(roles)
    for role, path in roles.items():
        raw = path.read_bytes()
        assert value['artifacts'][role] == {
            'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


@pytest.mark.parametrize('kind', ['empty', 'directory', 'symlink'])
def test_artifact_binding_refuses_nonregular_empty_or_indirect_bytes(tmp_path, kind):
    path = tmp_path / 'artifact'
    if kind == 'directory':
        path.mkdir()
    elif kind == 'symlink':
        target = tmp_path / 'target'
        target.write_bytes(b'not an authenticated path alias')
        path.symlink_to(target)
    else:
        path.touch()
    with pytest.raises((OSError, RuntimeError)):
        artifact_identity(path)


@pytest.mark.parametrize('count', [0, 2])
def test_publication_refuses_missing_or_ambiguous_control(tmp_path, capsys, count):
    (tmp_path / 'controls').mkdir()
    for index in range(count):
        (tmp_path / 'controls' / f'{index}.json').write_bytes(b'control')
    with pytest.raises(RuntimeError, match='exactly one actual control'):
        publish_control_artifacts('actual-node', tmp_path, None)
    assert capsys.readouterr().out == ''
