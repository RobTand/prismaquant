"""Selected-result byte binding, not a synthetic CUDA/provider qualification."""
import hashlib
import json
import subprocess
import sys

import pytest

from experiments.original_cuda_control import artifact_identity, publish_control_artifacts, retain_artifact


def test_publication_binds_all_actual_node_sidecars(tmp_path, capsys):
    from prismabuild.core import PrismaBuildCAS

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
    cas = PrismaBuildCAS(tmp_path / 'cas')
    profile = retain_artifact(cas, 'torch_trace', roles['torch_trace'])
    publish_control_artifacts('actual-node', tmp_path, profile, cas)
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
        digest = hashlib.sha256(raw).hexdigest()
        durable = cas.input_path({'id': 'published-' + role, 'sha256': digest, 'bytes': len(raw)})
        assert value['artifacts'][role] == {
            'path': str(durable), 'bytes': len(raw), 'sha256': digest}
        assert durable.read_bytes() == raw
        assert durable.stat().st_mode & 0o222 == 0


def test_original_publication_reader_keeps_ascii_strict_unicode_bytes():
    from prismaquant import source_generation as sg

    node = 'actual-café-\ud800'
    artifacts = {'control': {'path': '/actual/control', 'sha256': 'a' * 64, 'bytes': 1}}
    publication = {'schema': 'prismaquant.original_source_artifact_publication.v1',
                   'node_id': node, 'artifacts': artifacts}
    raw = json.dumps(publication, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    assert b'caf\\u00e9-\\ud800' in raw
    assert sg._original_artifact_publication(sg.ORIGINAL_ARTIFACT_PUBLICATION_PREFIX + raw,
        node_id=node, roles={'control'}) == artifacts


@pytest.mark.parametrize(('raw', 'message', 'cause'), [
    (b'{"schema":1,"schema":2}', 'artifact publication duplicate key schema', None),
    (b'{"schema":NaN}', 'artifact publication invalid constant NaN', None),
    (b'{"schema":Infinity}', 'artifact publication invalid constant Infinity', None),
    (b'{"schema":-Infinity}', 'artifact publication invalid constant -Infinity', None),
    (b'\xff', 'selected artifact publication is not strict JSON', UnicodeDecodeError),
    (b'{', 'selected artifact publication is not strict JSON', json.JSONDecodeError),
])
def test_original_publication_reader_keeps_strict_errors(raw, message, cause):
    from prismaquant import source_generation as sg

    with pytest.raises(RuntimeError) as caught:
        sg._original_artifact_publication(sg.ORIGINAL_ARTIFACT_PUBLICATION_PREFIX + raw,
                                         node_id='actual-node', roles={'control'})
    assert str(caught.value) == message
    assert (None if caught.value.__cause__ is None else type(caught.value.__cause__)) is cause


@pytest.mark.parametrize('encoding', ['spaced', 'literal-unicode'])
def test_original_publication_reader_refuses_noncanonical_encoding(encoding):
    from prismaquant import source_generation as sg

    publication = {'schema': 'prismaquant.original_source_artifact_publication.v1',
                   'node_id': 'actual-café', 'artifacts': {}}
    options = {'sort_keys': True, 'allow_nan': False}
    if encoding == 'literal-unicode':
        options.update(separators=(',', ':'), ensure_ascii=False)
    raw = json.dumps(publication, **options).encode()
    with pytest.raises(RuntimeError) as caught:
        sg._original_artifact_publication(sg.ORIGINAL_ARTIFACT_PUBLICATION_PREFIX + raw,
                                         node_id=publication['node_id'], roles=set())
    assert str(caught.value) == 'original generation: canonical actual artifact publication: identity mismatch'



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
        publish_control_artifacts('actual-node', tmp_path, None, None)
    assert capsys.readouterr().out == ''


def test_retained_artifact_refuses_replaced_granted_bytes(tmp_path):
    from prismabuild.core import PrismaBuildCAS

    path = tmp_path / 'granted-trace'
    path.write_bytes(b'actual original granted bytes')
    identity = artifact_identity(path)
    path.write_bytes(b'different replacement bytes')
    with pytest.raises((ValueError, RuntimeError), match="ingested input"):
        retain_artifact(PrismaBuildCAS(tmp_path / 'cas'), 'torch_trace', path, identity)


def test_artifact_sdk_loader_uses_actual_launch_without_numeric_host_imports(tmp_path):
    program = (
        'import sys; from experiments.original_cuda_control import artifact_cas; '
        'cas=artifact_cas(sys.argv[1]); '
        'from prismabuild.client import SDK_VERSION; assert SDK_VERSION==4; '
        'assert "prismaquant" not in sys.modules and "torch" not in sys.modules; '
        'print(str(cas.root))')
    done = subprocess.run([sys.executable, '-S', '-c', program, str(tmp_path / 'cas')],
                          capture_output=True, text=True, check=True)
    assert done.stdout.strip() == str(tmp_path / 'cas')
