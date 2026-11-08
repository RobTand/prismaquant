"""Bounded publication bytes and measured runtime policy, independent of CUDA."""
import copy
import hashlib
from pathlib import Path
import sys

import pytest

from prismaquant import native_results as nr


def test_measured_interpreter_is_the_actual_executable():
    runtime = nr.measure_runtime(dependencies=('pytest',))
    assert nr.validate_runtime(runtime) == runtime
    executable = Path(sys.executable).resolve(strict=True)
    raw = executable.read_bytes()
    assert runtime['python']['path'] == str(executable)
    assert runtime['python']['bytes'] == len(raw)
    assert runtime['python']['sha256'] == hashlib.sha256(raw).hexdigest()
    package = runtime['dependencies']['pytest']
    assert b'Name: pytest' in Path(package['metadata']['path']).read_bytes()
    for role in ('metadata', 'record'):
        raw = Path(package[role]['path']).read_bytes()
        assert package[role]['bytes'] == len(raw)
        assert package[role]['sha256'] == hashlib.sha256(raw).hexdigest()
    import pytest as installed
    entry = package['modules']['pytest/__init__.py']
    raw = Path(installed.__file__).read_bytes()
    assert entry['path'] == str(Path(installed.__file__).resolve())
    assert entry['bytes'] == len(raw)
    assert entry['sha256'] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('raw', [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":-Infinity}'])
def test_publication_json_is_strict(raw):
    with pytest.raises(ValueError):
        nr._document(raw)


@pytest.mark.parametrize('mutation', ['missing', 'changed', 'replaced', 'symlink', 'oversize'])
def test_owned_artifact_starts_with_independent_positive_bytes(tmp_path, mutation):
    path = tmp_path / 'artifact'
    raw = b'known-artifact\x00\xff'
    path.write_bytes(raw)
    row = nr.bind_result_artifact(path, max_bytes=128)
    binding = {key: row[key] for key in ('path', 'sha256', 'bytes')}
    assert nr._owned(binding, 128, identity=row['file_identity']) == raw
    if mutation == 'oversize':
        binding['bytes'] = 129
    elif mutation == 'changed':
        path.write_bytes(b'wrong-artifact\x00\xff')
    elif mutation == 'replaced':
        # Both inodes coexist before replacement. No inode-reuse or clock
        # assumption supplies the distinction this fence must observe.
        replacement = tmp_path / 'replacement'
        replacement.write_bytes(raw)
        assert replacement.stat().st_ino != path.stat().st_ino
        replacement.replace(path)
    else:
        path.unlink()
        if mutation == 'symlink':
            target = tmp_path / 'target'; target.write_bytes(raw); path.symlink_to(target)
    with pytest.raises((OSError, ValueError)):
        nr._owned(binding, 128, identity=row['file_identity'])


@pytest.mark.parametrize('mutation', ['extra', 'duplicate-path', 'aggregate', 'zero', 'digest'])
def test_explicit_input_roster_is_closed_and_bounded(mutation):
    roster = {'known': {'path': '/known/input', 'bytes': 3,
                         'sha256': hashlib.sha256(b'abc').hexdigest()}}
    nr._roster(roster, limit=3)
    if mutation == 'extra':
        roster['known']['unexpected'] = True
    elif mutation == 'duplicate-path':
        roster['other'] = dict(roster['known'])
    elif mutation == 'aggregate':
        roster['other'] = dict(roster['known'], path='/known/other')
    elif mutation == 'zero':
        roster['known']['bytes'] = 0
    else:
        roster['known']['sha256'] = 'not-a-digest'
    with pytest.raises(ValueError):
        nr._roster(roster, limit=3)


def test_selection_cannot_adopt_legacy_fields():
    for selection in ({}, {'schema': nr.SELECTION_SCHEMA, 'action_key': 'a' * 64}):
        with pytest.raises(nr.ResultUnavailable):
            nr.read_selected_result(selection)
