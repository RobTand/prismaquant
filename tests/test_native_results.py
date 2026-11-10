"""Bounded publication bytes and measured runtime policy, independent of CUDA."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from prismaquant import native_results as nr
from prismaquant.mount_facts import find_mount


@pytest.fixture
def mapped_library(tmp_path, monkeypatch):
    directory = tmp_path / 'sub volume'
    directory.mkdir()
    library = directory / 'library.so'
    library.write_bytes(b'known mapped bytes')
    table = tmp_path / 'mountinfo'
    point = str(directory).replace(' ', r'\040')

    def configure(*, stat_device=31, stat_inode=4242, maps_device='00:1e',
                  mount_device='0:30', filesystem='btrfs', deleted=False, extra_mounts='',
                  mountpoint=point):
        raw = (f'1000-2000 r-xp 00000000 {maps_device} 4242 {library}'
               + (' (deleted)' if deleted else '') + '\n').encode()
        table.write_text('1 0 8:1 / / rw - ext4 /dev/root rw\n'
            f'2 1 {mount_device} /@ {mountpoint} rw - {filesystem} /dev/data rw\n'
            + extra_mounts)
        info = SimpleNamespace(st_dev=stat_device, st_ino=stat_inode,
                               st_size=18, st_mtime_ns=100, st_ctime_ns=200)
        monkeypatch.setattr(nr, 'open', lambda path, mode: io.BytesIO(raw), raising=False)
        monkeypatch.setattr(nr, 'os', SimpleNamespace(stat=lambda path: info,
                                                     major=os.major, minor=os.minor))
        monkeypatch.setattr(nr, 'find_mount', lambda path: find_mount(path, table=table))
        return library

    return configure


@pytest.mark.parametrize('filesystem, stat_device', [('btrfs', 31), ('ext4', 30),
                                                    ('overlay', 30), ('zfs', 30)])
def test_mapped_library_identity_accepts_exact_device_or_btrfs_mount(mapped_library, filesystem, stat_device):
    library = mapped_library(filesystem=filesystem, stat_device=stat_device)
    assert nr._runtime_libraries() == {str(library): (stat_device, 4242, 18, 100, 200)}


@pytest.mark.parametrize('change', [
    {'stat_inode': 4243},
    {'stat_inode': 4243, 'stat_device': 30},
    {'maps_device': '00:20'},
    {'mount_device': '0:32'},
    {'mount_device': 'bad'},
    {'filesystem': 'ext4'},
    {'filesystem': 'overlay'},
    {'deleted': True},
])
def test_mapped_library_identity_refuses_replaced_or_unverifiable_owner(mapped_library, change):
    mapped_library(**change)
    with pytest.raises(ValueError, match='mapped library replaced|unsupported mapped implementation owner'):
        nr._runtime_libraries()


def test_mapped_library_mount_must_contain_resolved_target(mapped_library, tmp_path):
    library = mapped_library()
    target = tmp_path / 'other' / library.name
    target.parent.mkdir()
    target.write_bytes(b'other file, same synthetic inode')
    library.unlink()
    library.symlink_to(target)
    # Even an equal inode cannot borrow the old path's btrfs mount device.
    with pytest.raises(ValueError, match='mapped library replaced'):
        nr._runtime_libraries()


def test_mapped_library_does_not_borrow_a_path_prefix_mount(mapped_library, tmp_path):
    mapped_library(mountpoint=str(tmp_path / 'sub'))
    with pytest.raises(ValueError, match='mapped library replaced'):
        nr._runtime_libraries()


def test_mapped_library_uses_topmost_mount_at_same_path(mapped_library, tmp_path):
    point = str(tmp_path / 'sub volume').replace(' ', r'\040')
    mapped_library(extra_mounts=f'3 2 0:32 / {point} rw - btrfs /dev/other rw\n')
    with pytest.raises(ValueError, match='mapped library replaced'):
        nr._runtime_libraries()


@pytest.mark.parametrize('stat_device', [30, 31])
def test_mapped_library_missing_mountinfo_only_allows_exact_device(mapped_library, tmp_path, stat_device):
    library = mapped_library(stat_device=stat_device)
    (tmp_path / 'mountinfo').unlink()
    if stat_device == 30:
        assert nr._runtime_libraries() == {str(library): (30, 4242, 18, 100, 200)}
    else:
        with pytest.raises(ValueError, match='mapped library replaced'):
            nr._runtime_libraries()


def test_measured_interpreter_is_the_actual_executable():
    measured = subprocess.run([sys.executable, '-s', '-B', '-c',
        'import runpy; runpy.run_path("tests/native_runtime_observer.py", run_name="__main__")'],
        check=False, capture_output=True, text=True)
    assert measured.returncode == 0, measured.stderr
    runtime = json.loads(measured.stdout)
    assert nr.validate_runtime(runtime) == runtime
    executable = Path(sys.executable).resolve(strict=True)
    raw = executable.read_bytes()
    assert runtime['python']['path'] == str(executable)
    assert runtime['python']['bytes'] == len(raw)
    assert runtime['python']['sha256'] == hashlib.sha256(raw).hexdigest()
    package = runtime['dependencies']['pytest']
    assert b'Name: pytest' in Path(package['metadata']).read_bytes()
    for role in ('metadata', 'record'):
        raw = Path(package[role]).read_bytes()
        assert runtime['files'][package[role]]['bytes'] == len(raw)
        assert runtime['files'][package[role]]['sha256'] == hashlib.sha256(raw).hexdigest()
    import pytest as installed
    entry = package['imports']['pytest']
    raw = Path(installed.__file__).read_bytes()
    assert entry['origin'] == str(Path(installed.__file__).resolve())
    assert runtime['files'][entry['origin']]['bytes'] == len(raw)
    assert runtime['files'][entry['origin']]['sha256'] == hashlib.sha256(raw).hexdigest()


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


def test_publish_result_refuses_an_existing_publication_without_replacing_it(tmp_path, monkeypatch, capsys):
    """A publication path that exists is never replaced: its bytes and the announced result are unchanged.

    Terminal authority (the running job and its runtime) is a controlled double; the publisher's file
    writing, role roster, byte bounds and exclusive-link publication are real.
    """
    monkeypatch.setattr(nr, 'execution_context', lambda: {'schema': nr.PRODUCER_SCHEMA, 'job_id': 'f2'})
    monkeypatch.setattr(nr, 'measure_runtime', lambda *, dependencies=(): {})
    source = tmp_path / 'input.bin'
    source.write_bytes(b'input')
    artifact = tmp_path / 'artifact.bin'
    artifact.write_bytes(b'artifact')
    inputs = {'source': {'path': str(source), 'sha256': hashlib.sha256(b'input').hexdigest(), 'bytes': 5}}
    path = tmp_path / 'publication.json'
    first = nr.publish_result(path, producer_id='p', inputs=inputs, artifacts={'result': artifact},
                              max_artifact_bytes=1 << 20)
    announced = capsys.readouterr().out
    assert nr.ANNOUNCEMENT_PREFIX.decode() in announced
    before = path.read_bytes()
    artifact.write_bytes(b'artifact-changed')
    with pytest.raises(FileExistsError):
        nr.publish_result(path, producer_id='p', inputs=inputs, artifacts={'result': artifact},
                          max_artifact_bytes=1 << 20)
    assert path.read_bytes() == before
    assert hashlib.sha256(before).hexdigest() == first['sha256']
    assert capsys.readouterr().out == ''
