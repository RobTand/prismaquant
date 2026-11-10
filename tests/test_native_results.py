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


@pytest.fixture
def mapped_library(tmp_path, monkeypatch):
    library = tmp_path / 'library.so'
    library.write_bytes(b'known mapped bytes')
    regular = nr.open_regular

    def configure(*, stat_device=31, stat_inode=4242, maps_device='00:1e',
                  mount_device='0:30', filesystem='btrfs', deleted=False,
                  fdinfo='mnt_id:\t10\n', mountinfo=Ellipsis, extra_mounts=''):
        maps = (f'1000-2000 r-xp 00000000 {maps_device} 4242 {library}'
                + (' (deleted)' if deleted else '') + '\n')
        if mountinfo is Ellipsis:
            mountinfo = (f'10 1 {mount_device} /@ /unrelated/mountpoint rw - {filesystem} /dev/data rw\n'
                         + extra_mounts)
        info = SimpleNamespace(st_dev=stat_device, st_ino=stat_inode,
                               st_size=18, st_mtime_ns=100, st_ctime_ns=200)
        state = SimpleNamespace(library=library, opened=[], path_info=info, on_fdinfo=None,
                                proc={'maps': maps, 'mountinfo': mountinfo, 'fdinfo': fdinfo})

        def proc_open(path, mode):
            assert mode == 'rb'
            assert path.startswith('/proc/self/')
            kind = 'fdinfo' if path.startswith('/proc/self/fdinfo/') else path.rsplit('/', 1)[1]
            if kind == 'fdinfo' and state.on_fdinfo is not None:
                state.on_fdinfo()
            raw = state.proc[kind]
            if raw is None:
                raise FileNotFoundError(path)
            return io.BytesIO(raw.encode() if isinstance(raw, str) else raw)

        def acquire(path):
            descriptor = regular(path)
            state.opened.append(descriptor)
            return descriptor

        monkeypatch.setattr(nr, 'open', proc_open, raising=False)
        monkeypatch.setattr(nr, 'open_regular', acquire)
        monkeypatch.setattr(nr, 'os', SimpleNamespace(fstat=lambda fd: info,
            stat=lambda path, **kwargs: state.path_info, close=os.close))
        return state

    return configure


@pytest.mark.parametrize('filesystem, stat_device', [('btrfs', 31), ('ext4', 30),
                                                    ('overlay', 31), ('zfs', 30)])
def test_mapped_library_identity_uses_fd_mount_device_for_every_filesystem(mapped_library, filesystem, stat_device):
    state = mapped_library(filesystem=filesystem, stat_device=stat_device)
    assert nr._runtime_libraries() == {str(state.library): (stat_device, 4242, 18, 100, 200)}
    assert len(state.opened) == 1
    with pytest.raises(OSError):
        os.fstat(state.opened[0])


def test_mapped_library_compares_hex_maps_device_with_decimal_mount_device(mapped_library):
    state = mapped_library(maps_device='08:1f', mount_device='8:31')
    assert nr._runtime_libraries() == {str(state.library): (31, 4242, 18, 100, 200)}


@pytest.mark.parametrize('change', [
    {'stat_inode': 4243},
    {'stat_inode': 4243, 'stat_device': 30},
    {'maps_device': '00:20'},
    {'mount_device': '0:32'},
    {'deleted': True},
])
def test_mapped_library_identity_refuses_replaced_owner(mapped_library, change):
    state = mapped_library(**change)
    with pytest.raises(ValueError, match='mapped library replaced|unsupported mapped implementation owner'):
        nr._runtime_libraries()
    for fd in state.opened:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize('change', [
    {'fdinfo': None},
    {'fdinfo': ''},
    {'fdinfo': 'mnt_id: bad\n'},
    {'fdinfo': 'mnt_id: 10\nmnt_id: 10\n'},
    {'fdinfo': 'mnt_id: 20\n'},
    {'mountinfo': None},
    {'mountinfo': ''},
    {'mountinfo': '10 1 0:30\n'},
    {'mountinfo': '10 1 0:30 / / rw - btrfs /dev/data\n'},
    {'mount_device': 'bad'},
    {'mount_device': '0:30:1'},
    {'mount_device': '-1:30'},
    {'extra_mounts': '10 1 0:30 / / rw - btrfs /dev/data rw\n'},
])
def test_mapped_library_refuses_missing_malformed_or_ambiguous_mount_provenance(mapped_library, change):
    # Even a stat device equal to maps cannot replace descriptor provenance.
    state = mapped_library(stat_device=30, **change)
    with pytest.raises((OSError, ValueError)):
        nr._runtime_libraries()
    assert len(state.opened) == 1
    with pytest.raises(OSError):
        os.fstat(state.opened[0])


def test_mapped_library_selects_held_mount_id_among_same_path_mounts(mapped_library):
    state = mapped_library(extra_mounts='20 1 0:32 / /unrelated/mountpoint rw - btrfs /dev/other rw\n')
    assert nr._runtime_libraries() == {str(state.library): (31, 4242, 18, 100, 200)}


def test_mapped_library_different_path_target_cannot_borrow_another_mount_device(mapped_library, tmp_path):
    state = mapped_library(fdinfo='mnt_id: 20\n',
        extra_mounts='20 1 0:32 / /other rw - btrfs /dev/other rw\n')
    target = tmp_path / 'target.so'
    target.write_bytes(b'different target with the same synthetic inode')
    state.library.unlink()
    state.library.symlink_to(target)
    with pytest.raises(ValueError, match='mapped library replaced'):
        nr._runtime_libraries()


def test_mapped_library_path_replacement_while_descriptor_is_held_is_refused(mapped_library, tmp_path):
    state = mapped_library()
    replacement = tmp_path / 'replacement.so'
    replacement.write_bytes(b'replacement bytes')

    def replace():
        replacement.replace(state.library)
        state.path_info = SimpleNamespace(st_dev=31, st_ino=4243,
                                         st_size=18, st_mtime_ns=100, st_ctime_ns=200)

    state.on_fdinfo = replace
    with pytest.raises(ValueError, match='mapped library replaced'):
        nr._runtime_libraries()
    assert len(state.opened) == 1
    with pytest.raises(OSError):
        os.fstat(state.opened[0])


@pytest.mark.parametrize('kind', ['maps', 'fdinfo', 'mountinfo'])
def test_mapped_library_proc_metadata_is_bounded(mapped_library, kind):
    state = mapped_library()
    state.proc[kind] = b'x' * ((4 << 20) + 1)
    with pytest.raises(ValueError, match='proc metadata exceeds bound'):
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
