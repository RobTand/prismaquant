"""Progress may follow only a durably published measurement record."""
import json
import os
import stat

import pytest

from experiments import profile_expert_routing_1931 as control


def test_measurement_record_fsyncs_file_then_rename_then_directory(tmp_path, monkeypatch):
    target = tmp_path / 'partial.json'
    events = []
    fsync, replace = os.fsync, os.replace

    def observed_fsync(fd):
        kind = 'directory' if stat.S_ISDIR(os.fstat(fd).st_mode) else 'file'
        if kind == 'directory':
            assert json.loads(target.read_text()) == {'committed': 1}
        fsync(fd)
        events.append(kind)

    def observed_replace(source, destination):
        replace(source, destination)
        events.append('rename')

    monkeypatch.setattr(control.os, 'fsync', observed_fsync)
    monkeypatch.setattr(control.os, 'replace', observed_replace)
    control._write(target, {'committed': 1})
    events.append('returned')
    assert events == ['file', 'rename', 'directory', 'returned']
    assert not target.with_name(target.name + '.writing').exists()


def test_failed_directory_fsync_prevents_progress_and_closes_fd(tmp_path, monkeypatch):
    fsync = os.fsync
    directory_fds = []
    committed = []

    def failing_directory_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            directory_fds.append(fd)
            raise OSError('injected directory persistence failure')
        fsync(fd)

    monkeypatch.setattr(control.os, 'fsync', failing_directory_fsync)
    with pytest.raises(OSError, match='directory persistence failure'):
        control._write(tmp_path / 'partial.json', {'committed': 1})
        committed.append(1)
    assert not committed
    assert len(directory_fds) == 1
    with pytest.raises(OSError):
        os.fstat(directory_fds[0])
