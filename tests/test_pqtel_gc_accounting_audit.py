"""Real GC CLI and owned temporary CSVs; one exact unlink can be denied."""

import os
import re
import time
from types import SimpleNamespace

from pqtel import cli


def _csv_directory(monkeypatch, tmp_path):
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    monkeypatch.setattr(cli, "CSV_DIR", str(csv_dir))
    real_statvfs = os.statvfs

    def statvfs(path):
        if os.fspath(path) == str(csv_dir):
            return SimpleNamespace(f_bavail=50, f_blocks=100, f_frsize=2**30)
        return real_statvfs(path)

    monkeypatch.setattr(cli.os, "statvfs", statvfs)
    return csv_dir


def _old_csv(directory, name, size):
    path = directory / name
    path.write_bytes(b"x" * size)
    old = time.time() - 40 * 86400
    os.utime(path, (old, old))
    return path


def test_failed_unlink_is_not_reported_as_removed_or_freed(monkeypatch, tmp_path, capsys):
    directory = _csv_directory(monkeypatch, tmp_path)
    blocked = _old_csv(directory, "blocked.csv", 1_000_000)
    removable = _old_csv(directory, "removable.csv", 2_000_000)
    real_unlink = os.unlink

    def unlink(path, *args, **kwargs):
        if os.fspath(path) == str(blocked):
            raise PermissionError("audit denied owned CSV")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(cli.os, "unlink", unlink)
    result = cli.main(["gc"])
    output = capsys.readouterr().out
    assert blocked.exists()
    assert not removable.exists()
    assert "could not remove blocked.csv" in output
    assert "  removed blocked.csv" not in output
    assert re.search(r"\b1 removed,", output), output
    assert "2.0 MB freed" in output
    assert result != 0


def test_successful_unlink_reports_actual_owned_csv(monkeypatch, tmp_path, capsys):
    directory = _csv_directory(monkeypatch, tmp_path)
    old = _old_csv(directory, "old.csv", 1_000_000)
    current = directory / "current.csv"
    current.write_bytes(b"epoch,value\n")
    assert cli.main(["gc"]) == 0
    output = capsys.readouterr().out
    assert not old.exists()
    assert current.exists()
    assert "  removed old.csv" in output
    assert "1 removed, 1 kept, 1.0 MB freed" in output


def test_empty_owned_csv_directory_reports_no_removals(monkeypatch, tmp_path, capsys):
    _csv_directory(monkeypatch, tmp_path)
    assert cli.main(["gc"]) == 0
    assert "0 removed, 0 kept, 0.0 MB freed" in capsys.readouterr().out


def test_dry_run_keeps_owned_csv_and_labels_the_plan(monkeypatch, tmp_path, capsys):
    directory = _csv_directory(monkeypatch, tmp_path)
    old = _old_csv(directory, "old.csv", 1_000_000)
    assert cli.main(["gc", "--dry-run"]) == 0
    output = capsys.readouterr().out
    assert old.exists()
    assert "DRY RUN" in output
    assert "  would remove old.csv" in output
