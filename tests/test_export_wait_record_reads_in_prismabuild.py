"""The export-wait record PQ writes is the one PrismaBuild reads (PQ #1240).

PrismaBuild #1035 (PB ``b0079ca``) lets an owner blocked on its own
produced-output exports say so: ``<progress path>.export-wait``
(``prismabuild.export_wait.v1``), which the worker's ``no_progress`` rung
reads through ``prismabuild.pool.read_export_wait``. PrismaQuant writes that
record from ``prismabuild_progress.declare_export_wait``, against the wire
format and without importing PrismaBuild, as ``declare_staged_wait`` does:
the rows run where PrismaBuild may not be importable.

The wire-format tests read the record as JSON and need nothing else. The
round-trip tests read it through PrismaBuild's own reader, from whatever
``prismabuild`` this interpreter imports, and skip, naming it, when that one
is absent or predates #1035: then nothing on this interpreter reads the
record, and there is nothing to round-trip against.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prismaquant import prismabuild_progress

# This module imports whatever ``prismabuild`` the interpreter resolves; one
# process holds only one, so it runs apart from the pinned-bundle harnesses
# (PQ #1008, tests/conftest.py).
pytestmark = pytest.mark.own_process

TOKEN = "t" * 32
KEYS = ["cd" * 32, "ab" * 32, "cd" * 32]
SINCE = 1_790_000_000.25


@pytest.fixture
def progress(tmp_path, monkeypatch):
    """This action's progress channel: the progress report's path."""
    path = tmp_path / "action.progress"
    monkeypatch.setenv(prismabuild_progress.PATH_ENV, str(path))
    monkeypatch.setenv(prismabuild_progress.TOKEN_ENV, TOKEN)
    return path


def _prismabuild():
    """PrismaBuild's ``pool`` and ``progress``, or skip naming why not."""
    pool = pytest.importorskip("prismabuild.pool")
    pb_progress = pytest.importorskip("prismabuild.progress")
    if not hasattr(pool, "read_export_wait"):
        pytest.skip(
            f"the prismabuild this interpreter imports ({pool.__file__}) "
            "predates PrismaBuild #1035 (b0079ca): it has no "
            "read_export_wait, so nothing here reads the record")
    return pool, pb_progress


# -- the wire format, read as JSON ----------------------------------------

def test_the_record_is_the_wire_format(progress):
    assert prismabuild_progress.declare_export_wait(KEYS, since_unix=SINCE) is True
    record = json.loads(Path(str(progress) + ".export-wait").read_text())
    assert record == {"schema": "prismabuild.export_wait.v1", "token": TOKEN,
                      "since_unix": SINCE, "exports": sorted(set(KEYS))}
    # Beside the staged-wait record, never in it.
    assert not Path(str(progress) + ".staged-wait").exists()


def test_a_later_declaration_replaces_the_earlier(progress):
    prismabuild_progress.declare_export_wait(KEYS[:1], since_unix=SINCE)
    prismabuild_progress.declare_export_wait(KEYS[1:2], since_unix=SINCE + 1)
    record = json.loads(Path(str(progress) + ".export-wait").read_text())
    assert (record["exports"], record["since_unix"]) == ([KEYS[1]], SINCE + 1)


def test_clear_removes_the_record_and_is_idempotent(progress):
    prismabuild_progress.declare_export_wait(KEYS, since_unix=SINCE)
    assert prismabuild_progress.clear_export_wait() is True
    assert not Path(str(progress) + ".export-wait").exists()
    assert prismabuild_progress.clear_export_wait() is True


def test_without_a_channel_or_a_key_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.delenv(prismabuild_progress.PATH_ENV, raising=False)
    monkeypatch.delenv(prismabuild_progress.TOKEN_ENV, raising=False)
    assert prismabuild_progress.declare_export_wait(KEYS, since_unix=SINCE) is False
    assert prismabuild_progress.clear_export_wait() is False
    path = tmp_path / "action.progress"
    monkeypatch.setenv(prismabuild_progress.PATH_ENV, str(path))
    assert prismabuild_progress.declare_export_wait(KEYS, since_unix=SINCE) is False
    monkeypatch.setenv(prismabuild_progress.TOKEN_ENV, TOKEN)
    assert prismabuild_progress.declare_export_wait([], since_unix=SINCE) is False
    assert list(tmp_path.iterdir()) == []


def test_an_unwritable_destination_does_not_raise(tmp_path, monkeypatch):
    monkeypatch.setenv(prismabuild_progress.PATH_ENV,
                       str(tmp_path / "absent" / "action.progress"))
    monkeypatch.setenv(prismabuild_progress.TOKEN_ENV, TOKEN)
    assert prismabuild_progress.declare_export_wait(KEYS, since_unix=SINCE) is False


# -- read back through PrismaBuild's own reader ---------------------------

def test_prismabuild_names_the_same_record():
    pool, pb_progress = _prismabuild()
    assert pb_progress.EXPORT_WAIT_SCHEMA_V1 == prismabuild_progress.EXPORT_WAIT_SCHEMA
    assert pb_progress.EXPORT_WAIT_SUFFIX == prismabuild_progress.EXPORT_WAIT_SUFFIX
    assert pb_progress.ACTION_PROGRESS_PATH_ENV == prismabuild_progress.PATH_ENV
    assert pb_progress.ACTION_PROGRESS_TOKEN_ENV == prismabuild_progress.TOKEN_ENV


def test_prismabuild_reads_the_record_pq_writes(progress):
    pool, pb_progress = _prismabuild()
    assert prismabuild_progress.declare_export_wait(KEYS, since_unix=SINCE) is True
    where = Path(pb_progress.export_wait_path(str(progress)))
    assert pool.read_export_wait(where, token=TOKEN) == (
        {"exports": sorted(set(KEYS)), "since_unix": SINCE}, "")


def test_prismabuild_reads_a_cleared_wait_as_none(progress):
    pool, pb_progress = _prismabuild()
    prismabuild_progress.declare_export_wait(KEYS, since_unix=SINCE)
    prismabuild_progress.clear_export_wait()
    where = Path(pb_progress.export_wait_path(str(progress)))
    assert pool.read_export_wait(where, token=TOKEN) == (None, "")


def test_prismabuild_refuses_the_record_under_another_launch(progress):
    pool, pb_progress = _prismabuild()
    prismabuild_progress.declare_export_wait(KEYS, since_unix=SINCE)
    where = Path(pb_progress.export_wait_path(str(progress)))
    assert pool.read_export_wait(where, token="u" * 32) == (None, "foreign token")


def test_the_export_wait_and_the_staged_wait_are_two_records(progress):
    """An owner can wait on both at once (PB #1035): each reads on its own,
    and ending one leaves the other."""
    pool, pb_progress = _prismabuild()
    movers = ["ef" * 32]
    prismabuild_progress.declare_staged_wait(movers, since_unix=SINCE)
    prismabuild_progress.declare_export_wait(KEYS, since_unix=SINCE + 2)
    staged = Path(pb_progress.staged_wait_path(str(progress)))
    exports = Path(pb_progress.export_wait_path(str(progress)))
    assert pool.read_staged_wait(staged, token=TOKEN) == (
        {"movers": movers, "since_unix": SINCE}, "")
    assert pool.read_export_wait(exports, token=TOKEN) == (
        {"exports": sorted(set(KEYS)), "since_unix": SINCE + 2}, "")
    prismabuild_progress.clear_export_wait()
    assert pool.read_export_wait(exports, token=TOKEN) == (None, "")
    assert pool.read_staged_wait(staged, token=TOKEN)[0] is not None
