"""PrismaSnap publication contracts retained from 05b62b47427 for PRI-226."""
from __future__ import annotations

import os
import stat

import pytest

from prismaquant import prismasnap_checkpoint as checkpoint


PAYLOAD = {"z": -0.0, "a": ["é", "line\n", None]}
ENCODED = b'{\n  "a": [\n    "\xc3\xa9",\n    "line\\n",\n    null\n  ],\n  "z": -0.0\n}'


def test_json_uses_the_byte_publishers_failure_boundary(monkeypatch, tmp_path):
    output = tmp_path / "receipt.json"
    calls = []

    def refuse(path, data):
        calls.append((path, data))
        raise OSError("byte publication refused")

    monkeypatch.setattr(checkpoint, "_atomic_bytes", refuse)
    with pytest.raises(OSError, match="^byte publication refused$"):
        checkpoint._atomic_json(output, PAYLOAD)
    assert calls == [(output, ENCODED)]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", [None, "file_sync", "replace", "directory_sync"])
def test_json_publication_retains_bytes_and_durability_order(
    monkeypatch, tmp_path, failure,
):
    output = tmp_path / "receipt.json"
    temporary = tmp_path / ".receipt.json.tmp"
    output.write_bytes(b"previous")
    events = []
    real_fsync = os.fsync
    real_replace = os.replace

    def fsync(fd):
        event = "directory_sync" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file_sync"
        events.append(event)
        if event == "file_sync":
            assert temporary.read_bytes() == ENCODED
            assert output.read_bytes() == b"previous"
        else:
            assert not temporary.exists()
            assert output.read_bytes() == ENCODED
        if failure == event:
            raise OSError(event)
        real_fsync(fd)

    def replace(source, destination):
        events.append("replace")
        assert source == temporary
        assert destination == output
        assert temporary.read_bytes() == ENCODED
        assert output.read_bytes() == b"previous"
        if failure == "replace":
            raise OSError("replace")
        real_replace(source, destination)

    monkeypatch.setattr(checkpoint.os, "fsync", fsync)
    monkeypatch.setattr(checkpoint.os, "replace", replace)
    if failure is None:
        checkpoint._atomic_json(output, PAYLOAD)
    else:
        with pytest.raises(OSError, match=f"^{failure}$"):
            checkpoint._atomic_json(output, PAYLOAD)

    ordered = ["file_sync", "replace", "directory_sync"]
    expected_events = ordered if failure is None else ordered[:ordered.index(failure) + 1]
    assert events == expected_events
    unpublished = failure in {"file_sync", "replace"}
    assert output.read_bytes() == (b"previous" if unpublished else ENCODED)
    assert temporary.exists() == unpublished
    if unpublished:
        assert temporary.read_bytes() == ENCODED


@pytest.mark.parametrize("kind", ["file", "directory", "dangling_symlink"])
def test_json_refuses_stale_temporaries_without_overwriting(tmp_path, kind):
    output = tmp_path / "receipt.json"
    temporary = tmp_path / ".receipt.json.tmp"
    output.write_bytes(b"previous")
    if kind == "file":
        temporary.write_bytes(b"stale")
    elif kind == "directory":
        temporary.mkdir()
    else:
        temporary.symlink_to(tmp_path / "absent")
    with pytest.raises(RuntimeError) as caught:
        checkpoint._atomic_json(output, PAYLOAD)
    assert str(caught.value) == f"refusing stale PrismaSnap temporary file {temporary}"
    assert output.read_bytes() == b"previous"
    assert os.path.lexists(temporary)
    if kind == "file":
        assert temporary.read_bytes() == b"stale"
    elif kind == "directory":
        assert temporary.is_dir()
    else:
        assert temporary.is_symlink()


@pytest.mark.parametrize("payload,error", [(float("nan"), ValueError), (object(), TypeError)])
def test_json_encoding_failure_precedes_stale_temporary_refusal(tmp_path, payload, error):
    output = tmp_path / "receipt.json"
    temporary = tmp_path / ".receipt.json.tmp"
    output.write_bytes(b"previous")
    temporary.write_bytes(b"stale")
    with pytest.raises(error):
        checkpoint._atomic_json(output, payload)
    assert output.read_bytes() == b"previous"
    assert temporary.read_bytes() == b"stale"
