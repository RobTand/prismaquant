"""Census publication keeps its owned bytes, refusal order and config fence."""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pytest

from tests.golden_table import GoldenTable
from test_tessera_census_cache import _build_tool
from prismaquant.cluster_campaign import CampaignContractError


@pytest.fixture(scope="module")
def golden():
    return GoldenTable("census_publisher_digest_owner_1301")


@pytest.fixture
def tool():
    return _build_tool()


@pytest.mark.parametrize("raw", [b"", b"\x00\xff\r\n", "é😀\n".encode()])
def test_old_bound_input_bytes_and_wrong_digest(tmp_path, tool, golden, raw):
    path = tmp_path / "input.bin"
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    golden.call(lambda: tool._bound(str(path), digest, "input"), tmp=tmp_path)
    golden.call(lambda: tool._bound(str(path), "0" * 64, "input"), tmp=tmp_path)


@pytest.mark.parametrize("directory", [False, True])
def test_old_bound_open_errors(tmp_path, tool, golden, directory):
    path = tmp_path / "input.bin"
    if directory:
        path.mkdir()
    golden.call(lambda: tool._bound(str(path), "0" * 64, "input"), tmp=tmp_path)


@pytest.mark.parametrize("raw", [b"", b"\x00\xff\r\n", "é😀\n".encode()])
def test_old_atomic_publication_digest_and_existing_file(tmp_path, tool, golden, raw):
    path = tmp_path / "nested" / "output.bin"
    golden.call(lambda: (tool._write(path, raw), path.read_bytes()))
    golden.call(lambda: tool._write(path, b"replacement"), tmp=tmp_path)
    assert path.read_bytes() == raw
    assert list(path.parent.iterdir()) == [path]


def test_old_atomic_publication_refuses_dangling_symlink(tmp_path, tool, golden):
    path = tmp_path / "output.bin"
    path.symlink_to(tmp_path / "absent")
    golden.call(lambda: tool._write(path, b"replacement"), tmp=tmp_path)
    assert path.is_symlink() and not path.exists()


def _projection_inputs(tmp_path, tool, monkeypatch, *, mutate=False):
    config = {"x": {"data_type": "float", "bits": 16}}
    raw = json.dumps(config).encode()
    inputs = {
        "costs": pickle.dumps({"costs": {"x": {}}}),
        "census": b'{"unit_shapes": {}}',
        "roster": json.dumps({"identity_sha256": "e" * 64}).encode(), "layer-config": raw,
    }
    argv = []
    for flag, content in inputs.items():
        path = tmp_path / flag
        path.write_bytes(content)
        argv += [f"--{flag}", str(path), f"--{flag}-sha256", hashlib.sha256(content).hexdigest()]
    plan = tmp_path / "plan.json"
    argv += ["--out-dir", str(tmp_path / "out"), "--checkpoint-parts", str(tmp_path / "parts"),
             "--plan-layer-config-out", str(plan)]
    monkeypatch.setattr(tool, "load_assignment", lambda _: {"x": "BF16"})
    monkeypatch.setattr(tool, "read_layer_config_metadata", lambda _: {})

    def selected(*_):
        if mutate:
            (tmp_path / "layer-config").write_bytes(raw + b"\n")
        return ({"x": "BF16"},)

    monkeypatch.setattr(tool, "selected_census_assignment", selected)
    monkeypatch.setattr(tool, "plan_layer_config_projection", lambda value, _: (value, []))

    def stop_after_projection(*_, **__):
        raise RuntimeError("fixture stopped after the checked projection")

    monkeypatch.setattr(tool, "load_selected_wire_records", stop_after_projection)
    return argv, raw, plan


@pytest.mark.parametrize("mutate", [False, True])
def test_old_main_post_binding_config_fence(tmp_path, tool, golden, monkeypatch, mutate):
    argv, _, plan = _projection_inputs(tmp_path, tool, monkeypatch, mutate=mutate)
    golden.call(lambda: tool.main(argv), tmp=tmp_path)
    golden.value(plan.read_bytes() if plan.exists() else None)


def test_bound_routes_already_read_bytes_once(tmp_path, tool, monkeypatch):
    path = tmp_path / "input.bin"
    raw = b"\x00\xff\r\n"
    path.write_bytes(raw)
    calls = []
    monkeypatch.setattr(tool, "bytes_sha256hex", lambda value: calls.append(value) or "owner")
    assert tool._bound(str(path), "owner", "input") == raw
    assert calls == [raw]


def test_write_publishes_before_owner_without_readback(tmp_path, tool, monkeypatch):
    events = []
    path, raw = tmp_path / "output.bin", b"\x00\xff\r\n"
    publish = tool._atomic_write_new_bytes

    def observed_publication(target, value):
        publish(target, value)
        events.append(("published", target, value))

    def owner(value):
        events.append(("digested", value))
        return "owner"

    def forbidden_readback(_):
        raise AssertionError("publication must digest its owned bytes, not reread the file")

    monkeypatch.setattr(tool, "_atomic_write_new_bytes", observed_publication)
    monkeypatch.setattr(tool, "bytes_sha256hex", owner)
    monkeypatch.setattr(Path, "read_bytes", forbidden_readback)
    assert tool._write(path, raw) == "owner"
    assert events == [("published", path, raw), ("digested", raw)]
    events.clear()
    with pytest.raises(CampaignContractError):
        tool._write(path, b"replacement")
    assert events == []


def test_main_rechecks_config_and_digests_projection_through_owner(tmp_path, tool, monkeypatch):
    argv, raw, plan = _projection_inputs(tmp_path, tool, monkeypatch)
    calls = []

    def owner(value):
        calls.append(value)
        return hashlib.sha256(value).hexdigest()

    monkeypatch.setattr(tool, "bytes_sha256hex", owner)
    with pytest.raises(RuntimeError, match="fixture stopped after the checked projection"):
        tool.main(argv)
    projected = plan.read_bytes()
    assert calls[-3:] == [raw, raw, projected]
