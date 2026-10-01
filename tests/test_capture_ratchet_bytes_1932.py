"""Byte and refusal goldens for the shrink-only capture-chain repair."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant import capture_layer_chain as chain
from prismaquant import digests
from prismaquant import tessera_campaign

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import dispatch_capture_chain as dispatch  # noqa: E402
import dispatch_stage_a_split as split_dispatch  # noqa: E402

_PRETTY_CASES = [
    ({}, b"{}\n"),
    ({"z": 1, "a": [True, None]},
     b'{\n  "a": [\n    true,\n    null\n  ],\n  "z": 1\n}\n'),
    ({"\u00e9": "\u96ea\x00\n"}, b'{\n  "\\u00e9": "\\u96ea\\u0000\\n"\n}\n'),
    ({"values": [float("nan"), float("inf"), -float("inf"), -0.0]},
     b'{\n  "values": [\n    NaN,\n    Infinity,\n    -Infinity,\n    -0.0\n  ]\n}\n'),
    ({"surrogate": "\ud800"}, b'{\n  "surrogate": "\\ud800"\n}\n'),
]


@pytest.mark.parametrize("value,expected", _PRETTY_CASES)
def test_existing_dispatch_writers_emit_literal_bytes(tmp_path, value, expected):
    for index, writer in enumerate((dispatch._write_json, split_dispatch._write_json)):
        path = tmp_path / str(index) / "record.json"
        writer(path, value)
        assert path.read_bytes() == expected


def test_named_pretty_profile_encodes_literal_legacy_bytes():
    profile = digests.DIRECT_ASCII_INDENT2_LAX
    for value, expected in _PRETTY_CASES:
        assert profile.text(value) + "\n" == expected.decode("ascii")
        assert profile.encoded(value) + b"\n" == expected
        assert profile.sha256(value) == hashlib.sha256(expected[:-1]).hexdigest()
        assert profile.sha256_streamed(value) == profile.sha256(value)


def test_capture_writer_reuses_the_existing_dispatch_owner():
    assert dispatch._write_json is split_dispatch._write_json


@pytest.mark.parametrize("value", [{"bad": object()}, {"bad": {1, 2}}])
def test_json_writer_native_refusal_after_parent_creation(tmp_path, value):
    path = tmp_path / "parent" / "record.json"
    with pytest.raises(TypeError) as refused:
        dispatch._write_json(path, value)
    assert refused.value.__cause__ is None
    assert path.parent.is_dir()
    assert not path.exists()


def test_capture_execution_receipt_emits_literal_bytes(tmp_path):
    writer = SimpleNamespace(load_execution={"rows": ["\u00e9", float("nan")]},
                             seal_load_execution={"sha": "x"})
    tessera_campaign._write_capture_load_execution(
        SimpleNamespace(cache_dir=tmp_path), writer, guard=None, resources={"z": 1})
    assert (tmp_path / "capture-load-execution.json").read_bytes() == (
        b'{\n  "memory_guard": null,\n  "replay": {\n    "rows": [\n'
        b'      "\\u00e9",\n      NaN\n    ]\n  },\n'
        b'  "resources": {\n    "z": 1\n  },\n'
        b'  "schema": "prismaquant.capture_load_run.v1",\n'
        b'  "seal": {\n    "sha": "x"\n  }\n}\n')


@pytest.mark.parametrize("writer", [None, SimpleNamespace(load_execution=None)])
def test_capture_execution_absence_has_no_io(tmp_path, writer):
    tessera_campaign._write_capture_load_execution(
        SimpleNamespace(cache_dir=tmp_path), writer, guard=None, resources={})
    assert not list(tmp_path.iterdir())


def test_capture_seal_and_reader_preserve_literal_body_and_digest(tmp_path):
    body = {"z": [True, None, -0.0], "schema": "golden", "\u00e9": "\u96ea"}
    encoded = b'{"schema":"golden","z":[true,null,-0.0],"\xc3\xa9":"\xe9\x9b\xaa"}'
    expected = {"schema": "golden", "z": [True, None, -0.0], "\u00e9": "\u96ea",
                "seal": hashlib.sha256(encoded).hexdigest()}
    document = chain._seal(body, where="golden body", field="seal")
    assert document == expected
    path = tmp_path / "sealed.json"
    path.write_bytes(json.dumps(document).encode("utf-8"))
    assert chain._read_sealed(path, schema="golden", field="seal", what="golden body") == expected


@pytest.mark.parametrize("raw,message,cause", [
    (b"{", "no readable golden body at", json.JSONDecodeError),
    (b"\xff", "no readable golden body at", UnicodeDecodeError),
    (b"[]", "is not a golden document", None),
    (b'{"schema":"golden","seal":"wrong"}', "does not seal its own content", None),
])
def test_capture_sealed_reader_preserves_refusals(tmp_path, raw, message, cause):
    path = tmp_path / "sealed.json"
    path.write_bytes(raw)
    with pytest.raises(chain.CaptureChainRefused, match=message) as refused:
        chain._read_sealed(path, schema="golden", field="seal", what="golden body")
    if cause is None:
        assert refused.value.__cause__ is None
    else:
        assert isinstance(refused.value.__cause__, cause)


def test_capture_seal_hashes_an_existing_selected_field():
    body = {"schema": "golden", "seal": "previous"}
    encoded = b'{"schema":"golden","seal":"previous"}'
    assert chain._seal(body, where="golden body", field="seal") == {
        "schema": "golden", "seal": hashlib.sha256(encoded).hexdigest()}


@pytest.mark.parametrize("stdout,returncode,detach", [
    ('noise\n[]\n{}\n{"action_key":"first"}\n{broken\n'
     '{"action_key":"last","note":"\\u00e9"}\n', 0,
     {"action_key": "last", "note": "\u00e9"}),
    ('{"action_key":null}\n', 0, {"action_key": None}),
    ("noise\n{}\n", 0, None),
    (None, 0, None),
    ('{"action_key":"accepted-but-nonzero"}\n', 3,
     {"action_key": "accepted-but-nonzero"}),
])
def test_capture_submission_record_precedes_native_refusal(
        tmp_path, monkeypatch, stdout, returncode, detach):
    completed = SimpleNamespace(stdout=stdout, stderr="error detail", returncode=returncode)
    calls = []
    written = []
    original_writer = dispatch._write_json

    def write(path, value):
        written.append(path)
        original_writer(path, value)

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return completed

    monkeypatch.setattr(dispatch, "_write_json", write)
    entry = {"name": "prep", "row": {"payload": "\u96ea", "value": float("nan")}}
    if returncode != 0 or detach is None:
        with pytest.raises(dispatch.ChainDispatchRefused) as refused:
            dispatch._submit_row(tmp_path, entry, run=run)
        assert str(refused.value) == f"row prep was not submitted (exit {returncode}): error detail"
        assert refused.value.__cause__ is None
    else:
        result = dispatch._submit_row(tmp_path, entry, run=run)
        assert result["detach"] == detach
    command = [sys.executable, str(dispatch.PBCAMPAIGN), "--detach",
               str(tmp_path / "manifests/prep.json")]
    assert calls == [(command, {"capture_output": True, "text": True})]
    record = {"name": "prep", "argv": command, "returncode": returncode,
              "stdout": stdout, "stderr": "error detail", "detach": detach}
    output = dispatch.submission_path(tmp_path, "prep")
    assert output.read_bytes() == (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    assert written == [tmp_path / "manifests/prep.json", output]
    assert (tmp_path / "manifests/prep.json").read_bytes() == (
        b'[\n  {\n    "payload": "\\u96ea",\n    "value": NaN\n  }\n]\n')


def test_capture_submission_uses_legacy_path_lookup_before_refusal(tmp_path, monkeypatch):
    redirected = tmp_path / "redirected/attempt.json"
    looked_up = []

    def locate(round_dir, name):
        looked_up.append((round_dir, name))
        return redirected

    monkeypatch.setattr(dispatch, "submission_path", locate)
    completed = SimpleNamespace(stdout="no detach", stderr="", returncode=0)
    with pytest.raises(dispatch.ChainDispatchRefused, match="row prep was not submitted"):
        dispatch._submit_row(tmp_path, {"name": "prep", "row": {}},
                             run=lambda *args, **kwargs: completed)
    assert looked_up == [(tmp_path, "prep")]
    assert json.loads(redirected.read_text())["stdout"] == "no detach"
    assert not (tmp_path / "submissions/prep.json").exists()


def test_legacy_capture_aliases_retain_function_identity():
    for old, primary in [
        ("fragment_path", "capture_fragment_path"),
        ("generation_directory", "capture_generation_directory"),
        ("owner_status_path", "capture_owner_status_path"),
        ("_seal", "_seal_capture_document"),
        ("_read_sealed", "_read_capture_document"),
        ("prepare", "prepare_capture_chain"),
    ]:
        assert getattr(chain, old) is getattr(chain, primary)
    for old, primary in [
        ("load_round", "load_capture_round"),
        ("require_ready", "require_capture_ready"),
        ("submit", "submit_capture_chain"),
    ]:
        assert getattr(dispatch, old) is getattr(dispatch, primary)


def test_split_help_does_not_import_package_dependencies(tmp_path):
    env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    script_name = split_dispatch.__file__
    assert script_name is not None
    script = Path(script_name).resolve()
    completed = subprocess.run([sys.executable, "-S", str(script), "--help"],
                               cwd=tmp_path, env=env, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "seal-forward" in completed.stdout


def test_capture_layout_public_contracts(tmp_path):
    prep = {"boundary_storage": {"directory": str(tmp_path / "windows")},
            "session": {"generation": "generation-\u00e9"}}
    assert chain.fragment_path(tmp_path, 1, 12) == tmp_path / "chain/capture-001-012.fragment.json"
    assert chain.generation_directory(prep) == tmp_path / "windows/generation-\u00e9"
    assert chain.owner_status_path(prep, 1, 12) == (
        tmp_path / "windows/generation-\u00e9/owners/capture-001-012.json")
