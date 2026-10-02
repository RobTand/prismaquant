"""Exact old outcomes and owner routing for preparation JSON/SHA sites."""
from __future__ import annotations

import hashlib
import json
import pickle
from types import SimpleNamespace

import pytest

from prismaquant import cost_stage_checkpoint as journal, digests
import tools.build_tessera_selected_cache as selected
from tests.golden_table import GoldenTable


@pytest.fixture(scope="module")
def golden():
    return GoldenTable("digest_prepare_io_1301")


@pytest.mark.parametrize("raw", [b"", b"a\r\n\x00\xff", "café\u2028".encode()])
def test_selected_bound_old_bytes_and_refusal(tmp_path, raw, golden):
    path = tmp_path / "input-é"
    path.write_bytes(raw)
    golden.call(lambda: selected._bound(str(path), hashlib.sha256(raw).hexdigest(), "input-é"))
    golden.call(lambda: selected._bound(str(path), "0" * 64, "input-é"))
    golden.call(lambda: selected._bound(str(tmp_path / "missing"), "0" * 64, "input-é"), tmp=tmp_path)


@pytest.mark.parametrize("pins", [
    {"é": [1, -0.0, True, None]}, {10: "ten", 9: "nine"},
    {"value": float("nan")}, {"value": "\ud800"},
    {1: "int", "1": "str"}, {"bad": {1, 2}},
])
def test_migration_pin_dedup_old_outcomes(pins, golden):
    first = {"proof_bundle_sha256": "a" * 64, "old_pins": pins,
             "new_pins": {"z": 1, "a": 2}, "run_id": "a", "keep": "first"}
    second = dict(first, run_id="b", keep="second")
    golden.call(lambda: journal.merge_identity_migrations({"b": [second], "a": [first]}))


@pytest.mark.parametrize("identity", [
    {"é": [1, -0.0, True, None], "emoji": "😀"},
    {10: "ten", 9: "nine"}, {"nonfinite": float("nan")},
    {"surrogate": "\ud800"},
])
def test_journal_manifest_old_bytes_and_refusals(tmp_path, identity, golden):
    def write():
        root, seal, states = journal.prepare_journal(tmp_path / "journal",
            stage="prépare", resume=False, identity=identity, qnames=["unit-é", "unit.z"])
        return seal, (root / "manifest.json").read_bytes(), states
    golden.call(write, tmp=tmp_path)


def _selected_run(tmp_path, monkeypatch, capsys, *, rooted=False, value=None):
    wire = tmp_path / "wire"
    wire.mkdir()
    handoff = {"provenance": {"wire_dir": str(wire),
        "tessera_joint_anchors": {"inputs": {"source": "same"}}}}
    raw = pickle.dumps(handoff)
    path = tmp_path / "handoff.pkl"
    path.write_bytes(raw)
    manifest = {"units": {"unit-é": {}}, "source": {"label": "café"},
                "geometry": [1, -0.0, True, None], "value": value}
    monkeypatch.setattr(selected, "_bind_selected_assignment", lambda *_: ({}, {}, {}))
    monkeypatch.setattr(selected, "load_measured_anchor_input", lambda *_a, **_k: object())
    monkeypatch.setattr(selected, "selected_cached_units_manifest", lambda *_a, **_k: manifest)
    monkeypatch.setattr(selected, "read_cached_unit_bundle", lambda *_a, **_k:
        SimpleNamespace(encoder_source_proof_mode="fixture", warnings=["é"]))
    out = wire / "selected.json"
    argv = ["--handoff", str(path), "--handoff-sha256", hashlib.sha256(raw).hexdigest(),
            "--assignment", "assignment.json", "--assignment-sha256", "b" * 64,
            "--out", str(out)]
    paths = tmp_path / "read-paths.json"
    if rooted:
        import prismaquant.joint_catalog_extension as catalog
        monkeypatch.setattr(catalog, "selected_cache_read_paths", lambda _m: ["/inputs/z-é", "/inputs/a"])
        for name in ("catalog-extension", "producer-packages"):
            binding = tmp_path / (name + ".json")
            binding.write_bytes(b"{}")
            argv += ["--" + name, str(binding), "--" + name + "-sha256", hashlib.sha256(b"{}").hexdigest()]
        argv += ["--read-paths-out", str(paths)]
    code = selected.main(argv)
    stdout = capsys.readouterr().out
    # This control pickle names the test's temporary wire root. Verify its
    # actual acquired-byte digest independently, then mask only that varying
    # input field in the golden output. Manifest identity remains unmasked.
    assert json.loads(stdout)["handoff_sha256"] == hashlib.sha256(raw).hexdigest()
    stdout = stdout.replace(hashlib.sha256(raw).hexdigest(), "<handoff-sha256>")
    return code, out.read_bytes(), paths.read_bytes() if rooted else None, stdout


@pytest.mark.parametrize("rooted", [False, True])
@pytest.mark.parametrize("value", [None, "\ud800", float("nan")])
def test_selected_publication_old_bytes_and_refusals(tmp_path, monkeypatch, capsys, golden,
                                                   rooted, value):
    golden.call(lambda: _selected_run(tmp_path, monkeypatch, capsys, rooted=rooted, value=value),
                tmp=tmp_path)


def test_bound_routes_owned_bytes_to_sha_owner(tmp_path, monkeypatch):
    path = tmp_path / "input"
    raw = b"exact\r\n\x00\xff"
    path.write_bytes(raw)
    calls = []
    monkeypatch.setattr(selected, "bytes_sha256hex", lambda data: calls.append(data) or digests.bytes_sha256hex(data))
    assert selected._bound(str(path), hashlib.sha256(raw).hexdigest(), "fixture") == raw
    assert calls == [raw]


def test_selected_publication_routes_distinct_profiles(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(selected, "indent2_json_file_bytes",
        lambda value: calls.append(("strict-file", value)) or digests.indent2_json_file_bytes(value))
    for alias in ("DIRECT_ASCII_INDENT2_LAX", "DIRECT_ASCII_SPACED_LAX"):
        profile = getattr(digests, alias)
        monkeypatch.setattr(selected, alias, SimpleNamespace(text=
            lambda value, p=profile: calls.append((p.name, value)) or p.text(value)))
    _selected_run(tmp_path, monkeypatch, capsys, rooted=True)
    assert [name for name, _ in calls] == ["strict-file", "direct-ascii-indent2-lax", "direct-ascii-spaced-lax"]


def test_journal_manifest_routes_utf8_without_trailing_lf(tmp_path, monkeypatch):
    profile = digests.DIRECT_UTF8_INDENT2_STRICT
    calls = []
    monkeypatch.setattr(journal, "DIRECT_UTF8_INDENT2_STRICT", SimpleNamespace(encoded=
        lambda value: calls.append(value) or profile.encoded(value)))
    root, _, _ = journal.prepare_journal(tmp_path / "journal", stage="é", resume=False,
                                        identity={"label": "é"}, qnames=["unit-é"])
    assert len(calls) == 1
    raw = (root / "manifest.json").read_bytes()
    assert "é".encode() in raw and not raw.endswith(b"\n")


def test_migration_pin_keys_route_spaced_lax_profile(monkeypatch):
    calls = []
    monkeypatch.setattr(journal, "DIRECT_ASCII_SPACED_LAX", SimpleNamespace(text=
        lambda value: calls.append(value) or digests.DIRECT_ASCII_SPACED_LAX.text(value)))
    row = {"proof_bundle_sha256": "a" * 64, "old_pins": {"é": 1}, "new_pins": None}
    assert journal.merge_identity_migrations({"row": [row]}) == [row]
    assert calls == [row["old_pins"], None]
