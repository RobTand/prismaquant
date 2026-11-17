"""The Stage A package tools publish the same JSON bytes through their bound profiles.

The literals below come from the unedited tools. A profile must not change
one byte of a file, of a printed line or of a digest.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import pytest

from prismaquant.digests import JsonProfile
from tools import build_stagea_seed_package as seed
from tools import build_stagea_split_package as split
from tools import compare_stage_a_checkpoints as compare

NAN, INF = float("nan"), float("inf")
NEST = [[1, [2, {}]], {"b": [], "a": {"c": None}}]
SPEC = {"through": 2, "checkpoint": {"path": "k.json"}, "compare": None, "e": "\u00e9\U0001F600", "n": NAN, "i": INF, "m": -INF,
        "z": {}, "x": [1.5, [], {"b": 1, "a": {}}], "B": 2, "10": 3}
DROPPED = {"entries": 2, "bytes": NAN, "u": "\u00e9"}
ANNOTATIONS = {"nan": NAN, "inf": INF, "ninf": -INF, "empty": {}, "nest": NEST, "10": 1, "2": 2,
               "B": True, "\u00e9": "\U0001F600"}
SEED_MANIFEST = {
    "annotations": ANNOTATIONS, "entry_count": 1, "total_bytes": NAN,
    "entries": [{"path": "/\u00e9", "offset": 0, "bytes": 3, "sha256": None}],
    "read_plan": {"phases": [{"name": "head", "bytes": 3, "entry_indices": [0]},
                             {"name": "chain-\u00e9", "bytes": 4, "entry_indices": []}],
                  "read_bytes": 7}}
SOURCE = b'{"entries": []}'
SOURCE_SHA = hashlib.sha256(SOURCE).hexdigest()
RESULT = {"boundary": 2, "equal": 3, "different": NAN, "relative_l2": INF, "max_abs": -INF,
          "probes": [[1.5, []], {"b": 1, "a": {}}], "note": "\u00e9\U0001F600", "B": 1, "10": 2,
          "empty": {}}
COMPARE_ARGV = ["--reference", "r.json", "--reference-sha256", "a" * 64, "--candidate", "c.json",
                "--candidate-sha256", "b" * 64, "--output", "out.json", "--device", "cpu"]
SPLIT_ENTRY = {"path": "/x/\u00e9.safetensors", "offset": 0, "bytes": 5, "sha256": None}
SPLIT_PREP = {
    "annotations": {**ANNOTATIONS, "chain_split": {"boundaries": [4, 2]}}, "entry_count": 1,
    "entries": [SPLIT_ENTRY],
    "read_plan": {"phases": [{"name": "head", "bytes": 5, "entry_indices": [0]}], "read_bytes": 5}}
SPLIT_QUANTUM = {
    "annotations": {"nan": NAN, "\u00e9": [{}]}, "entry_count": 1, "entries": [SPLIT_ENTRY],
    "read_plan": {"phases": [{"name": "chain-003", "bytes": 5, "entry_indices": [0]}],
                  "read_bytes": 5}}
SPLIT_LOADED = {"boundary": 4, "state_entry": {"path": "/run/chain-state.json"},
                "state": {"chain_state_sha256": "e" * 64, "n_batches": 3},
                "sealed": {4: {"cotangent_sha256": "f" * 64}}}


def _canon(value):
    """Give a text that compares a whole value, NaN included."""
    return json.dumps(value, sort_keys=True)


def _spy(mp):
    calls = []
    real = JsonProfile.text

    def text(self, value):
        calls.append((self.name, value))
        return real(self, value)

    mp.setattr(JsonProfile, "text", text)
    return calls


def _seed_setup(root, mp):
    mp.chdir(root)
    Path("source.json").write_bytes(SOURCE)
    entries = []

    def derive(original, *, spec, spec_entry, checkpoint, compare):
        entries.append(spec_entry)
        return SEED_MANIFEST

    mp.setattr(seed, "drop_source_head_walk_reads", lambda original, plan: (original, DROPPED))
    mp.setattr(seed, "normalize_seed_spec", lambda raw: SPEC)
    mp.setattr(seed, "load_pinned_checkpoint", lambda pin, what: {})
    mp.setattr(seed, "seed_manifest", derive)
    return entries


def _seed_build(sha=SOURCE_SHA, output="package", validate=None):
    return seed.build(
        original_manifest="source.json", original_manifest_sha256=sha,
        plan={"path": "plan.json", "sha256": "0" * 64}, checkpoint={}, capsule={}, through=2,
        implementation_from=None, implementation_to=None, compare=None, output=output,
        validate=validate)


def _split_setup(root, mp):
    mp.chdir(root)
    mp.setattr(split, "_load_original",
               lambda manifest, sha, plan: ({"entries": [SPLIT_ENTRY]}, DROPPED))
    mp.setattr(split, "load_round", lambda *args: SPLIT_LOADED)
    mp.setattr(split, "round_manifests",
               lambda original, loaded, through, ranges: (SPLIT_PREP, {(0, 2): SPLIT_QUANTUM}))
    mp.setattr(split, "stat_check", lambda manifest, paths: {"checked": 1, "missing": 0})


def _split_build(sha=SOURCE_SHA, output="package", validate=None):
    return split.build(
        original_manifest="source.json", original_manifest_sha256=sha,
        plan={"path": "plan.json", "sha256": "0" * 64}, output_root="run",
        chain_state_sha256="1" * 64, checkpoint_sha256="2" * 64, through=2, ranges=[(0, 2)],
        output=output, validate=validate)


def _compare_setup(root, mp):
    mp.chdir(root)
    mp.setattr(compare, "checkpoint_plane_distance",
               lambda reference, candidate, device, read_ahead: dict(RESULT))


SEED_ENTRY = {"path": "package/seed-spec.json", "offset": 0, "bytes": 261,
              "sha256": "fdc3db6d1c47e07059dd6c5c77618cfd8b9f69738fff26266d68be8695b9de22"}
SEED_SPEC_BYTES = b'{\n  "10": 3,\n  "B": 2,\n  "checkpoint": {\n    "path": "k.json"\n  },\n  "compare": null,\n  "e": "\\u00e9\\ud83d\\ude00",\n  "i": Infinity,\n  "m": -Infinity,\n  "n": NaN,\n  "through": 2,\n  "x": [\n    1.5,\n    [],\n    {\n      "a": {},\n      "b": 1\n    }\n  ],\n  "z": {}\n}\n'
SEED_PACKAGE_BYTES = b'{\n  "data_manifest": {\n    "path": "package/seed-manifest.json.gz",\n    "sha256": "b61a9d8bc6a80cbe1bca82e588d7ea1341b78162016b5b5af3546e7137657995"\n  },\n  "head_walk_reads_dropped": {\n    "bytes": NaN,\n    "entries": 2,\n    "u": "\\u00e9"\n  },\n  "peak_consecutive_phase_bytes": 7,\n  "phases": [\n    {\n      "bytes": 3,\n      "entries": 1,\n      "name": "head"\n    },\n    {\n      "bytes": 4,\n      "entries": 0,\n      "name": "chain-\\u00e9"\n    }\n  ],\n  "read_bytes": 7,\n  "schema": "prismaquant.stage_a.seed_package.v1",\n  "seed_spec": {\n    "path": "package/seed-spec.json",\n    "sha256": "fdc3db6d1c47e07059dd6c5c77618cfd8b9f69738fff26266d68be8695b9de22"\n  },\n  "source_manifest": {\n    "path": "source.json",\n    "sha256": "e3eb12da99e044ecc7d50cea407bf17f33c546e5309aa7ee661234baed2b7750"\n  }\n}\n'
SEED_WIRE_PLAIN = b'{"annotations":{"10":1,"2":2,"B":true,"empty":{},"inf":Infinity,"nan":NaN,"nest":[[1,[2,{}]],{"a":{"c":null},"b":[]}],"ninf":-Infinity,"\\u00e9":"\\ud83d\\ude00"},"entries":[{"bytes":3,"offset":0,"path":"/\\u00e9","sha256":null}],"entry_count":1,"read_plan":{"phases":[{"bytes":3,"entry_indices":[0],"name":"head"},{"bytes":4,"entry_indices":[],"name":"chain-\\u00e9"}],"read_bytes":7},"total_bytes":NaN}\n'
SEED_WIRE_HEADER = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\x03"
SEED_WIRE_SHA = "b61a9d8bc6a80cbe1bca82e588d7ea1341b78162016b5b5af3546e7137657995"
COMPARE_FILE_BYTES = b'{\n  "10": 2,\n  "B": 1,\n  "boundary": 2,\n  "device": "cpu",\n  "different": NaN,\n  "empty": {},\n  "equal": 3,\n  "max_abs": -Infinity,\n  "note": "\\u00e9\\ud83d\\ude00",\n  "probes": [\n    [\n      1.5,\n      []\n    ],\n    {\n      "a": {},\n      "b": 1\n    }\n  ],\n  "relative_l2": Infinity\n}\n'
COMPARE_OUT_BYTES = b'{"boundary": 2, "different": NaN, "equal": 3, "max_abs": -Infinity, "relative_l2": Infinity}\n'


def test_seed_package_bytes_stay_equal(tmp_path, monkeypatch):
    entries = _seed_setup(tmp_path, monkeypatch)
    _seed_build()
    assert entries == [SEED_ENTRY]
    assert Path("package", seed.SPEC_NAME).read_bytes() == SEED_SPEC_BYTES
    assert Path("package", seed.PACKAGE_NAME).read_bytes() == SEED_PACKAGE_BYTES
    wire = Path("package", seed.MANIFEST_NAME).read_bytes()
    assert wire[:10] == SEED_WIRE_HEADER
    assert gzip.decompress(wire) == SEED_WIRE_PLAIN
    assert hashlib.sha256(wire).hexdigest() == SEED_WIRE_SHA
    assert seed.manifest_wire(SEED_MANIFEST) == wire
    for path in Path("package").iterdir():
        assert path.stat().st_mode == 0o100664


def test_seed_refusals_leave_no_file(tmp_path, monkeypatch):
    _seed_setup(tmp_path, monkeypatch)
    with pytest.raises(seed.SeedPackageRefused) as info:
        _seed_build(sha="0" * 64)
    assert str(info.value) == "the source manifest does not have the pinned digest"
    assert info.value.__cause__ is None
    assert not Path("package").exists()

    def refuse(manifest):
        raise RuntimeError("no")

    with pytest.raises(RuntimeError) as info:
        _seed_build(validate=refuse)
    assert str(info.value) == "no"
    assert info.value.__cause__ is None
    assert not Path("package").exists()


def test_compare_record_bytes_stay_equal(tmp_path, monkeypatch, capsysbinary):
    _compare_setup(tmp_path, monkeypatch)
    capsysbinary.readouterr()
    assert compare.main(COMPARE_ARGV) == 0
    assert capsysbinary.readouterr().out == COMPARE_OUT_BYTES
    assert Path("out.json").read_bytes() == COMPARE_FILE_BYTES
    assert Path("out.json").stat().st_mode == 0o100664


def test_compare_record_is_never_replaced(tmp_path, monkeypatch, capsysbinary):
    _compare_setup(tmp_path, monkeypatch)
    assert compare.main(COMPARE_ARGV) == 0
    capsysbinary.readouterr()
    with pytest.raises(SystemExit) as info:
        compare.main(COMPARE_ARGV)
    assert info.value.code == 2
    assert info.value.__cause__ is None
    captured = capsysbinary.readouterr()
    assert captured.out == b""
    assert captured.err.endswith(b"out.json already exists; a distance record is never replaced\n")
    assert Path("out.json").read_bytes() == COMPARE_FILE_BYTES
    assert Path("out.json").stat().st_mode == 0o100664
    assert sorted(path.name for path in Path(".").iterdir()) == ["out.json"]


def test_record_seed_and_compare(tmp_path, monkeypatch, capsysbinary):
    (tmp_path / "a").mkdir()
    entries = _seed_setup(tmp_path / "a", monkeypatch)
    _seed_build()
    assert entries == [SEED_ENTRY]
    assert Path("package", seed.SPEC_NAME).read_bytes() == SEED_SPEC_BYTES
    assert Path("package", seed.PACKAGE_NAME).read_bytes() == SEED_PACKAGE_BYTES
    wire = Path("package", seed.MANIFEST_NAME).read_bytes()
    assert hashlib.sha256(wire).hexdigest() == SEED_WIRE_SHA
    (tmp_path / "b").mkdir()
    _compare_setup(tmp_path / "b", monkeypatch)
    capsysbinary.readouterr()
    assert compare.main(COMPARE_ARGV) == 0
    assert capsysbinary.readouterr().out == COMPARE_OUT_BYTES
    assert Path("out.json").read_bytes() == COMPARE_FILE_BYTES


SPLIT_PACKAGE_BYTES = b'{\n  "boundaries": [\n    4,\n    2\n  ],\n  "chain_state": {\n    "chain_state_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",\n    "path": "/run/chain-state.json",\n    "sha256": "1111111111111111111111111111111111111111111111111111111111111111"\n  },\n  "checkpoint": {\n    "boundary": 4,\n    "cotangent_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",\n    "sha256": "2222222222222222222222222222222222222222222222222222222222222222"\n  },\n  "from": 4,\n  "head_walk_reads_dropped": {\n    "bytes": NaN,\n    "entries": 2,\n    "u": "\\u00e9"\n  },\n  "n_batches": 3,\n  "prep": {\n    "data_manifest": {\n      "path": "package/prep-manifest.json.gz",\n      "sha256": "ca1bbfa471b002e0b6fd8b4aaf3b26df71537032ffa5b6959f5f4b058833cafe"\n    },\n    "entry_count": 1,\n    "peak_consecutive_phase_bytes": 5,\n    "phases": [\n      {\n        "bytes": 5,\n        "entries": 1,\n        "name": "head"\n      }\n    ],\n    "read_bytes": 5,\n    "source_bytes": 0,\n    "stat_check": {\n      "checked": 1,\n      "missing": 0\n    }\n  },\n  "quanta": [\n    {\n      "data_manifest": {\n        "path": "package/quantum-samples-000000-000002-manifest.json.gz",\n        "sha256": "696495a274a20e95049788d1cf3f6303274e30c2fa9114faa1fa738af652647f"\n      },\n      "entry_count": 1,\n      "peak_consecutive_phase_bytes": 5,\n      "phases": [\n        {\n          "bytes": 5,\n          "entries": 1,\n          "name": "chain-003"\n        }\n      ],\n      "read_bytes": 5,\n      "samples": [\n        0,\n        2\n      ],\n      "source_bytes": 5,\n      "stat_check": {\n        "checked": 1,\n        "missing": 0\n      }\n    }\n  ],\n  "ranges": [\n    [\n      0,\n      2\n    ]\n  ],\n  "round_source_bytes": 5,\n  "schema": "prismaquant.stage_a.split_package.v1",\n  "source_manifest": {\n    "path": "source.json",\n    "sha256": "e3eb12da99e044ecc7d50cea407bf17f33c546e5309aa7ee661234baed2b7750"\n  },\n  "through": 2\n}\n'


def test_record_split(tmp_path, monkeypatch, capsysbinary):
    _split_setup(tmp_path, monkeypatch)
    _split_build()
    assert Path("package", split.PACKAGE_NAME).read_bytes() == SPLIT_PACKAGE_BYTES
    expected = [
        ("prep-manifest.json.gz", "ca1bbfa471b002e0b6fd8b4aaf3b26df71537032ffa5b6959f5f4b058833cafe",
         "1229dea666a6624f78b5f959563ae9f59463bde7a89af163364669e321d6be30"),
        ("quantum-samples-000000-000002-manifest.json.gz",
         "696495a274a20e95049788d1cf3f6303274e30c2fa9114faa1fa738af652647f",
         "52a1ea721420e59c07a044c77d2c1c3f01c14158a458b73223b9f9e127f09067"),
    ]
    for name, wire_sha, plain_sha in expected:
        raw = Path("package", name).read_bytes()
        assert raw[:10] == SEED_WIRE_HEADER
        assert hashlib.sha256(raw).hexdigest() == wire_sha
        plain = gzip.decompress(raw)
        assert plain.endswith(b"\n") and not plain.endswith(b"\n\n")
        assert hashlib.sha256(plain).hexdigest() == plain_sha
    for path in Path("package").iterdir():
        assert path.stat().st_mode == 0o100664


def test_split_refusal_leaves_no_file(tmp_path, monkeypatch):
    _split_setup(tmp_path, monkeypatch)

    def refuse(manifest):
        raise RuntimeError("no")

    with pytest.raises(RuntimeError) as info:
        _split_build(validate=refuse)
    assert type(info.value) is RuntimeError
    assert str(info.value) == "no"
    assert info.value.__cause__ is None
    assert list(tmp_path.iterdir()) == []


def test_stage_a_packages_profile_calls(tmp_path, monkeypatch):
    calls = _spy(monkeypatch)
    (tmp_path / "a").mkdir()
    _seed_setup(tmp_path / "a", monkeypatch)
    _seed_build()
    assert [name for name, _ in calls] == [
        "direct-ascii-indent2-lax", "direct-ascii-lax", "direct-ascii-indent2-lax"]
    assert calls[0][1] is SPEC
    assert calls[1][1] is SEED_MANIFEST
    assert _canon(calls[2][1]) == _canon(json.loads(Path("package", seed.PACKAGE_NAME).read_bytes()))
    assert calls[2][1]["head_walk_reads_dropped"] is DROPPED
    (tmp_path / "b").mkdir()
    _split_setup(tmp_path / "b", monkeypatch)
    calls.clear()
    _split_build()
    assert [name for name, _ in calls] == [
        "direct-ascii-lax", "direct-ascii-lax", "direct-ascii-indent2-lax"]
    assert calls[0][1] is SPLIT_PREP
    assert calls[1][1] is SPLIT_QUANTUM
    assert _canon(calls[2][1]) == _canon(json.loads(Path("package", split.PACKAGE_NAME).read_bytes()))
    (tmp_path / "c").mkdir()
    _compare_setup(tmp_path / "c", monkeypatch)
    calls.clear()
    assert compare.main(COMPARE_ARGV) == 0
    assert [name for name, _ in calls] == ["direct-ascii-indent2-lax", "direct-ascii-spaced-lax"]
    assert _canon(calls[0][1]) == _canon({**RESULT, "device": "cpu"})
    assert _canon(calls[0][1]) == _canon(json.loads(Path("out.json").read_bytes()))
