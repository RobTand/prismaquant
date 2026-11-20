"""The Stage A recovery builder publishes the same JSON bytes through its bound profiles.

The literals below come from the unedited tool. A profile must not change
one byte of a file, of a printed line or of a digest.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from prismaquant.digests import JsonProfile
from tools import build_stagea_forward_recovery_package as tool

NAN, INF = float("nan"), float("inf")
CAPSULE_SHA = "c" * 64
HEAD = "9" * 40
MANIFEST = {"\u00e9": "\U0001F600", "nan": NAN, "inf": INF, "ninf": -INF, "empty": {}, "B": True,
            "nest": [[1, [2, {}]], {"b": [], "a": {"c": None}}], "10": 1, "2": 2,
            "read_plan": {"phases": [{"name": "head", "bytes": 5}, {"name": "forward-003", "bytes": 9}],
                          "read_bytes": 14}}
TEMPLATE = {"template_id": "tpl-\u00e9-r9", "nan": NAN, "inf": INF, "ninf": -INF, "empty": {},
            "nest": [1.5, [], {"b": 1, "a": {}}], "B": 2, "10": 3, "2": 4, "e": "\U0001F600"}
DROPPED = {"entries": 2, "bytes": 110, "ratio": NAN, "\u00e9": [{}, []]}
CAMPAIGN = {"artifact_budget_bytes": "4096", "campaign_bindings": {"plan_sha256": "a" * 64},
            "campaign_name": "CPU fixture", "campaign_scope": {"fixture": True, "\u00e9": NAN},
            "original_manifest_sha256": "b" * 64, "panel": "/fixture/panel", "records": "records",
            "run": "run", "scope_refusal": "campaign scope differs: "}
PRIOR = {"prefetch-override-r7.json": b'{"prefetch": 1}\n', "stage-a-spec-r7.json": b'{"spec": [1, 2]}\n',
         "stage-a-dispatch-config.json": b'{"dispatch": {}}\n'}


def _setup(root, mp, campaign=CAMPAIGN, *, validate_manifest=False):
    mp.chdir(root)
    Path("prior").mkdir()
    for name, raw in PRIOR.items():
        (Path("prior") / name).write_bytes(raw)
    (Path("prior") / "stagea-512-template-r7.json").write_text(json.dumps(TEMPLATE))
    Path("campaign.json").write_text(json.dumps(campaign))
    mp.setattr(sys, "argv", [
        "build", "--capsule", "capsule.json", "--capsule-sha256", CAPSULE_SHA,
        "--original-manifest", "original.gz", "--plan", "plan.json", "--plan-sha256", "d" * 64,
        "--campaign-fields", "campaign.json", "--r9-package", "prior", "--output", "out/pkg",
        "--source-head", HEAD, "--label", "r11"])
    Path("template.txt").write_bytes(b"template\n")
    mp.setattr(tool, "TEMPLATE", Path("template.txt").resolve())
    mp.setattr(tool, "_read", lambda path, sha: ({"frontier": 3}, None))
    mp.setattr(tool, "load_original", lambda path, campaign, plan: ({"original": True}, DROPPED))
    mp.setattr(tool, "recovery_manifest", lambda original, capsule, bound: MANIFEST)
    if not validate_manifest:
        mp.setattr("prismaquant.data_manifest_contract.check_manifest_v2", lambda manifest: None)
    mp.setattr("prismaquant.stage_a_produced_output.validate_declaration", lambda template: None)


def _spy(mp):
    calls = []
    real = JsonProfile.text

    def text(self, value):
        calls.append((self.name, value))
        return real(self, value)

    mp.setattr(JsonProfile, "text", text)
    return calls


def _canon(value):
    """Give a text that compares a whole value, NaN included."""
    return json.dumps(value, sort_keys=True)


STDOUT = b'{"capsule": {"path": "capsule.json", "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"}, "data_manifest_sha256": "c56e71884b7b4282d13740f84f29ce1bcd4f7daab8d9f8b07070f4c08fe6fb11", "frontier": 3, "head_walk_reads_dropped": {"bytes": 110, "entries": 2, "ratio": NaN, "\\u00e9": [{}, []]}, "max_phase_bytes": 9, "package": "out/pkg", "phases": 2, "runtime_pin_pending": true, "source_head": "9999999999999999999999999999999999999999"}\n'
MANIFEST_TEXT = b'{"10":1,"2":2,"B":true,"empty":{},"inf":Infinity,"nan":NaN,"nest":[[1,[2,{}]],{"a":{"c":null},"b":[]}],"ninf":-Infinity,"read_plan":{"phases":[{"bytes":5,"name":"head"},{"bytes":9,"name":"forward-003"}],"read_bytes":14},"\\u00e9":"\\ud83d\\ude00"}\n'
MANIFEST_SHA = "c56e71884b7b4282d13740f84f29ce1bcd4f7daab8d9f8b07070f4c08fe6fb11"
TEMPLATE_BYTES = b'{\n  "10": 3,\n  "2": 4,\n  "B": 2,\n  "e": "\\ud83d\\ude00",\n  "empty": {},\n  "inf": Infinity,\n  "nan": NaN,\n  "nest": [\n    1.5,\n    [],\n    {\n      "a": {},\n      "b": 1\n    }\n  ],\n  "ninf": -Infinity,\n  "template_id": "tpl-\\u00e9-r11"\n}\n'
FIELDS_BYTES = b'{\n  "campaign": {\n    "artifact_budget_bytes": "4096",\n    "campaign_bindings": {\n      "plan_sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n    },\n    "campaign_name": "CPU fixture",\n    "campaign_scope": {\n      "fixture": true,\n      "\\u00e9": NaN\n    },\n    "original_manifest_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",\n    "panel": "/fixture/panel",\n    "records": "records",\n    "run": "run",\n    "scope_refusal": "campaign scope differs: "\n  },\n  "label": "r11",\n  "source_head": "9999999999999999999999999999999999999999",\n  "manifest_sha256": "c56e71884b7b4282d13740f84f29ce1bcd4f7daab8d9f8b07070f4c08fe6fb11",\n  "capsule": {\n    "path": "capsule.json",\n    "sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"\n  },\n  "head_walk_reads_dropped": {\n    "entries": 2,\n    "bytes": 110,\n    "ratio": NaN,\n    "\\u00e9": [\n      {},\n      []\n    ]\n  },\n  "template_sha256": "90d69c78fb9ef942c8ce0a0d88a7455572ada06044bc1516473472664ddd013b"\n}\n'
PINS_BYTES = b'{\n  "launch-fields.json": "29842d38e9197717b8b620da2443ffa294f84830d31929d57dda46c3e9f5cf22",\n  "prefetch-override-r7.json": "2f0765642be5200e0d3fcfef9a6d25c908f62664316dedbe014f0f883b1ed7f0",\n  "stage-a-dispatch-config.json": "5c498c755b356e79838989c7634ba22642c2003a2d5061d8adc11eaf9d5ee0f1",\n  "stage-a-spec-r7.json": "b028d7bc6062218b428160fbb1dff8ea1d4b110e77b04ad8476dd7f54becb527",\n  "stagea-512-template-r7.json": "ba94f432325fc8f12945ccfabaf83a290f2e19a988a2e57631e1123c856db04a"\n}\n'
FILES = ["adjoint-recovery-manifest.json.gz", "launch-fields.json", "launch-r11.py", "package-pins.json",
         "prefetch-override-r7.json", "stage-a-dispatch-config.json", "stage-a-spec-r7.json",
         "stagea-512-template-r7.json"]


def test_stage_a_recovery_bytes_stay_equal(tmp_path, monkeypatch, capsysbinary):
    _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(tool, "render_launcher", lambda *a, **k: "launcher\n")
    tool.main()
    assert capsysbinary.readouterr().out == STDOUT
    pkg = Path("out/pkg")
    wire = (pkg / tool.MANIFEST_NAME).read_bytes()
    assert wire[4:8] == b"\0\0\0\0"
    assert gzip.decompress(wire) == MANIFEST_TEXT
    assert hashlib.sha256(wire).hexdigest() == MANIFEST_SHA
    assert (pkg / "stagea-512-template-r7.json").read_bytes() == TEMPLATE_BYTES
    assert (pkg / "launch-fields.json").read_bytes() == FIELDS_BYTES
    assert (pkg / "package-pins.json").read_bytes() == PINS_BYTES
    assert (pkg / "launch-r11.py").read_bytes() == b"launcher\n"
    assert sorted(p.name for p in pkg.iterdir()) == FILES
    umask = os.umask(0)
    os.umask(umask)
    assert {p.stat().st_mode & 0o777 for p in pkg.iterdir()} == {0o666 & ~umask}
    with pytest.raises(FileExistsError) as again:
        tool.main()
    assert str(again.value) == "[Errno 17] File exists: 'out/pkg'"
    assert again.value.__cause__ is None


def test_stage_a_recovery_refusal_leaves_no_package(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, {k: v for k, v in CAMPAIGN.items() if k != "run"})
    with pytest.raises(ValueError) as refusal:
        tool.main()
    assert str(refusal.value) == "launcher fields differ from the declared set: missing ['run'], unexpected []"
    assert refusal.value.__cause__ is None
    assert not Path("out").exists()


def test_recovery_manifest_v2_rejects_repeats_within_one_phase(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, validate_manifest=True)
    manifest = {
        "schema": "prismaquant.prismabuild.data_manifest.v2",
        "produced_by": {},
        "annotations": {},
        "mount_prefix": "/fixture",
        "entries": [{"path": "/fixture/input", "offset": 0, "bytes": 3,
                     "sha256": hashlib.sha256(b"abc").hexdigest()}],
        "entry_count": 1,
        "total_bytes": 3,
        "read_plan": {
            "phases": [
                {"name": "head", "entry_indices": [0], "bytes": 3, "cumulative_bytes": 3},
                {"name": "recover", "entry_indices": [0, 0], "bytes": 6, "cumulative_bytes": 9},
            ],
            "read_bytes": 9,
        },
    }
    monkeypatch.setattr(tool, "recovery_manifest", lambda original, capsule, bound: manifest)
    monkeypatch.setattr(tool, "render_launcher", lambda *a, **k: "launcher\n")
    with pytest.raises(RuntimeError, match="invalid or repeated phase entry index"):
        tool.main()
    assert not Path("out").exists()

    plan = manifest["read_plan"]
    plan["phases"][1].update(entry_indices=[0], bytes=3, cumulative_bytes=6)
    plan["read_bytes"] = 6
    tool.main()
    wire = (Path("out/pkg") / tool.MANIFEST_NAME).read_bytes()
    assert json.loads(gzip.decompress(wire)) == manifest


def test_stage_a_recovery_profile_calls(tmp_path, monkeypatch, capsysbinary):
    _setup(tmp_path, monkeypatch)
    calls = _spy(monkeypatch)
    monkeypatch.setattr(tool, "render_launcher", lambda *a, **k: "launcher\n")
    tool.main()
    out = capsysbinary.readouterr().out
    pkg = Path("out/pkg")
    pins = {name: hashlib.sha256((pkg / name).read_bytes()).hexdigest() for name in (
        *PRIOR, "stagea-512-template-r7.json", "launch-fields.json")}
    template = {**TEMPLATE, "template_id": "tpl-\u00e9-r11"}
    assert [(name, _canon(value)) for name, value in calls] == [
        ("direct-ascii-lax", _canon(MANIFEST)),
        ("direct-ascii-indent2-lax", _canon(template)),
        ("insertion-ascii-indent2-lax", _canon(json.loads(FIELDS_BYTES))),
        ("direct-ascii-indent2-lax", _canon(pins)),
        ("direct-ascii-spaced-lax", _canon(json.loads(out)))]
