"""The joint dispatcher publishes the same JSON bytes through bound profiles.

The literals below come from the unedited tool. Patches of the template
builder, the record loader and the argv builder give the sites a fixed input
with Unicode, int keys, NaN, Infinity, nested lists and empty dicts.

CPU-only; every file is in ``tmp_path``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from prismaquant import streamed_boundary_artifacts as boundary  # noqa: E402
from prismaquant.digests import JsonProfile  # noqa: E402
from tools import dispatch_joint_quanta as dj  # noqa: E402

QID = "q-é"
ID = "pq-stageb-handoff-q-é-9348e70f923da519"
NAME = "q-é.handoff-template.82459d9ab7f73ec2.json"
EDGE = {"schema": "edge", "status": "edge-status", "text": "caf\u00e9 \u2603 \U0001f600 \"q\" \\ \n\t",
        "ints": {10: "ten", 2: "two", 1: "one"}, "mixed": {"b": 1, "a": [1, 2.5, [], {}, None, True], "": {}},
        "nan": float("nan"), "inf": float("inf"), "ninf": float("-inf"), "empty": {},
        "nested": [[1, [2]], {"z": {"y": []}}]}
ANSWER = {**EDGE, "status": "queued"}
ROW = {"argv": ["python3", "--quantum", "q-a.json"],
       "env": {"text": "caf\u00e9 \u2603", "nan": float("nan"), "ints": {10: "ten", 2: "two"},
               "nested": [[1, [2]], {"z": {"y": []}}], "empty": {}}}
REASON = "wait caf\u00e9"
PAYLOAD = b'{\n  "empty": {},\n  "inf": Infinity,\n  "ints": {\n    "1": "one",\n    "2": "two",\n    "10": "ten"\n  },\n  "mixed": {\n    "": {},\n    "a": [\n      1,\n      2.5,\n      [],\n      {},\n      null,\n      true\n    ],\n    "b": 1\n  },\n  "nan": NaN,\n  "nested": [\n    [\n      1,\n      [\n        2\n      ]\n    ],\n    {\n      "z": {\n        "y": []\n      }\n    }\n  ],\n  "ninf": -Infinity,\n  "schema": "edge",\n  "status": "edge-status",\n  "template_id": "pq-stageb-handoff-q-\\u00e9-9348e70f923da519",\n  "text": "caf\\u00e9 \\u2603 \\ud83d\\ude00 \\"q\\" \\\\ \\n\\t"\n}\n'
DRY = b'{\n  "band_serial_pending": [],\n  "consumer_tags": [\n    "gb10"\n  ],\n  "execution_plan": null,\n  "priority": 5,\n  "rows": [\n    {\n      "band_serial_readset": null,\n      "chain_layers": [],\n      "checkpoint_boundary": 2,\n      "cotangent_source": {\n        "mode": "chain"\n      },\n      "handoff_template": null,\n      "handoff_template_document": null,\n      "handoff_template_sha256": null,\n      "identity_sha256": "1111111111111111111111111111111111111111111111111111111111111111",\n      "kind": "quantum",\n      "link": {\n        "readers": 1\n      },\n      "manifest_sha256": "2222222222222222222222222222222222222222222222222222222222222222",\n      "pilot_admission": {\n        "fanout": 1,\n        "mode": "override",\n        "schema": "prismaquant.joint_dispatch_pilot.v2"\n      },\n      "progress_grace": null,\n      "quantum_id": "q-a",\n      "row": {\n        "argv": [\n          "python3",\n          "--quantum",\n          "q-a.json"\n        ],\n        "env": {\n          "empty": {},\n          "ints": {\n            "2": "two",\n            "10": "ten"\n          },\n          "nan": NaN,\n          "nested": [\n            [\n              1,\n              [\n                2\n              ]\n            ],\n            {\n              "z": {\n                "y": []\n              }\n            }\n          ],\n          "text": "caf\\u00e9 \\u2603"\n        }\n      },\n      "slice_sha256": "5555555555555555555555555555555555555555555555555555555555555555"\n    }\n  ],\n  "stage_a_pending": [\n    "q-b"\n  ],\n  "stage_a_terminal": false\n}\n'
LIVE = b'{"published": "q-a", "status": "queued", "submission": {"empty": {}, "inf": Infinity, "ints": {"1": "one", "2": "two", "10": "ten"}, "mixed": {"": {}, "a": [1, 2.5, [], {}, null, true], "b": 1}, "nan": NaN, "nested": [[1, [2]], {"z": {"y": []}}], "ninf": -Infinity, "schema": "edge", "status": "queued", "text": "caf\\u00e9 \\u2603 \\ud83d\\ude00 \\"q\\" \\\\ \\n\\t"}}\n{"band_serial_pending": "q-b", "reason": "wait caf\\u00e9"}\n'
REFUSALS = (
    ({"plan": {}}, "quantum 'q-é': the plan seals no boundary storage for a handoff", None),
    ({"adjoint_slice": {}}, "quantum 'q-é': no handoff template derives from the plan and slice: 'checkpoint'",
     "KeyError('checkpoint')"))


def _template(tmp_path, monkeypatch, **kwargs):
    built = []

    def build(**intent):
        built.append(intent["template_id"])
        return {**EDGE, "template_id": intent["template_id"]}

    monkeypatch.setattr(dj, "build_stage_a_produced_template", build)
    monkeypatch.setattr(boundary, "normalize_boundary_storage", lambda config: {
        "directory": "/d", "max_artifact_bytes": 7, "prefetch_batches": 2})
    slice_ = {"checkpoint": {"activation_entries": [{"tensor_bytes": 5}]}}
    plan = {"execution": {"boundary_storage": {}}}
    kwargs = {"plan": plan, "adjoint_slice": slice_, "tier": "t", "output_root": tmp_path, **kwargs}
    return built, dj.handoff_template({"quantum_id": QID, "layer": 3}, **kwargs)


class _Gateway(dj.FakeGateway):
    def submit(self, row, *, state_dir, afterok=()):
        return ANSWER


def _dispatch(tmp_path, monkeypatch, *, live):
    ids = ("q-a", "q-b")
    records = [(Path(f"{q}.json"), {
        "quantum_id": q, "layer": layer, "identity_sha256": "1" * 64, "read_set": {"manifest_sha256": "2" * 64},
        "adjoint": {"checkpoint_boundary": layer, "chain_layers": []},
        **({"executable_readset": {"manifest_sha256": "3" * 64}} if live else {})})
        for q, layer in zip(ids, (2, 1))]
    monkeypatch.setattr(dj, "load_records", lambda directory: records)
    monkeypatch.setattr(dj, "load_stage_a_proofs", lambda *args, **kwargs: None)
    monkeypatch.setattr(dj, "check_stage_a_proofs",
                        lambda *args, **kwargs: {q: "5" * 64 for q in (ids if live else ids[:1])})
    monkeypatch.setattr(dj, "quantum_argv", lambda record, **kwargs: ROW)
    monkeypatch.setattr(dj, "link_readers", lambda **kwargs: {"readers": 1})
    argv = ["--records", str(tmp_path), "--output-root", str(tmp_path / "out"), "--adjoint-receipt",
            str(tmp_path / "receipt.json"), "--timeout-s", "60", "--priority", "5", "--force-unverified-pilot"]
    if live:
        monkeypatch.setattr(dj, "_spec_capture_batch", lambda path: 1)
        monkeypatch.setattr(dj, "_coverage_row", lambda *args, **kwargs: {})
        monkeypatch.setattr(dj, "_append_state", lambda *args: None)
        monkeypatch.setattr(dj, "fresh_band_role", lambda record, **kwargs: (
            (None, None) if record["quantum_id"] == "q-a" else (None, REASON)))
        argv += ["--band-serial", "--handoff-tier", "t"]
    else:
        argv.append("--dry-run")
    return dj.main(argv, _gateway=_Gateway(), _coverage=lambda rows: [])


def _flows(tmp_path, monkeypatch, capsysbinary):
    built, (path, payload) = _template(tmp_path, monkeypatch)
    template_id = dj.handoff_template_id(QID, EDGE)
    assert _dispatch(tmp_path, monkeypatch, live=False) == 0
    dry = capsysbinary.readouterr().out
    monkeypatch.undo()
    assert _dispatch(tmp_path, monkeypatch, live=True) == 0
    live = capsysbinary.readouterr().out
    monkeypatch.undo()
    return built, path.name, payload, template_id, dry, live


def test_joint_dispatch_bytes(tmp_path, monkeypatch, capsysbinary):
    built, name, payload, template_id, dry, live = _flows(tmp_path, monkeypatch, capsysbinary)
    assert built == [dj.HANDOFF_TEMPLATE_ID_PREFIX + QID, ID]
    assert (name, payload, template_id, dry, live) == (NAME, PAYLOAD, ID, DRY, LIVE)


@pytest.mark.parametrize(("kwargs", "message", "cause"), REFUSALS)
def test_joint_dispatch_refusals(tmp_path, monkeypatch, kwargs, message, cause):
    with pytest.raises(dj.DispatchRefused) as info:
        _template(tmp_path, monkeypatch, **kwargs)
    actual = info.value.__cause__
    assert (str(info.value), None if actual is None else repr(actual)) == (message, cause)
    assert not list(tmp_path.rglob("*"))


def test_joint_dispatch_profile_calls(tmp_path, monkeypatch, capsysbinary):
    calls, real = [], JsonProfile.text

    def text(self, value):
        if isinstance(value, dict) and {"schema", "rows", "published", "band_serial_pending"} & set(value):
            calls.append((self.name, value))
        return real(self, value)

    with pytest.MonkeyPatch.context() as spy:
        spy.setattr(JsonProfile, "text", text)
        _flows(tmp_path, monkeypatch, capsysbinary)
    names = [name for name, _ in calls]
    assert names == ["direct-ascii-lax", "direct-ascii-indent2-lax", "direct-ascii-lax",
                     "direct-ascii-indent2-lax", "direct-ascii-spaced-lax", "direct-ascii-spaced-lax"]
    assert calls[0][1] == EDGE
    assert calls[1][1] == {**EDGE, "template_id": ID}
    assert calls[2][1] == EDGE
    assert calls[3][1] == {
        "consumer_tags": ["gb10"], "priority": 5, "execution_plan": None, "stage_a_terminal": False,
        "stage_a_pending": ["q-b"], "band_serial_pending": [],
        "rows": [{"kind": "quantum", "quantum_id": "q-a", "identity_sha256": "1" * 64,
                  "manifest_sha256": "2" * 64, "cotangent_source": {"mode": "chain"},
                  "handoff_template": None, "handoff_template_sha256": None,
                  "handoff_template_document": None, "band_serial_readset": None,
                  "link": {"readers": 1}, "progress_grace": None,
                  "pilot_admission": {"fanout": 1, "mode": "override",
                                      "schema": "prismaquant.joint_dispatch_pilot.v2"},
                  "slice_sha256": "5" * 64, "checkpoint_boundary": 2, "chain_layers": [],
                  "row": ROW}]}
    assert calls[4][1] == {"published": "q-a", "submission": ANSWER, "status": "queued"}
    assert calls[5][1] == {"band_serial_pending": "q-b", "reason": REASON}
