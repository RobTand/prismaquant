"""The stage report tools publish the same JSON bytes through bound profiles.

The literals come from the unedited tools. Stubs replace the heavy inputs of
each ``main``, so that the bytes do not depend on paths or times. CPU-only.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from prismaquant import io_spans, tessera_joint_aura as aura
from prismaquant.digests import JsonProfile
from tools import dispatch_stage_a_split as split
from tools import profile_stage_b_head as head
from tools import reselect_mtp_fixed as reselect

EDGE = {"text": "caf\u00e9 \u2603 \U0001f600", "ints": {10: "ten", 2: "two"}, "nan": float("nan"),
        "inf": float("inf"), "mixed": {"b": [1, 2.5, [], {}], "a": {}}, "nested": [[1, [2]], {"z": {}}]}
STAGE_A = (b'{\n  "inf": Infinity,\n  "ints": {\n    "2": "two",\n    "10": "ten"\n  },\n  "mismatches": 0,\n'
           b'  "mixed": {\n    "a": {},\n    "b": [\n      1,\n      2.5,\n      [],\n      {}\n    ]\n  },\n'
           b'  "nan": NaN,\n  "nested": [\n    [\n      1,\n      [\n        2\n      ]\n    ],\n'
           b'    {\n      "z": {}\n    }\n  ],\n  "text": "caf\\u00e9 \\u2603 \\ud83d\\ude00"\n}\n')
RESELECT = (b'{"E": NaN, "body_sha256": "b1425e8e14afbf5c4ab02bfcafae5fe88eb79dca7389127417f9820481bf56e7", '
            b'"mtp_cost_sha256": "cost-\\u00e9\\u2603", "output": "selected.json", "resident_bytes": 7, '
            b'"rung": "caf\\u00e9 \\u2603", "selected_mtp_wires": 1, "sha256": '
            b'"07bc94012cecce13e8e2abfdaa4cd76718bf7384ad87cc1c47385f01b8d7e471", '
            b'"whole_artifact_budget_bytes": 2, "whole_artifact_payload_bytes": 2, '
            b'"whole_artifact_reserve_bytes": 50}\n')
RESELECT_SHA = "07bc94012cecce13e8e2abfdaa4cd76718bf7384ad87cc1c47385f01b8d7e471"
HEAD = (b'STAGE_B_HEAD_PROFILE {"cpu_s": 0.5, "edge": {"inf": Infinity, "ints": {"2": "two", "10": "ten"}, '
        b'"mixed": {"a": {}, "b": [1, 2.5, [], {}]}, "nan": NaN, "nested": [[1, [2]], {"z": {}}], '
        b'"text": "caf\\u00e9 \\u2603 \\ud83d\\ude00"}, "host": "h\\u00f6st", "layer": 3, '
        b'"loadavg_after": [1.0, 0.5, 0.25], "loadavg_before": [0.5, 0.25, 0.125], '
        b'"local_journal_written": {"bytes": 0, "files": 0}, "marks": {"slice_s": 1}, "mode": "slice", '
        b'"nfs_ops_mnt_shared": {"READ": 5, "_bytes": 4}, "nfs_ops_total": 5, "peak_rss_kib": 5, '
        b'"proc_io": {"read": 15}, "quantum_sha256": '
        b'"35876f8ae64dee705748500171ce65c651a74c0d009c03317411a8769c0202b4", '
        b'"schema": "prismaquant.stage_b_head_profile.v1", "units": 2, "wall_s": 2.5}\n')
BAD_TYPE = "Object of type object is not JSON serializable"
# The second value mixes int and str keys. The sort of the keys refuses it.
BAD = ((object(), BAD_TYPE), ({1: "a", "b": 2}, "'<' not supported between instances of 'str' and 'int'"))


RESULT = {"marks": {"slice_s": 1}, "units": 2, "edge": EDGE}


@pytest.fixture
def spy(monkeypatch):
    calls = []
    real = JsonProfile.text

    def text(self, value):
        calls.append((self.name, value))
        return real(self, value)

    monkeypatch.setattr(JsonProfile, "text", text)
    return calls


def _stage_a(monkeypatch, report):
    monkeypatch.setattr(split, "load_round", lambda round_dir: {})
    monkeypatch.setattr(split, "compare_digests", lambda document, baseline: report)
    return split.main(["compare", "--round-dir", "r", "--baseline", "b"])


def test_stage_a_report_bytes_and_refusal(monkeypatch, capsysbinary):
    assert _stage_a(monkeypatch, {**EDGE, "mismatches": 0}) == 0
    assert capsysbinary.readouterr().out == STAGE_A
    assert _stage_a(monkeypatch, {**EDGE, "mismatches": 3}) == 4
    assert json.loads(capsysbinary.readouterr().out)["mismatches"] == 3
    for value, text in BAD:
        with pytest.raises(TypeError) as bad:
            _stage_a(monkeypatch, {"mismatches": 0, "bad": value})
        assert str(bad.value) == text and bad.value.__cause__ is None
        assert capsysbinary.readouterr().out == b""


def _reselect(tmp_path, monkeypatch, rung="caf\u00e9 \u2603"):
    monkeypatch.chdir(tmp_path)
    body = json.dumps({"__prismaquant__": {
        "target_profile": "tp", "assignment_payload_bits_total": 8.0,
        "tessera_serving_scope": {"target": "T", "by_unit": {}},
        "serving_lane_provenance": {"by_unit": {}, "activation_pricing_branches": {}}}}).encode()
    for name, text in (("body.json", body.decode()), ("fixed.json", "{}"), ("constants.json", "{}")):
        (tmp_path / name).write_text(text)
    record = {"assignment": {}, "resident_bytes": 7, "rung": rung, "E": EDGE["nan"]}
    stubs = {
        "load_mtp_cost": lambda path, expected_sha256: SimpleNamespace(
            payload={"groups": {}, "costs": {}}, sha256="cost-\u00e9\u2603"),
        "detect_profile": lambda model: "profile",
        "load_serving_profile": lambda name: SimpleNamespace(target_platform="sm121"),
        "serving_target_from_args": lambda args, target_platform: object(),
        "_mtp_scope_inputs": lambda *a, **k: ({}, {}),
        "_mtp_rung_attestation": lambda *a: None,
        "select_mtp_rungs": lambda *a, **k: dict(record),
        "allocation_rung_allowability": lambda args: None,
        "whole_artifact_budget_from_assignment_payload": lambda *a, **k: {
            "selection_tensor_payload_bytes": 100},
        "recursive_regular_file_bytes": lambda path: 150,
        "mtp_selection_rebased_bytes": lambda *a, **k: {"mtp_source_bytes": 1, "total_bytes": 2},
        "scope_provenance": lambda target, contexts: {"target": "T", "by_unit": {}},
        "selection_serving_lane_provenance": lambda *a, **k: {"by_unit": {}},
        "whole_artifact_budget_stamp": lambda **k: {},
        "backfill_mtp_selection_wires": lambda result, cost: {
            "__prismaquant__": {"mtp_selection": {"mtp_expert_wires": {"w": 1}}}, "t": "caf\u00e9"},
    }
    for name, stub in stubs.items():
        monkeypatch.setattr(reselect, name, stub)
    argv = ["reselect_mtp_fixed.py", "--body-layer-config", "body.json", "--expect-body-sha256",
            hashlib.sha256(body).hexdigest(), "--mtp-joint-cost", "cost.pkl", "--expect-cost-sha256", "0",
            "--mtp-byte-budget", "1", "--whole-artifact-budget-bytes", "2", "--observed-body-export", "e",
            "--new-metadata-allowance-bytes", "0", "--mtp-serve-constants", "constants.json",
            "--fixed-formats", "fixed.json", "--model", "m", "--target-profile", "tp",
            "--output", "selected.json", "--tessera-platform", "sm121", "--tessera-runtime-image", "i@sha256:" + "a" * 64,
            "--tessera-execution-mode", "eager", "--tessera-residency", "resident"]
    monkeypatch.setattr("sys.argv", argv)
    return tmp_path / "selected.json"


def test_reselect_summary_bytes_and_refusal(tmp_path, monkeypatch, capsysbinary):
    output = _reselect(tmp_path, monkeypatch)
    assert reselect.main() == 0
    assert capsysbinary.readouterr().out == RESELECT
    assert hashlib.sha256(output.read_bytes()).hexdigest() == RESELECT_SHA
    other = tmp_path / "other"
    other.mkdir()
    output = _reselect(other, monkeypatch, rung=object())
    with pytest.raises(TypeError) as bad:
        reselect.main()
    assert str(bad.value) == BAD_TYPE and bad.value.__cause__ is None
    assert hashlib.sha256(output.read_bytes()).hexdigest() == RESELECT_SHA
    assert capsysbinary.readouterr().out == b""


def _head(tmp_path, monkeypatch, result):
    quantum = tmp_path / "q.json"
    raw = json.dumps({"layer": 3, "campaign": {"prepared_path": "p", "prepared_sha256": "s"}}).encode()
    quantum.write_bytes(raw)

    def seq(*values):
        items = iter(values)
        return lambda *a, **k: next(items)

    monkeypatch.setattr(io_spans, "read_proc_io", seq({"read": 10}, {"read": 25}))
    monkeypatch.setattr(io_spans, "read_proc_status", lambda: {"VmHWM": 5120})
    monkeypatch.setattr(aura, "load_joint_anchor_plan", lambda *a, **k: {})
    monkeypatch.setattr(head, "mount_ops", seq({"_bytes": 5, "READ": 2}, {"_bytes": 9, "READ": 7}))
    monkeypatch.setattr(head, "slice_intake", lambda *a, **k: result)
    monkeypatch.setattr(head, "time", SimpleNamespace(monotonic=seq(1.0, 3.5), process_time=seq(0.5, 1.0)))
    monkeypatch.setattr(head.os, "getloadavg", seq((0.5, 0.25, 0.125), (1.0, 0.5, 0.25)))
    monkeypatch.setattr(head.os, "uname", lambda: SimpleNamespace(nodename="h\u00f6st"))
    return ["--mode", "slice", "--quantum", str(quantum), "--quantum-sha256",
                 hashlib.sha256(raw).hexdigest(), "--plan", "plan.json", "--plan-sha256", "p" * 8,
                 "--scratch", str(tmp_path / "scratch")]


def test_head_profile_line_bytes_and_refusal(tmp_path, monkeypatch, capsysbinary):
    argv = _head(tmp_path, monkeypatch, RESULT)
    prints = []
    monkeypatch.setattr(head, "print", lambda *a, **k: prints.append(k) or print(*a, **k), raising=False)
    assert head.main(argv) == 0
    assert capsysbinary.readouterr().out == HEAD
    assert prints == [{"flush": True}]
    for value, text in BAD:
        argv = _head(tmp_path, monkeypatch, {"bad": value})
        with pytest.raises(TypeError) as bad:
            head.main(argv)
        assert str(bad.value) == text and bad.value.__cause__ is None
        assert capsysbinary.readouterr().out == b""


def test_reselect_publication_refusal(tmp_path, monkeypatch, capsysbinary):
    output = _reselect(tmp_path, monkeypatch)
    select = reselect.select_mtp_rungs

    def race(*args, **kwargs):
        output.write_bytes(b"other")
        return select(*args, **kwargs)

    monkeypatch.setattr(reselect, "select_mtp_rungs", race)
    with pytest.raises(SystemExit) as refused:
        reselect.main()
    assert refused.value.code == 2 and refused.value.__cause__ is None
    captured = capsysbinary.readouterr()
    assert captured.out == b""
    assert captured.err.endswith(b"error: output already exists; refusing overwrite\n")
    assert output.read_bytes() == b"other"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "body.json", "constants.json", "fixed.json", "selected.json"]


def test_stage_reports_profile_calls(tmp_path, monkeypatch, spy):
    report = {**EDGE, "mismatches": 0}
    _stage_a(monkeypatch, report)
    other = tmp_path / "r"
    other.mkdir()
    _reselect(other, monkeypatch)
    reselect.main()
    third = tmp_path / "h"
    third.mkdir()
    head.main(_head(third, monkeypatch, RESULT))
    summary = {**json.loads(RESELECT), "E": EDGE["nan"]}
    profile = {**json.loads(HEAD.split(b" ", 1)[1]), "edge": EDGE,
               "loadavg_before": (0.5, 0.25, 0.125), "loadavg_after": (1.0, 0.5, 0.25)}
    assert spy == [("direct-ascii-indent2-lax", report), ("direct-ascii-spaced-lax", summary),
                   ("direct-ascii-spaced-lax", profile)]
