"""Telemetry JSON lines: bytes from the unedited tools, and one profile call each."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from prismaquant.digests import JsonProfile  # noqa: E402
from tools import band_serial_handoff_live_pair as pair  # noqa: E402
from tools import staged_exact_read_bench as bench  # noqa: E402

EDGE = {"schema": "edge", "status": "edge-status", "text": "caf\u00e9 \u2603 \U0001f600 \"q\" \\ \n\t",
        "ints": {10: "ten", 2: "two", 1: "one"}, "mixed": {"b": 1, "a": [1, 2.5, [], {}, None, True], "": {}},
        "nan": float("nan"), "inf": float("inf"), "ninf": float("-inf"), "empty": {},
        "nested": [[1, [2]], {"z": {"y": []}}]}
BODY = (b'"empty": {}, "inf": Infinity, "ints": {"1": "one", "2": "two", "10": "ten"}, '
        b'"mixed": {"": {}, "a": [1, 2.5, [], {}, null, true], "b": 1}, "nan": NaN, '
        b'"nested": [[1, [2]], {"z": {"y": []}}], "ninf": -Infinity, ')
TEXT = b'"text": "caf\\u00e9 \\u2603 \\ud83d\\ude00 \\"q\\" \\\\ \\n\\t"'
DERIVE_BYTES = b'{' + BODY + b'"schema": "edge", "status": "edge-status", ' + TEXT + b'}\n'
CHILD_BYTES = (b'STAGED-EXACT-READ-BENCH {' + BODY + b'"pid": 4242, "schema": "edge", '
               b'"status": "edge-status", ' + TEXT + b', "tree": "/fake/tree"}\n')
DERIVE_SHA = "5739005435c3709b551a3d7e06c2352cdb55d7772677331194b95af33529cf77"
CHILD_SHA = "7496bb0f5c90260b16ba63add72b5bd369e8aa8ddda32dd591246eb27fc5af6c"
REFUSAL = (TypeError, "Object of type object is not JSON serializable", None)
NAME = "direct-ascii-spaced-lax"
DERIVE_ARGV = ["derive", "--root", "r", "--handoff", "h", "--handoff-sha256", "s"]


def _child_args():
    return SimpleNamespace(child="paths", slice="p", slice_sha256="s")


def test_telemetry_profile_calls(monkeypatch, capsysbinary):
    calls = []
    real = JsonProfile.text

    def spy(self, value):
        calls.append((self.name, value))
        return real(self, value)

    monkeypatch.setattr(JsonProfile, "text", spy)
    made = []

    def child(args, doc):
        made.append(dict(EDGE) if not made else {"x": object()})
        return made[-1]

    monkeypatch.setattr(pair, "derive", lambda root, handoff, sha: EDGE)
    monkeypatch.setitem(bench.CHILDREN, "paths", child)
    monkeypatch.setattr(bench, "load_slice", lambda path, sha: {})
    monkeypatch.setattr(bench, "os", SimpleNamespace(getpid=lambda: 4242))
    monkeypatch.setitem(sys.modules, "prismaquant", SimpleNamespace(__file__="/fake/tree/prismaquant/__init__.py"))

    assert pair.main(DERIVE_ARGV) == 0
    out = capsysbinary.readouterr().out
    assert (out, hashlib.sha256(out).hexdigest()) == (DERIVE_BYTES, DERIVE_SHA)
    assert len(calls) == 1 and calls[0][0] == NAME and calls[0][1] is EDGE

    calls.clear()
    assert bench.child_main(_child_args()) == 0
    out = capsysbinary.readouterr().out
    assert (out, hashlib.sha256(out).hexdigest()) == (CHILD_BYTES, CHILD_SHA)
    assert len(calls) == 1 and calls[0][0] == NAME and calls[0][1] is made[0]

    calls.clear()
    monkeypatch.setattr(pair, "derive", lambda root, handoff, sha: {"x": object()})
    with pytest.raises(TypeError) as err:
        pair.main(DERIVE_ARGV)
    assert (type(err.value), str(err.value), err.value.__cause__) == REFUSAL
    assert capsysbinary.readouterr().out == b""
    assert [name for name, _ in calls] == [NAME]

    calls.clear()
    with pytest.raises(TypeError) as err:
        bench.child_main(_child_args())
    assert (type(err.value), str(err.value), err.value.__cause__) == REFUSAL
    assert capsysbinary.readouterr().out == b""
    assert [name for name, _ in calls] == [NAME]
