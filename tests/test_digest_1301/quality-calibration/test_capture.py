"""CPU differential tests of actual consumer blocks, not GPU qualification.

Fixtures are verbatim source slices at the ledger commit. Only heavyweight
intake/publication/time dependencies are controlled; serializers, tensor bytes,
and the consumer's sequencing and error propagation execute unchanged.
"""
import ast
import contextlib
import hashlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from prismaquant import digests

ROOT = Path(__file__).resolve().parents[3]
FIX = ROOT / "tests/fixtures/digest_1301/quality-calibration/capture"


def original(name):
    return textwrap.dedent((FIX / (name + ".py")).read_text())


def actual(module, function, start=None):
    source = (ROOT / "tools" / (module + ".py")).read_text()
    tree = ast.parse(source)
    # First lexical match selects the legacy publish rather than original capture.
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == function)
    if start is None:
        return textwrap.dedent("\n".join(source.splitlines()[node.lineno-1:node.end_lineno]))
    lines = source.splitlines()[node.lineno-1:node.end_lineno]
    begin = next(i for i, line in enumerate(lines) if line.strip().startswith(start))
    # replay block ends before finally; other slices run through the function end.
    end = next((i for i in range(begin, len(lines)) if lines[i].strip() == "finally:"), len(lines))
    return textwrap.dedent("\n".join(lines[begin:end]))


def namespace():
    return dict(json=json, hashlib=hashlib, torch=torch, io=io, Path=Path,
                bytes_sha256hex=digests.bytes_sha256hex,
                DIRECT_ASCII_SPACED_STRICT=digests.DIRECT_ASCII_SPACED_STRICT,
                DIRECT_ASCII_SPACED_LAX=digests.DIRECT_ASCII_SPACED_LAX)


def run_block(code, values, *, returns=False):
    ns = namespace()
    ns.update(values)
    output = io.StringIO()
    if returns:
        code = "def block():\n" + textwrap.indent(code, "    ")
    with contextlib.redirect_stdout(output):
        exec(code, ns)
        result = ns["block"]() if returns else None
    return ns, result, output.getvalue()


def publisher(events, refuse=None):
    def publish(path, raw):
        events.append((str(path), raw))
        return len(events) != refuse
    return publish


def test_routed_publish():
    codes = [original("routed_publish"), actual("capture_glm_routed_layers", "publish")]
    for refuse in (None, 1):
        results = []
        for code in codes:
            events, entries = [], []
            values = dict(root=Path("capture"), entries=entries, calibration={}, manifest={}, source={},
                          started=1, time=SimpleNamespace(monotonic=lambda: 3),
                          publish_new_bytes=publisher(events, refuse),
                          routed_boundary_inputs=lambda *a, **k: events.append("intake"),
                          report=lambda *a, **k: events.append("report"))
            # Refusal is indexed among publication events, not intake.
            if refuse:
                values["publish_new_bytes"] = lambda p, r: (events.append((str(p), r)) or False)
            ns, _, _ = run_block(code, values)
            out = io.StringIO()
            try:
                with contextlib.redirect_stdout(out):
                    ns["publish"](7, {"metadata": {"unit": "héllo"}, "tensors": {"z": torch.arange(4)}})
                error = None
            except ValueError as exc:
                error = str(exc)
            results.append((events, entries, out.getvalue(), error))
        assert results[0] == results[1]
        events, entries, stdout, error = results[1]
        assert events[0] == "intake"
        if refuse:
            assert not entries and not stdout and "already exists" in error
        else:
            assert entries[0]["sha256"] == hashlib.sha256(events[1][1]).hexdigest()
            assert entries[0]["bytes"] == len(events[1][1])
            assert stdout.startswith('{"captured": 7, "count": 1, "elapsed_s": ')
            assert events[-1] == "report"


def test_routed_receipt():
    codes = [original("routed_receipt"), actual("capture_glm_routed_layers", "main", "raw = (DIRECT_")]
    for value in ("héllo", float("nan")):
        for refuse in (None, 1):
            outcomes = []
            for code in codes:
                events = []
                try:
                    _, result, stdout = run_block(code, dict(receipt={"z": value, "a": -0.0},
                        root=Path("capture"), entries=[1], publish_new_bytes=publisher(events, refuse)), returns=True)
                    error = None
                except ValueError as exc:
                    result, stdout, error = None, "", str(exc)
                outcomes.append((events, result, stdout, error))
            assert outcomes[0] == outcomes[1]
            if isinstance(value, str) and not refuse:
                assert outcomes[1][0][0][1] == b'{"a": -0.0, "z": "h\\u00e9llo"}\n'
                assert outcomes[1][2].startswith('{"status": "complete", "captures": 1, "receipt": ')
            else:
                assert outcomes[1][3]


def test_replay_publish():
    codes = [original("replay_publish"), actual("capture_glm_routing_replay", "main", "buffer = io.BytesIO()")]
    for refuse in (None, 1, 2):
        outcomes = []
        for code in codes:
            events = []
            try:
                _, _, stdout = run_block(code, dict(root=Path("replay"),
                    result={"metadata": {"z": "héllo", "a": float("nan")}}, spec_binding={"z": 1, "a": 2},
                    bf16_reduction_stamp=lambda: {}, publish_new_bytes=publisher(events, refuse)))
                error = None
            except ValueError as exc:
                stdout, error = "", str(exc)
            outcomes.append((events, stdout, error))
        assert outcomes[0] == outcomes[1]
        events, stdout, error = outcomes[1]
        if not refuse:
            receipt = events[1][1]
            assert receipt.endswith(b"\n") and b"h\\u00e9llo" in receipt and b"NaN" in receipt
            assert json.loads(receipt)["boundary"]["sha256"] == hashlib.sha256(events[0][1]).hexdigest()
            assert stdout.startswith('{"boundary": ')
        else:
            assert len(events) == refuse and error and not stdout


# These children have no site-packages or repository import path. Fail even an
# attempted package import, rather than relying on whether torch is installed.
_ISOLATED_STARTUP = """
import hashlib
import importlib.abc
import importlib.util
import runpy
import sys
from pathlib import Path

class NoRuntimePackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'prismaquant', 'torch', 'tools'}:
            raise AssertionError('startup attempted package import: ' + fullname)

sys.meta_path.insert(0, NoRuntimePackages())
path = Path(sys.argv[1])
mode = sys.argv[2]
before = list(sys.path)
if mode == 'import':
    spec = importlib.util.spec_from_file_location('isolated_capture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    owner = sys.modules['pq_profile_shared_digest_owner']
    assert module.bytes_sha256hex is owner.bytes_sha256hex
    raw = b'capture\\x00bytes\\xff'
    assert module.bytes_sha256hex(raw) == hashlib.sha256(raw).hexdigest()
    if hasattr(module, '_hash'):
        module._tensor_bytes = lambda tensor: raw
        assert module._hash(object()) == hashlib.sha256(raw).hexdigest()
else:
    sys.argv = [str(path), '--help']
    try:
        runpy.run_path(str(path), run_name='__main__')
    except SystemExit as exc:
        assert exc.code == 0, exc.code
    else:
        raise AssertionError('--help did not exit')
assert sys.path == before
assert not any(name.split('.')[0] in {'prismaquant', 'torch', 'tools'}
               for name in sys.modules)
print('stdlib startup verified')
"""


@pytest.mark.parametrize("module", ["exl3_grouped_down_replay", "glm_mtp_capture"])
def test_stdlib_file_spec_import(module, tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", _ISOLATED_STARTUP,
         str(ROOT / "tools" / (module + ".py")), "import"],
        cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "stdlib startup verified" in result.stdout


def test_mtp_stdlib_help(tmp_path):
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", _ISOLATED_STARTUP,
         str(ROOT / "tools/glm_mtp_capture.py"), "help"],
        cwd=tmp_path, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--phase {final-hidden,projection,capture}" in result.stdout
    assert "stdlib startup verified" in result.stdout


def test_tensor_hash():
    module = importlib.import_module("tools.exl3_grouped_down_replay")
    ns = namespace()
    exec(original("tensor_bytes") + "\n" + original("tensor_hash"), ns)
    for tensor in (torch.arange(12).reshape(3, 4).t(),
                   torch.tensor([0., -0., float("nan")], dtype=torch.bfloat16),
                   torch.empty(0), torch.tensor(3), torch.arange(8)[::2]):
        assert module._tensor_bytes(tensor) == ns["_tensor_bytes"](tensor)
        assert module._hash(tensor) == ns["_hash"](tensor)
        assert module._hash(tensor) == hashlib.sha256(module._tensor_bytes(tensor)).hexdigest()


def test_result_json_gap():
    module = importlib.import_module("tools.exl3_grouped_down_replay")
    ns = namespace()
    exec(original("result_json"), ns)
    record = {"z": "héllo", "a": -0.0}
    assert module.result_json(record) == ns["result_json"](record)
    assert module.result_json(record) == '{\n  "a": -0.0,\n  "z": "h\\u00e9llo"\n}'
    assert module.result_json(record) != digests.DIRECT_UTF8_INDENT2_STRICT.text(record)
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            module.result_json({"z": value})
        with pytest.raises(ValueError):
            ns["result_json"]({"z": value})


def test_projection_publish(tmp_path, monkeypatch):
    codes = [original("projection_publish"), actual("glm_mtp_capture", "projection_phase", "raw = (json.dumps")]
    for value in ("héllo", float("nan")):
        for refuse in (None, 1):
            outcomes = []
            for code in codes:
                events = []
                def publish(path, raw, **kwargs):
                    events.append((str(path), raw, kwargs))
                    return len(events) != refuse
                monkeypatch.setattr("prismaquant.cost_stage_checkpoint.publish_new_bytes", publish)
                try:
                    _, result, stdout = run_block(code, dict(carried={"z": value, "stacks": {"z": [1], "a": [2, 3]},
                        "request": {}, "tool": {}}, path=tmp_path / "projection.json",
                        args=SimpleNamespace(plan="plan"), plan_sha256="plan-digest",
                        time=SimpleNamespace(monotonic=lambda: 3), started=1), returns=True)
                    error = None
                except (ValueError, RuntimeError) as exc:
                    result, stdout, error = None, "", str(exc)
                outcomes.append((events, result, stdout, error))
            assert outcomes[0] == outcomes[1]
            events, result, _, error = outcomes[1]
            if isinstance(value, str) and not refuse:
                assert events[0][1].endswith(b"\n") and b"h\\u00e9llo" in events[0][1]
                assert result["projection"]["sha256"] == hashlib.sha256(events[0][1]).hexdigest()
                assert result["stacks"] == ["a", "z"] and result["units"] == 3
            else:
                assert error
                if not isinstance(value, str):
                    assert not events


def test_mtp_main_gap(tmp_path):
    module = importlib.import_module("tools.glm_mtp_capture")
    for value in ("héllo", float("nan")):
        outcomes = []
        for code in (original("mtp_main"), actual("glm_mtp_capture", "main")):
            report = tmp_path / "projection-run.json"
            if report.exists():
                report.unlink()
            summary = {"z": value, "a": -0.0}
            args = SimpleNamespace(phase="projection", out=str(tmp_path))
            values = dict(_capture_arguments=lambda argv: args, sys=SimpleNamespace(argv=["tool"]),
                          final_hidden_phase=lambda a: summary, projection_phase=lambda a: summary,
                          capture_phase=lambda a: summary)
            ns, _, _ = run_block(code, values)
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    result = ns["main"]([])
                error = None
            except ValueError as exc:
                result, error = None, str(exc)
            outcomes.append((report.read_bytes() if report.exists() else None, output.getvalue(), result, error))
        assert outcomes[0] == outcomes[1]
        if isinstance(value, str):
            raw, stdout, result, _ = outcomes[1]
            assert raw == b'{\n  "a": -0.0,\n  "z": "h\\u00e9llo"\n}\n'
            assert stdout == raw.decode() + "\n" and result == 0
        else:
            assert outcomes[1][0] is None and outcomes[1][3]
