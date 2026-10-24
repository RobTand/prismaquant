"""Behavioral replay against independent, pinned original consumer SOURCE."""
import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests/fixtures/digest_1301/read-measurement/subset_reports_source.json"
SOURCES = json.loads(FIXTURE.read_text())


def original(module, file, name):
    namespace = dict(vars(module))
    exec(compile(SOURCES[file]["SOURCE"], str(FIXTURE), "exec"), namespace)
    return namespace[name]


def test_subset_reports_file_hash(tmp_path):
    module = importlib.import_module("tools.benchmarks.joint_validation_pair")
    before = original(module, "tools/benchmarks/joint_validation_pair.py", "sha")
    # Original SOURCE resolves stdlib hashlib, not the migrated owner.
    before.__globals__["hashlib"] = hashlib
    path = tmp_path / "payload"
    for payload in (b"", b"\x00\xff\r\n", bytes(range(256)) * 40000):
        path.write_bytes(payload)
        assert module.sha(path) == before(path)
    for missing in (tmp_path / "missing", tmp_path):
        for fn in (module.sha, before):
            with pytest.raises((FileNotFoundError, IsADirectoryError)):
                fn(missing)


@pytest.mark.parametrize("bad", [False, True])
def test_subset_reports_gate_writer(tmp_path, monkeypatch, capsys, bad):
    module = importlib.import_module("tools.compare_joint_layer_gate")
    monkeypatch.setattr(module, "_journal_identity", lambda path: {
        "units": {"model.layers.1.x"}, "identity_sha256": "a" * 64})
    verdict = {"layer": 1, "verdict": "match", "matched": 1,
               "units_shared": 1, "units_single": 1, "units_quantum": 1,
               "pending": [], "differed": [],
               "z": object() if bad else ["é\n", float("nan"), -0.0], "a": 1}
    monkeypatch.setattr(module, "compare_layer", lambda *a, **k: copy.deepcopy(verdict))
    before = original(module, "tools/compare_joint_layer_gate.py", "main")
    output = tmp_path / "gate.json"
    argv = ["--single-run", str(tmp_path / "one"), "--quantum", str(tmp_path / "two"),
            "--layers", "1", "--json", str(output)]
    results = []
    for fn in (before, module.main):
        if bad:
            with pytest.raises(TypeError):
                fn(argv)
            assert not output.exists()
        else:
            assert fn(argv) == 0
            results.append((output.read_bytes(), capsys.readouterr().out))
            output.unlink()
    if not bad:
        assert results[0] == results[1]
        assert results[1][0].endswith(b"\n") and not results[1][0].endswith(b"\n\n")
        assert b"\\u00e9" in results[1][0] and b"NaN" in results[1][0]


@pytest.mark.parametrize("bad", [False, True])
def test_subset_reports_stage_writer(tmp_path, monkeypatch, capsys, bad):
    module = importlib.import_module("tools.compare_stage_a_checkpoints")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)))
    value = {"boundary": "é", "equal": 1, "different": 0,
             "relative_l2": -0.0, "max_abs": float("inf"),
             "z": object() if bad else {"z": ["line\n"], "a": 2}}
    monkeypatch.setattr(module, "checkpoint_plane_distance",
                        lambda *a, **k: copy.deepcopy(value))
    published = []
    monkeypatch.setattr(module, "publish_new_bytes",
                        lambda path, data: published.append(data) or True)
    before = original(module, "tools/compare_stage_a_checkpoints.py", "main")
    argv = ["--reference", "r", "--reference-sha256", "a" * 64,
            "--candidate", "c", "--candidate-sha256", "b" * 64,
            "--output", str(tmp_path / "distance.json"), "--device", "cpu"]
    stdout = []
    for fn in (before, module.main):
        if bad:
            with pytest.raises(TypeError):
                fn(argv)
        else:
            assert fn(argv) == 0
            stdout.append(capsys.readouterr().out)
    if bad:
        assert not published
    else:
        assert published[0] == published[1]
        assert stdout[0] == stdout[1]
        assert published[1].endswith(b"\n")
        assert b"\\u00e9" in published[1] and b"Infinity" in published[1]
        # Publication refusal stays with the consumer, not the JSON owner.
        monkeypatch.setattr(module, "publish_new_bytes", lambda *a: False)
        with pytest.raises(SystemExit) as error:
            module.main(argv)
        assert error.value.code == 2
