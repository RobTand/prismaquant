"""Allocator publications against independent, verbatim research-base source.

No historical Git objects or replacement-owner-derived expectations are needed.
Only the real publication AST is isolated from main's model/probe setup.
"""
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from prismaquant import allocator

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests/fixtures/digest_allocator_pareto_1301.py"
PUBLICATIONS = ROOT / "tests/fixtures/digest_1301/allocator-pareto/original_publications.json"


def _tree():
    return ast.parse((ROOT / "prismaquant/allocator.py").read_text())


def _block():
    main = next(n for n in _tree().body if isinstance(n, ast.FunctionDef) and n.name == "main")
    return ast.Module(body=[next(n for n in main.body if isinstance(n, ast.If)
        and ast.unparse(n.test) == "args.pareto_output_dir"
        and any(isinstance(c, ast.Name) and c.id == "seen_payloads" for c in ast.walk(n)))],
        type_ignores=[])


def _records():
    base = dict(target_bits=4.25, achieved_bits=4., bits_total=40., bits_total_with_aux=41.,
        predicted_dloss=.125, total_predicted_dloss_with_aux=.25, variable_predicted_dloss=.1,
        aux_fixed_predicted_dloss=.025, fixed_predicted_dloss=.125, format_counts={"NVFP4": 1},
        artifact_bytes=128, artifact_tensor_payload_bytes=96, whole_artifact_upper_bound_bytes=160)
    return [{**base, "assignment": {"z": "FP8", "é": "NVFP4"}},
        {**base, "target_bits": 4.5, "assignment": {"é": "NVFP4", "z": "FP8"}},
        {**base, "assignment": {"z": "NVFP4", "é": "FP8"}}]


def _exercise(block, directory, records, *, budget=False):
    events, feeds = [], []
    real_sha = hashlib.sha256
    real_write = Path.write_text

    def sha(data=b"", **kwargs):
        feed = bytearray(data)
        feeds.append(feed)
        events.append(("hash", bytes(data)))
        delegate = real_sha(data, **kwargs)

        class Digest:
            def update(self, raw):
                feed.extend(raw)
                delegate.update(raw)

            def hexdigest(self):
                return delegate.hexdigest()

            def digest(self):
                return delegate.digest()
        return Digest()

    def write(path, text, *args, **kwargs):
        events.append(("write", path.name, text.replace(str(directory), "<OUTPUT>")))
        return real_write(path, text, *args, **kwargs)

    ns = dict(vars(allocator))
    # The original fixture owns its own raw hashlib/json recipes.
    ns.update(hashlib=hashlib, args=SimpleNamespace(pareto_output_dir=str(directory),
        target_disk_gb=1e-6 if budget else None, artifact_overhead_reserve_bytes=64,
        exclude_source_prefix=[], probe="probe-input", costs="cost-input"),
        pareto_seed_records=copy.deepcopy(records), target_profile="fixture",
        measured_runtime_table=None, specs_sorted=[SimpleNamespace(name="NVFP4")],
        targets=[4.25, 4.5], knee_summary={"enabled": False})
    stdout = io.StringIO()
    error = None
    with patch.object(hashlib, "sha256", sha), patch.object(Path, "write_text", write), contextlib.redirect_stdout(stdout):
        try:
            exec(compile(block, "<allocator-publication>", "exec"), ns)
        except Exception as exc:
            error = (type(exc), str(exc))
    files = {p.name: p.read_bytes().replace(str(directory).encode(), b"<OUTPUT>")
             for p in sorted(directory.iterdir())}
    return files, stdout.getvalue().replace(str(directory), "<OUTPUT>"), list(map(bytes, feeds)), events, error


def test_pareto_publication_matches_original(tmp_path):
    """Full files, hash feeds, labels, dedup, output and operation order."""
    old = ast.parse(FIXTURE.read_text())
    records = _records()
    pristine = copy.deepcopy(records)
    for budget in (False, True):
        expected = _exercise(old, tmp_path / f"old-{budget}", records, budget=budget)
        actual = _exercise(_block(), tmp_path / f"new-{budget}", records, budget=budget)
        assert actual == expected
        assert actual[-1] is None
        manifest = json.loads(actual[0]["manifest.json"])
        assert len(manifest["candidates"]) == 2
        assert len(actual[2]) >= 3
        assert actual[2].count(b'{"z":"FP8","\\u00e9":"NVFP4"}') == 2
        assert all(len(c["label"].rsplit("_", 1)[1]) == 12 for c in manifest["candidates"])
        assert actual[3][-1][0:2] == ("write", "manifest.json")
        assert actual[1].endswith("(2 unique)\n")
    assert records == pristine


def test_pareto_refusals_match_original(tmp_path):
    old = ast.parse(FIXTURE.read_text())
    cycle = []; cycle.append(cycle)
    for index, bad in enumerate(({1: "NVFP4", "z": "FP8"}, {"z": object()}, {"z": cycle})):
        records = [{**_records()[0], "assignment": bad}]
        expected = _exercise(old, tmp_path / f"old-{index}", records)
        actual = _exercise(_block(), tmp_path / f"new-{index}", records)
        assert actual == expected
        assert actual[-1] is not None
        assert actual[0] == {} and actual[1] == ""


def test_pareto_ascii_lax_values_match_original(tmp_path):
    records = [{**_records()[0], "assignment": {"é": "\ud800", "x": float("inf")}}]
    expected = _exercise(ast.parse(FIXTURE.read_text()), tmp_path / "old", records)
    assert _exercise(_block(), tmp_path / "new", records) == expected
    assert expected[-1] is None
    assert b'\\ud800' in expected[2][0] and b'Infinity' in expected[2][0]


def _publication_statements():
    tree = _tree()
    result = []
    for call in ast.walk(tree):
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "DIRECT_ASCII_INDENT2_LAX" and call.func.attr == "text"):
            stmt = min((n for n in ast.walk(tree) if isinstance(n, ast.stmt)
                and n.lineno <= call.lineno <= n.end_lineno), key=lambda n: n.end_lineno - n.lineno)
            result.append(stmt)
    return sorted(result, key=lambda n: n.lineno)


def test_all_sorted_publication_statements_match_original(tmp_path):
    """Execute each real writer statement, not just its profile in isolation."""
    frozen = json.loads(PUBLICATIONS.read_text())["statements"]
    live = _publication_statements()
    assert len(live) == len(frozen) == 7
    cycle = []; cycle.append(cycle)
    values = [{"z": [float("nan"), float("inf"), -0.0], "é": "\ud800"},
              {1: "int", "z": "mixed"}, {"z": object()}, {"z": cycle}]
    for index, (row, statement) in enumerate(zip(frozen, live)):
        for case, value in enumerate(values):
            def exercise(block, directory):
                directory.mkdir(exist_ok=True)
                ns = dict(vars(allocator))
                ns.update(args=SimpleNamespace(probe="p", costs="c", target_disk_gb=None,
                    artifact_overhead_reserve_bytes=None), out_dir=directory,
                    path=directory / "assignment.json", sel_path=directory / "selection.json",
                    knee_path=directory / "knees.json", applicability_report_path=directory / "applicability.json",
                    applicability_payload=value, knee_summary=value, payload=value, selection=value,
                    manifest_rows=[value], target_profile="fixture", targets=[4.25],
                    specs_sorted=[SimpleNamespace(name="NVFP4")])
                error = None
                try:
                    exec(compile(block, "<allocator-writer>", "exec"), ns)
                except Exception as exc:
                    error = (type(exc), str(exc))
                return {p.name: p.read_bytes() for p in directory.iterdir()}, error
            expected = exercise(ast.parse(row["source"]), tmp_path / f"old-{index}-{case}")
            actual = exercise(ast.Module(body=[statement], type_ignores=[]), tmp_path / f"new-{index}-{case}")
            assert actual == expected
            if expected[1] is None:
                assert all(raw.endswith(b"\n") for raw in actual[0].values())


def test_pareto_uses_exact_owner():
    block = _block()
    raw = [n for n in ast.walk(block) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
        and n.func.value.id in {"json", "hashlib"}]
    assert not raw
    assert "DIRECT_ASCII_LAX.sha256(assignment)[:12]" in ast.unparse(block)
