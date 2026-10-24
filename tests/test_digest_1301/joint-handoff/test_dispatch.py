"""Frozen dispatcher recipes exercised against their original byte-level outcomes."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from prismaquant import digests
from tools import dispatch_joint_quanta as joint
from tools import dispatch_stage_a_split as split

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/fixtures/digest_1301/joint-handoff"


def _expression(file, scope, index, original=False):
    path = FIXTURES / (Path(file).stem + ".original.txt") if original else ROOT / file
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == scope)
    calls = sorted((n for n in ast.walk(function) if isinstance(n, ast.Call)
                    and (ast.unparse(n.func) == "json.dumps" or
                         (isinstance(n.func, ast.Attribute) and n.func.attr == "text" and ast.unparse(n.func).startswith("DIRECT_ASCII_")))),
                   key=lambda n: (n.lineno, n.col_offset))
    return compile(ast.Expression(calls[index]), str(path), "eval")


def _compare(scope, index, context, file="tools/dispatch_joint_quanta.py"):
    namespace = {"json": json, **vars(digests), **context}
    before = eval(_expression(file, scope, index, True), namespace)
    after = eval(_expression(file, scope, index), namespace)
    assert after.encode("utf-8") == before.encode("utf-8")
    return after


def test_quantum_argv_grace_recipe():
    values = {"z": "雪", "a": [True, -0.0, None]}
    assert _compare("quantum_argv", 0, {"grace_stamps": values}) == '{"a":[true,-0.0,null],"z":"\\u96ea"}'
    for bad in (float("nan"), float("inf")):
        for original in (True, False):
            with pytest.raises(ValueError):
                eval(_expression("tools/dispatch_joint_quanta.py", "quantum_argv", 0, original),
                     {"json": json, **vars(digests), "grace_stamps": bad})


def test_handoff_template_identity():
    template = {"template_id": "pilot-雪", "note": "é"}
    body = {"note": "é"}
    text = _compare("handoff_template_id", 0, {"body": body})
    assert joint.handoff_template_id("q", template) == joint.HANDOFF_TEMPLATE_ID_PREFIX + "q-" + hashlib.sha256(text.encode()).hexdigest()[:16]


def test_handoff_template_written_bytes(tmp_path, monkeypatch):
    # SDK generation is outside this slice: retain the real dispatcher policy,
    # derivation, hashing and publication, with a deterministic SDK builder seam.
    import prismaquant.stage_a_produced_output as produced
    monkeypatch.setattr(produced, "build_boundary_template", lambda **kwargs: kwargs)
    record = {"quantum_id": "q", "layer": 2}
    storage = {"schema": "prismaquant.aura.boundary_storage.v1", "directory": "unused",
               "max_resident_bytes": 4096, "max_auxiliary_bytes": 4096,
               "max_artifact_bytes": 4096, "prefetch_batches": 1}
    kwargs = {"plan": {"execution": {"boundary_storage": storage}},
              "adjoint_slice": {"checkpoint": {"activation_entries": [{"tensor_bytes": 16}]}},
              "tier": "fixture-stage", "output_root": tmp_path}
    tree = ast.parse((FIXTURES / "dispatch_joint_quanta.original.txt").read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("handoff_template_id", "handoff_template")]
    namespace = dict(vars(joint))
    exec(compile(ast.Module(body=functions, type_ignores=[]), "original", "exec"), namespace)
    expected = namespace["handoff_template"](record, **kwargs)
    path, payload = joint.handoff_template(record, **kwargs)
    assert (path, payload) == expected
    assert payload.endswith(b"\n") and not path.exists()
    assert joint.handoff_template_path(record, **kwargs) == path
    assert path.read_bytes() == payload
    assert joint.handoff_template_path(record, **kwargs) == path
    path.write_text("changed")
    with pytest.raises(joint.DispatchRefused):
        joint.handoff_template_path(record, **kwargs)


def _report_consumer(monkeypatch, tmp_path, *, dry_run=False, pending=False):
    """Run the actual dispatch branch, stubbing external admission/plan seams."""
    record = {"quantum_id": "q", "campaign": {}, "executable_readset": {}}
    monkeypatch.setattr(joint, "_plan_block", lambda _path: {"consumer_tags": ["fixture-雪"]})
    monkeypatch.setattr(joint, "load_records", lambda _path: [(tmp_path / "q.json", record)])
    monkeypatch.setattr(joint, "_plan_output_root", lambda _path: tmp_path)
    monkeypatch.setattr(joint, "load_stage_a_proofs", lambda *a, **kw: [])
    monkeypatch.setattr(joint, "check_stage_a_proofs", lambda *a, **kw: ({"q": "sha"} if pending else {}))
    monkeypatch.setattr(joint, "band_serial_roles", lambda *a, **kw: {"q": {}})
    monkeypatch.setattr(joint, "fresh_band_role", lambda *a, **kw: (None, "wait 雪"))
    monkeypatch.setattr(joint, "_stage_manifest_binding", lambda *a, **kw: {"z": "雪"})
    monkeypatch.setattr(joint, "stage_a_argv", lambda *a, **kw: ["joint_adjoint_capture", "雪"])
    def admit(args, rows, *a, **kwargs):
        for row in rows:
            row["pilot_admission"] = {"fixture": True}
    monkeypatch.setattr(joint, "_admit_dispatch_pilots", admit)
    class Gateway:
        def __init__(self):
            self.submitted = []
        def submit(self, argv):
            self.submitted.append(argv)
            return {"action_key": "action-雪", "status": "submitted"}
    gateway = Gateway()
    args = ["--plan", str(tmp_path / "plan.json"), "--records", str(tmp_path),
            "--output-root", str(tmp_path / "out"), "--state", str(tmp_path / "state.jsonl"),
            "--priority", "-5"]
    if pending:
        monkeypatch.setattr(joint, "_spec_capture_batch", lambda _path: 1)
        args.extend(["--band-serial", "--handoff-tier", "fixture-stage"])
    if dry_run:
        args.append("--dry-run")
    assert joint.main(args, _gateway=gateway, _coverage=lambda _rows: []) == 0
    return gateway


def test_dispatch_dry_run_document(monkeypatch, tmp_path, capsys):
    gateway = _report_consumer(monkeypatch, tmp_path, dry_run=True)
    actual = capsys.readouterr().out
    context = dict(tags=["fixture-雪"], priority=-5, execution_stamp=None,
                   stage_a_keys=[], receipt_ok={}, records=[(None, {"quantum_id": "q"})],
                   publishable=set(), band_pending=[], by_id={},
                   rows=[{"kind": "stage-a", "argv": ["joint_adjoint_capture", "雪"],
                          "pilot_admission": {"fixture": True}}])
    assert actual == _compare("_dispatch", 0, context) + "\n"
    assert not gateway.submitted


def test_dispatch_published_status(monkeypatch, tmp_path, capsys):
    gateway = _report_consumer(monkeypatch, tmp_path)
    expected = _compare("_dispatch", 1, {"row": {"kind": "stage-a"},
                                       "answer": {"action_key": "action-雪", "status": "submitted"}})
    assert capsys.readouterr().out == expected + "\n"
    assert gateway.submitted == [["joint_adjoint_capture", "雪"]]
    state = [json.loads(line) for line in (tmp_path / "state.jsonl").read_text().splitlines()]
    assert state[0]["action_key"] == "action-雪"


def test_dispatch_waiting_status(monkeypatch, tmp_path, capsys):
    gateway = _report_consumer(monkeypatch, tmp_path, pending=True)
    expected = _compare("_dispatch", 2, {"waiting": {"quantum_id": "q", "reason": "wait 雪"}})
    assert capsys.readouterr().out == expected + "\n" + '{"note": "nothing publishable", "published": []}\n'
    assert not gateway.submitted
    assert not (tmp_path / "state.jsonl").exists()


def test_quantum_argv_grace_consumer(tmp_path, monkeypatch, capsys):
    # Existing consumer fixture builds actual slices, receipts, and manifests.
    from test_compute_phase_grace_1165 import _dispatch, FIXTURE_CEILINGS
    _dispatcher, _manifest, argv = _dispatch(
        tmp_path, monkeypatch, capsys, replay_mode="spill", ceilings=FIXTURE_CEILINGS)
    wire = argv[argv.index("--progress-grace-derivation") + 1]
    stamps = json.loads(wire)
    assert stamps
    assert wire == _compare("quantum_argv", 0, {"grace_stamps": stamps})


def test_split_main_report(monkeypatch, capsys):
    report = {"z": ["雪", None, -0.0], "a": True, "mismatches": ["雪"]}
    text = _compare("main", 4, {"report": report}, "tools/dispatch_stage_a_split.py")
    calls = []
    monkeypatch.setattr(split, "load_round", lambda path: calls.append(("load", path)) or {"rows": []})
    monkeypatch.setattr(split, "compare_digests", lambda document, baseline: calls.append(("compare", baseline)) or report)
    argv = ["compare", "--round-dir", "fixture-round", "--baseline", "fixture-baseline"]
    assert split.main(argv) == 4
    assert capsys.readouterr().out == text + "\n"
    assert calls == [("load", "fixture-round"), ("compare", "fixture-baseline")]
    report["mismatches"] = []
    assert split.main(argv) == 0
    assert capsys.readouterr().out == json.dumps(report, indent=2, sort_keys=True) + "\n"


def test_embedded_band_set_target_deployment(tmp_path):
    """The target uses only the two existing runtime imports, never digests."""
    source = ast.parse((FIXTURES / "dispatch_stage_a_split.original.txt").read_text())
    original = next(ast.literal_eval(n.value) for n in source.body
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_BAND_SET" for t in n.targets))
    assert split._BAND_SET == original
    # Execute the actual remote -c payload in an isolated target with the existing
    # roster stubbed. No digests.py is installed on this target.
    package = tmp_path / "prismaquant"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "joint_adjoint_slices.py").write_text("def band_set(bands):\n    assert [b['id'] for b in bands] == ['second', 'first']\n    return {4: None, 2: None}\ndef stage_a_run_header(band):\n    return band['id']\ndef stage_a_run_header_sha256(header):\n    return header\n")
    for name in ("second", "first"):
        (tmp_path / (name + ".json")).write_text(json.dumps({"id": name}))
    result = subprocess.run([sys.executable, "-I", "-c", "import sys, importlib.util; sys.path.insert(0, sys.argv.pop(1)); assert importlib.util.find_spec('prismaquant.digests') is None; " + split._BAND_SET, str(tmp_path), "second.json", "first.json"], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == '{"boundaries": [2, 4], "run_header_sha256": "second"}\n'


def test_unresolved_unsorted_payloads():
    # Eligible pending owner recipes, not justified retention.
    value = {"z": "雪", "a": -0.0}
    assert _compare("_container_wrap", 0, {"spec": value}) == json.dumps(value)
    assert _compare("row_template", 0, {"base_template": value}, "tools/dispatch_stage_a_split.py") == json.dumps(value)
    context = {"args": type("Args", (), {"round_dir": "round"})(), "Path": Path,
               "ROUND_NAME": "round.json", "document": {"rows": [{"name": "雪"}]},
               "row": {"name": "雪", "argv": ["python", "-c"]},
               "result": {"name": "雪", "detach": {"z": 1, "a": 2}}}
    for index in range(4):
        _compare("main", index, context, "tools/dispatch_stage_a_split.py")


def test_ledger_provenance_and_resolvable_nodes():
    ledger = json.loads((FIXTURES / "ledger_dispatch.json").read_text())
    registry = json.loads((ROOT / ledger["registry"]).read_text())
    expected = {c["id"]: c for c in registry["calls"]
                if c["file"] in ("tools/dispatch_joint_quanta.py", "tools/dispatch_stage_a_split.py")}
    assert len(ledger["entries"]) == len(expected) + 1
    nodes = {n.name for n in ast.parse(Path(__file__).read_text()).body if isinstance(n, ast.FunctionDef)}
    for entry in ledger["entries"]:
        if entry["original_id"] in expected:
            original = expected[entry["original_id"]]
            assert entry["source_commit"] == registry["source_commit"]
            assert entry["source_blob"] == original["source_blob"]
            source = FIXTURES / (Path(original["file"]).stem + ".original.txt")
            tree = ast.parse(source.read_text())
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == original["scope"])
            digests_found = {hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
                             for n in ast.walk(function)
                             if isinstance(n, ast.Call) and ast.unparse(n.func) == original["callee"]}
            assert original["syntax_sha256"] in digests_found
        for node in entry["tests"]:
            file, name = node.split("::")
            assert ROOT / file == Path(__file__)
            assert name in nodes
        for fixture in entry["fixtures"]:
            assert (ROOT / fixture).is_file()
    assert sum(e["status"] == "replaced" for e in ledger["entries"]) == 7


def test_frozen_original_outcomes():
    outcomes = json.loads((FIXTURES / "original_outcomes.json").read_text())
    for key, profile in (("compact_strict", digests.DIRECT_ASCII_STRICT),
                         ("compact_lax", digests.DIRECT_ASCII_LAX),
                         ("indent2_lax", digests.DIRECT_ASCII_INDENT2_LAX),
                         ("spaced_lax", digests.DIRECT_ASCII_SPACED_LAX)):
        assert profile.text(outcomes["input"]) == outcomes[key]


def test_unresolved_coverage_default_str():
    value = {"path": Path("雪"), "nan": float("nan")}
    for original in (True, False):
        text = eval(_expression("tools/dispatch_joint_quanta.py", "check_source_coverage", 0, original),
                    {"json": json, "gaps_report": lambda *a, **kw: value, "gaps": [], "coverage_rows": []})
        assert text == json.dumps(value, indent=2, sort_keys=True, default=str)


def test_minimized_original_excerpts():
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    for excerpt in provenance["excerpts"]:
        lines = (FIXTURES / excerpt["fixture"]).read_text().splitlines()
        selected = lines[excerpt["fixture_start_line"] - 1:excerpt["fixture_end_line"]]
        text = "\n".join(line[excerpt["added_indent"]:] for line in selected)
        assert hashlib.sha256(text.encode()).hexdigest() == excerpt["text_sha256"]
    for call in provenance["calls"]:
        tree = ast.parse((FIXTURES / (Path(call["file"]).stem + ".original.txt")).read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == call["scope"])
        syntax = {hashlib.sha256(ast.dump(n, include_attributes=False).encode()).hexdigest()
                  for n in ast.walk(function)
                  if isinstance(n, ast.Call) and ast.unparse(n.func) == call["callee"]}
        assert call["syntax_sha256"] in syntax, call["id"]
    assert (FIXTURES / "dispatch_joint_quanta.original.txt").stat().st_size < 10000
    assert (FIXTURES / "dispatch_stage_a_split.original.txt").stat().st_size < 3000


def test_dispatch_classification_and_blocker():
    ledger = json.loads((FIXTURES / "ledger_dispatch.json").read_text())
    gaps = [e for e in ledger["entries"] if e["status"] == "unresolved_api_gap"]
    assert len(gaps) == 7
    assert all(e["blocker"] == "pending-owner-recipes" for e in gaps)
    assert sum(e["original_id"].startswith("supplemental-json:") for e in gaps) == 6
    retained = [e for e in ledger["entries"] if e["status"] == "retained"]
    assert len(retained) == 1 and retained[0]["original_id"].startswith("embedded:")
    assert ledger["counts"] == dict(registry_calls=14, additional_embedded_sites=1,
                                    replaced=7, eligible_unresolved_api_gaps=7,
                                    justified_deployment_retentions=1)
    blocker, = ledger["blockers"]
    assert blocker["id"] == "pending-owner-recipes" and blocker["status"] == "pending"
    assert [r["sites"] for r in blocker["required_recipes"]] == [6, 1]
