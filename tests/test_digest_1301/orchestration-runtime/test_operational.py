"""Operational consumer slices, frozen independently at the requested source commit.

These execute original consumer expressions/statements, not a reimplementation of
an owner. Heavy model/runtime imports are deliberately outside this CPU test.
The seven insertion-order recipes remain blockers, not approximate migrations.
"""
import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant import digests

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/fixtures/digest_1301/orchestration-runtime/operational"
LEDGER = json.loads((FIXTURES / "ledger.json").read_text())
PRIMARY = [entry for entry in LEDGER if entry["status"] == "already_routed"]
SUPPLEMENTAL = [entry for entry in LEDGER if entry["status"] == "blocked"]


def _current_call(entry):
    tree = ast.parse((ROOT / entry["file"]).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and node.lineno == entry["line"]
             and ast.unparse(node.func) == entry["callee"]]
    assert len(calls) == 1, entry["original_id"]
    return calls[0], tree


def _call(tree, callee):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                and ast.unparse(node.func) == callee)


def _eval(node, env):
    return eval(compile(ast.Expression(node), "<consumer slice>", "eval"), env)


@pytest.mark.parametrize("entry", PRIMARY, ids=lambda e: e["original_id"])
def test_primary_consumer_compatibility(entry, tmp_path):
    original = ast.parse((FIXTURES / entry["fixtures"][0]).read_text())
    old_call = _call(original, entry["callee"])
    current, _ = _current_call(entry)
    assert ast.dump(current) == ast.dump(old_call)
    value = {"z": ["é", 3], "a": {"x": True}}
    path = tmp_path / "input.bin"
    path.write_bytes(b"original\x00consumer\xff")
    env = dict(vars(digests))
    env["_initialization_digest"] = digests.DIRECT_ASCII_STRICT.sha256
    # Bind only the arguments used by the real call expression. Calls and their
    # owner selection are executed unmodified from both original and current AST.
    for node in ast.walk(old_call.args[0]):
        if isinstance(node, ast.Name):
            env[node.id] = (path if entry["callee"] == "file_sha256hex" else
                            b"original\x00consumer\xff" if entry["callee"] == "bytes_sha256hex"
                            else value)
    env["self"] = SimpleNamespace(records=value)
    first = _eval(old_call.args[0], env)
    callee = entry["callee"]
    if callee == "file_sha256hex":
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
    elif callee == "bytes_sha256hex":
        expected = hashlib.sha256(first).hexdigest()
    elif callee == "indent2_json_file_bytes":
        expected = (json.dumps(first, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    elif callee == "DIRECT_ASCII_INDENT2_LAX.text":
        expected = json.dumps(first, sort_keys=True, indent=2)
    else:
        expected = hashlib.sha256(json.dumps(
            first, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert _eval(old_call, env) == expected
    assert _eval(current, env) == expected
    assert entry["owner_route"].startswith("prismaquant.digests.")


def _statement_result(statement, env):
    statement = copy.deepcopy(statement)
    # Retain the consumer's print, assignment, or return and its nesting.
    body = [statement]
    if isinstance(statement, ast.Assign):
        body.append(ast.Return(ast.Name("state", ast.Load())))
    fn = ast.FunctionDef(name="consumer", args=ast.arguments(
        posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=body, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(module, "<consumer statement>", "exec"), env)
        result = env["consumer"]()
    return result, output.getvalue()


@pytest.mark.parametrize("entry", SUPPLEMENTAL, ids=lambda e: e["original_id"])
@pytest.mark.parametrize("case", ["ordered", "nonfinite", "invalid"])
def test_supplemental_consumer_compatibility(entry, case):
    fixture = ast.parse((FIXTURES / entry["fixtures"][0]).read_text())
    call, tree = _current_call(entry)
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    statement = call
    while not isinstance(statement, ast.stmt):
        statement = parents[statement]
    value = {"z": ("é", 2), "a": {"tail": 1, "first": 0}}
    if case == "nonfinite":
        value["n"] = float("inf")
    if case == "invalid":
        value["bad"] = object()
    report = dict(value, status="refused", problems=["é"])
    def environment():
        return dict(json=json, diag=value, report=report, value=value,
                    self=SimpleNamespace(records=value), evidence=[{"z": 1, "a": 2}],
                    exc=ValueError("é"))
    def observe(stmt):
        try:
            return ("ok", _statement_result(stmt, environment()))
        except Exception as exc:
            return (type(exc).__name__, str(exc))
    assert observe(statement) == observe(fixture.body[0])
    # Exact insertion order, escaped Unicode, and tuple-to-list normalization.
    if case == "ordered":
        result, stdout = _statement_result(statement, environment())
        if result is not None:
            assert list(result) == ["z", "a"]
            assert result["z"] == ["é", 2]
            assert result is not value
        elif entry["file"].endswith(("autoscale.py", "day0_model_intake.py")):
            assert stdout.startswith('{\n  "z":')
            assert '\\u00e9' in stdout
        elif entry["file"].endswith("release_receipts.py"):
            assert stdout == '{"status": "error", "error": "\\u00e9"}\n'
        else:
            assert stdout.startswith('{"status": "refused", "evidence":')
    assert entry["owner_route"] is None


def test_frozen_registry_and_executable_ledger():
    registry = json.loads((ROOT / "tests/fixtures/digest_parent_completion_1301.json").read_text())
    scoped_files = {e["file"] for e in LEDGER}
    frozen = {e["id"]: e for e in registry["calls"] if e["file"] in scoped_files}
    assert {e["original_id"] for e in SUPPLEMENTAL} == set(frozen)
    functions = {n.name for n in ast.walk(ast.parse(Path(__file__).read_text()))
                 if isinstance(n, ast.FunctionDef)}
    for entry in LEDGER:
        assert entry["source_commit"] == "6d8288adc94b3c4b624c143ed372fb6513f32784"
        assert len(entry["source_blob"]) == 40
        source = (ROOT / entry["file"]).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(source)).encode() + b"\0" + source).hexdigest()
        assert blob == entry["source_blob"]
        for fixture in entry["fixtures"]:
            ast.parse((FIXTURES / fixture).read_text())
        for node in entry["test_nodes"]:
            assert node.split("::")[1] in functions
        if entry["status"] == "blocked":
            assert entry["source_blob"] == frozen[entry["original_id"]]["source_blob"]
            call, _ = _current_call(entry)
            syntax = ast.dump(call, annotate_fields=True, include_attributes=False)
            assert hashlib.sha256(syntax.encode()).hexdigest() == entry["syntax_sha256"]
            assert entry["blocker"]


def _load_function(source, name, env):
    function = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<consumer>", "exec"), env)
    return env[name]


@pytest.mark.parametrize("status", ["ready", "refused", "error"])
def test_release_main(status):
    outputs = []
    for source in ((FIXTURES / "release_receipts_main_original.py").read_text(),
                   (ROOT / "prismaquant/release_receipts.py").read_text()):
        calls = []
        def ingest(shipcard, **kwargs):
            calls.append((shipcard, kwargs))
            if status == "error":
                raise ValueError("é")
            return {"z": "é", "status": status}
        env = dict(argparse=argparse, json=json, ingest_receipts=ingest,
                   DIRECT_ASCII_INDENT2_LAX=digests.DIRECT_ASCII_INDENT2_LAX)
        main = _load_function(source, "main", env)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            code = main(["card.json", "--window", "window.json", "--apply"])
        outputs.append((code, stdout.getvalue(), calls))
    assert outputs[0] == outputs[1]
    assert outputs[0][0] == {"ready": 0, "refused": 1, "error": 2}[status]
    assert outputs[0][2] == [("card.json", dict(window="window.json", records_dir=None, apply=True))]
    if status == "error":
        assert outputs[0][1] == '{"status": "error", "error": "\\u00e9"}\n'.replace("é", "\\u00e9")


@pytest.mark.parametrize("invalid", [False, True])
def test_mtp_validator(invalid):
    import prismaquant.streaming_initialization as initialization
    state = {"z": {"shape": [2], "dtype": "float32", "sha256": "a" * 64}}
    value = dict(schema=initialization._MTP_LAYER_INITIALIZATION_SCHEMA,
                 scope="mtp_layer_checkpoint_load", status="completed",
                 transformers_version="test", model_class="Test", dtype="float32",
                 layer_prefix="mtp", experts_implementation="test", checkpoint_tensors=1,
                 state=state, state_sha256=digests.DIRECT_ASCII_STRICT.sha256(state),
                 source_map_sha256="b" * 64,
                 input={"schema": "test", "path": "input.json", "sha256": "c" * 64})
    if invalid:
        value["state_sha256"] = "d" * 64
    old = _load_function((FIXTURES / "streaming_initialization_validate_mtp_layer_initialization_contract_original.py").read_text(),
                         "validate_mtp_layer_initialization_contract", dict(vars(initialization)))
    results = []
    for consumer in (old, initialization.validate_mtp_layer_initialization_contract):
        if invalid:
            with pytest.raises(ValueError, match="state digest differs"):
                consumer(value)
        else:
            result = consumer(value)
            assert result == value and result is not value
            assert result["state"] is not value["state"]
            results.append(list(result))
    if not invalid:
        assert results == [list(value), list(value)]


def test_sorted_owners_are_not_insertion_order_recipes():
    value = {"z": 1, "a": 2}
    for profile in (digests.DIRECT_ASCII_LAX, digests.DIRECT_ASCII_STRICT,
                    digests.DIRECT_ASCII_INDENT2_LAX):
        assert profile.text(value).index('"a"') < profile.text(value).index('"z"')
    assert json.dumps(value).startswith('{"z":')
