"""Executable compatibility evidence from frozen original consumer snippets.

These are blocked eligible sites awaiting shared owner integration, not justified
retentions or completed migrations. Expected behavior executes original source,
never a replacement owner. No GPU, weights, serving process or package init needed.
"""
import ast
import hashlib
import io
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/fixtures/digest_1301/serving-evidence"
LEDGER = json.loads((FIXTURES / "ledger_values.json").read_text())
CASES = [json.loads((ROOT / row["fixtures"][0]).read_text()) for row in LEDGER["calls"]]
IDS = [Path(row["fixtures"][0]).stem for row in LEDGER["calls"]]


def encoder_calls(tree):
    return sorted(
        (n for n in ast.walk(tree) if isinstance(n, ast.Call)
         and isinstance(n.func, ast.Attribute)
         and isinstance(n.func.value, ast.Name)
         and n.func.value.id == "json"
         and n.func.attr in ("dump", "dumps", "JSONEncoder")),
        key=lambda n: (n.lineno, n.col_offset))


def function_source(case):
    tree = ast.parse((ROOT / case["file"]).read_text())
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == case["scope"])


def current_expression(case):
    original = ast.parse(case["function_source"])
    old_calls = encoder_calls(original)
    # The frozen enclosing expression identifies each distinct call, including
    # the two copies in make_uniform_control_record.
    wanted = ast.dump(ast.parse(case["expression"], mode="eval").body)
    parents = {child: parent for parent in ast.walk(original)
               for child in ast.iter_child_nodes(parent)}
    index = next(i for i, call in enumerate(old_calls)
                 if ast.dump(parents[call] if case["roundtrip"] else call) == wanted)
    current = function_source(case)
    calls = encoder_calls(current)
    parent_map = {child: parent for parent in ast.walk(current)
                  for child in ast.iter_child_nodes(parent)}
    node = calls[index]
    return parent_map[node] if case["roundtrip"] else node


def execute_expression(node, value):
    handle = io.StringIO()
    env = dict(json=json, value=value, identity=value, control_block=value,
               control_arm=value, record=value, receipt=value,
               row={"sweep": value}, handle=handle)
    try:
        result = eval(compile(ast.Expression(node), "<consumer-expression>", "eval"), env)
        if handle.tell():
            result = handle.getvalue()
        # repr captures insertion order, tuple-to-list normalization, NaN and
        # escaped versus literal surrogates without JSON re-normalizing a result.
        return ("ok", type(result).__name__, repr(result))
    except Exception as exc:
        return ("error", type(exc).__name__, exc.args, handle.getvalue())


def values():
    cycle = []
    cycle.append(cycle)
    return [
        {"z": "é😀", "a": [True, None, -0.0]},
        {10: "ten", 9: "nine"},
        {"nested": {"z": (1, 2), "a": "\\\n\t"}},
        {"z": float("nan"), "a": float("inf")},
        {"v": float("-inf")},
        {"v": "\ud800"},
        {"v": object()},
        {"v": cycle},
        {"v": {1, 2}},
        {(1, 2): "bad key"},
        {"a": 1, 2: "mixed keys accepted without sorting"},
        {},
    ]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_original_consumer_behavior(case):
    original = ast.parse(case["expression"], mode="eval").body
    current = current_expression(case)
    for value in values():
        assert execute_expression(current, value) == execute_expression(original, value)
    # Explicit discriminator: neither direct JSON nor its loads roundtrip sorts.
    value = {"z": 1, "a": "é"}
    outcome = execute_expression(current, value)
    assert outcome[0] == "ok"
    assert outcome[2].index("z") < outcome[2].index("a")
    if case["roundtrip"]:
        assert outcome[1] == "dict"
        assert "é" in outcome[2]
    else:
        assert outcome[1] == "str"
        assert "\\u00e9" in outcome[2]


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_frozen_provenance_and_recipe_gap(case):
    registry = json.loads((ROOT / "tests/fixtures/digest_parent_completion_1301.json").read_text())
    entry = next(c for c in registry["calls"] if c["id"] == case["original_id"])
    assert case["source_commit"] == registry["source_commit"]
    assert case["source_blob"] == entry["source_blob"]
    assert case["syntax_sha256"] == entry["syntax_sha256"]
    # Frozen registry fingerprints use attribute-free AST syntax, not owner output.
    node = next(n for n in encoder_calls(ast.parse(case["function_source"]))
                if ast.dump(n) == ast.dump(next(iter(encoder_calls(
                    ast.parse(case["expression"], mode="eval"))))))
    assert hashlib.sha256(ast.dump(node).encode()).hexdigest() == entry["syntax_sha256"]
    for call in (node, next(iter(encoder_calls(ast.Expression(current_expression(case)))))):
        keywords = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        assert keywords.get("sort_keys", False) is False
    owner = ast.parse((ROOT / "prismaquant/digests.py").read_text())
    profile = next(n for n in owner.body if isinstance(n, ast.ClassDef) and n.name == "JsonProfile")
    encoder = next(n for n in profile.body if isinstance(n, ast.FunctionDef) and n.name == "_encoder")
    call = next(n for n in ast.walk(encoder) if isinstance(n, ast.Call))
    assert next(kw.value.value for kw in call.keywords if kw.arg == "sort_keys") is True


def test_whole_file_raw_call_accounting():
    assert len(CASES) == 8
    assert len(LEDGER["whole_file_raw_call_accounting"]) == 5
    for account in LEDGER["whole_file_raw_call_accounting"]:
        calls = encoder_calls(ast.parse((ROOT / account["file"]).read_text()))
        assert len(calls) == account["current_raw_json_encoder_calls"]
        assert account["original_raw_json_encoder_calls"] == len(calls)
        assert account["replaced"] == 0
        assert account["blocked_eligible"] == len(calls)
        assert set(account["original_ids"]) == {
            c["original_id"] for c in CASES if c["file"] == account["file"]}
    assert LEDGER["disposition"] == "blocked_eligible"
    assert LEDGER["retained_calls"] == []
    assert LEDGER["production_replacements"] == 0
    assert LEDGER["integration_status"] == "awaiting_shared_owner_integration"
    for row, case in zip(LEDGER["calls"], CASES):
        assert row["owner_route"] is None
        assert row["disposition"] == "blocked_eligible"
        assert row["integration_status"] == LEDGER["integration_status"]
        for key in ("original_id", "file", "scope", "source_commit", "source_blob"):
            assert row[key] == case[key]
        recipe = row["requested_recipe"]
        assert recipe["owner"] == LEDGER["owner"]
        assert recipe["consumer_expression"] == case["expression"]
        assert recipe["result"] == ("loads_roundtrip" if case["roundtrip"] else "direct_text")
        call = encoder_calls(ast.parse(case["expression"]))[0]
        assert recipe["encoder"] == call.func.attr
        options = dict(skipkeys=False, ensure_ascii=True, check_circular=True,
                       allow_nan=True, cls=None, indent=None, default=None, sort_keys=False)
        options.update({kw.arg: ast.literal_eval(kw.value) for kw in call.keywords})
        options["separators"] = [", ", ": "] if options["indent"] is None else [",", ": "]
        assert recipe["options"] == options


def load_function(source, name):
    namespace = dict(json=json, Path=Path, sys=sys, Any=object)
    # Compile the real consumer body, avoiding heavyweight package imports.
    tree = ast.parse("from __future__ import annotations\n" + source)
    exec(compile(tree, "<consumer-function>", "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("scope", ["_write", "write_sweep"])
def test_actual_publication_functions(scope, tmp_path, capsys):
    case = next(c for c in CASES if c["scope"] == scope)
    old = load_function(case["function_source"], scope)
    new = load_function(ast.unparse(function_source(case)), scope)
    record = {"z": "é", "a": {"z": [1, 2], "a": float("nan")}}
    old_path, new_path = tmp_path / "old.json", tmp_path / "new.json"
    if scope == "_write":
        old(record, str(old_path))
        new(record, str(new_path))
        old(record, "-")
        expected = capsys.readouterr().out
        new(record, "-")
        assert capsys.readouterr().out == expected
    else:
        assert old(str(old_path), record) == str(old_path)
        assert new(str(new_path), record) == str(new_path)
    assert new_path.read_bytes() == old_path.read_bytes()
    assert new_path.read_bytes().endswith(b"\n")
    assert b"\\u00e9" in new_path.read_bytes()


def test_actual_normalization_function():
    case = next(c for c in CASES if c["scope"] == "normalize_source_derivative")
    def require(ok, message):
        if not ok:
            raise ValueError("GLM source derivative: " + message)
    functions = [load_function(source, case["scope"]) for source in
                 (case["function_source"], ast.unparse(function_source(case)))]
    for function in functions:
        function.__globals__.update(_require=require, SCHEMA="schema", VERSION="version")
        assert function(None) is None
        value = dict(version="version", schema="schema", image_build={"path": "é", "sha256": "a"})
        result = function(value)
        assert result == value and result is not value
        assert result["image_build"] is not value["image_build"]
        assert list(result) == list(value)
        value["image_build"]["path"] = float("nan")
        with pytest.raises(ValueError, match="Out of range float values"):
            function(value)
        with pytest.raises(ValueError, match="closed policy required"):
            function({})
