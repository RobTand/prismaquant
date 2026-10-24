"""Independent original-source oracles for the remaining campaign routes."""
from __future__ import annotations

import argparse
import ast
import contextlib
from collections import Counter
import copy
import hashlib
import io
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant import digests

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/fixtures/digest_1301/tessera-supply/campaigns"
LEDGER = json.loads((FIXTURES / "ledger.json").read_text())


def namespace():
    return dict(vars(digests), argparse=argparse, hashlib=hashlib, json=json,
                pickle=pickle, Path=Path, sys=sys, __doc__="campaign oracle")


def functions(name, old, extra=None):
    source = ((FIXTURES / (name + ".py")) if old else
              ROOT / "tools" / (name + ".py")).read_text()
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    ns = namespace()
    ns.update(extra or {})
    exec(compile(ast.fix_missing_locations(tree), name, "exec"), ns)
    ns.update(extra or {})
    return ns


def calls_in_scope(path, scope):
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == scope)
    return [n for n in ast.walk(fn) if isinstance(n, ast.Call)]


def outcome(call):
    try:
        return ("ok", call())
    except (ValueError, TypeError, UnicodeError, OverflowError) as exc:
        return (type(exc).__name__, str(exc))


def test_original_fragments():
    registry = json.loads((ROOT / "tests/fixtures/digest_parent_completion_1301.json").read_text())
    registered = {c["id"]: c for c in registry["calls"]}
    all_entries = LEDGER["replacements"] + LEDGER["unresolved"]
    assert len({x["original_id"] for x in all_entries}) == len(all_entries)
    assigned = {x["file"] for x in all_entries}
    assert {x["original_id"] for x in all_entries} == {
        c["id"] for c in registry["calls"] if c["file"] in assigned}
    for entry in all_entries:
        rid = entry["original_id"]
        assert entry["source_commit"] == "6d8288adc94b3c4b624c143ed372fb6513f32784"
        assert entry["source_blob"] == registered[rid]["source_blob"]
        assert "fixture" not in entry
        assert entry["fixtures"] and all((ROOT / path).is_file() for path in entry["fixtures"])
        fixture = json.loads(next(ROOT / path for path in entry["fixtures"] if path.endswith(".json")).read_text())
        source = next(f["source"] for f in fixture["fragments"] if f["original_id"] == rid)
        original = ast.parse(source, mode="eval").body
        current_calls = calls_in_scope(ROOT / entry["file"], entry["scope"])
        if "reason" in entry:
            assert any(ast.dump(n) == ast.dump(original) for n in current_calls)
            continue
        route = entry["owner_route"].rsplit(".", 1)[1]
        matches = [n for n in current_calls if
                   (isinstance(n.func, ast.Name) and n.func.id == route) or
                   (isinstance(n.func, ast.Attribute) and n.func.attr == "text" and
                    isinstance(n.func.value, ast.Name) and n.func.value.id == route)]
        assert matches, rid
        # Execute the actual new call node and the independently retained old
        # expression, substituting only the payload, never encoding options.
        new = copy.deepcopy(matches[0])
        new.args = [ast.Name(id="value", ctx=ast.Load())]
        if registered[rid]["callee"] == "hashlib.sha256":
            old_expr = ast.parse("hashlib.sha256(value).hexdigest()", mode="eval").body
            samples = [b"", b"\x00\xff\r\n", b"abc" * 10000]
        else:
            old_expr = copy.deepcopy(original)
            old_expr.args = [ast.Name(id="value", ctx=ast.Load())]
            samples = [{"z": "é雪\\\n", "a": [True, None, -0.0, 1.25]},
                       {"z": float("nan"), "a": float("inf")},
                       {"bad": object()}, {1: "x", "a": "y"}, {"s": "\ud800"}]
        old_code = compile(ast.fix_missing_locations(ast.Expression(old_expr)), "original", "eval")
        new_code = compile(ast.fix_missing_locations(ast.Expression(new)), "current", "eval")
        for value in samples:
            env = namespace() | {"value": value}
            old_result = outcome(lambda: eval(old_code, env))
            def evaluate_new():
                result = eval(new_code, env)
                return result.decode("utf-8")[:-1] if route == "indent2_json_file_bytes" else result
            assert outcome(evaluate_new) == old_result, rid


def test_bound_read_before_unpickle_and_refusal(tmp_path):
    raw = pickle.dumps({"é": [1, 2]})
    path = tmp_path / "input.pkl"
    path.write_bytes(raw)
    for old in (True, False):
        events = []
        def loads(data):
            events.append(data)
            return pickle.loads(data)
        ns = functions("expand_probe_onto_tessera_census", old,
                       {"pickle": SimpleNamespace(loads=loads)})
        with pytest.raises(SystemExit, match="expected wrong"):
            ns["_bound"](str(path), "wrong", "probe")
        assert events == []
        assert ns["_bound"](str(path), hashlib.sha256(raw).hexdigest(), "probe") == {"é": [1, 2]}
        assert events == [raw]
        grace = functions("derive_tessera_progress_grace", old)
        assert grace["sha256"](path) == hashlib.sha256(raw).hexdigest()


def test_seal_publication_order_bytes_and_existing_output(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b'{"identity":"snow \\u96ea"}\r\n')
    out = tmp_path / "roster.json"
    results = []
    for old in (True, False):
        events = []
        def write(path, raw):
            events.append((str(path), raw))
        ns = functions("seal_tessera_census_identity", old, {
            "seal_roster": lambda value: {"units": {"z": {}}, "identity_sha256": "é雪"},
            "_atomic_write_new_bytes": write})
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            assert ns["main"](["--manifest", str(manifest), "--out", str(out)]) == 0
        assert len(events) == 2
        assert events[0][0] == str(out)
        assert events[1] == (str(out) + ".sha256",
                             f"{hashlib.sha256(events[0][1]).hexdigest()}  {out.name}\n".encode())
        roster = json.loads(events[0][1])
        assert roster["checkpoint_manifest"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
        results.append((events, stdout.getvalue()))
        def refuse(path, raw):
            raise FileExistsError("existing output")
        ns["_atomic_write_new_bytes"] = refuse
        with pytest.raises(FileExistsError):
            ns["main"](["--manifest", str(manifest), "--out", str(out)])
    assert results[0] == results[1]


def test_replay_publication_and_refusals(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    out = tmp_path / "report.json"
    args = ["--costs", "cost", "--checkpoint", "checkpoint", "--plan", str(plan), "--out", str(out)]
    results = []
    for old in (True, False):
        events = []
        def publish(path, raw, **kw):
            events.append((path, raw, kw))
            return True
        ns = functions("tessera_surface_replay", old, {
            "replay_campaign": lambda *args: {"z": "é雪", "a": -0.0},
            "publish_new_bytes": publish})
        assert ns["main"](args) == 0
        assert events[0][2] == {"nofollow": True}
        results.append(events)
        ns["publish_new_bytes"] = lambda *args, **kw: False
        out.write_bytes(events[0][1])
        assert ns["main"](args) == 0
        out.write_bytes(b"different")
        with pytest.raises(SystemExit) as exc:
            ns["main"](args)
        assert exc.value.code == 2
        ns["replay_campaign"] = lambda *args: {"bad": float("nan")}
        with pytest.raises(SystemExit) as exc:
            ns["main"](args)
        assert exc.value.code == 2
    assert results[0] == results[1]


def test_dispatch_resources_before_submit_and_capture_refusal(tmp_path):
    results = []
    for old in (True, False):
        workspace = tmp_path / "workspace"
        workspace.mkdir(exist_ok=True)
        args = SimpleNamespace(spec="spec", workspace=str(workspace), timeout_s=10,
                               submit=True, wait_s=2)
        events = []
        adapter = SimpleNamespace(
            load_spec=lambda path: {"model": "model", "campaign_argv": ["--streaming"]},
            prismabuild_row=lambda *args, **kw: {"z": "雪", "a": kw},
            row_memory_gb=lambda *args: 12,
            streamed_resource_plan=lambda *args: {"z": "é雪", "a": float("inf")})
        def submit(manifest, **kw):
            resource = workspace / manifest.name.replace("manifest", "resources")
            events.append((manifest.name, manifest.read_bytes(), resource.read_bytes(), kw))
            return 7
        ns = functions("dispatch_tessera_campaign", old, {
            "campaign_rows": adapter, "explicit_campaign_options": lambda argv: {"streaming": True},
            "_pbcampaign": submit})
        with contextlib.redirect_stdout(io.StringIO()):
            assert ns["cmd_census"](args) == 7
            (workspace / "census.json").write_text('{"model":"model","counts":{"b":2,"a":1}}')
            assert ns["cmd_capture"](args) == 7
        assert len(events) == 2
        results.append(events)
        (workspace / "census.json").write_text('{"model":"different"}')
        with pytest.raises(RuntimeError, match="different models"):
            ns["cmd_capture"](args)
        assert len(events) == 2
    assert results[0] == results[1]


def test_expand_main_publication_bytes_and_validation(tmp_path):
    probe = tmp_path / "probe.pkl"
    cost = tmp_path / "cost.pkl"
    probe.write_bytes(pickle.dumps({"probe": 1}))
    cost.write_bytes(pickle.dumps({"cost": 2}))
    output = tmp_path / "expanded.pkl"
    args = ["--probe", str(probe), "--probe-sha256", hashlib.sha256(probe.read_bytes()).hexdigest(),
            "--costs", str(cost), "--costs-sha256", hashlib.sha256(cost.read_bytes()).hexdigest(),
            "--structure-spec", "spec", "--out", str(output)]
    results = []
    for old in (True, False):
        events = []
        def expanded(probe, cost, **kw):
            assert probe == {"probe": 1} and cost == {"cost": 2}
            return {"meta": {"test": {"dropped_rows": ["x"], "z": "雪"}},
                    "stats": {"row": {"n_params": 12}}}
        ns = functions("expand_probe_onto_tessera_census", old, {
            "load_structure_spec": lambda name: {}, "META_KEY": "test",
            "expand_probe_onto_census": expanded,
            "validate_probe_payload": lambda *args: events.append("validate"),
            "_atomic_write_new_bytes": lambda path, raw: events.append((path, raw))})
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            assert ns["main"](args) == 0
        assert events[0] == "validate"
        assert events[1][0] == output
        raw = events[1][1]
        assert events[2] == (output.with_name(output.name + ".sha256"),
                             f"{hashlib.sha256(raw).hexdigest()}  {output.name}\n".encode())
        results.append((events, stream.getvalue()))
        def refuse(*args):
            raise ValueError("invalid probe")
        ns["validate_probe_payload"] = refuse
        events.clear()
        with pytest.raises(ValueError, match="invalid probe"):
            ns["main"](args)
        assert not events
    assert results[0] == results[1]


def test_union_and_prismasnap_output_and_refusals(tmp_path):
    outputs = []
    class Refused(ValueError):
        pass
    for old in (True, False):
        events = []
        ns = functions("union_tessera_census_tables", old, {
            "UnionRefused": Refused,
            "build_plan": lambda *args: SimpleNamespace(summary={"z": Path("雪"), "a": -0.0}),
            "write_plan": lambda *args, **kw: events.append(kw) or 3})
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            assert ns["main"](["--table", "a=x", "--out", str(tmp_path / "new"), "--dry-run"]) == 0
            snap = functions("prismasnap", old)
            snap["_print"]({"z": "é雪", "a": -0.0})
        assert not events
        outputs.append(stream.getvalue())
        with pytest.raises(ValueError):
            snap["_print"]({"bad": float("nan")})
        with contextlib.redirect_stderr(io.StringIO()):
            assert ns["main"](["--table", "a=x", "--out", str(tmp_path)]) == 2
    assert outputs[0] == outputs[1]


def test_function_source_provenance_and_unresolved_gaps():
    for entry in json.loads((FIXTURES / "function_sources.json").read_text()):
        source = (ROOT / entry["fixture"]).read_text()
        node = next(n for n in ast.parse(source).body
                    if isinstance(n, ast.FunctionDef) and n.name == entry["function"])
        assert hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest() == entry["sha256"]
        assert entry["source_commit"] == "6d8288adc94b3c4b624c143ed372fb6513f32784"
        assert entry["source_blob"] in {e["source_blob"] for e in LEDGER["replacements"]}
    assert len(LEDGER["unresolved"]) == 16
    assert sum(e["reason"].startswith("Unsorted") for e in LEDGER["unresolved"]) == 15
    assert sum(e["reason"].startswith("Indent1 strict") for e in LEDGER["unresolved"]) == 1


def test_dispatch_plan_publication_and_refusal(tmp_path):
    workspace = tmp_path / "plan-workspace"
    workspace.mkdir()
    census = {"model": "model", "layer_stride": 1,
              "anchor_groups": {"z": ["雪"], "a": ["é"]}}
    args = SimpleNamespace(spec="spec", workspace=str(workspace), groups_per_row=1,
                           rows_per_box=1, stack_sample=None, seed_checkpoint=None,
                           seed_wire_dir=None, timeout_s=10, stack_sample_seed=3,
                           audit_rate=0, probe=None)
    results = []
    for old in (True, False):
        (workspace / "census.json").write_text(json.dumps(census))
        events = []
        adapter = SimpleNamespace(
            load_spec=lambda path: {"model": "model", "campaign_argv": []},
            row_memory_gb=lambda *a, **k: 12,
            prismabuild_row=lambda spec, argv, **kw: {"argv": argv, "demand": {"mem_gb": 12}},
            process_baseline=lambda spec: (0, "test"), guard_margin_bytes=lambda: 0)
        def manifests(workspace, plan, selections, rows):
            # Derivation must precede all publication, including selections.
            assert not (workspace / "units").exists()
            assert len(selections) == len(rows) == 2
            events.append(copy.deepcopy(plan))
            return rows, [(workspace / "data.json", b'{"z":"test"}\n')]
        ns = functions("dispatch_tessera_campaign", old, {
            "campaign_rows": adapter, "explicit_campaign_options": lambda argv: {},
            "_planned_campaign_acquisition": lambda *a: (None, None),
            "_calibration_cache_binding": lambda *a: None,
            "_planned_source_identity_cache": lambda *a: None,
            "_source_identity_cache_binding": lambda *a: None,
            "partition_rows_by_fit": lambda demand, *a: (list(demand), []),
            "planned_data_manifests": manifests,
            "UNITS_SCHEMA": "prismaquant.tessera_campaign_units.v1", "PLAN_SCHEMA": "test"})
        with contextlib.redirect_stdout(io.StringIO()):
            assert ns["cmd_plan"](args) == 0
        files = {str(p.relative_to(workspace)): p.read_bytes()
                 for p in workspace.rglob("*") if p.is_file()}
        assert json.loads(files["units/row-0000.json"])["groups"][0]["key"] == "a"
        assert len(json.loads(files["manifest.json"])) == 2
        assert json.loads(files["plan.json"])["rows"][1]["members"] == ["雪"]
        results.append((files, events))
        def refuse(*a):
            raise RuntimeError("unbound read set")
        ns["planned_data_manifests"] = refuse
        with pytest.raises(RuntimeError, match="unbound read set"):
            ns["cmd_plan"](args)
        assert files == {str(p.relative_to(workspace)): p.read_bytes()
                         for p in workspace.rglob("*") if p.is_file()}
        (workspace / "census.json").write_text('{"model":"different"}')
        with pytest.raises(RuntimeError, match="spec names"):
            ns["cmd_plan"](args)
        for p in sorted(workspace.rglob("*"), reverse=True):
            p.rmdir() if p.is_dir() else p.unlink()
    assert results[0] == results[1]


def test_build_cache_main_publication_and_refusals(tmp_path):
    cost = {"costs": {"unit": {}}, "provenance": {"wire_dir": str(tmp_path / "wires")}}
    inputs = {"costs": pickle.dumps(cost), "census": b'{"unit_shapes":{}}',
              "roster": b'{"identity_sha256":"seal"}',
              "layer-config": '{"unit":"BF16","__meta__":{"z":"雪"}}'.encode()}
    argv = ["--checkpoint-parts", str(tmp_path / "parts"), "--out-dir", str(tmp_path / "out"),
            "--plan-layer-config-out", str(tmp_path / "projected.json")]
    for label, raw in inputs.items():
        path = tmp_path / label
        path.write_bytes(raw)
        argv += ["--" + label, str(path), "--" + label + "-sha256", hashlib.sha256(raw).hexdigest()]
    results = []
    for old in (True, False):
        events = []
        def write(path, raw):
            events.append((path.name, raw))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(raw)
        def manifest(*a, **kw):
            return {"z": "é雪", "units": {"unit": {"blob_bytes": 7}}, "source": "test"}
        ns = functions("build_tessera_census_cache", old, {
            "Counter": Counter, "time": SimpleNamespace(monotonic=lambda: 10),
            "_atomic_write_new_bytes": write,
            "validate_layer_config_payload": lambda *a: events.append("validate"),
            "canonicalize_assignment": lambda c: {"unit": "BF16"},
            "layer_config_metadata": lambda c: {"z": "雪"},
            "selected_census_assignment": lambda *a: ({"unit": "BF16"},),
            "plan_layer_config_projection": lambda c, cost: (c, []),
            "load_selected_wire_records": lambda *a, **kw: {},
            "census_selected_cached_units_manifest": manifest,
            "CachedUnitBundle": lambda *a: events.append("bundle"),
            "INPUT_SCHEMA": "input", "ENCODING_INPUT_SCHEMA": "encoding", "CACHE_SCHEMA": "cache"})
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            assert ns["main"](argv) == 0
        assert [e if isinstance(e, str) else e[0] for e in events] == [
            "validate", "projected.json", "bundle", "manifest.json", "summary.json"]
        published = {e[0]: e[1] for e in events if isinstance(e, tuple)}
        summary = json.loads(published["summary.json"])
        assert summary["sha256"]["manifest"] == hashlib.sha256(published["manifest.json"]).hexdigest()
        assert summary["sha256"]["plan_layer_config"] == hashlib.sha256(published["projected.json"]).hexdigest()
        assert json.loads(stream.getvalue().splitlines()[1])["summary_sha256"] == hashlib.sha256(
            published["summary.json"]).hexdigest()
        results.append((events.copy(), stream.getvalue()))
        before = {p: p.read_bytes() for p in (tmp_path / "out").iterdir()}
        with pytest.raises(FileExistsError):
            ns["main"](argv)
        assert before == {p: p.read_bytes() for p in (tmp_path / "out").iterdir()}
        events.clear()
        bad = argv.copy()
        bad[bad.index("--costs-sha256") + 1] = "wrong"
        with pytest.raises(SystemExit, match="expected wrong"):
            ns["main"](bad)
        assert events == []
        def refuse(*a):
            raise ValueError("outside roster")
        ns["selected_census_assignment"] = refuse
        with pytest.raises(ValueError, match="outside roster"):
            ns["main"](argv)
        assert events == ["validate"]
        for path in (tmp_path / "out").iterdir():
            path.unlink()
        (tmp_path / "projected.json").unlink()
    assert results[0] == results[1]


def test_reselect_main_publication_and_refusals(tmp_path, monkeypatch):
    body = {"body": {"bits": 16}, "__prismaquant__": {
        "target_profile": "test", "assignment_payload_bits_total": 80,
        "tessera_serving_scope": {"target": "test", "by_unit": {"body": "scope"}},
        "serving_lane_provenance": {"by_unit": {"body": "lane"},
                                    "activation_pricing_branches": {"joint_aura": 1}}}}
    body_path = tmp_path / "body.json"
    body_path.write_text(json.dumps(body))
    for name, value in (("fixed", {"group": "BF16"}), ("constants", {})):
        (tmp_path / name).write_text(json.dumps(value))
    output = tmp_path / "result.json"
    argv = ["reselect", "--body-layer-config", str(body_path), "--expect-body-sha256",
            hashlib.sha256(body_path.read_bytes()).hexdigest(), "--mtp-joint-cost", "cost",
            "--expect-cost-sha256", "cost-sha", "--mtp-byte-budget", "5",
            "--whole-artifact-budget-bytes", "100", "--observed-body-export", str(tmp_path),
            "--new-metadata-allowance-bytes", "2", "--mtp-serve-constants", str(tmp_path / "constants"),
            "--fixed-formats", str(tmp_path / "fixed"), "--model", "model",
            "--target-profile", "test", "--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    results = []
    for old in (True, False):
        events = []
        def publish(path, raw):
            events.append((path.name, raw))
            with path.open("xb") as stream:
                stream.write(raw)
            return True
        def backfill(result, cost):
            result["__prismaquant__"]["mtp_selection"]["mtp_expert_wires"] = ["雪"]
            return result
        ns = functions("reselect_mtp_fixed", old, {
            "add_serving_scope_arguments": lambda p: None,
            "add_mtp_scope_arguments": lambda p: p.set_defaults(mtp_routing=None, mtp_regime=None, mtp_tensor_parallel=1),
            "allocation_allowability_arguments": lambda p: None,
            "load_mtp_cost": lambda *a, **kw: SimpleNamespace(payload={"groups": {"group": {}}}, sha256="cost-sha"),
            "detect_profile": lambda m: None,
            "load_serving_profile": lambda p: SimpleNamespace(target_platform="test"),
            "serving_target_from_args": lambda *a, **kw: SimpleNamespace(context=lambda u: "scope"),
            "_mtp_scope_inputs": lambda *a, **kw: ({}, {}),
            "_mtp_rung_attestation": lambda *a: None, "allocation_rung_allowability": lambda a: None,
            "select_mtp_rungs": lambda *a, **kw: {"assignment": {"mtp": "BF16"},
                                                  "resident_bytes": 5, "rung": "é雪", "E": -0.0},
            "fr": SimpleNamespace(get_format=lambda fmt: SimpleNamespace(autoround_config=lambda: {"bits": 16})),
            "canonicalize_format": lambda v: "BF16",
            "whole_artifact_budget_from_assignment_payload": lambda *a, **kw: {"selection_tensor_payload_bytes": 10},
            "recursive_regular_file_bytes": lambda p: 12,
            "mtp_selection_rebased_bytes": lambda *a, **kw: {"mtp_source_bytes": 2, "total_bytes": 15},
            "unit_structure_from_profile": lambda n, p: n,
            "scope_provenance": lambda t, c: {"target": "test", "by_unit": {n: "scope" for n in c}},
            "selection_serving_lane_provenance": lambda a, **kw: {"by_unit": {n: "lane" for n in a}},
            "whole_artifact_budget_stamp": lambda **kw: kw,
            "backfill_mtp_selection_wires": backfill, "publish_new_bytes": publish})
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            assert ns["main"]() == 0
        raw = output.read_bytes()
        result = json.loads(raw)
        assert result["body"] == body["body"]
        assert result["__prismaquant__"]["whole_artifact_budget"]["selection_non_tensor_reserve_bytes"] == 4
        assert json.loads(stream.getvalue())["sha256"] == hashlib.sha256(raw).hexdigest()
        assert events == [(output.name, raw)]
        results.append((raw, stream.getvalue()))
        events.clear()
        with pytest.raises(SystemExit) as exc:
            ns["main"]()
        assert exc.value.code == 2 and not events and output.read_bytes() == raw
        output.unlink()
        argv[argv.index("--expect-body-sha256") + 1] = "wrong"
        with pytest.raises(SystemExit) as exc:
            ns["main"]()
        assert exc.value.code == 2 and not events and not output.exists()
        argv[argv.index("--expect-body-sha256") + 1] = hashlib.sha256(body_path.read_bytes()).hexdigest()
        ns["publish_new_bytes"] = lambda *a: False
        with pytest.raises(SystemExit) as exc:
            ns["main"]()
        assert exc.value.code == 2 and not output.exists()
        def changed(result, cost):
            result["body"] = {"bits": 4}
            return result
        ns["backfill_mtp_selection_wires"] = changed
        with pytest.raises(ValueError, match="changed a body format"):
            ns["main"]()
        assert not output.exists()
    assert results[0] == results[1]
