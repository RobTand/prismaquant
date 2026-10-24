"""Execute frozen research-base consumers independently of the digest owners."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant import artifact_registry, cluster_campaign, pipeline


ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/fixtures/digest_1301/orchestration-runtime/publications"
PAYLOADS = [
    {"z": [1, None, True], "a": {"unicode": "雪é", "escape": "\n\\\""}},
    {},
    {"nonfinite": [float("nan"), float("inf"), -float("inf")], "negative_zero": -0.0},
    {"unsupported": object()},
    {1: "integer", "a": "mixed keys"},
]


def original(module, fixture, function):
    # The entire original consumer function is executable, not an expression
    # reconstructed using a digest owner's recipe.
    namespace = dict(vars(module))
    exec(compile((FIXTURES / fixture).read_text(), fixture, "exec"), namespace)
    return namespace[function]


def outcome(call):
    try:
        return ("ok", call())
    except (TypeError, ValueError) as exc:
        return (type(exc).__name__, str(exc))


def spec_for(payload):
    return SimpleNamespace(
        to_dict=lambda: payload,
        validate=lambda: SimpleNamespace(errors=(), warnings=(), ok=True),
    )


@pytest.mark.parametrize("payload", PAYLOADS)
def test_pipeline_file_publication(tmp_path, payload):
    old = original(pipeline, "pipeline_write_pipeline_spec.py", "write_pipeline_spec")
    before, after = tmp_path / "old.json", tmp_path / "new.json"
    assert outcome(lambda: old(spec_for(payload), before)) == outcome(
        lambda: pipeline.write_pipeline_spec(spec_for(payload), after)
    )
    # Include partial files on rejection: dump streams before TypeError.
    assert before.read_bytes() == after.read_bytes()
    if "unsupported" not in payload and 1 not in payload:
        assert after.read_bytes().endswith(b"\n")
        assert b"\r\n" not in after.read_bytes()
        assert list(json.loads(after.read_bytes())) == sorted(payload)


@pytest.mark.parametrize("payload", PAYLOADS)
def test_pipeline_stdout_publication(monkeypatch, payload):
    monkeypatch.setattr(pipeline, "production_pipeline_spec_from_config", lambda **kw: spec_for(payload))
    old = original(pipeline, "pipeline_main.py", "main")
    outputs = []
    for call in (old, pipeline.main):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = outcome(lambda: call([]))
        outputs.append((result, stream.getvalue()))
    assert outputs[0] == outputs[1]
    if outputs[1][0][0] == "ok":
        assert outputs[1][1].endswith("\n")


@pytest.mark.parametrize("command", ["run", "status"])
@pytest.mark.parametrize("payload", PAYLOADS)
def test_campaign_publication(monkeypatch, command, payload):
    args = SimpleNamespace(command=command, manifest=Path("manifest.json"), state=Path("state.json"))
    monkeypatch.setattr(cluster_campaign, "_build_parser", lambda: SimpleNamespace(parse_args=lambda argv: args))
    monkeypatch.setattr(cluster_campaign, "load_campaign_manifest_v2", lambda path: {})
    monkeypatch.setattr(cluster_campaign, "run_campaign_v2", lambda manifest, state: payload)

    class Store:
        def __init__(self, path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def load(self, manifest):
            return payload

    monkeypatch.setattr(cluster_campaign, "_StateStore", Store)
    old = original(cluster_campaign, "cluster_campaign_main.py", "main")
    outputs = []
    for call in (old, cluster_campaign.main):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            result = outcome(lambda: call([]))
        outputs.append((result, stream.getvalue()))
    assert outputs[0] == outputs[1]
    if outputs[1][0][0] == "ok":
        assert outputs[1][1].endswith("\n")
        assert "雪" not in outputs[1][1]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_registry_publication(tmp_path, payload):
    old = original(artifact_registry, "artifact_registry_ArtifactRegistry__write.py", "_write")
    record = SimpleNamespace(to_dict=lambda: payload)
    paths = [tmp_path / "old.json", tmp_path / "new.json"]
    results = []
    for call, path in zip((old, artifact_registry.ArtifactRegistry._write), paths):
        # Atomic replace must keep an existing destination intact on failure.
        path.write_bytes(b"original\n")
        registry = object.__new__(artifact_registry.ArtifactRegistry)
        registry.path = path
        results.append(outcome(lambda: call(registry, [record])))
    assert results[0] == results[1]
    assert paths[0].read_bytes() == paths[1].read_bytes()
    assert not list(tmp_path.glob(".*.tmp"))
    if results[1][0] == "ok":
        assert paths[1].read_bytes().endswith(b"\n")
        assert b"\r\n" not in paths[1].read_bytes()
    else:
        assert paths[1].read_bytes() == b"original\n"


@pytest.mark.parametrize("payload", PAYLOADS)
@pytest.mark.parametrize("fail_write", [None, 4])
def test_registry_sink_trace(tmp_path, monkeypatch, payload, fail_write):
    """Compare partitions, partial tmp bytes, LF, flush/fsync/replace ordering."""
    import builtins
    import os

    old = original(artifact_registry, "artifact_registry_ArtifactRegistry__write.py", "_write")
    traces = []
    results = []
    for index, call in enumerate((old, artifact_registry.ArtifactRegistry._write)):
        trace = []
        path = tmp_path / str(index) / "registry.json"
        path.parent.mkdir()
        path.write_bytes(b"original\n")
        writes = 0

        class Sink:
            def __init__(self, target, mode):
                self.file = builtins.open(target, mode)
                self.target = target
                trace.append(("open", target.name, mode))

            def __enter__(self):
                return self

            def write(self, text):
                nonlocal writes
                writes += 1
                trace.append(("write", text))
                if writes == fail_write:
                    raise OSError("injected write failure")
                return self.file.write(text)

            def flush(self):
                trace.append(("flush",))
                self.file.flush()

            def fileno(self):
                trace.append(("fileno",))
                return self.file.fileno()

            def __exit__(self, *args):
                self.file.close()
                trace.append(("close", self.target.read_bytes()))

        def fsync(fd):
            trace.append(("fsync",))
            os.fsync(fd)

        def replace(src, dst):
            trace.append(("replace", src.name, dst.name))
            os.replace(src, dst)

        # Frozen functions have their own globals; patch both namespaces.
        globals_ = call.__globals__
        monkeypatch.setitem(globals_, "open", Sink)
        monkeypatch.setitem(globals_, "os", SimpleNamespace(fsync=fsync, replace=replace))
        monkeypatch.setitem(globals_, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="fixed")))
        registry = object.__new__(artifact_registry.ArtifactRegistry)
        registry.path = path
        try:
            result = outcome(lambda: call(registry, [SimpleNamespace(to_dict=lambda: payload)]))
        except OSError as exc:
            result = (type(exc).__name__, str(exc))
        results.append((result, path.read_bytes()))
        assert not list(path.parent.glob(".*.tmp"))
        traces.append(trace)
    assert results[0] == results[1]
    assert traces[0] == traces[1]
    assert len([event for event in trace if event[0] == "write"]) > 1
    if results[1][0][0] == "ok":
        assert trace[-5][0] == "flush"
        assert [event[0] for event in trace[-5:]] == ["flush", "fileno", "fsync", "close", "replace"]
        assert [event for event in trace if event[0] == "write"][-1] == ("write", "\n")
    else:
        assert not any(event[0] in ("fsync", "replace") for event in trace)
        assert results[1][1] == b"original\n"


def test_pipeline_real_spec_roundtrip(tmp_path):
    spec = pipeline.production_pipeline_spec_from_config()
    old = original(pipeline, "pipeline_write_pipeline_spec.py", "write_pipeline_spec")
    before, after = tmp_path / "old.json", tmp_path / "new.json"
    old(spec, before)
    pipeline.write_pipeline_spec(spec, after)
    assert before.read_bytes() == after.read_bytes()
    assert pipeline.load_pipeline_spec(after).to_dict() == spec.to_dict()


@pytest.mark.parametrize("payload", PAYLOADS)
@pytest.mark.parametrize("route", ["settings", "reuse", "main"])
def test_pipeline_indent1_gaps(tmp_path, monkeypatch, payload, route):
    monkeypatch.setattr(pipeline, "time", SimpleNamespace(time=lambda: 123))
    monkeypatch.setattr(pipeline, "stage_settings_document", lambda settings: payload)
    name = {"settings": "_record_stage_settings", "reuse": "_record_unverified_reuse", "main": "main"}[route]
    fixture = "pipeline_main.py" if route == "main" else "pipeline" + name + ".py"
    old = original(pipeline, fixture, name)
    observations = []
    for index, call in enumerate((old, getattr(pipeline, name))):
        path = tmp_path / str(index) / "settings.json"
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            if route == "settings":
                result = outcome(lambda: call(path, "stage", payload))
            elif route == "reuse":
                result = outcome(lambda: call(Path("artifact"), path, "stage", payload, "legacy"))
            else:
                result = outcome(lambda: call(["--write-stage-settings", str(path)]))
        observations.append((result, path.read_bytes() if path.exists() else None))
    assert observations[0] == observations[1]
    if observations[1][0][0] == "ok":
        assert observations[1][1].endswith(b"\n")
        if observations[1][1] != b"{}\n":
            assert b'\n "' in observations[1][1]
    else:
        assert observations[1][1] is None


def test_ledger_frozen_provenance():
    ledger = json.loads((FIXTURES / "ledger.json").read_text())
    frozen = json.loads((ROOT / "tests/fixtures/digest_parent_completion_1301.json").read_text())
    calls = {call["id"]: call for call in frozen["calls"]}
    assert len(ledger["publications"]) == 3
    assert len(ledger["recipe_gaps"]) == 5
    for entry in ledger["publications"] + ledger["recipe_gaps"]:
        call = calls[entry["original_id"]]
        assert entry["source_commit"] == frozen["source_commit"]
        assert entry["source_blob"] == call["source_blob"]
        # Snippets were extracted from the frozen blob; do not require git
        # history in installed or snapshot test environments.
        fixture = (ROOT / entry["fixtures"][0]).read_bytes()
        assert hashlib.sha256(fixture).hexdigest() == entry["fixture_sha256"]
        assert isinstance(ast.parse(fixture).body[0], ast.FunctionDef)
        if entry in ledger["publications"]:
            assert entry["owner_route"].startswith("prismaquant.digests.DIRECT_ASCII_")
        else:
            assert entry["reason"]
        assert entry["tests"]
