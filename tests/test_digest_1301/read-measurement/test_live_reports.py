"""Original-consumer snippet oracles; no fleet, GPU or publication claim.

Role report tails run as executable boundaries, not full producer publication.
The consumer integration below additionally exercises the real guarded reader.
"""
import ast
import contextlib
import hashlib
import inspect
import io
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from prismaquant import digests
from tools import band_serial_handoff_live_pair as band
from tools import joint_prepare_startup_probe as startup
from tools import profile_stage_b_head as profile
from tools import stagea_produced_live_cycle as cycle

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "tests/fixtures/digest_1301/read-measurement/live_reports_source.json"
SNIPPETS = json.loads(FIXTURE.read_text())["snippets"]


def snippet(scope):
    return next(s for s in SNIPPETS if scope in s["original_id"])


def original_print(scope, **values):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(snippet(scope)["SOURCE"], {"json": json, **values})
    return stream.getvalue()


@pytest.mark.parametrize("raw", [b"", b"\x00\xff\xc3\xa9", b'{ "z": 1, "a": 2 }\n'])
def test_byte_consumers(raw, tmp_path, monkeypatch):
    from prismaquant import autoscale, tessera_joint_aura

    events = []
    owner = digests.bytes_sha256hex
    monkeypatch.setattr(digests, "bytes_sha256hex",
                        lambda value: events.append(("hash", value)) or owner(value))
    oracle = {"hashlib": hashlib, "raw": raw}
    exec("def original():\n" + textwrap.indent(snippet("::_digest@")["SOURCE"], "    "), oracle)
    expected = oracle["original"]()
    assert cycle._digest(raw) == expected
    assert events == [("hash", raw)]
    path = tmp_path / "record.json"
    path.write_bytes(raw)
    startup_oracle = {"hashlib": hashlib, "plan_path": path}
    exec(snippet("joint_prepare_startup_probe.py::main@46")["SOURCE"], startup_oracle)
    assert startup_oracle["actual"] == expected
    events.clear()
    read = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda p: events.append(("read", p)) or read(p))
    with pytest.raises(SystemExit, match="record digest mismatch"):
        profile.load_record(path, "not-the-digest")
    assert events == [("read", path), ("hash", raw)]

    monkeypatch.setattr(autoscale, "require_bounded_capture_environment",
                        lambda env: events.append(("environment", None)))
    def loaded(*args, **kwargs):
        events.append(("load", None))
        raise RuntimeError("plan loader reached")
    monkeypatch.setattr(tessera_joint_aura, "load_joint_anchor_plan", loaded)
    events.clear()
    with pytest.raises(SystemExit, match="hashes to"):
        startup.main(["--plan", str(path), "--plan-sha256", "bad"])
    assert events == [("environment", None), ("read", path), ("hash", raw)]
    events.clear()
    with pytest.raises(RuntimeError, match="plan loader reached"):
        startup.main(["--plan", str(path), "--plan-sha256", expected])
    assert events == [("environment", None), ("read", path), ("hash", raw), ("load", None)]
    events.clear()
    with pytest.raises(RuntimeError, match="plan loader reached"):
        startup.main(["--plan", str(path)])
    assert events == [("environment", None), ("load", None)]


def test_record_acceptance_and_refusals(tmp_path):
    path = tmp_path / "record.json"
    record = {"schema": "prismaquant.tessera_joint_quantum.v1", "mode": "layer_campaign"}
    raw = json.dumps(record, indent=1).encode()
    path.write_bytes(raw)
    expected = eval(snippet("::load_record@")['expression'],
                    {"hashlib": hashlib, "raw": raw}).hexdigest()
    exec(snippet("::load_record@")["SOURCE"],
         {"hashlib": hashlib, "raw": raw, "sha256": expected, "path": path})
    with pytest.raises(SystemExit, match="record digest mismatch"):
        exec(snippet("::load_record@")["SOURCE"],
             {"hashlib": hashlib, "raw": raw, "sha256": "bad", "path": path})
    assert profile.load_record(path, expected) == record
    path.write_bytes(b"not json")
    with pytest.raises(ValueError):
        profile.load_record(path, hashlib.sha256(b"not json").hexdigest())
    # This reader does not validate schema/mode; preserve that acceptance.
    path.write_bytes(b"{}")
    assert profile.load_record(path, hashlib.sha256(b"{}").hexdigest()) == {}
    with pytest.raises(TypeError):
        cycle._digest("not bytes")


@pytest.mark.parametrize("role", ["producer_role", "consumer_role"])
@pytest.mark.parametrize("ok", [True, False])
def test_role_report_boundaries(role, ok, monkeypatch):
    # Execute the real terminal report block, including ok/return policy, with
    # precomputed checks. Publication and reconstruction are intentionally absent.
    fn = getattr(band, role)
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    body = tree.body[0].body
    start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == "out['checks']")
    tree.body[0].body = body[start:]
    tree.body[0].args = ast.arguments(posonlyargs=[], args=[ast.arg(arg="out"), ast.arg(arg="checks")], kwonlyargs=[], kw_defaults=[], defaults=[])
    ns = {}
    exec(compile(ast.fix_missing_locations(tree), "<actual-role-report-tail>", "exec"), ns)
    payload = {"z": Path("portable-path"), "unicode": "café", "nan": float("nan")}
    checks = {"read": ok}
    expected_payload = {**payload, "checks": checks, "ok": ok}
    expected = original_print("::" + role + "@", out=expected_payload)
    calls = []
    owner = digests.DIRECT_ASCII_SPACED_LAX_DEFAULT_STR
    monkeypatch.setattr(digests, "DIRECT_ASCII_SPACED_LAX_DEFAULT_STR",
                        SimpleNamespace(text=lambda v: calls.append(dict(v)) or owner.text(v)))
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        rc = ns[role](payload, checks)
    assert stream.getvalue() == expected
    assert rc == (0 if ok else 1)
    assert calls == [expected_payload]
    circular = []; circular.append(circular)
    with pytest.raises(ValueError, match="Circular"):
        ns[role]({"circular": circular}, checks)


def test_derive_report(tmp_path, monkeypatch, capsys):
    payload = {"z": ["é", float("inf")], "a": 1}
    calls = []
    monkeypatch.setattr(band, "derive", lambda *a: calls.append(a) or payload)
    args = SimpleNamespace(root=tmp_path, handoff="handoff", handoff_sha256="digest")
    expected = original_print("band_serial_handoff_live_pair.py::main@", args=args, derive=lambda *a: payload)
    assert band.main(["derive", "--root", str(tmp_path), "--handoff", "handoff",
                      "--handoff-sha256", "digest"]) == 0
    assert capsys.readouterr().out == expected
    assert calls == [(tmp_path, "handoff", "digest")]
    monkeypatch.setattr(band, "derive", lambda *a: {"bad": object()})
    with pytest.raises(TypeError):
        band.main(["derive", "--root", str(tmp_path), "--handoff", "h", "--handoff-sha256", "d"])
    assert capsys.readouterr().out == ""


def test_profile_report(tmp_path, monkeypatch, capsys):
    from prismaquant import io_spans, tessera_joint_aura
    events = []
    monkeypatch.setattr(profile, "load_record", lambda *a: events.append("record") or {
        "layer": 2, "campaign": {"prepared_path": "prepared", "prepared_sha256": "digest"}})
    monkeypatch.setattr(tessera_joint_aura, "load_joint_anchor_plan",
                        lambda *a, **k: events.append("plan") or {})
    monkeypatch.setattr(profile, "walk_intake", lambda *a, **k: events.append("walk") or {"extra": "é", "nan": float("nan")})
    monkeypatch.setattr(profile, "mount_ops", lambda: {})
    monkeypatch.setattr(io_spans, "read_proc_io", lambda: {})
    monkeypatch.setattr(io_spans, "read_proc_status", lambda: {})
    owner = digests.DIRECT_ASCII_SPACED_LAX
    reports = []
    monkeypatch.setattr(digests, "DIRECT_ASCII_SPACED_LAX", SimpleNamespace(
        text=lambda report: events.append("serialize") or reports.append(report) or owner.text(report)))
    assert profile.main(["--mode", "walk", "--quantum", "q", "--quantum-sha256", "qd",
                         "--plan", "p", "--plan-sha256", "pd", "--scratch", str(tmp_path)]) == 0
    assert events == ["record", "plan", "walk", "serialize"]
    assert capsys.readouterr().out == original_print("profile_stage_b_head.py::main@500", report=reports[0])


def test_frozen_registry_provenance():
    registry = json.loads((ROOT / "tests/fixtures/digest_parent_completion_1301.json").read_text())
    files = {s["file"] for s in SNIPPETS}
    calls = {r["id"]: r for r in registry["calls"] if r["file"] in files}
    assert set(calls) == {s["original_id"] for s in SNIPPETS}
    for source in SNIPPETS:
        row = calls[source["original_id"]]
        assert source["source_commit"] == registry["source_commit"]
        assert source["source_blob"] == row["source_blob"]
        assert source["file"] == row["file"]
        assert source["line"] == int(row["id"].rsplit("@", 1)[1].split(":")[0])
        expression = ast.parse(source["expression"], mode="eval").body
        syntax = ast.dump(expression, include_attributes=False)
        assert hashlib.sha256(syntax.encode()).hexdigest() == row["syntax_sha256"]
        assert ast.unparse(expression.func) == row["callee"]
        assert any(ast.dump(n, include_attributes=False) == syntax
                   for n in ast.walk(ast.parse(source["SOURCE"])) if isinstance(n, ast.Call))


def test_fixture_ledger_integrity():
    ledger = json.loads(Path(__file__).with_name("live_reports_ledger.json").read_text())["entries"]
    assert len({e["original_id"] for e in ledger}) == len(ledger) == 23
    assert sum(e["status"] == "replaced" for e in ledger) == 7
    for entry, source in zip(ledger, SNIPPETS):
        assert entry["fixtures"] == [
            f"{FIXTURE.relative_to(ROOT).as_posix()}#{source['original_id']}"]
        for key in ["original_id", "source_commit", "source_blob"]:
            assert entry[key] == source[key]
        assert source["source_commit"] == "6d8288adc94b3c4b624c143ed372fb6513f32784"
        assert len(source["source_blob"]) == 40
        assert len(source["SOURCE"].splitlines()) < 15
        if entry["status"] == "blocker":
            assert entry["owner_route"] is None and entry["blocker"]
        else:
            assert entry["tests"] and entry["fixtures"] and entry["owner_route"]
