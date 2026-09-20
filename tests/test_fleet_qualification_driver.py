"""Driver tests: per-host plans, verified receipts, exact completeness."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = HERE.parent / "tools" / "fleet_qualification_driver.py"
FIXTURES = HERE / "fixtures" / "cas-r5"

#: Real filed actions (tiny pins probes through PB's machinery).
KEY_A = "c74acd1a38ae4aa13e182b8625fbd403e2ed1b4d2ef4b0abc24c98114ea80505"
KEY_B = "554fa0c61f3472a609e9eba92d3371cb0de4964b00151cb381c9fff3ec5451b1"
KEY_C = "1ad62808565187698210a9bff83be35ad501e733511981d9b52cddae4dca812d"
GEN = "0467e9e2316c-1789881139-d2055704fb70"


def _driver():
    spec = importlib.util.spec_from_file_location(
        "fleet_qualification_driver", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pins():
    return {"pb_candidate": {"rev": "c" * 40}, "pq": {"head": "d" * 40}}


def _plan(driver, out_dir="/tmp/driver-out"):
    return driver.build_plan(pins=_pins(), checkout="/tmp/checkout",
                             python="/tmp/python", out_dir=out_dir)


def _declared(head="d" * 40, generation=GEN):
    return {"pq_head": head, "pb_rev": "c" * 40, "generation": generation}


def _tmp_cas(driver, tmp_path, tag):
    """Real filed artifacts materialized as a tmp CAS tree (read-only)."""
    src = FIXTURES / tag
    key = {"A": KEY_A, "B": KEY_B, "C": KEY_C}[tag]
    root = tmp_path / f"cas-{tag}"
    (root / "requests" / key[:2]).mkdir(parents=True)
    (root / "actions" / "v3" / key[:2]).mkdir(parents=True)
    request = (src / "request.json").read_bytes()
    receipt = json.loads((src / "receipt.json").read_bytes())
    digest = receipt["result"]["sha256"]
    (root / "blobs" / digest[:2]).mkdir(parents=True)
    blob = (src / "result.bin").read_bytes()
    (root / "requests" / key[:2] / f"{key}.json").write_bytes(request)
    (root / "actions" / "v3" / key[:2] / f"{key}.json").write_bytes(
        json.dumps(receipt).encode())
    (root / "blobs" / digest[:2] / digest).write_bytes(blob)
    for path in root.rglob("*"):
        if path.is_file():
            os.chmod(path, 0o444)
    return root, key


def _terminal(key, host="sparklina", rc=0, parent="d" * 40):
    return {"action_key": key, "status": "executed",
            "finished_host": host, "claimed_host": host,
            "detail": {"returncode": rc, "stdout": "", "argv": []},
            "checkout_snapshot": {"commit": "f" * 40, "parent": parent}}


def _shard(key, files=None):
    return {"files": files or ["tests/test_x.py"],
            "output": f'pbrun: queued {key[:8]}\n'
                      f'{{"action_key": "{key}"}}\n',
            "returncode": 0, "shard": 0, "summary": "squinted"}


def test_plan_is_deterministic_for_declared_inputs():
    """Same inputs in, same plan out: byte-identical JSON."""
    driver = _driver()
    first = _plan(driver)
    second = _plan(driver)
    assert json.dumps(first, sort_keys=True) == json.dumps(
        second, sort_keys=True)
    assert first["pins"] == _pins()
    assert first["schema"] == driver.PLAN_SCHEMA


def test_runs_cover_both_hosts_with_real_client_flags():
    """One tagged pbtest row per required host; placement stays fleet-owned."""
    driver = _driver()
    plan = _plan(driver)
    assert plan["required_hosts"] == ["sparky", "sparklina"]
    assert [run["host"] for run in plan["runs"]] == ["sparky", "sparklina"]
    for run in plan["runs"]:
        assert run["client"] == driver.SUPPORTED_CLIENT
        argv = run["argv"]
        assert argv[0].endswith("tools/pbtest.py"), argv
        assert "run" not in argv
        assert argv[argv.index("--tag") + 1] == run["host"]
        assert int(argv[argv.index("--priority") + 1]) == -10
        assert argv[argv.index("--json") + 1].startswith("/tmp/driver-out/")
        assert not any("{" in arg or "}" in arg for arg in argv), argv
        assert "host" not in [a.lstrip("-") for a in argv
                              if a.startswith("--")], argv


def test_unimplemented_legs_are_records_without_invocations():
    """Membership/native-container legs never get runnable commands."""
    driver = _driver()
    plan = _plan(driver)
    names = {r["name"] for r in plan["unimplemented"]}
    assert any("membership" in n for n in names)
    assert any("native-container" in n for n in names)
    for record in plan["unimplemented"]:
        assert record["reason"]
        assert "argv" not in record
    lines = driver.format_invocations(plan)
    runnable = [line for line in lines
                if not line.startswith("# unimplemented")
                and not line.startswith("# host")]
    assert not any("--help" in line for line in runnable)
    for record in plan["unimplemented"]:
        assert any(line.startswith("# unimplemented")
                   and record["name"] in line
                   for line in lines)


def test_valid_chain_qualifies_with_receipt_evidence(tmp_path, monkeypatch):
    """Filed action + verified receipt + result + terminal all agree."""
    driver = _driver()
    root, key = _tmp_cas(driver, tmp_path, "A")
    monkeypatch.setattr(driver, "_CAS_ROOT", root)
    terminal = _terminal(key)
    monkeypatch.setattr(driver, "_read_terminal",
                        lambda k: (terminal, "done"))
    verdict = driver.verify_shard(shard=_shard(key), host="sparklina",
                                  declared=_declared())
    assert verdict["status"] == "qualified", verdict["reason"]
    assert verdict["action_key"] == key
    assert verdict["generation"] == GEN
    assert verdict["snapshot_parent"] == "d" * 40
    assert len(verdict["receipt_sha256"]) == 64
    assert verdict["passed"] == 2


def test_substituted_payload_fails_integrity(tmp_path, monkeypatch):
    """Valid bytes from another action do not verify under this receipt."""
    driver = _driver()
    root, key = _tmp_cas(driver, tmp_path, "A")
    other = (FIXTURES / "B" / "result.bin").read_bytes()
    digest = json.loads(
        (FIXTURES / "A" / "receipt.json").read_bytes())["result"]["sha256"]
    target = root / "blobs" / digest[:2] / digest
    os.chmod(target, 0o644)
    target.write_bytes(other)
    os.chmod(target, 0o444)
    monkeypatch.setattr(driver, "_CAS_ROOT", root)
    monkeypatch.setattr(driver, "_read_terminal",
                        lambda k: (_terminal(key), "done"))
    verdict = driver.verify_shard(shard=_shard(key), host="sparklina",
                                  declared=_declared())
    assert verdict["status"] == "failed"
    assert "receipt verification" in verdict["reason"]


def test_swapped_receipt_fails_action_binding(tmp_path, monkeypatch):
    """Another action's receipt at this path mismatches the request."""
    driver = _driver()
    root, key = _tmp_cas(driver, tmp_path, "A")
    receipt_path = root / "actions" / "v3" / key[:2] / f"{key}.json"
    os.chmod(receipt_path, 0o644)
    receipt_path.write_bytes((FIXTURES / "B" / "receipt.json").read_bytes())
    os.chmod(receipt_path, 0o444)
    monkeypatch.setattr(driver, "_CAS_ROOT", root)
    monkeypatch.setattr(driver, "_read_terminal",
                        lambda k: (_terminal(key), "done"))
    verdict = driver.verify_shard(shard=_shard(key), host="sparklina",
                                  declared=_declared())
    assert verdict["status"] == "failed"


def test_corrupt_receipt_fails(tmp_path, monkeypatch):
    """A flipped receipt byte breaks canonical receipt verification."""
    driver = _driver()
    root, key = _tmp_cas(driver, tmp_path, "A")
    receipt_path = root / "actions" / "v3" / key[:2] / f"{key}.json"
    raw = bytearray(receipt_path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    os.chmod(receipt_path, 0o644)
    receipt_path.write_bytes(bytes(raw))
    os.chmod(receipt_path, 0o444)
    monkeypatch.setattr(driver, "_CAS_ROOT", root)
    monkeypatch.setattr(driver, "_read_terminal",
                        lambda k: (_terminal(key), "done"))
    verdict = driver.verify_shard(shard=_shard(key), host="sparklina",
                                  declared=_declared())
    assert verdict["status"] == "failed"


def test_wrong_generation_or_source_never_qualifies(tmp_path, monkeypatch):
    """Declared pins that disagree with receipt/terminal evidence fail."""
    driver = _driver()
    root, key = _tmp_cas(driver, tmp_path, "A")
    monkeypatch.setattr(driver, "_CAS_ROOT", root)
    terminal = _terminal(key)
    monkeypatch.setattr(driver, "_read_terminal",
                        lambda k: (terminal, "done"))
    base = {"shard": _shard(key), "host": "sparklina"}
    assert driver.verify_shard(declared=_declared(), **base)["status"] \
        == "qualified"
    assert "generation" in driver.verify_shard(
        declared=_declared(generation="f" * 40), **base)["reason"]
    assert "source" in driver.verify_shard(
        declared=_declared(head="0" * 40), **base)["reason"]


def test_wrong_host_or_failed_terminal_never_qualifies(tmp_path,
                                                        monkeypatch):
    """Terminal host and executed status bind the verdict."""
    driver = _driver()
    root, key = _tmp_cas(driver, tmp_path, "A")
    monkeypatch.setattr(driver, "_CAS_ROOT", root)
    terminal = _terminal(key, host="sparky")
    monkeypatch.setattr(driver, "_read_terminal",
                        lambda k: (terminal, "done"))
    assert "host" in driver.verify_shard(
        shard=_shard(key), host="sparklina",
        declared=_declared())["reason"]
    terminal["finished_host"] = "sparklina"
    terminal["status"] = "failed"
    verdict = driver.verify_shard(shard=_shard(key), host="sparklina",
                                  declared=_declared())
    assert verdict["status"] == "failed"


def test_empty_collection_never_qualifies(tmp_path, monkeypatch):
    """A collect-only result has no outcome: unqualified, not green."""
    driver = _driver()
    root, key = _tmp_cas(driver, tmp_path, "C")
    monkeypatch.setattr(driver, "_CAS_ROOT", root)
    terminal = _terminal(key)
    monkeypatch.setattr(driver, "_read_terminal",
                        lambda k: (terminal, "done"))
    verdict = driver.verify_shard(shard=_shard(key), host="sparky",
                                  declared=_declared())
    assert verdict["status"] in ("nonqualified", "failed"), verdict


def test_malformed_records_and_replays_rejected(monkeypatch):
    """Garbage shapes and keyless replays never reach verification."""
    driver = _driver()
    declared = _declared()
    assert driver.verify_shard(
        shard={"files": "nope", "output": None, "returncode": "0",
               "summary": None},
        host="sparky", declared=declared)["status"] == "failed"
    replay = driver.verify_shard(
        shard={"files": ["tests/test_x.py"], "output": "2 passed in 1s",
               "returncode": 0, "shard": 0, "summary": "2 passed in 1s"},
        host="sparky", declared=declared)
    assert replay["status"] == "nonqualified"


def test_summary_counts_cover_singular_plural_and_usage():
    """Anchored summary stems: pass/fail/skip/error, both numbers."""
    driver = _driver()
    pbcore, pbtest_mod = driver._published()
    assert pbtest_mod.pytest_summary(
        ["1 failed, 531 passed, 1 skipped in 17.82s"]) == \
        "1 failed, 531 passed, 1 skipped in 17.82s"
    assert pbtest_mod.pytest_summary(
        ["python -m pytest: error: unrecognized arguments: x"]) == ""
    counts = driver._summary_counts("1 failed, 531 passed, 1 skipped")
    assert (counts["failed"], counts["passed"],
            counts["skipped"]) == (1, 531, 1)
    assert driver._summary_counts("2 errors")["error"] == 2
    assert driver._summary_counts("1 error")["error"] == 1
    assert driver._summary_counts("2 passed, 3 warnings in 0.5s") == \
        {"passed": 2, "failed": 0, "skipped": 0, "error": 0}


def test_completeness_regression_flags_a_dropped_case():
    """Five planned files minus one is a named gap, not silent coverage."""
    driver = _driver()
    plan = _plan(driver)
    verdicts = []
    for host in ("sparky", "sparklina"):
        for name, files in driver.COMPONENT_CASES:
            if (name, host) == ("matrix", "sparklina"):
                continue
            verdicts.append({"host": host, "file": files[0],
                             "status": "qualified", "reason": "",
                             "passed": 2, "failed": 0, "skipped": 0,
                             "action_key": "a" * 64,
                             "terminal": "/tmp/t.json",
                             "snapshot_commit": "f" * 40,
                             "snapshot_parent": "d" * 40,
                             "generation": GEN, "receipt_sha256": "e" * 64})
    report = driver.assemble_report(plan=plan,
                                    verdicts=driver.complete_cases(
                                        plan, verdicts),
                                    started_unix=0.0)
    assert report["covered_hosts"] == ["sparky"]
    assert report["missing_hosts"] == ["sparklina"]
    dropped = [c for c in report["cases"]
               if c["name"] == "tests/test_fleet_acceptance_matrix.py"
               "@sparklina"]
    assert dropped and dropped[0]["status"] == "nonqualified"
    assert "missing case file" in dropped[0]["reason"]
    assert report["complete"] is False
    assert driver.exit_code_for(report) == 0
