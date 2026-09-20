"""Driver tests: per-host plans, strict verification, plain statuses."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = HERE.parent / "tools" / "fleet_qualification_driver.py"
KEY = "a" * 64


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


def _declared(driver=None):
    return {"pq_head": "d" * 40, "pb_rev": "c" * 40,
            "generation": "0467e9e2316c-1789881139-d2055704fb70"}


def _payload_path(text: str) -> tuple[str, bytes]:
    """Content-addressed CAS path for result text (existing protocol)."""
    data = text.encode()
    digest = hashlib.sha256(data).hexdigest()
    return f"/mnt/shared/cas/blobs/{digest[:2]}/{digest}", data


_STORE: dict[str, bytes] = {}


def _blob(key=KEY, payload_text="2 passed in 1s"):
    content = (f"header\n{payload_text}\n"
               + json.dumps({"receipt": {"action_key": key}}) + "\n")
    path, data = _payload_path(content)
    _STORE[path] = data
    return json.dumps({"payload_path": path,
                       "receipt": {"action_key": key},
                       "receipt_sha256": "e" * 64, "status": "published"})


def _shard(summary="2 passed in 1s", key=KEY, files=None,
           payload_text="2 passed in 1s"):
    return {"files": files or ["tests/test_x.py"],
            "output": f"pbrun: queued {key[:8]}\n{summary}\n"
                      f"{_blob(key, payload_text)}\n",
            "returncode": 0, "shard": 0, "summary": summary}


def _terminal(host="sparky", rc=0, key=KEY, parent="d" * 40,
              generation="0467e9e2316c-1789881139-d2055704fb70",
              side="executed"):
    return {"action_key": key, "status": side,
            "finished_host": host, "claimed_host": host,
            "detail": {"returncode": rc,
                       "stdout": "2 passed in 1s",
                       "argv": ["/usr/bin/python3",
                                "/mnt/shared/prismabuild-fleet/"
                                "runtime-generations/" + generation
                                + "/tools/resource_exec.py"]},
            "checkout_snapshot": {"commit": "f" * 40, "parent": parent}}


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


def test_full_chain_qualifies_with_plain_statuses(tmp_path, monkeypatch):
    """Matching action/source/generation/host qualifies; statuses stay plain."""
    driver = _driver()
    terminal = _terminal()
    monkeypatch.setattr(driver, "_read_bytes", _STORE.get)
    monkeypatch.setattr(driver, "_read_json_file",
                        lambda path: terminal)
    verdict = driver.verify_shard(shard=_shard(), host="sparky",
                                  declared=_declared())
    assert verdict["status"] == "qualified"
    assert verdict["action_key"] == KEY
    assert set(verdict) >= {"host", "status", "reason", "passed", "failed",
                            "skipped", "action_key", "terminal",
                            "snapshot_commit", "snapshot_parent",
                            "generation"}


def test_wrong_action_source_generation_or_host_never_qualifies(
        tmp_path, monkeypatch):
    """Every attribution break fails closed with its own reason."""
    driver = _driver()
    terminal = _terminal()

    def fake_read(path):
        return terminal

    monkeypatch.setattr(driver, "_read_bytes", _STORE.get)
    monkeypatch.setattr(driver, "_read_json_file", fake_read)
    base = {"shard": _shard(), "host": "sparky",
            "declared": _declared()}
    assert driver.verify_shard(**base)["status"] == "qualified"
    terminal["action_key"] = "b" * 64
    assert driver.verify_shard(**base)["status"] == "failed"
    terminal["action_key"] = KEY
    terminal["finished_host"] = "sparklina"
    assert "host" in driver.verify_shard(**base)["reason"]
    terminal["finished_host"] = "sparky"
    terminal["checkout_snapshot"]["parent"] = "0" * 40
    assert "source" in driver.verify_shard(**base)["reason"]
    terminal["checkout_snapshot"]["parent"] = "d" * 40
    terminal["detail"]["argv"][1] = "/elsewhere/x.py"
    assert "generation" in driver.verify_shard(**base)["reason"]


def test_empty_collection_and_replays_never_qualify(monkeypatch):
    """Zero passing tests and unattributable replays stay unqualified."""
    driver = _driver()
    monkeypatch.setattr(driver, "_read_bytes", _STORE.get)
    monkeypatch.setattr(driver, "_read_json_file", lambda path: None)
    empty = driver.verify_shard(
        shard=_shard(summary="1 skipped in 1s",
                     payload_text="1 skipped in 1s"),
        host="sparky", declared=_declared())
    assert empty["status"] == "nonqualified"
    replay = driver.verify_shard(
        shard={"files": ["tests/test_x.py"], "output": "2 passed in 1s",
               "returncode": 0, "shard": 0, "summary": "2 passed in 1s"},
        host="sparky", declared=_declared())
    assert replay["status"] == "nonqualified"
    malformed = driver.verify_shard(
        shard={"files": "nope", "output": None, "returncode": "0",
               "summary": None},
        host="sparky", declared=_declared())
    assert malformed["status"] == "failed"


def test_case_by_host_matrix_and_exit_codes():
    """Covered hosts need every file qualified; failures exit nonzero."""
    driver = _driver()
    plan = _plan(driver)
    both = []
    for host in ("sparky", "sparklina"):
        for name, files in driver.COMPONENT_CASES:
            both.append({"host": host, "file": files[0],
                         "status": "qualified", "reason": "",
                         "passed": 2, "failed": 0, "skipped": 0,
                         "action_key": KEY, "terminal": "/tmp/t.json",
                         "snapshot_commit": "f" * 40,
                         "snapshot_parent": "d" * 40,
                         "generation": "g"})
    report = driver.assemble_report(plan=plan, verdicts=both,
                                    started_unix=0.0)
    assert report["covered_hosts"] == ["sparklina", "sparky"]
    assert report["missing_hosts"] == []
    assert any("unimplemented leg" in r
               for r in report["incomplete_reasons"])
    assert report["complete"] is False
    assert driver.exit_code_for(report) == 0
    assert [c["name"] for c in report["cases"]][:2] == [
        "tests/test_fleet_acceptance_pins.py@sparky",
        "tests/test_fleet_acceptance_runner.py@sparky"]
    failed = [dict(both[0], status="failed", reason="rc=1")]
    report = driver.assemble_report(plan=plan, verdicts=failed,
                                    started_unix=0.0)
    assert report["missing_hosts"] == ["sparky", "sparklina"]
    assert driver.exit_code_for(report) == 1
