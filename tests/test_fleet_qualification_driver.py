"""Driver tests: real argv, unimplemented records, coverage arithmetic."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVER = HERE.parent / "tools" / "fleet_qualification_driver.py"


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


def _shard(host=None, rc=0, summary="2 passed in 1s",
           actions=(), payload=None):
    blob = ""
    if host is not None:
        blob += json.dumps({"hostname": host}) + "\n"
    for action in actions:
        blob += f"pbrun: queued {action[:8]} tags=['gb10']\n"
        blob += json.dumps({"action_key": action}) + "\n"
    if payload is not None:
        blob += json.dumps({"payload_path": payload,
                            "receipt_sha256": "e" * 64,
                            "status": "published"}) + "\n"
    return {"files": ["tests/test_x.py"], "output": blob,
            "returncode": rc, "shard": 0, "summary": summary}


def test_plan_is_deterministic_for_declared_inputs():
    """Same inputs in, same plan out: byte-identical JSON."""
    driver = _driver()
    first = _plan(driver)
    second = _plan(driver)
    assert json.dumps(first, sort_keys=True) == json.dumps(
        second, sort_keys=True)
    assert first["pins"] == _pins()
    assert first["schema"] == driver.PLAN_SCHEMA


def test_runnable_rows_carry_real_client_flags():
    """Rows invoke the published client with concrete bounds, no fiction."""
    driver = _driver()
    plan = _plan(driver)
    assert plan["required_hosts"] == ["sparky", "sparklina"]
    assert driver.HOSTS == ("sparky", "sparklina")
    for case in plan["cases"]:
        assert case["client"] == driver.SUPPORTED_CLIENT
        argv = case["argv"]
        assert argv[0].endswith("tools/pbtest.py"), argv
        assert "run" not in argv
        assert argv[argv.index("--tag") + 1] == "gb10"
        assert int(argv[argv.index("--priority") + 1]) == -10
        assert argv[argv.index("--json") + 1].startswith("/tmp/driver-out/")
        assert not any("{" in arg or "}" in arg for arg in argv), argv
        assert case["files"] and all(
            f.endswith(".py") for f in case["files"])


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
                if not line.startswith("# unimplemented")]
    assert not any("--help" in line for line in runnable)
    for record in plan["unimplemented"]:
        assert any(line.startswith("# unimplemented")
                   and record["name"] in line
                   for line in lines)


def test_host_coverage_marks_the_missing_box():
    """Attribution from shard receipts; uncovered hosts stay gaps."""
    driver = _driver()
    plan = _plan(driver)
    report = driver.assemble_report(
        plan=plan,
        receipts={"pins": [_shard(host="sparky", actions=["a" * 64],
                                  payload="/mnt/shared/x")]},
        started_unix=0.0)
    assert report["covered_hosts"] == ["sparky"]
    assert report["missing_hosts"] == ["sparklina"]
    assert any("sparklina" in reason
               for reason in report["incomplete_reasons"])
    assert report["complete"] is False


def test_failed_shards_fail_the_case_and_the_exit():
    """A nonzero shard is a failed case; clean runs exit zero, never complete."""
    driver = _driver()
    plan = _plan(driver)
    bad = driver.assemble_report(
        plan=plan,
        receipts={"pins": [_shard(host="sparky", rc=1,
                                  summary="1 failed in 1s")]},
        started_unix=0.0)
    assert bad["cases"][0]["status"] == "failed"
    assert driver.exit_code_for(bad) == 1
    good = driver.assemble_report(
        plan=plan,
        receipts={case["name"]: [_shard(host="sparky")]
                  for case in plan["cases"]},
        started_unix=0.0)
    assert all(c["status"] == "qualified" for c in good["cases"])
    assert good["complete"] is False
    assert good["unimplemented"]
    assert driver.exit_code_for(good) == 0


def test_parse_shard_reads_receipt_facts():
    """Action keys, payloads, hosts come from the shard record itself."""
    driver = _driver()
    parsed = driver.parse_shard(
        _shard(host="sparklina", actions=["f" * 64],
               payload="/mnt/shared/cas/blobs/aa/x"))
    assert parsed["hosts"] == ["sparklina"]
    assert parsed["actions"] == ["f" * 64]
    assert parsed["payload_paths"] == ["/mnt/shared/cas/blobs/aa/x"]
    assert parsed["published"] is True
    assert parsed["counts"]["passed"] == 2
