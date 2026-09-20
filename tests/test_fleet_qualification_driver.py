"""Driver tests: real client flags, detach/terminal parsing, coverage."""
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


def _detach_line(key="a" * 64, status="submitted"):
    return json.dumps({
        "schema": "prismaquant.prismabuild.pbrun_detach.v1",
        "action_key": key,
        "done": f"/mnt/shared/pb-queue/done/{key}.json",
        "failed": f"/mnt/shared/pb-queue/failed/{key}.json",
        "status": status})


def _terminal(host="sparky", rc=0, summary="2 passed in 1s",
              payload="/mnt/shared/cas/blobs/aa/x"):
    return {"status": "executed",
            "finished_host": host, "claimed_host": host,
            "detail": {"returncode": rc,
                       "stdout": f"collected stuff\n{summary}\n"
                                 + json.dumps({"payload_path": payload})}}


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
    """Rows invoke pbrun with concrete bounds; placement stays fleet-owned."""
    driver = _driver()
    plan = _plan(driver)
    assert plan["required_hosts"] == ["sparky", "sparklina"]
    assert driver.HOSTS == ("sparky", "sparklina")
    for case in plan["cases"]:
        assert case["client"] == driver.SUPPORTED_CLIENT
        argv = case["argv"]
        assert argv[0].endswith("tools/pbrun.py"), argv
        assert "run" not in argv
        assert argv[argv.index("--tag") + 1] == "gb10"
        assert int(argv[argv.index("--priority") + 1]) == -10
        assert "--cwd" in argv and "--demand" in argv
        assert "--cpus" in argv and "--detach" in argv
        assert "--wait-s" in argv and "--env" in argv
        dash = argv.index("--")
        rest = argv[dash + 1:]
        assert rest[0] == "/tmp/python"
        assert rest[1:3] == ["-m", "pytest"]
        assert rest[3:] and all(f.endswith(".py") for f in rest[3:])
        assert not any("{" in arg or "}" in arg for arg in argv), argv
        assert "host" not in [a.lstrip("-") for a in argv], argv


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


def test_detach_and_terminal_parse_to_attributable_facts():
    """Action key, paths, host, counts come from the artifacts themselves."""
    driver = _driver()
    detach = driver.parse_detach(
        "pbrun: queued aaaa1111 tags=['gb10']\n" + _detach_line())
    assert detach["action_key"] == "a" * 64
    assert detach["done"].endswith(".json")
    assert detach["status"] == "submitted"
    term = driver.parse_terminal(_terminal(host="sparklina"), side="done")
    assert term["host"] == "sparklina"
    assert term["returncode"] == 0
    assert term["counts"]["passed"] == 2
    assert term["payload_paths"] == ["/mnt/shared/cas/blobs/aa/x"]


def test_host_coverage_marks_the_missing_box():
    """Only terminal-attested hosts count; uncovered hosts stay gaps."""
    driver = _driver()
    plan = _plan(driver)
    detach = driver.parse_detach(_detach_line())
    cases = [driver._case_from_terminal(
        "pins", detach,
        driver.parse_terminal(_terminal(host="sparky"), side="done"))]
    report = driver.assemble_report(plan=plan, cases=cases,
                                    started_unix=0.0)
    assert report["covered_hosts"] == ["sparky"]
    assert report["missing_hosts"] == ["sparklina"]
    assert any("sparklina" in reason
               for reason in report["incomplete_reasons"])
    assert report["complete"] is False


def test_failed_terminals_fail_the_case_and_the_exit():
    """A failed terminal is a failed case; clean runs exit zero, never complete."""
    driver = _driver()
    plan = _plan(driver)
    detach = driver.parse_detach(_detach_line())
    bad_term = driver.parse_terminal(
        _terminal(rc=1, summary="1 failed in 1s"), side="failed")
    bad = driver.assemble_report(
        plan=plan,
        cases=[driver._case_from_terminal("pins", detach, bad_term)],
        started_unix=0.0)
    assert bad["cases"][0]["status"] == "failed"
    assert driver.exit_code_for(bad) == 1
    good_term = driver.parse_terminal(_terminal(), side="done")
    good = driver.assemble_report(
        plan=plan,
        cases=[driver._case_from_terminal(c["name"], detach, good_term)
               for c in plan["cases"]],
        started_unix=0.0)
    assert all(c["status"] == "qualified" for c in good["cases"])
    assert good["complete"] is False
    assert good["unimplemented"]
    assert driver.exit_code_for(good) == 0
