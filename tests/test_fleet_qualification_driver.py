"""Driver tests: per-host plans, verified receipts, exact completeness."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
DRIVER = HERE.parent / "tools" / "fleet_qualification_driver.py"
FIXTURES = HERE / "fixtures" / "cas-r5"

#: Real filed actions (tiny pins probes through PB's machinery).
KEY_A = "c74acd1a38ae4aa13e182b8625fbd403e2ed1b4d2ef4b0abc24c98114ea80505"
KEY_B = "554fa0c61f3472a609e9eba92d3371cb0de4964b00151cb381c9fff3ec5451b1"
KEY_C = "1ad62808565187698210a9bff83be35ad501e733511981d9b52cddae4dca812d"
KEY_D = "d11c8a497ffe9866f391c0f7d53ebcfe56aa713b98ac312a1e6913660f201d30"
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


REAL_PYTHON = "/home/rob/venvs/pq846-pb461728e4/bin/python"
PINS_FILE = "tests/test_fleet_acceptance_pins.py"
LEVEL1_FILE = "tests/test_fleet_acceptance_level1.py"
RUNNER_FILE = "tests/test_fleet_acceptance_runner.py"
INPUT_A = "35f2fca765407ad485b0da2778ffbceb7ca1de570154e5626fda99b704fc4cc9"
INPUT_D = "d19e26fc5a9f6fd84dabad9adb9024f89db3b07527b38333c77cd239694e69b2"
GEN_NEW = "495461327539-1789901949-c5b033dd06ba"


def _declared(head="d" * 40, generation=GEN, python=REAL_PYTHON):
    return {"pq_head": head, "pb_rev": "c" * 40, "generation": generation,
            "python": python}


def _tmp_cas(driver, tmp_path, tag):
    """Real filed artifacts materialized as a tmp CAS tree (read-only)."""
    src = FIXTURES / tag
    key = {"A": KEY_A, "B": KEY_B, "C": KEY_C, "D": KEY_D}[tag]
    root = tmp_path / f"cas-{tag}"
    (root / "requests" / key[:2]).mkdir(parents=True)
    (root / "actions" / "v3" / key[:2]).mkdir(parents=True)
    request = (src / "request.json").read_bytes()
    receipt = json.loads((src / "receipt.json").read_bytes())
    digest = receipt["result"]["sha256"]
    (root / "blobs" / digest[:2]).mkdir(parents=True)
    blob = (src / "result.bin").read_bytes()
    (root / "requests" / key[:2] / f"{key}.json").write_bytes(request)
    # Receipt bytes must stay byte-identical: the verifier checks the
    # filed bytes are canonical JSON, and any reserialization breaks it.
    shutil.copyfile(src / "receipt.json",
                    root / "actions" / "v3" / key[:2] / f"{key}.json")
    (root / "blobs" / digest[:2] / digest).write_bytes(blob)
    for path in root.rglob("*"):
        if path.is_file():
            os.chmod(path, 0o444)
    return root, key


def _terminal(key, host="sparklina", rc=0, parent="d" * 40,
              input_sha=INPUT_A):
    return {"action_key": key, "status": "executed",
            "finished_host": host, "claimed_host": host,
            "detail": {"returncode": rc, "stdout": "", "argv": []},
            "checkout_snapshot": {
                "commit": "f" * 40, "parent": parent,
                "input": {"id": "pbrun.checkout-snapshot",
                          "sha256": input_sha}}}


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
                                  expected_file=PINS_FILE,
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
                                  expected_file=PINS_FILE,
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
                                  expected_file=PINS_FILE,
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
                                  expected_file=PINS_FILE,
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
    base = {"shard": _shard(key), "host": "sparklina",
            "expected_file": PINS_FILE}
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
        expected_file=PINS_FILE,
        declared=_declared())["reason"]
    terminal["finished_host"] = "sparklina"
    terminal["status"] = "failed"
    verdict = driver.verify_shard(shard=_shard(key), host="sparklina",
                                  expected_file=PINS_FILE,
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
    terminal = _terminal(key, host="sparky")
    monkeypatch.setattr(driver, "_read_terminal",
                        lambda k: (terminal, "done"))
    verdict = driver.verify_shard(shard=_shard(key), host="sparky",
                                  expected_file=PINS_FILE,
                                  declared=_declared())
    assert verdict["status"] in ("nonqualified", "failed"), verdict


def test_malformed_records_and_replays_rejected(monkeypatch):
    """Garbage shapes and keyless replays never reach verification."""
    driver = _driver()
    declared = _declared()
    assert driver.verify_shard(
        shard={"files": "nope", "output": None, "returncode": "0",
               "summary": None},
        host="sparky", expected_file=PINS_FILE,
        declared=declared)["status"] == "failed"
    replay = driver.verify_shard(
        shard={"files": ["tests/test_x.py"], "output": "2 passed in 1s",
               "returncode": 0, "shard": 0, "summary": "2 passed in 1s"},
        host="sparky", expected_file=PINS_FILE,
        declared=declared)
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
    for host_idx, host in enumerate(("sparky", "sparklina")):
        for idx, (name, files) in enumerate(driver.COMPONENT_CASES):
            if (name, host) == ("matrix", "sparklina"):
                continue
            verdicts.append({"host": host, "file": files[0],
                             "status": "qualified", "reason": "",
                             "passed": 2, "failed": 0, "skipped": 0,
                             "action_key": f"{host_idx:x}{idx:x}" + "b" * 62,
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


def _terminal_for(driver, tmp_path, monkeypatch, tag, key, host,
                  head="d" * 40):
    """Inline terminal mirroring the real record shape for one fixture."""
    inputs = {"A": INPUT_A, "D": INPUT_D}
    terminal = _terminal(key, host=host, parent=head,
                         input_sha=inputs[tag])
    monkeypatch.setattr(driver, "_read_terminal",
                        lambda k: (terminal, "done"))
    return terminal


def _fixture_action(tag):
    import json as _json
    return _json.loads((FIXTURES / tag / "request.json").read_bytes())


def test_canonical_script_matches_filed_actions():
    """Rebuilt publisher expression equals real sealed scripts byte-wise."""
    driver = _driver()
    for tag in ("A", "B", "C", "D"):
        action = _fixture_action(tag)
        expected, problem = driver._canonical_script(action)
        assert problem == "", (tag, problem)
        assert expected == action["task"]["argv"][4], tag
    assert driver._canonical_script(None)[0] is None
    assert driver._canonical_script({})[0] is None


def test_canonical_script_rejects_unexecuted_text():
    """Echo/branch/substitution text breaks canonical equality."""
    driver = _driver()
    action = _fixture_action("A")
    script = action["task"]["argv"][4]
    head = script.rpartition(" 2>&1 | tee ")[0]
    tampered = dict(action)
    tampered["task"] = dict(action["task"])
    tampered["task"]["argv"] = list(action["task"]["argv"])
    # Appended echo is not part of the sealed command.
    tampered["task"]["argv"][4] = (
        head + "; echo " + RUNNER_FILE + " 2>&1 | tee log.txt; "
        "exit ${PIPESTATUS[0]}")
    expected, _ = driver._canonical_script(tampered)
    assert expected != tampered["task"]["argv"][4]
    # Command substitution where the canonical form has plain words.
    substituted = dict(tampered["task"])
    substituted["argv"] = list(tampered["task"]["argv"])
    substituted["argv"][4] = head.replace(
        "/home/rob/venvs/pq846-pb461728e4/bin/python",
        "$(echo /home/rob/venvs/pq846-pb461728e4/bin/python)", 1) \
        + " 2>&1 | tee log.txt; exit ${PIPESTATUS[0]}"
    tampered2 = dict(action, task=substituted)
    expected2, _ = driver._canonical_script(tampered2)
    assert expected2 != substituted["argv"][4]
    # A shell comment carrying the filename is not canonical either.
    commented = dict(tampered["task"])
    commented["argv"] = list(tampered["task"]["argv"])
    commented["argv"][4] = (
        head + " # " + RUNNER_FILE + " 2>&1 | tee log.txt; "
        "exit ${PIPESTATUS[0]}")
    tampered3 = dict(action, task=commented)
    expected3, _ = driver._canonical_script(tampered3)
    assert expected3 != commented["argv"][4]


def test_command_operands_accepts_both_published_forms():
    """Structured operands from real filed actions, both entries."""
    driver = _driver()
    for tag, expected, entry in (("A", [PINS_FILE], "plain"),
                                 ("B", [PINS_FILE], "plain"),
                                 ("C", [PINS_FILE], "plain"),
                                 ("D", [LEVEL1_FILE], "guard")):
        operands, problem = driver._command_operands(
            _fixture_action(tag))
        assert problem == "", (tag, problem)
        assert operands is not None
        assert operands["files"] == expected, (tag, operands)
        assert operands["entry"] == entry, (tag, operands)
        assert operands["interpreter"] == REAL_PYTHON, (tag, operands)


def test_command_operands_rejects_extra_operands():
    """Extra tests or options beyond the declared case never parse."""
    driver = _driver()
    action = _fixture_action("A")
    command = list(action["params"]["command"])
    extra = dict(action, params=dict(action["params"]))
    extra["params"]["command"] = command + [RUNNER_FILE]
    operands, problem = driver._command_operands(extra)
    assert operands is not None
    assert operands["files"] == [PINS_FILE, RUNNER_FILE]
    flagged = dict(action, params=dict(action["params"]))
    flagged["params"]["command"] = command + ["-k", "foo"]
    assert driver._command_operands(flagged)[0] is None
    assert driver._command_operands(
        {"params": {"command": ["x"]}})[0] is None
    assert driver._command_operands(None)[0] is None


def test_guard_bytes_reject_fake_entry():
    """Only the bound generation's exact guard program is admitted."""
    driver = _driver()
    action = _fixture_action("D")
    operands, _ = driver._command_operands(action)
    assert operands is not None
    assert driver._guard_bytes_ok(operands["guard"], GEN_NEW) is True
    fake = operands["guard"].replace(
        "2 passed", "5 passed", 1) if "2 passed" in operands["guard"] \
        else "print('5 passed in 1s')"
    assert driver._guard_bytes_ok(fake, GEN_NEW) is False
    assert driver._guard_bytes_ok("", GEN_NEW) is False
    assert driver._guard_bytes_ok(operands["guard"], "no-such-gen") is False


@pytest.mark.parametrize("where,assignment", [
    ("inner", "PYTEST_ADDOPTS=-k one_test"),
    ("sealed", "PYTEST_ADDOPTS=-k one_test"),
    ("inner", "PYTHONPATH=/unrelated/source"),
    ("sealed", "PYTHONOPTIMIZE=1"),
    ("inner", "OMP_NUM_THREADS=8"),
    ("inner", "PRISMABUILD_TEST_TIMEOUT_S=nan"),
])
def test_environment_cannot_reduce_declared_case(where, assignment):
    """A matching file operand does not authorize a different test scope."""
    driver = _driver()
    action = _fixture_action("D")
    if where == "inner":
        command = action["params"]["command"]
        name = assignment.split("=", 1)[0] + "="
        found = next((i for i, word in enumerate(command)
                      if word.startswith(name)), None)
        if found is None:
            command.insert(1, assignment)
        else:
            command[found] = assignment
    else:
        name, value = assignment.split("=", 1)
        action["environment"]["variables"][name] = value
    # Canonical quoting is satisfied; the environment must still be checked.
    script, problem = driver._canonical_script(action)
    assert not problem
    action["task"]["argv"][4] = script
    operands, problem = driver._command_operands(action)
    assert operands is None, "altered selection/import/thread environment qualified"
    assert problem


def test_relabeled_file_fails_binding(tmp_path, monkeypatch):
    """A pins receipt relabeled as the runner file must not qualify."""
    driver = _driver()
    _tmp_cas(driver, tmp_path, "A")
    monkeypatch.setattr(driver, "_CAS_ROOT",
                        tmp_path / "cas-A")
    _terminal_for(driver, tmp_path, monkeypatch, "A", KEY_A, "sparklina")
    verdict = driver.verify_shard(
        shard=_shard(KEY_A), host="sparklina",
        expected_file=RUNNER_FILE, declared=_declared())
    assert verdict["status"] == "failed"
    assert "does not execute" in verdict["reason"]


def test_swapped_pair_and_unrelated_action_fail(tmp_path, monkeypatch):
    """Cross-labeled genuine receipts fail; matching labels qualify."""
    driver = _driver()
    _tmp_cas(driver, tmp_path, "A")
    _tmp_cas(driver, tmp_path, "D")
    monkeypatch.setattr(driver, "_CAS_ROOT", tmp_path / "cas-A")
    _terminal_for(driver, tmp_path, monkeypatch, "A", KEY_A, "sparklina")
    assert driver.verify_shard(
        shard=_shard(KEY_A), host="sparklina",
        expected_file=RUNNER_FILE,
        declared=_declared())["status"] == "failed"
    monkeypatch.setattr(driver, "_CAS_ROOT", tmp_path / "cas-D")
    _terminal_for(driver, tmp_path, monkeypatch, "D", KEY_D, "sparky",
                  head="e" * 40)
    assert driver.verify_shard(
        shard=_shard(KEY_D), host="sparky",
        expected_file=PINS_FILE,
        declared=_declared(head="e" * 40, generation=GEN_NEW))["status"] \
        == "failed"
    assert driver.verify_shard(
        shard=_shard(KEY_D), host="sparky",
        expected_file=LEVEL1_FILE,
        declared=_declared(head="e" * 40, generation=GEN_NEW))["status"] \
        == "qualified"


def test_duplicate_action_keys_across_cases_refused():
    """One action cannot qualify two different case labels."""
    driver = _driver()
    plan = _plan(driver)
    first = {"host": "sparky", "file": PINS_FILE, "status": "qualified",
             "reason": "", "passed": 2, "failed": 0, "skipped": 0,
             "action_key": KEY_A, "terminal": "/tmp/t.json",
             "snapshot_commit": "f" * 40, "snapshot_parent": "d" * 40,
             "generation": GEN, "receipt_sha256": "e" * 64}
    second = dict(first, file=RUNNER_FILE)
    report = driver.assemble_report(
        plan=plan, verdicts=[first, second], started_unix=0.0)
    duped = [c for c in report["cases"]
             if c["file"] == RUNNER_FILE][0]
    assert duped["status"] == "failed"
    assert "duplicate action key" in duped["reason"]


def test_snapshot_disagreement_and_interpreter_mismatch_fail(
        tmp_path, monkeypatch):
    """Terminal snapshot input and sealed interpreter bind the verdict."""
    driver = _driver()
    _tmp_cas(driver, tmp_path, "A")
    monkeypatch.setattr(driver, "_CAS_ROOT", tmp_path / "cas-A")
    _terminal_for(driver, tmp_path, monkeypatch, "A", KEY_A, "sparklina")
    terminal = _terminal(KEY_A, host="sparklina", parent="d" * 40,
                         input_sha="0" * 64)
    monkeypatch.setattr(driver, "_read_terminal",
                        lambda k: (terminal, "done"))
    assert "snapshot input disagrees" in driver.verify_shard(
        shard=_shard(KEY_A), host="sparklina",
        expected_file=PINS_FILE,
        declared=_declared())["reason"]
    assert "interpreter mismatch" in driver.verify_shard(
        shard=_shard(KEY_A), host="sparklina",
        expected_file=PINS_FILE,
        declared=_declared(
            python="/elsewhere/python"))["reason"]
