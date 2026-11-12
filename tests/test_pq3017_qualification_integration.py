"""Offline acceptance for #3017; CPU provenance checks, never new GPU evidence.

The native-v2 fixture is the unmodified preserved observer. While the package is
absent it lets the behavioral regressions demonstrate the actual preserved bug,
not an import error. Once recovered, ONLY the live package harness is exercised.
The independent integration checks still require the package and all four gates.
No test reads Git history or external measurement directories.
"""
from __future__ import annotations

import copy
import errno
import hashlib
import importlib.util
import json
import mmap
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "docs/measurements/pq1317-gpu-tests"
FIXTURE = ROOT / "tests/fixtures/pq3017"
GATES = [
    "test_pq1317_gpu_qualification_package.py",
    "test_pq1317_native_identity.py",
    "test_pq1317_bf16_identity_requalification.py",
    "test_pq1317_native_v2_package.py",
]


def read(path):
    assert path.is_file(), f"qualification package integration is missing {path.relative_to(ROOT)}"
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def modules(monkeypatch):
    harness = PACKAGE / "harness"
    if not harness.exists():
        harness = FIXTURE / "native-v2"
    loaded = {}
    for name in ("native_probe", "bf16_oracle", "container_entry", "run_suite"):
        path = harness / f"{name}.py"
        assert path.is_file(), f"incomplete qualification harness: {path}"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        loaded[name] = module
    return loaded


@pytest.fixture
def real_mapping(tmp_path, modules):
    path = tmp_path / "tessera_terminal.so"
    path.write_bytes(b"original".ljust(mmap.PAGESIZE, b"x"))
    with path.open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as memory:
        entry, = [e for e in modules["native_probe"]._maps() if e["path"] == str(path)]
        yield path, memory, entry


def inventory(record):
    return {"tessera_terminal": {
        "path": record["path"], "nfs_renamed": False, "deleted": False,
        "identity_verified": record["identity_verified"], "sha256": record["sha256"],
        "mappings": [record],
    }}


def records(libraries):
    result = []
    for worker in ("gw0", "gw1"):
        common = {"worker": worker, "pid": 1, "native_libraries_mapped": libraries}
        result += [
            {**common, "kind": "test", "nodeid": f"synthetic.py::{worker}", "when": "call",
             "outcome": "passed", "assertions": [], "assertions_dropped": 0},
            {**common, "kind": "session", "executes_tests": True, "exitstatus": 0},
        ]
    return copy.deepcopy(result)


def merged(modules, out, libraries):
    (out / "probe").mkdir()
    (out / "probe/probe.main.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records(libraries)))
    return modules["container_entry"].merge_probe(out, ["synthetic.py::gw0", "synthetic.py::gw1"])


def denied_map_files(monkeypatch):
    original = Path.open

    def opened(path, *args, **kwargs):
        if str(path).startswith("/proc/self/map_files/"):
            raise PermissionError(errno.EPERM, "map_files denied", str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", opened)


@pytest.mark.parametrize("gate", ["observer", "qualification", "merge", "build"])
def test_one_time_mutation_after_successful_same_inode_open_is_terminal(
        gate, modules, real_mapping, monkeypatch, tmp_path):
    probe = modules["native_probe"]
    path, memory, entry = real_mapping
    # Positive control within this regression: no privileges are required.
    with monkeypatch.context() as allowed:
        denied_map_files(allowed)
        good = probe._digest(entry)
        assert good["identity_verified"] and good["reader"] == "pathname"
        assert probe.native_identity_errors(inventory(good)) == []

    original_open = Path.open
    original_hash = probe.hashlib.file_digest
    calls = []

    def opened(target, *args, **kwargs):
        if str(target).startswith("/proc/self/map_files/"):
            # Emulate an AVAILABLE map_files descriptor to the ACTUAL mapped inode.
            handle = original_open(path, "rb")
            assert os.fstat(handle.fileno()).st_ino == entry["inode"]
            return handle
        return original_open(target, *args, **kwargs)

    def mutate_once(handle, algorithm):
        result = original_hash(handle, algorithm)
        calls.append(handle.name)
        if len(calls) == 1:
            # Appending changes size deterministically on any supported filesystem;
            # do not truncate a live mmap or depend on clock resolution.
            with original_open(path, "ab") as writer:
                writer.write(b"changed-once")
        return result

    monkeypatch.setattr(Path, "open", opened)
    # Also preserve the available-reader route; refusing everything is not a fix.
    unchanged = probe._digest(entry)
    assert unchanged["identity_verified"] and unchanged["reader"] == "map_files"
    assert probe.native_identity_errors(inventory(unchanged)) == []
    control = tmp_path / "unchanged"
    control.mkdir()
    assert merged(modules, control, inventory(unchanged))["qualification_ok"] is True
    assert modules["run_suite"].native_inventory_errors(
        control, [{"path": str(path), "sha256": unchanged["sha256"]}]) == []
    monkeypatch.setattr(probe.hashlib, "file_digest", mutate_once)
    record = probe._digest(entry)
    assert memory[:8] == b"original"
    assert any("changed while hashing" in a.get("error", "") for a in record["attempts"])
    libraries = inventory(record)
    if gate == "observer":
        assert record["identity_verified"] is False, "integrity failure was laundered by pathname retry"
        assert record["sha256"] is None
        assert len(calls) == 1, "terminal refusal must not hash another reader"
        assert all(a["reader"] != "pathname" for a in record["attempts"])
    elif gate == "qualification":
        assert probe.native_identity_errors(libraries), "gate accepted a detected mutation"
    elif gate == "merge":
        summary = merged(modules, tmp_path, libraries)
        assert summary["qualification_ok"] is False, "passing pytest/session records erased mutation"
        assert summary["errors"]
    else:
        summary = merged(modules, tmp_path, libraries)
        # The independent build join must examine provenance, not trust a green flag.
        summary.update(qualification_ok=True, errors=[], native_libraries_by_worker={"gw0": libraries})
        (tmp_path / "probe-summary.json").write_text(json.dumps(summary))
        built = [{"path": str(path), "sha256": libraries["tessera_terminal"]["sha256"]}]
        assert modules["run_suite"].native_inventory_errors(tmp_path, built)


@pytest.mark.parametrize("reason", [
    "backing file changed while hashing",
    "mapping changed while hashing",
    "mapping has no regular backing file",
    "opened file device/inode differs from mapping",
])
@pytest.mark.parametrize("gate", ["qualification", "merge", "build"])
def test_later_success_cannot_hide_a_recorded_integrity_failure(
        reason, gate, modules, real_mapping, monkeypatch, tmp_path):
    probe = modules["native_probe"]
    _, _, entry = real_mapping
    denied_map_files(monkeypatch)
    good = probe._digest(entry)
    assert probe.native_identity_errors(inventory(good)) == []
    corrupted = copy.deepcopy(good)
    # Simulate the old observer's retained format. This is NOT a new claim about
    # the measured runs and not the withdrawn different-inode mmap objection.
    corrupted["attempts"][0]["opened_file"] = copy.deepcopy(good["backing_file"])
    corrupted["attempts"][0]["error"] = reason
    libraries = inventory(corrupted)
    if gate == "qualification":
        assert probe.native_identity_errors(libraries)
    elif gate == "merge":
        summary = merged(modules, tmp_path, libraries)
        assert summary["qualification_ok"] is False and summary["errors"]
    else:
        summary = {"qualification_ok": True, "errors": [],
                   "native_libraries_by_worker": {"gw0": libraries}}
        (tmp_path / "probe-summary.json").write_text(json.dumps(summary))
        built = [{"path": good["path"], "sha256": good["sha256"]}]
        assert modules["run_suite"].native_inventory_errors(tmp_path, built)


def test_retained_qualification_validator_refuses_laundered_mapping(
        modules, real_mapping, monkeypatch, tmp_path):
    source = ROOT / "tests/test_pq1317_native_v2_package.py"
    assert source.is_file(), "recover the independent retained-record validator"
    spec = importlib.util.spec_from_file_location("pq3017_retained_validator", source)
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    probe = modules["native_probe"]
    _, _, entry = real_mapping
    denied_map_files(monkeypatch)
    record = probe._digest(entry)
    # Schema-only relocation into the validator's expected recorded build layout.
    # No CUDA execution, external library read or different-inode claim is made.
    relative = "torch-ext/gw0/tessera_terminal.so"
    recorded_path = str(tmp_path / relative)
    record["path"] = recorded_path
    record["attempts"][-1]["path"] = recorded_path
    libraries = inventory(record)
    built = {relative: record["sha256"]}
    validator.check_native(libraries, "gw0", str(tmp_path), built)
    record["attempts"][0].update(
        opened_file=copy.deepcopy(record["backing_file"]),
        error="backing file changed while hashing")
    with pytest.raises(AssertionError):
        validator.check_native(libraries, "gw0", str(tmp_path), built)


def test_preserved_measurements_and_receipt_identities_are_not_rewritten():
    expected = read(FIXTURE / "preserved-files.json")
    for name, identity in expected.items():
        path = PACKAGE / name
        assert path.is_file(), f"retained package artifact missing: {name}"
        assert {"sha256": digest(path), "bytes": path.stat().st_size} == identity, name
    published = read(PACKAGE / "results/log-digests.json")["files"]
    on_disk = {str(p.relative_to(PACKAGE)) for p in (PACKAGE / "results/logs").rglob("*") if p.is_file()}
    assert set(published) == on_disk, "digest coverage must include ALL retained log trees"
    for name in on_disk:
        path = PACKAGE / name
        assert published[name] == {"sha256": digest(path), "bytes": path.stat().st_size}


def test_recorded_and_current_harness_bindings_are_distinct():
    # This small additive publication contract leaves every launch manifest intact.
    bindings = read(PACKAGE / "results/harness-bindings.json")
    expected = read(FIXTURE / "snapshot-digests.json")
    index = read(PACKAGE / "results/requalification.json")
    assert index["harness_binding"] == "retained_native_v2"
    assert index["validation_binding"] == "current_validation"
    roots = []
    for key, snapshot in (("historical_run2", "run2"), ("retained_native_v2", "native-v2")):
        binding = bindings[key]
        root = (PACKAGE / binding["root"]).resolve()
        assert root.is_relative_to(PACKAGE.resolve())
        roots.append(root)
        actual = {p.name: digest(p) for p in root.glob("*.py")}
        assert binding["sha256"] == actual == expected[snapshot]
    current = bindings["current_validation"]
    live = (PACKAGE / current["root"]).resolve()
    assert live == (PACKAGE / "harness").resolve()
    assert current["sha256"] == {p.name: digest(p) for p in live.glob("*.py")}
    assert live not in roots and roots[0] != roots[1]
    assert current["sha256"]["native_probe.py"] != expected["native-v2"]["native_probe.py"]
    assert index["harness_sha256"] == expected["native-v2"]
    for suite in ("moe", "dense"):
        manifest = read(PACKAGE / f"results/logs/run2-{suite}/run-manifest.json")
        assert manifest["harness_sha256"] == expected["run2"]
    for round_ in index["rounds"]:
        for run in round_["runs"]:
            manifest = read(PACKAGE / "results" / run["artifact_root"] / "run-manifest.json")
            assert manifest["harness_sha256"] == bindings[index["harness_binding"]]["sha256"]


def test_retained_index_replays_under_current_integrity_rules(modules):
    index = read(PACKAGE / "results/requalification.json")
    preserved = read(FIXTURE / "requalification.json")
    for field in ("candidate", "inputs", "operations", "preflights", "rounds",
                  "correction_coverage_per_round", "probe_tests_encoding", "harness_sha256"):
        assert index[field] == preserved[field], f"measured index input changed: {field}"
    for round_ in index["rounds"]:
        assert round_["total_nodes"] == 136
        for run in round_["runs"]:
            base = PACKAGE / "results" / run["artifact_root"]
            tests = read(base / "probe-tests.json")
            inventories = read(base / "native-inventories.json")["inventories"]
            for key, value in inventories.items():
                assert hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == key
            expanded = copy.deepcopy(tests)
            for test in expanded:
                reference = test["native_libraries_mapped"]
                test["native_libraries_mapped"] = inventories[reference["$native_inventory"]]
            raw = (json.dumps(expanded, sort_keys=True, indent=1) + "\n").encode()
            assert hashlib.sha256(raw).hexdigest() == run["raw_probe_tests"]["sha256"]
            assert len(raw) == run["raw_probe_tests"]["bytes"]
            expected = read(base / "expected-nodes.json")
            assert sorted(t["nodeid"] for t in expanded) == sorted(expected)
            assert len(expected) == {"moe": 71, "dense": 65}[run["suite"]]
            assert modules["bf16_oracle"].bf16_oracle_errors(expanded, expected) == []
            for record in expanded + read(base / "probe-sessions.json"):
                assert modules["native_probe"].native_identity_errors(record["native_libraries_mapped"]) == []


def test_how_to_check_names_the_complete_gate_and_separates_claims():
    readme = PACKAGE / "README.md"
    assert readme.is_file(), "recover and update the qualification README"
    text = readme.read_text()
    assert "how to check" in text.lower()
    section = text.lower().split("how to check", 1)[1].split("\n## ", 1)[0]
    for name in GATES + [Path(__file__).name]:
        assert name in section, f"How to check omits {name}"
    assert "python -m pytest" in section
    assert "historical" in text.lower() and "current" in text.lower()
    assert "fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb" in text


def test_complete_recovered_package_gate_passes_offline(tmp_path):
    paths = [ROOT / "tests" / name for name in GATES]
    assert all(p.is_file() for p in paths), "recover all four preserved qualification/regression gates"
    environment = dict(os.environ)
    environment.pop("PQ_OWN_PROCESS_REPORT", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--basetemp", str(tmp_path / "package-gate"),
         *[str(p.relative_to(ROOT)) for p in paths]],
        cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    assert " skipped" not in result.stdout, "CPU package regressions must execute, not skip"
