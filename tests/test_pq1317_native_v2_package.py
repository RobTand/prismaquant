"""Offline CPU validation of the fresh native-v2 evidence, not a GPU rerun.

Re-expand every probe record and reproduce the ORIGINAL raw file bytes/digest.
Then independently derive qualification from JUnit, roster/correction inputs,
per-mapping reader/backing-file provenance and the existing BF16 oracle helper.
No external paths, Docker, torch, candidate source, or library binaries required.
"""
from collections import Counter, defaultdict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import xml.etree.ElementTree as ET

import pytest

from pq1317_bindings import harness_binding

PACKAGE = Path(__file__).resolve().parents[1] / "docs/measurements/pq1317-gpu-tests"
RESULTS = PACKAGE / "results"


def read(path):
    return json.loads(path.read_bytes())


def digest(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


INDEX = read(RESULTS / "requalification.json")
RETAINED_HARNESS, RETAINED_SHA256 = harness_binding(PACKAGE, INDEX["harness_binding"])
VALIDATION_HARNESS, VALIDATION_SHA256 = harness_binding(PACKAGE, INDEX["validation_binding"])
ROSTER = read(PACKAGE / "roster.json")
CANDIDATE = read(PACKAGE / "candidate.json")
RUNS = [run for rd in INDEX["rounds"] for run in rd["runs"]]
PREFLIGHTS = INDEX["preflights"]
DENSE = [run for run in RUNS if run["suite"] == "dense"]
RAW_SHA256 = {
    "run-moe.4yrogwtx": "f736c43b23c70faf464cee01ba205968663ca79b96a19053b723130ab4a1d665",
    "run-dense.8iph6q0c": "6543b278d528d0cc237f4442ec644b00a58b7d42075d19c2ece8ff7001f0a7c5",
    "run-moe.lze3_4vo": "6af1a3e167da2371d281736084c80e787b30696ea588114d38184d33d90e7cd4",
    "run-dense.qg186gcx": "407128bc441bb6653f925a40661de5520fdcc376e8426f87e372f34ee3bebe4a",
}
spec = importlib.util.spec_from_file_location("pq1317_native_v2_oracle", VALIDATION_HARNESS / "bf16_oracle.py")
oracle = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oracle)


def root(run):
    return RESULTS / run["artifact_root"]


def expand(run, records=None, inventories=None):
    """Resolve only the explicit field; preserve all records, fields and ordering."""
    if records is None:
        records = read(root(run) / "probe-tests.json")
    if inventories is None:
        sidecar = read(root(run) / "native-inventories.json")
        assert sidecar["schema"] == "pq1317.native-inventories.v1"
        inventories = sidecar["inventories"]
    for key, value in inventories.items():
        assert key == digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())["sha256"]
    expanded, used = copy.deepcopy(records), set()
    for record in expanded:
        ref = record["native_libraries_mapped"]
        assert set(ref) == {"$native_inventory"}
        key = ref["$native_inventory"]
        record["native_libraries_mapped"] = copy.deepcopy(inventories[key])
        used.add(key)
    assert used == set(inventories)  # No unreferenced native snapshots.
    raw = (json.dumps(expanded, sort_keys=True, indent=1) + "\n").encode()
    assert run["raw_probe_tests"]["sha256"] == RAW_SHA256[run["name"]]
    assert digest(raw) == {k: run["raw_probe_tests"][k] for k in ("sha256", "bytes")}
    return expanded


def junit(run):
    tree = ET.parse(root(run) / "junit.xml").getroot()
    outcomes = {}
    for case in tree.iter("testcase"):
        nodeid = case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"]
        assert nodeid not in outcomes
        outcomes[nodeid] = next((key for tag, key in (("failure", "failed"), ("error", "error"),
                                                     ("skipped", "skipped"))
                                 if case.find(tag) is not None), "passed")
    for suite in tree.iter("testsuite"):
        assert int(suite.attrib["tests"]) == len(outcomes)
        assert all(int(suite.attrib[key]) == 0 for key in ("errors", "failures", "skipped"))
    return outcomes


def expected(run):
    return {node["id"] for node in ROSTER["nodes"] if node["suite"] == run["suite"]}


def test_index_inputs_and_independent_chronological_rounds():
    assert INDEX["schema"] == "pq1317.native-requalification.v1"
    assert INDEX["candidate"] == CANDIDATE["tessera_commit"] == ROSTER["candidate"]
    for name, recorded in INDEX["inputs"].items():
        assert digest((PACKAGE / name).read_bytes()) == recorded
    assert INDEX["harness_binding"] == "retained_native_v2"
    assert INDEX["validation_binding"] == "current_validation"
    assert RETAINED_HARNESS != VALIDATION_HARNESS
    assert INDEX["harness_sha256"] == RETAINED_SHA256
    assert set(VALIDATION_SHA256) == set(RETAINED_SHA256)
    for name, sha in INDEX["harness_sha256"].items():
        assert digest((RETAINED_HARNESS / name).read_bytes())["sha256"] == sha
    assert set(INDEX["harness_sha256"]) == {
        "bf16_oracle.py", "container_entry.py", "native_probe.py", "probe_selftest.py", "run_suite.py"}
    assert INDEX["probe_tests_encoding"]["reference_field"] == "$native_inventory"
    assert INDEX["probe_tests_encoding"]["sidecar"] == "native-inventories.json"
    assert INDEX["operations"]["runs"]["operation"] == "a577b8d9db2049265c77f9c211f8a7994b45374c25575163ddc6310664be1cb7"
    assert INDEX["operations"]["preflights"]["operation"] == "5267a884d103f822f34a4ff5022bdd80bc6006db6f937d89c417547e02ef01cf"
    assert [rd["round"] for rd in INDEX["rounds"]] == [1, 2]
    assert {r["name"] for r in RUNS} == set(RAW_SHA256)
    assert len({r["artifact_root"] for r in RUNS}) == 4
    previous = 0
    for rd in INDEX["rounds"]:
        assert {r["suite"] for r in rd["runs"]} == {"dense", "moe"}
        assert rd["total_nodes"] == sum(len(expected(r)) for r in rd["runs"]) == 136
        assert rd["started_unix"] == rd["runs"][0]["started_unix"] > previous
        for run in rd["runs"]:
            assert previous < run["started_unix"] < run["finished_unix"]
            previous = run["finished_unix"]
        assert rd["finished_unix"] == previous
    assert max(r["finished_unix"] for r in PREFLIGHTS) < RUNS[0]["started_unix"]
    for key, entries in (("runs", RUNS), ("preflights", PREFLIGHTS)):
        operation = INDEX["operations"][key]
        assert operation["launches"] == len(entries) and operation["exit_status"] == 0
        assert all(r["operation"] == operation["operation"] for r in entries)
        assert all("action" not in r for r in entries)  # No invented per-launch PB IDs.


@pytest.mark.parametrize("run", RUNS + PREFLIGHTS, ids=lambda r: r["name"])
def test_artifact_digests_and_minimal_retention(run):
    files = {p.name for p in root(run).iterdir()}
    required = {"collect.txt", "container-report.json", "run-manifest.json"}
    if run["mode"] == "run":
        required |= {"junit.xml", "pytest.log", "surface.json", "probe-summary.json", "probe-sessions.json",
                     "native-so.sha256", "expected-nodes.json", "triton-kernels.json", "probe-tests.json",
                     "native-inventories.json"}
    assert files == required == set(run["artifacts"])
    for name, recorded in run["artifacts"].items():
        assert digest((root(run) / name).read_bytes()) == recorded
    manifest = read(root(run) / "run-manifest.json")
    raw_paths = set()
    for item in run["external_raw_jsonl"]:
        path = PurePosixPath(item["path"])
        assert path.is_absolute() and path.is_relative_to(manifest["out"])
        assert path.suffix == ".jsonl" and path not in raw_paths
        assert re.fullmatch("[0-9a-f]{64}", item["sha256"]) and item["bytes"] > 0
        raw_paths.add(path)
    assert len(raw_paths) == (7 if run["mode"] == "run" else 4)
    if run["mode"] == "run":
        assert run["raw_probe_tests"]["path"] == str(PurePosixPath(manifest["out"]) / "probe-tests.json")
        expand(run)  # Raw whitespace, all fields and all test records must be reversible.


@pytest.mark.parametrize("run", RUNS + PREFLIGHTS, ids=lambda r: r["name"])
def test_source_image_harness_and_cpu_preflight_identity(run):
    m, c = (read(root(run) / n) for n in ("run-manifest.json", "container-report.json"))
    assert m["candidate"] == c["candidate"] == INDEX["candidate"]
    assert m["suite"] == c["suite"] == run["suite"]
    assert m["mode"] == c["mode"] == run["mode"]
    assert m["exit_code"] == m["docker_returncode"] == c["exit_code"] == 0
    assert m["started_unix"] == run["started_unix"] < c["started_unix"]
    assert c["finished_unix"] < run["finished_unix"] == m["finished_unix"]
    assert m["harness_sha256"] == INDEX["harness_sha256"]
    assert all(m["package_sha256"][n] == INDEX["inputs"][n]["sha256"] for n in ("candidate.json", "roster.json"))
    assert run["launch_snapshot"] == {key: json.loads(value) for key, value in m["closure_files"].items()}
    # A sealed harness snapshot may be dirty or have a different HEAD. Bind bytes,
    # do not require its closure HEAD to equal the Tessera candidate or be clean.
    assert run["source"] == {"checkout": m["source"], "verified_identity": c["identity"]}
    assert m["source"]["head"] == INDEX["candidate"]
    assert m["source"]["tree"] == CANDIDATE["tessera_commit_facts"]["git_tree_oid"]
    identity = c["identity"]
    assert identity["verified_at_candidate"] is identity["contract_matches_pin"] is True
    assert identity["contract_sha256"] == CANDIDATE["contract_sha256"]
    source = identity["suite_source"]
    assert source["verification"] == "verified" and source["snapshot_commit"] == INDEX["candidate"]
    assert source["files_verified"] == m["source"]["tracked_files"] == 2005
    assert re.fullmatch("[0-9a-f]{64}", source["sha256"])
    assert run["image"] == m["image_inspect"]
    assert m["image"] == run["image"]["reference"] == CANDIDATE["image_digest"]
    assert m["image"] in run["image"]["repo_digests"]
    assert re.fullmatch("sha256:[0-9a-f]{64}", run["image"]["id"])  # Config ID != RepoDigest is valid.
    assert c["probe_loaded_in_collection"] is True
    assert c["probe_selftest"]["ok"] is True and c["probe_selftest"]["returncode"] == 0
    assert c["probe_selftest"]["workers"] == ["gw0", "gw1"]
    roster_check = c["roster_check"]
    assert roster_check == dict(collected=len(expected(run)), expected=len(expected(run)),
                                equal=True, duplicates=[], missing=[], unexpected=[])
    collected = [line for line in (root(run) / "collect.txt").read_text().splitlines() if line.startswith("tests/")]
    assert Counter(collected) == Counter(expected(run))
    assert run["roster_count"] == len(collected)
    gpu = run["mode"] == "run"
    assert c["runtime"]["cuda_available"] is run["cuda_available"] is gpu
    assert ("--gpus" in m["docker_command"]) is gpu
    assert run["device"] == c["runtime"].get("device")
    if gpu:
        assert run["device"]["name"] == CANDIDATE["target"]["device"]
        assert run["device"]["capability"] == [12, 1]
    else:
        assert run["device"] is None and m["native_libraries_built"] == []


@pytest.mark.parametrize("run", RUNS, ids=lambda r: r["name"])
def test_outcomes_roster_sessions_and_surface_are_rederived(run):
    records, outcomes = expand(run), junit(run)
    assert set(outcomes) == expected(run) and set(outcomes.values()) == {"passed"}
    counts = dict.fromkeys(("passed", "failed", "error", "skipped"), 0)
    counts.update(Counter(outcomes.values()))
    assert run["outcomes"] == counts
    assert Counter(read(root(run) / "expected-nodes.json")) == Counter(expected(run))
    assert Counter(t["nodeid"] for t in records) == Counter(expected(run))
    sessions = read(root(run) / "probe-sessions.json")
    assert Counter(s["worker"] for s in sessions) == Counter(["gw0", "gw1", "main"])
    for session in sessions:
        assert session["kind"] == "session" and session["exitstatus"] == 0
        assert session["executes_tests"] is (session["worker"] != "main")
    workers = {s["worker"]: s["pid"] for s in sessions if s["executes_tests"]}
    for t in records:
        assert t["kind"] == "test" and t["when"] == "call" and t["outcome"] == outcomes[t["nodeid"]]
        assert t["worker"] in workers and t["pid"] == workers[t["worker"]]
        assert t["assertions_dropped"] == 0
    summary = read(root(run) / "probe-summary.json")
    assert summary["tests_recorded"] == len(records)
    assert summary["assertions_recorded"] == sum(len(t["assertions"]) for t in records)
    assert summary["errors"] == [] and summary["assertions_dropped"] == 0 and summary["qualification_ok"] is True
    assert summary["workers"] == sorted(workers)
    assert read(root(run) / "container-report.json")["probe"] == summary
    surface = read(root(run) / "surface.json")
    assert surface["counts"] == dict(counts, xfailed=0, xpassed=0)
    assert surface["cuda"] is surface["strict_cuda"] is True
    assert surface["commit"] == INDEX["candidate"] and surface["xdist_workers"] == 2
    identity = surface["source_identity"]
    assert identity["verification"] == "verified" and identity["workers"] == {w: "agrees" for w in workers}
    assert identity["measurement_span"]["agrees"] is True
    assert identity["measurement_span"]["entry_sha256"] == identity["sha256"]
    assert identity["sha256"] == run["source"]["verified_identity"]["suite_source"]["sha256"]
    assert surface["not_collected"] == [] and surface["skip_reasons"] == {}
    assert surface["cuda_surface"]["is_a_floor"] is True
    assert surface["cuda_surface"]["executed"] == run["cuda_allocating_nodes_floor"] == (46 if run["suite"] == "dense" else 38)
    assert f"{len(records)} passed" in (root(run) / "pytest.log").read_text()


def test_correction_maps_are_covered_by_passing_nodes_in_both_rounds():
    corrections = read(PACKAGE / "corrections.json")["corrections"]
    for rd in INDEX["rounds"]:
        outcomes = {node: outcome for run in rd["runs"] for node, outcome in junit(run).items()}
        for correction in corrections:
            mapped = {node for failure in correction["original_failures"] for node in failure["candidate_nodes"]}
            assert all(outcomes[node] == "passed" for node in mapped)
            suite = next(s for s, v in ROSTER["suites"].items() if v["issue"] == correction["issue"]["number"])
            nodes = [n for n in ROSTER["nodes"] if n["suite"] == suite and n["tier"] == "correction"]
            assert mapped == {n["id"] for n in nodes}
            elements = {e for n in nodes if outcomes[n["id"]] == "passed" for e in n["elements"]}
            assert elements == set(correction["elements"])
            assert INDEX["correction_coverage_per_round"][str(correction["issue"]["number"])] == {
                "original_failures": len(correction["original_failures"]),
                "candidate_nodes_passed": len(mapped), "elements": sorted(elements)}


def check_native(inventory, worker, out, built):
    """Independent verifier: do not trust the probe's identity_verified flag alone."""
    for module, info in inventory.items():
        assert info["identity_verified"] is True and info["deleted"] is info["nfs_renamed"] is False
        path = PurePosixPath(info["path"]).relative_to(out)
        assert path.parts[:2] == ("torch-ext", worker)
        assert built[str(path)] == info["sha256"] and path.name == module + ".so"
        assert info["mappings"]
        identities = set()
        for mapping in info["mappings"]:
            assert mapping["identity_verified"] is True
            assert mapping["module"] == module and mapping["path"] == info["path"]
            assert mapping["sha256"] == info["sha256"]
            assert mapping["deleted"] is mapping["nfs_renamed"] is False
            backing = mapping["backing_file"]
            keys = ("device_major", "device_minor", "inode")
            assert all(type(mapping[k]) is int and mapping[k] == backing[k] for k in keys)
            assert mapping["inode"] > 0 and backing["size"] > 0
            identities.add(tuple(backing[k] for k in keys))
            assert mapping["reader"] in {"map_files", "pathname"}
            # Exactly one final success, preceded only by reader unavailability.
            attempts = mapping["attempts"]
            readers = (["map_files"] if mapping["reader"] == "map_files"
                       else ["map_files", "pathname"])
            assert [a["reader"] for a in attempts] == readers
            assert "error" not in attempts[-1] and "error_stage" not in attempts[-1]
            # Later success must never erase a prior opened-file/race failure.
            for attempt in attempts:
                if "error" in attempt:
                    assert "opened_file" not in attempt
                    assert attempt.get("error_stage") in (None, "acquisition")
                    assert attempt.get("reader") == "map_files"
                    assert attempt.get("path") == "/proc/self/map_files/" + mapping["address"]
                    assert str(attempt["error"]).startswith((
                        "[Errno 1]", "[Errno 2]", "[Errno 13]"))
            success = [a for a in mapping["attempts"] if "error" not in a]
            assert len(success) == 1 and success[0]["reader"] == mapping["reader"]
            assert success[0]["opened_file"] == backing
            if mapping["reader"] == "pathname":
                assert success[0]["path"] == mapping["path"]
                assert any(a["reader"] == "map_files" and a["error"] for a in mapping["attempts"] if "error" in a)
            else:
                assert success[0]["path"] == "/proc/self/map_files/" + mapping["address"]
        assert len(identities) == 1


@pytest.mark.parametrize("error,opened", [
    ("opened file device/inode differs from mapping", True),
    ("backing file changed while hashing", True),
    ("mapping changed while hashing", True),
    ("mapping changed while hashing", False),
    ("[Errno 5] Input/output error", False),
])
def test_independent_verifier_refuses_recovered_integrity_failure(error, opened):
    run = next(r for r in RUNS if r["suite"] == "dense")
    manifest = read(root(run) / "run-manifest.json")
    built = {item["path"]: item["sha256"] for item in manifest["native_libraries_built"]}
    record = next(t for t in expand(run) if t["native_libraries_mapped"])
    inventory = copy.deepcopy(record["native_libraries_mapped"])
    check_native(inventory, record["worker"], manifest["out"], built)
    mapping = next(iter(inventory.values()))["mappings"][0]
    failed = {"reader": "map_files", "path": "/proc/self/map_files/" + mapping["address"],
              "error": error}
    if opened:
        failed["opened_file"] = copy.deepcopy(mapping["backing_file"])
    mapping["attempts"].insert(0, failed)
    with pytest.raises(AssertionError):
        check_native(inventory, record["worker"], manifest["out"], built)


@pytest.mark.parametrize("run", RUNS, ids=lambda r: r["name"])
def test_native_backing_identity_and_built_inventory(run):
    m = read(root(run) / "run-manifest.json")
    built = {item["path"]: item["sha256"] for item in m["native_libraries_built"]}
    lines = [line.split(maxsplit=1) for line in (root(run) / "native-so.sha256").read_text().splitlines()]
    assert built == {path: sha for sha, path in lines}
    assert len(built) == len(m["native_libraries_built"]) == len(lines)
    records, sessions = expand(run), read(root(run) / "probe-sessions.json")
    by_worker, seen, mapped, observations = defaultdict(dict), defaultdict(set), set(), []
    for t in records + sessions:
        inventory = t["native_libraries_mapped"]
        check_native(inventory, t["worker"], m["out"], built)
        by_worker[t["worker"]].update(inventory)
        for module, info in inventory.items():
            seen[module].add(info["sha256"])
            mapped.add((str(PurePosixPath(info["path"]).relative_to(m["out"])), info["sha256"]))
            observations.extend(info["mappings"])
    assert mapped == set(built.items())  # Neither mapped-only nor built-only entries.
    summary = read(root(run) / "probe-summary.json")
    assert dict(by_worker) == summary["native_libraries_by_worker"]
    assert {k: sorted(v) for k, v in seen.items()} == summary["native_library_sha256_seen"]
    assert run["native"]["evidence_qualified"] is m["native_evidence_qualified"] is True
    assert run["native"]["errors"] == m["native_evidence_errors"] == []
    assert run["native"]["mapping_observations"] == len(observations)
    assert run["native"]["readers"] == sorted({o["reader"] for o in observations})
    assert run["native"]["tests_with_mappings"] == sum(bool(t["native_libraries_mapped"]) for t in records)
    assert run["native"]["built_libraries"] == len(built) == (4 if run["suite"] == "dense" else 0)
    triton = read(root(run) / "triton-kernels.json")
    assert triton == m["triton"]
    assert all(json.loads(t)["backend"] == "cuda" and json.loads(t)["arch"] == 121 for t in triton["targets"])
    kernel = "_window_gemm_kernel" if run["suite"] == "dense" else "_grouped_window_gemm_kernel"
    assert triton["kernels"][kernel] > 0


@pytest.mark.parametrize("run", RUNS, ids=lambda r: r["name"])
def test_bf16_numerical_measurements_using_existing_helper(run):
    records = expand(run)
    assert oracle.bf16_oracle_errors(records, expected(run)) == []
    measurements = []
    for record in records:
        for assertion in record["assertions"]:
            if "numerical" in assertion:
                value = assertion["numerical"]
                assert value["nodeid"] == record["nodeid"]
                assert value["line"] == assertion["line"] and value["orig"] == assertion["assertion"]
                measurements.append(value)
    nodes = {m["nodeid"] for m in measurements}
    assert nodes == expected(run) & oracle.EXPECTED_COMPARISONS.keys()
    assert run["bf16_oracle"] == {"nodes": len(nodes), "measurements": len(measurements)}
    assert (len(nodes), len(measurements)) == ((19, 23) if run["suite"] == "dense" else (0, 0))


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "cpu", "nonfinite", "violation"])
def test_bf16_gate_refuses_damaged_real_evidence(mutation):
    run = DENSE[0]
    records = expand(run)
    record = next(t for t in records if any("numerical" in a for a in t["assertions"]))
    assertion = next(a for a in record["assertions"] if "numerical" in a)
    if mutation == "missing":
        record["assertions"].remove(assertion)
    elif mutation == "duplicate":
        record["assertions"].append(copy.deepcopy(assertion))
    else:
        key, value = {"cpu": ("device", "cpu"), "nonfinite": ("max_abs_error", float("nan")),
                      "violation": ("violation_count", 1)}[mutation]
        assertion["numerical"][key] = value
    assert oracle.bf16_oracle_errors(records, expected(run))


@pytest.mark.parametrize("mutation", ["record", "inventory"])
def test_lossless_encoding_refuses_tampering(mutation):
    run = DENSE[0]
    records = read(root(run) / "probe-tests.json")
    inventories = read(root(run) / "native-inventories.json")["inventories"]
    if mutation == "record":
        records[0]["duration_s"] += 1
    else:
        next(v for v in inventories.values() if v).clear()
    with pytest.raises(AssertionError):
        expand(run, records, inventories)


@pytest.mark.parametrize("mutation", ["inode", "device_minor", "reader", "sha256", "opened_file"])
def test_native_gate_refuses_bad_provenance_even_with_true_flags(mutation):
    run = DENSE[0]
    record = next(t for t in expand(run) if t["native_libraries_mapped"])
    m = read(root(run) / "run-manifest.json")
    built = {item["path"]: item["sha256"] for item in m["native_libraries_built"]}
    inventory = record["native_libraries_mapped"]
    mapping = next(iter(inventory.values()))["mappings"][0]
    if mutation in ("inode", "device_minor"):
        mapping[mutation] += 1
    elif mutation == "opened_file":
        next(a for a in mapping["attempts"] if "opened_file" in a)["opened_file"]["inode"] += 1
    else:
        mapping[mutation] = "unverified"
    assert mapping["identity_verified"] is True
    with pytest.raises(AssertionError):
        check_native(inventory, record["worker"], m["out"], built)
