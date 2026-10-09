"""The pq1317 qualification package agrees with itself and with its retained raw logs.

The package (docs/measurements/pq1317-gpu-tests) qualifies one Tessera candidate for
tessera#610 and #611 on image X. These tests hold the record honest: every node outcome
is re-derived from the retained junit files, every retained log matches its digest, and
the identity chain, the correction map, the native-library records and the probe plugin
are checked against each other. They read files only; no GPU and no Tessera is needed.
"""

import hashlib
import importlib.util
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "docs" / "measurements" / "pq1317-gpu-tests"
RESULTS = PKG / "results"
LOGS = RESULTS / "logs"
HARNESS = PKG / "harness"

CANDIDATE = "fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb"
CONTRACT = "ee065629b081d913a0351e43160c5c6e1bd38fa628cafd51e756e9caf3bb334e"
IMAGE = "localhost/prismaquant/spark-vllm-nccl230@sha256:f8dbe1a02e33ccb7416ab40b72a83e8c725dcb6fed3e90bae4a658cce5e1b7f5"
SUITES = {"moe": 71, "dense": 65}
RUNS = {"run2": "accepted run, instrument verified", "run1": "corroborating run of attempt 2"}
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()

# The dense route's own launch pairs at the candidate (bf16_route and fp8_route DENSE_LAUNCHES).
FUSED_BF16 = ["tessera::fused_window_dense", "native_fused_window_dense_folded"]
TRITON_BF16 = ["tessera::window_gemm_dense", "native_window_gemm_folded"]
FUSED_FP8_MMA = ["tessera::fused_window_dense", "native_fused_window_dense_e4m3mma"]


def load(*parts):
    return json.loads(PKG.joinpath(*parts).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def junit_nodes(path):
    """``{node id: (outcome, seconds, cuda allocated)}`` from a retained junit file."""
    root = ET.parse(path).getroot()
    suite = next(root.iter("testsuite"))
    nodes = {}
    for case in root.iter("testcase"):
        node = case.get("classname").replace(".", "/") + ".py::" + case.get("name")
        outcome = "passed"
        for child in case:
            if child.tag in ("failure", "error", "skipped"):
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[child.tag]
        props = {p.get("name"): p.get("value") for p in case.iter("property")}
        nodes[node] = (outcome, float(case.get("time")), props.get("tessera_cuda_executed") == "True")
    return nodes, {k: int(suite.get(k)) for k in ("tests", "failures", "errors", "skipped")}


def run_dir(run, suite):
    return LOGS / f"{run}-{suite}"


@pytest.fixture(scope="module")
def roster():
    return load("roster.json")


@pytest.fixture(scope="module")
def results():
    return {suite: load("results", f"{suite}.json") for suite in SUITES}


def roster_nodes(roster, suite):
    return [n["id"] for n in roster["nodes"] if n["suite"] == suite]


# ----------------------------------------------------------------- identity ----

def test_one_candidate_binds_every_record(roster, results):
    candidate = load("candidate.json")
    assert candidate["tessera_commit"] == CANDIDATE
    assert candidate["contract_sha256"] == candidate["contract"]["sha256"] == CONTRACT
    assert candidate["image_digest"] == IMAGE
    assert candidate["contract"]["contract_version"] == 60
    assert candidate["source_tree"]["shared_pin_dir_equals_archive"] is True
    assert len(candidate["serving_native_extensions"]) == 4
    assert roster["candidate"] == load("corrections.json")["candidate"] == CANDIDATE
    summary, manifest = load("results", "summary.json"), load("results", "candidate-manifest.json")
    assert summary["candidate"] == manifest["candidate"] == CANDIDATE
    assert summary["image"] == manifest["image"] == IMAGE
    assert summary["contract"]["sha256"] == manifest["contract_sha256"] == CONTRACT
    for suite, document in results.items():
        assert document["candidate"] == CANDIDATE and document["image"] == IMAGE, suite


def test_the_readme_names_the_candidate_and_the_accepted_actions():
    readme = (PKG / "README.md").read_text(encoding="utf-8")
    assert "Status: **complete**" in readme
    assert CANDIDATE in readme and IMAGE in readme and CONTRACT in readme
    summary = load("results", "summary.json")
    for action in summary["primary_runs"].values():
        assert action in readme
    commands = (PKG / "gpu-commands.md").read_text(encoding="utf-8")
    for action in summary["primary_runs"].values():
        assert action in commands


@pytest.mark.parametrize("suite", SUITES)
def test_the_source_is_verified_at_the_candidate_in_the_accepted_run(suite):
    surface = json.loads((run_dir("run2", suite) / "surface.json").read_text())
    identity = surface["source_identity"]
    assert surface["commit"] == identity["snapshot_commit"] == CANDIDATE
    assert identity["verification"] == "verified" and identity["files_verified"] == 2005
    assert identity["measurement_span"]["agrees"] is True
    assert set(identity["workers"].values()) == {"agrees"} and len(identity["workers"]) == 2
    report = json.loads((run_dir("run2", suite) / "container-report.json").read_text())
    assert report["exit_code"] == 0 and report["mode"] == "run" and report["candidate"] == CANDIDATE
    assert report["identity"]["verified_at_candidate"] is True
    assert report["identity"]["contract_matches_pin"] is True
    assert report["identity"]["contract_sha256"] == CONTRACT
    assert report["roster_check"]["equal"] is True and report["probe_loaded_in_collection"] is True
    assert report["probe_selftest"]["ok"] is True
    assert report["runtime"]["device"]["capability"] == [12, 1]
    assert report["runtime"]["device"]["name"] == "NVIDIA GB10"
    assert report["runtime"]["torch"] == "2.13.0+cu130" and report["runtime"]["tessera_environment"] == {}


def test_both_accepted_suites_read_the_same_source():
    sources = {suite: json.loads((run_dir("run2", suite) / "surface.json").read_text())["source_identity"]["sha256"]
               for suite in SUITES}
    assert len(set(sources.values())) == 1 and set(sources.values()) == {
        load("results", "summary.json")["source_identity"]["sha256"]}


def test_the_accepted_runs_used_a_clean_snapshot_of_the_recorded_harness():
    digests = load("results", "log-digests.json")
    heads = set()
    for suite in SUITES:
        manifest = json.loads((run_dir("run2", suite) / "run-manifest.json").read_text())
        closure = json.loads(next(iter(manifest["closure_files"].values())))
        assert closure["dirty_sha256"] == EMPTY_SHA256
        heads.add(closure["head"])
        assert manifest["candidate"] == CANDIDATE and manifest["image"] == IMAGE
        assert manifest["source"]["head"] == CANDIDATE and manifest["docker_returncode"] == 0
        assert manifest["image_inspect"]["id"] == IMAGE.split("@")[1]
        for name, digest in manifest["harness_sha256"].items():
            assert digest == sha256(HARNESS / name), f"{name} differs from the harness that ran"
    assert len(heads) == 1 and len(next(iter(heads))) == 40
    assert digests["harness"] == {f"harness/{p.name}": sha256(p) for p in sorted(HARNESS.glob("*.py"))}


# ------------------------------------------------------------------- nodes -----

def test_the_roster_is_the_collected_population(roster):
    nodes = [n["id"] for n in roster["nodes"]]
    assert len(nodes) == len(set(nodes)) == roster["required_node_count"] == sum(SUITES.values())
    for suite, count in SUITES.items():
        assert roster["suites"][suite]["node_count"] == len(roster_nodes(roster, suite)) == count
        files = roster["suites"][suite]["files"]
        assert all(n["file"] in files for n in roster["nodes"] if n["suite"] == suite)
    assert roster["owner_roster"]["published"] is False
    for node in roster["nodes"]:
        assert node["id"].startswith(node["file"] + "::") and node["line"] > 0
        assert (node["tier"] == "correction") == bool(node["elements"])


@pytest.mark.parametrize("run", RUNS)
@pytest.mark.parametrize("suite", SUITES)
def test_every_required_node_passed_in_the_retained_junit(roster, run, suite):
    nodes, totals = junit_nodes(run_dir(run, suite) / "junit.xml")
    assert set(nodes) == set(roster_nodes(roster, suite))
    assert {outcome for outcome, _, _ in nodes.values()} == {"passed"}
    assert totals == {"tests": SUITES[suite], "failures": 0, "errors": 0, "skipped": 0}
    surface = json.loads((run_dir(run, suite) / "surface.json").read_text())
    assert surface["strict_cuda"] is True and surface["cuda"] is True
    assert surface["counts"] == {"passed": SUITES[suite], "failed": 0, "error": 0, "skipped": 0,
                                 "xfailed": 0, "xpassed": 0}
    assert surface["not_collected"] == [] and surface["skip_reasons"] == {}
    assert surface["cuda_surface"]["executed"] == sum(1 for _, _, cuda in nodes.values() if cuda)


@pytest.mark.parametrize("suite", SUITES)
def test_the_collected_nodes_equal_the_roster(roster, suite):
    expected = roster_nodes(roster, suite)
    assert json.loads((run_dir("run2", suite) / "expected-nodes.json").read_text()) == expected
    collected = [line.strip() for line in (run_dir("run2", suite) / "collect.txt").read_text().splitlines()
                 if line.startswith("tests/") and "::" in line]
    assert sorted(collected) == sorted(expected)


@pytest.mark.parametrize("suite", SUITES)
def test_the_per_node_results_equal_the_retained_junit(roster, results, suite):
    document = results[suite]
    run2, _ = junit_nodes(run_dir("run2", suite) / "junit.xml")
    run1, _ = junit_nodes(run_dir("run1", suite) / "junit.xml")
    assert set(document["nodes"]) == set(roster_nodes(roster, suite))
    by_id = {n["id"]: n for n in roster["nodes"]}
    classes = {"device-allocating": 0, "cuda-gated-no-allocation": 0, "host-only": 0}
    for node, record in document["nodes"].items():
        assert (record["outcome"], record["seconds"], record["cuda_allocated"]) == run2[node], node
        assert (record["run1"]["outcome"], record["run1"]["seconds"], record["run1"]["cuda_allocated"]) == run1[node]
        assert record["run1"]["cuda_allocated"] == record["cuda_allocated"], f"runs disagree on {node}"
        assert record["tier"] == by_id[node]["tier"] and record["cuda_gated"] == by_id[node]["cuda_gated"]
        expected_class = ("device-allocating" if record["cuda_allocated"] else
                          "cuda-gated-no-allocation" if record["cuda_gated"] else "host-only")
        assert record["gpu_class"] == expected_class
        classes[expected_class] += 1
    assert document["gpu_classes"] == classes
    assert document["counts"] == {"passed": SUITES[suite], "failed": 0, "error": 0, "skipped": 0}
    assert document["primary_run"]["cuda_surface_executed"] == classes["device-allocating"]


def test_a_cuda_gated_node_that_did_not_allocate_is_not_counted_as_gpu_proof(results):
    for suite, document in results.items():
        for node, record in document["nodes"].items():
            if record["gpu_class"] == "cuda-gated-no-allocation":
                function = node.split("::")[1]
                assert any(word in function for word in ("refuse", "requires", "does_not_allocate",
                                                         "integrity_gates", "before_first_parse")), node
    summary = load("results", "summary.json")
    assert summary["gpu_classes"] == {"device-allocating": 84, "cuda-gated-no-allocation": 18, "host-only": 34}
    assert sum(summary["gpu_classes"].values()) == summary["passed"] == 136


# ------------------------------------------------------------ corrections ------

def test_every_original_failure_maps_to_candidate_nodes(roster):
    corrections = load("corrections.json")
    nodes = {n["id"]: n for n in roster["nodes"]}
    expected = {610: 4, 611: 39}
    mapped = set()
    for record in corrections["corrections"]:
        issue, pr = record["issue"], record["fix_pr"]
        assert issue["state"] == "CLOSED" and pr["state"] == "MERGED" and pr["closes"] == [issue["number"]]
        assert issue["closed_at"] and pr["merged_at"] and len(pr["merge_commit"]) == 40
        ancestry = record["ancestor_of_candidate"]
        assert ancestry["status"] == "ahead" and ancestry["behind_by"] == 0
        assert ancestry["merge_base"] == pr["merge_commit"] and ancestry["ahead_by"] > 0
        assert record["green_receipt_at_fix"]["binds_candidate"] is False
        failures = record["original_failures"]
        assert len(failures) == expected[issue["number"]] == len({f["id"] for f in failures})
        for failure in failures:
            assert failure["candidate_nodes"], failure["id"]
            mapped.update(failure["candidate_nodes"])
        assert set(record["elements"]) and all(k.startswith(str(issue["number"])) for k in record["elements"])
    assert all(n in nodes for n in mapped)
    assert mapped == {n for n, v in nodes.items() if v["tier"] == "correction"} and len(mapped) == 54
    for node in mapped:
        issue = "610" if nodes[node]["suite"] == "moe" else "611"
        assert nodes[node]["elements"] and all(e.startswith(issue + ".") for e in nodes[node]["elements"])


def test_the_obsolete_launch_assertions_are_accounted_for():
    corrections = load("corrections.json")
    obsolete = corrections["obsolete_assertions"]
    assert obsolete["retired_tests"].startswith("None")
    assert "No node asserts that a served launch equals a retired symbol" in obsolete["status_at_candidate"]
    assert {e["item"].split()[0] for e in corrections["exclusions"]} >= {"tessera#638", "Full-model", "Native"}
    history = corrections["test_file_history_after_fix"]["files"]
    assert history["tests/test_native_window_moe.py"] == []
    assert len(history["tests/test_serving_bf16_gemv.py"]) == 3 and len(history["tests/test_serving_fp8_gemv.py"]) == 5


def test_every_correction_node_is_device_allocating_in_both_runs(results):
    for suite, document in results.items():
        for node, record in document["nodes"].items():
            if record["tier"] == "correction":
                assert record["cuda_allocated"] and record["run1"]["cuda_allocated"], node


# ------------------------------------------------------------ native path ------

@pytest.mark.parametrize("suite", SUITES)
def test_the_mapped_native_libraries_are_the_built_ones(results, suite):
    run = results[suite]["primary_run"]
    built = {entry["sha256"]: entry["path"] for entry in run["native_libraries_built"]}
    listed = {}
    for line in (run_dir("run2", suite) / "native-so.sha256").read_text().splitlines():
        digest, path = line.split("  ", 1)
        listed[digest] = path
    assert built == listed and run["mapped_digests_are_built_digests"] is True
    for worker, libraries in run["native_libraries_mapped_by_worker"].items():
        for module, info in libraries.items():
            assert info["sha256"] in built and info["nfs_renamed"] is False
            assert info["path"].startswith(f"torch-ext/{worker}/{module}_sm_121_tessera_guarded_v1/"), module
    assert not run["probe_errors"]
    for kernel_target in run["triton_targets"]:
        assert json.loads(kernel_target) == {"arch": 121, "backend": "cuda", "warp_size": 32}


def test_the_dense_run_mapped_the_fused_libraries_and_the_moe_run_mapped_none(results):
    dense = results["dense"]["primary_run"]
    by_worker = dense["native_libraries_mapped_by_worker"]
    owner = dense["worker_of_file"]
    bf16, = owner["tests/test_serving_bf16_gemv.py"]
    fp8, = owner["tests/test_serving_fp8_gemv.py"]
    assert bf16 != fp8
    assert set(by_worker[bf16]) == {"tessera_routed_fused_value", "tessera_window_gemv"}
    assert set(by_worker[fp8]) == {"tessera_routed_fused_mma_e4m3", "tessera_window_gemv"}
    moe = results["moe"]["primary_run"]
    assert all(not libraries for libraries in moe["native_libraries_mapped_by_worker"].values())
    assert moe["native_libraries_built"] == [] and moe["triton_kernels"]["_grouped_window_gemm_kernel"] > 0
    assert "fused_moe_kernel" in moe["triton_kernels"]


def test_the_stamped_dense_launches_match_the_lanes_the_tests_selected(results):
    nodes = results["dense"]["nodes"]
    stamped = {}
    for node, record in nodes.items():
        function, _, params = node.split("::")[1].partition("[")
        file = "bf16" if "bf16" in node.split("::")[0] else "fp8"
        for pair in record["stamped_launch"]:
            stamped.setdefault((file, tuple(pair)), []).append((function, params.rstrip("]")))
        if function == "test_decode_regime_serves_the_folded_native_window_gemm":
            want = FUSED_BF16 if params.endswith("fused]") else TRITON_BF16
            assert record["stamped_launch"] == [want], node
    counts = {key: len(value) for key, value in stamped.items()}
    assert counts == {("bf16", tuple(FUSED_BF16)): 12, ("bf16", tuple(TRITON_BF16)): 6,
                      ("fp8", tuple(FUSED_FP8_MMA)): 11}
    summary = load("results", "summary.json")["native_evidence"]["dense"]["stamped_launches"]
    assert sum(item["nodes"] for item in summary) == 29


def test_the_oracle_values_are_recorded_for_the_corrected_moe_nodes(results):
    nodes = results["moe"]["nodes"]
    for node, record in nodes.items():
        if "matches_the_oracle_fused_and_split" in node:
            assert len(record["float_comparisons"]) == 2, node
            for comparison in record["float_comparisons"]:
                measured, bound = comparison["evaluated"].split(" < ")
                assert float(measured) < float(bound)
        if "router_weight_on_input" in node or "native_loader_shape" in node:
            assert record["float_comparisons"], node
    for node, record in results["dense"]["nodes"].items():
        for comparison in record["float_comparisons"]:
            assert "<" in comparison["evaluated"] or ">" in comparison["evaluated"], node
    assert "max_abs=0.000e+00 max_rel=0.000e+00" in (run_dir("run2", "dense") / "pytest.log").read_text()


# --------------------------------------------------------------- receipts ------

def test_every_accepted_action_ended_green_on_gb10_in_image_x():
    document = load("results", "pb-receipts.json")
    summary = load("results", "summary.json")
    harness_head = summary["harness_head"]
    for run, action in document["actions"].items():
        assert action["state"] == "done" and action["returncode"] == 0 and action["attempts"] == 1, run
        assert action["container_images"] == [IMAGE]
        assert action["placement"]["required_tags"] == ["container-image-v1", "gb10"]
        assert len(action["action_key"]) == 64 and len(action["cas_receipt_sha256"]) == 64
        assert action["accelerators"][0]["compute_capability"] == "12.1"
        assert action["accelerators"][0]["driver_version"] == summary["driver_version"]
        if "preflight" in run:
            assert "gpu" not in action["demand"], run
        else:
            assert action["demand"]["gpu"] == 1, run
        if run.startswith("run2"):
            assert action["snapshot"]["parent"] == harness_head, run
    assert {a["action_key"] for r, a in document["actions"].items() if r in ("run2-moe", "run2-dense")} == \
        set(summary["primary_runs"].values())
    development = document["development_actions"]
    assert len(development) == 12 and set(summary["development_actions"]) == set(development)
    assert sorted(d["state"] for d in development.values()).count("failed") == 2


def test_the_summary_counts_agree_with_the_per_suite_results(results):
    summary = load("results", "summary.json")
    assert (summary["required_nodes"], summary["passed"], summary["failed"], summary["skipped"],
            summary["not_collected"]) == (136, 136, 0, 0, 0)
    assert summary["verdict"] == "complete" and summary["snapshot_clean"] is True
    assert summary["correction_nodes"] == {"total": 54, "passed": 54, "device_allocating": 54,
                                           "by_issue": {"610": 9, "611": 45}}
    assert summary["original_failures_covered"] == {"610": 4, "611": 39}
    for suite, document in results.items():
        assert summary["primary_runs"][suite] == document["primary_run"]["pb_action"]
        assert summary["corroborating_runs"][suite] == document["corroborating_run"]["pb_action"]


# ----------------------------------------------------------------- digests -----

def test_every_retained_log_matches_its_digest():
    digests = load("results", "log-digests.json")
    on_disk = {str(p.relative_to(PKG)) for p in LOGS.rglob("*") if p.is_file()}
    assert set(digests["files"]) == on_disk
    for name, entry in digests["files"].items():
        path = PKG / name
        assert sha256(path) == entry["sha256"] and path.stat().st_size == entry["bytes"], name
    for run in ("run2-moe", "run2-dense"):
        assert len(digests["raw_files_not_retained"][run]["raw_probe_tests_sha256"]) == 64


def test_the_manifest_binds_the_result_files():
    manifest = load("results", "candidate-manifest.json")
    for name, digest in manifest["files"].items():
        assert sha256(PKG / name) == digest, name
    assert {"results/moe.json", "results/dense.json", "results/summary.json"} <= set(manifest["files"])
    assert "requalify" in manifest["requalify_rule"]


# ------------------------------------------------------------- probe plugin ----

def _load_plugin():
    """The harness plugin, loaded by path so the test session's sys.path stays untouched."""
    spec = importlib.util.spec_from_file_location("pq1317_native_probe", HARNESS / "native_probe.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native_probe = _load_plugin()

MAPS = """\
ffff8c000000-ffff8c800000 r-xp 00000000 00:2a 123 /mnt/shared/m/torch-ext/gw1/tessera_routed_fused_value_sm_121_tessera_guarded_v1/.nfs0000000000f553e400000118
ffff8d000000-ffff8d100000 r--p 00000000 00:2a 124 /mnt/shared/m/torch-ext/gw1/tessera_window_gemv_sm_121_tessera_guarded_v1/tessera_window_gemv.so (deleted)
ffff8e000000-ffff8e100000 r--p 00000000 00:2a 125 /usr/lib/aarch64-linux-gnu/libc.so.6
ffff8f000000-ffff8f100000 r--p 00000000 00:2a 126 /home/x/.triton/cache/ABC/cuda_utils.cpython-312-aarch64-linux-gnu.so
ffff90000000-ffff90100000 r--p 00000000 00:2a 127 /mnt/shared/m/.nfs0000000000000001
"""


def test_the_probe_finds_tessera_libraries_even_when_nfs_renames_them():
    found = {entry["module"]: entry for entry in native_probe.parse_maps(MAPS.splitlines())}
    assert set(found) == {"tessera_routed_fused_value", "tessera_window_gemv"}
    assert found["tessera_routed_fused_value"]["nfs_renamed"] is True
    assert found["tessera_window_gemv"]["deleted"] is True and found["tessera_window_gemv"]["nfs_renamed"] is False


def test_the_probe_records_float_and_launch_assertions_and_skips_the_rest(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "probe_selftest.py").write_bytes((HARNESS / "probe_selftest.py").read_bytes())
    probe_dir, ext_root = tmp_path / "probe", tmp_path / "ext"
    environment = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(HARNESS), "PQ1317_PROBE_DIR": str(probe_dir),
                   "PQ1317_EXT_DIR_ROOT": str(ext_root), "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                   "HOME": str(tmp_path)}
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "native_probe", "-o", "enable_assertion_pass_hook=true",
         "-o", "verbosity_assertions=2", "-q", "probe_selftest.py"],
        cwd=tmp_path, env=environment, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    records = [json.loads(line) for line in (probe_dir / "probe.main.jsonl").read_text().splitlines()]
    assert not [r for r in records if r["kind"] == "error"]
    tests = {r["nodeid"]: r for r in records if r["kind"] == "test"}
    assert set(tests) == {f"probe_selftest.py::{name}" for name in (
        "test_a_float_assertion_is_recorded", "test_an_integer_assertion_is_not_recorded",
        "test_a_launch_pair_assertion_is_recorded", "test_the_extension_directory_belongs_to_the_worker")}
    assert tests["probe_selftest.py::test_a_float_assertion_is_recorded"]["assertions"][0]["evaluated"] == "0.125 < 0.25"
    assert tests["probe_selftest.py::test_an_integer_assertion_is_not_recorded"]["assertions"] == []
    pair = tests["probe_selftest.py::test_a_launch_pair_assertion_is_recorded"]["assertions"][0]["evaluated"]
    assert "'tessera::fused_window_dense'" in pair and "'native_fused_window_dense'" in pair
    assert [r for r in records if r["kind"] == "session"]
