#!/usr/bin/env python3
"""Container side of the pq1317 qualification harness. It runs inside image X.

``run_suite.py`` starts it with the Tessera source mounted read-only. Steps:

1. validate the pinned pytest tools already supplied by the image or a read-only tool directory;
2. bind the source: Tessera's own ``measured_source`` must say ``verified`` at the
   candidate commit, and the packaged runtime contract must hash to the pinned digest;
3. record the runtime: torch, CUDA device, vLLM, Triton, Tessera;
4. self-test the probe plugin on two synthetic tests under xdist;
5. collect the suite's nodes and compare them with the roster, exactly;
6. in ``run`` mode only: run the suite under ``--strict-cuda`` and merge the probe records.
   ``--dist loadfile`` fixes which worker runs which file, so process-level evidence
   (the mapped native libraries) belongs to one file.

Exit codes: 0 ok, 3 roster mismatch, 4 identity refused, 5 probe self-test failed,
6 unqualified probe evidence. A pytest failure returns pytest's own code; passing
tests alone cannot override failed attestation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

#: module name -> pin. The same pins attempt 2 and the Tessera owners used.
TEST_TOOLS = {
    "pytest": "pytest==8.4.2",
    "xdist": "pytest-xdist==3.8.0",
    "execnet": "execnet==2.1.1",
    "pluggy": "pluggy==1.6.0",
    "iniconfig": "iniconfig==2.1.0",
    "pygments": "Pygments==2.19.2",
    "packaging": "packaging==25.0",
}
CONTRACT_IN_TREE = "src/tessera/serving/runtime_contract.json"
WORKERS = 2
#: Without it pytest elides long values in an assertion's explanation: ('tessera::fu...window_dense').
ASSERTION_VERBOSITY = "verbosity_assertions=2"


def say(tag: str, payload) -> None:
    print(f"{tag} {json.dumps(payload, sort_keys=True)}", flush=True)


def install_test_tools(out: Path) -> dict:
    # Qualification must not resolve new dependencies from a package index.
    versions = {}
    for module, pin in TEST_TOOLS.items():
        distribution, expected = pin.split("==", 1)
        if importlib.util.find_spec(module) is None:
            raise RuntimeError(f"missing pre-provisioned test tool: {pin}")
        actual = importlib.metadata.version(distribution)
        # Rendering/version-parsing dependencies come from immutable image X,
        # as in the historical runs. Pin the supplied pytest/xdist toolchain.
        if module not in ("pygments", "packaging") and actual != expected:
            raise RuntimeError(f"test tool {distribution}: expected {expected}, got {actual}")
        versions[distribution] = actual
    return {"installed": [], "versions": versions,
            "pytest": versions["pytest"], "xdist": versions["pytest-xdist"]}


def bind_source(source: Path, candidate: str, contract_sha256: str) -> dict:
    from tessera._dev.suite_source import measured_source

    record = measured_source(source)
    contract = hashlib.sha256((source / CONTRACT_IN_TREE).read_bytes()).hexdigest()
    return {
        "suite_source": record,
        "contract_sha256": contract,
        "contract_matches_pin": contract == contract_sha256,
        "verified_at_candidate": (record.get("verification") == "verified"
                                  and record.get("snapshot_commit") == candidate),
    }


def runtime_facts() -> dict:
    facts: dict = {"python": sys.version.split()[0], "machine": platform.machine(),
                   "uid": os.getuid(), "gid": os.getgid(),
                   "tessera_environment": {k: v for k, v in sorted(os.environ.items())
                                           if k.startswith("TESSERA_")}}
    import torch

    facts.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                 cuda_available=torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        facts["device"] = {
            "name": props.name, "capability": list(torch.cuda.get_device_capability(0)),
            "total_memory": props.total_memory, "multi_processor_count": props.multi_processor_count,
            "uuid": str(getattr(props, "uuid", None)),
        }
    for module in ("vllm", "triton"):
        try:
            facts[module] = importlib.import_module(module).__version__
        except Exception as error:  # noqa: BLE001 -- recorded, not hidden
            facts[module] = f"unavailable: {error!r}"
    import tessera

    facts["tessera_file"] = tessera.__file__
    for dist in ("tessera-quant", "tessera"):
        try:
            facts["tessera_dist"] = {dist: importlib.metadata.version(dist)}
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    return facts


def pytest_command(out: Path, *args: str) -> list[str]:
    """pytest with its cache outside the read-only source tree."""
    return [sys.executable, "-m", "pytest", "-o", f"cache_dir={out / 'pytest-cache'}", *args]


def selftest_probe(out: Path) -> dict:
    """Run the probe on two synthetic tests under xdist and read its records back."""
    directory = out / "selftest"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (directory / "probe_selftest.py").write_bytes((Path(__file__).parent / "probe_selftest.py").read_bytes())
    probe_dir = out / "probe-selftest"
    ext_root = out / "selftest-ext"
    env = {**os.environ, "PQ1317_PROBE_DIR": str(probe_dir), "PQ1317_EXT_DIR_ROOT": str(ext_root)}
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "xdist.plugin", "-p", "native_probe", "-n", str(WORKERS),
         "-o", "enable_assertion_pass_hook=true", "-o", ASSERTION_VERBOSITY, "-q", "probe_selftest.py"],
        cwd=directory, env=env, capture_output=True, text=True)
    records = [json.loads(line) for path in sorted(probe_dir.glob("probe.*.jsonl"))
               for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tests = {r["nodeid"]: r for r in records if r.get("kind") == "test"}
    float_test = tests.get("probe_selftest.py::test_a_float_assertion_is_recorded", {})
    int_test = tests.get("probe_selftest.py::test_an_integer_assertion_is_not_recorded", {})
    pair_test = tests.get("probe_selftest.py::test_a_launch_pair_assertion_is_recorded", {})
    pair_text = " ".join(a["evaluated"] for a in pair_test.get("assertions", []))
    workers = sorted({r["worker"] for r in records if r.get("kind") == "test"})
    ok = (done.returncode == 0
          and len(float_test.get("assertions", [])) == 1
          and all(number in float_test["assertions"][0]["evaluated"] for number in ("0.125", "0.25"))
          and int_test.get("assertions") == []
          and "tessera::fused_window_dense" in pair_text and "native_fused_window_dense" in pair_text
          and sorted(p.name for p in ext_root.iterdir()) == ["gw0", "gw1"]
          and all(r.get("kind") != "error" for r in records))
    return {"ok": ok, "returncode": done.returncode, "workers": workers,
            "float_assertion": (float_test.get("assertions") or [None])[0],
            "stdout_tail": done.stdout[-400:]}


def collect_nodes(files: list[str], out: Path) -> list[str]:
    env = {**os.environ, "PQ1317_PROBE_DIR": str(out / "probe-collect")}
    done = subprocess.run(
        pytest_command(out, "-p", "native_probe", "--collect-only", "-q", *files),
        env=env, capture_output=True, text=True)
    (out / "collect.txt").write_text(done.stdout + done.stderr, encoding="utf-8")
    if done.returncode != 0:
        say("PQ1317_COLLECT_FAILED", {"returncode": done.returncode, "tail": done.stdout[-600:]})
    return [line.strip() for line in done.stdout.splitlines()
            if line.startswith("tests/") and "::" in line]


def probe_loaded_in_collection(out: Path) -> bool:
    """The probe wrote its session record during collection, so ``-p native_probe`` took effect."""
    return any('"kind": "session"' in path.read_text(encoding="utf-8")
               for path in (out / "probe-collect").glob("probe.*.jsonl"))


def compare_roster(collected: list[str], expected: list[str]) -> dict:
    missing = sorted(set(expected) - set(collected))
    unexpected = sorted(set(collected) - set(expected))
    duplicates = sorted({n for n in collected if collected.count(n) > 1})
    return {"expected": len(expected), "collected": len(collected), "missing": missing,
            "unexpected": unexpected, "duplicates": duplicates,
            "equal": not (missing or unexpected or duplicates) and len(collected) == len(expected)}


def run_suite(files: list[str], out: Path) -> int:
    command = pytest_command(
        out, "-p", "xdist.plugin", "-p", "native_probe", "-n", str(WORKERS), "--dist", "loadfile",
        "--durations=20", "--strict-cuda", "--surface-json", str(out / "surface.json"),
        "--basetemp", str(out / "tmp" / "pytest"), "--junitxml", str(out / "junit.xml"),
        "-o", "enable_assertion_pass_hook=true", "-o", ASSERTION_VERBOSITY, "-o", "log_level=INFO",
        "-rP", *files)
    env = {**os.environ, "PQ1317_PROBE_DIR": str(out / "probe"), "PQ1317_EXT_DIR_ROOT": str(out / "torch-ext")}
    say("PQ1317_PYTEST_COMMAND", command)
    with (out / "pytest.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log.write(line)
        returncode = process.wait()
    surface = out / "surface.json"
    say("PQ1317_SURFACE", json.loads(surface.read_text()) if surface.exists() else {"missing": str(surface)})
    return returncode


def merge_probe(out: Path, expected: list[str]) -> dict:
    # Import only when merging so this module can still be inspected by path.
    from native_probe import FILE_IDENTITY, native_identity_errors
    from bf16_oracle import bf16_oracle_errors

    records = [json.loads(line) for path in sorted((out / "probe").glob("probe.*.jsonl"))
               for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    tests = [r for r in records if r.get("kind") == "test"]
    sessions = [r for r in records if r.get("kind") == "session"]
    errors = [r for r in records if r.get("kind") == "error"]
    for name, data in (("probe-tests.json", tests), ("probe-sessions.json", sessions)):
        (out / name).write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    def refuse(message: str) -> None:
        errors.append({"kind": "error", "where": "merge_probe", "error": message})

    roster = compare_roster([r["nodeid"] for r in tests], expected)
    if not roster["equal"]:
        refuse(f"probe node roster mismatch: {roster}")
    workers = {f"gw{i}" for i in range(WORKERS)}
    if {r["worker"] for r in tests} != workers:
        refuse("probe did not record all test workers")
    for worker in workers:
        ended = [r for r in sessions if r["worker"] == worker and r.get("executes_tests") is True]
        if len(ended) != 1 or ended[0].get("exitstatus") != 0:
            refuse(f"{worker}: missing or unsuccessful probe session")
    for record in tests:
        if record.get("when") != "call" or record.get("outcome") != "passed":
            refuse(f"{record['nodeid']}: required node did not pass its call")
        if record.get("assertions_dropped") != 0:
            refuse(f"{record['nodeid']}: assertions were dropped")

    for error in bf16_oracle_errors(tests, expected):
        refuse(error)
    by_worker: dict[str, dict[str, dict]] = {}
    seen: dict[str, set] = {}
    identities: dict[tuple, tuple] = {}
    for record in tests + sessions:
        worker = record["worker"]
        libraries = record.get("native_libraries_mapped")
        if not isinstance(libraries, dict):
            refuse(f"{worker}: missing native mapping inventory")
            continue
        for error in native_identity_errors(libraries):
            refuse(f"{worker}: {record.get('nodeid', 'session')}: {error}")
        inventory = by_worker.setdefault(worker, {})
        for module, info in libraries.items():
            # Every observation participates. Never let a later valid record hide
            # an earlier failure, or let dict.update erase a changed binary.
            if native_identity_errors({module: info}):
                continue
            mapping = info["mappings"][0]
            identity = (info["sha256"], *(mapping[k] for k in FILE_IDENTITY))
            key = (worker, module)
            if key in identities and identities[key] != identity:
                refuse(f"{worker}: {module}: mapped backing file changed during the run")
            identities[key] = identity
            inventory.setdefault(module, info)
            seen.setdefault(module, set()).add(info["sha256"])
    summary = {
        "qualification_ok": not errors,
        "tests_recorded": len(tests),
        "workers": sorted({r["worker"] for r in tests}),
        "errors": errors,
        "assertions_recorded": sum(len(r["assertions"]) for r in tests),
        "assertions_dropped": sum(r["assertions_dropped"] for r in tests),
        "native_libraries_by_worker": {w: dict(sorted(libs.items())) for w, libs in sorted(by_worker.items())},
        "native_library_sha256_seen": {m: sorted(h) for m, h in sorted(seen.items())},
    }
    (out / "probe-summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--suite", required=True)
    parser.add_argument("--mode", choices=("collect", "run"), required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--nodes-json", required=True, type=Path,
                        help="JSON list of the node ids the suite must collect")
    args = parser.parse_args(argv)

    expected = json.loads(args.nodes_json.read_text(encoding="utf-8"))
    files = list(dict.fromkeys(node.split("::", 1)[0] for node in expected))
    report: dict = {"schema": "pq1317.container.v1", "suite": args.suite, "mode": args.mode,
                    "candidate": args.candidate, "files": files, "started_unix": time.time()}

    report["test_tools"] = install_test_tools(args.out)
    say("PQ1317_TEST_TOOLS", report["test_tools"])

    report["identity"] = bind_source(args.source, args.candidate, args.contract_sha256)
    say("PQ1317_IDENTITY", report["identity"])
    report["runtime"] = runtime_facts()
    say("PQ1317_RUNTIME", report["runtime"])

    def finish(code: int) -> int:
        report.update(exit_code=code, finished_unix=time.time())
        (args.out / "container-report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n",
                                                        encoding="utf-8")
        say("PQ1317_CONTAINER_EXIT", {"exit_code": code})
        return code

    identity = report["identity"]
    if not (identity["verified_at_candidate"] and identity["contract_matches_pin"]):
        say("PQ1317_REFUSED", {"reason": "source or contract identity does not match the candidate"})
        return finish(4)

    report["probe_selftest"] = selftest_probe(args.out)
    say("PQ1317_PROBE_SELFTEST", report["probe_selftest"])
    if not report["probe_selftest"]["ok"]:
        return finish(5)

    collected = collect_nodes(files, args.out)
    report["probe_loaded_in_collection"] = probe_loaded_in_collection(args.out)
    report["roster_check"] = compare_roster(collected, expected)
    say("PQ1317_ROSTER_CHECK", {k: v for k, v in report["roster_check"].items() if k != "expected"})
    if not (report["roster_check"]["equal"] and report["probe_loaded_in_collection"]):
        return finish(3)
    if args.mode == "collect":
        return finish(0)

    code = run_suite(files, args.out)
    report["probe"] = merge_probe(args.out, expected)
    say("PQ1317_PROBE", report["probe"])
    return finish(code or (0 if report["probe"]["qualification_ok"] else 6))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
