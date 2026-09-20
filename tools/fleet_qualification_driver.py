"""Executable fleet qualification driver (level 2, component scope).

Builds the hardware-phase qualification plan from declared pins and, with
``--submit``, EXECUTES the runnable component cases and writes an
attributable report.

Mechanism (all published interfaces, no invented flags):

- One ``pbtest.py`` invocation per required host with a host-tag
  dependency (``--tag sparky`` / ``--tag sparklina``): the same small
  integration suite is given to PB on each host. Tags are sealed into
  action identity, so each host executes fresh shard actions with full
  evidence; PB still owns admission, execution, and fanout, and the
  driver never assigns work to hosts itself. Both invocations are
  admitted together, never serially awaited.
- In-shard exact-pin guards run automatically (pbtest executes the
  checkout's ``tools/resolve_*_dev_pin.py`` resolvers and refuses
  provenance mismatches), so source qualification uses PB's real
  machinery on each host.
- Per shard, the driver verifies the existing PB artifacts instead of
  trusting summary text: the CAS blob in the shard output names the
  action key; the CAS payload file must parse with a matching receipt;
  the queue terminal record must name the same action, report
  ``executed`` with returncode 0, show the REQUESTED host, the DECLARED
  source (snapshot parent), and the DECLARED runtime generation; and the
  pytest counts must show a nonzero collection with zero failures.
  Wrong action/source/generation, missing receipt, empty collection,
  failed/unrelated terminals, and unattributable cached replays never
  become qualified.

``--submit`` refuses dirty checkouts and declared pins that do not match
the checkout HEAD (exit 2): unattributable snapshots must not run.
Exit 1 on failed cases or driver errors, else 0. Completeness is NEVER
claimed here: the report always carries ``complete: false`` with the
incomplete reasons (unimplemented legs, uncovered hosts). Skips inside
a passing case are the harness's named nonqualified legs, recorded,
never green conformance. Do NOT run live membership/deploy here.

``--print-plan`` emits the exact invocations without executing them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve()
CHECKOUT = HERE.parents[1]
PBTEST = Path("/mnt/shared/prismabuild-fleet/repo/tools/pbtest.py")

#: Fresh exact-pin env (PQ846 strict PB reader SDK 461728e4). Never modify
#: a shared active env; this path names the provisioned one. It must exist
#: on every worker the cases land on -- a missing interpreter is a named
#: driver error, never a silent substitution.
PYTHON = "/home/rob/venvs/pq846-pb461728e4/bin/python"

ENTRYPOINTS = ("tools/pbtest.py", "tools/pbcampaign.py", "tools/pbrun.py")
SUPPORTED_CLIENT = "tools/pbtest.py"

#: Hardware coverage is the purpose: both Sparks are required dependencies
#: of a complete report, each served by its own tagged client invocation.
HOSTS = ("sparky", "sparklina")

TAG = "gb10"
PRIORITY = -10
WAIT_S = 1500
MEM_GB = 8

#: Runnable component cases: (name, files). One pbtest invocation runs all
#: files per host; distinct files fan out to distinct shard actions.
COMPONENT_CASES: tuple = (
    ("pins", ("tests/test_fleet_acceptance_pins.py",)),
    ("runner-table", ("tests/test_fleet_acceptance_runner.py",)),
    ("matrix", ("tests/test_fleet_acceptance_matrix.py",)),
    ("driver-self", ("tests/test_fleet_qualification_driver.py",)),
    ("level1-connected", ("tests/test_fleet_acceptance_level1.py",)),
)
CASE_FILES: tuple = tuple(f for _, files in COMPONENT_CASES for f in files)

#: Legs that cannot run yet: explicit records, never invocations.
UNIMPLEMENTED: tuple = (
    ("membership-join-resign-handoff",
     "PB728 published membership API (JOIN/RESIGN/chained handoff); "
     "no live membership mutation until root authorizes the acceptance "
     "phase"),
    ("native-container-readers",
     "no container validation claimed: a --help text is not reader "
     "validation and --image is not this PB client's container contract; "
     "requires the accepted reader SDK, its image contract, and root's "
     "hardware-container acceptance"),
)

REPORT_SCHEMA = "prismaquant.fleet_qualification.report.v1"
PLAN_SCHEMA = "prismaquant.fleet_qualification.plan.v1"


def _case_argv(*, host: str, out_json: Path, checkout: str,
               python: str) -> list[str]:
    """The exact published-client invocation for one host's suite."""
    return [str(PBTEST),
            "--checkout", checkout,
            "--python", python,
            "--tag", host,
            "--priority", str(PRIORITY),
            "--shards", str(len(COMPONENT_CASES)),
            "--workers-per-shard", "1",
            "--threads-per-shard", "1",
            "--mem-gb", str(MEM_GB),
            "--wait-s", str(WAIT_S),
            "--json", str(out_json),
            *[str(f) for f in CASE_FILES]]


def build_plan(*, pins: dict, checkout: str, python: str = PYTHON,
               out_dir: str, priority: int = PRIORITY) -> dict:
    """The executable plan: one runnable row per host plus records."""
    out = Path(out_dir)
    runs = []
    for host in HOSTS:
        argv = _case_argv(host=host,
                          out_json=out / f"{host}.pbtest.json",
                          checkout=checkout, python=python)
        if priority != PRIORITY:
            argv[argv.index("--priority") + 1] = str(priority)
        runs.append({"host": host, "client": SUPPORTED_CLIENT,
                     "argv": argv,
                     "receipt": str(out / f"{host}.pbtest.json")})
    return {"schema": PLAN_SCHEMA, "pins": pins,
            "checkout": checkout, "python": python,
            "required_hosts": list(HOSTS),
            "cases": [{"name": name, "files": list(files)}
                      for name, files in COMPONENT_CASES],
            "runs": runs,
            "unimplemented": [{"name": name, "reason": reason}
                              for name, reason in UNIMPLEMENTED]}


def format_invocations(plan: dict) -> list[str]:
    """Exact shell lines per host; records for the rest."""
    lines = []
    for run in plan["runs"]:
        lines.append(f"# host: {run['host']}")
        lines.append("python3 " + shlex.join(run["argv"]))
    for record in plan["unimplemented"]:
        lines.append(f"# unimplemented {record['name']}: {record['reason']}")
    return lines


def _is_hex(value: object, length: int) -> bool:
    return (isinstance(value, str) and len(value) == length
            and re.fullmatch(r"[0-9a-f]+", value) is not None)


def _counts(summary: object) -> dict[str, int] | None:
    """Pass/fail/skip/error counts, or None on malformed input."""
    if not isinstance(summary, str):
        return None
    out = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for match in re.finditer(r"(\d+)\s+(passed|failed|skipped|error)\b",
                             summary):
        out[match.group(2)] += int(match.group(1))
    return out


def _balanced_objects(text: str) -> list[dict]:
    """Top-level ``{...}`` JSON objects in order (brace matching)."""
    found = []
    depth, start = 0, -1
    for pos, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = pos
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    doc = json.loads(text[start:pos + 1])
                except ValueError:
                    pass
                else:
                    if isinstance(doc, dict):
                        found.append(doc)
                start = -1
    return found


def _cas_blob(output: object) -> dict | None:
    """The shard's CAS blob: receipt action key, payload, status.

    Real blobs nest (``receipt.action_key``); the extractor brace-matches
    instead of assuming a flat shape, and rejects anything malformed.
    """
    if not isinstance(output, str):
        return None
    for doc in _balanced_objects(output):
        receipt = doc.get("receipt")
        if (isinstance(receipt, dict)
                and _is_hex(receipt.get("action_key"), 64)
                and isinstance(doc.get("payload_path"), str)
                and isinstance(doc.get("status"), str)):
            return {"action_key": receipt["action_key"],
                    "receipt_sha256": doc.get("receipt_sha256", ""),
                    "payload_path": doc["payload_path"],
                    "status": doc["status"]}
    return None


def _read_json_file(path: str) -> dict | list | None:
    try:
        with open(path) as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def _read_bytes(path: str) -> bytes | None:
    try:
        with open(path, "rb") as stream:
            return stream.read()
    except OSError:
        return None


def _cas_text(payload_path: object, key: str) -> tuple[str | None, str]:
    """The CAS-stored shard result, validated by the existing protocol.

    Content-address integrity (sha256(content) == basename), non-empty,
    decodable text. Returns ``(text, "")`` or ``(None, reason)``.
    """
    if not isinstance(payload_path, str) or not payload_path:
        return None, "CAS payload unreadable"
    data = _read_bytes(payload_path)
    if not data:
        return None, "CAS payload unreadable"
    if hashlib.sha256(data).hexdigest() != Path(payload_path).name:
        return None, "CAS integrity mismatch"
    try:
        return data.decode("utf-8"), ""
    except ValueError:
        return None, "CAS payload unreadable"


def _generation_of_terminal(doc: dict) -> str | None:
    """Deployed generation id from the terminal's runtime paths."""
    detail = doc.get("detail")
    argv = detail.get("argv") if isinstance(detail, dict) else None
    if not isinstance(argv, list):
        return None
    for word in argv:
        if not isinstance(word, str):
            continue
        match = re.search(r"runtime-generations/([^/\"'\s]+)", word)
        if match:
            return match.group(1)
    return None


def verify_shard(*, shard: object, host: str, declared: dict) -> dict:
    """qualified / nonqualified / failed for one shard on one host.

    Every check reads existing PB artifacts; malformed objects are
    rejected, never coerced. Only a fully attributed chain qualifies.
    """
    def nope(status: str, reason: str, **extra: object) -> dict:
        base: dict = {"host": host, "status": status, "reason": reason,
                      "passed": 0, "failed": 0, "skipped": 0,
                      "action_key": "", "terminal": "",
                      "snapshot_commit": "", "snapshot_parent": "",
                      "generation": ""}
        base.update(extra)
        return base

    if not isinstance(shard, dict):
        return nope("failed", "malformed shard record: not an object")
    files = shard.get("files")
    output = shard.get("output")
    returncode = shard.get("returncode")
    if (not isinstance(files, list) or not files
            or not all(isinstance(f, str) for f in files)):
        return nope("failed", "malformed shard record: files")
    if not isinstance(output, str) or not isinstance(returncode, int):
        return nope("failed", "malformed shard record: output/returncode")
    counts = _counts(shard.get("summary"))
    if counts is None:
        return nope("failed", "malformed shard record: summary")
    # The submission record's returncode gates first; verdict counts come
    # from the CAS-stored result text below, never console rendering.
    if returncode != 0:
        return nope("failed", f"returncode={returncode}")
    blob = _cas_blob(output)
    if blob is None:
        return nope("nonqualified",
                    "unattributed cached replay: no CAS blob with action "
                    "key; resubmission at a new snapshot required for "
                    "fresh terminals")
    key = blob["action_key"]
    if blob.get("status") != "published":
        return nope("failed", "CAS receipt not published",
                    action_key=key)
    text, problem = _cas_text(blob.get("payload_path"), key)
    if problem:
        return nope("failed", problem, action_key=key)
    counts = _counts(text)
    if counts is None:
        return nope("failed", "CAS payload unreadable", action_key=key)
    if counts["failed"] or counts["error"]:
        return nope("failed",
                    f"failed={counts['failed']} errors={counts['error']}",
                    action_key=key,
                    passed=counts["passed"], failed=counts["failed"],
                    skipped=counts["skipped"])
    if not counts["passed"]:
        return nope("nonqualified", "no passing tests collected",
                    action_key=key, skipped=counts["skipped"])
    term = _read_json_file(
        f"/mnt/shared/prismabuild-fleet/pb-queue/done/{key}.json")
    side = "done"
    if term is None:
        term = _read_json_file(
            f"/mnt/shared/prismabuild-fleet/pb-queue/failed/{key}.json")
        side = "failed"
    if not isinstance(term, dict):
        return nope("failed", "no terminal for recorded action",
                    action_key=key, passed=counts["passed"],
                    skipped=counts["skipped"])
    if term.get("action_key") != key:
        return nope("failed", "stale unrelated terminal",
                    action_key=key, passed=counts["passed"],
                    skipped=counts["skipped"])
    if side != "done" or term.get("status") != "executed":
        return nope("failed",
                    f"terminal side={side} status={term.get('status')}",
                    action_key=key, passed=counts["passed"],
                    skipped=counts["skipped"])
    detail = term.get("detail")
    if (not isinstance(detail, dict)
            or detail.get("returncode") != 0):
        return nope("failed", "terminal returncode not zero",
                    action_key=key, passed=counts["passed"],
                    skipped=counts["skipped"])
    if term.get("finished_host") != host:
        return nope("failed",
                    f"wrong host: terminal served by "
                    f"{term.get('finished_host')}",
                    action_key=key, passed=counts["passed"],
                    skipped=counts["skipped"])
    snapshot = term.get("checkout_snapshot")
    parent = snapshot.get("parent") if isinstance(snapshot, dict) else None
    commit = snapshot.get("commit") if isinstance(snapshot, dict) else None
    if parent != declared.get("pq_head"):
        return nope("failed",
                    f"source mismatch: declared {declared.get('pq_head')} "
                    f"vs executed parent {parent}",
                    action_key=key, passed=counts["passed"],
                    skipped=counts["skipped"],
                    snapshot_commit=commit or "",
                    snapshot_parent=parent or "")
    generation = _generation_of_terminal(term)
    if generation != declared.get("generation"):
        return nope("failed",
                    f"generation mismatch: declared "
                    f"{declared.get('generation')} vs {generation}",
                    action_key=key, passed=counts["passed"],
                    skipped=counts["skipped"],
                    snapshot_commit=commit or "",
                    snapshot_parent=parent or "")
    return {"host": host, "status": "qualified",
            "reason": (f"{counts['skipped']} named nonqualified skip(s)"
                       if counts["skipped"] else ""),
            "passed": counts["passed"], "failed": 0,
            "skipped": counts["skipped"], "action_key": key,
            "terminal": f"/mnt/shared/prismabuild-fleet/pb-queue/done/"
                        f"{key}.json",
            "snapshot_commit": commit or "",
            "snapshot_parent": parent or "",
            "generation": generation or ""}


def exit_code_for(report: dict) -> int:
    """1 on failed cases or driver errors, else 0.

    Unimplemented legs and host-coverage gaps are reported
    nonqualification, never hidden -- but they are not failures, so a
    fully executed component run with named gaps still exits zero while
    ``complete`` stays false.
    """
    failed = [c["name"] for c in report["cases"]
              if c["status"] == "failed"]
    return 1 if (failed or report.get("driver_errors")) else 0


def assemble_report(*, plan: dict, verdicts: list[dict],
                    started_unix: float) -> dict:
    """Report from verified cases; completeness never claimed here."""
    by_host: dict[str, list[dict]] = {}
    for verdict in verdicts:
        by_host.setdefault(verdict["host"], []).append(verdict)
    covered = sorted(
        host for host in plan["required_hosts"]
        if host in by_host and by_host[host]
        and all(v["status"] == "qualified" for v in by_host[host]))
    missing = [h for h in plan["required_hosts"] if h not in covered]
    incomplete = [f"unimplemented leg: {r['name']}"
                  for r in plan["unimplemented"]]
    if missing:
        incomplete.append(
            "host coverage gap: no qualified suite on "
            + ", ".join(missing))
    cases = []
    for verdict in verdicts:
        record = dict(verdict)
        record["name"] = (f"{verdict.get('file', '')}"
                          f"@{verdict['host']}")
        cases.append(record)
    return {"schema": REPORT_SCHEMA, "pins": plan["pins"],
            "driver": {"checkout": plan["checkout"],
                       "python": plan["python"],
                       "host": socket.gethostname(),
                       "started_unix": started_unix,
                       "finished_unix": time.time()},
            "cases": cases,
            "unimplemented": plan["unimplemented"],
            "required_hosts": list(plan["required_hosts"]),
            "covered_hosts": covered, "missing_hosts": missing,
            "complete": False, "incomplete_reasons": incomplete}


def _pins_module() -> object:
    sys.path.insert(0, str(HERE.parents[1] / "tests"))
    try:
        import fleet_acceptance_pins as pins
        return pins
    finally:
        sys.path.remove(str(HERE.parents[1] / "tests"))


def check_clean_checkout(checkout: str, pins_doc: dict) -> dict:
    """Refuse unattributable snapshots: dirty tree or pins mismatch.

    Returns the executed source pins (PQ HEAD + PB rev). Raises
    ``SystemExit(2)`` on refusal with the reason on stderr.
    """
    pins = _pins_module()
    try:
        snapshots = pins.record_snapshots(checkout=Path(checkout))
    except pins.NonQualified as exc:
        raise SystemExit(f"refusing unattributable snapshot: {exc.reason}")
    if snapshots["pq_head"] != pins_doc.get("pq", {}).get("head"):
        raise SystemExit(
            "refusing pins mismatch: declared pq.head "
            f"{pins_doc.get('pq', {}).get('head')} vs checkout "
            f"{snapshots['pq_head']}")
    if pins.PB_CANDIDATE_REV != pins_doc.get("pb_candidate", {}).get("rev"):
        raise SystemExit(
            "refusing pins mismatch: declared pb_candidate.rev "
            f"{pins_doc.get('pb_candidate', {}).get('rev')} vs harness "
            f"{pins.PB_CANDIDATE_REV}")
    if not pins_doc.get("published_generation", {}).get("generation"):
        raise SystemExit("refusing pins mismatch: no published_generation")
    return {"pq_head": snapshots["pq_head"],
            "pb_rev": pins.PB_CANDIDATE_REV,
            "generation": pins_doc["published_generation"]["generation"]}


def submit(plan: dict, out_dir: str, declared: dict) -> tuple[dict, int]:
    """Execute one tagged suite per host together; write the report.

    Returns ``(report, exit_code)``: 1 on failed cases or driver errors,
    else 0. A zero exit with ``complete: false`` is normal today: it
    means the machinery worked and the gaps are named, not that the
    fleet is qualified.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    procs: dict[str, object] = {}
    for run in plan["runs"]:
        log_path = out / f"{run['host']}.submit.txt"
        log_path.write_text(
            "$ python3 " + shlex.join(
                [sys.executable, *run["argv"]]) + "\n")
        procs[run["host"]] = subprocess.Popen(
            [sys.executable, *run["argv"]],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    receipts: dict[str, list[dict] | None] = {}
    errors: list[str] = []
    for run in plan["runs"]:
        host = run["host"]
        log_path = out / f"{host}.submit.txt"
        try:
            proc = procs[host]
            assert isinstance(proc, subprocess.Popen)
            stdout, stderr = proc.communicate(timeout=WAIT_S + 900)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{host}: submission error: {exc!r}")
            continue
        with open(log_path, "a") as stream:
            stream.write(f"returncode={proc.returncode}\n"
                         f"--- stdout ---\n{stdout[-4000:]}\n"
                         f"--- stderr ---\n{stderr[-2000:]}")
        receipt_path = out / f"{host}.pbtest.json"
        data = _read_json_file(str(receipt_path))
        if proc.returncode != 0 and data is None:
            errors.append(
                f"{host}: pbtest exited {proc.returncode} with no receipt")
            continue
        receipts[host] = data if isinstance(data, list) else (
            [data] if isinstance(data, dict) else None)
        if receipts[host] is None:
            errors.append(f"{host}: receipt unreadable")
    verdicts: list[dict] = []
    for run in plan["runs"]:
        host = run["host"]
        shards = receipts.get(host)
        if not shards:
            continue
        by_file: dict[str, list[dict]] = {}
        for shard in shards:
            files = shard.get("files") if isinstance(shard, dict) else None
            if (not isinstance(files, list) or len(files) != 1
                    or not isinstance(files[0], str)):
                verdicts.append({"host": host, "file": "?",
                                 "status": "failed",
                                 "reason": "unexpected fanout",
                                 "passed": 0, "failed": 0, "skipped": 0,
                                 "action_key": "", "terminal": "",
                                 "snapshot_commit": "", "snapshot_parent": "",
                                 "generation": ""})
                continue
            by_file.setdefault(files[0], []).append(shard)
        for filename, group in sorted(by_file.items()):
            if len(group) != 1:
                verdicts.append({"host": host, "file": filename,
                                 "status": "failed",
                                 "reason": "unexpected fanout",
                                 "passed": 0, "failed": 0, "skipped": 0,
                                 "action_key": "", "terminal": "",
                                 "snapshot_commit": "", "snapshot_parent": "",
                                 "generation": ""})
                continue
            verdict = verify_shard(shard=group[0], host=host,
                                   declared=declared)
            verdict["file"] = filename
            verdicts.append(verdict)
    report = assemble_report(plan=plan, verdicts=verdicts,
                             started_unix=started)
    if errors:
        report["driver_errors"] = errors
    (out / "report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n")
    return report, exit_code_for(report)


def print_summary(report: dict) -> None:
    """Human lines for the console; gaps are loud, never hidden."""
    for case in report["cases"]:
        flag = {"qualified": "QUALIFIED", "nonqualified": "NONQUALIFIED"}.get(
            case["status"], "FAILED")
        hosts = case["host"]
        print(f"{flag} {case['name']}: "
              f"passed={case['passed']} failed={case['failed']} "
              f"skipped={case['skipped']} hosts={hosts} "
              f"{case['action_key'][:8]} {case['reason']}")
    for record in report["unimplemented"]:
        print(f"NONQUALIFIED {record['name']}: {record['reason']}")
    if report["missing_hosts"]:
        print("NONQUALIFIED host coverage gap: "
              + ", ".join(report["missing_hosts"]))
    if report.get("driver_errors"):
        for line in report["driver_errors"]:
            print(f"DRIVER-ERROR {line}")
    print("COMPLETE: false (" + "; ".join(report["incomplete_reasons"])
          + ")")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", required=True,
                        help="declared pins JSON (PQ HEAD, PB rev, generation)")
    parser.add_argument("--checkout", default=str(CHECKOUT))
    parser.add_argument("--python", default=PYTHON)
    parser.add_argument("--out-dir", required=True,
                        help="unique attributable outputs land here")
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    pins_doc = json.loads(Path(args.pins).read_text())
    plan = build_plan(pins=pins_doc, checkout=args.checkout,
                      python=args.python, out_dir=args.out_dir)
    if args.print_plan and not args.submit:
        print("\n".join(format_invocations(plan)))
        return 0
    if args.submit and not args.print_plan:
        try:
            declared = check_clean_checkout(args.checkout, pins_doc)
        except SystemExit as exc:
            print(f"DRIVER-ERROR {exc}")
            return 2
        report, exit_code = submit(plan, args.out_dir, declared)
        print_summary(report)
        return exit_code
    parser.error("exactly one of --print-plan or --submit is required")


if __name__ == "__main__":
    raise SystemExit(main())
