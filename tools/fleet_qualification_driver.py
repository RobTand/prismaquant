"""Executable fleet qualification driver (level 2, component scope).

Builds the hardware-phase qualification plan from declared pins and, with
``--submit``, EXECUTES the runnable component cases through the published
PB clients only and writes an attributable report.

Mechanism per case (all published interfaces, no invented flags):

- ``pbrun.py`` with real bounds (``--cwd`` checkout, ``--demand``,
  ``--cpus``, ``--tag gb10``, ``--priority -10``, ``--wait-s``,
  ``OMP/MKL/OPENBLAS_NUM_THREADS=1``, ``--detach``) runs
  ``<python> -m pytest <files>``. The detach line always names the
  action key and the done/failed terminal paths -- even on ``cache_hit``,
  which submits nothing and reuses the recorded terminal.
- ``pbwait.py`` waits for the terminal; the driver then reads the
  done/failed record itself: ``status``, ``finished_host`` (hardware
  attribution), ``returncode``, full pytest stdout (pass/fail/skip
  counts), and the CAS payload path.

Placement stays PB-owned; the driver never schedules, shards, or pins
hosts. It ATTRIBUTES serving hosts from terminal records and reports
uncovered required hosts as nonqualified gaps. Cases that cannot run yet
(membership JOIN/RESIGN/handoff, native-container readers) are explicit
``unimplemented`` case records with reasons -- never runnable
invocations, never ``--help``/``--image`` pseudo-validation.

``--submit`` exit code: 1 on failed cases or driver errors, 0 otherwise.
Completeness is NEVER claimed here: the report always carries
``complete: false`` with the incomplete reasons. Skips inside a passing
case are the harness's named nonqualified legs, recorded, never green
conformance. Do NOT run live membership/deploy from this driver.

``--print-plan`` emits the exact invocations without executing them.
"""
from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve()
CHECKOUT = HERE.parents[1]
PBRUN = Path("/mnt/shared/prismabuild-fleet/repo/tools/pbrun.py")
PBWAIT = Path("/mnt/shared/prismabuild-fleet/repo/tools/pbwait.py")

#: Fresh exact-pin env (PQ846 strict PB reader SDK 461728e4). Never modify
#: a shared active env; this path names the provisioned one. It must exist
#: on every worker the cases land on -- a missing interpreter is a named
#: driver error, never a silent substitution.
PYTHON = "/home/rob/venvs/pq846-pb461728e4/bin/python"

ENTRYPOINTS = ("tools/pbtest.py", "tools/pbcampaign.py", "tools/pbrun.py")
SUPPORTED_CLIENT = "tools/pbrun.py"
SUPPORTED_WAITER = "tools/pbwait.py"

#: Hardware coverage is the purpose: both Sparks are required dependencies
#: of a complete report. The driver attributes serving hosts from terminal
#: records and reports uncovered hosts as nonqualified gaps.
HOSTS = ("sparky", "sparklina")

TAG = "gb10"
PRIORITY = -10
WAIT_S = 1500
MEM_GB = 8
CPUS = 1
THREAD_ENV = (("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"),
              ("OPENBLAS_NUM_THREADS", "1"))

#: Runnable component cases: (name, files). One action per case; distinct
#: commands are distinct actions, so the fleet distributes them and host
#: coverage emerges from terminal records.
COMPONENT_CASES: tuple = (
    ("pins", ("tests/test_fleet_acceptance_pins.py",)),
    ("runner-table", ("tests/test_fleet_acceptance_runner.py",)),
    ("matrix", ("tests/test_fleet_acceptance_matrix.py",)),
    ("driver-self", ("tests/test_fleet_qualification_driver.py",)),
    ("level1-connected", ("tests/test_fleet_acceptance_level1.py",)),
)

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


def _case_argv(*, files: tuple[str, ...], checkout: str,
               python: str) -> list[str]:
    """The exact published-client invocation for one component case."""
    argv = [str(PBRUN),
            "--cwd", checkout,
            "--demand", f"mem_gb={MEM_GB}",
            "--cpus", str(CPUS),
            "--tag", TAG,
            "--priority", str(PRIORITY),
            "--wait-s", str(WAIT_S)]
    for key, value in THREAD_ENV:
        argv += ["--env", f"{key}={value}"]
    argv += ["--detach", "--", python, "-m", "pytest",
             *[str(f) for f in files]]
    return argv


def build_plan(*, pins: dict, checkout: str, python: str = PYTHON,
               out_dir: str, tag: str = TAG,
               priority: int = PRIORITY) -> dict:
    """The executable plan: runnable rows plus unimplemented records."""
    cases = []
    for name, files in COMPONENT_CASES:
        argv = _case_argv(files=files, checkout=checkout, python=python)
        if tag != TAG:
            argv[argv.index("--tag") + 1] = tag
        if priority != PRIORITY:
            argv[argv.index("--priority") + 1] = str(priority)
        cases.append({"name": name, "client": SUPPORTED_CLIENT,
                      "waiter": SUPPORTED_WAITER, "argv": argv,
                      "files": list(files)})
    return {"schema": PLAN_SCHEMA, "pins": pins,
            "checkout": checkout, "python": python,
            "required_hosts": list(HOSTS),
            "cases": cases,
            "unimplemented": [{"name": name, "reason": reason}
                              for name, reason in UNIMPLEMENTED]}


def format_invocations(plan: dict) -> list[str]:
    """The exact shell lines for runnable rows; records for the rest."""
    lines = []
    for case in plan["cases"]:
        lines.append(f"# case: {case['name']}")
        lines.append("python3 " + " ".join(case["argv"]))
    for record in plan["unimplemented"]:
        lines.append(f"# unimplemented {record['name']}: {record['reason']}")
    return lines


def parse_detach(text: str) -> dict:
    """Action key + terminal paths from a ``pbrun --detach`` JSON line."""
    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            doc = json.loads(line)
        except ValueError:
            continue
        if doc.get("schema") != "prismaquant.prismabuild.pbrun_detach.v1":
            continue
        return {"action_key": doc.get("action_key", ""),
                "done": doc.get("done", ""),
                "failed": doc.get("failed", ""),
                "status": doc.get("status", "")}
    return {"action_key": "", "done": "", "failed": "",
            "status": "no-detach-line"}


def _counts(stdout: str) -> dict[str, int]:
    out = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for match in re.finditer(r"(\d+)\s+(passed|failed|skipped|error)\b",
                             stdout):
        out[match.group(2)] += int(match.group(1))
    return out


def parse_terminal(doc: dict, *, side: str) -> dict:
    """Attributable facts from a done/failed terminal record."""
    detail = doc.get("detail") if isinstance(doc.get("detail"), dict) else {}
    stdout = str(detail.get("stdout", "") or "")
    payloads = sorted(set(re.findall(r'"payload_path":\s*"([^"]+)"',
                                     stdout)))
    hosts = [doc.get("finished_host"), doc.get("claimed_host")]
    return {"side": side,
            "status": doc.get("status", ""),
            "host": next((h for h in hosts
                          if isinstance(h, str) and h), ""),
            "returncode": detail.get("returncode"),
            "counts": _counts(stdout),
            "payload_paths": payloads}


def _case_from_terminal(name: str, detach: dict, term: dict) -> dict:
    counts, host = term["counts"], term["host"]
    passed, failed = counts["passed"], counts["failed"] + counts["error"]
    if (term["side"] != "done" or term["returncode"] != 0
            or failed or not host):
        reason = "; ".join(part for part in (
            f"terminal side={term['side']}" if term["side"] != "done"
            else "",
            f"returncode={term['returncode']}"
            if term["returncode"] != 0 else "",
            f"failed={failed}" if failed else "",
            "host unattributed" if not host else "") if part)
        return {"name": name, "status": "failed",
                "reason": reason or "unreadable terminal",
                "detach": detach["status"],
                "action_key": detach["action_key"],
                "terminal": detach["done"] or detach["failed"],
                "passed": passed, "failed": failed,
                "skipped": counts["skipped"], "hosts": [host] if host else [],
                "payload_paths": term["payload_paths"]}
    return {"name": name, "status": "qualified",
            "reason": (f"{counts['skipped']} named nonqualified skip(s)"
                       if counts["skipped"] else ""),
            "detach": detach["status"],
            "action_key": detach["action_key"],
            "terminal": detach["done"],
            "passed": passed, "failed": 0,
            "skipped": counts["skipped"], "hosts": [host],
            "payload_paths": term["payload_paths"]}


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


def assemble_report(*, plan: dict, cases: list[dict],
                    started_unix: float) -> dict:
    """Report from executed cases; completeness never claimed here."""
    covered = sorted({h for c in cases for h in c["hosts"]
                      if h in plan["required_hosts"]})
    missing = [h for h in plan["required_hosts"] if h not in covered]
    incomplete = [f"unimplemented leg: {r['name']}"
                  for r in plan["unimplemented"]]
    if missing:
        incomplete.append(
            "host coverage gap: no terminal served by "
            + ", ".join(missing))
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


def submit(plan: dict, out_dir: str) -> tuple[dict, int]:
    """Execute runnable cases via published clients; write the report.

    Returns ``(report, exit_code)``: 1 on failed cases or driver errors,
    else 0. A zero exit with ``complete: false`` is normal today: it
    means the machinery worked and the gaps are named, not that the
    fleet is qualified.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    cases: list[dict] = []
    errors: list[str] = []
    for case in plan["cases"]:
        name = case["name"]
        log_path = out / f"{name}.submit.txt"
        try:
            done = subprocess.run(
                [sys.executable, *case["argv"]],
                capture_output=True, text=True, timeout=600)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{name}: submission error: {exc!r}")
            continue
        log_path.write_text(
            f"$ python3 {' '.join(case['argv'])}\n"
            f"returncode={done.returncode}\n"
            f"--- stdout ---\n{done.stdout[-3000:]}\n"
            f"--- stderr ---\n{done.stderr[-2000:]}")
        detach = parse_detach(done.stdout)
        if not detach["action_key"]:
            errors.append(
                f"{name}: no detach line: rc={done.returncode} "
                f"{done.stderr.strip()[-300:]}")
            continue
        waited = subprocess.run(
            [sys.executable, str(PBWAIT),
             "--wait-s", str(WAIT_S), detach["action_key"]],
            capture_output=True, text=True, timeout=WAIT_S + 300)
        _ = waited
        term, side = None, ""
        for candidate, label in ((detach["done"], "done"),
                                (detach["failed"], "failed")):
            if candidate:
                try:
                    with open(candidate) as stream:
                        term, side = json.load(stream), label
                    break
                except (OSError, ValueError):
                    continue
        if term is None:
            errors.append(f"{name}: no terminal at recorded paths")
            continue
        cases.append(_case_from_terminal(
            name, detach, parse_terminal(term, side=side)))
    report = assemble_report(plan=plan, cases=cases, started_unix=started)
    if errors:
        report["driver_errors"] = errors
    (out / "report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n")
    return report, exit_code_for(report)


def print_summary(report: dict) -> None:
    """Human lines for the console; gaps are loud, never hidden."""
    for case in report["cases"]:
        flag = ("QUALIFIED" if case["status"] == "qualified" else "FAILED")
        print(f"{flag} {case['name']}: "
              f"passed={case['passed']} failed={case['failed']} "
              f"skipped={case['skipped']} "
              f"hosts={','.join(case['hosts']) or 'unattributed'} "
              f"{case['detach']} {case['action_key'][:8]} "
              f"{case['reason']}")
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
                        help="pins JSON (record_pins output)")
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
        report, exit_code = submit(plan, args.out_dir)
        print_summary(report)
        return exit_code
    parser.error("exactly one of --print-plan or --submit is required")


if __name__ == "__main__":
    raise SystemExit(main())
