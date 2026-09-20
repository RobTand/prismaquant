"""Executable fleet qualification driver (level 2, component scope).

Builds the hardware-phase qualification plan from declared pins and, with
``--submit``, EXECUTES the runnable component cases through the published
PB clients only (``pbtest.py`` subprocesses -- no scheduler, no model-layer
sharding, no placement control) and writes an attributable report.

Every runnable row carries real flags (checkout, interpreter, tag,
priority, shards, threads, memory, wait, per-case receipt JSON); there
are no ``{placeholders}``, no ``run`` subcommand, no ``--tag cpu``, and
no container ``--image``/``--help`` pseudo-validation. Cases that cannot
run yet (membership JOIN/RESIGN/handoff, native-container readers) are
explicit ``unimplemented`` case records with reasons -- never runnable
invocations, never proof.

``--submit`` exit code: 1 on failed cases or driver errors, 0 otherwise.
Completeness is NEVER claimed here: the report always carries
``complete: false`` with the incomplete reasons (unimplemented legs,
uncovered hosts). Skips inside a passing case are the harness's named
nonqualified legs (e.g. the cgroup-gated ticket case), recorded, never
green conformance. Do NOT run live membership/deploy from this driver.

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
PB_CLIENT = Path("/mnt/shared/prismabuild-fleet/repo/tools/pbtest.py")

#: Fresh exact-pin env (PQ846 strict PB reader SDK 461728e4). Never modify
#: a shared active env; this path names the provisioned one. It must exist
#: on every worker the cases land on -- a missing interpreter is a named
#: driver error, never a silent substitution.
PYTHON = "/home/rob/venvs/pq846-pb461728e4/bin/python"

ENTRYPOINTS = ("tools/pbtest.py", "tools/pbcampaign.py", "tools/pbrun.py")
SUPPORTED_CLIENT = "tools/pbtest.py"

#: Hardware coverage is the purpose: both Sparks are required dependencies
#: of a complete report. Placement stays PB-owned; the driver ATTRIBUTES
#: serving hosts from shard receipts and reports uncovered hosts as
#: nonqualified gaps.
HOSTS = ("sparky", "sparklina")

TAG = "gb10"
PRIORITY = -10
WAIT_S = 1500
MEM_GB = 8

#: Runnable component cases: (name, files, mem_gb). One shard per file;
#: distinct files are distinct actions, so the fleet distributes them and
#: host coverage emerges from receipts.
COMPONENT_CASES: tuple = (
    ("pins", ("tests/test_fleet_acceptance_pins.py",), MEM_GB),
    ("runner-table", ("tests/test_fleet_acceptance_runner.py",), MEM_GB),
    ("matrix", ("tests/test_fleet_acceptance_matrix.py",), MEM_GB),
    ("driver-self", ("tests/test_fleet_qualification_driver.py",), MEM_GB),
    ("level1-connected", ("tests/test_fleet_acceptance_level1.py",), MEM_GB),
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


def _case_argv(*, out_json: Path, files: tuple[str, ...],
               checkout: str, python: str) -> list[str]:
    """The exact published-client invocation for one component case."""
    return [str(PB_CLIENT),
            "--checkout", checkout,
            "--python", python,
            "--tag", TAG,
            "--priority", str(PRIORITY),
            "--shards", "1",
            "--workers-per-shard", "1",
            "--threads-per-shard", "1",
            "--mem-gb", str(MEM_GB),
            "--wait-s", str(WAIT_S),
            "--json", str(out_json),
            *[str(f) for f in files]]


def build_plan(*, pins: dict, checkout: str, python: str = PYTHON,
               out_dir: str, tag: str = TAG,
               priority: int = PRIORITY) -> dict:
    """The executable plan: runnable rows plus unimplemented records."""
    out = Path(out_dir)
    cases = []
    for name, files, _mem in COMPONENT_CASES:
        argv = _case_argv(out_json=out / f"{name}.pbtest.json",
                          files=files, checkout=checkout, python=python)
        if tag != TAG:
            argv[argv.index("--tag") + 1] = tag
        if priority != PRIORITY:
            argv[argv.index("--priority") + 1] = str(priority)
        cases.append({"name": name, "client": SUPPORTED_CLIENT,
                      "argv": argv, "files": list(files),
                      "receipt": str(out / f"{name}.pbtest.json")})
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


def _counts(summary: str) -> dict[str, int]:
    out = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for match in re.finditer(r"(\d+)\s+(passed|failed|skipped|error)\b",
                             summary):
        out[match.group(2)] += int(match.group(1))
    return out


def parse_shard(shard: dict) -> dict:
    """Attributable facts from one pbtest shard record (no verdicts)."""
    output = str(shard.get("output", ""))
    summary = str(shard.get("summary", ""))
    hosts = set(re.findall(r'"hostname":\s*"([\w.-]+)"', output))
    hosts.update(re.findall(r"executed on (\w[\w.-]*)", output))
    actions = sorted(set(re.findall(r'"action_key":\s*"([0-9a-f]{64})"',
                                    output)))
    queued = sorted(set(re.findall(r"queued ([0-9a-f]{8,40})\b", output)))
    payloads = sorted(set(re.findall(r'"payload_path":\s*"([^"]+)"',
                                     output)))
    receipts = sorted(set(re.findall(r'"receipt_sha256":\s*"([^"]+)"',
                                     output)))
    published = '"status": "published"' in output
    return {"returncode": shard.get("returncode"),
            "counts": _counts(summary),
            "hosts": sorted(hosts),
            "actions": actions, "queued": queued,
            "payload_paths": payloads, "receipt_sha256": receipts,
            "published": published}


def _case_status(name: str, shards: list[dict]) -> dict:
    """qualified / failed for one executed case (skips stay visible)."""
    parsed = [parse_shard(s) for s in shards]
    passed = sum(p["counts"]["passed"] for p in parsed)
    failed = sum(p["counts"]["failed"] for p in parsed)
    skipped = sum(p["counts"]["skipped"] for p in parsed)
    errors = sum(p["counts"]["error"] for p in parsed)
    bad_rc = [p["returncode"] for p in parsed if p["returncode"] != 0]
    hosts = sorted({h for p in parsed for h in p["hosts"]})
    actions = sorted({a for p in parsed for a in p["actions"]})
    queued = sorted({q for p in parsed for q in p["queued"]})
    payloads = sorted({u for p in parsed for u in p["payload_paths"]})
    receipts = sorted({u for p in parsed for u in p["receipt_sha256"]})
    published = all(p["published"] for p in parsed) if parsed else False
    if not parsed or bad_rc or failed or errors:
        return {"name": name, "status": "failed",
                "reason": ("no shard records" if not parsed else
                           f"returncodes={bad_rc} failed={failed} "
                           f"errors={errors}"),
                "passed": passed, "failed": failed, "skipped": skipped,
                "hosts": hosts, "actions": actions, "queued": queued,
                "payload_paths": payloads, "receipt_sha256": receipts,
                "published": published}
    return {"name": name, "status": "qualified",
            "reason": (f"{skipped} named nonqualified skip(s)"
                       if skipped else ""),
            "passed": passed, "failed": 0, "skipped": skipped,
            "hosts": hosts, "actions": actions, "queued": queued,
            "payload_paths": payloads, "receipt_sha256": receipts,
            "published": published}


def assemble_report(*, plan: dict, receipts: dict[str, list[dict]],
                    started_unix: float) -> dict:
    """Report from executed cases; completeness never claimed here."""
    cases = [_case_status(case["name"],
                          receipts.get(case["name"], []))
             for case in plan["cases"]]
    covered = sorted({h for c in cases for h in c["hosts"]
                      if h in plan["required_hosts"]})
    missing = [h for h in plan["required_hosts"] if h not in covered]
    incomplete = [f"unimplemented leg: {r['name']}"
                  for r in plan["unimplemented"]]
    if missing:
        incomplete.append(
            "host coverage gap: no shard served by "
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


def submit(plan: dict, out_dir: str,
           *, wait_slack_s: int = 900) -> tuple[dict, int]:
    """Execute runnable cases via the published client; write the report.

    Returns ``(report, exit_code)``: 1 on failed cases or driver errors,
    else 0. A zero exit with ``complete: false`` is normal today: it
    means the machinery worked and the gaps are named, not that the
    fleet is qualified.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    receipts: dict[str, list[dict]] = {}
    errors: list[str] = []
    for case in plan["cases"]:
        name = case["name"]
        receipt_path = Path(case["receipt"])
        log_path = out / f"{name}.stdout.txt"
        try:
            done = subprocess.run(
                [sys.executable, *case["argv"]],
                capture_output=True, text=True,
                timeout=WAIT_S + wait_slack_s)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{name}: submission error: {exc!r}")
            continue
        log_path.write_text(
            f"$ python3 {' '.join(case['argv'])}\n"
            f"returncode={done.returncode}\n"
            f"--- stdout ---\n{done.stdout[-6000:]}\n"
            f"--- stderr ---\n{done.stderr[-2000:]}")
        if done.returncode != 0 and not receipt_path.is_file():
            errors.append(
                f"{name}: pbtest exited {done.returncode} with no receipt: "
                f"{done.stderr.strip()[-300:]}")
            continue
        try:
            data = json.loads(receipt_path.read_text())
            shards = data if isinstance(data, list) else [data]
        except (OSError, ValueError) as exc:
            errors.append(f"{name}: receipt unreadable: {exc!r}")
            continue
        receipts[name] = shards
    report = assemble_report(plan=plan, receipts=receipts,
                             started_unix=started)
    if errors:
        report["driver_errors"] = errors
    report_path = out / "report.json"
    report_path.write_text(json.dumps(report, indent=1, sort_keys=True)
                           + "\n")
    return report, exit_code_for(report)


def print_summary(report: dict) -> None:
    """Human lines for the console; gaps are loud, never hidden."""
    for case in report["cases"]:
        flag = ("QUALIFIED" if case["status"] == "qualified" else "FAILED")
        print(f"{flag} {case['name']}: "
              f"passed={case['passed']} failed={case['failed']} "
              f"skipped={case['skipped']} "
              f"hosts={','.join(case['hosts']) or 'unattributed'} "
              f"{case['reason']}")
    for record in report["unimplemented"]:
        print(f"NONQUALIFIED {record['name']}: {record['reason']}")
    if report["missing_hosts"]:
        print("NONQUALIFIED host coverage gap: "
              + ", ".join(report["missing_hosts"]))
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
