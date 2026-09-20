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

#: Published PB client sources, imported read-only for verification only:
#: the CAS verifier (``prismabuild.core.PrismaBuildCAS``) and the anchored
#: pytest summary grammar (``pbtest.pytest_summary``). Never the candidate
#: tree under test; the driver process imports no candidate modules.
_PB_SRC = Path("/mnt/shared/prismabuild-fleet/repo/src")
_PB_TOOLS = Path("/mnt/shared/prismabuild-fleet/repo/tools")

#: Content-addressed store and queue roots (PB-published defaults, the same
#: roots ``pbwait``/``pbstatus`` read). Terminal and receipt paths below
#: derive from action keys; nothing is written here.
_CAS_ROOT = Path("/mnt/shared/prismabuild-fleet/cas")
_QUEUE_ROOT = Path("/mnt/shared/prismabuild-fleet/pb-queue")

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


def _published():
    """Published PB verification imports (read-only use).

    The CAS verifier (``prismabuild.core.PrismaBuildCAS``) and the anchored
    pytest summary grammar (``pbtest.pytest_summary``). Never the candidate
    tree under test; the driver process imports no candidate modules.
    """
    for entry in (str(_PB_SRC), str(_PB_TOOLS)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    from prismabuild import core as pbcore
    import pbtest as pbtest_mod
    return pbcore, pbtest_mod


def _sealed_command(action: object) -> tuple[dict | None, str]:
    """Parse the sealed test command (closed published forms).

    Accepts exactly the two shapes the published generators seal for
    test work, validated token by token after shell lexing (comments
    stripped, quoting respected):

    - pbtest suite form: ``export PATH=...:$PATH; env K=V... <interp>
      -c <guard> -q --no-header -p no:cacheprovider <files>`` -- the
      dependency-guard entry admitted since this checkout carries
      ``resolve_*_dev_pin.py`` resolvers;
    - pbrun-direct bare form: ``export PATH=...:$PATH; <interp> -m
      pytest <files>``.

    Returns ``({"interpreter", "files", "log", "entry"}, "")`` where
    entry is ``"guard"`` or ``"plain"``. Anything else returns
    ``(None, reason)``; callers treat shape drift as nonqualified,
    never as proof. This is a small verifier for explicit versioned
    forms, not a general Bash analyzer.
    """
    if not isinstance(action, dict):
        return None, "sealed action malformed"
    task = action.get("task")
    argv = task.get("argv") if isinstance(task, dict) else None
    if (not isinstance(argv, list) or len(argv) != 5
            or argv[:4] != ["/bin/bash", "--noprofile", "--norc", "-c"]
            or not isinstance(argv[4], str)):
        return None, "sealed wrapper shape differs"
    script = argv[4]
    head, exit_sep, exit_tail = script.rpartition("; exit ${PIPESTATUS[0]}")
    if not exit_sep or exit_tail != "":
        return None, "sealed exit handling differs"
    cmdline, tee_sep, log = head.rpartition(" 2>&1 | tee ")
    if not tee_sep or not log or " " in log:
        return None, "sealed log separator differs"
    try:
        tokens = shlex.split(cmdline, comments=True, posix=True)
    except ValueError:
        return None, "sealed command unlexable"
    if (len(tokens) < 4 or tokens[0] != "export"
            or not tokens[1].startswith("PATH=")
            or not tokens[1].endswith(":$PATH") or tokens[2] != ";"):
        return None, "sealed prefix differs"
    rest = tokens[3:]
    if rest[:1] == ["env"]:
        index = 1
        while index < len(rest) and re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]*=.*", rest[index]):
            index += 1
        if index == 1:
            return None, "sealed env assignments differ"
        body = rest[index:]
        if (len(body) < 7 or not body[0] or body[1] != "-c"
                or not body[2]
                or body[3:7] != ["-q", "--no-header", "-p",
                                 "no:cacheprovider"]):
            return None, "sealed pytest entry differs"
        files = body[7:]
        entry = "guard"
    else:
        if (len(rest) < 4 or not rest[0]
                or rest[1:3] != ["-m", "pytest"]):
            return None, "sealed pytest entry differs"
        files = rest[3:]
        entry = "plain"
    if not files or any(not f.endswith(".py") for f in files):
        return None, "sealed test operands differ"
    return {"interpreter": rest[0] if entry == "plain" else body[0],
            "files": files, "log": log, "entry": entry}, ""


def _snapshot_agreement(action: object,
                        terminal: dict) -> tuple[str | None, str]:
    """Action snapshot input vs the terminal snapshot descriptor.

    Returns ``(sha256, "")`` when the filed action's
    ``pbrun.checkout-snapshot`` input digest equals the digest in the
    terminal record used for attribution, else ``(None, reason)``.
    """
    inputs = action.get("inputs") if isinstance(action, dict) else None
    snaps = [entry for entry in inputs
             if isinstance(entry, dict)
             and entry.get("id") == "pbrun.checkout-snapshot"
             and isinstance(entry.get("sha256"), str)]
    if len(snaps) != 1:
        return None, "action snapshot input missing"
    snapshot = terminal.get("checkout_snapshot")
    descriptor = snapshot.get("input") if isinstance(snapshot, dict) else None
    digest = descriptor.get("sha256") if isinstance(descriptor, dict) else None
    if digest != snaps[0]["sha256"]:
        return None, "snapshot input disagrees with terminal descriptor"
    return str(digest), ""
    """Distinct 64-hex action keys named in console output.

    Console JSON only LOCATES a candidate key; nothing here authorizes
    anything. None on malformed input.
    """
    if not isinstance(output, str):
        return None
    return set(re.findall(r'"action_key":\s*"([0-9a-f]{64})"', output))


def _recorded_action(cas, key: str) -> dict | None:
    """The sealed action PB filed for this key, or None.

    Same path convention ``pbwait.recorded_action`` reads
    (``requests/<xx>/<key>.json``); all validation happens in
    ``cas.lookup``, never here.
    """
    try:
        value = json.loads(
            (Path(cas.root) / "requests" / key[:2]
             / f"{key}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _summary_counts(line: str) -> dict[str, int]:
    """Outcome counts from one anchored pytest summary line.

    Stems cover singular and plural (``1 error`` / ``2 errors``);
    collection words (warnings, xfailed, deselected, subtests-as-such)
    never count as outcomes.
    """
    out = {"passed": 0, "failed": 0, "skipped": 0, "error": 0}
    for match in re.finditer(r"(\d+)\s+([A-Za-z][\w-]*)", line):
        stem = match.group(2).lower()
        if stem.startswith("pass"):
            out["passed"] += int(match.group(1))
        elif stem.startswith("fail"):
            out["failed"] += int(match.group(1))
        elif stem.startswith("skip"):
            out["skipped"] += int(match.group(1))
        elif stem.startswith("error"):
            out["error"] += int(match.group(1))
    return out


def _receipt_generation(receipt: object) -> str | None:
    """Deployed generation from the authenticated receipt runtime evidence.

    Reads ``producer.runtime`` (core/launcher paths) inside the receipt
    ``cas.lookup`` already integrity-checked -- never argv text.
    """
    if not isinstance(receipt, dict):
        return None
    producer = receipt.get("producer")
    runtime = producer.get("runtime") if isinstance(producer, dict) else None
    if not isinstance(runtime, dict):
        return None
    for section in ("core", "launcher"):
        part = runtime.get(section)
        path = part.get("path") if isinstance(part, dict) else None
        if not isinstance(path, str):
            continue
        match = re.search(r"runtime-generations/([^/\"'\s]+)", path)
        if match:
            return match.group(1)
    return None


def _read_json_file(path: str) -> dict | list | None:
    try:
        with open(path) as stream:
            return json.load(stream)
    except (OSError, ValueError):
        return None


def _read_terminal(key: str) -> tuple[dict | None, str]:
    """The queue terminal record for one action key, if any."""
    for side in ("done", "failed"):
        try:
            with open(_QUEUE_ROOT / side / f"{key}.json") as stream:
                doc = json.load(stream)
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict):
            return doc, side
    return None, ""


def verify_shard(*, shard: object, host: str, declared: dict,
                 expected_file: str) -> dict:
    """qualified / nonqualified / failed for one shard on one host.

    The chain, all through existing PB machinery: console output locates
    the candidate action key; the filed action comes from the CAS
    requests store; ``cas.lookup`` verifies the receipt, its manifest
    binding, the producer attestation, and the result blob; the verified
    result bytes carry the anchored pytest summary; the queue terminal
    confirms executed status, host, and source; the generation comes
    from the authenticated receipt runtime evidence. Malformed objects
    are rejected, never coerced. Only a fully attributed chain
    qualifies.
    """
    def nope(status: str, reason: str, **extra: object) -> dict:
        base: dict = {"host": host, "status": status, "reason": reason,
                      "passed": 0, "failed": 0, "skipped": 0,
                      "action_key": "", "terminal": "",
                      "snapshot_commit": "", "snapshot_parent": "",
                      "generation": "", "receipt_sha256": ""}
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
    if not isinstance(shard.get("summary"), str):
        return nope("failed", "malformed shard record: summary")
    # The submission record's returncode gates first; verdict counts come
    # from the verified result bytes below, never console rendering.
    if returncode != 0:
        return nope("failed", f"returncode={returncode}")
    keys = _candidate_keys(output)
    assert keys is not None
    if not keys:
        return nope("nonqualified",
                    "unattributed cached replay: no action key in output; "
                    "resubmission at a new snapshot required for fresh "
                    "terminals")
    if len(keys) != 1:
        return nope("failed", "ambiguous candidate actions in output")
    key = next(iter(keys))
    try:
        pbcore, pbtest_mod = _published()
    except Exception as exc:  # noqa: BLE001 -- fail closed, name it
        return nope("failed",
                    f"published verifier unavailable: {exc!r}",
                    action_key=key)
    cas = pbcore.PrismaBuildCAS(_CAS_ROOT)
    action = _recorded_action(cas, key)
    if action is None:
        return nope("failed", "no filed action for recorded key",
                    action_key=key)
    try:
        receipt = cas.lookup(action)
    except Exception as exc:  # noqa: BLE001 -- tamper reads as failure
        return nope(
            "failed",
            f"receipt verification: {type(exc).__name__}: "
            f"{str(exc)[:200]}",
            action_key=key)
    if receipt is None:
        return nope("failed", "no CAS receipt for filed action",
                    action_key=key)
    receipt_sha = str(receipt.get("receipt_sha256", ""))
    command, problem = _sealed_command(action)
    if problem:
        return nope("nonqualified",
                    f"command shape differs: {problem}",
                    action_key=key, receipt_sha256=receipt_sha)
    assert command is not None
    if command["files"] != [expected_file]:
        return nope("failed",
                    f"sealed command does not execute {expected_file}",
                    action_key=key, receipt_sha256=receipt_sha)
    python = declared.get("python")
    if not isinstance(python, str) or command["interpreter"] != python:
        return nope("failed", "sealed command interpreter mismatch",
                    action_key=key, receipt_sha256=receipt_sha)
    task = action.get("task") if isinstance(action, dict) else None
    result_path = task.get("result_path") if isinstance(task, dict) else None
    if result_path != command["log"]:
        return nope("failed", "sealed log differs from task result_path",
                    action_key=key, receipt_sha256=receipt_sha)
    try:
        result_path = cas.result_path(receipt, action)
        text = Path(result_path).read_bytes().decode("utf-8")
    except Exception as exc:  # noqa: BLE001 -- fail closed, name it
        return nope(
            "failed",
            f"result unreadable: {type(exc).__name__}: {str(exc)[:200]}",
            action_key=key, receipt_sha256=receipt_sha)
    if command["entry"] == "guard" and "pbtest dependency pin:" not in text:
        return nope("failed", "pin-guard evidence absent from result",
                    action_key=key, receipt_sha256=receipt_sha)
    summary = pbtest_mod.pytest_summary(text.splitlines())
    if not summary:
        return nope("nonqualified",
                    "no terminal summary in verified result",
                    action_key=key, receipt_sha256=receipt_sha)
    counts = _summary_counts(summary)
    if counts["failed"] or counts["error"]:
        return nope("failed",
                    f"failed={counts['failed']} errors={counts['error']}",
                    action_key=key, receipt_sha256=receipt_sha,
                    passed=counts["passed"], failed=counts["failed"],
                    skipped=counts["skipped"])
    if not counts["passed"]:
        return nope("nonqualified", "no passing tests collected",
                    action_key=key, receipt_sha256=receipt_sha,
                    skipped=counts["skipped"])
    term, side = _read_terminal(key)
    if term is None:
        return nope("failed", "no terminal for recorded action",
                    action_key=key, receipt_sha256=receipt_sha,
                    passed=counts["passed"], skipped=counts["skipped"])
    if term.get("action_key") != key:
        return nope("failed", "stale unrelated terminal",
                    action_key=key, receipt_sha256=receipt_sha,
                    passed=counts["passed"], skipped=counts["skipped"])
    if side != "done" or term.get("status") != "executed":
        return nope("failed",
                    f"terminal side={side} status={term.get('status')}",
                    action_key=key, receipt_sha256=receipt_sha,
                    passed=counts["passed"], skipped=counts["skipped"])
    detail = term.get("detail")
    if (not isinstance(detail, dict)
            or detail.get("returncode") != 0):
        return nope("failed", "terminal returncode not zero",
                    action_key=key, receipt_sha256=receipt_sha,
                    passed=counts["passed"], skipped=counts["skipped"])
    if term.get("finished_host") != host:
        return nope("failed",
                    f"wrong host: terminal served by "
                    f"{term.get('finished_host')}",
                    action_key=key, receipt_sha256=receipt_sha,
                    passed=counts["passed"], skipped=counts["skipped"])
    snapshot = term.get("checkout_snapshot")
    parent = snapshot.get("parent") if isinstance(snapshot, dict) else None
    commit = snapshot.get("commit") if isinstance(snapshot, dict) else None
    if parent != declared.get("pq_head"):
        return nope("failed",
                    f"source mismatch: declared {declared.get('pq_head')} "
                    f"vs executed parent {parent}",
                    action_key=key, receipt_sha256=receipt_sha,
                    passed=counts["passed"], skipped=counts["skipped"],
                    snapshot_commit=commit or "",
                    snapshot_parent=parent or "")
    agreement, problem = _snapshot_agreement(action, term)
    if problem:
        return nope("failed", problem,
                    action_key=key, receipt_sha256=receipt_sha,
                    passed=counts["passed"], skipped=counts["skipped"],
                    snapshot_commit=commit or "",
                    snapshot_parent=parent or "")
    generation = _receipt_generation(receipt)
    if generation != declared.get("generation"):
        return nope("failed",
                    f"generation mismatch: declared "
                    f"{declared.get('generation')} vs {generation}",
                    action_key=key, receipt_sha256=receipt_sha,
                    passed=counts["passed"], skipped=counts["skipped"],
                    snapshot_commit=commit or "",
                    snapshot_parent=parent or "")
    evidence_host = None
    producer = receipt.get("producer")
    if isinstance(producer, dict):
        evidence = producer.get("evidence")
        if isinstance(evidence, dict):
            evidence_host = evidence.get("hostname")
    if evidence_host != host:
        return nope("failed",
                    f"host evidence disagreement: receipt says "
                    f"{evidence_host}",
                    action_key=key, receipt_sha256=receipt_sha,
                    passed=counts["passed"], skipped=counts["skipped"],
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
            "generation": generation or "",
            "receipt_sha256": receipt_sha}


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
    """Report from verified cases; completeness never claimed here.

    A host is covered only when EVERY planned case file qualifies on it;
    ``complete_cases`` turns dropped shards into explicit nonqualified
    records first, so a shrinking denominator can never fake coverage.
    """
    qualified = {(str(v.get("file")), str(v.get("host")))
                 for v in verdicts if v.get("status") == "qualified"}
    covered = sorted(
        host for host in plan["required_hosts"]
        if all((filename, host) in qualified
               for filename in CASE_FILES))
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
    first_use: dict[str, tuple] = {}
    for record in cases:
        key = record.get("action_key")
        if not isinstance(key, str) or not key:
            continue
        owner = (record.get("file"), record.get("host"))
        if key in first_use and first_use[key] != owner:
            record["status"] = "failed"
            record["reason"] = (
                f"duplicate action key also used by "
                f"{first_use[key][0]}@{first_use[key][1]}")
        else:
            first_use.setdefault(key, owner)
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


def _group_failure(host: str, filename: str, reason: str) -> dict:
    return {"host": host, "file": filename, "status": "failed",
            "reason": reason, "passed": 0, "failed": 0, "skipped": 0,
            "action_key": "", "terminal": "",
            "snapshot_commit": "", "snapshot_parent": "",
            "generation": "", "receipt_sha256": ""}


def group_shards(host: str, shards: object) -> tuple[list, list[dict]]:
    """Split one host's shards into verifiable items and failures.

    Pure: returns ``([(filename, shard)], [failure verdicts])``.
    Unexpected files, duplicates, and malformed fanout fail here, so
    only exactly-planned single-file shards reach verification.
    """
    if not isinstance(shards, list):
        return [], [_group_failure(host, "?", "receipt unreadable")]
    items: list = []
    failures: list[dict] = []
    seen: dict[str, int] = {}
    for shard in shards:
        files = shard.get("files") if isinstance(shard, dict) else None
        if (not isinstance(files, list) or len(files) != 1
                or not isinstance(files[0], str)):
            failures.append(_group_failure(host, "?", "unexpected fanout"))
            continue
        filename = files[0]
        if filename not in CASE_FILES:
            failures.append(_group_failure(host, filename,
                                           "unexpected case file"))
            continue
        seen[filename] = seen.get(filename, 0) + 1
        if seen[filename] > 1:
            failures.append(_group_failure(host, filename,
                                           "duplicate case file"))
            continue
        items.append((filename, shard))
    return items, failures


def complete_cases(plan: dict, verdicts: list[dict]) -> list[dict]:
    """Add explicit nonqualified records for missing planned files.

    Expected coverage is the entire CASE_FILES x required-hosts matrix;
    a dropped shard reports its host uncovered instead of silently
    shrinking the denominator.
    """
    expected = {(filename, host)
                for filename in CASE_FILES
                for host in plan["required_hosts"]}
    present = {(str(v.get("file")), str(v.get("host"))) for v in verdicts}
    out = list(verdicts)
    for filename, host in sorted(expected - present):
        out.append({"host": host, "file": filename,
                    "status": "nonqualified",
                    "reason": "missing case file",
                    "passed": 0, "failed": 0, "skipped": 0,
                    "action_key": "", "terminal": "",
                    "snapshot_commit": "", "snapshot_parent": "",
                    "generation": "", "receipt_sha256": ""})
    return out


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
        items, failures = group_shards(host, receipts.get(host))
        verdicts.extend(failures)
        for filename, shard in items:
            verdict = verify_shard(shard=shard, host=host,
                                   declared=declared,
                                   expected_file=filename)
            verdict["file"] = filename
            verdicts.append(verdict)
    verdicts = complete_cases(plan, verdicts)
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
        declared["python"] = args.python
        report, exit_code = submit(plan, args.out_dir, declared)
        print_summary(report)
        return exit_code
    parser.error("exactly one of --print-plan or --submit is required")


if __name__ == "__main__":
    raise SystemExit(main())
