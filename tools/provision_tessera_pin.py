#!/usr/bin/env python3
"""Install the Tessera the pin names into an interpreter, and check the bytes.

GitHub CI checks out ``RobTand/tessera`` at the commit
``tools/resolve_tessera_dev_pin.py`` prints and runs
``pip install --no-deps`` on that directory.  Nothing did the same for the
fleet interpreters, so ``/home/rob/venvs/pq-cpu312`` sat on a Tessera from an
ancestor of the pin: answer-equivalent, different bytes, and every
byte-identity assertion red in PrismaBuild while green in CI (issue #455).

This is that missing step, written so a re-pin can carry it.  A re-pin is
already a reviewed change to the pin JSON and the module constants together;
run this on the same change and the fleet moves with them instead of behind
them.

The pin is read the way the resolver reads it -- parsed out of
``prismaquant/tessera_runtime_contract.py`` -- so this tool imports neither
PrismaQuant nor Tessera and runs on an interpreter that has neither.  The
reviewed contract hash is read from the same file for the same reason, and it
is the check that actually bites: a matching version string proves nothing,
because the pin's version is static ``0.1.0`` and so was the stale install's.

Usage::

    python tools/provision_tessera_pin.py --python /home/rob/venvs/pq-cpu312/bin/python

By default the source is materialised under ``/mnt/shared/tessera-pins/<commit>``
from a local Tessera clone, which is content-addressed by construction: a
directory named for a commit either holds that commit's tree or does not exist.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_SOURCE = ROOT / "prismaquant" / "tessera_runtime_contract.py"
PIN_NAME = "TESSERA_DEV_PIN_COMMIT"
SHA_NAME = "TESSERA_DEV_PIN_CONTRACT_SHA256"
DEFAULT_PINS_ROOT = Path("/mnt/shared/tessera-pins")
DEFAULT_CLONE = Path("/home/rob/tessera")
CONTRACT_IN_TREE = Path("src/tessera/serving/runtime_contract.json")


def _literal(name: str, source: Path = PIN_SOURCE) -> str:
    """The one literal ``name`` assignment in ``source``, as a string."""

    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    values: list[object] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in targets
        ):
            values.append(ast.literal_eval(node.value))
    if len(values) != 1:
        raise SystemExit(f"{source}: expected exactly one literal {name}")
    value = values[0]
    if not isinstance(value, str):
        raise SystemExit(f"{source}: {name} is not a string literal")
    return value


def reviewed_pin(source: Path = PIN_SOURCE) -> tuple[str, str]:
    """The reviewed ``(commit, contract_sha256)`` pair."""

    commit = _literal(PIN_NAME, source)
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise SystemExit(f"{source}: {PIN_NAME} must be a full lowercase Git SHA")
    sha = _literal(SHA_NAME, source)
    if re.fullmatch(r"[0-9a-f]{64}", sha) is None:
        raise SystemExit(f"{source}: {SHA_NAME} must be a lowercase sha256")
    return commit, sha


def materialise(commit: str, clone: Path, pins_root: Path) -> Path:
    """Lay down ``commit``'s tree under ``pins_root``, or reuse what is there."""

    target = pins_root / commit
    marker = target / CONTRACT_IN_TREE
    if marker.exists():
        return target
    if not (clone / ".git").exists():
        raise SystemExit(
            f"{clone} is not a Tessera clone; pass --clone or pre-populate {target}"
        )
    target.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(clone),
         "archive", "--format=tar", commit],
        check=True, stdout=subprocess.PIPE,
    )
    subprocess.run(["tar", "-x", "-C", str(target)],
                   check=True, input=archive.stdout)
    if not marker.exists():
        raise SystemExit(f"{commit} carries no {CONTRACT_IN_TREE}")
    return target


def installed_contract(python: str) -> tuple[str | None, str | None]:
    """That interpreter's packaged contract: ``(sha256, reason it is absent)``.

    Exactly one of the two is set.  The reason is carried rather than dropped
    because an absent contract has at least three causes an operator has to
    act on differently -- the venv does not exist, Tessera is not installed in
    it, or Tessera is installed without its packaged contract -- and a bare
    ``null`` reads as all three at once.  The GB10 venv reported ``null`` on
    its first audit; it was the second cause, and the report did not say so.
    """

    probe = (
        "import hashlib,json,sys\n"
        "from importlib import resources\n"
        "try:\n"
        "    p = resources.files('tessera.serving')"
        ".joinpath('runtime_contract.json')\n"
        "    out = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}\n"
        "except Exception as exc:\n"
        "    out = {'reason': type(exc).__name__ + ': ' + str(exc)}\n"
        "sys.stdout.write(json.dumps(out))\n"
    )
    try:
        done = subprocess.run([python, "-c", probe],
                              capture_output=True, text=True)
    except OSError as exc:
        # No such interpreter, or one that will not start.  The caller wants
        # "not the reviewed bytes", and an exception here would make a routine
        # audit of a machine that has no such venv look like a tool failure.
        return None, f"{type(exc).__name__}: {exc}"
    try:
        out = json.loads(done.stdout.strip() or "{}")
    except ValueError:
        out = {}
    sha = out.get("sha256")
    if sha:
        return sha, None
    reason = out.get("reason") or (done.stderr.strip().splitlines() or
                                   ["the probe printed nothing"])[-1]
    return None, reason


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True,
                        help="interpreter to provision")
    parser.add_argument("--clone", type=Path, default=DEFAULT_CLONE,
                        help="local Tessera clone to archive the pin from")
    parser.add_argument("--pins-root", type=Path, default=DEFAULT_PINS_ROOT,
                        help="where pinned source trees are materialised")
    parser.add_argument("--check-only", action="store_true",
                        help="report the installed contract, install nothing")
    args = parser.parse_args(argv)

    commit, reviewed_sha = reviewed_pin()
    before, before_absent = installed_contract(args.python)
    report = {
        "python": args.python,
        "reviewed_commit": commit,
        "reviewed_contract_sha256": reviewed_sha,
        "installed_contract_sha256_before": before,
    }
    if before_absent is not None:
        report["installed_contract_absent_before"] = before_absent

    if before == reviewed_sha:
        report["action"] = "none, already the reviewed bytes"
        print(json.dumps(report, indent=1))
        return 0
    if args.check_only:
        report["action"] = "none, --check-only"
        print(json.dumps(report, indent=1))
        return 1

    source = materialise(commit, args.clone, args.pins_root)
    on_disk = hashlib.sha256(
        (source / CONTRACT_IN_TREE).read_bytes()
    ).hexdigest()
    if on_disk != reviewed_sha:
        raise SystemExit(
            f"{source} publishes contract {on_disk}, reviewed is {reviewed_sha}"
        )
    report["source"] = str(source)

    subprocess.run(
        [args.python, "-m", "pip", "install", "--no-deps",
         "--no-build-isolation", "--force-reinstall", str(source)],
        check=True,
    )

    after, after_absent = installed_contract(args.python)
    report["installed_contract_sha256_after"] = after
    if after_absent is not None:
        report["installed_contract_absent_after"] = after_absent
    if after != reviewed_sha:
        report["action"] = "installed, and it did NOT take"
        print(json.dumps(report, indent=1))
        return 1
    report["action"] = "installed the reviewed bytes"
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
