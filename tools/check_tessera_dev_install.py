#!/usr/bin/env python3
"""Fail fast when a test interpreter cannot pass pbtest's Tessera pin gate.

``pbtest.py`` refuses a PrismaQuant checkout before pytest unless the target
interpreter carries ``tessera-quant`` as a non-editable Git install at the
reviewed commit (``tools/resolve_tessera_dev_pin.py``). A local-directory
install records no Git commit, so the gate can never pass for it -- that is
RobTand/prismaquant#753 (fleet env ``pq-cpu312``) and RobTand/prismabuild#658.

This check asks the same provenance question locally, before a submission
burns a fleet slot learning the answer. It checks PROVENANCE only: the
byte-identity audit stays worker-side in pbtest, and re-provisioning a shared
interpreter stays the fleet owner's idle-window work
(``tools/provision_tessera_pin.py``), never this tool's.

Usage::

    python tools/check_tessera_dev_install.py --python /home/rob/venvs/pq-cpu312/bin/python

Exit 0 when the interpreter carries the reviewed commit; exit 1 with the
remediation (provision command, idle-window requirement, owning issue) when
it does not. Imports neither PrismaQuant nor Tessera, so it runs anywhere.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_RESOLVER = ROOT / "tools" / "resolve_tessera_dev_pin.py"
DIST_NAME = "tessera-quant"
PB_ISSUE = "RobTand/prismabuild#658"
PQ_ISSUE = "RobTand/prismaquant#753"

#: The probe runs in the target interpreter WITHOUT ``-I``: pbtest's own
#: gate inspects the live import environment, including a shadowing checkout
#: on ``PYTHONPATH`` (which it refuses), so a pre-submit check that hid the
#: shadow would certify an interpreter the gate then refuses.
_PROBE = (
    "import json, sys\n"
    "from importlib import metadata\n"
    "try:\n"
    f"    dist = metadata.distribution({DIST_NAME!r})\n"
    "    direct = json.loads(dist.read_text('direct_url.json') or '{}')\n"
    "    out = {'distribution': dist.metadata['Name'], 'direct_url': direct}\n"
    "except Exception as exc:\n"
    "    out = {'reason': type(exc).__name__ + ': ' + str(exc)}\n"
    "sys.stdout.write(json.dumps(out))\n"
)


def reviewed_commit() -> str:
    """The commit pbtest will demand, read the way pbtest reads it."""
    done = subprocess.run([sys.executable, str(PIN_RESOLVER)],
                          capture_output=True, text=True, check=False)
    if done.returncode or not done.stdout.strip():
        raise SystemExit(f"{PIN_RESOLVER} failed: {done.stderr.strip()}")
    return done.stdout.strip()


def probe_direct_url(python: str) -> dict:
    """That interpreter's ``tessera-quant`` provenance, or its reason."""
    try:
        done = subprocess.run([python, "-c", _PROBE],
                              capture_output=True, text=True)
    except OSError as exc:
        return {"reason": f"{type(exc).__name__}: {exc}"}
    try:
        out = json.loads(done.stdout.strip() or "{}")
    except ValueError:
        out = {}
    if isinstance(out.get("direct_url"), dict):
        return out
    reason = out.get("reason") or (done.stderr.strip().splitlines() or
                                   ["the probe printed nothing"])[-1]
    return {"reason": reason}


def verdict(provenance: dict, expected: str, python: str) -> tuple[bool, str]:
    """Whether this provenance passes the gate, and the message either way.

    The rule is pbtest's (prismabuild ``tools/pbtest_pins.py``): exactly one
    owning distribution, a ``vcs_info`` naming git at the reviewed commit,
    and not an editable install. A local-directory install carries no
    ``vcs_info`` commit, so it refuses here exactly as it refuses there.
    """
    reason = provenance.get("reason")
    if reason is not None:
        return False, (
            f"{python}: no {DIST_NAME} provenance ({reason}); expected a "
            f"non-editable Git install at {expected}. {remediation(python)}")
    direct = provenance.get("direct_url") or {}
    vcs = direct.get("vcs_info") or {}
    observed = vcs.get("commit_id")
    editable = bool(direct.get("dir_info", {}).get("editable"))
    if vcs.get("vcs") == "git" and observed == expected and not editable:
        return True, (
            f"{python}: {DIST_NAME} is a non-editable Git install at the "
            f"reviewed commit {expected}")
    if observed is None:
        detail = ("a local-directory install, which records no Git commit"
                  if vcs == {} else
                  f"a non-git install ({direct!r})")
    elif editable:
        detail = f"an editable install at {observed}"
    else:
        detail = f"commit {observed}, expected {expected}"
    return False, (
        f"{python}: {DIST_NAME} is {detail}; pbtest refuses before pytest "
        f"({PQ_ISSUE}). {remediation(python)}")


def remediation(python: str) -> str:
    """The one repair and who may run it, pinned in the refusal itself."""
    return (
        f"Repair: python tools/provision_tessera_pin.py --python {python} "
        f"on the box that owns that interpreter, with every user of it idle "
        f"-- never under a running shard. Re-provisioning the shared fleet "
        f"interpreter is the fleet owner's call ({PB_ISSUE}); this check "
        f"only reports.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True,
                        help="interpreter a pbtest shard would run")
    args = parser.parse_args(argv)
    expected = reviewed_commit()
    ok, message = verdict(probe_direct_url(args.python), expected, args.python)
    print(message, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
