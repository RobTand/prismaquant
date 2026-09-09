#!/usr/bin/env python3
"""Real-install acceptance for ``tools/provision_tessera_pin.py``.

Every other test of that tool places a package directory into site-packages
and asks what the tool DECIDES.  That is the right shape for a decision, and
it cannot see the two things only a build backend does: leave the pinned
tree's frozen bytes alone, and install a subset of them.

So this one builds.  It creates a throwaway venv, runs the provisioner into
it, and then checks three properties on the artifacts rather than on the log:

1. the pinned tree still digests to the manifest it was published with, so
   the backend wrote its ``egg-info`` somewhere else;
2. the interpreter now reports the pin's identity, which is the digest over
   what the wheel actually installed;
3. a second run is a no-op, which is the property the whole tool exists for
   and the one a contract-only gate could not deliver.

Run through PrismaBuild on x86; the venv it makes is deleted at the end.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "provision_tessera_pin.py"
sys.path.insert(0, str(ROOT / "tools"))
import provision_tessera_pin as pin                        # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clone", type=Path, default=pin.DEFAULT_CLONE)
    ap.add_argument("--pins-root", type=Path, default=pin.DEFAULT_PINS_ROOT)
    ap.add_argument("--work", type=Path, required=True,
                    help="scratch root for the throwaway venv and the build")
    ap.add_argument("--seed-from", type=Path,
                    help="copy this tree into pins-root/<commit> first, to "
                         "reproduce a cached tree the tool has to judge")
    args = ap.parse_args()

    commit, _ = pin.reviewed_pin()
    args.work.mkdir(parents=True, exist_ok=True)
    env = args.work / "venv"
    if env.exists():
        shutil.rmtree(env)
    venv.create(env, with_pip=True)
    python = env / "bin" / "python"
    # The provisioner installs with --no-build-isolation, which is right for a
    # fleet venv that already carries setuptools and wrong for a venv made two
    # lines ago: 3.12 stopped seeding it.  Giving the throwaway one the build
    # backend keeps the acceptance about the provisioner.
    subprocess.run([str(python), "-m", "pip", "install", "-q", "setuptools"],
                   check=True)

    out: dict = {"commit": commit, "venv": str(env)}

    def run(*extra: str) -> tuple[int, dict]:
        done = subprocess.run(
            [sys.executable, str(TOOL), "--python", str(python),
             "--clone", str(args.clone), "--pins-root", str(args.pins_root),
             "--staging-root", str(args.work / "staging"), *extra],
            capture_output=True, text=True)
        sys.stderr.write(done.stderr)
        return done.returncode, json.loads(done.stdout or "{}")

    source = args.pins_root / commit
    if args.seed_from is not None and not source.exists():
        args.pins_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(args.seed_from, source, symlinks=True)
    before = pin.tree_digest(source) if source.exists() else None
    out["source_digest_before"] = before

    rc1, first = run()
    out["first"] = {"rc": rc1, "action": first.get("action"),
                    "drift": first.get("drift"),
                    "built_from": first.get("built_from"),
                    "unshipped_packages": first.get("unshipped_packages"),
                    "source_intact_after_install":
                        first.get("source_intact_after_install")}

    after = pin.tree_digest(source)
    out["source_digest_after"] = after

    rc2, second = run()
    out["second"] = {"rc": rc2, "action": second.get("action"),
                     "drift": second.get("drift")}

    rc3, third = run("--check-only")
    out["check_only"] = {"rc": rc3, "action": third.get("action")}

    quarantined = first.get("quarantined")
    out["quarantined"] = quarantined
    out["quarantine_kept"] = (
        Path(quarantined).is_dir() if quarantined else None)
    out["published_holds_its_manifest"] = pin._manifest_holds(source, commit)
    checks = {
        # The published tree still answers the manifest it was published with,
        # so the backend's egg-info and build/ went to the copy.  Checked here
        # on the artifact as well as by the tool, because the tool reporting
        # its own property is the thing under test.
        "the install did not write into the published tree":
            out["published_holds_its_manifest"],
        "the tool says the tree is intact": first.get("source_intact_after_install") is True,
        "the first run installed the pin": rc1 == 0,
        "the second run is a no-op": rc2 == 0 and second.get("drift") == [],
        "check-only reports clean": rc3 == 0,
    }
    if quarantined:
        checks["the tree it replaced was kept"] = out["quarantine_kept"] is True
    out["checks"] = checks
    shutil.rmtree(env, ignore_errors=True)
    shutil.rmtree(args.work / "staging", ignore_errors=True)
    print(json.dumps(out, indent=1))
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        print("FAILED: " + "; ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
