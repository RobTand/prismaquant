#!/usr/bin/env python3
"""Check installed Tessera bytes and import ownership against the reviewed Git pin.

Expected bytes come from a read-only Git clone archived into private scratch
by the existing provisioner. This check installs nothing and imports neither
Tessera nor its serving stack.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import subprocess
import tempfile

pin = runpy.run_path(str(Path(__file__).with_name("provision_tessera_pin.py")))


def reviewed_commit() -> str:
    return pin["reviewed_pin"]()[0]


def check_install(python: str, *, clone: Path,
                  pin_source: Path = pin["PIN_SOURCE"]) -> tuple[bool, dict]:
    commit, contract_sha = pin["reviewed_pin"](pin_source)
    report = {"python": python, "reviewed_commit": commit,
              "expected": None, "installed": None, "refusal": None, "passed": False}
    try:
        with tempfile.TemporaryDirectory(prefix="tessera-pin-check-") as temporary:
            source = Path(temporary) / "source"
            pin["_archive_into"](commit, clone, source)
            unshipped = pin["unshipped_packages"](source)
            expected = pin["expected_identity"](source, unshipped)
            report["expected"] = expected
            if expected["contract_sha256"] != contract_sha:
                raise ValueError("pinned Git source contract differs from reviewed digest")
            installed, absent = pin["installed_identity"](python, unshipped)
            report["installed"] = installed
            if absent is not None:
                raise ValueError(absent)
            if installed["distribution"] != "tessera-quant" or installed["import_owners"] != ["tessera-quant"]:
                raise ValueError("missing or ambiguous tessera-quant import ownership")
            package = Path(installed["package_path"])
            owner = Path(installed["distribution_package_path"])
            if package.is_symlink() or package.resolve(strict=True) != owner.resolve(strict=True):
                raise ValueError("Tessera import is shadowed or symlinked outside its distribution")
            direct = installed["direct_url"]
            if not isinstance(direct, dict) or not isinstance(direct.get("dir_info", {}), dict):
                raise ValueError("invalid installed provenance")
            if direct.get("dir_info", {}).get("editable"):
                raise ValueError("editable install refused")
            vcs = direct.get("vcs_info")
            if vcs is not None and (not isinstance(vcs, dict) or vcs.get("vcs") != "git" or vcs.get("commit_id") != commit):
                raise ValueError("installed Git provenance differs from reviewed commit")
            fields = pin["drift"](installed, expected)
            if fields:
                raise ValueError("installed bytes differ: " + ", ".join(fields))
            report["passed"] = True
    except (OSError, ValueError, subprocess.SubprocessError, SystemExit) as exc:
        report["refusal"] = f"{type(exc).__name__}: {exc}"
    return report["passed"], report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True)
    parser.add_argument("--clone", type=Path, default=pin["DEFAULT_CLONE"])
    args = parser.parse_args(argv)
    ok, report = check_install(args.python, clone=args.clone)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
