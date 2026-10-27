#!/usr/bin/env python3
"""Build and validate a committed candidate using a temporary, local-only tag."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from packaging.utils import parse_sdist_filename, parse_wheel_filename
from packaging.version import Version


def isolated_environment(home: Path) -> dict[str, str]:
    return {
        "PATH": os.defpath,
        "HOME": str(home),
        "TMPDIR": str(home),
        "LANG": "C.UTF-8",
        "PYTHONNOUSERSITE": "1",
        "PIP_NO_INDEX": "1",
        "PIP_CONFIG_FILE": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    }


def validate_tag(tag: str, dist: Path) -> str:
    if not re.fullmatch(r"v[0-9][A-Za-z0-9.!+_-]*", tag):
        raise ValueError("tag must be a local v-prefixed package version")
    expected = Version(tag[1:])
    wheels = list(dist.glob("*.whl"))
    sdists = list(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("expected exactly one wheel and one sdist")
    wheel_name, wheel_version, _, _ = parse_wheel_filename(wheels[0].name)
    sdist_name, sdist_version = parse_sdist_filename(sdists[0].name)
    if wheel_name != "prismaquant" or sdist_name != "prismaquant":
        raise ValueError("candidate artifacts must both be prismaquant")
    if expected != wheel_version or expected != sdist_version:
        raise ValueError(
            f"tag {tag} disagrees with wheel {wheel_version} or sdist {sdist_version}"
        )
    return str(expected)


def dry_run(repository: Path, revision: str, tag: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    receipt = {"schema": "prismaquant.release_dry_run.v1", "tag": tag, "result": "FAIL"}
    try:
        with tempfile.TemporaryDirectory(prefix="pq-release-dry-run-") as scratch:
            home = Path(scratch)
            environment = isolated_environment(home)
            source = home / "candidate"
            dist = output / "dist"
            with (output / "validation.log").open("w") as log:
                def execute(*arguments: str, cwd: Path = home) -> str:
                    completed = subprocess.run(
                        arguments, cwd=cwd, env=environment, text=True,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    )
                    log.write(f"$ {' '.join(arguments)}\n{completed.stdout}\n")
                    log.flush()
                    if completed.returncode:
                        raise RuntimeError(
                            f"{arguments[0]} {arguments[1]} failed ({completed.returncode}); "
                            "see validation.log"
                        )
                    return completed.stdout.strip()

                commit = execute(
                    "git", "-C", str(repository), "rev-parse", "--verify",
                    "--end-of-options", f"{revision}^{{commit}}",
                )
                if not re.fullmatch(r"[0-9a-f]{40}", commit):
                    raise ValueError("candidate must resolve to one full commit SHA")
                if not re.fullmatch(r"v[0-9][A-Za-z0-9.!+_-]*", tag):
                    raise ValueError("tag must be a local v-prefixed package version")
                Version(tag[1:])
                receipt["commit"] = commit
                execute("git", "init", "--quiet", str(source))
                execute(
                    "git", "-c", "protocol.file.allow=always", "fetch", "--no-tags",
                    str(repository), commit, cwd=source,
                )
                execute("git", "-c", "tag.gpgSign=false", "tag", "--", tag, commit, cwd=source)
                execute("git", "checkout", "--quiet", "--detach", f"refs/tags/{tag}", cwd=source)
                if execute("git", "rev-parse", "HEAD", cwd=source) != commit:
                    raise ValueError("local tag checkout does not match candidate SHA")
                execute(sys.executable, "-m", "build", "--no-isolation", "--outdir", str(dist), cwd=source)
                artifacts = sorted(dist.iterdir())
                execute(sys.executable, "-m", "twine", "check", "--strict", *(str(path) for path in artifacts))
                execute(sys.executable, str(Path(__file__).with_name("check_dist.py")), str(dist), str(source))
                receipt["version"] = validate_tag(tag, dist)
                receipt["artifacts"] = {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in artifacts
                }
                receipt["result"] = "PASS"
    except Exception as error:
        receipt["error"] = str(error)
        raise
    finally:
        (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    dry_run(arguments.repository.resolve(), arguments.revision, arguments.tag, arguments.output.resolve())


if __name__ == "__main__":
    main()
