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
PrismaQuant nor Tessera and runs on an interpreter that has neither.

**The pin names a commit, and the check has to be the commit.**  An earlier
version of this tool compared the packaged ``runtime_contract.json`` alone and
stopped there.  That file is one file in the commit's tree, and two commits can
publish identical contract bytes while differing everywhere else -- tessera#437
rewrites the window encoder and leaves the contract untouched -- so a
contract-only gate reports "already the reviewed bytes" for a re-pin whose
entire content is new encoder code, and leaves the old encoder installed.  What
is compared now is a digest over the installed ``tessera`` package's own files
as well, taken under ``-I`` so it is the installed distribution answering and
not a checkout that happens to be on ``PYTHONPATH``.  The version string is no
check at all: the pin's version is static ``0.1.0`` and so was the stale
install's.

Usage::

    python tools/provision_tessera_pin.py --python /home/rob/venvs/pq-cpu312/bin/python

By default the source is materialised under ``/mnt/shared/tessera-pins/<commit>``
from a local Tessera clone.  That directory is a cache and not a
content-addressed store, whatever its name suggests: an interrupted ``tar``
leaves one with the right name and some of the right files.  So each tree is
written beside a manifest of its own digest, published with an atomic
``os.replace``, and reused only while it still digests to what the manifest
says.  A tree that does not, or that has no manifest, is re-materialised, which
needs the clone.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
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


MANIFEST = ".pinned-source.json"
#: Compiled bytecode is not source and does not travel with a commit, so it is
#: excluded from every digest here.  A tree's identity would otherwise depend
#: on whether anything had imported it, which is how an encoder identity hash
#: once came to cover ``.pyc`` files.
SKIP_DIRS = {"__pycache__", ".git"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
#: Excluded by name: it is written into the tree it describes.
SKIP_NAMES = {MANIFEST}


def tree_digest(root: Path) -> tuple[str, int]:
    """``(sha256 over the tree's paths and bytes, file count)``.

    Path-and-content, not content alone: a tree that lost a file entirely, and
    a tree whose bytes moved, are both things this has to see, and a digest
    over concatenated contents would miss a rename.
    """
    h = hashlib.sha256()
    n = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root)
        if (set(rel.parts) & SKIP_DIRS or path.suffix in SKIP_SUFFIXES
                or path.name in SKIP_NAMES):
            continue
        h.update(str(rel).encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        h.update(b"\n")
        n += 1
    return h.hexdigest(), n


def package_digest(root: Path) -> tuple[str, int]:
    """The digest of the ``tessera`` package alone, as an install would see it.

    A source tree carries ``pyproject.toml``, tests and CI that an install
    does not, so the two are only comparable over the package directory.  Paths
    are taken relative to it, so the same package under ``src/`` and under
    ``site-packages/`` digests identically.
    """
    return tree_digest(root)


def _archive_into(commit: str, clone: Path, dest: Path) -> None:
    if not (clone / ".git").exists():
        raise SystemExit(f"{clone} is not a Tessera clone; pass --clone")
    dest.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(clone),
         "archive", "--format=tar", commit],
        check=True, stdout=subprocess.PIPE,
    )
    subprocess.run(["tar", "-x", "-C", str(dest)],
                   check=True, input=archive.stdout)


def materialise(commit: str, clone: Path, pins_root: Path) -> Path:
    """Lay down ``commit``'s tree under ``pins_root``, verified, or repair it.

    The first version returned any directory named for the commit that had a
    ``runtime_contract.json`` somewhere in it.  A tree left half-extracted by
    an interrupted ``tar``, or edited afterwards, satisfies both of those and
    is not the commit -- so it would have installed something else under the
    commit's label, which is the one thing a pin exists to prevent.

    So the tree is written beside its own manifest, and a cached tree is
    reused only when it still digests to what the manifest says.  Extraction
    goes to a scratch directory and is moved into place with ``os.replace``,
    which is atomic within a filesystem: a reader either sees the previous
    tree or the new one, never a partial one, and an interrupted run leaves
    scratch behind rather than a plausible-looking half tree.
    """
    target = pins_root / commit
    manifest = target / MANIFEST
    if manifest.exists():
        try:
            held = json.loads(manifest.read_text(encoding="utf-8"))
        except ValueError:
            held = {}
        if held.get("commit") == commit:
            digest, count = tree_digest(target)
            if (digest == held.get("tree_sha256")
                    and count == held.get("files")):
                return target
        # Fall through and repair.  Saying which way it failed is worth a line
        # to whoever reads the log: a digest mismatch and a missing manifest
        # are different accidents.
        print(f"{target}: cached tree does not match its manifest, replacing",
              file=sys.stderr)
    elif target.exists():
        print(f"{target}: cached tree has no manifest, replacing",
              file=sys.stderr)

    scratch = pins_root / f".materialising-{commit}-{os.getpid()}"
    if scratch.exists():
        shutil.rmtree(scratch)
    try:
        _archive_into(commit, clone, scratch)
        if not (scratch / CONTRACT_IN_TREE).exists():
            raise SystemExit(f"{commit} carries no {CONTRACT_IN_TREE}")
        # The manifest lives inside the tree it describes, so ``tree_digest``
        # excludes it by name; otherwise writing it would change the thing it
        # records and no cached tree could ever verify.
        digest, count = tree_digest(scratch)
        (scratch / MANIFEST).write_text(json.dumps(
            {"commit": commit, "tree_sha256": digest, "files": count},
            indent=1), encoding="utf-8")
        if target.exists():
            doomed = pins_root / f".replaced-{commit}-{os.getpid()}"
            os.replace(target, doomed)
            shutil.rmtree(doomed, ignore_errors=True)
        os.replace(scratch, target)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return target


def installed_identity(python: str) -> tuple[dict | None, str | None]:
    """That interpreter's Tessera: ``(identity, reason it is absent)``.

    Exactly one of the two is set.  The identity carries the packaged
    contract's sha256 AND a digest over the installed package's own files,
    because the pin names a COMMIT and the contract is one file in that
    commit's tree.  Two commits can publish identical contract bytes and
    differ everywhere else -- tessera#437 rewrites the window encoder and
    leaves the contract alone -- so a contract-only check reports "already the
    reviewed bytes" for a re-pin whose whole content is new encoder code, and
    leaves the old encoder installed.  That was this tool's own defect.

    The reason is carried rather than dropped because an absent Tessera has
    several causes an operator acts on differently: no such interpreter, no
    Tessera in it, or a Tessera installed without its packaged contract.  The
    GB10 venv reported a bare ``null`` on its first audit; it was the second.

    Run under ``-I``, so what is measured is the INSTALLED distribution.
    Without it a checkout on ``PYTHONPATH`` -- or a shadowing directory in the
    caller's cwd -- answers instead, and the tool would certify a package the
    fleet does not import.
    """

    probe = (
        "import hashlib,json,sys\n"
        "from importlib import resources\n"
        "from pathlib import Path\n"
        "SKIP_DIRS={'__pycache__','.git'}\n"
        "SKIP_SUF={'.pyc','.pyo'}\n"
        "def digest(root):\n"
        "    h=hashlib.sha256(); n=0\n"
        "    for p in sorted(q for q in root.rglob('*') if q.is_file()):\n"
        "        rel=p.relative_to(root)\n"
        "        if set(rel.parts)&SKIP_DIRS or p.suffix in SKIP_SUF: continue\n"
        "        h.update(str(rel).encode()); h.update(b'\\0')\n"
        "        h.update(hashlib.sha256(p.read_bytes()).hexdigest().encode())\n"
        "        h.update(b'\\n'); n+=1\n"
        "    return h.hexdigest(), n\n"
        "try:\n"
        "    c = resources.files('tessera.serving')"
        ".joinpath('runtime_contract.json')\n"
        "    pkg = Path(str(resources.files('tessera')))\n"
        "    d, n = digest(pkg)\n"
        "    out = {'contract_sha256': hashlib.sha256(c.read_bytes()).hexdigest(),\n"
        "           'package_sha256': d, 'package_files': n,\n"
        "           'package_path': str(pkg)}\n"
        "except Exception as exc:\n"
        "    out = {'reason': type(exc).__name__ + ': ' + str(exc)}\n"
        "sys.stdout.write(json.dumps(out))\n"
    )
    try:
        done = subprocess.run([python, "-I", "-c", probe],
                              capture_output=True, text=True)
    except OSError as exc:
        # No such interpreter, or one that will not start.  The caller wants
        # "not the reviewed source", and an exception here would make a routine
        # audit of a machine that has no such venv look like a tool failure.
        return None, f"{type(exc).__name__}: {exc}"
    try:
        out = json.loads(done.stdout.strip() or "{}")
    except ValueError:
        out = {}
    if out.get("contract_sha256") and out.get("package_sha256"):
        return out, None
    reason = out.get("reason") or (done.stderr.strip().splitlines() or
                                   ["the probe printed nothing"])[-1]
    return None, reason


def expected_identity(source: Path) -> dict:
    """What a correct install of the pinned tree would report."""

    pkg = source / "src" / "tessera"
    if not pkg.is_dir():
        raise SystemExit(f"{source} carries no src/tessera")
    digest, count = package_digest(pkg)
    return {
        "contract_sha256": hashlib.sha256(
            (source / CONTRACT_IN_TREE).read_bytes()).hexdigest(),
        "package_sha256": digest,
        "package_files": count,
    }


def drift(installed: dict | None, expected: dict) -> list[str]:
    """Which fields disagree, named, so a report says what to repair."""

    if installed is None:
        return ["no Tessera to compare"]
    return [
        field for field in ("contract_sha256", "package_sha256")
        if installed.get(field) != expected[field]
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", required=True,
                        help="interpreter to provision")
    parser.add_argument("--clone", type=Path, default=DEFAULT_CLONE,
                        help="local Tessera clone to archive the pin from")
    parser.add_argument("--pins-root", type=Path, default=DEFAULT_PINS_ROOT,
                        help="where pinned source trees are materialised")
    parser.add_argument("--pin-source", type=Path, default=PIN_SOURCE,
                        help="module holding the reviewed pin literals")
    parser.add_argument("--check-only", action="store_true",
                        help="report what is installed, install nothing")
    args = parser.parse_args(argv)

    commit, reviewed_sha = reviewed_pin(args.pin_source)
    installed, absent = installed_identity(args.python)
    report = {
        "python": args.python,
        "reviewed_commit": commit,
        "reviewed_contract_sha256": reviewed_sha,
        "installed_before": installed,
    }
    if absent is not None:
        report["installed_absent_before"] = absent

    # The pinned tree is materialised BEFORE the decision, not after it,
    # because the decision needs the commit's package digest and only the tree
    # has it.  The contract hash in the pin is a second, independent check on
    # the tree, and it is checked here rather than trusted.
    source = materialise(commit, args.clone, args.pins_root)
    expected = expected_identity(source)
    report["source"] = str(source)
    report["expected"] = expected
    if expected["contract_sha256"] != reviewed_sha:
        raise SystemExit(
            f"{source} publishes contract {expected['contract_sha256']}, "
            f"reviewed is {reviewed_sha}"
        )

    fields = drift(installed, expected)
    report["drift"] = fields
    if not fields:
        report["action"] = "none, already the reviewed source"
        print(json.dumps(report, indent=1))
        return 0
    if args.check_only:
        report["action"] = "none, --check-only"
        print(json.dumps(report, indent=1))
        return 1

    subprocess.run(
        [args.python, "-m", "pip", "install", "--no-deps",
         "--no-build-isolation", "--force-reinstall", str(source)],
        check=True,
    )

    after, after_absent = installed_identity(args.python)
    report["installed_after"] = after
    if after_absent is not None:
        report["installed_absent_after"] = after_absent
    remaining = drift(after, expected)
    report["drift_after"] = remaining
    if remaining:
        report["action"] = "installed, and it did NOT take"
        print(json.dumps(report, indent=1))
        return 1
    report["action"] = "installed the reviewed source"
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
