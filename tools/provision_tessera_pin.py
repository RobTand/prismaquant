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
import tempfile
import time
import tomllib
from fnmatch import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIN_SOURCE = ROOT / "prismaquant" / "tessera_runtime_contract.py"
PIN_NAME = "TESSERA_DEV_PIN_COMMIT"
SHA_NAME = "TESSERA_DEV_PIN_CONTRACT_SHA256"
DEFAULT_PINS_ROOT = Path("/mnt/shared/tessera-pins")
DEFAULT_CLONE = Path("/home/rob/tessera")
CONTRACT_IN_TREE = Path("src/tessera/serving/runtime_contract.json")
PACKAGE_IN_TREE = Path("src/tessera")
PYPROJECT_IN_TREE = Path("pyproject.toml")
DEFAULT_STAGING_ROOT = Path("/home/rob/tmp")


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


def tree_digest(root: Path, skip=None) -> tuple[str, int]:
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
        if skip is not None and skip(rel):
            continue
        h.update(str(rel).encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("ascii"))
        h.update(b"\n")
        n += 1
    return h.hexdigest(), n


def unshipped_packages(source: Path) -> list[str]:
    """Dotted subpackages the pinned tree's build config does not install.

    Read from the tree rather than known here.  Tessera excludes
    ``tessera._dev*`` -- its own merge-suite and import-graph tooling, which
    lives under ``src/`` because ``tools/`` imports it by module name.  A
    digest that counted those five files would compare a 76-file source tree
    against a 71-file install and report drift on a correctly provisioned
    interpreter, every run, forever.  The exclusion is a fact about the build
    backend, so it is taken from the backend's own configuration; Tessera's
    ``tools/check_wheel.py`` proves it on the built artifact.
    """

    pyproject = source / PYPROJECT_IN_TREE
    if not pyproject.is_file():
        raise SystemExit(f"{source} carries no {PYPROJECT_IN_TREE}")
    config = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    find = (config.get("tool", {}).get("setuptools", {})
            .get("packages", {}).get("find", {}))
    return list(find.get("exclude", []))


def _package_skip(patterns: list[str], package: str = "tessera"):
    """A ``tree_digest`` predicate for those patterns, over package paths."""

    if not patterns:
        return None

    def skip(rel: Path) -> bool:
        dotted = ".".join((package, *rel.parts[:-1]))
        return any(fnmatch(dotted, pattern) for pattern in patterns)

    return skip


def package_digest(root: Path, unshipped: list[str] | None = None):
    """The digest of the ``tessera`` package alone, as an install would see it.

    A source tree carries ``pyproject.toml``, tests and CI that an install
    does not, so the two are only comparable over the package directory.  Paths
    are taken relative to it, so the same package under ``src/`` and under
    ``site-packages/`` digests identically -- and subpackages the build config
    excludes are dropped on both sides, because an install never has them and
    a source tree always does.
    """
    return tree_digest(root, _package_skip(unshipped or []))


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


def _manifest_holds(target: Path, commit: str) -> bool:
    """Does ``target`` still digest to what its own manifest recorded?"""

    manifest = target / MANIFEST
    if not manifest.is_file():
        return False
    try:
        held = json.loads(manifest.read_text(encoding="utf-8"))
    except ValueError:
        return False
    if held.get("commit") != commit:
        return False
    digest, count = tree_digest(target)
    return digest == held.get("tree_sha256") and count == held.get("files")


def _write_manifest(target: Path, commit: str) -> None:
    """Record what ``target`` digests to, atomically, beside itself.

    The manifest lives inside the tree it describes, so ``tree_digest``
    excludes it by name; otherwise writing it would change the thing it
    records and no cached tree could ever verify.
    """
    digest, count = tree_digest(target)
    body = json.dumps({"commit": commit, "tree_sha256": digest,
                       "files": count}, indent=1)
    scratch = target / f".{MANIFEST}.writing"
    scratch.write_text(body, encoding="utf-8")
    os.replace(scratch, target / MANIFEST)


class _PinsLock:
    """One writer at a time per commit, across hosts on the shared root.

    Two boxes provisioning the same pin at once is the ordinary case, not the
    exotic one: PrismaBuild runs actions concurrently and the pins root is
    NFS.  Without this they race to publish the same directory, and the loser
    quarantines the winner's freshly published tree as an intruder.
    """

    def __init__(self, pins_root: Path, commit: str) -> None:
        self.path = pins_root / f".lock-{commit}"

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a+")
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        import fcntl
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        return False


def verified_source(commit: str, pins_root: Path) -> Path | None:
    """``pins_root/commit`` if it verifies against its own manifest, else None.

    Reads only.  ``--check-only`` runs on this, because a read-only check that
    writes the pinned tree into a shared directory to learn what to compare
    against is not read-only.
    """
    target = pins_root / commit
    return target if _manifest_holds(target, commit) else None


def materialise(commit: str, clone: Path, pins_root: Path,
                report: dict | None = None) -> Path:
    """Lay down ``commit``'s tree under ``pins_root``, verified, or repair it.

    Three states, three different right answers, and the first version of this
    collapsed them into one:

    * **Manifested and matching.** Reused, no clone needed.
    * **No manifest, right bytes.** ``/mnt/shared/tessera-pins/07ad344c...``
      predates the manifest and is correct.  Its content is compared against a
      fresh ``git archive`` of the commit and, when equal, the manifest is
      written beside it.  The bytes are not replaced: rewriting a shared
      directory other boxes may be reading, to end with what is already there,
      is churn with a window in it.
    * **Anything else.** The tree is moved aside under a
      ``.quarantine-<commit>-...`` name and the fresh archive published in its
      place.  It is not deleted.  Something wrote to a shared pins root and
      those bytes are the only record of what; the quarantine path is returned
      in ``report`` so a run says where it went.

    Publication is ``os.replace`` of a directory that is complete and already
    manifested, so a reader sees the old tree or the new one.  Staging is
    ``mkdtemp`` under the pins root -- unique and owned, where the earlier
    PID-named scratch could collide between hosts on NFS and was removed by
    name, which is a directory another box may be extracting into.
    """
    target = pins_root / commit
    if _manifest_holds(target, commit):
        return target

    pins_root.mkdir(parents=True, exist_ok=True)
    with _PinsLock(pins_root, commit):
        # Re-checked under the lock: another box may have published it while
        # this one waited, and that tree is as good as one written here.
        if _manifest_holds(target, commit):
            return target

        staging = Path(tempfile.mkdtemp(dir=pins_root,
                                        prefix=f".materialising-{commit}-"))
        fresh = staging / "tree"
        try:
            _archive_into(commit, clone, fresh)
            if not (fresh / CONTRACT_IN_TREE).exists():
                raise SystemExit(f"{commit} carries no {CONTRACT_IN_TREE}")

            if target.exists():
                if tree_digest(target)[0] == tree_digest(fresh)[0]:
                    # Already the commit, only unattested.  Attest in place.
                    _write_manifest(target, commit)
                    if report is not None:
                        report["source_state"] = "verified against a fresh archive"
                    return target
                quarantine = pins_root / (
                    f".quarantine-{commit}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
                    f"-{staging.name.rsplit('-', 1)[-1]}")
                os.replace(target, quarantine)
                print(f"{target}: does not hold {commit}; kept at {quarantine}",
                      file=sys.stderr)
                if report is not None:
                    report["quarantined"] = str(quarantine)

            _write_manifest(fresh, commit)
            os.replace(fresh, target)
            if report is not None:
                report.setdefault("source_state", "published from the clone")
            return target
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def installed_identity(python: str,
                       unshipped: list[str] | None = None,
                       ) -> tuple[dict | None, str | None]:
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

    # The exclusion travels INTO the probe rather than being applied to its
    # answer: an editable install points at a checkout that does carry the
    # unshipped subpackage, and only the digest that skipped it on both sides
    # compares the two halves of one pin.
    probe = (
        "import hashlib,json,sys\n"
        "from fnmatch import fnmatch\n"
        "from importlib import resources\n"
        "from pathlib import Path\n"
        "SKIP_DIRS={'__pycache__','.git'}\n"
        "SKIP_SUF={'.pyc','.pyo'}\n"
        f"UNSHIPPED={list(unshipped or [])!r}\n"
        "def digest(root):\n"
        "    h=hashlib.sha256(); n=0\n"
        "    for p in sorted(q for q in root.rglob('*') if q.is_file()):\n"
        "        rel=p.relative_to(root)\n"
        "        if set(rel.parts)&SKIP_DIRS or p.suffix in SKIP_SUF: continue\n"
        "        dotted='.'.join(('tessera',)+rel.parts[:-1])\n"
        "        if any(fnmatch(dotted,x) for x in UNSHIPPED): continue\n"
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


def expected_identity(source: Path,
                      unshipped: list[str] | None = None) -> dict:
    """What a correct install of the pinned tree would report."""

    pkg = source / PACKAGE_IN_TREE
    if not pkg.is_dir():
        raise SystemExit(f"{source} carries no {PACKAGE_IN_TREE}")
    digest, count = package_digest(pkg, unshipped)
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
    parser.add_argument("--staging-root", type=Path,
                        default=DEFAULT_STAGING_ROOT,
                        help="where the build's copy of the pinned tree goes")
    parser.add_argument("--check-only", action="store_true",
                        help="report what is installed, write nothing")
    args = parser.parse_args(argv)

    commit, reviewed_sha = reviewed_pin(args.pin_source)
    report = {
        "python": args.python,
        "reviewed_commit": commit,
        "reviewed_contract_sha256": reviewed_sha,
    }

    # --check-only reads.  It does not materialise, because the pins root is
    # shared and a read-only check that lays a tree down in it is not one; an
    # operator running it to see where a box stands would find a directory
    # written by the question.  So it answers from a source that already
    # verifies, and says so when there is none rather than making one.
    if args.check_only:
        source = verified_source(commit, args.pins_root)
        if source is None:
            report["installed_before"], absent = installed_identity(args.python)
            if absent is not None:
                report["installed_absent_before"] = absent
            report["action"] = "none, --check-only and no verified pinned source"
            report["repair"] = (
                f"run without --check-only to publish {args.pins_root / commit} "
                "from the clone")
            print(json.dumps(report, indent=1))
            return 1
    else:
        source = materialise(commit, args.clone, args.pins_root, report)

    unshipped = unshipped_packages(source)
    report["source"] = str(source)
    report["unshipped_packages"] = unshipped
    expected = expected_identity(source, unshipped)
    report["expected"] = expected
    if expected["contract_sha256"] != reviewed_sha:
        raise SystemExit(
            f"{source} publishes contract {expected['contract_sha256']}, "
            f"reviewed is {reviewed_sha}"
        )

    installed, absent = installed_identity(args.python, unshipped)
    report["installed_before"] = installed
    if absent is not None:
        report["installed_absent_before"] = absent

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

    # The build gets a copy.  ``pip install <dir>`` runs the backend IN that
    # directory, and setuptools writes ``*.egg-info`` (and, uncached, a
    # ``build/``) into it -- so pointing pip at the pinned tree makes the tree
    # stop matching the manifest it was published with, and the next run
    # reports the pin's own source corrupt.  The frozen tree is an input.
    args.staging_root.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(dir=args.staging_root,
                                  prefix=f"tessera-build-{commit[:12]}-"))
    try:
        copy = build / "tree"
        shutil.copytree(source, copy)
        report["built_from"] = str(copy)
        subprocess.run(
            [args.python, "-m", "pip", "install", "--no-deps",
             "--no-build-isolation", "--force-reinstall", str(copy)],
            check=True,
        )
    finally:
        shutil.rmtree(build, ignore_errors=True)

    # And proved rather than argued: the tree still answers its own manifest.
    report["source_intact_after_install"] = _manifest_holds(source, commit)

    after, after_absent = installed_identity(args.python, unshipped)
    report["installed_after"] = after
    if after_absent is not None:
        report["installed_absent_after"] = after_absent
    remaining = drift(after, expected)
    report["drift_after"] = remaining
    if remaining or not report["source_intact_after_install"]:
        report["action"] = "installed, and it did NOT take"
        print(json.dumps(report, indent=1))
        return 1
    report["action"] = "installed the reviewed source"
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
