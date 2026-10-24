# Original-source witnesses, extracted without transforming bodies.
import hashlib, json, os, sys, subprocess
from pathlib import Path
from typing import *
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
        "    c = resources.files('tessera')"
        ".joinpath('serving', 'runtime_contract.json')\n"
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

