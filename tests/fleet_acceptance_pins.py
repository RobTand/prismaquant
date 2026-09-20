"""Declared immutable pins for the fleet acceptance harness (PQ #859).

The PB candidate is resolved from a declared pinned checkout's git
objects -- no branch fetch, no remotes, no mutable paths:

- ``PB_PINNED_CHECKOUT``: the checkout whose object store carries the
  pinned commit (retired detached, read-only; never edited here).
- ``PB_CANDIDATE_REV``: the exact commit executed (R7 now; R8 when the
  author commits -- one line changes, the machinery does not).

Resolution verifies the WHOLE extracted tree against the revision
(archive digest recorded, file list equal to ``ls-tree``, every reused
byte under ``src/prismabuild`` + ``tools/fleet`` hash-checked), and the
run records the exact executed PB snapshot (rev + tree digest) and PQ
snapshot (worktree HEAD, required clean). A missing checkout, a missing
commit, or a verification mismatch is :class:`NonQualified` machine
output -- never a green row, never a silent substitution.

``git`` is invoked with ``--git-dir`` against the pinned checkout only;
no fetch, no checkout mutation, no remote required.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


PB_PINNED_CHECKOUT = Path("/home/rob/tmp/pb-reader-lease-pin-20260920")
PB_CANDIDATE_REPO = "https://github.com/RobTand/prismabuild.git"
PB_CANDIDATE_BRANCH = "fix/pb-reader-lifetime-20260920"
#: PB741 merge ("refs_for_holder signals unknown census", corrected SDK
#: 461728e4, root census 7+179 qualified). Full immutable pin for the final
#: harness: supersedes R8 0bf6fc81 (typed export proof, terminal replay),
#: R9 ee944cd8 (hold unproven claims with live refs for the worker
#: reaper), and R10 c50a7759 (production settlement path in connected
#: tests). The connected first-release scenario fails on R7 (a9bb83e9,
#: recorded in pqfleet-r7red-20260920.json) and passes here; the recovery
#: scenario needs the R9+ hold-and-reaper path in this tree. Deployed
#: runtime 0467e9e2316c claims no reader capability; this pin names
#: component behavior only until root accepts/deploys.
PB_CANDIDATE_REV = "c5012fbfc6986c551af5606ff9d6f0d874768116"

#: Tree paths this harness reuses (imports or executes). Every byte under
#: these prefixes is hash-verified against the pinned revision.
REUSED_PREFIXES = ("src/prismabuild/", "tools/fleet/")


class NonQualified(Exception):
    """A named nonqualified result: not failure, not conformance.

    Carries ``reason`` plus detail for the result JSON the runner writes.
    Callers (pytest scenarios) must surface it as SKIP with the reason,
    so a missing dependency can never read as a passing conformance test.
    """

    def __init__(self, reason: str, *, detail: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = dict(detail or {})


def _git(git_dir: Path, *args: str, timeout_s: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "--git-dir", str(git_dir), *args],
                          capture_output=True, text=True, timeout=timeout_s)


def _git_dir_of(checkout: Path) -> Path:
    """The true git dir, including for linked worktrees (whose .git is a file)."""
    dotgit = checkout / ".git"
    if dotgit.is_dir():
        return dotgit
    try:
        target = dotgit.read_text().strip()
    except OSError:
        raise NonQualified(
            "pinned PB checkout has no git metadata",
            detail={"checkout": str(checkout), "rev": PB_CANDIDATE_REV})
    if target.startswith("gitdir: "):
        resolved = Path(target[len("gitdir: "):])
        if not resolved.is_absolute():
            resolved = checkout / resolved
        if resolved.is_dir():
            return resolved
    raise NonQualified(
        "pinned PB checkout git metadata unreadable",
        detail={"checkout": str(checkout), "rev": PB_CANDIDATE_REV})


def _git_bytes(git_dir: Path, *args: str, timeout_s: int = 300) -> bytes:
    done = subprocess.run(["git", "--git-dir", str(git_dir), *args],
                          capture_output=True, timeout=timeout_s)
    if done.returncode != 0:
        raise NonQualified(
            f"git {' '.join(args[:2])} failed in pinned checkout",
            detail={"rev": PB_CANDIDATE_REV,
                    "stderr": done.stderr.decode()[-500:]})
    return done.stdout


def _resolve_via_network(dest: Path, *, timeout_s: int) -> Path:
    """Clone the candidate branch metadata; the pin must be in its history.

    Fallback when the pinned checkout is not visible from this worker.
    Branch deletion (retirement on merge) fails closed here: the pin then
    needs re-pointing at the merged home, never a silent substitution.
    """
    mirror = dest / "mirror.git"
    if not (mirror / "objects").is_dir():
        done = _run_git_no_dir(
            ["clone", "--bare", "--filter=blob:none", "--single-branch",
             "--branch", PB_CANDIDATE_BRANCH, PB_CANDIDATE_REPO,
             str(mirror)], timeout_s=timeout_s)
        if done.returncode != 0:
            raise NonQualified(
                "PB candidate unreachable",
                detail={"rev": PB_CANDIDATE_REV,
                        "stderr": done.stderr.strip()[-500:]})
    return mirror


def _run_git_no_dir(args: list[str], *, timeout_s: int) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          timeout=timeout_s)


def resolve_pb_candidate(dest: Path, *, timeout_s: int = 300) -> dict:
    """Extract the pinned commit beside nothing mutable; verify the tree.

    ``dest`` is caller-owned (pytest tmp). Returns ``{"tree": ...,
    "rev": ..., "tree_sha256": ...}`` where ``tree`` holds the exact
    revision content. Verification, all against the pinned revision:

    - the commit exists in the pinned object store;
    - the extracted file set equals ``ls-tree -r`` exactly;
    - every file under the reused prefixes hash-matches its blob.
    """
    try:
        git_dir = _git_dir_of(PB_PINNED_CHECKOUT)
        network = False
    except NonQualified:
        git_dir = None
        network = True
    if network:
        mirror = _resolve_via_network(dest, timeout_s=timeout_s)
        src_git_dir = mirror
    else:
        src_git_dir = git_dir
        mirror = None
    done = _git(src_git_dir, "cat-file", "-t", PB_CANDIDATE_REV,
                timeout_s=timeout_s)
    if done.returncode != 0 or done.stdout.strip() != "commit":
        raise NonQualified(
            "pinned commit missing from checkout objects",
            detail={"checkout": str(PB_PINNED_CHECKOUT),
                    "rev": PB_CANDIDATE_REV})
    dest.mkdir(parents=True, exist_ok=True)
    tree = dest / "tree"
    if not (tree / "src" / "prismabuild" / "reader_lease.py").is_file():
        import io
        import tarfile
        raw = _git_bytes(src_git_dir, "archive", PB_CANDIDATE_REV,
                         timeout_s=timeout_s)
        if not raw:
            raise NonQualified(
                "pinned commit archive empty",
                detail={"rev": PB_CANDIDATE_REV})
        tree_sha256 = hashlib.sha256(raw).hexdigest()
        tree.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r|") as member:
            member.extractall(path=str(tree))
        (dest / "tree.sha256").write_text(tree_sha256 + "\n")
    else:
        tree_sha256 = (dest / "tree.sha256").read_text().strip()
    listed = _git(src_git_dir, "ls-tree", "-r", "--name-only", PB_CANDIDATE_REV,
                  timeout_s=timeout_s)
    if listed.returncode != 0:
        raise NonQualified(
            "pinned commit tree unreadable",
            detail={"rev": PB_CANDIDATE_REV})
    expected = set(listed.stdout.split())
    actual = {str(p.relative_to(tree)) for p in tree.rglob("*") if p.is_file()}
    if actual != expected:
        raise NonQualified(
            "extracted tree differs from pinned revision",
            detail={"rev": PB_CANDIDATE_REV,
                    "missing": sorted(expected - actual)[:10],
                    "extra": sorted(actual - expected)[:10]})
    blobs = _git(src_git_dir, "ls-tree", "-r", PB_CANDIDATE_REV,
                 timeout_s=timeout_s)
    if blobs.returncode != 0:
        raise NonQualified(
            "pinned commit blobs unreadable",
            detail={"rev": PB_CANDIDATE_REV})
    digest_of = {}
    for line in blobs.stdout.splitlines():
        meta, _, name = line.partition("\t")
        mode, kind, digest = meta.split()
        if kind == "blob" and name.startswith(REUSED_PREFIXES):
            digest_of[name] = digest
    bad = []
    for name, digest in digest_of.items():
        data = (tree / name).read_bytes()
        if _git_blob_sha(data) != digest:
            bad.append(name)
    if bad:
        raise NonQualified(
            "reused bytes fail change detection",
            detail={"rev": PB_CANDIDATE_REV, "files": bad[:10]})
    if not digest_of:
        raise NonQualified(
            "no reused bytes found at pinned revision",
            detail={"rev": PB_CANDIDATE_REV})
    return {"tree": str(tree), "rev": PB_CANDIDATE_REV,
            "tree_sha256": tree_sha256}


def _git_blob_sha(data: bytes) -> str:
    """The git blob id bytes hash to (change detection primitive)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def record_snapshots(*, checkout: Path) -> dict:
    """Exact executed snapshots: PB rev + tree digest, PQ HEAD (must be clean)."""
    def _run(*args: str) -> str:
        done = subprocess.run(["git", "-C", str(checkout), *args],
                              capture_output=True, text=True, timeout=60)
        if done.returncode != 0:
            raise NonQualified(
                f"PQ checkout at {checkout} is not readable")
        return done.stdout.strip()
    head = _run("rev-parse", "HEAD")
    tracked = _run("status", "--porcelain", "--untracked-files=no")
    if tracked:
        raise NonQualified(
            "PQ checkout has uncommitted tracked changes; evidence would "
            "not name the executed tree",
            detail={"dirty": tracked.splitlines()[:10]})
    ignored_prefixes = ("__pycache__/", ".pytest_cache/")
    ignored_suffixes = (".pyc",)
    # Fleet-written receipts (pbrun_result.*.txt) are data the admission
    # layer drops beside the run; pytest never imports them.
    ignored_names = (".coverage", "coverage.xml")
    stray = []
    for line in _run("status", "--porcelain",
                     "--untracked-files=normal").splitlines():
        if not line.startswith("?? "):
            continue
        name = line[3:]
        if ("/__pycache__/" in f"/{name}" or name.startswith(ignored_prefixes)
                or name.endswith(ignored_suffixes)
                or Path(name).name in ignored_names
                or Path(name).name.startswith("pbrun_result.")):
            continue
        stray.append(name)
    if stray:
        raise NonQualified(
            "PQ checkout has uncommitted executable files; evidence would "
            "not name the executed tree",
            detail={"dirty": stray[:10]})
    return {"pb_rev": PB_CANDIDATE_REV, "pq_head": head}


def result_document(*, status: str, scenario: str, snapshots: dict,
                    reason: str = "", evidence: dict | None = None) -> dict:
    """One machine-readable scenario outcome for the report."""
    assert status in ("qualified", "nonqualified", "failed")
    return {"schema": "prismaquant.fleet_acceptance.result.v1",
            "scenario": scenario, "status": status, "reason": reason,
            "snapshots": snapshots, "evidence": dict(evidence or {})}


def nonqualified_doc(scenario: str, snapshots: dict, exc: NonQualified) -> dict:
    """The machine-output shape for a missing dependency or capability."""
    return result_document(status="nonqualified", scenario=scenario,
                           snapshots=snapshots, reason=exc.reason,
                           evidence={"detail": exc.detail})
