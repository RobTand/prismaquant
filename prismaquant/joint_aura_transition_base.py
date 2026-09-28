"""The byte-identical helpers the three closed joint source transitions share.

``joint_aura_retained_budget_transition``, ``joint_aura_run_transition`` and
``joint_aura_source_transition`` each carried their own copy of these spellings;
they now import them from here. The sealed source proofs pin each transition's
own contract and rewrite table, and those tables describe the campaign branch,
not ``main`` (see ``docs/design/joint_run_source_transition_2026-09-18.md``),
so moving a helper here changes no sealed bytes: every body below is verbatim
what the twins carried, and ``_actual_execution`` still lives per module
because it calls that module's own ``source_proof``.
"""
from __future__ import annotations

from pathlib import Path
import re
import subprocess

from .digests import file_sha256hex


_COMMIT = r"[0-9a-f]{40}|[0-9a-f]{64}"
_BYTES = ("producer_source_sha256", "reconstructed_source_sha256", "transition_module_sha256")


def _require(ok, message):
    if not ok:
        raise ValueError(f"joint source transition: {message}")


def _bound(record, label):
    _require(isinstance(record, dict) and set(record) == {"path", "sha256"},
             f"{label} requires independently bound path/SHA256")
    path = Path(record["path"])
    _require(path.is_file() and file_sha256hex(path) == record["sha256"], f"{label} bytes changed")
    return path


def _bytes_identity(execution):
    _require(isinstance(execution, dict) and all(
        isinstance(execution.get(key), str) and len(execution[key]) == 64 for key in _BYTES),
        "execution record lacks the package byte identity")
    return {key: execution[key] for key in _BYTES}


def checkout_head_commit(repo_root):
    """The sealed checkout's HEAD commit, read as files.

    The campaign image carries no git binary. A PrismaBuild checkout is
    detached at its snapshot commit (``.git/HEAD`` holds the id); a developer
    worktree may hold ``ref: refs/heads/...`` resolved through the loose ref,
    the common directory of a linked worktree, or ``packed-refs``.
    """
    root = Path(repo_root)
    git = root / ".git"
    if git.is_file():
        pointer = git.read_text().strip()
        _require(pointer.startswith("gitdir: "), "unreadable .git pointer")
        git = Path(pointer[len("gitdir: "):])
        if not git.is_absolute():
            git = root / git
    _require(git.is_dir() and (git / "HEAD").is_file(), "no sealed Git checkout at the package root")
    head = (git / "HEAD").read_text().strip()
    if re.fullmatch(_COMMIT, head):
        return head
    _require(head.startswith("ref: "), "unreadable HEAD")
    ref = head[len("ref: "):]
    common = git
    if (git / "commondir").is_file():
        common = (git / (git / "commondir").read_text().strip()).resolve()
    for candidate in (git / ref, common / ref):
        if candidate.is_file():
            value = candidate.read_text().strip()
            _require(re.fullmatch(_COMMIT, value) is not None, f"unreadable ref {ref}")
            return value
    packed = common / "packed-refs"
    if packed.is_file():
        for line in packed.read_text().splitlines():
            if not line or line[0] in "#^":
                continue
            value, _, name = line.partition(" ")
            if name == ref and re.fullmatch(_COMMIT, value):
                return value
    _require(False, f"HEAD ref {ref} is unresolved")


def _committed_package(repo_root):
    """Creating a receipt is a producer act: the package must be committed and clean."""
    try:
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "prismaquant"],
                                cwd=repo_root, check=True, capture_output=True, text=True, timeout=10).stdout
        parent = subprocess.run(["git", "rev-parse", "HEAD^"], cwd=repo_root, check=True,
                                capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        _require(False, f"creating a transition requires git and a committed checkout: {exc}")
    _require(not status.strip(), "producer package must be committed and clean")
    _require(re.fullmatch(_COMMIT, parent) is not None, "unreadable parent commit")
    # A PrismaBuild snapshot commit exists only in its bundle; its parent is
    # the branch commit a reader can find. Recorded, never compared.
    return {"git_parent_commit": parent}
