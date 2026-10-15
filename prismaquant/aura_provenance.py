"""Producer Git and package identity used by AURA and its checkpoints."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

def producer_git_commit() -> str | None:
    """Best-effort commit of the prismaquant tree this cost was computed by."""
    override = str(os.environ.get(
        "PRISMAQUANT_IDENTITY_GIT_COMMIT", ""
    )).strip().lower()
    if override:
        if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", override) is None:
            raise RuntimeError(
                "PRISMAQUANT_IDENTITY_GIT_COMMIT must be a full 40- or "
                "64-character hexadecimal commit id"
            )
        return override
    return producer_git_head_commit()


def producer_git_head_commit() -> str | None:
    """Best-effort ``git rev-parse HEAD`` of the package's checkout.

    The one owner of this recipe (PQ #1302): 5 s timeout, run from the package
    directory, ``None`` when git is absent, fails or prints nothing.
    """
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        return None


def checkpoint_git_commit() -> str:
    commit = producer_git_commit()
    if commit is None:
        raise RuntimeError("AURA checkpoint identity cannot resolve git commit")
    if os.environ.get("PRISMAQUANT_IDENTITY_GIT_COMMIT"):
        return commit
    repo_root = Path(__file__).resolve().parents[1]
    clean = subprocess.run(
        ["git", "diff", "--quiet", commit, "--", "prismaquant/aura_cost.py"],
        cwd=repo_root,
        check=False,
        timeout=10,
    )
    if clean.returncode != 0:
        refusal = RuntimeError(
            "AURA checkpoint git identity is not exact: aura_cost.py differs "
            f"from commit {commit}; commit it before checkpoint/resume"
        )
        if clean.returncode != 1:
            raise refusal
        # Only status one reports source drift. Git errors still refuse.
        from prismaquant.dev_mode import seal_check

        seal_check(
            "AURA checkpoint producer source", commit, "changed working tree",
            where="prismaquant/aura_cost.py", refusal=refusal,
        )
    return commit


def producer_package_sha256() -> str:
    """Bind AURA to the complete producer package, including dependencies."""
    from prismaquant.production_weight_cache import (
        _production_cache_source_sha256,
    )

    return _production_cache_source_sha256()
