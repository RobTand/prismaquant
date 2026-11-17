"""Producer Git and package identity used by AURA and its checkpoints."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from prismaquant.digests import LengthFramedSourceSha256, bytes_sha256hex
from prismaquant.file_identity import _production_cache_identity_paths


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


#: The sources whose bytes an AURA checkpoint's commit certifies: the cost
#: passes, the journal (identity, manifest and unit codecs, resume admission),
#: the price step (unit restore and capture, joint row completion) and this
#: recipe. The whole-package digest below binds every other module.
CHECKPOINT_PRODUCER_SOURCES = (
    "prismaquant/aura_cost.py",
    "prismaquant/aura_checkpoint_journal.py",
    "prismaquant/joint_price_step.py",
    "prismaquant/aura_provenance.py",
)


def checkpoint_git_commit() -> str:
    commit = producer_git_commit()
    if commit is None:
        raise RuntimeError("AURA checkpoint identity cannot resolve git commit")
    if os.environ.get("PRISMAQUANT_IDENTITY_GIT_COMMIT"):
        return commit
    repo_root = Path(__file__).resolve().parents[1]
    clean = subprocess.run(
        ["git", "diff", "--name-only", "--exit-code", commit, "--",
         *CHECKPOINT_PRODUCER_SOURCES],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        text=True,
        check=False,
        timeout=10,
    )
    if clean.returncode != 0:
        drifted = (clean.stdout.split() if clean.returncode == 1 else []) or list(
            CHECKPOINT_PRODUCER_SOURCES)
        names = ", ".join(Path(path).name for path in drifted)
        refusal = RuntimeError(
            f"AURA checkpoint git identity is not exact: {names} "
            f"{'differs' if len(drifted) == 1 else 'differ'} from commit "
            f"{commit}; commit it before checkpoint/resume"
        )
        if clean.returncode != 1:
            raise refusal
        # Only status one reports source drift. Git errors still refuse.
        from prismaquant.dev_mode import seal_check

        seal_check(
            "AURA checkpoint producer source", commit, "changed working tree",
            where=", ".join(drifted), refusal=refusal,
        )
    return commit


def _production_cache_source_root(package_root: Path | None) -> Path:
    root = (
        Path(package_root)
        if package_root is not None
        else Path(__file__).resolve().parent
    )
    if not root.is_dir():
        raise RuntimeError(
            f"production source identity cannot read package root {root}"
        )
    return root


def producer_package_sha256(
    package_root: Path | None = None,
) -> str:
    """Hash every durable PrismaQuant package input used by the producer.

    A hand-maintained import list is not fail-closed: a newly introduced
    transitive renderer dependency could otherwise change bytes without
    changing a resumable pair's identity.  Hash the complete installed
    package tree instead, excluding only interpreter bytecode caches.  This
    also binds packaged lattice/codebook data and model-profile JSON.
    """
    return _production_cache_source_profile(package_root)[0]


def _production_cache_source_profile(
    package_root: Path | None = None,
) -> tuple[str, dict[str, str]]:
    """Aggregate digest and per-file digests from one read of the source tree.

    The aggregate is byte-identical to ``producer_package_sha256``'s
    framing; the per-file map is what a checkpoint manifest stores beside it
    so a later mismatch can name the first file whose bytes moved (#2218).
    One walk, one read per file, so the map and its aggregate always describe
    the same pass over the tree.
    """
    root = _production_cache_source_root(package_root)
    identity_paths = _production_cache_identity_paths(root)
    if not identity_paths:
        raise RuntimeError(
            f"production source identity found no files under {root}"
        )
    digest = LengthFramedSourceSha256()
    digests: dict[str, str] = {}
    for path in identity_paths:
        relative = path.relative_to(root).as_posix()
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(
                f"production source identity cannot read {relative}"
            ) from exc
        digest.update(relative, payload)
        digests[relative] = bytes_sha256hex(payload)
    return digest.hexdigest(), digests
