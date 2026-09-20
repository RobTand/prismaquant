"""Resolve published PB stdlib modules from one immutable generation.

The fleet publishes exactly one runtime generation behind
``/mnt/shared/prismabuild-fleet/repo``. This helper resolves that link
ONCE per admitted action to a full immutable root, derives the generation
identity from that same root's own name, inserts its ``src`` and
``tools/fleet`` ahead of ``sys.path``, and verifies every published module
actually imported comes from under that same root. A module loaded from
anywhere else fails loudly: an unresolvable or drifting PB is a failed
qualification, never a silent substitution.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

REPO_LINK = Path("/mnt/shared/prismabuild-fleet/repo")

#: Resolved once per process (one admitted action); never re-resolved.
_ROOT: Path | None = None


def _resolve_once() -> Path:
    """The single immutable root every PB import in this process comes from."""
    global _ROOT
    if _ROOT is None:
        root = REPO_LINK.resolve()
        assert root.is_dir(), f"published PB root missing under {REPO_LINK}"
        assert root.name != REPO_LINK.name, (
            f"PB repo link {REPO_LINK} does not name a generation")
        assert (root / "src" / "prismabuild" / "core.py").is_file(), (
            f"published PB stdlib missing under {root / 'src'}")
        assert (root / "tools" / "fleet" / "stage_move.py").is_file(), (
            f"published PB fleet tools missing under {root / 'tools' / 'fleet'}")
        _ROOT = root
    return _ROOT


def generation() -> str:
    """The immutable generation id, derived from the resolved root's name."""
    return _resolve_once().name


def require_paths() -> dict[str, str]:
    """Insert the resolved root's src/fleet tools; verify imports land there."""
    root = _resolve_once()
    src = root / "src"
    fleet = root / "tools" / "fleet"
    import sys
    for entry in (str(src), str(fleet)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    import prismabuild.core as core  # noqa: E402
    import stage_move  # noqa: E402
    for module in (core, stage_move):
        location = Path(getattr(module, "__file__", "")).resolve()
        assert location.is_relative_to(root), (
            f"{module.__name__} loaded from {location}, not the "
            f"resolved root {root}")
    info = {"generation": root.name, "root": str(root),
            "src": str(src), "fleet": str(fleet),
            "core_file": str(Path(core.__file__).resolve())}
    print(f"[pb-generation] {info['generation']} root={info['root']} "
          f"core={info['core_file']}", flush=True)
    return info


def runtime_version_digest() -> str:
    """Digest of the resolved root's own runtime version record, when filed."""
    root = _resolve_once()
    record = root / "RUNTIME_VERSION.json"
    if record.is_file():
        return hashlib.sha256(record.read_bytes()).hexdigest()
    return ""
