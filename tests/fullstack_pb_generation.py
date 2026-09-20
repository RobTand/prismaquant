"""Resolve published PB stdlib modules from their immutable generation.

The fleet publishes exactly one runtime generation behind
``/mnt/shared/prismabuild-fleet/repo``. This helper resolves that link
once per admitted action, records the generation id for receipts, and
imports real PB modules (stdlib ``src`` plus ``tools/fleet`` movers) from
it — never a mutable per-host checkout. Missing link or unresolvable
generation fails loudly: an unresolvable PB is a failed qualification,
never a silent substitution.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_LINK = Path("/mnt/shared/prismabuild-fleet/repo")

#: Resolved once per process (one admitted action); never re-resolved.
_GENERATION: str | None = None


def generation() -> str:
    """The immutable generation id, e.g. ``0467e9e2316c-…-…``."""
    global _GENERATION
    if _GENERATION is None:
        target = os.readlink(REPO_LINK)
        _GENERATION = Path(target).name
        assert _GENERATION and _GENERATION != REPO_LINK.name, (
            f"PB repo link {REPO_LINK} does not name a generation")
    return _GENERATION


def require_paths() -> dict[str, str]:
    """Insert published PB src and fleet tools; return what was recorded."""
    root = REPO_LINK.resolve()
    src = root / "src"
    fleet = root / "tools" / "fleet"
    assert (src / "prismabuild" / "core.py").is_file(), (
        f"published PB stdlib missing under {src}")
    assert (fleet / "stage_move.py").is_file(), (
        f"published PB fleet tools missing under {fleet}")
    import sys
    for entry in (str(src), str(fleet)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    return {"generation": generation(), "src": str(src), "fleet": str(fleet)}
