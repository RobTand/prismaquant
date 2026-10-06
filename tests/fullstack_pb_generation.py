"""Resolve PB stdlib and fleet tools from the existing shared source pin.

Connected CPU fixtures exercise the same reviewed SDK source as their PQ
reader, not whichever older generation currently serves the fleet. The
shared ``pb_runtime_generation_pin.json`` owns the root and file digests.
Resolve it once per admitted action, then verify every imported module is
under that one root. The root may be staged: a passing source-bound fixture
never establishes live deployment, activation or fleet qualification.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from fleet_sdk import require_prismabuild_sdk

PIN_PATH = Path(__file__).with_name("pb_runtime_generation_pin.json")

#: Resolved once per process (one admitted action); never re-resolved.
_ROOT: Path | None = None


def _resolve_once() -> Path:
    """The single immutable root every PB import in this process comes from."""
    require_prismabuild_sdk()
    global _ROOT
    if _ROOT is None:
        pin = json.loads(PIN_PATH.read_text())
        root = Path(pin["bundle_root"]).resolve()
        assert root.is_dir(), f"pinned PB root missing: {root}"
        assert root.name == pin["runtime_generation"], (
            f"PB root {root} differs from its declared generation")
        for name, digest in pin["files"].items():
            assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
        assert (root / "src" / "prismabuild" / "core.py").is_file(), (
            f"published PB stdlib missing under {root / 'src'}")
        assert (root / "tools" / "fleet" / "stage_move.py").is_file(), (
            f"published PB fleet tools missing under {root / 'tools' / 'fleet'}")
        _ROOT = root
    return _ROOT


@contextmanager
def source_bound():
    """Own a fresh canonical import graph for the authenticated source pin.

    Fixtures construct all PB objects inside this scope and finish their work
    before it exits. The installed SDK graph is restored, not substituted or
    accepted as this source tree.
    """
    from fleet_sdk import prismabuild_imports_restored

    root = _resolve_once()  # Authenticate every declared byte before detaching.
    with prismabuild_imports_restored(source_root=root):
        yield


@contextmanager
def reader_sdk_bound():
    """Bind PQ's existing explicit SDK owner to this fixture's one source root."""
    from prismaquant.staged_lease import set_lease_helper_root
    set_lease_helper_root(_resolve_once())
    try:
        yield
    finally:
        set_lease_helper_root(None)


def generation() -> str:
    """The immutable generation id, derived from the resolved root's name."""
    return _resolve_once().name


def require_paths() -> dict[str, str]:
    """Insert the pinned root's src/fleet tools; verify imports land there."""
    require_prismabuild_sdk()
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
