"""Where ``import prismabuild`` resolves, put back after a test (PQ #1281).

A helper module, not a test module, and standard library only: ``tests/conftest.py``
wraps every test and every test module in ``prismabuild_imports_restored``, so
whatever this module imports, every test depends on. It lived in
``tests/fleet_sdk.py`` until #3215, where it brought ``prismaquant.staged_lease``
(and, through it, most of ``prismaquant``) into every test's inputs.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
import sys


def prismabuild_entries() -> dict:
    """Every ``prismabuild`` entry ``sys.modules`` holds, name to module.

    The same entries ``staged_lease._prismabuild_modules`` names;
    ``tests/test_conftest_reach.py`` pins that the two agree.
    """

    return {name: sys.modules[name] for name in list(sys.modules)
            if name == "prismabuild" or name.startswith("prismabuild.")}


def _hosts_prismabuild(entry: str) -> bool:
    try:
        return (Path(entry or ".") / "prismabuild").is_dir()
    except OSError:
        return False


@contextlib.contextmanager
def prismabuild_imports_restored(*, source_root: Path | None = None,
                               detach: bool = False):
    """Leave ``import prismabuild`` resolving where it did before (PQ #1281).

    Several tests import PrismaBuild from a sealed generation tree: they
    put its ``src`` on ``sys.path`` and import ``prismabuild.*`` from it
    (``test_quantum_executable_readset._pb``, ``staged_lease._sdk_from_tree``
    under the PB-injected ``PRISMABUILD_READER_HELPER_ROOT``). Nothing took
    either back out, so every later test in that process imported the
    generation's ``prismabuild`` in place of the installed one, and
    ``staged_lease.inject_installed_sdk_for_tests`` bound it as the
    installed SDK. The generation reads its queue from PB's own
    ``PRISMABUILD_QUEUE_ROOT``, so each strict-reader fixture's claim was
    looked for in the live fleet queue: ``lease-context-unavailable:
    no-claim-context``. The reverse order fails too: an installed module
    left behind refuses a later sealed-tree resolution as divergent
    (PQ #963, PQ #1032).

    On exit, the ``prismabuild`` entries of ``sys.modules`` are put back
    exactly: added ones are dropped, replaced or removed ones restored, and
    a surviving parent package's attribute for each follows its entry. Each
    ``sys.path`` entry added inside the block that hosts a ``prismabuild``
    package is dropped; every other path change stays as the test left it.
    Restoring is silent and unconditional, like the other restore fixtures
    in ``tests/conftest.py``: importing from a sealed tree is legitimate,
    only its escape from the test is the defect.

    ``detach`` gives a fixture ownership of an initially unbound graph. It
    removes pre-existing package entries without choosing a source or adding
    a prerequisite. Resolvers still authenticate and refuse origins themselves.
    The previous graph and parent edges return on exit; without ``source_root``
    unrelated path changes retain the preservation-only behavior above.

    ``source_root`` is an explicit fixture-owned binding, authenticated by
    its caller before entry. It detaches the canonical PB graph and sibling
    fleet tools for this scope, then restores their identities and path order
    exactly. Callers must finish using its objects and join their work before
    leaving; production resolution and its divergence refusals are unchanged.
    """

    from types import ModuleType

    saved_path = list(sys.path)
    tool_names = set()
    if source_root is not None:
        source_root = Path(source_root).resolve(strict=True)
        fleet = source_root / "tools" / "fleet"
        tool_names = {path.stem for path in fleet.glob("*.py")}
    saved_modules = prismabuild_entries()
    saved_modules.update({name: sys.modules[name] for name in tool_names
                          if name in sys.modules})
    missing = object()
    saved_attributes = {}
    for name, parent in saved_modules.items():
        if parent is None or not name.startswith("prismabuild"):
            continue
        for leaf, value in vars(parent).items():
            if isinstance(value, ModuleType) and value.__name__ == f"{name}.{leaf}":
                saved_attributes[f"{name}.{leaf}"] = (parent, leaf, value)
    for name in saved_modules:
        parent_name, _, leaf = name.rpartition(".")
        parent = saved_modules.get(parent_name)
        if parent is not None:
            saved_attributes[name] = (parent, leaf, getattr(parent, leaf, missing))
    if source_root is not None or detach:
        for name in saved_modules:
            sys.modules.pop(name, None)
    if source_root is not None:
        sys.path[:0] = [str(source_root / "src"), str(fleet)]
    try:
        yield
    finally:
        if source_root is not None:
            sys.path[:] = saved_path
        else:
            added_paths = {entry for entry in sys.path if entry not in saved_path}
            dropped = {entry for entry in added_paths if _hosts_prismabuild(entry)}
            if dropped:
                sys.path[:] = [entry for entry in sys.path if entry not in dropped]
        current = prismabuild_entries()
        current.update({name: sys.modules[name] for name in tool_names
                        if name in sys.modules})
        for name in set(current) | set(saved_modules):
            if name in saved_modules:
                sys.modules[name] = saved_modules[name]
            else:
                sys.modules.pop(name, None)
        # A loader or injection can already have removed an entry while its
        # parent still holds the replacement. Restore those orphan edges too.
        for name, parent in saved_modules.items():
            if parent is None or not name.startswith("prismabuild"):
                continue
            for leaf, value in tuple(vars(parent).items()):
                edge = f"{name}.{leaf}"
                if (isinstance(value, ModuleType) and value.__name__ == edge
                        and edge not in saved_attributes):
                    delattr(parent, leaf)
        for parent, leaf, value in saved_attributes.values():
            if value is missing:
                if hasattr(parent, leaf):
                    delattr(parent, leaf)
            else:
                setattr(parent, leaf, value)
