"""No test's ``prismabuild`` imports reach the next test (PQ #1281).

A test that imports PrismaBuild from a sealed generation tree put that
tree's ``src`` on ``sys.path`` and left ``prismabuild.*`` from it in
``sys.modules``. ``staged_lease.inject_installed_sdk_for_tests`` then bound
the generation's ``reader_lease`` as the installed SDK in every later test
on that worker, and the strict-reader fixtures refused
``lease-context-unavailable: no-claim-context``. These tests build their own
stand-in tree under ``tmp_path``: they read nothing from the fleet.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest

from fleet_sdk import (
    prismabuild_entries, prismabuild_imports_restored, require_prismabuild_sdk)


def _foreign_tree(tmp_path: Path) -> Path:
    """A ``src`` holding a ``prismabuild`` package, as a generation has.

    Its ``reader_lease`` carries every name the injection checks for, so
    only the check on where the module came from can refuse it.
    """

    from prismaquant.staged_lease import _REQUIRED_NAMES

    src = tmp_path / "generation" / "src"
    package = src / "prismabuild"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("FOREIGN = True\n")
    (package / "reader_lease.py").write_text("FOREIGN = True\n" + "".join(
        f"{name} = None\n" for name in _REQUIRED_NAMES))
    (package / "pool.py").write_text("FOREIGN = True\n")
    return src


def _import_from(src: Path):
    """Import ``prismabuild`` from ``src`` the way a sealed-tree helper does."""

    for name in prismabuild_entries():
        del sys.modules[name]
    sys.path.insert(0, str(src))
    importlib.invalidate_caches()
    package = importlib.import_module("prismabuild")
    importlib.import_module("prismabuild.pool")
    return package


def test_a_foreign_prismabuild_does_not_outlive_its_block(tmp_path):
    src = _foreign_tree(tmp_path)
    path_before = list(sys.path)
    entries_before = prismabuild_entries()

    with prismabuild_imports_restored():
        package = _import_from(src)
        assert package.FOREIGN
        assert sys.path[0] == str(src)

    assert sys.path == path_before
    after = prismabuild_entries()
    assert after.keys() == entries_before.keys()
    assert all(after[name] is entries_before[name] for name in after)
    for module in after.values():
        assert not getattr(module, "FOREIGN", False)


def test_every_test_and_module_runs_inside_the_restore(request):
    """``tests/conftest.py`` applies the restore to every test and module."""

    assert "_no_prismabuild_import_carried_between_tests" in request.fixturenames
    assert "_no_prismabuild_import_carried_between_modules" in request.fixturenames


def test_injection_refuses_a_prismabuild_it_did_not_install(tmp_path):
    """A shadowing ``prismabuild`` fails the injection by name, never binds."""

    require_prismabuild_sdk()
    from prismaquant import staged_lease

    src = _foreign_tree(tmp_path)
    # Put back, not cleared: a session-scoped injection elsewhere on this
    # worker must survive this test.
    saved = (staged_lease._INJECTED, staged_lease._INJECTED_MODULES_BEFORE)
    with prismabuild_imports_restored():
        try:
            _import_from(src)
            with pytest.raises(RuntimeError, match="PQ #1281") as refused:
                staged_lease.inject_installed_sdk_for_tests()
            foreign = (src / "prismabuild" / "reader_lease.py").resolve()
            assert str(foreign) in str(refused.value)
            assert staged_lease._INJECTED is saved[0]
        finally:
            staged_lease._INJECTED, staged_lease._INJECTED_MODULES_BEFORE = saved
