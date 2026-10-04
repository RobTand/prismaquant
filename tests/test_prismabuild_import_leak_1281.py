"""No test's ``prismabuild`` imports reach the next test (PQ #1281).

A test that imports PrismaBuild from a sealed generation tree put that
tree's ``src`` on ``sys.path`` and left ``prismabuild.*`` from it in
``sys.modules``. ``staged_lease.inject_installed_sdk_for_tests`` then bound
the generation's ``reader_lease`` as the installed SDK in every later test
on that worker, and the strict-reader fixtures refused
``lease-context-unavailable: no-claim-context``. Parent-edge controls build
their own stand-in tree under ``tmp_path``. The source-scope controls also
exercise the existing authenticated shared PB pin and reviewed installed SDK.
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

    Its ``client`` (PB #1254) carries every name and the SDK version the
    injection checks for, so only the check on where the module came from
    can refuse it.
    """

    from prismaquant.staged_lease import PB_CLIENT_SDK_VERSION, _REQUIRED_NAMES

    src = tmp_path / "generation" / "src"
    package = src / "prismabuild"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("FOREIGN = True\n")
    (package / "reader_lease.py").write_text("FOREIGN = True\n" + "".join(
        f"{name} = None\n" for name in _REQUIRED_NAMES))
    (package / "client.py").write_text(
        f"FOREIGN = True\nSDK_VERSION = {PB_CLIENT_SDK_VERSION}\n" + "".join(
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
            foreign = (src / "prismabuild" / "client.py").resolve()
            assert str(foreign) in str(refused.value)
            assert staged_lease._INJECTED is saved[0]
        finally:
            staged_lease._INJECTED, staged_lease._INJECTED_MODULES_BEFORE = saved


@pytest.mark.parametrize("change", ["parent-only", "removed", "replaced"])
def test_restore_recovers_parent_edges_even_without_a_live_replacement(tmp_path, change):
    from types import ModuleType

    src = _foreign_tree(tmp_path)
    with prismabuild_imports_restored():
        package = _import_from(src)
        original = sys.modules["prismabuild.pool"]
        with prismabuild_imports_restored():
            replacement = ModuleType("prismabuild.pool")
            package.pool = replacement
            if change == "removed":
                sys.modules.pop("prismabuild.pool")
            elif change == "replaced":
                sys.modules["prismabuild.pool"] = replacement
        assert sys.modules["prismabuild.pool"] is original
        assert package.pool is original


def test_restore_drops_an_added_parent_edge_after_its_entry_was_removed(tmp_path):
    from types import ModuleType

    src = _foreign_tree(tmp_path)
    with prismabuild_imports_restored():
        package = _import_from(src)
        with prismabuild_imports_restored():
            package.temporary = ModuleType("prismabuild.temporary")
            sys.modules["prismabuild.temporary"] = package.temporary
            sys.modules.pop("prismabuild.temporary")
        assert "prismabuild.temporary" not in sys.modules
        assert not hasattr(package, "temporary")


def test_authenticated_source_scope_restores_installed_identity_and_tools(
        installed_client_sdk, monkeypatch):
    from types import ModuleType
    from threading import Thread
    import fullstack_pb_generation as published
    from prismaquant import staged_lease

    old_tool = ModuleType("stage_move")
    monkeypatch.setitem(sys.modules, "stage_move", old_tool)
    before = prismabuild_entries()
    before_path = list(sys.path)
    for _ in range(2):
        with published.source_bound(), published.reader_sdk_bound():
            info = published.require_paths()
            core = importlib.import_module("prismabuild.core")
            tool = importlib.import_module("stage_move")
            sdk = staged_lease.client_sdk()
            assert sdk is not installed_client_sdk
            assert tool is not old_tool and tool.pb is core
            assert sdk.canonical_sha256 is core.canonical_sha256
            for module in (core, tool, sdk):
                assert Path(module.__file__).resolve().is_relative_to(Path(info["root"]))
            observed = []
            child = Thread(target=lambda: observed.append(core.canonical_sha256({"scope": "owned"})))
            child.start()
            child.join()  # No source-bound object is used after graph restoration.
            assert observed == [core.canonical_sha256({"scope": "owned"})]
        assert sys.path == before_path
        after = prismabuild_entries()
        assert after.keys() == before.keys()
        assert all(after[name] is before[name] for name in before)
        assert sys.modules["stage_move"] is old_tool
        assert staged_lease.client_sdk() is installed_client_sdk


def test_source_scope_authenticates_bytes_before_detaching_modules(
        tmp_path, monkeypatch, installed_client_sdk):
    import json
    import fullstack_pb_generation as published

    pin = json.loads(published.PIN_PATH.read_text())
    pin["files"]["src/prismabuild/core.py"] = "0" * 64
    path = tmp_path / "wrong-source-pin.json"
    path.write_text(json.dumps(pin))
    monkeypatch.setattr(published, "PIN_PATH", path)
    monkeypatch.setattr(published, "_ROOT", None)
    before = prismabuild_entries()
    before_path = list(sys.path)
    with pytest.raises(AssertionError, match="src/prismabuild/core.py"):
        with published.source_bound():
            pytest.fail("a mismatching source pin cannot enter the context")
    assert sys.path == before_path
    after = prismabuild_entries()
    assert after.keys() == before.keys()
    assert all(after[name] is before[name] for name in before)


def test_source_scope_keeps_production_divergent_origin_refusal(
        tmp_path, monkeypatch, installed_client_sdk):
    from types import ModuleType
    import fullstack_pb_generation as published
    from prismaquant import staged_lease

    with published.source_bound(), published.reader_sdk_bound():
        staged_lease.client_sdk()
        foreign = ModuleType("prismabuild.foreign")
        foreign.__file__ = str(tmp_path / "foreign.py")
        with monkeypatch.context() as owned:
            owned.setitem(sys.modules, foreign.__name__, foreign)
            with pytest.raises(staged_lease.LeaseRefused, match="lease-helper-divergent"):
                staged_lease.client_sdk()
    assert staged_lease.client_sdk() is installed_client_sdk
