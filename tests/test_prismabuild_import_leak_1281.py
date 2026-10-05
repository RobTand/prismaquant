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

import ast
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


def test_tests_do_not_import_prismabuild_during_collection():
    """Collection precedes the restorers: SDK imports must be test-owned."""
    tests_root = Path(__file__).parent
    imports = []
    for path in sorted(tests_root.rglob("*.py")):
        pending = list(ast.parse(path.read_text(encoding="utf-8")).body)
        while pending:
            node = pending.pop()
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                pending.extend(ast.iter_child_nodes(node))
                continue
            if any(name == "prismabuild" or name.startswith("prismabuild.")
                   for name in names):
                imports.append(f"{path.relative_to(tests_root)}:{node.lineno}")
    assert not imports, "collection-time PrismaBuild imports: " + ", ".join(imports)


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


def _source_refusal_representatives(tests_root):
    """Derive origin-refusing families and their pytest consumers, not a roster."""
    from collections import deque

    nodes, imports, fixtures, autouse = {}, {}, {}, {}
    def imported_names(tree):
        result = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result[alias.asname or alias.name.split('.')[0]] = alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    if alias.name != '*':
                        result[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        return result
    def normalized(name):
        return name.removeprefix('tests.')
    def dotted(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = dotted(node.value)
            return f"{parent}.{node.attr}" if parent else ''
        return ''
    def collect(body, module, path, classes=()):
        for node in body:
            if isinstance(node, ast.ClassDef):
                collect(node.body, module, path, (*classes, node.name))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = f"{module}.{'.'.join((*classes, node.name))}"
                nodes[key] = (node, module, path, classes)
                for decorator in node.decorator_list:
                    target = decorator.func if isinstance(decorator, ast.Call) else decorator
                    if dotted(target).endswith('fixture'):
                        name = node.name
                        if isinstance(decorator, ast.Call):
                            for keyword in decorator.keywords:
                                if keyword.arg == 'name' and isinstance(keyword.value, ast.Constant):
                                    name = keyword.value.value
                                if keyword.arg == 'autouse' and isinstance(keyword.value, ast.Constant) and keyword.value.value:
                                    autouse.setdefault(module, set()).add(key)
                        fixtures[(module, name)] = key
    for path in sorted(tests_root.rglob('*.py')):
        module = '.'.join(path.relative_to(tests_root).with_suffix('').parts)
        tree = ast.parse(path.read_text(encoding='utf-8'))
        imports[module] = imported_names(ast.Module(body=[node for node in tree.body
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))], type_ignores=[]))
        collect(tree.body, module, path)
    edges, roots = {}, set()
    for key, (node, module, path, classes) in nodes.items():
        aliases = imports[module] | imported_names(node)
        def resolve(name):
            first, _, suffix = name.partition('.')
            if first in ('self', 'cls') and classes:
                return f"{module}.{'.'.join(classes)}.{suffix}"
            if first in aliases:
                return normalized(aliases[first] + (f'.{suffix}' if suffix else ''))
            return f"{module}.{name}"
        calls = {resolve(dotted(item.func)) for item in ast.walk(node)
                 if isinstance(item, ast.Call) and dotted(item.func)}
        dependencies = calls & nodes.keys()
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            provider = fixtures.get((module, argument.arg)) or fixtures.get(('conftest', argument.arg))
            imported = normalized(aliases.get(argument.arg, ''))
            if imported in nodes and imported in fixtures.values():
                provider = imported
            if provider:
                dependencies.add(provider)
        dependencies |= autouse.get(module, set()) - {key}
        edges[key] = dependencies
        package_import = any(
            (isinstance(item, ast.Import) and any(alias.name == 'prismabuild' or alias.name.startswith('prismabuild.') for alias in item.names)) or
            (isinstance(item, ast.ImportFrom) and item.module and (item.module == 'prismabuild' or item.module.startswith('prismabuild.')))
            for item in ast.walk(node))
        file_reference = any((isinstance(item, ast.Attribute) and item.attr == '__file__') or
            (isinstance(item, ast.Call) and dotted(item.func) == 'getattr' and len(item.args) > 1 and isinstance(item.args[1], ast.Constant) and item.args[1].value == '__file__')
            for item in ast.walk(node))
        relative_check = any(isinstance(item, ast.Call) and isinstance(item.func, ast.Attribute) and item.func.attr == 'is_relative_to' for item in ast.walk(node))
        file_comparison = any(isinstance(item, ast.Compare) and any(isinstance(part, ast.Attribute) and part.attr == '__file__' for part in ast.walk(item)) for item in ast.walk(node))
        refusal = any(isinstance(item, ast.Assert) or (isinstance(item, ast.Call) and dotted(item.func) == 'pytest.fail') for item in ast.walk(node))
        if package_import and file_reference and (relative_check or file_comparison) and refusal:
            roots.add(key)
    tests = {key: record for key, record in nodes.items()
             if record[2].name.startswith('test_') and record[0].name.startswith('test_')}
    families, unreachable = {}, []
    for family in sorted(roots):
        candidates = []
        for key, record in tests.items():
            queue, seen = deque([(key, 0)]), set()
            while queue:
                current, distance = queue.popleft()
                if current in seen:
                    continue
                seen.add(current)
                if current == family:
                    node, module, path, classes = record
                    target = str(path.relative_to(tests_root.parent)) + '::' + '::'.join((*classes, node.name))
                    candidates.append((distance, module != nodes[family][1], node.lineno, target))
                    break
                queue.extend((dependency, distance + 1) for dependency in edges[current])
        if candidates:
            families[family] = min(candidates)[-1]
        else:
            unreachable.append(family)
    # These helpers are command entry points, not pytest-process consumers.
    # Unmapped roots in test modules fail closed instead of disappearing.
    assert not [name for name in unreachable if name.split('.')[0].startswith('test_')], unreachable
    assert families, 'origin-refusing source families must execute'
    return families, unreachable


def test_source_consumers_restore_the_public_plugin_import_graph():
    """Every discovered source family runs with the real public plugin loaded."""
    import os
    import subprocess

    require_prismabuild_sdk()
    root = Path(__file__).resolve().parents[1]
    families, command_helpers = _source_refusal_representatives(Path(__file__).parent)
    targets = list(dict.fromkeys(families.values()))
    print("source ownership families:", families, "command-only helpers:", command_helpers)
    program = (
        "import importlib, sys, pytest\n"
        "import prismabuild.core as core\n"
        "import prismabuild.pytest_test_bound as plugin\n"
        "package = sys.modules['prismabuild']\n"
        "before = {n: m for n, m in sys.modules.items() "
        "if n == 'prismabuild' or n.startswith('prismabuild.')}\n"
        "status = pytest.main(['-p', 'prismabuild.pytest_test_bound', "
        "'-p', 'no:xdist', '-p', 'no:cacheprovider', *sys.argv[1:]])\n"
        "assert status == 0, status\n"
        "after = {n: m for n, m in sys.modules.items() "
        "if n == 'prismabuild' or n.startswith('prismabuild.')}\n"
        "assert after.keys() == before.keys()\n"
        "assert all(after[n] is before[n] for n in before)\n"
        "assert importlib.import_module('prismabuild.core') is core\n"
        "assert package.core is core\n"
        "assert sys.modules['prismabuild.pytest_test_bound'] is plugin\n"
        "print('public plugin graph restored')\n"
    )
    env = {name: value for name, value in os.environ.items()
           if not name.startswith("PYTEST_") and name != "PQ_OWN_PROCESS_REPORT"}
    phase_bound = 120
    env["PRISMABUILD_TEST_TIMEOUT_S"] = str(phase_bound)
    done = subprocess.run(
        [sys.executable, "-c", program, *targets], cwd=root, env=env,
        capture_output=True, text=True,
        timeout=phase_bound * (3 * len(targets) + 1))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "public plugin graph restored" in done.stdout


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
