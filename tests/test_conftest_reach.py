"""``tests/conftest.py`` reaches no ``prismaquant`` or ``tools`` module (#3215).

Bazel, and the gate's selector, run a test again when anything in its static
inputs changes, and every test's inputs include ``tests/conftest.py`` and every
module it imports, at module level or inside a fixture. Its autouse fixtures
imported ``prismaquant`` modules only to reset or snapshot them, and each of
those enters the large ``prismaquant`` import cycle, so every test depended on
about 270 source modules whatever it imported itself.

Conftest now registers each reset with ``_when_loaded`` instead: a module
already loaded is reset at once, and one the test loads later is reset as its
body finishes, through a loader hook (PrismaBuild #2585's pattern). Fixtures
only some tests use live in helper modules those tests import. These tests
pin the cut, and that the resets still win when a module loads late.
"""
from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
SOURCE_ROOTS = ("prismaquant", "tools")


def _names_imported(path: Path) -> set[str]:
    """Every module an import statement in ``path`` names, anywhere in the file."""

    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _test_helper(name: str) -> Path | None:
    """The ``tests/`` helper a name imports, by bare name or as ``tests.<name>``."""

    stem = name.removeprefix("tests.")
    candidate = TESTS / (stem.replace(".", "/") + ".py")
    return candidate if candidate.is_file() else None


def _source_module(name: str) -> bool:
    first = name.split(".", 1)[0]
    if first in SOURCE_ROOTS:
        return True
    # A bare name that is a tools/ module (several tests put tools/ on sys.path).
    return any(True for _ in (ROOT / "tools").rglob(f"{first}.py"))


def test_conftest_and_the_helpers_it_imports_name_no_source_module():
    """Static: conftest, and each ``tests/`` helper it imports, transitively."""

    found, seen, pending = [], set(), [(TESTS / "conftest.py", ())]
    while pending:
        path, chain = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        for name in sorted(_names_imported(path)):
            helper = _test_helper(name)
            if helper is not None and helper != path:
                pending.append((helper, (*chain, helper.name)))
            elif helper is None and _source_module(name):
                found.append(" -> ".join(("conftest.py", *chain, name)))
    assert not found, "conftest's import chain reaches source modules:\n" + "\n".join(found)


def _child_pytest(tmp_path: Path, files: dict[str, str]) -> subprocess.CompletedProcess:
    """A real pytest over ``files``, loading this tree's conftest as its plugin."""

    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "pytest.ini").write_text("[pytest]\n")
    (suite / "conftest.py").write_text('pytest_plugins = ("tests.conftest",)\n')
    for name, text in files.items():
        (suite / name).write_text(textwrap.dedent(text))
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("PYTEST_") and key not in (
               "PQ_OWN_PROCESS_REPORT", "PRISMAQUANT_PROD_ACT_SCALES")}
    env.update({"PYTHONPATH": os.pathsep.join([str(ROOT), str(TESTS), env.get("PYTHONPATH", "")]),
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PQ_CONFTEST_REACH_ROOT": str(ROOT),
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "-c", str(suite / "pytest.ini"), "--rootdir", str(suite),
         "--basetemp", str(tmp_path / "base"), *sorted(files)],
        cwd=suite, env=env, capture_output=True, text=True, timeout=300)


_NOTHING_FROM_SOURCE = '''
import os
import sys
from pathlib import Path

ROOT = Path(os.environ["PQ_CONFTEST_REACH_ROOT"])


def test_no_source_module_is_loaded():
    loaded = sorted(
        name for name, module in list(sys.modules.items())
        if getattr(module, "__file__", None) and any(
            Path(module.__file__).resolve().is_relative_to(ROOT / top)
            for top in ("prismaquant", "tools")))
    assert not loaded, loaded
'''


def test_running_tests_under_conftest_loads_no_source_module(tmp_path):
    """Runtime: after conftest's hooks and autouse fixtures, including a module's teardown.

    The second file's test runs after the first module's teardown, so an
    import in a module-scoped teardown shows there.
    """

    done = _child_pytest(tmp_path, {"test_first.py": _NOTHING_FROM_SOURCE,
                                    "test_second.py": _NOTHING_FROM_SOURCE})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "2 passed" in done.stdout, done.stdout + done.stderr


_LATE_LOADS = '''
import os
import sys

POLICY = "prismaquant.staged_tier_policy"
SERVED = "prismaquant.nvfp4_activation_contract"
REGISTRY = "prismaquant.model_profiles.registry"
VENDORED = "prismaquant.vendored"
PRISTINE = {}


def test_a_leaks_state_into_modules_it_loads_itself():
    assert not [name for name in (POLICY, SERVED, REGISTRY, VENDORED) if name in sys.modules]
    from prismaquant import nvfp4_activation_contract as served
    from prismaquant import staged_tier_policy as policy
    import prismaquant.vendored as vendored
    from prismaquant.model_profiles import registry

    PRISTINE.update(registered=list(registry._REGISTERED),
                    generation=registry._REGISTRY_GENERATION,
                    order=registry._DETECTION_ORDER_CACHE)
    policy.activate_staged_tier_policy("ram")
    served._ACTIVE_SERVED_QUANTIZER = object()
    served._RESOLVED_SERVED_QUANTIZER = object()
    registry._REGISTERED.append(object)
    registry._REGISTRY_GENERATION += 1
    registry._DETECTION_ORDER_CACHE = (registry._REGISTRY_GENERATION, ())
    vendored.OVERRIDE_ERRORS["qwen3"] = "left behind"
    vendored._QWEN3_REGISTERED = True
    os.environ["PRISMAQUANT_PROD_ACT_SCALES"] = "0"


def test_b_starts_clean():
    from prismaquant import nvfp4_activation_contract as served
    from prismaquant import staged_tier_policy as policy
    import prismaquant.vendored as vendored
    from prismaquant.model_profiles import registry

    assert policy._ACTIVE is None and policy._OWNER is None and policy._STACK == []
    assert served._ACTIVE_SERVED_QUANTIZER is None
    assert served._RESOLVED_SERVED_QUANTIZER is None
    assert registry._REGISTERED == PRISTINE["registered"]
    assert registry._REGISTRY_GENERATION == PRISTINE["generation"]
    assert registry._DETECTION_ORDER_CACHE == PRISTINE["order"]
    assert vendored.OVERRIDE_ERRORS == {}
    assert vendored._QWEN3_REGISTERED is False
    assert "PRISMAQUANT_PROD_ACT_SCALES" not in os.environ
'''


def test_resets_still_win_for_modules_a_test_loads_itself(tmp_path):
    """A module first loaded inside a test is put back after it, as before the cut."""

    done = _child_pytest(tmp_path, {"test_late_loads.py": _LATE_LOADS})
    assert done.returncode == 0, done.stdout + done.stderr
    assert "2 passed" in done.stdout, done.stdout + done.stderr


def _probe(tmp_path: Path, name: str) -> None:
    (tmp_path / f"{name}.py").write_text("VALUE = 'module'\nSEEN_AT_LOAD = VALUE\n")


def test_when_loaded_patches_a_loaded_module_at_once_and_a_later_one_on_load(tmp_path):
    import conftest

    _probe(tmp_path, "pq_conftest_reach_early")
    _probe(tmp_path, "pq_conftest_reach_late")
    sys.path.insert(0, str(tmp_path))
    importlib.invalidate_caches()
    try:
        early = importlib.import_module("pq_conftest_reach_early")
        order = []
        with pytest.MonkeyPatch.context() as patcher:
            for label in ("first", "second"):
                for name in ("pq_conftest_reach_early", "pq_conftest_reach_late"):
                    conftest._when_loaded(patcher, name, lambda module, label=label: (
                        order.append((module.__name__, label)),
                        setattr(module, "VALUE", label)))
            assert early.VALUE == "second"
            late = importlib.import_module("pq_conftest_reach_late")
            # Applied after the module's own body ran, in registration order.
            assert late.SEEN_AT_LOAD == "module"
            assert late.VALUE == "second"
            assert order == [("pq_conftest_reach_early", "first"),
                             ("pq_conftest_reach_early", "second"),
                             ("pq_conftest_reach_late", "first"),
                             ("pq_conftest_reach_late", "second")]
        # The registrations end with their test.
        assert "pq_conftest_reach_late" not in conftest._LOAD_PATCHES
        assert "pq_conftest_reach_early" not in conftest._LOAD_PATCHES
        del sys.modules["pq_conftest_reach_late"]
        assert importlib.import_module("pq_conftest_reach_late").VALUE == "module"
    finally:
        sys.path.remove(str(tmp_path))
        for name in ("pq_conftest_reach_early", "pq_conftest_reach_late"):
            sys.modules.pop(name, None)


#: Names that left ``tests/conftest.py``, and their
#: one home now. A name kept in both places would let a test patch the old
#: copy and silently miss the code that reads the new one (PrismaBuild #2582's
#: lesson); with the old name gone, such a patch raises ``AttributeError``.
MOVED = {
    "conftest": {
        "synthetic_graph_lane": "graph_lane_fixtures",
        "legacy_capture_mechanism": "legacy_capture_fixtures",
        "_installed_contract": "lane_table_fixtures",
        "set_cell_census": "lane_table_fixtures",
        "down_convert_lane_table": "lane_table_fixtures",
        "lane_cells_on_one_image": "lane_table_fixtures",
        "project_lane_cells_onto_structures": "lane_table_fixtures",
        "legacy_v4_contract": "lane_table_fixtures",
        "legacy_v5_contract": "lane_table_fixtures",
    },
}


@pytest.mark.parametrize("old", sorted(MOVED))
def test_moved_names_have_one_home(old):
    module = importlib.import_module(old)
    for name, home in MOVED[old].items():
        assert not hasattr(module, name), f"{old}.{name} still exists beside {home}.{name}"
        assert hasattr(importlib.import_module(home), name), f"{home}.{name} is missing"


def test_conftest_restores_the_activation_scale_variable_the_joint_lease_reads():
    import conftest

    from prismaquant.tessera_joint_aura import ACTIVATION_SCALE_ENV

    assert conftest.ACTIVATION_SCALE_ENV == ACTIVATION_SCALE_ENV
