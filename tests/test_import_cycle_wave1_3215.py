"""Five lower modules no longer import back into higher ones (PQ #3215, wave 1).

A function-local import is still an import: when the function runs, the
importer loads the target and everything the target loads. 183 ``prismaquant``
modules formed one strongly connected component, held together mostly by such
imports in lower modules, so a test that reached any of them depended on all
of them. Each pair below was one of those back imports. The lower module now
owns what it needed, or reads it from a lower module, and the higher module
re-exports the moved names, so old import paths still work.

The graph is static (``ast``) and runs no package code. It has one edge for
every ``import`` and ``from ... import`` statement at any depth in a module,
except under ``if TYPE_CHECKING:``. It also has edges for a literal
``import_module`` or ``__import__`` name, for a name in the package's
``_LAZY_EXPORTS`` table, and for each parent package's ``__init__``, which
Python runs first.
"""
from __future__ import annotations

import pytest

from import_graph_support import _lazy_exports, _modules, _targets

# (lower module, higher module it must not reach through any chain of imports)
SEPARATED = [
    ("prismaquant.joint_adjoint_band", "prismaquant.joint_cost_stage_a"),
    ("prismaquant.tessera_joint_allocation", "prismaquant.joint_quanta_join"),
    ("prismaquant.staged_whole_file", "prismaquant.stage_inputs"),
    ("prismaquant.cost_stage_checkpoint", "prismaquant.production_weight_cache"),
    ("prismaquant.visual_linears", "prismaquant.allocator"),
]


@pytest.fixture(scope="module")
def graph() -> dict[str, set[str]]:
    modules = _modules()
    lazy = _lazy_exports(modules)
    return {module: _targets(module, path, modules, lazy) for module, path in modules.items()}


def _reach(graph: dict[str, set[str]], start: str) -> set[str]:
    seen, stack = {start}, [start]
    while stack:
        for target in graph[stack.pop()]:
            if target not in seen:
                seen.add(target)
                stack.append(target)
    return seen


@pytest.mark.parametrize("lower,higher", SEPARATED)
def test_the_lower_module_never_loads_the_higher_one(graph, lower, higher):
    assert lower in graph and higher in graph
    assert higher not in _reach(graph, lower)


def test_the_graph_counts_function_local_and_package_imports(graph):
    """Positive controls: imports this wave keeps, of each kind the oracle must see."""
    # ``producer_package_sha256`` imports the cache inside the function.
    assert "prismaquant.production_weight_cache" in graph["prismaquant.producer_package"]
    # ``detect_profile`` imports ``..staged_whole_file`` inside the function.
    assert "prismaquant.staged_whole_file" in graph["prismaquant.model_profiles.registry"]
    # A submodule runs its package's ``__init__`` first.
    assert "prismaquant.model_profiles" in graph["prismaquant.model_profiles.registry"]


def test_the_visual_roster_is_a_leaf_the_profiles_read_directly(graph):
    assert "prismaquant.visual_linears" in graph
    assert _reach(graph, "prismaquant.visual_linears") == {"prismaquant", "prismaquant.visual_linears"}
    assert "prismaquant.visual_linears" in graph["prismaquant.model_profiles.base"]
    assert "prismaquant.allocator" not in graph["prismaquant.model_profiles.base"]
