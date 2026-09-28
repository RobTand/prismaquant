"""Freeze where PrismaQuant core reaches into the Tessera lane (#1534).

PrismaQuant may use Tessera, but the use belongs in the lane: the
``tessera_*`` modules, the lane adapters and the campaign modules. Core
should reach a lane through declared seams, as GGUF does (``lane_spec``,
``serving_profile_specs``, the format registry,
``WEIGHTED_RENDER_FAMILIES``). This test is the mechanical half of that
rule, step 0 of the decoupling plan in the 2026-09-27 coupling inventory.

**Rule 1, core.** Over the explicit core set ``CORE`` (``model_profiles/**``
included), it parses each module with ``ast`` and finds:

- imports of ``tessera...``, ``.tessera_*`` (any relative depth) and
  ``prismaquant.tessera_*``, at any depth, including
  ``from . import tessera_x``;
- calls to ``is_tessera_format_name``;
- ``str``/``bytes`` constants outside docstrings matching
  ``^tessera\\.[a-z0-9_.]+\\.v\\d+$`` or starting ``TESSERA_``.

**Rule 2, private helpers.** Over every ``prismaquant/`` module that is not
itself in the lane (no ``tessera_*`` path component), it finds
``from .tessera_x import _private`` and its absolute spelling, one entry
per private name.

The ``tessera_*``, ``joint_*``, ``glm_*``, ``stage_*``, ``native_*`` and
``quality_prefill_*`` families are outside the rule-1 fence.

Each finding is one line of an allowlist under ``tests/boundary_allowlists/``
(``tessera_core.txt``, ``tessera_private.txt``), keyed by path and module,
call or constant, never by line number. The test requires exact multiset
equality:

- a new reference, or one more occurrence of an allowlisted one, fails;
- a removed reference fails until its line is deleted in the same change.

The allowlists only shrink; CI refuses a pull request that adds a line.
Step 6 (lane hooks in core) empties ``tessera_core.txt`` apart from the
seam-conformant ``pipeline.py`` settings; step 7 (generic helpers out of the
lane modules) empties ``tessera_private.txt``.
"""
from __future__ import annotations

import ast
import fnmatch
import re
from collections import Counter

from tests.boundary_gate import assert_equal, code_strings, escape, python_files

CORE = (
    "allocator*", "aura_cost", "cost_*", "prepriced_cost", "pipeline",
    "format_registry", "serving_profiles", "shipcard*", "production_weight_cache",
    "perturbed_x_cache", "weight_session", "select_validated_frontier",
    "layer_config", "autoscale", "unit_topology_restamp", "lane_spec",
    "kl_measurement", "export_native_compressed", "validate_*",
)
_LITERAL = re.compile(r"^tessera\.[a-z0-9_.]+\.v\d+$")
_PREDICATE = "is_tessera_format_name"


def is_core(rel: str) -> bool:
    if rel.startswith("prismaquant/model_profiles/"):
        return True
    top, _, name = rel.partition("/")
    if top != "prismaquant" or "/" in name:
        return False
    return any(fnmatch.fnmatchcase(name[:-3], pattern) for pattern in CORE)


def is_lane(rel: str) -> bool:
    return any(part.startswith("tessera_") for part in rel.split("/")[1:])


def _imports(node: ast.AST) -> list[tuple[str, list[str]]]:
    """``(module, names)`` for an import, spelled as written."""
    if isinstance(node, ast.Import):
        return [(alias.name, []) for alias in node.names]
    if not isinstance(node, ast.ImportFrom):
        return []
    dots = "." * node.level
    if node.module is None or node.module == "prismaquant":
        # ``from . import tessera_x`` or ``from prismaquant import tessera_x``
        # import the module itself.
        base = dots if node.module is None else "prismaquant."
        return [(base + alias.name, []) for alias in node.names]
    return [(dots + node.module, [alias.name for alias in node.names])]


def _lane_module(module: str) -> bool:
    if module.startswith("."):
        return module.lstrip(".").startswith("tessera_")
    return (module.split(".")[0] == "tessera"
            or module.startswith("prismaquant.tessera_"))


def _called(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def core_findings(source: str) -> list[tuple[str, str]]:
    tree = ast.parse(source)
    found: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        found += [("import", module) for module, _ in _imports(node)
                  if _lane_module(module)]
        if isinstance(node, ast.Call) and _called(node) == _PREDICATE:
            found.append(("call", _PREDICATE))
    found += [("literal", escape(text)) for _, text in code_strings(tree)
              if _LITERAL.match(text) or text.startswith("TESSERA_")]
    return found


def private_findings(source: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for node in ast.walk(ast.parse(source)):
        for module, names in _imports(node):
            if not _lane_module(module) or module.split(".")[0] == "tessera":
                continue
            found += [("private-import", f"{module}.{name}") for name in names
                      if name.startswith("_") and not name.startswith("__")]
    return found


def scan() -> tuple[Counter, Counter]:
    core: Counter = Counter()
    private: Counter = Counter()
    for rel, source in python_files("prismaquant"):
        if is_core(rel):
            for kind, value in core_findings(source):
                core[f"{rel}\t{kind}\t{value}"] += 1
        if not is_lane(rel):
            for kind, value in private_findings(source):
                private[f"{rel}\t{kind}\t{value}"] += 1
    return core, private


def test_scanner_sees_every_reference_form():
    source = (
        '"""TESSERA_ in a docstring is prose."""\n'
        "import tessera.serving.contract\n"
        "from .tessera_menu import menu_mode, _bound\n"
        "from .. import tessera_formats\n"
        "from prismaquant.tessera_render import render\n"
        "from .tessera_runtime.pin import PIN\n"
        "def f(name):\n"
        "    from prismaquant import tessera_campaign\n"
        "    if tf.is_tessera_format_name(name) or is_tessera_format_name(name):\n"
        "        return b'TESSERA_', 'tessera.uniform_control.v1', f'{name}TESSERA_X'\n"
        "X = ('xTESSERA_', 'tessera.no_version', '_TESSERA_OK')\n"
    )
    assert sorted(core_findings(source)) == [
        ("call", _PREDICATE), ("call", _PREDICATE),
        ("import", "..tessera_formats"), ("import", ".tessera_menu"),
        ("import", ".tessera_runtime.pin"), ("import", "prismaquant.tessera_campaign"),
        ("import", "prismaquant.tessera_render"), ("import", "tessera.serving.contract"),
        ("literal", "TESSERA_"), ("literal", "TESSERA_X"),
        ("literal", "tessera.uniform_control.v1"),
    ]
    assert private_findings(source) == [("private-import", ".tessera_menu._bound")]
    assert private_findings("from tessera.x import _y\nfrom .joint import _z\n") == []


def test_core_set_is_the_declared_fence():
    assert is_core("prismaquant/allocator_solver.py")
    assert is_core("prismaquant/model_profiles/specs/x.py")
    assert is_core("prismaquant/validate_native_export.py")
    assert not is_core("prismaquant/tessera_menu.py")
    assert not is_core("prismaquant/joint_cost_quantum.py")
    assert not is_core("prismaquant/tessera_runtime/allocator.py")
    assert is_lane("prismaquant/tessera_runtime/pin.py")
    assert not is_lane("prismaquant/glm_mtp_capture.py")


def test_core_lane_references_equal_the_frozen_allowlist():
    core, _ = scan()
    assert_equal(core, "tessera_core.txt",
                 "reach the Tessera lane through a declared seam (lane_spec, "
                 "serving_profile_specs, the format registry), not a core import")


def test_private_lane_imports_equal_the_frozen_allowlist():
    _, private = scan()
    assert_equal(private, "tessera_private.txt",
                 "a helper other modules need is not private to the lane: give "
                 "it a public name in a neutral module")
