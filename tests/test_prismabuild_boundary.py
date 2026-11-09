"""No Python module loads the retired PrismaBuild package, statically or dynamically.

The AST half of the ban (``tests/test_prismabuild_ban.py`` holds the text half and the exclusion
lists): the finding set over the live scope is empty and there is no allowlist.
"""
from __future__ import annotations

import ast

from tests.test_prismabuild_ban import excluded, read_text, tracked


def dynamic_loads(source: str) -> list[str]:
    """Imports of ``prismabuild``/``fleet_sdk`` and loads through ``import_module``/``sdk_submodule``."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            found += [n for n in names if n == "fleet_sdk" or n == "prismabuild" or n.startswith("prismabuild.")]
        elif isinstance(node, ast.Call) and node.args:
            callee = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            arg = node.args[0]
            head = (arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                    else arg.values[0].value if isinstance(arg, ast.JoinedStr) and arg.values
                    and isinstance(arg.values[0], ast.Constant) else "")
            if callee == "sdk_submodule" or (callee == "import_module" and str(head).startswith("prismabuild")):
                found.append(callee)
    return found


def scan() -> dict[str, list[str]]:
    found = {}
    for path in tracked():
        if not path.endswith(".py") or excluded(path):
            continue
        source = read_text(path)
        hits = dynamic_loads(source) if source is not None else []
        if hits:
            found[path] = sorted(set(hits))
    return found


def test_no_live_module_loads_prismabuild():
    assert scan() == {}


def test_the_scanner_sees_every_load_form():
    source = ("import importlib\nimport prismabuild.pool\nfrom prismabuild import core\n"
              "importlib.import_module(f'prismabuild.{name}')\nsdk_submodule('core')\n"
              "importlib.import_module('json')\nimport fleet_sdk\n")
    assert sorted(dynamic_loads(source)) == [
        "fleet_sdk", "import_module", "prismabuild", "prismabuild.pool", "sdk_submodule"]
