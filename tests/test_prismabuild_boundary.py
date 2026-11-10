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
    def retired(name):
        return (name == "fleet_sdk" or name == "prismabuild" or name.startswith("prismabuild.")
                or name.startswith("prismaquant.prismabuild"))
    tree = ast.parse(source)
    aliases = {alias.asname for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
               and node.module in {'importlib', 'importlib.util', 'builtins'}
               for alias in node.names if alias.asname and alias.name in {'import_module', 'find_spec', '__import__'}}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or "", *[(node.module or "") + "." + a.name for a in node.names]]
            if isinstance(node, ast.Import):
                found += [n for n in names if retired(n)]
            else:
                found += [names[0]] if retired(names[0]) else [n for n in names[1:] if retired(n)]
        elif isinstance(node, ast.Call) and node.args:
            callee = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            arg = node.args[0]
            head = (arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                    else arg.values[0].value if isinstance(arg, ast.JoinedStr) and arg.values
                    and isinstance(arg.values[0], ast.Constant) else "")
            if callee == "sdk_submodule" or (callee in {"import_module", "find_spec", "__import__", *aliases} and retired(str(head))):
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


def test_reexported_and_dynamic_package_loads_are_banned():
    cases = {
        "import prismaquant.prismabuild.client": "prismaquant.prismabuild.client",
        "from prismaquant import prismabuild_progress": "prismaquant.prismabuild_progress",
        "importlib.util.find_spec('prismabuild')": "find_spec",
        "__import__('prismabuild')": "__import__",
        "from importlib.util import find_spec as discover; discover('prismabuild')": "discover",
    }
    for source, expected in cases.items():
        assert expected in dynamic_loads(source)
    assert dynamic_loads("importlib.util.find_spec('json'); __import__('pathlib')") == []
