"""The package does not import fleet tools (T1: tools keep argv and printing).

Every ``prismaquant/`` import of a module under ``tools/``, at any depth, is
one line of ``tests/boundary_allowlists/package_tools.txt``. A new reach fails;
the allowlist only shrinks as the remaining owners move into the package.
"""
from __future__ import annotations

import ast
from collections import Counter

from tests.boundary_gate import assert_equal, python_files


def tools_imports(source: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names
                      if alias.name.split(".")[0] == "tools"]
        elif (isinstance(node, ast.ImportFrom) and node.level == 0
              and node.module and node.module.split(".")[0] == "tools"):
            found += ([f"tools.{alias.name}" for alias in node.names]
                      if node.module == "tools" else [node.module])
    return found


def test_scanner_sees_every_import_form():
    source = ("import tools.a\nfrom tools import b as c\n"
              "def f():\n    from tools.d import e\n"
              "from .tools import x\nfrom toolsmith import y\n")
    assert sorted(tools_imports(source)) == ["tools.a", "tools.b", "tools.d"]


def test_package_imports_of_tools_equal_the_frozen_allowlist():
    live = Counter(f"{rel}\timport\t{module}"
                   for rel, source in python_files("prismaquant")
                   for module in tools_imports(source))
    assert_equal(live, "package_tools.txt",
                 "move the logic into a package owner and leave tools argv and printing")
