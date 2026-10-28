"""The campaign table's private helpers are private to it.

A helper another module needs is a contract, so it gets a public name in
``tessera_campaign_table``. This holds every ``prismaquant/`` and ``tools/``
module other than the owner to the public names, including the fleet tools
that ``test_tessera_core_boundary`` does not scan.
"""
from __future__ import annotations

import ast

from tests.boundary_gate import python_files

OWNER = "prismaquant.tessera_campaign_table"
OWNER_PATH = "prismaquant/tessera_campaign_table.py"


def _private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def private_reaches(rel: str, source: str) -> list[str]:
    package = rel[:-3].replace("/", ".").rpartition(".")[0]
    aliases, found = set(), []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(alias.asname for alias in node.names
                           if alias.name == OWNER and alias.asname)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                parts = parts[:len(parts) - node.level + 1]
                base = ".".join(parts + ([node.module] if node.module else []))
            if base == OWNER:
                found += [alias.name for alias in node.names if _private(alias.name)]
            elif base + ".tessera_campaign_table" == OWNER:
                aliases.update(alias.asname or alias.name for alias in node.names
                               if alias.name == "tessera_campaign_table")
    found += [node.attr for node in ast.walk(tree)
              if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
              and node.value.id in aliases and _private(node.attr)]
    return found


def test_scanner_sees_imports_and_attribute_reads():
    source = ("from prismaquant.tessera_campaign_table import _a, b\n"
              "from prismaquant import tessera_campaign_table as table\n"
              "table._c(table.d)\n")
    assert private_reaches("tools/x.py", source) == ["_a", "_c"]
    assert private_reaches("prismaquant/y.py",
                           "from .tessera_campaign_table import _e\n") == ["_e"]


def test_no_module_reaches_a_private_campaign_table_name():
    found = sorted(f"{rel}: {name}"
                   for rel, source in python_files("prismaquant", "tools")
                   if rel != OWNER_PATH
                   for name in private_reaches(rel, source))
    assert found == [], "give the helper a public name in tessera_campaign_table"
