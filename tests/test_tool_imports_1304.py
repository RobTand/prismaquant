"""Every tool and script still resolves what it imports from PrismaQuant (#1328).

Step 2 of #1304 removed the retired codebook lane's modules, parameters,
helpers and branches. A tool that imported a removed name would fail only when
an operator next ran it, so this sweep checks every surviving file under
``tools/`` and ``scripts/``.

Several tools are flat scripts that parse arguments or read files at import
time, so the sweep never executes a tool. It parses each file, imports every
``prismaquant`` and ``tools`` module the file names, and checks that each
imported name, and each ``module.attribute`` the file reads through a module
alias, exists.
"""
from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FILES = sorted(
    path.relative_to(ROOT).as_posix()
    for top in ("tools", "scripts")
    for path in (ROOT / top).rglob("*.py")
    if "archive" not in path.relative_to(ROOT).parts
)
OWNED = ("prismaquant", "tools")


def _owned(module: str | None) -> bool:
    return bool(module) and module.split(".")[0] in OWNED


def _resolve(module: str, name: str) -> bool:
    if hasattr(importlib.import_module(module), name):
        return True
    try:
        importlib.import_module(f"{module}.{name}")
    except ModuleNotFoundError:
        return False
    return True


def _missing(path: str) -> list[str]:
    tree = ast.parse((ROOT / path).read_text(), path)
    aliases: dict[str, str] = {}
    missing: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and not node.level:
            if not _owned(node.module):
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                if not _resolve(node.module, alias.name):
                    missing.append(f"{node.lineno}: {node.module}.{alias.name}")
                    continue
                full = f"{node.module}.{alias.name}"
                try:
                    importlib.import_module(full)
                except ModuleNotFoundError:
                    continue
                aliases[alias.asname or alias.name] = full
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if not _owned(alias.name):
                    continue
                importlib.import_module(alias.name)
                if alias.asname:
                    aliases[alias.asname] = alias.name
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in aliases):
            module = importlib.import_module(aliases[node.value.id])
            if not hasattr(module, node.attr):
                missing.append(
                    f"{node.lineno}: {aliases[node.value.id]}.{node.attr}")
    return missing


@pytest.mark.parametrize("path", FILES)
def test_tool_resolves_its_prismaquant_imports(path):
    assert _missing(path) == []
