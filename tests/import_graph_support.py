"""Static import graph of the ``prismaquant`` package, shared by the import-cut tests (PQ #3215).

The graph is ``ast``-only and runs no package code. It has one edge for every ``import`` and
``from ... import`` statement at any depth in a module, except under ``if TYPE_CHECKING:``. It also
has edges for a literal ``import_module`` or ``__import__`` name, for a name in the package's
``_LAZY_EXPORTS`` table, and for each parent package's ``__init__``, which Python runs first.
It lives outside the ``test_*`` modules so tests share it without importing each other.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "prismaquant"


def _modules() -> dict[str, Path]:
    found = {}
    for path in PACKAGE.rglob("*.py"):
        parts = path.relative_to(ROOT).with_suffix("").parts
        found[".".join(parts[:-1] if parts[-1] == "__init__" else parts)] = path
    return found


def _lazy_exports(modules: dict[str, Path]) -> dict[str, str]:
    for node in ast.parse(modules["prismaquant"].read_text()).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "_LAZY_EXPORTS" for target in node.targets):
            return {name: "prismaquant" + relative for name, relative in ast.literal_eval(node.value).items()}
    return {}


def _statements(tree: ast.AST):
    """Import statements and literal dynamic-import calls outside ``if TYPE_CHECKING:``."""
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            for item in node.orelse:
                yield from _statements(ast.Module(body=[item], type_ignores=[]))
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call)):
            yield node
        yield from _statements(node)


def _targets(module: str, path: Path, modules: dict[str, Path], lazy: dict[str, str]) -> set[str]:
    package = module if path.name == "__init__.py" else module.rpartition(".")[0]
    names: set[str] = set()
    for node in _statements(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.split(".")[:len(package.split(".")) - node.level + 1]
                base = ".".join(anchor + ([node.module] if node.module else []))
            names.add(base)
            for alias in node.names:
                names.add(f"{base}.{alias.name}")
                if base == "prismaquant" and alias.name in lazy:
                    names.add(lazy[alias.name])
        elif (isinstance(node.func, (ast.Name, ast.Attribute))
              and (node.func.id if isinstance(node.func, ast.Name) else node.func.attr)
              in {"import_module", "__import__"}
              and node.args and isinstance(node.args[0], ast.Constant)
              and isinstance(node.args[0].value, str)):
            names.add(node.args[0].value)
    found = set()
    for name in names | {package}:
        parts = name.split(".")
        found |= {".".join(parts[:end]) for end in range(1, len(parts) + 1)} & set(modules)
    return found - {module}
