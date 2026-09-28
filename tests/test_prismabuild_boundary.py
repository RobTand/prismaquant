"""Freeze where PrismaQuant reaches into PrismaBuild internals (#1534).

PrismaBuild (PB) stands alone. PrismaQuant's allowed interface to it is
``pbrun``, ``pbtest``, ``pbcampaign`` and the progress helper; everything
else should go through a published, versioned PB client module, which does
not exist yet (step 2 of the decoupling plan in the 2026-09-27 coupling
inventory). This test is the mechanical half of that rule, step 0 of the
same plan.

Over ``prismaquant/`` and ``tools/`` it parses each module with ``ast`` and
finds:

- **imports** of ``prismabuild...``: ``import`` and ``from ... import`` at
  any depth, ``importlib.import_module`` on a ``prismabuild.`` string or
  f-string, and calls to ``staged_lease.sdk_submodule``, PQ's own loader
  for PB modules (a call with a computed name records ``prismabuild.*``).
  Modules named in ``PUBLIC_CLIENT`` are exempt; step 2 names the client
  there;
- **private access**: an attribute whose name starts with one underscore,
  read off a PB module or a name imported from one, and private names
  imported from PB directly. The binding is resolved within one module,
  flow-insensitively: import aliases, names and ``self.x`` attributes
  assigned from a PB import, and module functions that return one (the
  ``_pool_module()._read_json`` form). A static guard, not execution;
  a binding carried across modules is out of its reach;
- **literals**: ``str``/``bytes`` constants outside docstrings that contain
  ``prismabuild-fleet/pb-queue`` or ``prismabuild-fleet/cas``, or match
  ``prismaquant.prismabuild.(cas_receipt|worker_attestation|pool_outcome|
  residency_map)``.

Each finding is one line of ``tests/boundary_allowlists/prismabuild.txt``,
keyed by path and module, attribute or matched token, never by line number.
The test requires exact multiset equality:

- a new reference, or one more occurrence of an allowlisted one, fails;
- a removed reference fails until its line is deleted in the same change.

The allowlist only shrinks; CI refuses a pull request that adds a line.
Step 2 (the PB client SDK) empties it.
"""
from __future__ import annotations

import ast
import re
from collections import Counter

from tests.boundary_gate import assert_equal, code_strings, escape, python_files

PUBLIC_CLIENT: tuple[str, ...] = ()
_PB = "prismabuild"
_LOADER = "sdk_submodule"
_PATHS = ("prismabuild-fleet/pb-queue", "prismabuild-fleet/cas")
_RECORD = re.compile(
    r"prismaquant\.prismabuild\.(?:cas_receipt|worker_attestation|pool_outcome"
    r"|residency_map)[A-Za-z0-9_.]*")


def _private(name: str) -> bool:
    return name.startswith("_") and not name.startswith("__")


def _pb_module(name: str) -> bool:
    return (name == _PB or name.startswith(_PB + ".")) and name not in PUBLIC_CLIENT


def _called(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _loaded(node: ast.AST) -> str | None:
    """The PB module a call loads dynamically, or ``None``."""
    if not isinstance(node, ast.Call) or not node.args:
        return None
    name, arg = _called(node), node.args[0]
    if name == _LOADER:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return f"{_PB}.{arg.value}"
        return f"{_PB}.*"
    if name != "import_module":
        return None
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return arg.value if _pb_module(arg.value) else None
    if (isinstance(arg, ast.JoinedStr) and arg.values
            and isinstance(arg.values[0], ast.Constant)
            and str(arg.values[0].value).startswith(_PB + ".")):
        return f"{str(arg.values[0].value).rstrip('.')}.*"
    return None


def _target(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "self"):
        return f"self.{node.attr}"
    return None


def findings(source: str) -> list[tuple[str, str]]:
    tree = ast.parse(source)
    found: list[tuple[str, str]] = []
    bound: set[str] = set()          # names and self.x bound to PB objects
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _pb_module(alias.name):
                    found.append(("import", alias.name))
                    bound.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module == _PB:
                modules = [f"{_PB}.{alias.name}" for alias in node.names]
            else:
                modules = [node.module]
            for module in dict.fromkeys(modules):
                if _pb_module(module):
                    found.append(("import", module))
            if _pb_module(node.module) or node.module == _PB:
                for alias in node.names:
                    bound.add(alias.asname or alias.name)
                    if _private(alias.name):
                        found.append(("private", alias.name))
        elif (module := _loaded(node)) is not None:
            found.append(("import", module))

    def pb_valued(node: ast.AST | None) -> bool:
        if node is None:
            return False
        if _loaded(node) is not None or _target(node) in bound:
            return True
        if isinstance(node, ast.Attribute):     # prismabuild.pool._x
            return pb_valued(node.value)
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in returning)

    returning: set[str] = set()
    functions = [n for n in ast.walk(tree)
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    assignments = [n for n in ast.walk(tree) if isinstance(n, (ast.Assign, ast.AnnAssign))]
    changed = True
    while changed:
        changed = False
        for node in assignments:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if pb_valued(node.value):
                for name in filter(None, map(_target, targets)):
                    if name not in bound:
                        bound.add(name)
                        changed = True
        for function in functions:
            if function.name not in returning and any(
                    isinstance(n, ast.Return) and pb_valued(n.value)
                    for n in ast.walk(function)):
                returning.add(function.name)
                changed = True

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and _private(node.attr) and pb_valued(node.value):
            found.append(("private", node.attr))
    for _, text in code_strings(tree):
        for path in _PATHS:
            found += [("literal", path)] * text.count(path)
        found += [("literal", escape(m)) for m in _RECORD.findall(text)]
    return found


def scan() -> Counter:
    counts: Counter = Counter()
    for rel, source in python_files("prismaquant", "tools"):
        for kind, value in findings(source):
            counts[f"{rel}\t{kind}\t{value}"] += 1
    return counts


def test_scanner_sees_every_reference_form():
    source = (
        '"""prismabuild-fleet/cas in a docstring is prose."""\n'
        "import importlib\n"
        "import prismabuild.reader_lease as lease\n"
        "from prismabuild import pool as pool_mod, core\n"
        "from prismabuild.core import _ID_RE\n"
        "def _pool_module():\n"
        "    module = sdk_submodule('pool')\n"
        "    return module\n"
        "def load(name):\n"
        "    return importlib.import_module(f'prismabuild.{name}')\n"
        "class C:\n"
        "    def __init__(self):\n"
        "        self._po = sdk_submodule(NAME)\n"
        "    def f(self):\n"
        "        self._po._commit(); lease._pin; pool_mod.PoolQueue\n"
        "        return _pool_module()._read_json(b'/mnt/shared/prismabuild-fleet/pb-queue')\n"
        "X = 'prismaquant.prismabuild.cas_receipt.v3 or prismaquant.prismabuild.action.v2'\n"
        "Y = other._private\n"
        "def g():\n"
        "    import prismabuild.pool\n"
        "    return prismabuild.pool._read_json\n"
    )
    assert sorted(findings(source)) == [
        ("import", "prismabuild.*"), ("import", "prismabuild.*"),
        ("import", "prismabuild.core"), ("import", "prismabuild.core"),
        ("import", "prismabuild.pool"), ("import", "prismabuild.pool"),
        ("import", "prismabuild.pool"), ("import", "prismabuild.reader_lease"),
        ("literal", "prismabuild-fleet/pb-queue"),
        ("literal", "prismaquant.prismabuild.cas_receipt.v3"),
        ("private", "_ID_RE"), ("private", "_commit"), ("private", "_pin"),
        ("private", "_read_json"), ("private", "_read_json"),
    ]


def test_prismabuild_references_equal_the_frozen_allowlist():
    assert_equal(scan(), "prismabuild.txt",
                 "use PrismaBuild's public interface (pbrun, pbtest, pbcampaign, "
                 "the progress helper), or its published client module once step 2 "
                 "lands; never a PB internal")
