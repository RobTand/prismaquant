"""Shared mechanics for the shrink-only boundary gates (decoupling step 0).

The gates themselves are ``tests/test_tessera_core_boundary.py`` and
``tests/test_prismabuild_boundary.py``; this module holds what they share:
docstring detection, the allowlist file format and the exact comparison.

An allowlist is a text file under ``tests/boundary_allowlists/``. Each
non-comment line is one occurrence, ``path<TAB>kind<TAB>value``, so a
reference that appears twice is two identical lines. Keys never carry a
line number or a count: removing an occurrence is then a pure deletion,
which the CI step (``.github/workflows/ci.yml``, "boundary allowlists only
shrink") admits, while any added line is refused.
"""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLISTS = Path(__file__).with_name("boundary_allowlists")

_DOC_OWNERS = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def docstring_ids(tree: ast.AST) -> set[int]:
    """``id()`` of every docstring constant, which the gates never read."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, _DOC_OWNERS) and node.body:
            first = node.body[0]
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, (str, bytes))):
                ids.add(id(first.value))
    return ids


def code_strings(tree: ast.AST):
    """Every ``str`` or ``bytes`` constant outside docstrings, as text.

    Constants inside f-strings are included. Bytes decode as latin-1, so a
    byte pattern matches the way its source spelling reads.
    """
    docstrings = docstring_ids(tree)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes))
                and id(node) not in docstrings):
            value = node.value
            yield node, value.decode("latin-1") if isinstance(value, bytes) else value


def escape(value: str) -> str:
    """One allowlist line per value: tabs and newlines are escaped."""
    return value.encode("unicode_escape").decode("ascii")


def python_files(*tops: str, root: Path = ROOT):
    """``(relative path, source)`` under ``tops``, skipping archive walls."""
    for top in tops:
        for path in sorted((root / top).rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            if "/archive/" in f"/{rel}":
                continue
            yield rel, path.read_text(encoding="utf-8")


def allowed(name: str) -> Counter:
    lines = (ALLOWLISTS / name).read_text(encoding="utf-8").splitlines()
    return Counter(line for line in lines if line and not line.startswith("#"))


def assert_equal(live: Counter, name: str, remedy: str) -> None:
    """Hold ``live`` equal to the allowlist ``name``, naming every difference."""
    frozen = allowed(name)
    new = sorted(k for k in live if k not in frozen)
    grown = {k: (frozen[k], v) for k, v in live.items() if k in frozen and v > frozen[k]}
    shrunk = {k: (n, live.get(k, 0)) for k, n in frozen.items() if live.get(k, 0) < n}
    assert not new, f"new references not in {name}: {new}; {remedy}"
    assert not grown, f"allowlisted references in {name} gained occurrences (allowed, live): {grown}"
    assert not shrunk, (
        f"references removed (allowed, live): {shrunk}; delete their lines from "
        f"tests/boundary_allowlists/{name} in the same change")
