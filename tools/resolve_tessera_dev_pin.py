#!/usr/bin/env python3
"""Print PrismaQuant's reviewed Tessera commit without importing PrismaQuant.

``resolve_literal_pin`` is also the body of
``tools/resolve_prismabuild_dev_pin.py``: both resolvers read one literal
commit out of its owning source file, and pbtest's pin guard runs each
``tools/resolve_<module>_dev_pin.py`` before pytest starts.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path


PIN_NAME = "TESSERA_DEV_PIN_COMMIT"
PIN_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "prismaquant"
    / "tessera_runtime_contract.py"
)


def resolve_literal_pin(source: Path, name: str) -> str:
    """Read the single literal ``name`` commit from its owning source file."""

    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    values: list[object] = []
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in targets
        ):
            values.append(ast.literal_eval(node.value))
    if len(values) != 1:
        raise SystemExit(
            f"{source}: expected exactly one literal {name} assignment"
        )
    value = values[0]
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise SystemExit(
            f"{source}: {name} must be a full lowercase Git SHA"
        )
    return value


def resolve_tessera_dev_pin(source: Path = PIN_SOURCE) -> str:
    """Read the single literal development pin from its owning source file."""

    return resolve_literal_pin(source, PIN_NAME)


if __name__ == "__main__":
    print(resolve_tessera_dev_pin())
