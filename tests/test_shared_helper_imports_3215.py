"""Shared factories must not make their callers depend on test modules."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest


@pytest.mark.parametrize("stem", [
    "joint_aura_assignment_diagnostics",
    "tessera_expert_projection",
    "production_weight_cache",
    "footprint",
])
def test_shared_factories_are_imported_from_support_modules(stem):
    root = Path(__file__).resolve().parent
    module = "test_" + stem
    violations = []
    for path in sorted(root.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.ImportFrom) and node.module in {module, "tests." + module}:
                violations.append(f"{path.relative_to(root)}:{node.lineno}: "
                                  + ", ".join(name.name for name in node.names))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in {module, "tests." + module}:
                        violations.append(f"{path.relative_to(root)}:{node.lineno}: {alias.name}")
    assert not violations, "Import shared factories from support modules:\n" + "\n".join(violations)
