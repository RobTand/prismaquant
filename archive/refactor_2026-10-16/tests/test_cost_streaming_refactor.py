from __future__ import annotations

import ast

from tools.refactor_cost_streaming import migrate


def test_migration_preserves_unicode_and_python_line_boundaries(tmp_path):
    path = tmp_path / "digest_sites.py"
    path.write_text(
        'description = "\u2028"\n'
        'sites = (("\u03bb", "prismaquant.cost_streaming.canonical_fingerprint_key"),\n'
        '         ("prismaquant.cost_streaming", "canonical_fingerprint_key", {}))\n'
    )
    migrate(path, {"canonical_fingerprint_key": "streamed_model_identity"})
    values = {node.targets[0].id: ast.literal_eval(node.value)
              for node in ast.parse(path.read_text()).body}
    assert values == {
        "description": "\u2028",
        "sites": (("\u03bb", "prismaquant.streamed_model_identity.canonical_fingerprint_key"),
                  ("prismaquant.streamed_model_identity", "canonical_fingerprint_key", {})),
    }


def test_policy_inventory_migration_is_idempotent(tmp_path):
    path = tmp_path / "test_no_new_seals.py"
    modules = ["prismaquant/cost_streaming.py", *(f"module_{i}.py" for i in range(11))]
    path.write_text(f"MODULES = {tuple(modules)!r}\n")
    migrate(path, {})
    migrated = path.read_text()
    migrate(path, {})
    assert path.read_text() == migrated
    actual = ast.literal_eval(ast.parse(migrated).body[0].value)
    assert actual == (modules[0], "prismaquant/streamed_boundary_artifacts.py",
                      "prismaquant/streamed_model_identity.py", *modules[1:])
