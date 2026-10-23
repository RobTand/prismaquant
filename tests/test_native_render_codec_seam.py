"""The native codec's cross-module seam is its public names.

Cache fill, export, batched GPTQ, the GGUF lane and the fused kernel all
render through ``native_render_codec``.  Every name they call is public, so a
leading underscore keeps meaning "codec-internal": a live module that needs a
private helper publishes it instead of reaching in.  Tests may still exercise
private helpers directly.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODEC = "native_render_codec"


def _private_reaches(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases: set[str] = set()
    reaches: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.endswith(CODEC):
                reaches += [
                    f"{node.lineno} from-import {alias.name}"
                    for alias in node.names
                    if alias.name.startswith("_") and not alias.name.startswith("__")
                ]
            for alias in node.names:
                if alias.name == CODEC:
                    aliases.add(alias.asname or alias.name)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith(CODEC):
                    aliases.add(alias.asname or alias.name.rsplit(".", 1)[-1])
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr.startswith("_")
                and not node.attr.startswith("__")
                and isinstance(node.value, ast.Name) and node.value.id in aliases):
            reaches.append(f"{node.lineno} attribute {node.attr}")
    return reaches


def test_no_live_module_reaches_a_private_codec_name():
    offenders = []
    for top in ("prismaquant", "tools", "experiments"):
        for path in sorted((ROOT / top).rglob("*.py")):
            if "archive" in path.parts or path.name == f"{CODEC}.py":
                continue
            for reach in _private_reaches(path):
                offenders.append(f"{path.relative_to(ROOT)}:{reach}")
    assert not offenders, (
        "publish the codec name instead of reaching a private one:\n"
        + "\n".join(offenders)
    )
