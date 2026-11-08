"""Reject PB imports at test collection without importing PB itself."""
import ast
from pathlib import Path
import pytest


def _collection_prismabuild_imports(tests_root):
    """Scan executable collection syntax and statically named local helpers."""
    imports = set()
    for path in sorted(tests_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = {node.name: node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        aliases = {}
        for node in tree.body:
            if isinstance(node, ast.Import):
                aliases.update({alias.asname or alias.name: alias.name for alias in node.names})
            elif isinstance(node, ast.ImportFrom) and node.module:
                aliases.update({alias.asname or alias.name: f"{node.module}.{alias.name}"
                                for alias in node.names})
        pending, called = list(tree.body), set()
        while pending:
            node = pending.pop()
            if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
                test = node.test
                if (isinstance(test.left, ast.Name) and test.left.id == "__name__"
                        and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
                        and len(test.comparators) == 1
                        and isinstance(test.comparators[0], ast.Constant)
                        and test.comparators[0].value == "__main__"):
                    # A command entry point does not execute when pytest imports it.
                    pending.extend(node.orelse)
                    continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Decorators run now, unlike the body they decorate.
                pending.extend(node.decorator_list)
                continue
            if isinstance(node, ast.Lambda):
                continue
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Call):
                target = ast.unparse(node.func)
                first, _, suffix = target.partition(".")
                target = aliases.get(first, first) + (f".{suffix}" if suffix else "")
                if target in {"importlib.import_module", "__import__"} and node.args:
                    argument = node.args[0]
                    if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                        names = [argument.value]
                if isinstance(node.func, ast.Name) and node.func.id in functions:
                    if node.func.id not in called:
                        called.add(node.func.id)
                        pending.extend(functions[node.func.id].body)
            if any(name == "prismabuild" or name.startswith("prismabuild.")
                   for name in names):
                imports.add(f"{path.relative_to(tests_root)}:{node.lineno}")
            pending.extend(ast.iter_child_nodes(node))
    return sorted(imports)


def test_tests_do_not_import_prismabuild_during_collection():
    """Collection precedes the restorers: SDK imports must be test-owned."""
    imports = _collection_prismabuild_imports(Path(__file__).parent)
    assert not imports, "collection-time PrismaBuild imports: " + ", ".join(imports)


@pytest.mark.parametrize("source,line", [
    ('import importlib\nimportlib.import_module("prismabuild.core")\n', 2),
    ('__import__("prismabuild")\n', 1),
    ('import pytest\ndef helper():\n    import prismabuild.core\n    return True\n'
     '@pytest.mark.skipif(helper(), reason="fixture")\ndef test_case(): pass\n', 3),
    ('import importlib\nimport pytest\n'
     '@pytest.mark.skipif(importlib.import_module("prismabuild.core"), reason="fixture")\n'
     'async def test_case(): pass\n', 3),
    ('def main():\n    import prismabuild.core\n'
     'if __name__ == "__main__":\n    main()\n', None),
    ('def test_case():\n    import prismabuild.core\n', None),
])
def test_collection_scan_catches_dynamic_and_decorator_imports(tmp_path, source, line):
    module = tmp_path / "test_fixture.py"
    module.write_text(source)
    expected = [] if line is None else [f"test_fixture.py:{line}"]
    assert _collection_prismabuild_imports(tmp_path) == expected
