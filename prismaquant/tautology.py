from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    test: str
    rule: str


_SKIP = {".git", "node_modules", ".venv", "venv", "__pycache__"}
_COMPARE = {
    "assertEqual", "assertEquals", "assertIs", "assertCountEqual",
    "assertDictEqual", "assertListEqual", "assertSetEqual", "assertTupleEqual",
    "assertSequenceEqual", "assertMultiLineEqual", "assertGreaterEqual",
    "assertLessEqual",
}
_NONLITERAL = object()


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _self_compare(left: ast.AST, right: ast.AST) -> bool:
    return (
        not any(isinstance(node, ast.Call)
                for operand in (left, right) for node in ast.walk(operand))
        and ast.dump(left, annotate_fields=False) == ast.dump(right, annotate_fields=False)
    )


def _literal(node: ast.AST) -> object:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _NONLITERAL


def _assertion_rule(node: ast.AST) -> str | None:
    if isinstance(node, ast.Assert):
        value = _literal(node.test)
        if value is not _NONLITERAL and value:
            return "constant"
        condition = node.test
        if (isinstance(condition, ast.Compare) and len(condition.ops) == 1
                and isinstance(condition.ops[0], (ast.Eq, ast.Is, ast.GtE, ast.LtE))
                and _self_compare(condition.left, condition.comparators[0])):
            return "self_compare"
    elif isinstance(node, ast.Call):
        name = _name(node.func)
        if name in _COMPARE and len(node.args) >= 2:
            left, right = node.args[:2]
            if _self_compare(left, right):
                return "self_compare"
            if name in {"assertEqual", "assertEquals"}:
                left_value, right_value = _literal(left), _literal(right)
                if (left_value is not _NONLITERAL and right_value is not _NONLITERAL
                        and left_value == right_value):
                    return "constant"
        elif node.args:
            value = _literal(node.args[0])
            if value is not _NONLITERAL and (
                (name == "assertTrue" and bool(value))
                or (name == "assertFalse" and not value)
                or (name == "assertIsNone" and value is None)
                or (name == "assertIsNotNone" and value is not None)
            ):
                return "constant"
    return None


def _swallowed(try_node: ast.Try) -> list[int]:
    if not any(
        isinstance(node, ast.Assert)
        or (isinstance(node, ast.Call) and _name(node.func).startswith("assert"))
        for statement in try_node.body for node in ast.walk(statement)
    ):
        return []
    lines = []
    for handler in try_node.handlers:
        types = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
        catches_assertion = handler.type is None or any(
            _name(node) in {"Exception", "BaseException", "AssertionError"}
            for node in types
        )
        if catches_assertion and all(
            isinstance(statement, (ast.Pass, ast.Continue, ast.Return))
            or (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant))
            for statement in handler.body
        ):
            lines.append(handler.lineno)
    return lines


def scan(root: str | Path) -> list[Finding]:
    root = Path(root)
    findings = []
    for path in root.rglob("*.py"):
        if (_SKIP.intersection(path.parts) or not path.is_file()
                or not (path.name.startswith("test_") or path.name.endswith("_test.py"))):
            continue
        relative = path.relative_to(root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (SyntaxError, UnicodeDecodeError, ValueError):
            continue
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not function.name.startswith("test"):
                continue
            seen = set()
            for node in ast.walk(function):
                rule = _assertion_rule(node)
                if rule is not None:
                    seen.add((node.lineno, rule))
                if isinstance(node, ast.Try):
                    seen.update((line, "swallowed") for line in _swallowed(node))
            findings.extend(Finding(relative, line, function.name, rule) for line, rule in seen)
    return sorted(findings, key=lambda finding: (
        finding.path, finding.line, finding.test, finding.rule))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Find mechanical tautologies in Python tests.")
    parser.add_argument("root")
    findings = scan(parser.parse_args(argv).root)
    for finding in findings:
        print(json.dumps(asdict(finding)))
    print(json.dumps({"total": len(findings)}), file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
