"""Own-process hooks and their users have a directory-scoped Bazel input (#3215)."""
from __future__ import annotations

import ast
from pathlib import Path


TESTS = Path(__file__).resolve().parent
OWN_PROCESS = TESTS / 'own_process'
OWN_PROCESS_HELPERS = {
    'OwnProcessFailure', 'OwnProcessItem', '_OwnProcessGroup',
    '_own_process_root', '_fold_own_process_reports', '_run_own_process',
    '_own_process_measure_args', '_write_own_process_measurement',
    '_own_process_record', '_append_own_process_record',
}


def test_own_process_hooks_and_users_are_scoped_to_their_directory():
    outside = []
    for path in sorted(TESTS.rglob('*.py')):
        if path.is_relative_to(OWN_PROCESS):
            continue
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute) and node.attr in {'own_process', 'own_cgroup'}
                    and isinstance(node.value, ast.Attribute) and node.value.attr == 'mark'):
                outside.append(f'{path.relative_to(TESTS)}:{node.lineno}: {node.attr}')
            if isinstance(node, ast.Attribute) and node.attr in OWN_PROCESS_HELPERS:
                outside.append(f'{path.relative_to(TESTS)}:{node.lineno}: {node.attr}')
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in OWN_PROCESS_HELPERS:
                        outside.append(f'{path.relative_to(TESTS)}:{node.lineno}: {alias.name}')
    root = ast.parse((TESTS / 'conftest.py').read_text(encoding='utf-8'))
    definitions = {node.name for node in root.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    assert not definitions & OWN_PROCESS_HELPERS, sorted(definitions & OWN_PROCESS_HELPERS)
    assert not outside, '\n'.join(outside)
