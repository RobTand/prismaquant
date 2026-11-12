"""Check that every original test statement remains, in its original order."""
import ast
import json
from pathlib import Path
import subprocess
import sys

revision, path = sys.argv[1:]
original = ast.parse(subprocess.check_output(
    ['git', 'show', f'{revision}:{path}'], text=True))
current = ast.parse(Path(path).read_text())
assert len(current.body) >= len(original.body)
assert all(ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)
           for a, b in zip(original.body, current.body)), 'original test source changed'
print(json.dumps({
    'base': revision,
    'path': path,
    'original_module_statements': len(original.body),
    'original_assertions': sum(isinstance(n, ast.Assert) for n in ast.walk(original)),
    'original_source_preserved': True,
}, indent=2))
