"""Recompute hosted migration pin proposals without editing historical evidence.

Run in an admitted test environment. Review seal counts and source changes before
applying the proposed test files from --output-dir.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
import difflib
from pathlib import Path
import re
from pprint import pformat
import runpy
import sys
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reason', action='append', default=[], metavar='PATH=REVIEWED_REASON')
    parser.add_argument('--baseline-reason')
    args = parser.parse_args()
    reasons = dict(item.split('=', 1) for item in args.reason)
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / 'tests')]
    from prismaquant.digests import bytes_sha256hex, DIRECT_ASCII_STRICT
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seals = runpy.run_path(str(root / 'tests/test_no_new_seals.py'))
    print(json.dumps({'seal_violations': seals['violations'](seals['_campaign_sources']()),
        'seal_counts': {path: dict(Counter(scope for scope, _ in seals['seal_sites']((root / path).read_text())))
                        for path in seals['MODULES']}}, indent=2))
    name = 'test_guarded_read_digest_retention_2637.py'
    source = (root / 'tests' / name).read_text()
    pins = runpy.run_path(str(root / 'tests' / name))
    assignment = next(node for node in ast.parse(source).body
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'CURRENT_SOURCE_BLOBS'
            for target in node.targets))
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[:assignment.lineno - 1]))
    end = sum(map(len, lines[:assignment.end_lineno]))
    current = {}
    for path in pins['CURRENT_SOURCE_BLOBS']:
        raw = (root / path).read_bytes()
        blob = subprocess.run(['git', 'hash-object', '--stdin'], input=raw,
                              cwd=root, capture_output=True, check=True).stdout.decode().strip()
        current[path] = (blob, bytes_sha256hex(raw))
        old_blob = pins['CURRENT_SOURCE_BLOBS'][path][0]
        if current[path] != pins['CURRENT_SOURCE_BLOBS'][path]:
            if not reasons.get(path, '').strip():
                parser.error('changed pin requires --reason ' + path + '=REVIEWED_REASON')
            before = subprocess.run(['git', 'cat-file', 'blob', old_blob],
                cwd=root, capture_output=True, check=True).stdout.decode()
            print('Reviewed reason for ' + path + ': ' + reasons[path])
            print(''.join(difflib.unified_diff(before.splitlines(keepends=True),
                raw.decode().splitlines(keepends=True), fromfile=path + '@' + old_blob,
                tofile=path + '@' + blob)), end='')
    section = 'CURRENT_SOURCE_BLOBS = ' + pformat(current, sort_dicts=False, width=100) + '\n'
    proposed = source[:start] + section + source[end:]
    print(''.join(difflib.unified_diff(source.splitlines(keepends=True),
        proposed.splitlines(keepends=True), fromfile=name, tofile=name + '.proposed')), end='')
    (args.output_dir / name).write_text(proposed)
    name = 'test_tensor_digest_owner_recipes_2638.py'
    source = (root / 'tests' / name).read_text()
    baseline = json.loads((root / 'tests/fixtures/duplication_baseline.json').read_text())
    assert 'hashlib:prismaquant/tessera_hessian.py::token_ids_sha256' not in baseline['primitive_digest_sites']
    digest = DIRECT_ASCII_STRICT.sha256(baseline)
    old_source = source
    source = re.sub(r'TOKEN_ONLY_BASELINE_SHA256 = "[0-9a-f]{64}"',
                    f'TOKEN_ONLY_BASELINE_SHA256 = "{digest}"', source)
    if source != old_source:
        if not args.baseline_reason or not args.baseline_reason.strip():
            parser.error('baseline pin change requires --baseline-reason')
        print('Reviewed baseline reason: ' + args.baseline_reason)
        print(''.join(difflib.unified_diff(old_source.splitlines(keepends=True),
            source.splitlines(keepends=True), fromfile=name, tofile=name + '.proposed')), end='')
    (args.output_dir / name).write_text(source)


if __name__ == '__main__':
    main()
