"""Recompute hosted migration pin proposals without editing historical evidence.

Run in an admitted test environment. Review seal counts and source changes before
applying the proposed test files from --output-dir.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from pprint import pformat
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / 'tests')]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seals = runpy.run_path(str(root / 'tests/test_no_new_seals.py'))
    print(json.dumps({'seal_violations': seals['violations'](seals['_campaign_sources']()),
        'seal_counts': {path: dict(Counter(scope for scope, _ in seals['seal_sites']((root / path).read_text())))
                        for path in seals['MODULES']}}, indent=2))
    name = 'test_guarded_read_digest_retention_2637.py'
    source = (root / 'tests' / name).read_text()
    pins = runpy.run_path(str(root / 'tests' / name))
    start = source.index('CURRENT_SOURCE_BLOBS = {')
    end = source.index('\n}', start) + 2
    current = {}
    for path in pins['CURRENT_SOURCE_BLOBS']:
        raw = (root / path).read_bytes()
        current[path] = (hashlib.sha1(b'blob ' + str(len(raw)).encode('ascii') + b'\0' + raw).hexdigest(),
                         hashlib.sha256(raw).hexdigest())
    section = '# Current native-migration sources; historical manifest pins above remain unchanged.\nCURRENT_SOURCE_BLOBS = ' + pformat(current, sort_dicts=False, width=100)
    (args.output_dir / name).write_text(source[:start] + section + source[end:])
    name = 'test_tensor_digest_owner_recipes_2638.py'
    source = (root / 'tests' / name).read_text()
    baseline = json.loads((root / 'tests/fixtures/duplication_baseline.json').read_text())
    assert 'hashlib:prismaquant/tessera_hessian.py::token_ids_sha256' not in baseline['primitive_digest_sites']
    digest = hashlib.sha256(json.dumps(baseline, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    source = re.sub(r'TOKEN_ONLY_BASELINE_SHA256 = "[0-9a-f]{64}"',
                    f'TOKEN_ONLY_BASELINE_SHA256 = "{digest}"', source)
    (args.output_dir / name).write_text(source)


if __name__ == '__main__':
    main()
