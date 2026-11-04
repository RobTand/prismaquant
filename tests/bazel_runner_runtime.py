"""Exercise remote CLI scan boundaries in the pinned runner image, without pytest.

Run each role with the image's system Python. Only the external planning, test
execution and service calls are replaced; runtime bootstrap and both-key scans
execute unchanged. This proves runtime admission, not native Bazel selection.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.ci import bazel_artifacts, bazel_gate, bazel_selection, remote_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=('planner', 'shard', 'merger', 'post-merge'))
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 14):
        assert sys.version_info[:2] == (3, 12)
        assert importlib.util.find_spec('pip') is None
        assert shutil.which('python3.14') is None
        print('RUNNER_START ' + args.role + ' Python3.12 no-pip no-python3.14', flush=True)
    mode = 'main' if args.role == 'post-merge' else 'pr'
    head, base, label = 'a' * 40, 'b' * 40, '//tests:test_consumer'
    row = {'population': [label], 'tests': [label], 'shards': [[label]]}
    document = {'schema': 'prismaquant.bazel-shards.v1', 'mode': mode,
        'base': base, 'head': head, 'selection': 'impacted', 'scopes': ['//tests'],
        'all_scopes': False, 'reason': [], 'hosted': [], 'hosted_arm64': [],
        'architectures': {arch: row for arch in (
            ('amd64', 'arm64') if mode == 'main' else ('amd64',))}}
    raw = json.dumps(document).encode()
    result = {'schema': remote_gate.SCHEMA, 'mode': mode, 'base': base, 'head': head,
        'verdict': 'pass', 'selection': 'impacted', 'scopes': ['//tests'],
        'all_scopes': False, 'reason': [], 'budget_warnings': [],
        'hosted': [], 'hosted_arm64': [], 'targets': [label], 'error': None}
    scans = []
    original_scan = bazel_artifacts.scan_upload

    def scan(directory, required):
        assert sys.version_info[:2] == (3, 14)
        assert bazel_artifacts.scanner.zstd is not None
        assert set(required) == {'BUILDBUDDY_API_KEY', 'MERGIFY_TOKEN'}
        code = original_scan(directory, required)
        scans.append(code)
        return code

    with tempfile.TemporaryDirectory(prefix='pq-runtime-proof-') as directory:
        root = Path(directory)
        upload = root / 'upload'
        os.environ.update(BUILDBUDDY_API_KEY='runner-bb-public-sentinel',
                          MERGIFY_TOKEN='runner-mergify-public-sentinel',
                          BUILDBUDDY_ARTIFACTS_DIRECTORY=str(upload),
                          GITHUB_OUTPUT=str(root / 'outputs'))
        with patch.object(bazel_artifacts, 'scan_upload', scan):
            if args.role == 'planner':
                with patch.object(bazel_selection, 'plan_revisions', return_value=document):
                    code = bazel_selection.main(['plan', '--mode', mode, '--base', base,
                        '--head', head, '--branch', 'runner-proof'])
            elif args.role == 'shard':
                reference = {'mode': mode, 'base': base, 'head': head,
                    'branch': 'runner-proof', 'architecture': 'amd64', 'index': 0,
                    'require_execution': False,
                    'manifest_uri': 'bytestream://remote.buildbuddy.io/proof',
                    'manifest_sha256': hashlib.sha256(raw).hexdigest(),
                    'partition_sha256': hashlib.sha256((label + '\n').encode()).hexdigest()}
                os.environ['PQ_BAZEL_SHARD'] = json.dumps(reference)
                with patch.object(bazel_gate, 'fetch_artifact', return_value=raw), \
                     patch.object(bazel_artifacts, 'prepare_history'), \
                     patch.object(bazel_gate, 'execute_shard', return_value=result):
                    code = bazel_gate.main(['shard'])
            else:
                with patch.object(remote_gate, 'run_gate', return_value=result):
                    code = remote_gate.main([mode, '--base', base, '--head', head,
                        '--branch', 'runner-proof', '--output', str(upload)])
        assert code == 0, f'{args.role} refused with exit {code}'
        assert scans == [0, 0], scans
        receipts = list(upload.glob('*.json'))
        assert receipts
        for receipt in receipts:
            assert json.loads(receipt.read_text())['head'] == head
        print('RUNNER_PASS ' + args.role + ' Python3.14 native-zstd both-key-scans=2', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
