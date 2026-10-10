#!/usr/bin/env python3
"""Verify a PrismaBuild full-suite run against the exact head it claims to test.

The check every integrator merge ended with (pq-integrator, 2026-10-09). It is a
mechanical check; no judgment is needed, so it should run in code (D61).

For each shard result JSON (pbtest --json output) it checks:
  1. the shard returned 0 and its CAS receipt file exists;
  2. the receipt names a checkout-snapshot bundle whose sha256 equals the CAS blob;
  3. that snapshot commit has the claimed HEAD as its only parent and differs from
     HEAD by exactly one added `.pbrun-closure.<hex>.json` file (source agreement);
  4. it sums the 'N passed' counts of the shards.
It prints one JSON line and exits 0 only when nothing is bad and every receipt has an ok snapshot.
The collected/ran reconciliation is printed by pbtest in its log ("reconciliation:" line); read it there.

Usage:
  pq-verify-suite-receipts.py --checkout DIR --head SHA RESULT.json [RESULT.json ...]

Wording rule learned the hard way: report "ran" (the log reconciliation count, which
includes skips) separately from "passed" (this script's passed_sum). Never write
"passed" for the ran count.
"""
import argparse, hashlib, json, os, re, subprocess, sys, tempfile
from pathlib import Path

CAS = Path('/mnt/shared/prismabuild-fleet/cas/blobs')


def sh(*a, **k):
    return subprocess.run(a, capture_output=True, text=True, **k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkout', required=True, help='worktree that holds HEAD')
    ap.add_argument('--head', required=True, help='full 40-hex head sha')
    ap.add_argument('results', nargs='+')
    a = ap.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}', a.head):
        sys.exit('--head must be a full 40-hex sha')
    work = tempfile.mkdtemp(prefix='verify-suite-')
    repo = os.path.join(work, 'bare.git')
    sh('git', 'init', '-q', '--bare', repo)
    r = sh('git', '--git-dir', repo, 'fetch', '-q', '--no-tags', a.checkout, a.head)
    if r.returncode:
        sys.exit('cannot fetch head from checkout: ' + r.stderr[:200])
    bad, receipts, snaps, passed = [], 0, 0, 0
    for path in a.results:
        for e in json.load(open(path)):
            tag = (Path(path).name, e.get('shard'))
            rp = e.get('receipt_path')
            if str(e.get('returncode')) != '0' or not rp or not os.path.exists(rp):
                bad.append((tag, 'receipt or return code')); continue
            receipts += 1
            m = re.match(r'(\d+) passed', e.get('summary', ''))
            passed += int(m.group(1)) if m else 0
            rec = json.load(open(rp))
            for i in rec['producer']['inputs']:
                if i['id'] != 'pbrun.checkout-snapshot':
                    continue
                sha = i['sha256']
                blob = CAS / sha[:2] / sha
                h = hashlib.sha256()
                with open(blob, 'rb') as f:
                    for c in iter(lambda: f.read(8 << 20), b''):
                        h.update(c)
                if h.hexdigest() != sha:
                    bad.append((tag, 'bundle digest')); continue
                heads = sh('git', 'bundle', 'list-heads', str(blob)).stdout.split()
                commit = heads[0]
                sh('git', '--git-dir', repo, 'fetch', '-q', '--no-tags', str(blob), commit)
                parent = sh('git', '--git-dir', repo, 'log', '-1', '--format=%P', commit).stdout.strip()
                diff = sh('git', '--git-dir', repo, 'diff-tree', '-r', '--name-status',
                          a.head, commit).stdout.strip().splitlines()
                ok = (parent == a.head and len(diff) == 1
                      and re.match(r'^A\t\.pbrun-closure\.[0-9a-f]+\.json$', diff[0]))
                if ok:
                    snaps += 1
                else:
                    bad.append((tag, 'snapshot', parent[:8], diff[:2]))
    print(json.dumps({'receipts': receipts, 'snapshots_ok': snaps,
                      'passed_sum': passed, 'bad': bad}, sort_keys=True))
    sys.exit(1 if bad or snaps != receipts else 0)


if __name__ == '__main__':
    main()
