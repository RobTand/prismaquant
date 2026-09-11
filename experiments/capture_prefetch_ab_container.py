"""Run the capture prefetch A/B inside the row's own declared container.

The memory guard reads its cgroup, and a campaign row reads the one its
container gives it. Running the A/B anywhere else measures a different
environment, so this driver reuses the row's sealed container spec -- image,
mounts and environment -- and only appends the reader count.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def row_spec(base, row_id):
    manifest = json.loads((Path(base)/'first-proof-anchor-preparation-05/workspace'
                           /'manifest.json').read_text())
    action = next(entry for entry in manifest
                  if f'/rows/{row_id}/' in ' '.join(entry['argv']))
    argv = action['argv']
    return json.loads(argv[argv.index('--spec')+1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', default='/mnt/shared/tessera-measurements/glm-canonical-census-20260908')
    parser.add_argument('--row', required=True)
    parser.add_argument('--readers', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--arms', default='cold,warm')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--env', action='append', default=[],
                        metavar='NAME=VALUE',
                        help='extra environment entry for the container, repeatable')
    args = parser.parse_args()

    from tools.tessera_campaign_container import main as run_container
    spec = row_spec(args.base, args.row)
    extra = dict(item.split('=', 1) for item in args.env)
    spec['env'] = dict(spec['env'], PRISMAQUANT_CAPTURE_READ_THREADS=str(args.readers), **extra)
    command = ['python3', '-u', '-m', 'experiments.capture_prefetch_ab',
               '--base', args.base, '--row', args.row, '--readers', str(args.readers),
               '--arms', args.arms, '--out', args.out]
    if args.limit:
        command += ['--limit', str(args.limit)]
    if args.profile:
        command += ['--profile']
    return run_container(['--spec', json.dumps(spec, sort_keys=True), '--', *command])


if __name__ == '__main__':
    raise SystemExit(main())
