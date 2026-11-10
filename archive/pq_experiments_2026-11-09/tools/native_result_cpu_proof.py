"""Small native transport producer. This never issues original-source authority."""
from __future__ import annotations

import argparse
from pathlib import Path

from prismaquant.native_results import publish_result

PAYLOAD = b'PQ native result fixture\n\x00\xff'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--publication', type=Path, required=True)
    parser.add_argument('--late-failure', action='store_true')
    args = parser.parse_args()
    artifact = args.publication.parent / 'fixture.bin'
    artifact.write_bytes(PAYLOAD)
    publish_result(args.publication, producer_id='pq-native-cpu-fixture',
        inputs={'control': {'path': str(args.input), 'sha256': args.input_sha256,
                            'bytes': len(b'explicit fixture input\n')}},
        artifacts={'fixture': artifact}, max_artifact_bytes=4096)
    if args.late_failure:
        print('publication completed; actual execution fails afterward', flush=True)
        return 7
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
