# Original source fragments from 6d8288adc94b3c4b624c143ed372fb6513f32784:tools/build_tessera_mtp_cached_units.py
# Domain dependencies are injected by the behavioral tests.
from __future__ import annotations
import json
import hashlib
from pathlib import Path

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assignment', required=True)
    parser.add_argument('--assignment-sha256', required=True)
    parser.add_argument('--mtp-joint-cost', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    output = Path(args.out)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'MTP child output exists: {output}')
    raw = Path(args.assignment).read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.assignment_sha256:
        raise ValueError('MTP selected assignment SHA-256 differs')
    manifest = selected_mtp_child(
        json.loads(raw), cost_path=args.mtp_joint_cost, output=output)
    encoded = (json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                          allow_nan=False) + '\n').encode()
    if not publish_new_bytes(output, encoded):
        raise FileExistsError(f'MTP child output exists: {output}')
    print(json.dumps({'schema': 'prismaquant.mtp_cached_child_handoff.v1',
                      'manifest': str(output.resolve()),
                      'manifest_sha256': hashlib.sha256(encoded).hexdigest(),
                      'assignment_sha256': args.assignment_sha256,
                      'encoder_source_sha256': next(iter({
                          record['identity']['encoder_source_sha256']
                          for record in manifest['units'].values()})),
                      'units': len(manifest['units']),
                      'root': str(output.parent.resolve()),
                      'export_qualified': False, 'serving_qualified': False},
                     sort_keys=True))
    return 0
