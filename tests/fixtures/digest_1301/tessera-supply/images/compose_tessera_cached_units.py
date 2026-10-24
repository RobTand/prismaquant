# Original source fragments from 6d8288adc94b3c4b624c143ed372fb6513f32784:tools/compose_tessera_cached_units.py
# Domain dependencies are injected by the behavioral tests.
from __future__ import annotations
import json
import hashlib
from pathlib import Path

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--child-v1', nargs=4, action='append', default=[],
                        metavar=('MANIFEST', 'SHA256', 'PACKAGE', 'PACKAGE_SHA256'))
    parser.add_argument('--child-v2', nargs=2, action='append', default=[],
                        metavar=('MANIFEST', 'SHA256'))
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    children = [{'manifest': {'path': path, 'sha256': sha},
                 'producer_package': {'path': package, 'sha256': seal}}
                for path, sha, package, seal in args.child_v1]
    children.extend({'manifest': {'path': path, 'sha256': sha},
                     'producer_package': None} for path, sha in args.child_v2)
    output = Path(args.out)
    manifest, bundle = compose(children, output=output)
    raw = (json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                      allow_nan=False) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not publish_new_bytes(output, raw):
        raise FileExistsError(f'cached composition output exists: {output}')
    print(DIRECT_ASCII_SPACED_LAX.text({'schema': 'prismaquant.cached_unit_composition_handoff.v1',
                      'manifest': str(output.resolve()),
                      'manifest_sha256': hashlib.sha256(raw).hexdigest(),
                      'children': bundle.child_manifests,
                      'units': len(bundle.units),
                      'encoder_source_proof_mode': bundle.encoder_source_proof_mode,
                      'warnings': bundle.warnings,
                      'export_qualified': False, 'serving_qualified': False}))
    return 0
