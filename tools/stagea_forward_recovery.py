"""Inspect or freeze a contained Stage-A forward recovery authority."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.joint_forward_resume import (
    SCHEMA, _sdk, _read, _checked_group, _records, build_forward_recovery)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--specification')
    source.add_argument('--import-capsule',
                        help='continue this contained recovery capsule (chained segment)')
    parser.add_argument('--import-capsule-sha256')
    parser.add_argument('--owner-request',
                        help="sealed PB request of the action that imported --import-capsule")
    parser.add_argument('--spool-directory', required=True)
    parser.add_argument('--output')
    parser.add_argument('--inspect-live', action='store_true')
    parser.add_argument('--bind-current-implementation', action='store_true',
                        help='explicitly bind this reviewed snapshot as the recovery implementation')
    parser.add_argument('--frontier', type=int)
    args = parser.parse_args()
    if args.import_capsule:
        if not (args.import_capsule_sha256 and args.owner_request and args.output):
            parser.error('--import-capsule needs --import-capsule-sha256, --owner-request '
                         'and --output')
        if args.inspect_live:
            parser.error('--inspect-live takes a --specification')
        print(json.dumps(build_forward_recovery(
            imported={'path': args.import_capsule, 'sha256': args.import_capsule_sha256},
            owner_request=args.owner_request, spool_directory=args.spool_directory,
            output=args.output, bind_current_implementation=args.bind_current_implementation,
            frontier=args.frontier), sort_keys=True))
        return
    spec = _read(args.specification)[0]
    if not args.inspect_live:
        if not args.output:
            parser.error('--output is required to freeze recovery')
        print(json.dumps(build_forward_recovery(specification=spec,
            spool_directory=args.spool_directory, output=args.output,
            bind_current_implementation=args.bind_current_implementation,
            frontier=args.frontier), sort_keys=True))
        return
    sdk = _sdk()
    queue = sdk['pool'].PoolQueue(spec['queue_root'])
    instance = sdk['produced_output'].validate_instance(spec['instance'])
    commitments = _read(sdk['produced_output'].instance_dir(queue.root, instance) /
                        'commitments.json')[0]
    entries, groups = [], 0
    for path in sorted(Path(args.spool_directory).glob('*/manifest.json')):
        manifest, _ = _read(path)
        import re
        match = re.match(r'stagea-boundary-b(\d+)-g', manifest['batch_id'])
        if manifest['owner'] != instance['owner_action_key'] or not match or int(match[1]) > spec['frontier']:
            continue
        group = {'manifest': manifest, 'manifest_raw': path.read_text(),
                 'record': _read(path.parent / 'export.json')[0],
                 'receipt': _read(path.parent / 'receipt.json')[0]}
        entries.extend(_checked_group(group, queue=queue, instance=instance,
            template=spec['template'], commitments=commitments, sdk=sdk))
        groups += 1
    _records(spec, entries)
    print(json.dumps({'schema': SCHEMA, 'inspect_only': True,
        'authority_to_resume': False, 'groups': groups, 'entries': len(entries),
        'frontier': spec['frontier'], 'payload_bytes': sum(e['bytes'] for e in entries)}, sort_keys=True))


if __name__ == '__main__':
    main()
