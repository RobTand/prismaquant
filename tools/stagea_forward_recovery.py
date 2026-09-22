"""Inspect or freeze a contained Stage-A forward recovery authority."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.joint_forward_resume import (
    SCHEMA, ForwardRecoveryRefused, _sdk, _read, _checked_group, _records,
    _require_imported_by_owner, build_forward_recovery, chain_documents,
    chained_specification, require_contained)


def inspect_owner(spec, spool_directory, *, sdk):
    """Check one owner's committed groups without freezing any authority.

    ``spec`` is either an operator specification or the chained segment
    derived from an imported capsule. Only groups in the segment's own range
    (``first_boundary``..``frontier``) are this owner's; a chained segment
    must also be the owner that imported its capsule, and every imported
    link is checked.
    """
    queue = sdk['pool'].PoolQueue(spec['queue_root'])
    instance = sdk['produced_output'].validate_instance(spec['instance'])
    try:
        require_contained(queue, instance, sdk)
        contained = True
    except ForwardRecoveryRefused:
        contained = False
    if 'imported' in spec:
        _require_imported_by_owner(spec, sdk)
    segments = len(chain_documents(spec))
    commitments = _read(sdk['produced_output'].instance_dir(queue.root, instance) /
                        'commitments.json')[0]
    first = spec.get('first_boundary', 0)
    entries, groups = [], 0
    for path in sorted(Path(spool_directory).glob('*/manifest.json')):
        manifest, _ = _read(path)
        match = re.match(r'stagea-boundary-b(\d+)-g', manifest['batch_id'])
        if (manifest['owner'] != instance['owner_action_key'] or not match or
                not first <= int(match[1]) <= spec['frontier']):
            continue
        group = {'manifest': manifest, 'manifest_raw': path.read_text(),
                 'record': _read(path.parent / 'export.json')[0],
                 'receipt': _read(path.parent / 'receipt.json')[0]}
        entries.extend(_checked_group(group, queue=queue, instance=instance,
            template=spec['template'], commitments=commitments, sdk=sdk))
        groups += 1
    _records(spec, entries)
    return {'schema': SCHEMA, 'inspect_only': True, 'authority_to_resume': False,
            'contained': contained, 'first_boundary': first, 'frontier': spec['frontier'],
            'groups': groups, 'entries': len(entries), 'segments': segments,
            'payload_bytes': sum(e['bytes'] for e in entries)}


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
        if not (args.import_capsule_sha256 and args.owner_request):
            parser.error('--import-capsule needs --import-capsule-sha256 and --owner-request')
        imported = {'path': args.import_capsule, 'sha256': args.import_capsule_sha256}
        if args.inspect_live:
            spec = chained_specification(imported, spool_directory=args.spool_directory,
                                         owner_request=args.owner_request)
            if args.frontier is not None:
                spec['frontier'] = args.frontier
            print(json.dumps(inspect_owner(spec, args.spool_directory, sdk=_sdk()),
                             sort_keys=True))
            return
        if not args.output:
            parser.error('--output is required to freeze recovery')
        print(json.dumps(build_forward_recovery(
            imported=imported, owner_request=args.owner_request,
            spool_directory=args.spool_directory, output=args.output,
            bind_current_implementation=args.bind_current_implementation,
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
    print(json.dumps(inspect_owner(spec, args.spool_directory, sdk=_sdk()), sort_keys=True))


if __name__ == '__main__':
    main()
