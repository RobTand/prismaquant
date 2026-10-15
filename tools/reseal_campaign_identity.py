#!/usr/bin/env python3
"""Amend source-seal pins in Tessera campaign rows under an exact proof bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prismaquant import tessera_campaign_reseal as reseal


def cmd_hash_tree(args):
    print(json.dumps(reseal.hash_trees(prismaquant_root=args.prismaquant, producer_root=args.producer), indent=1))
    return 0


def cmd_dry_run(args):
    def display(plan):
        print(json.dumps(dict(row=plan['row'], kind=plan['kind'], journal_state=plan['journal_state'], state=plan['state'], strata=plan['strata'], shards=plan['shard_count'],
            receipt_seals=plan['receipt_seals'], cost_seals=plan['cost_seals'], bytes_to_write=plan['bytes_to_write'], old_identity_sha256=plan['old_identity_sha256'],
            new_identity_sha256=plan.get('new_identity_sha256'), plan_seconds=round(plan['plan_seconds'], 2))))
        if args.verbose:
            for edit in plan['edits']:
                print('  ' + json.dumps(edit, default=str))
    report = reseal.dry_run(args.pins, row_paths=args.row, workspace=args.workspace, allow_live=args.allow_live,
        checkpoint_audit=args.checkpoint_audit, proof_path=args.proof, on_row=display)
    if args.report:
        reseal.write_json(args.report, report)
    return 0


def cmd_migrate(args):
    def display(plan):
        if not plan.get('migrated'):
            print(json.dumps(dict(row=plan['row'], state=plan['state'], note=plan.get('note'))))
            return
        print(json.dumps(dict(row=plan['row'], kind=plan['kind'], journal_state=plan['journal_state'], shards=plan['shard_count'], receipt_seals=plan['receipt_seals'],
            cost_seals=plan['cost_seals'], bytes_written=plan['bytes_written'], seconds=round(plan['seconds'], 2), new_identity_sha256=plan['new_identity_sha256'])))
    report = reseal.migrate_rows(args.pins, proof_path=args.proof, row_paths=args.row, workspace=args.workspace, allow_live=args.allow_live,
        checkpoint_audit=args.checkpoint_audit, operator=args.operator, keep_previous=not args.discard_previous, on_row=display)
    if args.report:
        reseal.write_json(args.report, report)
    return 0


def cmd_verify(args):
    def display(result):
        print(json.dumps(dict(row=result['row'], ok=result['ok'], shards=result['shards'], wires=result['wires'], wire_bytes_checked=result['wire_bytes_checked'],
            seconds=round(result['seconds'], 2), failures=result['failures'][:3])))
    report = reseal.verify_rows(args.pins, row_paths=args.row, workspace=args.workspace, allow_live=args.allow_live,
        checkpoint_audit=args.checkpoint_audit, wire_root=args.wire_root, wire_bytes=not args.skip_wire_bytes,
        consumer_merge_dir=args.consumer_merge, on_row=display)
    if args.consumer_merge:
        key = 'consumer_merge_error' if 'error' in report['consumer_merge'] else 'consumer_merge'
        print(json.dumps({key: report['consumer_merge']['error'] if key.endswith('_error') else report['consumer_merge']}))
    if args.report:
        reseal.write_json(args.report, {k:v for k,v in report.items() if k != 'ok'})
    return 0 if report['ok'] else 1


def cmd_bundle(args):
    bundle = reseal.assemble_bundle(args.pins, arm_paths=args.arm, fixture_id=args.fixture_id, action_keys=args.action)
    reseal.write_json(args.out, bundle)
    print(json.dumps(dict(ok=bundle['ok'], cells=bundle['cell_count'], strata=bundle['strata'], missing=bundle['strata_missing'],
        engagement_missing=bundle['engagement_missing'], out=str(args.out))))
    return 0 if bundle['ok'] else 1


def _row_args(parser):
    parser.add_argument('--pins', required=True)
    parser.add_argument('--row', action='append')
    parser.add_argument('--workspace')
    parser.add_argument('--allow-live', action='store_true', help='allow rows inside the live campaign workspace')
    parser.add_argument('--checkpoint-audit', help='campaign checkpoint audit; rows it lists as withdrawn are partial journals')
    parser.add_argument('--report')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('hash-tree', help='compute the pins of a PrismaQuant checkout and/or a producer tree')
    p.add_argument('--prismaquant')
    p.add_argument('--producer')
    p = sub.add_parser('proof-bundle', help='assemble a proof bundle from the fleet arm results')
    p.add_argument('--pins', required=True)
    p.add_argument('--fixture-id', help='fixture-id arm result; required iff the encoder pin moves')
    p.add_argument('--arm', action='append', required=True)
    p.add_argument('--action', action='append', help='PB action key of an arm, for the record')
    p.add_argument('--out', required=True)
    p = sub.add_parser('dry-run', help='list every edit the migration would make')
    _row_args(p)
    p.add_argument('--proof')
    p.add_argument('--verbose', action='store_true')
    p = sub.add_parser('migrate', help='rewrite the pins under proof, one row at a time, atomically')
    _row_args(p)
    p.add_argument('--proof')
    p.add_argument('--operator')
    p.add_argument('--discard-previous', action='store_true', help='do not retain the pre-migration files beside the row')
    p = sub.add_parser('verify', help='re-read migrated rows the way their consumers do')
    _row_args(p)
    p.add_argument('--skip-wire-bytes', action='store_true')
    p.add_argument('--wire-root', help='row whose cache/wire holds the wires (defaults to the row itself)')
    p.add_argument('--consumer-merge', help='also merge verified campaign checkpoints into this directory')
    args = parser.parse_args(argv)
    try:
        return {'hash-tree': cmd_hash_tree, 'proof-bundle': cmd_bundle, 'dry-run': cmd_dry_run,
                'migrate': cmd_migrate, 'verify': cmd_verify}[args.command](args)
    except reseal.Refused as error:
        print(f'refused: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
