"""Dispatch a complete Tessera serving export as PrismaBuild layer quanta.

The caller supplies the whole source, plan, scales and pinned encoder and
image. The worker (``model_worker.py``, staged as ``worker.py``) derives
partition ownership from the producer's contract; this driver submits each
stage through ``pbcampaign``, requires every action of a stage to finish, and
only then admits assembly behind the complete-set barrier.

Moved from PrismaBuild's ``tools/fleet/dispatch_tessera_model.py`` on
2026-09-28 (RobTand/prismabuild#1076). Two things changed with the move,
because PrismaQuant reaches PrismaBuild only through its public commands:

- **Preparation hosts are named.** The source identity is prepared once on
  every host an encode may land on, and the hosts must agree. The old driver
  read PrismaBuild's live worker offers to find them; this one takes them as
  ``--prepare-host``, repeated, and seals the list into the export contract.
- **Results come from the worker's own records.** Each prepare action writes
  ``<out>.prepare/<host>.json`` and each part and the assembly write
  ``pb-result.json`` beside their bytes. The assembly action re-hashes every
  part against the barrier inside its admitted slot, as before.

Example::

    python3 -m tools.tessera_fleet.dispatch_model --workspace W \\
        --source S --plan P --encoder-checkout /home/rob/tessera \\
        --encoder-revision <40-hex> --image repo@sha256:<64-hex> --out O \\
        --prepare-host sparky --prepare-host sparklina
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

from tools.tessera_fleet import common
from tools.tessera_fleet import model_worker as model

WORKER = Path(model.__file__).resolve()


def stage_checkout(encoder, revision, workspace, plan, scales):
    workspace = Path(workspace)
    if workspace.exists():
        raise ValueError('workspace already exists; use --resume for an existing export')
    commit = subprocess.check_output(['git', '-C', str(encoder), 'rev-parse', revision + '^{commit}'], text=True).strip()
    if commit != revision:
        raise ValueError('--encoder-revision must be the full immutable commit ID')
    archive = subprocess.check_output(['git', '-C', str(encoder), 'archive', commit])
    workspace.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(workspace / 'encoder', filter='data')
    shutil.copyfile(WORKER, workspace / 'worker.py')
    partitioner = Path(__file__).resolve().parents[2] / 'prismaquant' / 'export_partition.py'
    shutil.copyfile(partitioner, workspace / 'export_partition.py')
    shutil.copyfile(plan, workspace / 'plan.json')
    if not isinstance(model.read_json(workspace / 'plan.json'), dict):
        raise ValueError('plan must be a JSON object')
    if scales:
        shutil.copyfile(scales, workspace / 'scales.safetensors')
    # pbrun snapshots tracked and untracked bytes. Nothing reaches main here.
    common.initialize_git(workspace, message='PrismaQuant sealed producer workspace')
    return commit


def campaign_row(workspace, spec, command, *, index=None, host=None):
    gpu = command == 'encode'
    argv = ['/usr/bin/python3', 'worker.py', command]
    if index is not None:
        argv += ['--index', str(index)]
    if host is not None:
        argv += ['--host', host]
    return {'cwd': str(workspace), 'argv': argv,
            'demand': {'cpu': spec['cpus'] if gpu else 1,
                       'mem_gb': spec['mem_gb'] if gpu else spec['assembly_mem_gb'],
                       **({'gpu': 1} if gpu else {})},
            'tags': sorted(set(spec['tags'] + ([host] if host else []))),
            'env': {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'},
            'retry_safe': True}


def prepare_source_identity(spec, workspace, wait_s):
    """Prepare the source identity on every named host, and require agreement."""
    rows = [campaign_row(workspace, spec, 'prepare', host=host) for host in spec['prepare_hosts']]
    common.run_stage(rows, workspace, 'prepare', wait_s=wait_s)
    prepared = [model.read_json(model.prepared_path(spec, host)) for host in spec['prepare_hosts']]
    if any(row != prepared[0] for row in prepared[1:]):
        raise ValueError('preparation hosts disagree on immutable source identity')
    return prepared[0]


def part_records(spec):
    records = []
    for index in range(spec['count']):
        record = model.read_json(Path(spec['parts']) / f'part-{index:05d}' / 'pb-result.json')
        if record.get('index') != index or record.get('contract') != spec['contract']:
            raise ValueError(f'part {index}: record belongs to another export')
        records.append(record)
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--source', type=Path, help='the complete Tessera source export to dispatch')
    parser.add_argument('--plan', type=Path, help='the layer plan the campaign partitions into quanta')
    parser.add_argument('--input-scales', type=Path, help='optional pre-computed scales to seal into the workspace')
    parser.add_argument('--encoder-checkout', type=Path, help='Git checkout the pinned encoder is archived from')
    parser.add_argument('--encoder-revision', help='full immutable commit ID of the encoder to pin')
    parser.add_argument('--image', help='qualified producer repository@sha256 digest')
    parser.add_argument('--out', type=Path, help='shared path the assembled model is written to')
    parser.add_argument('--workspace', type=Path, help='directory for the dispatch job, receipts and state')
    parser.add_argument('--resume', action='store_true', help='continue the dispatch already sealed in --workspace')
    parser.add_argument('--prepare-host', action='append', dest='prepare_hosts', default=None,
                        help='placement tag of a host an encode may run on; repeat for each. '
                             'The source identity is prepared on every one and must agree')
    parser.add_argument('--cpus', type=int, default=1, help='CPU reservation for each layer quantum')
    parser.add_argument('--mem-gb', type=int, default=16, help='memory reservation for each layer quantum, in GB')
    parser.add_argument('--assembly-mem-gb', type=int, default=4, help='memory reservation for the assembly action, in GB')
    parser.add_argument('--tag', action='append', dest='tags', default=None, help='placement tag; repeatable, defaults to gb10')
    parser.add_argument('--grid', default='E4M3', help='quantization grid the encoder is run with')
    parser.add_argument('--q256', type=int, default=1024, help='quanta per 256 rows the plan is partitioned at')
    parser.add_argument('--wait-s', type=float, default=86400., help="seconds to wait for a stage's actions before giving up")
    parser.add_argument('--dry-run', action='store_true',
                        help='derive source partitions and construction-census inputs; do not stage or submit')
    args = parser.parse_args(argv)
    if min(args.cpus, args.mem_gb, args.assembly_mem_gb) < 1:
        parser.error('resource reservations must be positive')
    if args.dry_run:
        if args.resume or args.source is None or args.plan is None:
            parser.error('--dry-run requires --source and --plan, without --resume')
        from prismaquant.tessera_export_lane import export_setup
        setup = export_setup(args.source, model.read_json(args.plan))
        spec = {'cpus': args.cpus, 'mem_gb': args.mem_gb,
                'assembly_mem_gb': args.assembly_mem_gb, 'tags': args.tags or ['gb10']}
        setup['encode_rows'] = [campaign_row(args.workspace or Path('<workspace>'), spec,
                                            'encode', index=part['index'])
                                for part in setup['partitions']]
        setup['execution_inputs'] = {'encoder_revision': args.encoder_revision,
                                     'image': args.image, 'out': str(args.out) if args.out else None}
        print(json.dumps(setup, indent=2))
        return 0
    if args.workspace is None:
        parser.error('--workspace is required for execution or resume')
    workspace = args.workspace.resolve()
    if args.resume:
        if any(getattr(args, name) is not None for name in ('source', 'plan', 'input_scales', 'encoder_checkout', 'encoder_revision', 'image', 'out', 'prepare_hosts')):
            parser.error('--resume uses the sealed workspace; input overrides require a new workspace')
        spec = model.read_json(workspace / 'job.json')
    else:
        for name, flag in (('source', '--source'), ('plan', '--plan'),
                           ('encoder_checkout', '--encoder-checkout'),
                           ('encoder_revision', '--encoder-revision'), ('image', '--image'),
                           ('out', '--out'), ('prepare_hosts', '--prepare-host')):
            if getattr(args, name) is None:
                parser.error(flag + ' is required')
        if not re.fullmatch(r'[^\s]+@sha256:[0-9a-f]{64}', args.image):
            parser.error('--image must be an immutable repository@sha256 digest')
        if any(Path(host).name != host or not host for host in args.prepare_hosts):
            parser.error('--prepare-host must be a bare placement tag')
        for path in (args.source.resolve(), args.out.resolve()):
            if not path.is_relative_to('/mnt/shared') or ':' in str(path):
                parser.error('source and output must be shared paths below /mnt/shared without colons')
        if args.out.exists():
            parser.error('--out already exists')
        commit = stage_checkout(args.encoder_checkout, args.encoder_revision, workspace,
                                args.plan, args.input_scales)
        spec = {'schema': model.SCHEMA, 'source': str(args.source.resolve()),
                'out': str(args.out.resolve()), 'parts': str(args.out.resolve()) + '.parts',
                'encoder_commit': commit, 'image': args.image,
                'adapter_sha256': model.digest_file(workspace / 'worker.py'),
                'plan_sha256': model.digest_file(workspace / 'plan.json'),
                'scales': model.digest_file(workspace / 'scales.safetensors') if args.input_scales else None,
                'cpus': args.cpus, 'mem_gb': args.mem_gb, 'assembly_mem_gb': args.assembly_mem_gb,
                'tags': args.tags or ['gb10'], 'grid': args.grid, 'q256': args.q256,
                'prepare_hosts': sorted(set(args.prepare_hosts))}
        model.atomic_json(workspace / 'job.json', spec)
    if 'contract' not in spec:
        spec.update(prepare_source_identity(spec, workspace, args.wait_s))
        spec['contract'] = model.digest_json(spec)
        model.atomic_json(workspace / 'job.json', spec)
    print(f"export {spec['contract']}: {spec['count']} whole-layer actions", flush=True)
    rows = [campaign_row(workspace, spec, 'encode', index=index) for index in range(spec['count'])]
    common.run_stage(rows, workspace, 'encode', wait_s=args.wait_s)
    # Full payload hashing runs in the admitted assembler, never as an
    # unreserved large validation stage on the submitting machine.
    parts = part_records(spec)
    assembly = workspace / common.STATE / 'assembly'
    if not assembly.exists():
        shutil.copytree(workspace, assembly, ignore=shutil.ignore_patterns('.git', common.STATE))
        common.initialize_git(assembly, message='PrismaQuant assembly barrier')
    model.atomic_json(assembly / 'barrier.json', parts)
    common.run_stage([campaign_row(assembly, spec, 'assemble')], workspace, 'assemble', wait_s=args.wait_s)
    result = model.read_json(Path(spec['out']) / 'pb-result.json')
    if result.get('contract') != spec['contract'] or result.get('index') is not None:
        raise ValueError('assembled output record belongs to another export')
    model.atomic_json(workspace / common.STATE / 'export-result.json', result)
    print(json.dumps({'out': spec['out'], 'contract': spec['contract'], 'workspace': str(workspace)}))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f'dispatch_model: {exc}')
