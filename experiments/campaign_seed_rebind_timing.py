"""Time the seed-rebind of one completed campaign row, through the real CLI.

A PrismaQuant source change moves ``prismaquant_source_sha256``, so a row
priced under the old source cannot resume its own journal; its rows are
offered to ``--seed-checkpoint`` one at a time and re-verified against this
run's inputs (``_adopt_seed_checkpoint``).  This runs exactly that on one
completed row -- the row's own argv with its outputs redirected, its original
checkpoint and wire directory as the seed, and the rollout configuration's
publication/identity flags -- and records where the time goes: the resident
prefetch, the identity bind, the adoption (per-anchor identity and wire
receipt re-verification, wire re-read bytes), and the final journal and cost
write.  Nothing is encoded when every anchor is adopted, so the elapsed time
is the rebind cost a rollout pays per row.

``prepare`` sizes the action through the dispatcher's own plan for the row
with the extra flags; ``run`` executes it inside the admitted action.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

from experiments.campaign_publication_ab import PhaseRecorder, arm_flags
from experiments.selected_snapshot_scope_ab import digest, write

SCHEMA = 'prismaquant.campaign_seed_rebind_timing.v1'


def prepare(spec_path, workspace, row_id, out, *, overlap_bytes, identity_bytes):
    from tools.dispatch_tessera_campaign import load_spec, _streamed_resource_plan
    spec = load_spec(spec_path)
    workspace, out = Path(workspace), Path(out)
    plan = json.loads((workspace/'plan.json').read_text())
    index, row = next((i, row) for i, row in enumerate(plan['rows']) if row['row_id'] == row_id)
    census = json.loads(Path(plan['census']).read_text())
    manifest = json.loads(Path(plan['manifest']).read_text())
    original = manifest[index]['argv']
    command = original[original.index('prismaquant.tessera_campaign')+1:]
    if '--seed-checkpoint' in command or '--publication-overlap-bytes' in command:
        raise ValueError('rebind timing requires the original fresh synchronous recipe')
    checkpoint = Path(command[command.index('--checkpoint')+1])
    if not checkpoint.is_file() or not checkpoint.with_name(checkpoint.name+'.parts').is_dir():
        raise ValueError(f'{row_id} has no completed checkpoint to rebind')
    cache_dir = Path(command[command.index('--cache-dir')+1])
    selected = json.loads(json.dumps(spec))
    selected['campaign_argv'] += arm_flags(overlap_bytes, identity_bytes)
    resources = _streamed_resource_plan(selected, census, row['members'], selected_source=True)
    memory = math.ceil((resources['memory_bytes']+plan['process_baseline_bytes'])/1024**3)
    if memory > spec['box_memory_gb']:
        raise ValueError(f'rebind workload needs {memory} GiB, above the recipe budget')
    seed_units = sorted(p.name for p in (checkpoint.with_name(checkpoint.name+'.parts')/'units').iterdir())
    inputs = [Path(spec_path), workspace/'plan.json', Path(plan['manifest']), Path(plan['census']),
              Path(row['units']), checkpoint]
    value = dict(schema=SCHEMA, command=command, row_id=row_id, groups=row['groups'],
        seed_checkpoint=str(checkpoint), seed_wire_dir=str(cache_dir/'wire'),
        seed_unit_shards=len(seed_units), seed_wire_files=sum(1 for _ in (cache_dir/'wire').iterdir()),
        seed_wire_bytes=sum(p.stat().st_size for p in (cache_dir/'wire').iterdir() if p.is_file()),
        input_sha256={str(path): digest(path) for path in inputs}, resources=resources,
        overlap_bytes=int(overlap_bytes), identity_bytes=int(identity_bytes),
        environment=spec['env'], container=spec['container'], requested_cpus=spec['cpus'],
        requested_memory_gib=memory, out=str(out/'native'))
    out.mkdir(parents=True, exist_ok=True)
    write(out/'plan.json', value)
    print(json.dumps(dict(plan=str(out/'plan.json'), sha256=digest(out/'plan.json'),
        memory_gib=memory, seed_unit_shards=len(seed_units),
        seed_wire_bytes=value['seed_wire_bytes'])), flush=True)


def _proc_io():
    try:
        return {line.split(':')[0]: int(line.split(':')[1]) for line in Path('/proc/self/io').read_text().splitlines()}
    except OSError:
        return {}


def run(path, expected):
    if digest(path) != expected:
        raise ValueError('rebind plan changed')
    p = json.loads(Path(path).read_text())
    if p['schema'] != SCHEMA:
        raise ValueError('unknown rebind timing schema')
    for name, sha in p['input_sha256'].items():
        if digest(name) != sha:
            raise ValueError('rebind input changed: '+name)
    for name, value in p['environment'].items():
        if os.environ.get(name) != value:
            raise ValueError('rebind environment changed: '+name)
    import torch
    from prismaquant import tessera_campaign as campaign
    from prismaquant import cost_stage_checkpoint
    if not torch.cuda.is_available():
        raise RuntimeError('rebind timing requires an admitted CUDA device')
    out = Path(p['out']); out.mkdir(parents=True, exist_ok=False)
    command = list(p['command'])
    for flag, value in {'--out': out/'cost.pkl', '--cache-dir': out/'cache',
            '--checkpoint': out/'cost.anchors.json'}.items():
        command[command.index(flag)+1] = str(value)
    command += ['--seed-checkpoint', p['seed_checkpoint'], '--seed-wire-dir', p['seed_wire_dir']]
    command += arm_flags(p['overlap_bytes'], p['identity_bytes'])
    recorder = PhaseRecorder(limit=200000)
    targets = [(campaign, name) for name in (
        '_prefetch_selected_capture', '_campaign_bound_identities', '_adopt_seed_checkpoint',
        '_checkpoint_anchor_identity', '_checkpoint_wire_record', '_link_seed_wire',
        '_campaign_checkpoint_identity', 'campaign_cost_payload')] + [
        (campaign._BoundCheckpointUnitIdentity, 'derive'),
        (cost_stage_checkpoint, 'write_unit'), (cost_stage_checkpoint, '_load_unit')]
    originals = [(owner, name, getattr(owner, name)) for owner, name in targets]
    io_before = _proc_io()
    started = time.time(); wall = time.perf_counter()
    returncode = None
    try:
        for owner, name, original in originals:
            setattr(owner, name, recorder.wrap(original, owner.__name__+'.'+name))
        returncode = campaign.main(command)
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)
        elapsed = time.perf_counter() - wall
        io_after = _proc_io()
        totals = {}
        for record in recorder.records:
            entry = totals.setdefault(record['phase'], dict(calls=0, seconds=0.0, thread_cpu_seconds=0.0,
                threads=set(), first_started_unix=record['started_unix'], last_finished_unix=0.0))
            entry['calls'] += 1
            entry['seconds'] += record['seconds']
            entry['thread_cpu_seconds'] += record['thread_cpu_seconds']
            entry['threads'].add(record['thread'])
            entry['first_started_unix'] = min(entry['first_started_unix'], record['started_unix'])
            entry['last_finished_unix'] = max(entry['last_finished_unix'], record['started_unix']+record['seconds'])
        for entry in totals.values():
            entry['threads'] = sorted(entry['threads'])
        summary = dict(schema='prismaquant.campaign_seed_rebind_timing_result.v1', plan_sha256=expected,
            row_id=p['row_id'], returncode=returncode, started_unix=started, wall_seconds=elapsed,
            command=command, phases=totals, process_io_delta={k: io_after.get(k, 0)-io_before.get(k, 0)
                for k in set(io_before) | set(io_after)},
            seed_wire_bytes=p['seed_wire_bytes'], seed_unit_shards=p['seed_unit_shards'],
            cuda_reserved_peak_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else None,
            span_scope='Inclusive nested wall and thread CPU times per phase; adoption contains the per-anchor identity and receipt spans; do not sum across phases.')
        write(out/'rebind-timing.json', summary)
        write(out/'phase-profile.json', dict(schema='prismaquant.publication_phase_profile.v1',
            spans='Inclusive nested wall and thread CPU times; no added CUDA synchronization.',
            records=recorder.records))
        print(json.dumps({k: v for k, v in summary.items() if k not in ('command', 'phases')}), flush=True)
        for phase, entry in sorted(totals.items(), key=lambda kv: -kv[1]['seconds']):
            print(json.dumps(dict(phase=phase, **entry)), flush=True)
    if returncode != 0:
        raise RuntimeError(f'rebind campaign returned {returncode}')
    manifest = json.loads((out/'cost.anchors.json').read_text())
    parts = out/'cost.anchors.json.parts'/'units'
    adopted = sum(1 for _ in parts.iterdir())
    if adopted != p['seed_unit_shards']:
        raise RuntimeError(f'rebind journalled {adopted} units of {p["seed_unit_shards"]} seeded')
    write(out/'result.json', dict(schema=SCHEMA, plan_sha256=expected, adopted_units=adopted,
        identity_sha256=manifest['identity_sha256'], wall_seconds=elapsed,
        scope='One completed row re-verified through --seed-checkpoint under a new package source; no anchor encoded.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--spec', required=True); prep.add_argument('--workspace', required=True)
    prep.add_argument('--row-id', required=True); prep.add_argument('--out', required=True)
    prep.add_argument('--overlap-bytes', type=int, default=268435456)
    prep.add_argument('--identity-bytes', type=int, default=268435456)
    execute = sub.add_parser('run')
    execute.add_argument('--plan', required=True); execute.add_argument('--plan-sha256', required=True)
    args = parser.parse_args()
    if args.mode == 'prepare':
        prepare(args.spec, args.workspace, args.row_id, args.out,
                overlap_bytes=args.overlap_bytes, identity_bytes=args.identity_bytes)
    else:
        run(args.plan, args.plan_sha256)


if __name__ == '__main__':
    main()
