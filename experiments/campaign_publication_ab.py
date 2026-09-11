"""Profile publication overlap on a bounded prefix of the real campaign.

Both arms retain the complete selected group's resident inputs. The explicit
batch-order permutation puts the requested shape first; no new encoder,
cache, calibration traversal, or fleet placement is implemented here.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import functools
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

from experiments.campaign_batch_prefix_ab import prefix_state
from experiments.selected_snapshot_scope_ab import digest, write


SCHEMA_V1 = 'prismaquant.campaign_publication_ab.v1'
SCHEMA_V2 = 'prismaquant.campaign_publication_ab.v2'
OVERLAP_BUDGET = 256*1024**2


def arm_flags(overlap, identity):
    """The campaign flags one arm adds to the original synchronous recipe."""
    flags = ['--publication-overlap-bytes', str(int(overlap))]
    if int(identity) > 0:
        flags += ['--campaign-identity-bytes', str(int(identity))]
    return flags


def plan_order(plan):
    """Every arm as an ``(overlap_bytes, identity_bytes)`` pair, v1 included."""
    if plan['schema'] == SCHEMA_V1:
        return [(int(budget), 0) for budget in plan['order']]
    if plan['schema'] != SCHEMA_V2:
        raise ValueError('unknown publication comparison schema')
    order = [tuple(int(v) for v in pair) for pair in plan['order']]
    if any(len(pair) != 2 or min(pair) < 0 for pair in order):
        raise ValueError('a v2 arm is an (overlap_bytes, identity_bytes) pair')
    return order


def arm_order(identity_bytes=0, override=None):
    """The arm list: the default matrix, or the caller's repeat of chosen arms.

    A repeat (e.g. one more pipelined arm on a quiet server) has its parity
    checked offline against the original run's synchronous arm, since ``run``
    compares only within one plan.
    """
    identity = int(identity_bytes)
    if identity < 0:
        raise ValueError('identity bytes cannot be negative')
    if override:
        order = [tuple(int(v) for v in pair) for pair in override]
        if any(len(pair) != 2 or min(pair) < 0 for pair in order):
            raise ValueError('an arm is an (overlap_bytes, identity_bytes) pair')
        return order
    budget = OVERLAP_BUDGET
    return ([(0, 0), (budget, 0), (budget, identity), (budget, identity), (budget, 0)]
            if identity else [(0, 0), (budget, 0), (budget, 0), (0, 0)])


def prepare(spec_path, workspace, row_id, out, *, identity_bytes=0, order=None):
    """Write the plan.  ``identity_bytes`` > 0 adds the pipelined arms.

    v1 order was ``[0, B, B, 0]`` (synchronous, overlap, overlap,
    synchronous).  With an identity reservation the order is
    ``[(0,0), (B,0), (B,I), (B,I), (B,0)]``: the synchronous parity arm, the
    overlap-only baseline, the pipelined arm twice, and the baseline again so
    the box's drift across the run brackets both.  Memory is computed for
    each distinct pair through the same dispatcher plan a row is admitted on.
    ``order`` overrides the arm list for a repeat run.
    """
    from tools.dispatch_tessera_campaign import load_spec, _streamed_resource_plan
    spec = load_spec(spec_path)
    workspace, out = Path(workspace), Path(out)
    plan = json.loads((workspace/'plan.json').read_text())
    index, row = next((i, row) for i, row in enumerate(plan['rows']) if row['row_id'] == row_id)
    if len(row['groups']) != 1 or not row['groups'][0].endswith('.mlp.experts'):
        raise ValueError('comparison requires one complete routed expert group')
    census = json.loads(Path(plan['census']).read_text())
    manifest = json.loads(Path(plan['manifest']).read_text())
    original = manifest[index]['argv']
    command = original[original.index('prismaquant.tessera_campaign')+1:]
    if '--publication-overlap-bytes' in command or '--seed-checkpoint' in command:
        raise ValueError('comparison requires the original fresh synchronous recipe')
    resources = {}
    order = arm_order(identity_bytes, order)
    for pair in sorted(set(order)):
        selected = copy.deepcopy(spec)
        selected['campaign_argv'] += arm_flags(*pair)
        resources['%d:%d' % pair] = _streamed_resource_plan(selected, census, row['members'], selected_source=True)
    # Existing measured process baseline plus a bounded CUDA trace and its
    # decoded profiler objects. This allowance is separate from staged bytes.
    workload_memory = math.ceil((max(r['memory_bytes'] for r in resources.values())+
        plan['process_baseline_bytes'])/1024**3)
    if workload_memory > spec['box_memory_gb']:
        raise ValueError(f'comparison workload needs {workload_memory} GiB, above the recipe budget')
    memory = workload_memory+2
    inputs = [Path(spec_path), workspace/'plan.json', Path(plan['manifest']),
        Path(plan['census']), Path(row['units']),
        Path(command[command.index('--calibration-cache')+1])]
    value = dict(schema=SCHEMA_V2, command=command,
        input_sha256={str(path): digest(path) for path in inputs}, resources=resources,
        row_id=row_id, groups=row['groups'], expected_source_units=864,
        limit_anchors=64, projection='gate_up', order=[list(pair) for pair in order],
        environment=spec['env'], container=spec['container'],
        requested_cpus=spec['cpus'], requested_memory_gib=memory, workload_memory_gib=workload_memory,
        out=str(out/'native'), profiler_allowance_gib=2)
    out.mkdir(parents=True, exist_ok=True)
    write(out/'plan.json', value)
    print(json.dumps(dict(plan=str(out/'plan.json'), sha256=digest(out/'plan.json'),
        memory_gib=memory, order=order,
        resident_anchors_bytes={k: sum(v['phases']['resident_anchors'].values()) for k, v in resources.items()},
        publication_staging_bytes={k:v['phases']['resident_anchors']['publication_staging_bytes']
        for k,v in resources.items()},
        campaign_identity_metadata_bytes={k:v['phases']['resident_anchors'].get('campaign_identity_metadata_bytes', 0)
        for k,v in resources.items()})), flush=True)


def preferred_batches(batches, projection):
    """Stable permutation of whole compatible batches, preserving membership."""
    if projection not in ('down_proj', 'gate_up'):
        raise ValueError('unknown measured projection class')
    def matches(batch):
        classes = {item[0].rsplit('.', 1)[-1] for item in batch}
        if not classes <= {'down_proj', 'gate_proj', 'up_proj'}:
            raise ValueError('publication benchmark requires routed expert projections')
        wanted = {'down_proj'} if projection == 'down_proj' else {'gate_proj', 'up_proj'}
        if classes & wanted and not classes <= wanted:
            raise ValueError('one producer batch crosses the requested shape classes')
        return classes <= wanted
    marked = [(matches(batch), batch) for batch in batches]
    if not any(match for match, _ in marked):
        raise ValueError('requested projection has no compatible batch')
    return [batch for match, batch in marked if match] + [batch for match, batch in marked if not match]


class PhaseRecorder:
    """Nested wall/CPU spans on both the encoding and publication threads.

    Spans are inclusive, may overlap, and must not be summed across threads.
    No CUDA synchronizations are inserted. Existing blocking boundaries retain
    their original semantics. Records flush only after the measured traversal.
    """
    def __init__(self, limit=100000):
        self.records = []
        self.limit = limit

    def wrap(self, function, name):
        @functools.wraps(function)
        def measured(*args, **kwargs):
            start = time.time()
            wall = time.perf_counter()
            cpu = time.thread_time()
            try:
                return function(*args, **kwargs)
            finally:
                record = dict(phase=name, started_unix=start,
                    seconds=time.perf_counter()-wall, thread_cpu_seconds=time.thread_time()-cpu,
                    thread=threading.current_thread().name, qname=kwargs.get('qname'),
                    format_name=kwargs.get('format_name'))
                if len(self.records) >= self.limit:
                    raise RuntimeError('phase profile exceeded its bounded record budget')
                self.records.append(record)
        return measured


@contextmanager
def instrument(campaign, recorder, projection):
    from prismaquant import production_weight_cache, perturbed_x_cache, cost_stage_checkpoint
    from prismaquant.tessera_publication import BoundedPublisher
    targets = [
        (campaign, name) for name in ('_checkpoint_anchor_identity', '_checkpoint_wire_record', '_finish_anchor',
                                      '_campaign_bound_identities', '_adopt_seed_checkpoint')
    ] + [
        # The per-anchor derivation off a sealed template.  Its thread name
        # in the record is the evidence of where the post-work ran.
        (campaign._BoundCheckpointUnitIdentity, 'derive'),
        (production_weight_cache, '_store_rendered_weight_entry'),
        (production_weight_cache, '_canonical_rendered_weight_tensor'),
        (production_weight_cache, '_local_forward_render_score'),
        (perturbed_x_cache, 'release_activation_cache_file_pages'),
        (cost_stage_checkpoint, 'write_unit'),
    ]
    from prismaquant import tessera_hessian
    targets.append((tessera_hessian, 'encoder_kwargs'))
    originals = [(owner, name, getattr(owner, name)) for owner, name in targets]
    batches = campaign._anchor_batches
    close = BoundedPublisher.close
    memo = campaign._activation_kwargs_memo
    memos = []

    def memo_with_record(*a, **kw):
        # The encoder memo (RobTand/prismaquant#389): its hit/miss counts are
        # read at the end of the arm, so the profile carries the memo policy's
        # measured effect and not only the factorization spans.
        built = memo(*a, **kw)
        memos.append(built)
        return built

    def close_with_stats(self):
        # The prefix ends by exception, before the campaign would stamp the
        # publisher's counters into provenance; take them at the close the
        # unwind always performs.  ``submit_blocked_seconds`` is how long the
        # encode thread waited on the byte budget, which is the writer
        # backpressuring the encode, and it has to be reported if nonzero.
        try:
            stats = self.stats()
        except Exception as error:  # pragma: no cover - diagnostic only
            stats = dict(error=repr(error))
        recorder.records.append(dict(phase='tessera_publication.BoundedPublisher.stats',
            started_unix=time.time(), seconds=0.0, thread_cpu_seconds=0.0,
            thread=threading.current_thread().name, qname=None, format_name=None, stats=stats))
        return close(self)

    try:
        for owner, name, original in originals:
            setattr(owner, name, recorder.wrap(original, owner.__name__+'.'+name))
        campaign._anchor_batches = lambda *a, **kw: preferred_batches(batches(*a, **kw), projection)
        campaign._activation_kwargs_memo = memo_with_record
        BoundedPublisher.close = close_with_stats
        yield
    finally:
        BoundedPublisher.close = close
        campaign._anchor_batches = batches
        campaign._activation_kwargs_memo = memo
        for owner, name, original in originals:
            setattr(owner, name, original)
        for built in memos:
            info = built.cache_info()
            recorder.records.append(dict(phase='tessera_campaign._activation_kwargs_memo.cache_info',
                started_unix=time.time(), seconds=0.0, thread_cpu_seconds=0.0,
                thread=threading.current_thread().name, qname=None, format_name=None,
                cache_info=dict(hits=info.hits, misses=info.misses, maxsize=info.maxsize, currsize=info.currsize)))


def profile_settings(plan):
    """Which anchor calls the observer traces and for how long.

    The default is the 0.25 s LDL window of call 0 the five-arm plan measured.
    A plan may carry a ``profile`` block -- ``calls`` (zero-based, must include
    0, at most four), ``window_seconds`` and ``trace_max_bytes`` -- so a later
    run can trace the head of a STEADY call (index 1 onward) long enough to
    resolve the early Viterbi chunks to kernel level, which the 0.25 s window
    could not.  Collection stays CUDA-only: the timed window is Kineto-wide.
    """
    settings = dict(profile_calls=[0], trace_max_bytes=512*1024**2, window_seconds=0.25)
    block = plan.get('profile')
    if block is None:
        return settings
    if not isinstance(block, dict) or set(block) - {'calls', 'window_seconds', 'trace_max_bytes'}:
        raise ValueError('plan profile block carries calls, window_seconds, trace_max_bytes only')
    if 'calls' in block:
        settings['profile_calls'] = [int(i) for i in block['calls']]
    if 'window_seconds' in block:
        settings['window_seconds'] = float(block['window_seconds'])
    if 'trace_max_bytes' in block:
        settings['trace_max_bytes'] = int(block['trace_max_bytes'])
    return settings


def arm(plan, out, budget, identity=0):
    import torch
    from experiments.campaign_prefix_profile import run_prefix
    from experiments.glm_full_capture_profile import AnchorObserver, selected_anchor_command
    from prismaquant import tessera_campaign as campaign
    if not torch.cuda.is_available():
        raise RuntimeError('publication comparison requires an admitted CUDA device')
    command = list(plan['command'])
    for flag, value in {'--out': out/'cost.pkl', '--cache-dir': out/'cache',
            '--checkpoint': out/'cost.anchors.json'}.items():
        command[command.index(flag)+1] = str(value)
    command += arm_flags(budget, identity)
    selected_anchor_command(command)
    recorder = PhaseRecorder()
    observer = AnchorObserver(out/'profile', command=command, cuda_only=True,
                              **profile_settings(plan))
    observer.result['python_sampler']['interval_seconds'] = 0.2
    try:
        with observer, instrument(campaign, recorder, plan['projection']):
            run_prefix(campaign, command, observer, limit=plan['limit_anchors'],
                expected_source_units=plan['expected_source_units'])
            if observer.result['resident_prefetch']['devices'] != ['cuda:0']:
                raise RuntimeError('comparison inputs were not fully GPU resident')
    finally:
        write(out/'phase-profile.json', dict(schema='prismaquant.publication_phase_profile.v1',
            spans='Inclusive nested wall and thread CPU times; no added CUDA synchronization.',
            records=recorder.records))


def run(path, expected):
    if digest(path) != expected:
        raise ValueError('publication plan changed')
    p = json.loads(Path(path).read_text())
    order = plan_order(p)
    for name, sha in p['input_sha256'].items():
        if digest(name) != sha:
            raise ValueError('publication input changed: '+name)
    for name, value in p['environment'].items():
        if os.environ.get(name) != value:
            raise ValueError('publication environment changed: '+name)
    root = Path(p['out']); root.mkdir(parents=True, exist_ok=False)
    rows, baseline = [], None
    for index, (budget, identity) in enumerate(order):
        out = root/f'arm-{index:02d}'; out.mkdir()
        command = [sys.executable, '-u', '-m', 'experiments.campaign_publication_ab', '--plan', str(path),
            '--plan-sha256', expected, '--arm-out', str(out), '--budget', str(budget),
            '--identity-bytes', str(identity)]
        env = dict(os.environ, TRITON_CACHE_DIR=str(out/'triton'),
            TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'))
        started = time.time()
        with (out/'command.log').open('x') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = dict(index=index, publication_overlap_bytes=budget, campaign_identity_bytes=identity,
            started_unix=started,
            finished_unix=time.time(), returncode=result.returncode, command=command)
        rows.append(row); write(out/'exit.json', row)
        if result.returncode:
            raise RuntimeError(f'publication arm {index} failed')
        files = list((out/'profile').glob('attempt-*/result.json'))
        if len(files) != 1:
            raise ValueError('observer result absent or ambiguous')
        observed = json.loads(files[0].read_text())
        if (observed['status'] != 'complete' or not observed['prefix_boundary_reached']
                or observed['completed_anchor_units'] != p['limit_anchors']):
            raise ValueError('measured publication prefix incomplete')
        current = prefix_state(out, observed)
        if baseline is None:
            baseline = current
        elif current != baseline:
            raise ValueError('publication changed wire bytes, scoring, or checkpoint identity')
        row.update(exact_parity=True, anchor_units=p['limit_anchors'], observer_result=str(files[0]))
        write(out/'parity.json', row); print(json.dumps(row), flush=True)
    write(root/'result.json', dict(schema=p['schema'], plan_sha256=expected,
        rows=rows, exact_parity=True, campaign_completed=False,
        scope='Bounded reordered prefix; complete selected resident source and original capture; fresh process/compiler caches per arm.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--arm-out', type=Path)
    parser.add_argument('--budget', type=int)
    parser.add_argument('--identity-bytes', type=int, default=0)
    args = parser.parse_args()
    if args.arm_out:
        if digest(args.plan) != args.plan_sha256:
            raise ValueError('arm plan changed')
        arm(json.loads(Path(args.plan).read_text()), args.arm_out, args.budget, args.identity_bytes)
    else:
        run(args.plan, args.plan_sha256)


if __name__ == '__main__':
    main()
