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
SCHEMA_V3 = 'prismaquant.campaign_publication_ab.v3'
OVERLAP_BUDGET = 256*1024**2
MEMORY_SAMPLE_SECONDS = 0.5


def arm_flags(overlap, identity):
    """The campaign flags one arm adds to the original synchronous recipe."""
    flags = ['--publication-overlap-bytes', str(int(overlap))]
    if int(identity) > 0:
        flags += ['--campaign-identity-bytes', str(int(identity))]
    return flags


def command_batch(command):
    """The anchor batch width the original recipe runs at."""
    command = list(command)
    width = int(command[command.index('--anchor-batch-size')+1])
    if width < 1:
        raise ValueError('the recipe batch width must be positive')
    return width


def with_batch(order, default):
    """Every arm as an ``(overlap_bytes, identity_bytes, batch)`` triple.

    A pair keeps the recipe's width; a triple names its own, so one plan can
    hold the same pipelined configuration at several widths and check them
    against each other in-plan.  The width is not part of the checkpoint
    identity (``tessera_campaign`` drops ``anchor_batch_size`` from the
    identity settings), so ``prefix_state`` compares across widths exactly.
    """
    triples = []
    for arm in order:
        arm = tuple(int(v) for v in arm)
        if len(arm) == 2:
            arm += (int(default),)
        if len(arm) != 3 or min(arm[:2]) < 0 or arm[2] < 1:
            raise ValueError('an arm is (overlap_bytes, identity_bytes[, anchor_batch_size])')
        triples.append(arm)
    return triples


def plan_order(plan):
    """Every arm as an ``(overlap_bytes, identity_bytes, batch)`` triple, v1/v2 included."""
    default = command_batch(plan['command'])
    if plan['schema'] == SCHEMA_V1:
        return with_batch([(int(budget), 0) for budget in plan['order']], default)
    if plan['schema'] == SCHEMA_V2:
        if any(len(pair) != 2 for pair in plan['order']):
            raise ValueError('a v2 arm is an (overlap_bytes, identity_bytes) pair')
        return with_batch(plan['order'], default)
    if plan['schema'] == SCHEMA_V3:
        if any(len(arm) != 3 for arm in plan['order']):
            raise ValueError('a v3 arm is an (overlap_bytes, identity_bytes, anchor_batch_size) triple')
        return with_batch(plan['order'], default)
    raise ValueError('unknown publication comparison schema')


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
        order = [tuple(int(v) for v in arm) for arm in override]
        if any(len(arm) not in (2, 3) or min(arm[:2]) < 0 or (len(arm) == 3 and arm[2] < 1)
               for arm in order):
            raise ValueError('an arm is (overlap_bytes, identity_bytes[, anchor_batch_size])')
        return order
    budget = OVERLAP_BUDGET
    return ([(0, 0), (budget, 0), (budget, identity), (budget, identity), (budget, 0)]
            if identity else [(0, 0), (budget, 0), (budget, 0), (0, 0)])


def prepare(spec_path, workspace, row_id, out, *, identity_bytes=0, order=None, profiler_allowance_gib=2):
    """Write the plan.  ``identity_bytes`` > 0 adds the pipelined arms.

    v1 order was ``[0, B, B, 0]`` (synchronous, overlap, overlap,
    synchronous).  With an identity reservation the order is
    ``[(0,0), (B,0), (B,I), (B,I), (B,0)]``: the synchronous parity arm, the
    overlap-only baseline, the pipelined arm twice, and the baseline again so
    the box's drift across the run brackets both.  Memory is computed for
    each distinct pair through the same dispatcher plan a row is admitted on.
    ``order`` overrides the arm list for a repeat run; an arm may be a
    ``(overlap, identity, anchor_batch_size)`` triple to run the same
    configuration at another batch width, checked in-plan for exact parity.
    ``profiler_allowance_gib`` is the bounded CUDA trace plus its decoded
    objects, charged on top of the workload; the trace cap is a quarter of it.
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
    order = with_batch(arm_order(identity_bytes, order), command_batch(command))
    for budget, identity, batch in sorted(set(order)):
        selected = copy.deepcopy(spec)
        argv = list(selected['campaign_argv'])
        argv[argv.index('--anchor-batch-size')+1] = str(batch)
        selected['campaign_argv'] = argv+arm_flags(budget, identity)
        resources['%d:%d:%d' % (budget, identity, batch)] = _streamed_resource_plan(
            selected, census, row['members'], selected_source=True)
    # Existing measured process baseline plus a bounded CUDA trace and its
    # decoded profiler objects. This allowance is separate from staged bytes.
    workload_memory = math.ceil((max(r['memory_bytes'] for r in resources.values())+
        plan['process_baseline_bytes'])/1024**3)
    if workload_memory > spec['box_memory_gb']:
        raise ValueError(f'comparison workload needs {workload_memory} GiB, above the recipe budget')
    allowance = int(profiler_allowance_gib)
    if allowance < 1:
        raise ValueError('the profiler allowance is at least 1 GiB')
    memory = workload_memory+allowance
    inputs = [Path(spec_path), workspace/'plan.json', Path(plan['manifest']),
        Path(plan['census']), Path(row['units']),
        Path(command[command.index('--calibration-cache')+1])]
    value = dict(schema=SCHEMA_V3, command=command,
        input_sha256={str(path): digest(path) for path in inputs}, resources=resources,
        row_id=row_id, groups=row['groups'], expected_source_units=864,
        limit_anchors=64, projection='gate_up', order=[list(arm) for arm in order],
        environment=spec['env'], container=spec['container'],
        requested_cpus=spec['cpus'], requested_memory_gib=memory, workload_memory_gib=workload_memory,
        out=str(out/'native'), profiler_allowance_gib=allowance)
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


class MemorySampler(threading.Thread):
    """Continuous process and box memory samples for one arm.

    The guard's checkpoints see the process between phases; the peak of a
    batch lives inside one.  This samples ``VmRSS``/``VmHWM`` of the arm
    process, the box's ``MemAvailable`` (on GB10 that is the device budget
    too), and the caching allocator's reserved bytes at a fixed cadence, and
    keeps the peak of each.  It never synchronizes the device.
    """
    def __init__(self, path, interval=MEMORY_SAMPLE_SECONDS):
        super().__init__(name='memory-sampler', daemon=True)
        self.path, self.interval = Path(path), float(interval)
        self.stopped = threading.Event()
        self.samples = []

    @staticmethod
    def read():
        sample = dict(unix=time.time())
        for line in Path('/proc/self/status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:')):
                sample[line.split(':')[0].lower()+'_bytes'] = int(line.split()[1])*1024
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                sample['mem_available_bytes'] = int(line.split()[1])*1024
        try:
            import torch
            if torch.cuda.is_available():
                sample['cuda_reserved_bytes'] = torch.cuda.memory_reserved()
                sample['cuda_allocated_bytes'] = torch.cuda.memory_allocated()
        except Exception:  # pragma: no cover - torch absent or device gone
            pass
        return sample

    def run(self):
        while not self.stopped.is_set():
            self.samples.append(self.read())
            self.stopped.wait(self.interval)

    def summary(self):
        keys = sorted({k for s in self.samples for k in s if k != 'unix'})
        peak = {k: max(s[k] for s in self.samples if k in s) for k in keys}
        peak['mem_available_bytes'] = min(s['mem_available_bytes'] for s in self.samples
                                          if 'mem_available_bytes' in s) if self.samples else None
        return dict(schema='prismaquant.publication_memory_samples.v1',
            interval_seconds=self.interval, samples=len(self.samples),
            peak=peak, note='peak is the max of each series, min for mem_available_bytes')

    def close(self):
        self.stopped.set()
        self.join()
        self.samples.append(self.read())
        write(self.path, dict(self.summary(), series=self.samples))


def arm(plan, out, budget, identity=0, batch=None):
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
    command[command.index('--anchor-batch-size')+1] = str(int(batch or command_batch(command)))
    command += arm_flags(budget, identity)
    selected_anchor_command(command)
    recorder = PhaseRecorder()
    memory = MemorySampler(out/'memory-samples.json')
    memory.start()
    observer = AnchorObserver(out/'profile', profile_calls=[0],
        trace_max_bytes=int(plan.get('profiler_allowance_gib', 2))*1024**3//4, command=command,
        cuda_only=True, window_seconds=0.25)
    observer.result['python_sampler']['interval_seconds'] = 0.2
    try:
        with observer, instrument(campaign, recorder, plan['projection']):
            run_prefix(campaign, command, observer, limit=plan['limit_anchors'],
                expected_source_units=plan['expected_source_units'])
            if observer.result['resident_prefetch']['devices'] != ['cuda:0']:
                raise RuntimeError('comparison inputs were not fully GPU resident')
    finally:
        memory.close()
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
    for index, (budget, identity, batch) in enumerate(order):
        out = root/f'arm-{index:02d}'; out.mkdir()
        command = [sys.executable, '-u', '-m', 'experiments.campaign_publication_ab', '--plan', str(path),
            '--plan-sha256', expected, '--arm-out', str(out), '--budget', str(budget),
            '--identity-bytes', str(identity), '--anchor-batch-size', str(batch)]
        env = dict(os.environ, TRITON_CACHE_DIR=str(out/'triton'),
            TORCHINDUCTOR_CACHE_DIR=str(out/'inductor'))
        started = time.time()
        with (out/'command.log').open('x') as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        row = dict(index=index, publication_overlap_bytes=budget, campaign_identity_bytes=identity,
            anchor_batch_size=batch, started_unix=started,
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
        memory = json.loads((out/'memory-samples.json').read_text())
        row.update(exact_parity=True, anchor_units=p['limit_anchors'], observer_result=str(files[0]),
            memory_peak=memory['peak'], memory_samples=memory['samples'])
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
    parser.add_argument('--anchor-batch-size', type=int, default=None,
        help='this arm\'s batch width; default is the recipe\'s')
    args = parser.parse_args()
    if args.arm_out:
        if digest(args.plan) != args.plan_sha256:
            raise ValueError('arm plan changed')
        arm(json.loads(Path(args.plan).read_text()), args.arm_out, args.budget, args.identity_bytes,
            args.anchor_batch_size)
    else:
        run(args.plan, args.plan_sha256)


if __name__ == '__main__':
    main()
