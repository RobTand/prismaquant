"""Native end-to-end joint operator windows on a genuine tiny original GLM.

The source and RTN PWC donors are real fixture bytes. This is a deterministic
synthetic calibration diagnostic, not full GLM fit, serving, KL or bpp evidence.
Legacy and windows use the same B1 execution partition, probes and donors.
Signed components have a stated FP32 tolerance; outgoing cotangents are exact.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import cProfile
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import pstats
import statistics
import sys
import threading
import time
import traceback
from unittest.mock import patch

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from experiments.workspace_netdata import NetdataWriter, sample_netdata


FORMATS = ('FP8_E4M3', 'NVFP4A16', 'BF16')
PROBES = 3
SEED_BASE = 7000
ROWS = 3
SEQUENCE = 17
RTOL = 3e-5
ATOL = 2e-9
POLICY_SCHEMA = 'prismaquant.joint_operator_windows.v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    from prismaquant.cost_stage_checkpoint import atomic_write_bytes
    atomic_write_bytes(Path(path), (json.dumps(value, sort_keys=True, indent=2,
                                               allow_nan=False) + '\n').encode())


def tensor_identity(value):
    value = value.detach().to('cpu').contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())


def reference_kernels(stack):
    from transformers.models.glm5_next import modeling_glm5_next as upstream
    result = {}
    for name in ('causal_conv1d_fn', 'causal_conv1d_update',
                 'chunk_kimi_delta_attention', 'recurrent_kimi_delta_attention'):
        reference = getattr(getattr(upstream, name), '__wrapped__', None)
        if reference is None:
            raise RuntimeError(f'existing tiny fixture reference kernel unavailable: {name}')
        stack.enter_context(patch.object(upstream, name, reference))
        result[name] = f'{reference.__module__}.{reference.__qualname__}'
    return result


def decoder_targets(modules, prefix='model.language_model.layers.'):
    result = {name: module for name, module in modules.items() if name.startswith(prefix)}
    if not result:
        raise RuntimeError('tiny fixture has no decoder target coverage')
    return result


def build_fixture(root, device):
    from prismaquant import aura_cost, format_registry as fr
    from prismaquant.model_profiles.glm5_next import Glm5NextProfile
    from prismaquant.production_weight_cache import _cache_weight_filename
    from prismaquant.routed_experts import profile_declared_packed_expert_projections
    tests = Path(__file__).resolve().parents[1] / 'tests'
    sys.path.insert(0, str(tests))
    try:
        from test_glm5_next_streamed_forward_parity import _build_tiny_model
        from test_glm_campaign_streaming import write_original_layout_checkpoint
    finally:
        sys.path.remove(str(tests))
    root.mkdir(parents=True, exist_ok=False)
    model = _build_tiny_model().to(torch.bfloat16)
    source, pwc = root / 'source', root / 'pwc'
    write_original_layout_checkpoint(model, source)
    pwc.mkdir()
    profile = Glm5NextProfile()
    modules = aura_cost._target_linears(model, include_lm_head=False,
        include_routed_experts=True, profile=profile)
    modules.update({member.qname: member for member in
                    profile_declared_packed_expert_projections(model, profile)})
    modules = decoder_targets(modules)
    if not any('.experts.' in name for name in modules) or not any('.experts.' not in name for name in modules):
        raise RuntimeError('fixture requires actual dense and routed targets')
    entries, shapes = [], {}
    with torch.no_grad():
        for name, module in sorted(modules.items()):
            weight = module.weight.detach()
            shapes[name] = list(weight.shape)
            source_identity = tensor_identity(weight)
            for fmt in FORMATS[:-1]:
                spec = fr.get_format(fmt)
                rendered = spec.quantize_dequantize(weight.to(device=device, dtype=torch.float32, copy=True))
                rendered = rendered.to(device='cpu', dtype=torch.bfloat16).contiguous()
                if tuple(rendered.shape) != tuple(weight.shape) or not bool(torch.isfinite(rendered).all()):
                    raise RuntimeError(f'invalid real RTN donor: {name}@{fmt}')
                path = pwc / _cache_weight_filename(name, fmt)
                torch.save(rendered, path)
                entries.append(dict(name=name, format=fmt, path=str(path),
                    file_sha256=sha(path), serialized_bytes=path.stat().st_size,
                    tensor=tensor_identity(rendered), source=source_identity,
                    quantizer=f'{spec.quantize_dequantize.__module__}.{spec.quantize_dequantize.__qualname__}'))
                del rendered
    ids = torch.arange(ROWS * SEQUENCE).remainder(126).add(2).reshape(ROWS, SEQUENCE)
    fixture = dict(source=str(source), pwc=str(pwc), entries=entries, shapes=shapes,
        ids=tensor_identity(ids), token_rule=f'arange({ROWS * SEQUENCE}).remainder(126).add(2).reshape({ROWS},{SEQUENCE})',
        source_kind='genuinely initialized original-layout tiny GLM, initialization seed 20260826',
        donors='existing registry RTN weights, stored BF16; no exported or served artifact',
        activation_max_abs=None, activation_policy='existing dynamic activation QDQ; no static clipping override',
        formats=list(FORMATS), probe_ids=list(range(SEED_BASE, SEED_BASE + PROBES)),
        probe_microbatch=1, boundary_storage=None)
    write_json(root / 'fixture.json', fixture)
    return fixture, ids


def input_inventory(fixture):
    paths = {Path(row['path']) for row in fixture['entries']}
    paths.update(path for path in Path(fixture['source']).rglob('*') if path.is_file())
    return {str(path): dict(bytes=path.stat().st_size, sha256=sha(path)) for path in sorted(paths)}


def make_policy(fixture):
    largest_elements = max(math.prod(shape) for shape in fixture['shapes'].values())
    # FP8 shares one activation operator and NVFP4A16 has identity activation.
    # Admit a whole largest target, forcing several packed-layer windows.
    return dict(schema=POLICY_SCHEMA, max_statistics_bytes=largest_elements * 8,
        max_candidate_bytes=largest_elements * 4,
        max_render_resident_bytes=max(row['serialized_bytes'] for row in fixture['entries']),
        max_load_buffer_bytes=max(row['serialized_bytes'] for row in fixture['entries']),
        workspace_reserve_bytes=64 * 1024**2,
        max_replay_cotangent_bytes=16 * 1024**2, prefetch_workers=2)


def physical_snapshot(device):
    fields = {}
    for line in Path('/proc/self/smaps_rollup').read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[2] == 'kB':
            fields[parts[0].removesuffix(':')] = int(parts[1]) * 1024
    result = dict(time_unix=time.time(), smaps_bytes=fields, process_io=Path('/proc/self/io').read_text())
    scope = Path('/sys/fs/cgroup')
    for name in ('memory.current', 'memory.peak', 'memory.max'):
        path = scope / name
        if path.is_file():
            value = path.read_text().strip()
            result[name] = value if value == 'max' else int(value)
    path = scope / 'memory.stat'
    if path.is_file():
        result['memory.stat'] = {key: int(value) for key, value in
                                 (line.split() for line in path.read_text().splitlines())}
    if device.type == 'cuda':
        result.update(cuda_allocated_bytes=torch.cuda.memory_allocated(device),
            cuda_reserved_bytes=torch.cuda.memory_reserved(device),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result


class StorageObserver:
    def __init__(self):
        self.storages, self.events = {}, []
        self.peaks = Counter()

    def register(self, kind, name, tensor):
        storage = tensor.untyped_storage()
        self.storages[kind, storage._cdata] = (StorageWeakRef(storage), storage.nbytes(), name)

    def live(self):
        sizes = Counter()
        for (kind, _), (reference, nbytes, _) in self.storages.items():
            if not reference.expired():
                sizes[kind] += nbytes
        for kind, size in sizes.items():
            self.peaks[kind] = max(self.peaks[kind], size)
        return dict(sizes)

    def record(self, label, **fields):
        row = dict(label=label, live_bytes=self.live(), **fields)
        self.events.append(row)
        return row


def cost_rows(payload):
    from prismaquant.joint_aura import validate_joint_aura_entry
    result = []
    for name, choices in sorted(payload['costs'].items()):
        for fmt, row in sorted(choices.items()):
            if not validate_joint_aura_entry(row):
                raise RuntimeError(f'invalid actual joint row: {name}@{fmt}')
            if row['probe_ids'] != list(range(SEED_BASE, SEED_BASE + PROBES)):
                raise RuntimeError('probe ids differ from the fixed experiment')
            result.append(dict(name=name, format=fmt,
                components=row['signed_components_per_probe'],
                signed=row['signed_per_probe'], operator=row['joint_operator_identity'],
                probe=row['probe_identity']))
    return result


def require_cost_parity(reference, observed, *, rtol=RTOL, atol=ATOL):
    if [(r['name'], r['format']) for r in reference] != [(r['name'], r['format']) for r in observed]:
        raise RuntimeError('operator-window candidate roster differs')
    errors = dict(max_absolute=0., max_relative_above_atol=0., compared_components=0)
    for before, after in zip(reference, observed):
        for label, removed in (('probe', {'arithmetic'}),
                               ('operator', {'arithmetic', 'probe_identity_sha256'})):
            if {k: v for k, v in before[label].items() if k not in removed} != {
                    k: v for k, v in after[label].items() if k not in removed}:
                raise RuntimeError(f'{label} input identity changed')
        if len(before['components']) != PROBES or len(after['components']) != PROBES:
            raise RuntimeError('missing signed probe components')
        for bterms, aterms in zip(before['components'], after['components']):
            for key in ('weight', 'activation', 'mixed', 'total'):
                b, a = bterms[key], aterms[key]
                if not (math.isfinite(b) and math.isfinite(a) and math.isclose(a, b, rel_tol=rtol, abs_tol=atol)):
                    raise RuntimeError(f'signed {key} differs for {before["name"]}@{before["format"]}: {a} != {b}')
                error = abs(a - b)
                errors['max_absolute'] = max(errors['max_absolute'], error)
                if abs(b) > atol:
                    errors['max_relative_above_atol'] = max(errors['max_relative_above_atol'], error / abs(b))
                errors['compared_components'] += 1
        for signed, terms in zip(after['signed'], after['components']):
            if signed != terms['total']:
                raise RuntimeError('signed sample differs from its components')
    return errors


def require_cotangent_parity(reference, observed):
    # Incoming cotangent bytes identify each probe independently of replay loop
    # order. Every replay copy must preserve its exact outgoing cotangent.
    def key(row):
        return (row['layer'], row['batch'], canonical_sha(row['incoming']))
    expected = {key(row): row['outgoing'] for row in reference}
    if len(expected) != len(reference):
        raise RuntimeError('baseline cotangents do not uniquely identify every probe')
    observed_keys = set()
    for row in observed:
        k = key(row)
        if k not in expected or row['outgoing'] != expected[k]:
            raise RuntimeError('outgoing probe cotangent bytes changed')
        observed_keys.add(k)
    if observed_keys != set(expected):
        raise RuntimeError('operator replay omitted a probe cotangent')
    return dict(unique_cotangents=len(expected), observed_cotangents=len(observed))


class ProbeObserver(StorageObserver):
    def __init__(self, mode, device, policy):
        super().__init__()
        self.mode, self.device, self.policy = mode, device, policy
        self.source_reads = Counter()
        self.leases, self.cotangents, self.tail_cotangents, self.forward_calls = [], [], [], []
        self.gradient_calls = 0

    def __enter__(self):
        from prismaquant import joint_aura as joint, streaming_model as sm, joint_statistics_replay as replay
        from prismaquant.production_weight_cache import ProductionWeightCache
        self.stack = ExitStack()
        observer = self
        original_read = sm._read_layer_to_device
        def read(prefix, *args, **kwargs):
            observer.source_reads[str(prefix)] += 1
            return original_read(prefix, *args, **kwargs)
        self.stack.enter_context(patch.object(sm, '_read_layer_to_device', read))
        original_load = ProductionWeightCache._record_file_load
        def load(cache, key, tensor, receipt):
            result = original_load(cache, key, tensor, receipt)
            observer.register('PWC', '|'.join(key), tensor)
            row = observer.record('pwc_loaded', key=list(key))
            if observer.mode == 'window' and row['live_bytes'].get('PWC', 0) > policy['max_render_resident_bytes']:
                raise RuntimeError('native PWC backing storage exceeded render budget')
            return result
        policy = self.policy
        self.stack.enter_context(patch.object(ProductionWeightCache, '_record_file_load', load))
        original_release = ProductionWeightCache.release_resident_tensors
        def release(cache, keys=None):
            result = original_release(cache, keys)
            observer.record('pwc_released')
            return result
        self.stack.enter_context(patch.object(ProductionWeightCache, 'release_resident_tensors', release))
        signed = joint.SignedJointProjectionLease
        class ObservedSigned(signed):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                for key, tensor in self.deltas.items():
                    observer.register('legacy_dW', '|'.join(key), tensor)
                observer.record('legacy_lease', names=list(self.modules))
        self.stack.enter_context(patch.object(joint, 'SignedJointProjectionLease', ObservedSigned))
        stats = joint.JointOperatorStatisticsLease
        class ObservedStatistics(stats):
            def __init__(self, *args, **kwargs):
                live = observer.live()
                if live.get('statistics') or live.get('candidate'):
                    raise RuntimeError('previous operator/candidate storage survived into a new statistics lease')
                super().__init__(*args, **kwargs)
                observer.leases.append(dict(names=list(self.modules), capacity=self.statistics_capacity_bytes))
                observer.record('statistics_lease', names=list(self.modules), physical=physical_snapshot(observer.device))

            def _accumulate(self, key, matrix):
                result = super()._accumulate(key, matrix)
                observer.register('statistics', str(key), self._operators[key])
                observer.record('statistics_accumulate')
                return result

            def project(self, delta_weights):
                for key, tensor in delta_weights.items():
                    observer.register('candidate', '|'.join(key), tensor)
                observer.record('candidate_project', keys=[list(key) for key in delta_weights])
                return super().project(delta_weights)

            def finish_projections(self):
                result = super().finish_projections()
                row = observer.record('statistics_finished')
                if row['live_bytes'].get('statistics'):
                    raise RuntimeError('finished statistics lease retained a matrix backing storage')
                return result

            def __exit__(self, *args):
                result = super().__exit__(*args)
                observer.record('statistics_closed')
                return result
        self.stack.enter_context(patch.object(joint, 'JointOperatorStatisticsLease', ObservedStatistics))
        self.stack.enter_context(patch.object(replay, 'JointOperatorStatisticsLease', ObservedStatistics))
        return self

    def attach_runner(self, runner, ids):
        # Runner initialization includes the visual tower, separately from
        # decoder checkpoint reads attributable to the complete probe action.
        self.initialization_reads = dict(self.source_reads)
        self.source_reads.clear()
        batches = {tensor_identity(row.reshape(1, -1))['sha256']: index for index, row in enumerate(ids)}
        tail_counts = Counter()
        tail = runner.tail_logits
        def observed_tail(batch, hidden):
            b = batches[tensor_identity(batch.input_ids)['sha256']]
            probe = tail_counts[b]
            tail_counts[b] += 1
            hidden.register_hook(lambda g, b=b, probe=probe: self.tail_cotangents.append(
                dict(batch=b, probe=probe, outgoing=tensor_identity(g))))
            return tail(batch, hidden)
        runner.tail_logits = observed_tail
        reverse = runner.isolated_layer
        def observed_reverse(batch, layer, hidden, *, pass_state):
            b = batches[tensor_identity(batch.input_ids)['sha256']]
            row = dict(layer=int(layer), batch=b)
            if self.mode == 'window' and any(p.requires_grad or p.grad is not None for p in runner.layers[layer].parameters()):
                raise RuntimeError('operator replay enabled a source parameter gradient plane')
            hidden.register_hook(lambda g: row.update(outgoing=tensor_identity(g)))
            output = reverse(batch, layer, hidden, pass_state=pass_state)
            output.register_hook(lambda g: row.update(incoming=tensor_identity(g)))
            self.cotangents.append(row)
            return output
        runner.isolated_layer = observed_reverse
        call = runner._call
        def observed_call(layer, hidden, *, batch, pass_state):
            b = batches[tensor_identity(batch.input_ids)['sha256']]
            self.forward_calls.append(dict(layer=int(layer), batch=b, grad_enabled=torch.is_grad_enabled()))
            return call(layer, hidden, batch=batch, pass_state=pass_state)
        runner._call = observed_call
        unload = runner.context.unload
        def observed_unload(layer):
            if any(p.grad is not None for p in runner.layers[layer].parameters()):
                raise RuntimeError('source layer unload retained a parameter gradient plane')
            return unload(layer)
        runner.context.unload = observed_unload

    def __exit__(self, *_args):
        self.stack.close()


def run_once(fixture, ids, mode, device, out, index, *, profiled):
    from prismaquant import aura_cost
    from prismaquant.cost_streaming import build_streamed_causal_lm, build_streamed_model_identity
    from prismaquant.model_profiles.glm5_next import Glm5NextProfile
    from prismaquant.production_weight_cache import ProductionWeightCache
    policy = make_policy(fixture)
    cache = ProductionWeightCache(weights={(row['name'], row['format']): row['path'] for row in fixture['entries']},
        levers={}, cache_dir=fixture['pwc'], activation_max_abs=None)
    with ProbeObserver(mode, device, policy) as observer:
        profile = Glm5NextProfile()
        runner = build_streamed_causal_lm(fixture['source'], device=device, dtype=torch.bfloat16,
            offload_folder=str(out / 'source-offload'), profile=profile,
            max_cache_slots=2, prefetch_workers=1, prefetch_lookahead=1,
            require_prefetched_residency=True, cache_headroom_gb=0,
            prefetch_min_available_gb=0, attn_implementation='eager')
        observer.attach_runner(runner, ids)
        model_identity = build_streamed_model_identity(runner, fixture['source'])
        result = dict(mode=mode, index=index, profiled=profiled, started_unix=time.time())
        profiler = cpu_profiler = None
        try:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            result['before'] = physical_snapshot(device)
            with ExitStack() as stack:
                if profiled:
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if device.type == 'cuda':
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    profiler = stack.enter_context(torch.profiler.profile(activities=activities,
                        profile_memory=True, record_shapes=True))
                    cpu_profiler = stack.enter_context(cProfile.Profile())
                started = time.perf_counter()
                with torch.profiler.record_function('joint_operator_probe_' + mode):
                    payload = aura_cost.compute_aura_cost_streamed(runner, ids, FORMATS,
                        n_probes=PROBES, probe_microbatch=1, seed_base=SEED_BASE,
                        min_free_gib=0, production_cache=cache, joint_activation=True,
                        collect_col_energy=True, include_routed_experts=True,
                        formats_by_qname={name: FORMATS for name in fixture['shapes']},
                        model_identity=model_identity, profile=profile,
                        **({'operator_windows': policy} if mode == 'window' else {}))
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                result['elapsed_seconds'] = time.perf_counter() - started
            result['after'] = physical_snapshot(device)
            result.update(cost_rows=cost_rows(payload), provenance=payload['provenance'],
                source_reads=dict(observer.source_reads), initialization_reads=observer.initialization_reads,
                cotangents=observer.cotangents, tail_cotangents=observer.tail_cotangents,
                source_forward_calls=observer.forward_calls, leases=observer.leases,
                ownership_events=observer.events, peak_storage_bytes=dict(observer.peaks),
                operator_windows=policy if mode == 'window' else None)
            ended = observer.record('probe_returned')
            if ended['live_bytes']:
                raise RuntimeError(f'completed probe retained tracked backing owners: {ended["live_bytes"]}')
            expected_prefixes = {f'{runner.context.layers_prefix}{layer}.' for layer in range(runner.num_layers)}
            if set(observer.source_reads) != expected_prefixes:
                raise RuntimeError('actual source checkpoint read prefix coverage differs')
            if len(observer.tail_cotangents) != ROWS * PROBES:
                raise RuntimeError('tail probe cotangent coverage differs')
            if mode == 'window':
                packed_windows = {tuple(row['names']) for row in observer.leases
                                  if any('.experts.' in name for name in row['names'])}
                if len(packed_windows) < 2:
                    raise RuntimeError('native fixture did not force multiple whole-target routed windows')
                result['distinct_packed_windows'] = len(packed_windows)
            if profiler is not None:
                trace = out / f'arm-{index}-{mode}.trace.json'
                profiler.export_chrome_trace(str(trace))
                cpu_profiler.dump_stats(str(out / f'arm-{index}-{mode}.cprofile'))
                with (out / f'arm-{index}-{mode}.cprofile.txt').open('w') as stream:
                    pstats.Stats(cpu_profiler, stream=stream).sort_stats('cumulative').print_stats(80)
                result['profile'] = dict(path=str(trace), sha256=sha(trace), events=[dict(key=row.key,
                    calls=row.count, self_cpu_us=row.self_cpu_time_total, self_device_us=row.self_device_time_total)
                    for row in profiler.key_averages()])
            del payload
        except BaseException:
            result.update(status='failed', traceback=traceback.format_exc(),
                source_reads=dict(observer.source_reads), ownership_events=observer.events,
                leases=observer.leases, cotangents=observer.cotangents)
            if profiler is not None:
                trace = out / f'failed-{index}-{mode}.trace.json'
                profiler.export_chrome_trace(str(trace))
                result['partial_profile'] = dict(path=str(trace), sha256=sha(trace))
            write_json(out / f'failed-{index}-{mode}.json', result)
            raise
        finally:
            runner.shutdown()
            del runner, cache, profiler, cpu_profiler
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        result['after_source_release'] = physical_snapshot(device)
        result['finished_unix'] = time.time()
        return result


def require_parity(reference, observed):
    for field in ('source_reads', 'initialization_reads', 'tail_cotangents'):
        if reference[field] != observed[field]:
            raise RuntimeError(f'native operator-window parity failed: {field}')
    return dict(components=require_cost_parity(reference['cost_rows'], observed['cost_rows']),
                cotangents=require_cotangent_parity(reference['cotangents'], observed['cotangents']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=10.)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 30:
        parser.error('--seconds must be between 1 and 30')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('native CUDA probe requires an admitted GPU')
    args.out.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    record = dict(schema='prismaquant.joint_operator_windows_profile.v1', status='running',
        scope=__doc__, started_unix=time.time(), torch=str(torch.__version__), cuda=torch.version.cuda,
        device=str(device), gpu_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        affinity=sorted(os.sched_getaffinity(0)), tolerance=dict(relative=RTOL, absolute=ATOL),
        arms=[], telemetry_errors=[])
    stop = threading.Event()
    def monitor():
        try:
            with (args.out / 'netdata.jsonl').open('x') as stream:
                writer = NetdataWriter(stream)
                while not stop.is_set():
                    for host in ('sparky', 'sparklina'):
                        writer.write(sample_netdata(host))
                    stop.wait(1.)
        except BaseException as exc:
            record['telemetry_errors'].append(repr(exc))
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        with ExitStack() as stack:
            record['fixture_forward_kernels'] = reference_kernels(stack)
            fixture, ids = build_fixture(args.out / 'fixture', device)
            inventory = input_inventory(fixture)
            record.update(fixture=fixture, input_inventory=inventory, input_inventory_sha256=canonical_sha(inventory))
            reference = None
            for index, mode in enumerate(('legacy', 'window', 'window', 'legacy')):
                if input_inventory(fixture) != inventory:
                    raise RuntimeError('source/PWC input bytes changed before an arm')
                arm = dict(index=index, mode=mode, started_unix=time.time(), invocations=[])
                record['arms'].append(arm)
                observed = run_once(fixture, ids, mode, device, args.out, index, profiled=True)
                if reference is None:
                    reference = observed
                observed['parity'] = require_parity(reference, observed)
                arm['profiled'] = observed
                write_json(args.out / 'progress.json', record)
                started = time.perf_counter()
                arm['measurement_started_unix'] = time.time()
                while time.perf_counter() - started < args.seconds:
                    observed = run_once(fixture, ids, mode, device, args.out, index, profiled=False)
                    observed['parity'] = require_parity(reference, observed)
                    arm['invocations'].append(observed)
                arm['measurement_finished_unix'] = time.time()
                arm['median_probe_seconds'] = statistics.median(r['elapsed_seconds'] for r in arm['invocations'])
                if input_inventory(fixture) != inventory:
                    raise RuntimeError('source/PWC input bytes changed during an arm')
                write_json(args.out / 'progress.json', record)
            record['status'] = 'complete'
    except BaseException:
        record.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        stop.set()
        thread.join(timeout=15)
        if thread.is_alive():
            record['telemetry_errors'].append('Netdata monitor did not stop')
        if record['telemetry_errors']:
            record['status'] = 'failed'
        record['finished_unix'] = time.time()
        write_json(args.out / 'result.json', record)
        print(json.dumps(dict(path=str(args.out / 'result.json'), status=record['status'],
                              sha256=sha(args.out / 'result.json'))), flush=True)
    if record['status'] != 'complete':
        raise RuntimeError('native operator-window evidence is incomplete')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
