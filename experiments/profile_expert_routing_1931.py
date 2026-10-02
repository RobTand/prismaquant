"""Bounded paired GLM MoE routing/projected-check measurement (#1931, #1935).

Run only through PB measurement admission, with independently publisher-bound
whole-file inputs delivered through the reviewed kernel-sealed shared reader.
The canonical shared-expert gate-input capture supplies the same MoE input
rows. This is one resident layer control, not whole-campaign qualification.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import datetime as dt
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request

import torch
from prismaquant import measure_quant_cost as mqc, tessera_campaign as campaign


def _write(path, value):
    temporary = path.with_name(path.name + '.writing')
    with temporary.open('w') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary,path)


def _test_module(filename, name):
    directory = Path(__file__).resolve().parents[1] / 'tests'
    # The reference module imports the common sync instrument as pytest does.
    # Loading it directly from an experiment must establish that same path.
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location(name, directory / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _same(a, b):
    return a['row_counts'] == b['row_counts'] and all(
        len(a[key]) == len(b[key]) and all(
            x.shape == y.shape and x.dtype == y.dtype and x.device == y.device
            and torch.equal(x, y) for x, y in zip(a[key], b[key]))
        for key in ('gate_up', 'down', 'gate_weights'))


def _references():
    routing = _test_module('test_derive_per_expert_routing_1931.py', 'routing_1931_reference')
    checking = _test_module('test_projected_unit_check_1935.py', 'checking_1935_reference')
    return routing, checking


def _reference_smoke():
    routing, checking = _references()
    e, x, indices, weights = routing._fixed_case('mixed', 'cpu', torch.float32)
    experts = routing._GateExperts(e)
    parent = routing._FixedRoute(indices, weights)
    assert _same(routing._reference_derive(experts, x, parent),
                 mqc.derive_per_expert_activations(experts, x, parent))
    assert callable(checking._reference_checked_units)
    print(json.dumps(dict(reference_imports=True, cpu_equality=True,
                         cuda_initialized=torch.cuda.is_initialized())), flush=True)


class _SealedDecoders:
    """Measurement transport of independently verified shared-reader deliveries.

    This adapter contributes no authority or production admission. All bytes
    must already be publisher-digest checked and kernel sealed by the shared
    whole-file reader. Each decoder reads that exact held immutable object.
    """
    def __init__(self, root, buffers):
        self.root, self.buffers = root, buffers

    @contextmanager
    def safe_open(self, factory, path, *args, **kwargs):
        from safetensors import safe_open
        path = Path(path)
        if path.parent != self.root or path.name not in self.buffers:
            raise RuntimeError('measurement decoder outside exact sealed source roster')
        if kwargs.get('device', 'cpu') != 'cpu':
            raise RuntimeError('source decoder must remain on the CPU')
        buffer = self.buffers[path.name]
        buffer.require_sealed()
        with safe_open(buffer.path, *args, **kwargs) as handle:
            yield handle
        buffer.require_sealed()

    def file_stat(self, path):
        return os.stat(self.buffers[Path(path).name].path)

    def descriptor_path(self, path):
        return self.buffers[Path(path).name].path


def _prepare(binding, stack, guard):
    from prismaquant.staged_whole_file import read_staged_sealed_file
    from prismaquant.layer_streaming import _source_safe_open
    from transformers.models.glm5_next import modeling_glm5_next as glm
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

    def sealed(row, label):
        guard.check('before_' + label, reserve_bytes=row['bytes'])
        buffer = read_staged_sealed_file(Path(row['path']), row['sha256'], row['bytes'], label=label)
        stack.callback(buffer.close)
        buffer.require_sealed()
        return buffer

    config_buffer = sealed(binding['config'], 'config')
    with config_buffer.readonly() as handle:
        config = Glm5NextTextConfig(**json.loads(handle.read())['text_config'])
    assert (config.num_local_experts, config.hidden_size, config.moe_intermediate_size,
            config.num_experts_per_tok) == (288, 4096, 2048, 8)
    buffers = {row['name']: sealed(row, row['name']) for row in binding['shards']}
    decoder = _SealedDecoders(Path(binding['model']), buffers)
    with torch.device('meta'):
        moe = glm.Glm5NextTextMoE(config)
    guard.check('before_model_allocation',reserve_bytes=binding['resources']['gpu_weight_bytes'] + 64*1024**2)
    moe = moe.to(torch.bfloat16).to_empty(device='cuda').eval().requires_grad_(False)
    prefix = f"model.language_model.layers.{binding['layer']}."
    inter = config.moe_intermediate_size
    files = {}
    with torch.no_grad():
        for row in binding['tensors']:
            key, file = row['name'], row['file']
            with _source_safe_open(Path(binding['model']) / file,
                    source_authentication=decoder, framework='pt', device='cpu') as handle:
                weight = handle.get_tensor(key)
                if tuple(weight.shape) != tuple(row['shape']) or weight.dtype != torch.bfloat16 and 'bias' not in key:
                    raise RuntimeError('sealed source tensor geometry/dtype differs from binding')
                if key == prefix + 'mlp.gate.weight':
                    moe.gate.weight.copy_(weight)
                elif key == prefix + 'mlp.gate.e_score_correction_bias':
                    moe.gate.register_buffer('e_score_correction_bias', weight.to('cuda'))
                else:
                    rest = key.removeprefix(prefix + 'mlp.experts.')
                    expert, projection, _ = rest.split('.')
                    views = dict(gate_proj=moe.experts.gate_up_proj[int(expert), :inter],
                        up_proj=moe.experts.gate_up_proj[int(expert), inter:],
                        down_proj=moe.experts.down_proj[int(expert)])
                    views[projection].copy_(weight)
                    files[key] = file
                del weight
            guard.check('after_source_tensor')
    capture_buffer = sealed(binding['capture'], 'capture')
    payload = torch.load(capture_buffer.path, mmap=True, map_location='cpu', weights_only=True)
    x = payload['inputs']
    if (payload['name'] != binding['capture']['name'] or payload['source'] != 'tessera_campaign_prefix_f32_v1'
            or x.shape != (512, 4096) or x.dtype != torch.float32
            or payload['count'] != binding['capture']['count']
            or payload['max_abs'] != binding['capture']['max_abs']
            or not torch.equal(x, x.to(torch.bfloat16).float())):
        raise RuntimeError('canonical shared-input capture coordinates/precision differ')
    inputs = x.to(device='cuda', dtype=torch.bfloat16).reshape(1, 512, 4096)
    del payload, x
    torch.cuda.synchronize()
    bound, live = {}, {}
    for e in range(config.num_local_experts):
        for projection, view in dict(gate_proj=moe.experts.gate_up_proj[e, :inter],
            up_proj=moe.experts.gate_up_proj[e, inter:], down_proj=moe.experts.down_proj[e]).items():
            key = f'{prefix}mlp.experts.{e}.{projection}.weight'
            bound.setdefault(projection, {})[key] = dict(source_tensor=key,
                rows=view.shape[0], cols=view.shape[1])
            live[key] = view
    assert len(binding['tensors']) == 866 and len(live) == len(files) == 864
    return moe, inputs, bound, live, dict(tensors=files), decoder


class _HookControlComplete(Exception):
    """End the bounded hook control before complete-capture output handoff."""


def _production_hook_phases(moe, inputs, binding, record, telemetry, guard, output, seconds):
    """Measure the actual installed collector, consume_packed and accumulate.

    The control supplies one existing captured input directly to the installed
    pre-hook. It does not perform another model forward or publish a capture.
    A deliberate bounded stop uses the collector's ordinary failure cleanup
    after the steady phases; a complete calibration artifact is out of scope.
    """
    from prismaquant.model_profiles import detect_profile
    from prismaquant.staged_whole_file import read_staged_whole_file
    config = json.loads(read_staged_whole_file(Path(binding['config']['path']),
        binding['config']['sha256'], label='hook-config'))
    profile = detect_profile(binding['model'], config=config)
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    model.model.language_model.layers = torch.nn.ModuleList([
        torch.nn.Module() for _ in range(binding['layer'] + 1)])
    model.model.language_model.layers[binding['layer']].mlp = moe
    prefix = f"model.language_model.layers.{binding['layer']}.mlp.experts"
    targets = {f'{prefix}.{e}.{projection}' for e in range(288)
        for projection in ('gate_proj','up_proj','down_proj')}
    original = mqc.derive_per_expert_activations
    arm = ['before']
    references = dict(before=_references()[0]._reference_derive, after=original)
    def selected(*args, **kwargs):
        return references[arm[0]](*args, **kwargs)
    def batches():
        registered = list(moe.experts._forward_pre_hooks.values())
        if len(registered) != 1:
            raise RuntimeError('production collector did not install exactly one expected hook')
        hook = registered[0]
        record['production_hook'] = dict(module=hook.__module__,
            file=hook.__code__.co_filename,line=hook.__code__.co_firstlineno,
            selected_units=len(targets),max_rows=512,shared_packed_inputs=True)
        collector = inspect.getclosurevars(hook).nonlocals['self']
        accumulate = inspect.getclosurevars(collector.row_consumer).nonlocals['accumulate']
        state = inspect.getclosurevars(accumulate).nonlocals
        def committed_state():
            return dict(rows_seen=dict(state['seen']),prefix_rows=dict(state['kept']),
                hessian_bytes=sum(value.numel()*value.element_size()
                    for value in state['hess'].values() if value is not None))
        probe = references['before'](moe.experts,inputs,moe)
        minimum = min(n for n in probe['row_counts'] if n)
        calls_to_fill = (512+minimum-1)//minimum
        del probe
        # Warm the genuine prefix owners to their real cap, on this input.
        # H/row counters come from the actual closure, never a mirror.
        guard.check('before_hook_residency',reserve_bytes=
            binding['resources']['hessian_bytes'] + binding['resources']['prefix_bytes']
            + 512*1024**2)
        for _ in range(calls_to_fill):
            hook(moe.experts,(inputs,))
        torch.cuda.synchronize()
        active = [name for name,value in state['seen'].items() if value]
        prefix_owners = [name for name in state['kept'] if state['seen'][name]]
        assert all(state['kept'][name] == 512 for name in prefix_owners)
        record['hook_warmup'] = dict(calls=calls_to_fill,minimum_nonzero_routed_rows=minimum,
            active_units=len(active),stable_prefix=True,state=committed_state())
        _write(output/'hook-warmup.json',record['hook_warmup'])
        for i, name in enumerate(('before','after','after','before')):
            arm[0] = name
            before = committed_state()
            phase = _phase(f'hook-{i}-{name}', lambda:hook(moe.experts,(inputs,)),
                seconds,telemetry,guard,output)
            phase.update(committed_state_before=before,committed_state_after=committed_state(),
                prefix_stable=before['prefix_rows'] == committed_state()['prefix_rows'])
            assert phase['prefix_stable']
            record['phases'].append(phase)
            _write(output/(phase['phase']+'.json'),phase)
            _write(output/'partial.json',record)
            # The iterator executes while the ordinary production hook scope
            # is installed. No-op forward avoids a second source-model draw.
            yield torch.zeros((1,1),dtype=torch.long)
        raise _HookControlComplete()
    mqc.derive_per_expert_activations = selected
    try:
        try:
            campaign._collect_activations(model, targets, batches(), 512, 'cuda',
                want_hessian=True,profile=profile,shared_packed_inputs=True,
                forward_batch=lambda _:None,on_forwards_complete=lambda:None)
        except _HookControlComplete:
            record['hook_control_cleanup'] = 'ordinary bounded-stop unwind; no capture artifact'
        else:
            raise RuntimeError('hook control did not end at the declared boundary')
    finally:
        mqc.derive_per_expert_activations = original
    if moe.experts._forward_pre_hooks:
        raise RuntimeError('production collector left a hook installed')
    gc.collect(); torch.cuda.empty_cache()


class _Power:
    def __init__(self):
        self.samples, self.errors = [], []
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        def observe():
            while not self.stop.is_set():
                start = time.time()
                try:
                    result = subprocess.run(['nvidia-smi', '--query-gpu=power.draw',
                        '--format=csv,noheader,nounits'], capture_output=True, text=True,
                        timeout=3, check=True)
                    self.samples.append(dict(requested_unix=start, observed_unix=time.time(),
                        watts=float(result.stdout.strip().splitlines()[0])))
                except Exception as error:
                    self.errors.append(str(error))
                self.stop.wait(.25)
        self.thread = threading.Thread(target=observe, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('power recorder failed to stop')


def _profile(fn, output):
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(output))
    averages = prof.key_averages()
    runtime = {event.key: event.count for event in averages if event.key.startswith('cuda')}
    return dict(trace=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        runtime_calls=runtime, self_device_seconds=sum(
            getattr(e, 'self_device_time_total', 0.) for e in averages) / 1e6,
        top_self_cpu=[dict(op=e.key, count=e.count, self_cpu_seconds=e.self_cpu_time_total/1e6)
            for e in sorted(averages, key=lambda e: e.self_cpu_time_total, reverse=True)[:12]])


def _raw_netdata(phase, output):
    # Preserve exact requests and untrimmed returned views/time/series. These
    # bucketed series are evidence only until the independent alignment gate
    # qualifies their actual bounds and cadence; no energy claim is derived.
    rows = []
    for host in ('sparky', 'sparklina'):
        from experiments.workspace_netdata import sample_netdata
        observed = sample_netdata(host)
        power_charts = [k for k in observed['metrics'] if k.startswith('nvidia_smi.') and k.endswith('_power_draw')]
        rows.append(dict(host=host, request_kind='allmetrics-live', response=observed))
        for chart in ('system.cpu', 'system.io', 'system.ram', 'system.cpu_some_pressure',
                      'system.io_some_pressure', *power_charts):
            query = dict(chart=chart, after=int(phase['started_unix'])-2,
                before=int(phase['finished_unix'])+2, points=0, group='average', format='json')
            url = f'http://{host}:19999/api/v1/data?' + urllib.parse.urlencode(query)
            row = dict(host=host, request=url, requested=query, requested_unix=time.time())
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    raw = response.read(4*1024**2+1)
                if len(raw) > 4*1024**2:
                    raise RuntimeError('raw Netdata response exceeds bound')
                row.update(returned_unix=time.time(), response=json.loads(raw))
            except Exception as error:
                row['error'] = str(error)
            rows.append(row)
    _write(output, rows)


def _phase(name, fn, seconds, telemetry, guard, output):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    guard.registered_gpu_subset_check('before_' + name)
    telemetry.collect(); telemetry.require_healthy()
    start, mono = time.time(), time.monotonic()
    times = []
    with _Power() as power:
        while time.monotonic()-mono < seconds:
            before = time.perf_counter()
            fn(); torch.cuda.synchronize()
            times.append(time.perf_counter()-before)
    end, end_mono = time.time(), time.monotonic()
    telemetry.collect(); telemetry.require_healthy()
    guard.registered_gpu_subset_check('after_' + name)
    row = dict(phase=name, started_unix=start, finished_unix=end,
        started_monotonic=mono, finished_monotonic=end_mono, calls=len(times),
        elapsed_seconds=end_mono-mono, wall_seconds_mean=statistics.fmean(times),
        wall_seconds_median=statistics.median(times), call_seconds=times,
        power_samples=power.samples, power_errors=power.errors, guard=guard.snapshot())
    row['profile'] = _profile(fn, output/(name+'.trace.json'))
    _raw_netdata(row, output/(name+'.netdata-raw.json'))
    _write(output/(name+'.json'), row)
    print(json.dumps({k:row[k] for k in ('phase','calls','wall_seconds_mean')}), flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path)
    parser.add_argument('--binding-sha256')
    parser.add_argument('--data-manifest-sha256')
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--references-only', action='store_true')
    args = parser.parse_args()
    if args.references_only:
        _reference_smoke(); return
    if args.seconds < 30 or args.seconds > 90:
        parser.error('steady arms must last 30..90 seconds')
    raw = args.binding.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.binding_sha256:
        raise RuntimeError('measurement binding changed')
    binding = json.loads(raw)
    if binding['schema'] != 'prismaquant.pq1934_measurement_binding.v1':
        raise RuntimeError('unsupported measurement binding')
    args.out.mkdir(parents=True, exist_ok=False)
    from prismaquant.residency_map import bind_residency_manifest, residency_resolver
    from prismaquant.staged_tier_policy import activate_staged_tier_policy
    from prismaquant.memory_management import CaptureMemoryGuard
    from experiments.glm_native_wire_screen_evidence import ScreenTelemetry
    activate_staged_tier_policy('ram,ssd')
    bind_residency_manifest(args.data_manifest_sha256)
    torch.set_num_threads(1)
    guard = CaptureMemoryGuard('cuda')
    if guard.cap_bytes < binding['resources']['aggregate_limit_bytes']:
        raise RuntimeError('PB enforced cgroup limit is smaller than the bound aggregate')
    def require_gpu_budget(label):
        if torch.cuda.memory_reserved() > binding['resources']['gpu_limit_bytes']:
            raise RuntimeError('CUDA reservation exceeds the bound GPU subset')
        return guard.check(label)
    guard.registered_gpu_subset_check = require_gpu_budget
    telemetry = ScreenTelemetry(args.out/'netdata-points.jsonl')
    record = dict(schema='prismaquant.pq1934_paired_measurement.v1', binding=binding,
        binding_sha256=args.binding_sha256, data_manifest_sha256=args.data_manifest_sha256,
        source_head=os.environ.get('PQ1931_GIT_HEAD'), torch=str(torch.__version__),
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), host=os.uname().nodename,
        gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), 'uuid', '')),
        affinity=sorted(os.sched_getaffinity(0)), native_threads=torch.get_num_threads(),
        phases=[], equality=[], energy_qualified=False,
        scope='one real resident GLM MoE layer; no full-campaign or serving claim')
    routing, checking = _references()
    try:
        telemetry.start()
        with ExitStack() as stack:
            try:
                moe, inputs, bound, live, source, decoder = _prepare(binding, stack, guard)
                for capture_down in (False, True):
                    for max_rows in (None, 16):
                        kwargs = dict(capture_down=capture_down, max_rows_per_expert=max_rows)
                        old = routing._reference_derive(moe.experts, inputs, moe, **kwargs)
                        new = mqc.derive_per_expert_activations(moe.experts, inputs, moe, **kwargs)
                        row = dict(capture_down=capture_down, max_rows=max_rows, equal=_same(old,new),
                            zero_row_experts=old['row_counts'].count(0))
                        record['equality'].append(row)
                        del old,new
                        if not row['equal']:
                            raise RuntimeError('routed outputs changed')
                calls = dict(before=lambda: routing._reference_derive(moe.experts, inputs, moe),
                             after=lambda: mqc.derive_per_expert_activations(moe.experts, inputs, moe))
                checks = {name:(lambda fn=fn: fn(bound, weights=live, model_path=binding['model'],
                    source=source, source_authentication=decoder)) for name,fn in dict(
                        before=checking._reference_checked_units, after=campaign._checked_projected_units).items()}
                for fn in checks.values():
                    assert len(fn()) == 864
                first = next(iter(campaign._measured_projected_units(bound)))
                live[first].view(torch.int16)[0,0] ^= 1
                messages = []
                try:
                    for fn in checks.values():
                        try:
                            fn()
                        except RuntimeError as error:
                            messages.append(str(error))
                    assert len(messages) == 2 and messages[0] == messages[1] and first in messages[0]
                finally:
                    live[first].view(torch.int16)[0,0] ^= 1
                record['mismatch_refusals'] = messages
                record['sync_sites'] = {kind:{name:routing.sync_sites(fn) for name,fn in functions.items()}
                    for kind,functions in dict(derive=calls,check=checks).items()}
                for kind,functions in dict(derive=calls,check=checks).items():
                    for i,arm in enumerate(('before','after','after','before')):
                        record['phases'].append(_phase(f'{kind}-{i}-{arm}', functions[arm],
                            args.seconds, telemetry, guard, args.out))
                        _write(args.out/'partial.json',record)
                torch.cuda.synchronize()
                record['phase_boundary_before_host_cache_release'] = guard.snapshot()
                torch._C._accelerator_emptyHostCache()
                torch.cuda.empty_cache(); gc.collect()
                guard.check('after_projected_check_host_cache_release')
                record['phase_boundary_after_host_cache_release'] = guard.snapshot()
                _production_hook_phases(moe,inputs,binding,record,telemetry,guard,args.out,args.seconds)
                # Every async CPU-buffer consumer completes before any sealed FD closes.
                torch.cuda.synchronize()
                del checks,calls,live,bound,source,decoder,moe,inputs
                gc.collect()
                record['residency'] = residency_resolver().report()
            finally:
                # Join even partial preparation, cancellation and guard failure
                # while every raw descriptor is still owned by this stack.
                torch.cuda.synchronize()
        record['sealed_buffers_closed'] = True
    finally:
        telemetry.collect()
        record['telemetry_coverage'] = telemetry.finish(record['phases'])
        _write(args.out/'result.json',record)
    if not record['telemetry_coverage']['passed']:
        raise RuntimeError('Netdata coverage failed; measurement remains unqualified')
    record['gain_fraction'] = {}
    for kind in ('derive','hook','check'):
        before = [p['wall_seconds_mean'] for p in record['phases'] if p['phase'].startswith(kind) and p['phase'].endswith('before')]
        after = [p['wall_seconds_mean'] for p in record['phases'] if p['phase'].startswith(kind) and p['phase'].endswith('after')]
        record['gain_fraction'][kind] = 1 - statistics.fmean(after)/statistics.fmean(before)
    record['hotpath_screen_passed'] = all(record['gain_fraction'][k] >= .02 for k in ('hook','check'))
    _write(args.out/'result.json',record)
    print(json.dumps(dict(gain_fraction=record['gain_fraction'],hotpath_screen_passed=record['hotpath_screen_passed'])),flush=True)


if __name__ == '__main__':
    main()
