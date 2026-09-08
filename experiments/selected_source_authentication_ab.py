"""One frozen, PB-admitted full-to-selected source authentication pilot pair.

No capture forward, X/H load, encoding, automatic retry, or CPU native mode.
The original complete-capture initialization witness is retained as provenance;
this source-only experiment does not claim a new initialization forward.
"""
from __future__ import annotations

import argparse
import cProfile
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import weakref
from unittest.mock import patch

from experiments.selected_source_authentication_ab_plan import sealed_json, GIB

SCHEMA = 'prismaquant.selected_source_authentication_ab.frozen.v1'
SELECTED_UNITS = ('model.language_model.layers.0.mlp.down_proj',
    'model.language_model.layers.0.mlp.gate_proj',
    'model.language_model.layers.6.mlp.experts.0.down_proj',
    'model.language_model.layers.6.mlp.experts.0.gate_proj')
ENVIRONMENT = dict(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
    PRISMAQUANT_LAYER_READ_THREADS='4', PRISMAQUANT_RELEASE_SOURCE_PAGES='1',
    MIMALLOC_PURGE_DELAY='0')
SOURCE_FILES = ('experiments/selected_source_authentication_ab.py',
    'experiments/selected_source_authentication_ab_plan.py',
    'experiments/glm_native_wire_screen_evidence.py',
    'experiments/workspace_netdata.py', 'prismaquant/tessera_calibration_cache.py',
    'prismaquant/cost_streaming.py', 'prismaquant/layer_streaming.py',
    'prismaquant/streaming_model.py', 'prismaquant/tessera_campaign.py',
    'prismaquant/tessera_expert_projection.py', 'prismaquant/memory_management.py',
    'prismaquant/calibration_data.py', 'prismaquant/production_weight_cache.py',
    'tools/tessera_campaign_container.py', 'tools/container_runtime_identity.py')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_stats(root):
    return {path.name:list((lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns))(
        path.stat())) for path in sorted(Path(root).iterdir()) if path.is_file()}


def _descriptor_open(fd):
    try:
        os.fstat(fd)
        return True
    except OSError as error:
        if error.errno != 9:
            raise
        return False


def proc_io():
    return {key:int(value) for key,value in (line.split(':') for line in
        Path('/proc/self/io').read_text().splitlines())}


def preflight(plan, *, repository):
    """All acceptance is metadata/token-only and must precede CUDA and payloads."""
    if plan.get('schema') != SCHEMA or plan.get('status') != 'FROZEN':
        raise ValueError('native A/B requires a reviewed FROZEN plan')
    if set(plan.get('source_files', {})) != set(SOURCE_FILES):
        raise ValueError('frozen source behavior roster differs')
    for name, expected in plan['source_files'].items():
        if digest(Path(repository)/name) != expected:
            raise ValueError('frozen source changed: '+name)
    if (not isinstance(plan.get('environment'), dict)
            or any(plan['environment'].get(k) != v for k,v in ENVIRONMENT.items())
            or any(os.environ.get(k, '') != v for k,v in plan['environment'].items())):
        raise ValueError('frozen thread/page-release environment differs')
    resources = sealed_json(plan['resources'])
    if (resources.get('schema') != 'prismaquant.selected_source_authentication_ab.resources.v1'
            or not resources['fits_104gib_conditional_envelope'] or resources['requested_mem_gib'] != 104
            or resources['order'] != ['full','selected'] or resources['automatic_followup']):
        raise ValueError('frozen conditional resource contract differs')
    inputs = resources['inputs']
    if plan.get('container') != inputs['container']:
        raise ValueError('frozen container differs from the resource input')
    census = sealed_json(inputs['census'])
    if census['model'] != inputs['model'] or any(census.get(k) != v for k,v in
            dict(nsamples=512,seqlen=512,seed=0,layer_stride=1).items()):
        raise ValueError('native A/B requires the original full census draw')
    if (set(inputs['selected_units']) != set(SELECTED_UNITS) or len(inputs['selected_units']) != 4
            or resources['selected_shapes'] != {n:census['unit_shapes'][n] for n in inputs['selected_units']}):
        raise ValueError('frozen selected source scope differs')
    import torch
    from prismaquant import tessera_calibration_cache as cc
    from prismaquant.calibration_data import load_calibration_input
    if torch.cuda.is_initialized():
        raise RuntimeError('complete capture preflight must precede CUDA initialization')
    capture = cc.require_capture_contract(plan['capture']['path'], expected_sha256=plan['capture']['sha256'])
    if capture['identity']['max_act_rows'] != 512:
        raise ValueError('source A/B requires the original 512-row capture prefix')
    if (any(inputs['calibration'].get(k) != v for k,v in dict(nsamples=512,seqlen=512,seed=0).items())
            or any(inputs['calibration'].get(k) != census[k] for k in ('fit_ids_sha256','text_sha256'))):
        raise ValueError('frozen calibration differs from the original census draw')
    ids, tokens = load_calibration_input(inputs['calibration_input']['path'],
        expected_sha256=inputs['calibration_input']['sha256'], n_samples=512, seqlen=512)
    del ids
    for k,v in inputs['calibration'].items():
        if capture['identity']['calibration'].get(k) != v or tokens['provenance'].get(k) != v:
            raise ValueError('capture and token provenance differ from the original draw: '+k)
    kwargs = owner_kwargs(inputs, census, plan['capture'], capture)
    # Public factory checks runtime, identity, complete units, index/config and
    # auxiliary bytes, but does not hash or read source tensor payloads.
    with cc.authenticate_selected_capture_source(**kwargs) as owner:
        if owner.receipt()['payload_bytes_hashed']:
            raise RuntimeError('preflight unexpectedly consumed source payloads')
    return resources, census, capture, tokens


def owner_kwargs(inputs, census, capture_binding, capture):
    return dict(census_path=inputs['census']['path'], capture_path=capture_binding['path'],
        expected_sha256=capture_binding['sha256'], model=inputs['model'],
        max_act_rows=capture['identity']['max_act_rows'],
        attention_implementation=census['attention_implementation'],
        calibration_parameters={k:census[k] for k in ('nsamples','seqlen','seed')})


class ResourceCheck:
    def __init__(self, guard, resources, telemetry, *, cuda):
        self.guard, self.resources, self.telemetry, self.cuda = guard, resources, telemetry, cuda
        self.started = time.monotonic()
        self.arm, self.phase, self.deadline = '', '', float('inf')
        self.max_file_charge = 0

    def file_charge(self):
        values = dict(line.split() for line in (self.guard.scope/'memory.stat').read_text().splitlines())
        value = int(values['file'])
        if value < 0:
            raise RuntimeError('cgroup file charge unavailable')
        self.max_file_charge = max(value, self.max_file_charge)
        return value

    def require_file_cap(self, cap):
        value = self.file_charge()
        if value > cap:
            raise RuntimeError(f'legacy full-source page residue refused: {value} > {cap}')
        return value

    def __call__(self, label, *, reserve_bytes=0):
        self.telemetry.require_healthy()
        now = time.monotonic()
        if now > self.deadline or now-self.started > self.resources['total_deadline_seconds']:
            raise RuntimeError('source A/B hard deadline exceeded')
        if 'capture_hash:' in label:
            reserve_bytes = max(reserve_bytes, self.resources['hash_read_buffer_bytes'])
        result = self.guard.check(label, reserve_bytes=reserve_bytes)
        if self.cuda.memory_reserved() > self.resources['requested_gpu_mem_gib']*GIB:
            raise RuntimeError('source A/B CUDA reservation exceeds its frozen subset')
        if self.arm == 'full' and self.phase == 'authentication':
            self.require_file_cap(self.resources['legacy_hash_absolute_file_cap_bytes'])
        return result


class SourceEvents:
    """Observe existing hash/read seams without retaining tensor or mmap owners."""
    def __init__(self, root, output, limit):
        self.root, self.output, self.limit = Path(root), Path(output), limit
        self.lock, self.rows, self.bytes = threading.Lock(), [], 0
        self.arm, self.phase = '', ''
        self.paths = {str(path):path.name for path in self.root.iterdir() if path.is_file()}
        self.realpaths = {str(Path(path).resolve()):name for path,name in self.paths.items()}

    def name(self, path):
        return self.paths.get(str(path)) or self.realpaths.get(str(Path(path).resolve()))

    def record(self, value):
        row = dict(value, arm=self.arm, phase=self.phase, monotonic=time.monotonic(),
                   thread=threading.get_ident())
        raw = json.dumps(row, separators=(',',':'))+'\n'
        with self.lock:
            self.bytes += len(raw.encode())
            if self.bytes > self.limit:
                raise RuntimeError('source event output exceeded frozen bound')
            self.stream.write(raw); self.stream.flush()
            if row['operation'] == 'sha256':
                self.rows.append(row)

    @contextmanager
    def observe(self):
        from contextlib import ExitStack
        import safetensors
        from prismaquant import tessera_calibration_cache as cc, layer_streaming as ls, streaming_model as sm
        original = cc.sha256
        def hashed(path, **kwargs):
            name = self.name(path)
            if name is None:
                return original(path, **kwargs)
            started, before = time.monotonic(), proc_io()
            profile = cProfile.Profile() if threading.current_thread() is not threading.main_thread() else None
            if profile is not None:
                profile.enable()
            try:
                value = original(path, **kwargs)
                self.record(dict(operation='sha256', name=name, bytes=Path(path).stat().st_size,
                    sha256=value, held_fd=kwargs.get('file_descriptor'),
                    started_monotonic=started, elapsed=time.monotonic()-started,
                    proc_io_before=before, proc_io_after=proc_io(),
                    proc_io_scope='Process-wide overlapping intervals; concurrent file hashes are not exclusive I/O.'))
                return value
            finally:
                if profile is not None:
                    profile.disable()
                    profile.dump_stats(str(self.output/f'{self.arm}-{threading.get_ident()}-{name}.cprofile'))
        def factory(original_open):
            def opened(path, *args, **kwargs):
                raw = original_open(path, *args, **kwargs)
                name = self.name(path)
                return raw if name is None else ObservedReader(raw, self, name, path)
            return opened
        with self.output.joinpath('source-events.jsonl').open('x') as self.stream, ExitStack() as stack:
            stack.enter_context(patch.object(cc, 'sha256', hashed))
            for module in (safetensors, ls, sm):
                stack.enter_context(patch.object(module, 'safe_open', factory(module.safe_open)))
            yield self


class ObservedReader:
    def __init__(self, raw, recorder, name, path):
        self.raw, self.recorder, self.name, self.path = raw, recorder, name, str(path)
    def event(self, operation, **kwargs):
        self.recorder.record(dict(operation=operation,name=self.name,
            held_fd_alias=self.path.startswith('/proc/self/fd/'), **kwargs))
    def __enter__(self):
        self.raw.__enter__(); self.event('header_open'); return self
    def __exit__(self, *args):
        try:
            return self.raw.__exit__(*args)
        finally:
            self.event('reader_close')
    def __getattr__(self, key):
        return getattr(self.raw, key)
    def get_tensor(self, key):
        value = self.raw.get_tensor(key)
        self.event('payload_tensor', key=key, shape=list(value.shape), dtype=str(value.dtype),
                   logical_bytes=value.numel()*value.element_size())
        return value
    def get_slice(self, key):
        return ObservedSlice(self.raw.get_slice(key), self, key)


class ObservedSlice:
    def __init__(self, raw, reader, key):
        self.raw, self.reader, self.key = raw, reader, key
    def __getattr__(self, key):
        return getattr(self.raw, key)
    def __getitem__(self, item):
        value = self.raw[item]
        self.reader.event('payload_slice', key=self.key, shape=list(value.shape),
            dtype=str(value.dtype), logical_bytes=value.numel()*value.element_size())
        return value


def prepare_arm(arm, *, inputs, census, capture_binding, capture, resources,
                output, check, phase, build, profile, device):
    """Inject only the model constructor in CPU fixtures; actual owners stay live."""
    import torch
    from prismaquant import tessera_calibration_cache as cc, tessera_campaign as tc
    owner = runner = None
    weights = None
    runner_ref = None
    before = source_stats(inputs['model'])
    manifest_before = digest(capture_binding['path'])
    report = dict(arm=arm, identity=None, closed=False, source_forward_count=0,
        source_stats_before=before, capture_sha256_before=manifest_before)
    try:
        if arm == 'selected':
            owner = phase('selected_authentication', lambda: cc.authenticate_selected_capture_source(
                **owner_kwargs(inputs,census,capture_binding,capture), resource_check=check,
                release_read_pages=True))
        elif arm != 'full':
            raise ValueError('unknown source A/B arm')
        runner = phase('source_build', lambda: build(inputs['model'], device=torch.device(device),
            dtype=torch.bfloat16, profile=profile, offload_folder=str(output/(arm+'-offload')),
            max_cache_slots=2,prefetch_workers=1,cache_headroom_gb=24,
            prefetch_min_available_gb=24,prefetch_lookahead=1,require_prefetched_residency=True,
            attn_implementation=census['attention_implementation'],source_authentication=owner))
        try:
            runner_ref = weakref.ref(runner)
        except TypeError:  # SimpleNamespace CPU fixture only.
            pass
        report['identity'] = phase('authentication', lambda: cc.capture_identity(
            inputs['census']['path'],calibration=capture['identity']['calibration'],
            max_act_rows=capture['identity']['max_act_rows'],
            model_load_contract=census['model_load_contract'],
            attention_implementation=census['attention_implementation'],resource_check=check,
            release_read_pages=True, source_authentication=owner))
        if report['identity'] != capture['identity']:
            raise RuntimeError('arm canonical full identity differs')
        if arm == 'full':
            check.require_file_cap(resources['legacy_post_hash_absolute_file_cap_bytes'])
        weights, report['snapshot'] = phase('snapshot', lambda: runner.snapshot_selected_weights(
            sorted(inputs['selected_units']), max_resident_bytes=resources['max_selected_bytes'],
            resource_check=check))
        phase('source_teardown', runner.shutdown)
        runner = None
        gc.collect()
        if torch.device(device).type == 'cuda':
            torch.cuda.empty_cache()
        if runner_ref is not None and runner_ref() is not None:
            raise RuntimeError('source runner retained after teardown')
        report['projection'] = phase('projection', lambda: tc._checked_projected_units(
            census['expert_projection']['stacks'], weights=weights, model_path=inputs['model'],
            source=census['expert_projection']['producer']['source'], measured=set(weights),
            resource_check=check,release_source_pages=True,source_authentication=owner))
        if owner is not None:
            report['source_authentication'] = owner.receipt()
        if (source_stats(inputs['model']) != before or digest(capture_binding['path']) != manifest_before
                or manifest_before != capture_binding['sha256']):
            raise RuntimeError('original source or complete capture changed during the arm')
        return weights, report
    finally:
        try:
            if runner is not None:
                runner.shutdown()
        finally:
            runner = None
            if owner is not None:
                owner.close()
                report['owner_closed'] = owner._closed
                report['owned_descriptor_count_after_close'] = sum(
                    _descriptor_open(state['fd']) for state in owner._files.values())
                if report['owned_descriptor_count_after_close']:
                    raise RuntimeError('source authentication retained an open descriptor')
            report['closed'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',required=True,type=Path)
    parser.add_argument('--plan-sha256',required=True)
    args = parser.parse_args()
    plan = sealed_json(dict(path=str(args.plan),sha256=args.plan_sha256))
    resources,census,capture,tokens = preflight(plan, repository=Path(__file__).resolve().parents[1])
    import torch
    from prismaquant.cost_streaming import build_streamed_causal_lm
    from prismaquant.model_profiles import detect_profile
    from prismaquant.memory_management import CaptureMemoryGuard
    from prismaquant.production_weight_cache import _cb_cache_tensor_identity
    from experiments.glm_native_wire_screen_evidence import ScreenTelemetry
    torch.set_num_threads(1); torch.cuda.init()
    guard = CaptureMemoryGuard('cuda')
    if guard.cap_bytes < resources['requested_mem_gib']*GIB:
        raise RuntimeError('PB cap smaller than frozen conditional guard envelope')
    output = Path(plan['output']); output.mkdir(parents=True,exist_ok=False)
    result = dict(schema='prismaquant.selected_source_authentication_ab.result.v1',passed=False,
        plan_sha256=args.plan_sha256, capture=plan['capture'], resources=plan['resources'],
        tokens=tokens, arms=[], phases=[], source_forward_count=0, anchor_render_count=0,
        original_source_stats=source_stats(census['model']),
        interpretation=resources['interpretation'],
        unmeasured=['GPU kernel attribution', 'order-independent speed', 'full anchor throughput'])
    telemetry = ScreenTelemetry(output/'netdata.jsonl')
    check = ResourceCheck(guard,resources,telemetry,cuda=torch.cuda)
    events = SourceEvents(census['model'],output,resources['max_source_event_bytes'])
    reference, weights = {}, None
    def phase(label, function):
        check.phase = events.phase = label
        future = (2*resources['physical_phases']['source_preparation']['nonbody_source_bytes']
                  if label == 'source_build' else 0)
        check('before_'+label, reserve_bytes=future); telemetry.collect(); telemetry.require_healthy()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start, mono, before = time.time(), time.monotonic(), proc_io()
        cpu = cProfile.Profile(); cpu.enable()
        try:
            value = function(); torch.cuda.synchronize()
            return value
        finally:
            end, endmono = time.time(), time.monotonic()
            cpu.disable(); cpu.dump_stats(str(output/(check.arm+'-'+label+'.cprofile')))
            row = dict(phase=check.arm+'-'+label, started_unix=start,finished_unix=end,
                started_monotonic=mono,finished_monotonic=endmono,elapsed=endmono-mono,
                proc_io_before=before,proc_io_after=proc_io(),guard=guard.snapshot(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            result['phases'].append(row)
            telemetry.collect(); write_json(output/'partial-result.json',result)
            check('after_'+label)
    try:
        telemetry.start(); check('before_reference', reserve_bytes=2*resources['reference_bytes'])
        reference = {name:torch.zeros(shape,device='cuda',dtype=torch.bfloat16)
                     for name,shape in resources['selected_shapes'].items()}
        torch.cuda.synchronize()
        with events.observe():
            for arm in ('full','selected'):
                check.arm = events.arm = arm
                check.deadline = time.monotonic()+resources['arm_deadlines_seconds'][arm]
                check.require_file_cap(resources['legacy_post_hash_absolute_file_cap_bytes'])
                if source_stats(census['model']) != result['original_source_stats']:
                    raise RuntimeError('source changed between the fixed A/B arms')
                weights, report = prepare_arm(arm,inputs=resources['inputs'],census=census,
                    capture_binding=plan['capture'],capture=capture,resources=resources,output=output,
                    check=check,phase=phase,build=build_streamed_causal_lm,
                    profile=detect_profile(census['model']),device='cuda')
                result['arms'].append(report)
                def parity():
                    if set(weights) != set(reference):
                        raise RuntimeError('source snapshot scope differs')
                    ids = {}
                    for name, expected in reference.items():
                        actual = weights[name]
                        if actual.shape != expected.shape or actual.dtype != expected.dtype:
                            raise RuntimeError('source snapshot shape/dtype differs')
                        if arm == 'full':
                            expected.copy_(actual)
                        elif not torch.equal(expected.view(torch.uint8),actual.view(torch.uint8)):
                            raise RuntimeError('source snapshot bit-exact parity failed')
                        ids[name] = _cb_cache_tensor_identity(actual)
                    return ids
                report['weight_identities'] = phase('parity',parity)
                weights = None; gc.collect(); torch.cuda.empty_cache()
                if arm == 'full':
                    check.require_file_cap(resources['legacy_post_hash_absolute_file_cap_bytes'])
            if result['arms'][0]['projection'] != result['arms'][1]['projection']:
                raise RuntimeError('source projection identities differ between arms')
            full = [r for r in events.rows if r['arm']=='full' and r['name'].endswith('.safetensors')]
            selected = [r for r in events.rows if r['arm']=='selected' and r['name'].endswith('.safetensors')]
            expected = {row['name']:row['sha256'] for row in resources['source_payload_roster']}
            if (len(full) != len(expected) or {r['name']:r['sha256'] for r in full} != expected
                    or len(selected) != len({r['name'] for r in selected})
                    or any(r['held_fd'] is None or expected.get(r['name']) != r['sha256'] for r in selected)):
                raise RuntimeError('full/selected source hash evidence differs from frozen roster')
            result['source_hashes'] = dict(full=full,selected=selected)
        if source_stats(census['model']) != result['original_source_stats']:
            raise RuntimeError('original source changed across the pair')
        result['passed'] = True
    except BaseException as error:
        result['failure'] = dict(type=type(error).__name__,message=str(error))
        raise
    finally:
        weights = reference = None; gc.collect(); torch.cuda.empty_cache()
        result['telemetry'] = telemetry.finish(result['phases'])
        result['passed'] = result['passed'] and result['telemetry']['passed']
        result['guard'] = guard.snapshot()
        result['maximum_observed_cgroup_file_bytes'] = check.max_file_charge
        result['final_source_stats'] = source_stats(census['model'])
        result['final_capture_sha256'] = digest(plan['capture']['path'])
        result['artifacts'] = {path.name:dict(sha256=digest(path),bytes=path.stat().st_size)
            for path in sorted(output.iterdir()) if path.is_file()
            and path.name not in ('result.json','partial-result.json')}
        write_json(output/'result.json',result)
    if not result['passed']:
        raise RuntimeError('source A/B evidence incomplete')
    print(json.dumps(dict(passed=True,result=str(output/'result.json'),sha256=digest(output/'result.json'))))

if __name__ == '__main__':
    main()
