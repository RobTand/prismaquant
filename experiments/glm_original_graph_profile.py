"""Bounded original GLM prefix graph qualification, only through PrismaBuild.

The deterministic boundary stimuli are not downstream Fisher adjoints. This
gate allocates no statistics or candidates and emits no cost/quality claim.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack, contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import traceback
from unittest.mock import patch

import torch
from torch.multiprocessing.reductions import StorageWeakRef

from experiments.glm_original_graph_source import (
    AuthenticatedSourceInputs, checked_json, derive_source_roster)
from experiments.joint_operator_windows_profile import tensor_identity, write_json, sha
from experiments.joint_allocator_turnover_profile import profile_phase, physical_snapshot
from experiments.workspace_netdata import NetdataWriter, sample_netdata

GIB = 1024**3
ROWS, LAYERS, SEEDS = (0, 511), (0, 3, 4), (7000, 7001, 7002, 7003)
ARMS = ('unobserved_isolated_baseline', 'nonfinal_fork_replay', 'final_original_owner_replay')
SHAPE = (1, 512, 4, 4096)
PLAN = Path(__file__).parent / 'measurements/glm-joint-original-graph-plan-20260908/plan.json'


class PrefixGraphQualificationComplete(Exception):
    """Normal bounded-prefix completion, never used for runtime failures."""


def schedule():
    return [(layer, row, seed, arm) for layer in LAYERS for row in ROWS
            for seed in SEEDS for arm in ARMS]


def boundary_policy(directory):
    return dict(schema='prismaquant.aura.boundary_storage.v2', capture_order='layer_major',
        directory=str(directory), max_resident_bytes=17 * GIB // 8,
        max_auxiliary_bytes=2 * GIB, max_artifact_bytes=GIB, prefetch_batches=1)


def require_empty_state(state):
    if state is not None and state != {}:
        raise RuntimeError('original GLM gate requires the declared empty cross-layer pass state')


def state_identity(value):
    if isinstance(value, torch.Tensor):
        return tensor_identity(value)
    if isinstance(value, dict):
        return {str(key): state_identity(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [state_identity(item) for item in value]
    if value is None or type(value) in (str, int, bool, float):
        return value
    raise TypeError(f'opaque source state: {type(value).__name__}')


def tensor_statistics(value):
    """Scalar diagnostic counts/extrema, never a retained activation copy."""
    if value is None:
        return dict(present=False)
    with torch.no_grad():
        value = value.detach()
        finite = torch.isfinite(value)
        count, total = int(finite.count_nonzero()), value.numel()
        zero = int((value == 0).count_nonzero())
        result = dict(present=True, shape=list(value.shape), dtype=str(value.dtype),
            elements=total, finite=count, nonfinite=total-count, zero=zero,
            finite_nonzero=count-zero, nan=int(torch.isnan(value).count_nonzero()),
            positive_inf=int(torch.isposinf(value).count_nonzero()),
            negative_inf=int(torch.isneginf(value).count_nonzero()),
            finite_min=None, finite_max=None)
        if count:
            result['finite_min'] = float(value.amin()) if count == total else float(
                torch.where(finite, value, float('inf')).amin())
            result['finite_max'] = float(value.amax()) if count == total else float(
                torch.where(finite, value, float('-inf')).amax())
        return result


class TensorBranchObservations:
    """Transparent tensor hooks on one actual original forward/backward.

    Module forward observers return None; tensor gradient hooks return None.
    No full-module backward hooks, output views, replacement or second graph.
    """
    NAMES = ('', 'attn_hc', 'input_layernorm', 'self_attn',
        'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
        'self_attn.forget_gate', 'self_attn.o_norm', 'self_attn.o_proj',
        'ffn_hc', 'post_attention_layernorm', 'mlp')

    def __init__(self, module, records):
        self.module, self.records, self.handles = module, records, []

    def observe(self, name, value):
        if isinstance(value, torch.Tensor):
            self.records.append(dict(site=name, phase='forward', statistics=tensor_statistics(value)))
            if value.requires_grad:
                def gradient(grad):
                    self.records.append(dict(site=name, phase='backward', statistics=tensor_statistics(grad)))
                    return None
                self.handles.append(value.register_hook(gradient))
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                self.observe(f'{name}[{index}]', child)

    def __enter__(self):
        for name in self.NAMES:
            try:
                module = self.module.get_submodule(name)
            except AttributeError:
                continue
            def hook(_module, args, kwargs, output, name=name):
                hidden = args[0] if args else kwargs.get('hidden_states')
                self.observe(f'{name or "layer"}.input', hidden)
                self.observe(f'{name or "layer"}.output', output)
                return None
            self.handles.append(module.register_forward_hook(hook, with_kwargs=True))
        return self

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.module = None


def module_identity(module):
    result = {}
    for kind, values in (('parameter', module.named_parameters()), ('buffer', module.named_buffers())):
        for name, value in values:
            if value.is_meta or (kind == 'parameter' and value.requires_grad):
                raise RuntimeError('graph gate requires resident frozen original source parameters')
            flat = value.detach().reshape(-1)
            count = min(64, flat.numel())
            positions = [i * (flat.numel()-1) // max(1, count-1) for i in range(count)]
            indices = torch.tensor(positions, dtype=torch.int64, device=flat.device)
            result[f'{kind}:{name}'] = dict(ptr=value.data_ptr(), version=value._version,
                shape=list(value.shape), dtype=str(value.dtype), samples=tensor_identity(flat[indices]))
    result['training'] = [name for name, item in module.named_modules() if item.training]
    if result['training']:
        raise RuntimeError('original source module is not in evaluation mode')
    return result


@contextmanager
def unchanged_execution(module, batch, pass_state):
    if torch.is_inference_mode_enabled():
        raise RuntimeError('original graph replay cannot run under inference_mode')
    require_empty_state(pass_state)
    before_module = module_identity(module)
    metadata = lambda: state_identity((batch.input_ids, batch.position_ids,
        batch.position_embeddings, batch.attention_mask, batch.shared_pass_state, pass_state))
    before_metadata = metadata()
    cpu = torch.get_rng_state()
    device = next(module.parameters()).device
    cuda = torch.cuda.get_rng_state(device) if device.type == 'cuda' else None
    yield
    if module_identity(module) != before_module or metadata() != before_metadata:
        raise RuntimeError('original graph replay mutated module/source metadata/pass state')
    if not torch.equal(cpu, torch.get_rng_state()) or (cuda is not None and
            not torch.equal(cuda, torch.cuda.get_rng_state(device))):
        raise RuntimeError('original graph replay consumed Torch RNG')


class Activity:
    """Observe original modules without substituting their decorated kernels."""
    def __init__(self, module):
        self.module, self.handles, self.rows = module, [], {}

    def __enter__(self):
        def shape_hook(name):
            def hook(_module, args, _output):
                self.rows.setdefault(name, []).append(list(args[0].shape))
            return hook
        for name in ('attn_hc', 'ffn_hc', 'mlp.down_proj', 'mlp.shared_experts'):
            try:
                item = self.module.get_submodule(name)
            except AttributeError:
                continue
            self.handles.append(item.register_forward_hook(shape_hook(name)))
        def cache_hook(_module, _args, kwargs):
            for key in ('past_key_values', 'cache_params', 'prev_topk_indices'):
                if kwargs.get(key) is not None:
                    raise RuntimeError('original gate unexpectedly received mutable KV/indexer cache')
            if kwargs.get('use_cache', False):
                raise RuntimeError('original graph gate unexpectedly enabled KV caching')
        self.handles.append(self.module.register_forward_pre_hook(cache_hook, with_kwargs=True))
        self.handles.append(self.module.self_attn.register_forward_pre_hook(cache_hook, with_kwargs=True))
        if hasattr(self.module.mlp, 'gate'):
            def router_hook(_module, _args, output):
                ids = output[2].detach().to('cpu')
                if ids.dtype not in (torch.int32, torch.int64) or tuple(ids.shape) != (512, 8):
                    raise RuntimeError('original router does not cover512 tokens x8 assignments')
                counts = torch.bincount(ids.flatten(), minlength=288)
                if len(counts) != 288 or int(counts.sum()) != 4096:
                    raise RuntimeError('original routed expert assignment coverage differs')
                self.rows['router'] = dict(ids=tensor_identity(ids), counts=counts.tolist(),
                    hit_experts=int((counts > 0).sum()), assignments=4096)
            self.handles.append(self.module.mlp.gate.register_forward_hook(router_hook))
        return self

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()

    def validate(self, layer):
        for name in ('attn_hc', 'ffn_hc'):
            if self.rows.get(name) != [list(SHAPE)]:
                raise RuntimeError('original HC graph shape/call count differs')
        if layer == 0 and self.rows.get('mlp.down_proj') != [[1, 512, 12288]]:
            raise RuntimeError('widest original dense MLP path was not exercised')
        if layer in (3, 4) and ('router' not in self.rows or
                self.rows.get('mlp.shared_experts') != [[1, 512, 4096]]):
            raise RuntimeError('original routed/shared MLP path was not exercised')


def replay_arm(runner, layer, hidden, batch, pass_state, seed, arm, owner, *,
               diagnostics=None, observe_branches=False):
    """One disposable graph through existing isolated_layer and shared owner API."""
    from prismaquant.sensitivity_probe import SharedStateCotangents
    if arm not in ARMS:
        raise ValueError('unknown original graph replay arm')
    if diagnostics is None:
        diagnostics = {}
    shared = (SharedStateCotangents(enabled=True) if arm == ARMS[0] else
        owner.fork_for_replay(max_resident_bytes=GIB//4) if arm == ARMS[1] else owner)
    leaf = output = delta = gradient = None
    try:
        with unchanged_execution(runner.layers[layer], batch, pass_state), torch.enable_grad(), ExitStack() as hooks:
            leaf = hidden.detach().requires_grad_(True)
            state = shared.graft(pass_state)
            generator = torch.Generator(device='cpu').manual_seed(seed)
            delta = (torch.randint(0, 2, tuple(hidden.shape), generator=generator,
                                   dtype=torch.int8).to(torch.bfloat16).mul_(2).sub_(1).div_(256)
                     ).to(hidden.device)
            diagnostics.update(input=tensor_statistics(leaf), stimulus=tensor_statistics(delta))
            if observe_branches:
                hooks.enter_context(TensorBranchObservations(runner.layers[layer],
                    diagnostics.setdefault('branches', [])))
            # Activity is separate from numerical identity; baseline has no graph hooks.
            if arm == ARMS[0]:
                output = runner.isolated_layer(batch, layer, leaf, pass_state=state)
                activity = None
            else:
                with Activity(runner.layers[layer]) as observed:
                    output = runner.isolated_layer(batch, layer, leaf, pass_state=state)
                observed.validate(layer)
                activity = observed.rows
            diagnostics['output'] = tensor_statistics(output)
            diagnostics['output_identity'] = tensor_identity(output)
            if 'primary_output_identity' in diagnostics:
                diagnostics['output_matches_primary'] = diagnostics['output_identity'] == diagnostics['primary_output_identity']
                if not diagnostics['output_matches_primary']:
                    raise RuntimeError('replayed output differs from original prefix output')
            if diagnostics['output']['nonfinite']:
                raise RuntimeError('original graph replay output contains nonfinite elements')
            roots, grads = shared.produced_roots()
            diagnostics['backward_started'] = True
            torch.autograd.backward([output, *roots], [delta, *grads])
            diagnostics['backward_completed'] = True
            shared.harvest()
            gradient = leaf.grad
            diagnostics['leaf_gradient'] = tensor_statistics(gradient)
            if gradient is None:
                raise RuntimeError('original graph input cotangent is absent')
            if diagnostics['leaf_gradient']['nonfinite']:
                raise RuntimeError('original graph input cotangent contains nonfinite elements')
            if not diagnostics['leaf_gradient']['finite_nonzero']:
                raise RuntimeError('original graph input cotangent is all zero')
            if shared.pending_keys() or shared.resident_tensors():
                raise RuntimeError('empty original GLM state produced retained shared adjoints')
            return dict(seed=seed, arm=arm, output=tensor_identity(output),
                cotangent=tensor_identity(gradient), stimulus=tensor_identity(delta), activity=activity,
                shared=dict(grafted=shared.n_grafted, harvested=shared.n_harvested, seeded=shared.n_seeded))
    finally:
        if leaf is not None:
            leaf.grad = None
        leaf = output = delta = gradient = None
        shared.release_resident_state()


class GraphObserver:
    def __init__(self, runner, out, result, settle, *, replay=replay_arm, diagnostic=False):
        self.runner, self.out, self.result, self.settle = runner, out, result, settle
        self.original, self.replay = runner._call, replay
        self.in_replay, self.row = False, None
        self.diagnostic = diagnostic
        self.layers, self.rows, self.seeds, self.arms = (
            ((0,), (0,), (7000,), (ARMS[0],)) if diagnostic else (LAYERS, ROWS, SEEDS, ARMS))

    def expected_schedule(self):
        return [(layer, row, seed, arm) for layer in self.layers for row in self.rows
                for seed in self.seeds for arm in self.arms]

    def __call__(self, layer, hidden, *, batch, pass_state):
        if self.in_replay:
            return self.original(layer, hidden, batch=batch, pass_state=pass_state)
        with unchanged_execution(self.runner.layers[layer], batch, pass_state):
            original = self.original(layer, hidden, batch=batch, pass_state=pass_state)
        if layer not in self.layers:
            return original
        if tuple(hidden.shape) != SHAPE or hidden.dtype != torch.bfloat16:
            raise RuntimeError('original graph boundary shape/dtype differs')
        from prismaquant.sensitivity_probe import SharedStateCotangents
        self.settle(layer)
        primary_stats = tensor_statistics(original)
        primary = tensor_identity(original)
        self.result.setdefault('primary_outputs', []).append(dict(layer=layer,
            original_row=self.row, statistics=primary_stats, identity=primary))
        if primary_stats['nonfinite']:
            raise RuntimeError('original prefix output contains nonfinite elements')
        incoming = tensor_identity(hidden)
        self.in_replay = True
        try:
            for seed in self.seeds:
                owner = SharedStateCotangents(enabled=True)
                baseline = None
                try:
                    for arm in self.arms:
                        diagnostics = dict(layer=layer, original_row=self.row, seed=seed, arm=arm,
                            primary_output_identity=primary)
                        self.result.setdefault('replay_diagnostics', []).append(diagnostics)
                        call = lambda: self.replay(self.runner, layer, hidden, batch, pass_state, seed, arm, owner,
                            diagnostics=diagnostics, observe_branches=self.diagnostic)
                        if self.row == 0 and seed == SEEDS[0]:
                            value, profile = profile_phase(self.out, f'layer{layer}_{arm}', call)
                        else:
                            value, profile = call(), None
                        if value['output'] != primary:
                            raise RuntimeError('replayed output differs from original prefix output')
                        if tensor_identity(hidden) != incoming:
                            raise RuntimeError('replay mutated the original incoming boundary')
                        if baseline is None:
                            baseline = value
                        elif value['cotangent'] != baseline['cotangent'] or value['stimulus'] != baseline['stimulus']:
                            raise RuntimeError('replay cotangent/stimulus differs from isolated baseline')
                        value.update(layer=layer, original_row=self.row, profile=profile)
                        self.result['backwards'].append(value)
                    if not self.diagnostic and self.result['backwards'][-1]['activity'] != self.result['backwards'][-2]['activity']:
                        raise RuntimeError('original route/activity differs across fork/final replay')
                finally:
                    owner.release_resident_state()
        finally:
            self.in_replay = False
        return original

    def visit(self, layer, forward_batch):
        for row, tokens in zip(self.rows, self.tokens):
            self.row = row
            forward_batch(tokens)
        self.result['progress'] = dict(phase='prefix_layer_completed', layer=layer,
            backward_calls=len(self.result['backwards']), time_unix=time.time())
        write_json(self.out/'progress.json', self.result)
        if layer == (0 if self.diagnostic else 4):
            actual = [(r['layer'], r['original_row'], r['seed'], r['arm']) for r in self.result['backwards']]
            if actual != self.expected_schedule():
                raise RuntimeError(f'original graph qualification did not complete its exact{len(self.expected_schedule())}-call schedule')
            raise PrefixGraphQualificationComplete()


def metadata_gate(runner, tokens, storage):
    from prismaquant.cost_streaming import StreamedForwardBoundaries, _state_tensors
    batches, weak = [], []
    storage.watch_auxiliary(batches, [])
    with torch.no_grad():
        for row in tokens:
            ids, positions, hidden, embeddings, mask = runner._prepare(row.unsqueeze(0))
            state = runner.profile.new_forward_pass_state()
            require_empty_state(state)
            batch = StreamedForwardBoundaries(ids, positions, embeddings, mask, [], state)
            batches.append(batch)
            storage.check_auxiliary(batches)
            weak.extend(StorageWeakRef(v.untyped_storage()) for v in _state_tensors(
                (ids, positions, embeddings, mask, state)))
            del hidden, ids, positions, embeddings, mask, state, batch
    result = dict(rows=len(batches), telemetry=dict(storage.telemetry), graph_calls=0)
    batches.clear()
    gc.collect()
    result['metadata_owners_expired'] = all(ref.expired() for ref in weak)
    if not result['metadata_owners_expired']:
        raise RuntimeError('full-row metadata gate retained metadata owners')
    return result


def close_source(runner, observer, *, owned=None):
    """Join existing source workers before unbinding readers or dropping owners."""
    if owned is None:
        owned = []
    owned.extend(StorageWeakRef(value.untyped_storage()) for value in
        [*runner.model.parameters(), *runner.model.buffers()] if not value.is_meta)
    for values in runner.context.layer_cache._cache.values():
        owned.extend(StorageWeakRef(value.untyped_storage()) for value in values.values())
    with runner.context._inflight_lock:
        for future in runner.context._inflight.values():
            if future.done() and not future.cancelled() and future.exception() is None:
                values = future.result()
                if values:
                    owned.extend(StorageWeakRef(value.untyped_storage()) for value in values.values())
    try:
        runner.shutdown()
        runner.context.reset_between_chunks(retain_cache=False)
    finally:
        if observer is not None:
            observer.runner = observer.original = None
            observer.tokens = []
    return owned


def finish_native_observation(result, stop, threads, samples, guard, owned):
    """A poisoned CUDA context must not suppress host telemetry or the first error."""
    stop.set()
    for thread in threads:
        thread.join(timeout=12)
    result['memory_samples'] = list(samples)
    result['source_owners_expired'] = all(ref.expired() for ref in owned) if owned else None
    for name, call in (
        ('gc', gc.collect), ('cuda_synchronize', torch.cuda.synchronize),
        ('peak_allocated_bytes', torch.cuda.max_memory_allocated),
        ('peak_reserved_bytes', torch.cuda.max_memory_reserved),
        ('empty_cuda_cache', torch.cuda.empty_cache), ('guard', guard.snapshot),
        ('after_cleanup', lambda: physical_snapshot(torch.device('cuda'))),
    ):
        try:
            value = call()
            if name not in ('gc', 'cuda_synchronize', 'empty_cuda_cache'):
                result[name] = value
        except BaseException as error:
            result.setdefault('cleanup_errors', []).append(dict(phase=name, error=repr(error)))
    result['source_owners_expired'] = all(ref.expired() for ref in owned) if owned else None


def preflight(plan, source, *, diagnostic=False):
    from prismaquant.model_profiles.glm5_next import Glm5NextProfile
    config = source.read_metadata('config.json')
    index = source.read_metadata('model.safetensors.index.json')['weight_map']
    for name, expected in (('config.json', plan['config_sha256']),
                           ('model.safetensors.index.json', plan['index_sha256'])):
        if source.authenticated[str(source.root/name)]['actual_sha256'] != expected:
            raise ValueError('pinned original model metadata differs from graph plan')
    text = config['text_config']
    if (text['num_hidden_layers'] != 45 or text['hidden_size'] != 4096 or text['hc_mult'] != 4
            or text['intermediate_size'] != 12288 or text['n_routed_experts'] != 288
            or text['num_experts_per_tok'] != 8 or set(text['indexer_types']) != {'full'}):
        raise ValueError('original full GLM config differs from qualified graph geometry')
    if ([text['layer_types'][i] for i in LAYERS] !=
            ['linear_attention', 'deepseek_sparse_attention', 'linear_attention'] or
            [text['mlp_layer_types'][i] for i in LAYERS] != ['dense', 'sparse', 'sparse']):
        raise ValueError('original GLM attention/MLP qualification layer types differ')
    profile = Glm5NextProfile()
    source.bind_roster(derive_source_roster(source.root, index, profile, last_layer=1 if diagnostic else 5))
    path = plan['calibration_input']['path']
    if sha(path) != plan['calibration_input']['sha256']:
        raise ValueError('sealed calibration input content differs')
    from safetensors.torch import load_file
    values = load_file(path)
    if len(values) != 1:
        raise ValueError('sealed calibration file has unexpected tensors')
    tokens = next(iter(values.values()))
    if tuple(tokens.shape) != (512, 512) or tokens.dtype != torch.int64:
        raise ValueError('sealed calibration tensor geometry differs')
    fit = hashlib.sha256(tokens.to(torch.int32).contiguous().numpy().tobytes()).hexdigest()
    if fit != plan['calibration_input']['fit_ids_sha256']:
        raise ValueError('sealed calibration token identities differ from canonical fit')
    return profile, tokens, {str(source.root / value) for value in index.values()}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu-preflight', action='store_true')
    parser.add_argument('--diagnostic-layer0', action='store_true',
        help='One original row0/layer0/seed7000 backward with transparent tensor hooks; not qualification')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    plan = json.loads(PLAN.read_text())
    manifest = checked_json(plan['source_files']['path'], plan['source_files']['sha256'])
    result = dict(schema='prismaquant.glm_original_graph_qualification.v1', status='running',
        scope=plan['limits'], backwards=[], telemetry_errors=[], cpu_preflight=args.cpu_preflight,
        mode='layer0_diagnostic_not_qualification' if args.diagnostic_layer0 else 'bounded_prefix_qualification')
    source = AuthenticatedSourceInputs(plan['model'], manifest)
    try:
        with source:
            profile, tokens, shards = preflight(plan, source, diagnostic=args.diagnostic_layer0)
            result['source_preflight'] = source.report()
            if args.cpu_preflight:
                result['status'] = 'cpu_preflight_complete_native_not_run'
                return
            run_native(args, plan, source, profile, tokens, shards, result)
            result['status'] = 'diagnostic_complete_not_qualification' if args.diagnostic_layer0 else 'complete'
    except BaseException:
        result.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        result['source_final'] = source.report()
        write_json(args.out/'result.json', result)


def run_native(args, plan, source, profile, tokens, shards, result):
    from prismaquant.autoscale import require_bounded_capture_environment
    from prismaquant.cost_streaming import build_streamed_causal_lm, StreamedBoundaryArtifacts
    from prismaquant.memory_management import CaptureMemoryGuard
    from transformers.models.glm5_next import modeling_glm5_next
    require_bounded_capture_environment(os.environ)
    if sha(modeling_glm5_next.__file__) != plan['image']['modeling_file_sha256']:
        raise RuntimeError('native original model code differs from pinned qualified image')
    if not torch.cuda.is_available() or torch.is_inference_mode_enabled():
        raise RuntimeError('native graph gate requires CUDA outside inference_mode')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    guard = CaptureMemoryGuard('cuda')
    if guard.cap_bytes != 104*GIB:
        raise RuntimeError('native graph cgroup is not the104GiB admission')
    stop, samples = threading.Event(), deque(maxlen=20000)
    def monitor():
        try:
            with (args.out/'netdata.jsonl').open('x') as stream:
                writer = NetdataWriter(stream)
                while not stop.is_set():
                    for host in ('sparky', 'sparklina'):
                        writer.write(sample_netdata(host))
                    stop.wait(1)
        except BaseException as error:
            result['telemetry_errors'].append(repr(error))
    def memory_monitor():
        try:
            while not stop.is_set():
                samples.append(dict(time=time.time(), **physical_snapshot(torch.device('cuda'))))
                stop.wait(.1)
        except BaseException as error:
            result['telemetry_errors'].append(repr(error))
    threads = [threading.Thread(target=fn, daemon=True) for fn in (monitor, memory_monitor)]
    for thread in threads:
        thread.start()
    runner = observer = None
    owned = []
    source_cleanup_started = False
    def cleanup_source():
        nonlocal runner, observer, owned, source_cleanup_started
        if runner is None or source_cleanup_started:
            return
        source_cleanup_started = True
        try:
            close_source(runner, observer, owned=owned)
        except BaseException as error:
            result.setdefault('cleanup_errors', []).append(dict(phase='source_owners', error=repr(error)))
        finally:
            observer = None
            runner = None
    def check(label, reserve_bytes=0):
        guard.check(label, reserve_bytes=reserve_bytes)
        if torch.cuda.memory_reserved() + reserve_bytes > 92*GIB:
            raise RuntimeError('original graph GPU subset admission exceeded')
    try:
        result['progress'] = dict(phase='authenticating_original_payloads', time_unix=time.time())
        write_json(args.out/'progress.json', result)
        source.authenticate_payloads()
        result['source_authentication'] = source.report()['authenticated']
        result['progress'] = dict(phase='source_authenticated_loading_model', time_unix=time.time())
        write_json(args.out/'progress.json', result)
        with source.reader_binding(all_indexed_shards=shards), ExitStack() as cleanup:
            cleanup.callback(cleanup_source)
            check('before_original_fixed_source', reserve_bytes=48*GIB)
            runner = build_streamed_causal_lm(str(source.root), device=torch.device('cuda'),
                dtype=torch.bfloat16, offload_folder=str(args.out/'offload'), profile=profile,
                max_cache_slots=2, prefetch_workers=1, prefetch_min_available_gb=24,
                cache_headroom_gb=24, prefetch_lookahead=1,
                require_prefetched_residency=True, attn_implementation='eager')
            runner.model.eval().requires_grad_(False)
            # Observe actual reads after the existing prepare_for_load eviction.
            from prismaquant import streaming_model
            original_read = streaming_model._read_layer_to_device
            def guarded_read(*read_args, **read_kwargs):
                check('before_original_source_load',
                      reserve_bytes=runner.context.estimated_layer_bytes + 16*GIB)
                value = original_read(*read_args, **read_kwargs)
                check('after_original_source_load')
                return value
            if not args.diagnostic_layer0:
                with StreamedBoundaryArtifacts(boundary_policy(args.out/'metadata')) as storage:
                    storage.bind({'scope': 'original_all512_metadata_only'}, n_probes=4,
                                 check_memory=lambda label: check(label))
                    result['metadata'] = metadata_gate(runner, tokens, storage)
            result['progress'] = dict(phase='starting_layer0_diagnostic' if args.diagnostic_layer0 else
                'metadata_complete_starting_prefix', time_unix=time.time())
            write_json(args.out/'progress.json', result)
            def settle(layer):
                with runner.context._inflight_lock:
                    future = runner.context._inflight.get(layer+1)
                if future is None or future.result() is None:
                    raise RuntimeError('original graph measurement needs existing forward lookahead residency')
                source.require_unchanged()
                check('settled_original_graph_workspace', reserve_bytes=16*GIB)
                result.setdefault('source_residency', []).append(dict(layer=layer,
                    row=observer.row, snapshot=runner.context.source_residency_snapshot([layer, layer+1])))
            observer = GraphObserver(runner, args.out, result, settle, diagnostic=args.diagnostic_layer0)
            observer.tokens = [tokens[row].unsqueeze(0) for row in observer.rows]
            with StreamedBoundaryArtifacts(boundary_policy(args.out/'prefix')) as storage:
                storage.bind({'scope':'original_row0_layer0_diagnostic' if args.diagnostic_layer0 else
                    'original_rows0_511_prefix0_4'}, n_probes=4,
                             check_memory=lambda label: check(label))
                with patch.object(runner, '_call', observer), patch.object(
                        streaming_model, '_read_layer_to_device', guarded_read):
                    try:
                        runner.visit_layer_batches(observer.tokens, observer.visit, boundary_storage=storage)
                    except PrefixGraphQualificationComplete:
                        result['diagnostic_completed' if args.diagnostic_layer0 else 'bounded_prefix_completed'] = True
                    else:
                        raise RuntimeError('original prefix did not stop at its declared bound')
                result['boundary_telemetry'] = dict(storage.telemetry)
            check('original_graph_complete')
    finally:
        cleanup_source()
        finish_native_observation(result, stop, threads, samples, guard, owned)
    if any(thread.is_alive() for thread in threads) or result['telemetry_errors']:
        raise RuntimeError('original graph telemetry did not complete successfully')
    if not result['source_owners_expired']:
        raise RuntimeError('original graph cleanup retained original source owners')
    if result.get('cleanup_errors'):
        raise RuntimeError('original graph cleanup reported errors')


if __name__ == '__main__':
    main()
