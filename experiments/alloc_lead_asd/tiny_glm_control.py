"""Direct tiny-GLM cotangents versus the existing staged Stage A chain.

The spill control feeds the same owned operands into StageBReplaySpill's
existing recording seam. It does not certify the full layer-quantum driver.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load as load_safetensors
from transformers.models.glm5_next import Glm5NextConfig

from experiments.alloc_lead_asd.glm_row_binding import RoutingTap
from experiments.alloc_lead_asd.qualify_tiny_glm import DERIVATIVE, fixture_module
from experiments.alloc_lead_asd.run_pair import runtime_identity
from prismaquant import aura_cost, format_registry
from prismaquant.joint_aura import SignedJointProjectionLease, JointOperatorStatisticsLease, select_invocation_gradient
from prismaquant.perturbed_x_cache import _activation_qdq
from prismaquant.residency_map import bind_residency_manifest, residency_report
from prismaquant.staged_tier_policy import activate_staged_tier_policy
from prismaquant.staged_whole_file import read_staged_whole_file
from prismaquant.glm_source_derivative import bind_source_derivative
from prismaquant.model_profiles.glm5_next import Glm5NextProfile
from prismaquant.routed_experts import profile_declared_packed_expert_projections
from prismaquant.kl_fisher import fisher_probe_scalar
from prismaquant.stage_a_produced_output import BoundaryProducedPublication
from prismaquant.joint_replay_spill import StageBReplaySpill, spill_geometry
from prismaquant.joint_cost_stage_a import run_adjoint_capture_core
import prismaquant.joint_cost_stage_a as stage_a


SPEC = format_registry.get_format('FP8_E4M3')


def sha(value):
    return hashlib.sha256(value).hexdigest()


def dependency_identity():
    import xxhash
    root = Path('/mnt/shared/tessera-measurements/pq1962-sol-20261002/deps-xxhash37')
    record = json.loads((root / 'qualification.json').read_text())
    assert record['wheel_sha256'] == 'f3e7b689c3bce16699efcf736066f5c6cc4472c3840fe4b22bd8279daf4abdac'
    for name, expected in record['files'].items():
        assert sha((root / name).read_bytes()) == expected, f'scoped dependency changed: {name}'
    assert Path(xxhash.__file__).resolve().parent == root / 'xxhash'
    return record


def commit(units, phase):
    from prismaquant.prismabuild_progress import commit as report_progress
    report_progress(units, phase, 'durable_tiny_glm_control')


@contextlib.contextmanager
def profiled(path, device):
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profile = torch.profiler.profile(activities=activities)
    try:
        with profile:
            yield
    finally:
        trace = Path(str(path) + '.trace.json')
        profile.export_chrome_trace(str(trace))
        with trace.open('rb') as handle, gzip.open(str(trace) + '.gz', 'wb') as zipped:
            zipped.write(handle.read())
        trace.unlink()


def difference(a, b):
    a, b = a.double(), b.double()
    d = a - b
    return {'exact': bool(torch.equal(a, b)), 'max_abs': float(d.abs().max()) if d.numel() else 0,
            'relative_rms': float(d.square().mean().sqrt() / a.square().mean().sqrt().clamp_min(1e-30)) if d.numel() else 0}


class RowObservation(SignedJointProjectionLease):
    """Use the production packed observer and gradient selection, retain tiny rows."""

    def __init__(self, modules, model):
        self.records, self.sequences = [], list(range(4))
        self.routing = {}
        for layer, block in enumerate(model.model.language_model.layers):
            if hasattr(block.mlp, 'experts'):
                self.routing[layer] = RoutingTap(block.mlp.experts)
        super().__init__(modules, {name: {SPEC.name: SPEC} for name in modules}, {})

    def _validate_delta_coverage(self, name, module):
        pass  # This observer records operands, not a weight perturbation.

    def _observe(self, name, source_weight, x, output, output_slice=None, row_slice=None):
        if not output.requires_grad:
            return
        layer = int(name.split('.layers.')[1].split('.')[0])
        if '.experts.' in name:
            expert = int(name.split('.experts.')[1].split('.')[0])
            token, slot = self.routing[layer].rows(expert, row_slice, x.reshape(-1, x.shape[-1]).shape[0])
            if name.endswith(('gate_proj', 'up_proj')):
                assert torch.equal(x.reshape(-1, x.shape[-1]), self.routing[layer].hidden[token]), 'expert activation rows differ from routed token identities'
        else:
            token = torch.arange(x.reshape(-1, x.shape[-1]).shape[0], device=x.device)
            slot = torch.full_like(token, -1)
        seq = torch.tensor(self.sequences, device=token.device)[token // 32]
        coords = torch.stack((seq, token % 32, slot), -1).cpu()
        inputs = x.detach().clone()
        dx = (_activation_qdq(inputs, SPEC, {}, name).float() - inputs.float()).detach()
        probe = 0

        def observe(gradient):
            nonlocal probe
            selected = select_invocation_gradient(name, source_weight, inputs, gradient,
                output_slice=output_slice, row_slice=row_slice).detach()
            self.records.append({'unit': name, 'probe': probe, 'coordinates': coords.clone(),
                'x': inputs.cpu().clone(), 'dx': dx.cpu().clone(), 'g': selected.cpu().clone()})
            probe += 1
            return gradient
        output.register_hook(observe)

    def close(self):
        for tap in self.routing.values():
            tap.remove()


def model_and_runner(config, state, dtype, device):
    fixture = fixture_module()
    model = fixture._build_model(Glm5NextConfig.from_dict(config))
    model.load_state_dict(state, strict=True)
    model.config._attn_implementation = model.config.text_config._attn_implementation = 'eager'
    model.config._experts_implementation = model.config.text_config._experts_implementation = 'grouped_mm'
    model = model.to(device=device, dtype=dtype).eval()
    bind_source_derivative(model, Glm5NextProfile(), DERIVATIVE)
    runner = fixture._streamed_runner(model)
    runner.context.device, runner.context.dtype = device, dtype
    runner.device, runner.dtype = device, dtype
    runner.require_prefetched_residency = True
    runner.prefetch_lookahead = 1
    def resident(layer, **kwargs):
        if 0 <= layer < runner.num_layers:
            assert all(not p.is_meta and p.device == device for p in runner.layers[layer].parameters()), 'synthetic source layer is not resident'
        return None
    runner.context.install = resident
    runner.context.schedule_prefetch = resident
    runner.context.settle_prefetch_layers = lambda layers: None
    runner.context.settle_prefetched_layers = lambda layers, **kwargs: None
    runner.context.source_residency_snapshot = lambda layers, **kwargs: {'owners': [],
        'unique_storage_bytes': sum(p.numel() * p.element_size() for p in model.parameters())}
    # Synthetic fixture weights are already resident; the production runner,
    # layer isolation and capture owners are unchanged.
    modules = aura_cost._target_linears(model, include_lm_head=False,
        include_routed_experts=True, profile=runner.profile)
    modules.update({member.qname: member for member in
                    profile_declared_packed_expert_projections(model, runner.profile)})
    modules = {name: module for name, module in modules.items() if '.mlp.' in name}
    return model, runner, modules


def direct(model, modules, ids):
    observed = RowObservation(modules, model)
    boundaries, gradients = {}, {}
    embedding = model.model.language_model.embed_tokens.register_forward_hook(
        lambda module, args, value: value.detach().requires_grad_(True))
    handles = []
    for layer, block in enumerate(model.model.language_model.layers):
        def before(module, args, kwargs, layer=layer):
            value = args[0] if args else kwargs['hidden_states']
            boundaries[layer] = value.detach().cpu().clone()
            p = 0
            def store(g):
                nonlocal p
                gradients[layer, p] = g.detach().cpu().clone()
                p += 1
                return g
            value.register_hook(store)
        handles.append(block.register_forward_pre_hook(before, with_kwargs=True))
    try:
        with observed:
            observed.begin_probe()
            logits = model(input_ids=ids, use_cache=False).logits
            assert torch.isfinite(logits).all()
            for probe in range(2):
                fisher_probe_scalar(logits, seed=7000 + probe, token_scope='all', temperature=1.0,
                    distribution='rademacher', token_count_override=128, global_row_offset=0).backward(retain_graph=probe == 0)
        return {'boundaries': boundaries, 'gradients': gradients, 'rows': observed.records,
                'logits': logits.detach().cpu()}
    finally:
        observed.close()
        embedding.remove()
        for handle in handles:
            handle.remove()


def capture(runner, model, modules, ids, source, output, publication):
    observed = RowObservation(modules, model)
    rolled = {}
    original = stage_a.render_free_layer_roll
    def instrumented(*args, **kwargs):
        layer = kwargs['layer']
        original_roll = kwargs['roll']
        def roll(value, batch, probe):
            rolled[layer, batch, probe] = value.clone()
            return original_roll(value, batch, probe)
        kwargs['roll'] = roll
        return original(*args, **kwargs)
    execution = {'n_probes': 2, 'seed_base': 7000, 'probe_microbatch': 1,
        'boundary_storage': {'schema': 'prismaquant.aura.boundary_storage.v2',
            'directory': str(output / 'boundaries'), 'capture_order': 'layer_major',
            'max_artifact_bytes': 64 << 20, 'max_auxiliary_bytes': 4 << 20,
            'max_resident_bytes': 4 << 20, 'prefetch_batches': 4}}
    try:
        with observed, patch.object(stage_a, 'render_free_layer_roll', instrumented):
            observed.begin_probe()
            receipt = run_adjoint_capture_core(runner, ids.cpu(), execution=execution,
                output_root=output, stride=1, source_model_identity=source,
                unit_roster_sha256=sha(json.dumps(sorted(modules)).encode()),
                plan_sha256=sha(json.dumps(execution, sort_keys=True).encode()),
                prepared_sha256=sha(b'tiny GLM no production rendered candidates'),
                read_manifest_sha256=os.environ['TINY_GLM_MANIFEST_SHA256'],
                implementation_sha256=aura_cost._aura_source_sha256(),
                chain_batch_size=4, chain_probe_fusion=True, produced_output=publication)
        return {'receipt': receipt, 'rolled': rolled, 'rows': observed.records}
    finally:
        observed.close()


def components(rows, modules):
    result = {}
    for row in rows:
        name, p = row['unit'], row['probe']
        x = row['dx'].reshape(-1, row['dx'].shape[-1]).float()
        g = row['g'].reshape(-1, row['g'].shape[-1]).float()
        weight = modules[name].weight.detach().cpu().float()
        fixed = float((g.double() * (x @ weight.T).double()).sum())
        operator = float(((g.T @ x) * weight).sum())
        result[name, p] = result.get((name, p), 0.0) + fixed
        row['fixed_g_dot'] = fixed
        row['operator_dot'] = operator
    return result


def statistics(rows, modules, device, spill=None):
    result = {}
    specs = {name: {SPEC.name: SPEC} for name in modules}
    for probe in range(2):
        with JointOperatorStatisticsLease(modules, specs, max_statistics_bytes=32 << 20,
                max_candidate_bytes=8 << 20) as lease:
            lease.begin_probe()
            if spill is not None:
                spill.replay(0, probe, lease)
            else:
                for row in rows:
                    if row['probe'] == probe:
                        name = row['unit']
                        lease.observe_row_chunk(name, modules[name].weight,
                            row['x'].reshape(-1, row['x'].shape[-1]).to(device),
                            row['g'].reshape(-1, row['g'].shape[-1]).to(device), calls=1)
            lease.finish_observations()
            terms = lease.project({(name, SPEC.name): torch.zeros_like(module.weight, dtype=torch.float32)
                                   for name, module in modules.items()})
            lease.finish_projections()
            result.update({f'{name}/p{probe}': value['activation'] for (name, fmt), value in terms.items()})
    return result


def spill_control(rows, modules, device, root):
    # The existing spill admits 16-bit source operands. FP32's contraction
    # control uses the same statistics owner directly and records that limit.
    root.mkdir(parents=True, exist_ok=False)
    names, specs = tuple(modules), {name: {SPEC.name: SPEC} for name in modules}
    geometry = spill_geometry(modules, [names], pending=set(names), batch_tokens=[128],
        n_probes=2, element_size=2, experts_per_token=2)
    with StageBReplaySpill(root=root, max_bytes=16 << 20, geometry=geometry,
            window_names=[names], n_probes=2, dtype=torch.bfloat16, device=device,
            accumulation='operator_gemm', chunk_rows=65536, threads=False) as session:
        for probe in range(2):
            with session.capture(probe, modules, specs, activation_max_abs={}, projection_backend=None):
                for row in rows:
                    if row['probe'] == probe:
                        session._record(row['unit'], row['x'].to(device), row['g'].to(device))
                session._end_batch()
        values = statistics(rows, modules, device, session)
        session.close_replay_stream()
        # Swap two input rows in the existing scratch reader, retaining its
        # original checksum and metadata. The verified delivery must refuse.
        window = session._windows[0]
        owner, entry = next((owner, entry) for owner, entries in window.entries.items()
                            for entry in entries if entry.layout[0][0] >= 2)
        low = session._physical(window.x_runs[owner], window.x_starts[owner], entry.logical, entry.nbytes)
        row_bytes = entry.layout[0][-1] * 2
        real_read = session._scratch.read_into
        injected = False
        def permuted(offset, views, **kwargs):
            nonlocal injected
            got = real_read(offset, views, **kwargs)
            cursor = offset
            for view in views:
                if cursor <= low and low + 2 * row_bytes <= cursor + len(view):
                    at = low - cursor
                    first, second = bytes(view[at:at + row_bytes]), bytes(view[at + row_bytes:at + 2 * row_bytes])
                    assert first != second, 'selected permutation has identical rows'
                    view[at:at + row_bytes], view[at + row_bytes:at + 2 * row_bytes] = second, first
                    injected = True
                cursor += len(view)
            return got
        refusal = None
        with patch.object(session._scratch, 'read_into', permuted):
            try:
                for item in window.plan:
                    session._read_chunk(window, 0, item)
            except RuntimeError as exc:
                refusal = str(exc)
        assert injected and refusal and 'checksum mismatch' in refusal, (injected, refusal)
        return {'components': values, 'permutation_refused': refusal,
                'scope': 'existing operand recording/replay seam; full quantum capture driver not executed',
                'telemetry': session.telemetry}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-fp32-direct', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--only-dtype', choices=('float32', 'bfloat16'))
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    device = torch.device('cuda', torch.cuda.current_device()) if args.device == 'cuda' else torch.device('cpu')
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    activate_staged_tier_policy('ram,ssd')
    bind_residency_manifest(args.manifest_sha256)
    manifest_raw = args.manifest.read_bytes()
    assert sha(manifest_raw) == args.manifest_sha256
    raw = {}
    for entry in json.loads(manifest_raw)['entries']:
        data = read_staged_whole_file(Path(entry['path']), entry['sha256'], label='tiny-glm-input')
        assert sha(data) == entry['sha256']
        raw[Path(entry['path']).name] = data
    state, ids = load_safetensors(raw['model.safetensors']), load_safetensors(raw['tokens.safetensors'])['calibration_ids']
    config, source = json.loads(raw['config.json']), json.loads(raw['source.json'])
    assert sha(ids.numpy().tobytes()) == source['tokens_sha256'] and list(ids.shape) == [4, 32]
    os.environ['TINY_GLM_MANIFEST_SHA256'] = args.manifest_sha256
    publication = BoundaryProducedPublication.bind_from_admitted_owner()
    (args.output / 'inputs.ready.json').write_text(json.dumps({
        'manifest_sha256': args.manifest_sha256,
        'input_residency': residency_report(),
        'source_identity_sha256': source['model_identity']['content_sha256'],
        'tokens_sha256': source['tokens_sha256'],
    }, indent=2) + '\n')
    commit(1, 'startup')
    results = {'source': source, 'scoped_dependency': dependency_identity(), 'legs': {},
               'scope': 'synthetic tiny GLM; no original GLM tensor reads'}
    if device.type == 'cpu':
        from transformers.models.glm5_next import modeling_glm5_next as modeling
        for name in ('causal_conv1d_fn', 'causal_conv1d_update'):
            function = getattr(modeling, name)
            if hasattr(function, '__wrapped__'):
                setattr(modeling, name, function.__wrapped__)
        results['substitutions'] = ['CPU preflight only: library Torch short-convolution implementations']
    try:
        dtypes = [getattr(torch, args.only_dtype)] if args.only_dtype else (torch.float32, torch.bfloat16)
        for index, dtype in enumerate(dtypes):
            label = str(dtype).split('.')[-1]
            model, runner, modules = model_and_runner(config, state, dtype, device)
            with profiled(args.output / label, device):
                if dtype == torch.float32 and args.reuse_fp32_direct:
                    prior = json.loads(raw['inputs.ready.json'])
                    assert prior['source_identity_sha256'] == source['model_identity']['content_sha256']
                    assert prior['tokens_sha256'] == source['tokens_sha256']
                    direct_result = torch.load(io.BytesIO(raw['float32.direct.pt']), map_location='cpu', weights_only=True)
                    assert {r['unit'] for r in direct_result['rows']} == set(modules)
                else:
                    direct_result = direct(model, modules, ids.to(device))
                torch.save(direct_result, args.output / f'{label}.direct.pt')
                captured = capture(runner, model, modules, ids, source['model_identity'], args.output / label, publication)
                cotangent_differences = {f'L{layer}/p{p}': difference(direct_result['gradients'][layer, p],
                    torch.cat([captured['rolled'][layer, batch, p] for batch in range(4)]))
                    for layer in range(2) for p in range(2)}
                direct_components = components(direct_result['rows'], modules)
                capture_components = components(captured['rows'], modules)
                operator_components = statistics(captured['rows'], modules, device)
                spill = (spill_control(captured['rows'], modules, device,
                    Path(os.environ['PRISMAQUANT_STAGE_B_SPILL_ROOT']) / f'action-{os.getpid()}')
                    if dtype == torch.bfloat16 else {'scope': 'FP32 not admitted by production spill operand dtype contract'})
            tensors = {'direct': direct_result, 'capture': captured}
            torch.save(tensors, args.output / f'{label}.pt')
            results['legs'][label] = {'runtime': runtime_identity(model) if device.type == 'cuda' else {'device': 'cpu', 'torch': torch.__version__},
                'cotangent_differences': cotangent_differences,
                'direct_components': {f'{n}/p{p}': v for (n, p), v in direct_components.items()},
                'capture_components': {f'{n}/p{p}': v for (n, p), v in capture_components.items()},
                'operator_components': operator_components, 'spill': spill,
                'capture_receipt': captured['receipt']}
            if dtype == torch.float32 and args.reuse_fp32_direct:
                results['legs'][label]['direct_reuse'] = {'action': '51bb661a16323d1e59c84197dec752db71e6026baf05be08bdc89be33bcce4a2',
                    'source': 'fa16aac5cfe', 'scope': 'banked direct graph completed before capture refusal; same unchanged direct function/source/inputs/arithmetic'}
            (args.output / 'result.partial.json').write_text(json.dumps(results, indent=2) + '\n')
            commit(index + 2, 'control')
            del model, runner, direct_result, captured, tensors
            torch.cuda.empty_cache()
        results['complete'] = True
        (args.output / 'result.json').write_text(json.dumps(results, indent=2) + '\n')
        commit(2 + len(dtypes), 'publish')
        print('PASS: tiny GLM direct and production capture legs complete; inspect numerical differences')
    finally:
        results['publication_release'] = publication.release()


if __name__ == '__main__':
    main()
