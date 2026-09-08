"""PB-only original-wire screen; requires a frozen plan and complete capture.

The selected-source authentication API must be reviewed and integrated before
this entry point can run. It never treats a partial capture as canonical.
"""
from __future__ import annotations

import argparse
import cProfile
from contextlib import ExitStack
from dataclasses import asdict
import gc
import inspect
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace

from experiments.glm_native_wire_screen_plan import sealed_json, validate_cells
from experiments.glm_native_wire_screen_evidence import (
    ScreenTelemetry, telemetry_coverage, require_original_input_bytes)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def require_source_api(cc, tc, build):
    """No fallback to whole-checkpoint hashing or an unauthenticated reader."""
    if not callable(getattr(cc, 'authenticate_selected_capture_source', None)):
        raise RuntimeError('reviewed selected-source authentication is not integrated')
    if any('source_authentication' not in inspect.signature(fn).parameters
           for fn in (build, tc._checked_projected_units)):
        raise RuntimeError('reviewed selected-source consumer propagation is not integrated')
    if 'verified_load_policy' not in inspect.signature(cc.prefetch_capture).parameters:
        raise RuntimeError('verified capture loader is not integrated')


def require_frozen_environment(environment, activation_environment, environ):
    required = dict(activation_environment, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', PRISMAQUANT_LAYER_READ_THREADS='4',
        PRISMAQUANT_RELEASE_SOURCE_PAGES='1', MIMALLOC_PURGE_DELAY='0',
        PRISMAQUANT_NVFP4_INPUT_GSCALE_FP8_RANGE='0')
    if any(environment.get(name) != value for name, value in required.items()):
        raise ValueError('frozen environment omits a required activation/thread/release contract')
    for name, expected in environment.items():
        if environ.get(name, '') != expected:
            raise ValueError(f'frozen screen environment differs: {name}')


def integrated_cpu_preflight(environment, activation_environment, *, environ):
    """Check actual reviewed APIs and environment without opening a capture."""
    import torch
    from prismaquant import tessera_calibration_cache as cc, tessera_campaign as tc
    from prismaquant.cost_streaming import build_streamed_causal_lm
    from prismaquant.autoscale import require_bounded_capture_environment
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU preflight must precede CUDA initialization')
    require_source_api(cc, tc, build_streamed_causal_lm)
    require_frozen_environment(environment, activation_environment, environ)
    require_bounded_capture_environment(environ)
    return dict(schema='prismaquant.glm_screen_cpu_preflight.v1', passed=True,
        gpu_initialized=False, source_api_module=cc.authenticate_selected_capture_source.__module__,
        checked_environment=dict(environment),
        capture_acceptance='No capture accepted; native execution still requires the complete public contract.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--cpu-preflight', action='store_true')
    args = parser.parse_args()
    plan = sealed_json(args.plan, args.plan_sha256)
    schemas = {'prismaquant.glm_native_wire_screen.frozen.v1'}
    if args.cpu_preflight:
        schemas.add('prismaquant.glm_native_wire_screen.draft.v1')
    if plan.get('schema') not in schemas:
        raise ValueError('native execution requires the reviewed frozen screen envelope')
    resources = sealed_json(plan['resources']['path'], plan['resources']['sha256'])
    source_inputs = resources['inputs']
    proposal = sealed_json(source_inputs['proposal']['path'], source_inputs['proposal']['sha256'])
    census = sealed_json(source_inputs['census']['path'], source_inputs['census']['sha256'])
    groups = sealed_json(source_inputs['group_contract']['path'], source_inputs['group_contract']['sha256'])
    union = sealed_json(source_inputs['union_proposal']['path'], source_inputs['union_proposal']['sha256'])
    shapes = validate_cells(proposal, census, groups)
    if (resources['cells'] != proposal['cells'] or
            proposal['group_contract_sha256'] != source_inputs['group_contract']['sha256']):
        raise ValueError('resource and native proposal rosters differ')
    if any(census.get(key) != value for key, value in dict(nsamples=512, seqlen=512, seed=0).items()):
        raise ValueError('screen requires the original 512 by 512 seed-zero census')
    preflight = integrated_cpu_preflight(plan['environment'],
        union['activation_operators']['explicit_environment_for_binding'], environ=os.environ)
    if args.cpu_preflight:
        print(json.dumps(preflight))
        return

    import torch
    from prismaquant import tessera_calibration_cache as cc, tessera_campaign as tc
    from prismaquant import tessera_hessian as th
    from prismaquant.autoscale import require_bounded_capture_environment
    from prismaquant.cost_streaming import build_streamed_causal_lm
    from prismaquant.memory_management import CaptureMemoryGuard
    from prismaquant.model_profiles import detect_profile
    from prismaquant.perturbed_x_cache import VERIFIED_ACTIVATION_LOAD_SCHEMA
    from prismaquant.production_weight_cache import ProductionWeightCache, _cb_cache_tensor_identity
    from prismaquant.tessera_joint_aura import verify_anchor_render
    require_source_api(cc, tc, build_streamed_causal_lm)
    capture = cc.require_capture_contract(plan['capture']['path'],
        expected_sha256=plan['capture']['sha256'])
    if (capture['identity']['census_sha256'] != source_inputs['census']['sha256'] or
            capture['identity']['max_act_rows'] != 512):
        raise ValueError('complete capture does not match the frozen source census/prefix')
    for name in ('fit_ids_sha256', 'text_sha256', 'nsamples', 'seqlen', 'seed'):
        if capture['identity']['calibration'][name] != union['fixed']['calibration'][name]:
            raise ValueError('complete capture differs from the original sealed calibration draw')
    api = tc._checkpoint_identity_api()
    if api.encoder_source_sha256() != plan['encoder_source_sha256']:
        raise ValueError('actual Tessera encoder source differs from the frozen producer')
    from prismaquant.tessera_runtime_contract import contract_path
    if cc.sha256(contract_path()) != plan['producer_contract_sha256']:
        raise ValueError('actual packaged reader contract differs from the frozen producer')
    require_bounded_capture_environment(os.environ)
    torch.set_num_threads(1)
    torch.cuda.init()
    guard = CaptureMemoryGuard('cuda')
    if guard.cap_bytes < resources['requested_mem_gib']*1024**3:
        raise RuntimeError('PB physical cap is smaller than the frozen guard envelope')
    output = Path(plan['output'])
    output.mkdir(parents=True, exist_ok=False)
    cache_dir, wire_dir = output/'renders', output/'wire'
    cache_dir.mkdir(); wire_dir.mkdir()
    result = dict(schema='prismaquant.glm_native_wire_screen.result.v1', passed=False,
        plan_sha256=args.plan_sha256, source_forward_count=0, cells=[], phases=[],
        candidates=[dict(qname=cell['qname'], format=cell['format'], status='pending')
                    for cell in proposal['cells']],
        scope='28 diagnostic original-wire cells; no complete-group or serving qualification',
        environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda), cpu_preflight=preflight,
        producer=dict(encoder_source_sha256=plan['encoder_source_sha256'],
            contract_sha256=plan['producer_contract_sha256'],
            container=union['fixed']['producer_container']),
        resource_plan=resources, capture=plan['capture'], started_unix=time.time())

    def check(label, **kwargs):
        telemetry.require_healthy()
        value = guard.check(label, **kwargs)
        if torch.cuda.memory_reserved() > resources['gpu_gib']*1024**3:
            raise RuntimeError('screen CUDA reservation exceeds its frozen GPU subset cap')
        return value

    def phase(label, function):
        telemetry.collect(); telemetry.require_healthy()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start, start_mono = time.time(), time.monotonic()
        cpu = cProfile.Profile(); cpu.enable()
        try:
            value = function(); torch.cuda.synchronize()
            return value
        finally:
            end, end_mono = time.time(), time.monotonic()
            cpu.disable(); cpu.dump_stats(str(output/(label+'.cprofile')))
            telemetry.collect()
            result['phases'].append(dict(phase=label, started_unix=start,
                finished_unix=end, started_monotonic=start_mono, finished_monotonic=end_mono,
                profiler='cProfile', gpu_kernel_attribution='unavailable',
                max_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                max_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                guard=guard.snapshot()))
            write_json(output/'partial-result.json', result)
            check('after_'+label)

    telemetry = ScreenTelemetry(output/'netdata.jsonl')
    active_candidate = None
    try:
        telemetry.start()
        profile = detect_profile(census['model'])
        with ExitStack() as scope:
            owner = scope.enter_context(cc.authenticate_selected_capture_source(
                source_inputs['census']['path'], plan['capture']['path'],
                expected_sha256=plan['capture']['sha256'], model=census['model'],
                max_act_rows=512, attention_implementation=census['attention_implementation'],
                calibration_parameters=dict(nsamples=512, seqlen=512, seed=0),
                resource_check=check, release_read_pages=True))

            def prepare_source():
                runner = build_streamed_causal_lm(census['model'], device=torch.device('cuda'),
                    dtype=torch.bfloat16, profile=profile, offload_folder=str(output/'source-offload'),
                    max_cache_slots=2, prefetch_workers=1, cache_headroom_gb=24,
                    prefetch_min_available_gb=24, prefetch_lookahead=1,
                    require_prefetched_residency=True,
                    attn_implementation=census['attention_implementation'], source_authentication=owner)
                try:
                    return runner.snapshot_selected_weights(sorted(shapes),
                        max_resident_bytes=resources['selected_source_weight_bytes'], resource_check=check)
                finally:
                    runner.shutdown()

            weights, result['selected_source'] = phase('source_preparation', prepare_source)
            gc.collect(); torch.cuda.empty_cache()
            projection = census['expert_projection']
            projected = phase('source_projection_check', lambda: tc._checked_projected_units(
                projection['stacks'], weights=weights, model_path=census['model'],
                source=projection['producer']['source'], measured=set(shapes),
                resource_check=check, release_source_pages=True, source_authentication=owner))
            result['source_authentication'] = owner.receipt()
        policy = dict(schema=VERIFIED_ACTIVATION_LOAD_SCHEMA,
            max_buffer_bytes=resources['max_capture_file_bytes'],
            max_scratch_bytes=resources['max_validation_scratch_bytes'])
        values, result['capture_prefetch'] = phase('capture_prefetch', lambda: cc.prefetch_capture(
            plan['capture']['path'], expected_sha256=plan['capture']['sha256'],
            expected_identity=capture['identity'], census=census, names=sorted(shapes), device='cuda',
            resource_check=check, release_file_pages=True, verified_load_policy=policy))
        acts, hessians, counts, maxima = values
        # Full-census maxima are required for siblings outside this four-unit
        # screen; selected-only maxima would change a fused activation scale.
        scales, scale_policy = tc._static_input_scales(census['max_abs'], profile=profile)
        if scale_policy != groups['input_scale_policy']:
            raise ValueError('frozen full-census input scale policy differs')
        write_json(output/'input-scales.json', dict(policy=scale_policy, units=scales))
        calibration = th.activation_source(hessians, capture['identity']['calibration'])
        memo = tc._activation_kwargs_memo(calibration, weights, 'cuda', max_entries=1,
            resource_check=check, factor_scratch_bytes=resources['selected_anchor_resources'][
                'phases']['resident_anchors']['factorization_scratch_bytes'])
        cache = ProductionWeightCache(weights={}, levers={'tessera_campaign': True},
            activation_max_abs=census['max_abs'], cache_dir=str(cache_dir),
            metadata={'release_completed_anchor_file_pages': True})
        cache.enable_lru(resources['max_resident_render_bytes'])
        for unit_index, name in enumerate(sorted(shapes)):
            selected = sorted((cell for cell in proposal['cells'] if cell['qname'] == name),
                              key=lambda cell: cell['format'])
            inputs = dict(source_weight=_cb_cache_tensor_identity(weights[name]),
                hessian=_cb_cache_tensor_identity(hessians[name]),
                inputs=_cb_cache_tensor_identity(acts[name]), count=counts[name], max_abs=maxima[name],
                original_capture_entry=capture['entries'][name], projected_unit=projected.get(name))
            write_json(output/f'unit-{unit_index}-inputs.json', inputs)
            signatures = [tc._bound_tensor_signature(tensor)
                          for tensor in (weights[name], hessians[name], acts[name])]
            def require_original_inputs():
                if signatures != [tc._bound_tensor_signature(tensor)
                                  for tensor in (weights[name], hessians[name], acts[name])]:
                    raise RuntimeError('encoder changed an original source/H/X tensor')
            anchors = []
            for index, cell in enumerate(selected):
                fmt = cell['format']
                active_candidate = next(row for row in result['candidates']
                                        if (row['qname'], row['format']) == (name, fmt))
                active_candidate['status'] = 'encoding'
                anchor = phase(f'unit-{unit_index}-encode-{index}', lambda: tc._measure_anchor(
                    qname=name, weight=weights[name], activations=acts[name], format_name=fmt,
                    cache=cache, wire_dir=wire_dir, activation_kwargs_for=memo,
                    hessian_required=True, static_input_scale=scales.get(name)))
                if (anchor.memory_bytes != cell['memory_bytes'] or not math.isfinite(anchor.dloss) or
                        anchor.wire_bytes > min(resources['max_original_wire_file_bytes'],
                                               cell['memory_bytes']+64*1024**2)):
                    raise ValueError('actual native cell price, score or wire exceeds its frozen contract')
                require_original_inputs()
                active_candidate.update(status='encoded', anchor=asdict(anchor))
                anchors.append(anchor)
            with tc.bind_checkpoint_unit_identity(anchors, source_weight=weights[name],
                    calibration_source=calibration, projected_unit=projected.get(name),
                    static_scales=scales) as bound:
                for index, anchor in enumerate(anchors):
                    fmt = anchor.format_name
                    active_candidate = next(row for row in result['candidates']
                                            if (row['qname'], row['format']) == (name, fmt))
                    active_candidate['status'] = 'qualifying'
                    identity = tc._checkpoint_anchor_identity(anchor, weights=weights,
                        menus={name: [SimpleNamespace(format_name=fmt)]},
                        calibration_source=calibration, static_scales=scales,
                        projected_units=projected, bound_unit=bound)
                    record = tc._checkpoint_wire_record(anchor, wire_dir, identity)
                    render_path = cache_dir/cache.weights[(name, fmt)]
                    cell = dict(anchor=asdict(anchor), record=record,
                        wire=str(wire_dir/record['file']), render=str(render_path),
                        render_file_sha256=cc.sha256(render_path, release_read_pages=True))
                    def qualify():
                        rendered = cache.get(name, fmt).to('cuda')
                        return verify_anchor_render(cell, weights[name], rendered,
                            calibration_source=calibration, projected_unit=projected.get(name),
                            static_scales=scales, bound_unit=bound, release_file_pages=True)
                    cell['qualification'] = phase(f'unit-{unit_index}-qualify-{index}', qualify)
                    require_original_inputs()
                    if cell['qualification']['source_weight'] != inputs['source_weight']:
                        raise RuntimeError('qualified wire source differs from the original source snapshot')
                    result['cells'].append(cell)
                    active_candidate['status'] = 'qualified'
                    active_candidate = None
            result.setdefault('final_input_identities', {})[name] = phase(
                f'unit-{unit_index}-final-input-identity', lambda: require_original_input_bytes(
                    dict(source_weight=weights[name], hessian=hessians[name], inputs=acts[name]),
                    {key: inputs[key] for key in ('source_weight', 'hessian', 'inputs')},
                    identity=_cb_cache_tensor_identity))
            memo.cache_clear()
        if len(result['cells']) != 28:
            raise RuntimeError('native screen did not qualify all 28 cells')
        result['passed'] = True
    except BaseException as error:
        result['failure'] = dict(type=type(error).__name__, message=str(error))
        if active_candidate is not None:
            active_candidate.update(status='failed', failure=result['failure'])
        raise
    finally:
        coverage = telemetry.finish(result['phases'])
        result.update(finished_unix=time.time(), telemetry_errors=telemetry.errors,
                      telemetry_samples={host: len(rows) for host, rows in telemetry.samples.items()},
                      telemetry_coverage=coverage, guard=guard.snapshot())
        result['wire_cells_passed'] = result['passed']
        result['passed'] = result['passed'] and coverage['passed']
        write_json(output/'result.json', result)
    print(json.dumps(dict(passed=result['passed'], cells=len(result['cells']),
                         result_sha256=cc.sha256(output/'result.json'))))
    if not result['passed']:
        raise RuntimeError('native wire cells or required both-host telemetry are incomplete')


if __name__ == '__main__':
    main()
