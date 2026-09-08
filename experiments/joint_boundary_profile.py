"""Bounded genuine GLM before/after qualification, never a full-model fit gate.

Run only as an admitted PB measurement in the recorded producer container.
ABBA preserves the source/checkpoint/calibration/probes and scalar arithmetic.
The upstream Torch reference KDA/conv kernels deliberately avoid testing the
separate optional fused-kernel contract on this tiny fixture.
"""
from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import json
from pathlib import Path
import pstats
import sys
import time
import weakref

import torch

from experiments.glm_full_capture_profile import CaptureObserver
from prismaquant import aura_cost
from prismaquant.cost_streaming import (
    BOUNDARY_STORAGE_SCHEMA, build_streamed_causal_lm,
    build_streamed_model_identity,
)
from prismaquant.model_profiles.glm5_next import Glm5NextProfile
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.routed_experts import profile_declared_packed_expert_projections


def tensor_digest(value):
    value = value.detach().to('cpu').contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def process_state():
    return dict(time=time.time(), process_io=Path('/proc/self/io').read_text(),
        process_status=Path('/proc/self/status').read_text(),
        cuda_allocated=torch.cuda.memory_allocated(),
        cuda_reserved=torch.cuda.memory_reserved(),
        cuda_peak_allocated=torch.cuda.max_memory_allocated(),
        cuda_peak_reserved=torch.cuda.max_memory_reserved())


def run_arm(source, cache, ids, out, index, mode):
    profile = Glm5NextProfile()
    runner = build_streamed_causal_lm(str(source), device=torch.device('cuda'),
        dtype=torch.bfloat16, offload_folder=str(out/f'offload-{index}'),
        profile=profile, max_cache_slots=2, prefetch_workers=1,
        prefetch_min_available_gb=0, cache_headroom_gb=0,
        prefetch_lookahead=1, require_prefetched_residency=True,
        attn_implementation='eager')
    identity = build_streamed_model_identity(runner, str(source))
    calls, cotangents, owned = [], [], []
    capture_peaks = []
    original_call, original_capture = runner._call, runner.capture_boundaries
    original_tail, original_reverse = runner.tail_logits, runner.isolated_layer
    tail_calls = reverse_calls = 0

    def call(layer, hidden, *, batch, pass_state):
        calls.append((int(batch.input_ids[0, 0]), layer, torch.is_grad_enabled()))
        return original_call(layer, hidden, batch=batch, pass_state=pass_state)

    def capture(*args, **kwargs):
        batch = original_capture(*args, **kwargs)
        owned.extend(weakref.ref(t) for t in batch.activations_cpu if isinstance(t, torch.Tensor))
        capture_peaks.append(sum(t().untyped_storage().nbytes() for t in owned if t() is not None))
        return batch

    def observe(gradient, probe, batch, layer):
        cotangents.append(dict(probe=probe, batch=batch, layer=layer,
            shape=list(gradient.shape), dtype=str(gradient.dtype),
            sha256=tensor_digest(gradient)))

    def tail(batch, hidden):
        nonlocal tail_calls
        b, k = divmod(tail_calls, 4)
        hidden.register_hook(lambda g, b=b, k=k: observe(g, k, b, runner.num_layers))
        tail_calls += 1
        return original_tail(batch, hidden)

    def reverse(batch, layer, hidden, *, pass_state):
        nonlocal reverse_calls
        k, b = divmod(reverse_calls % (4*len(ids)), len(ids))
        hidden.register_hook(lambda g, k=k, b=b, layer=layer: observe(g, k, b, layer))
        reverse_calls += 1
        return original_reverse(batch, layer, hidden, pass_state=pass_state)

    runner._call, runner.capture_boundaries = call, capture
    runner.tail_logits, runner.isolated_layer = tail, reverse
    plane = ids.shape[1] * 4 * 64 * 2
    policy = dict(schema=BOUNDARY_STORAGE_SCHEMA, directory=str(out/f'artifacts-{index}'),
        max_resident_bytes=5*plane, max_auxiliary_bytes=4*1024**2,
        max_artifact_bytes=32*1024**2, prefetch_batches=2)
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    record = dict(index=index, mode=mode, before=process_state(),
        identity=identity, boundary_plane_bytes=plane)
    cpu_profiler = cProfile.Profile()
    started = time.perf_counter()
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=True,
                profile_memory=True) as profiler, cpu_profiler:
            payload = aura_cost.compute_aura_cost_streamed(runner, ids,
                ['FP8_DYNAMIC', 'NVFP4A16', 'BF16'], n_probes=4,
                probe_microbatch=1, seed_base=7000, min_free_gib=0,
                production_cache=cache, joint_activation=True,
                collect_col_energy=True, include_routed_experts=True,
                formats_by_qname={name: ['FP8_DYNAMIC', 'NVFP4A16', 'BF16']
                                  for name, _fmt in cache.weights},
                model_identity=identity, profile=profile,
                boundary_storage=policy if mode == 'exact' else None)
            torch.cuda.synchronize()
        record.update(seconds=time.perf_counter()-started, after=process_state(),
            source_calls=calls, cotangents=cotangents,
            legacy_boundary_capture_peak_bytes=max(capture_peaks),
            storage=payload['provenance'].get('streamed_boundary_storage'),
            targets=len(payload['costs']))
        profiler.export_chrome_trace(str(out/f'arm-{index}-{mode}.trace.json'))
        (out/f'arm-{index}-{mode}.profile.txt').write_text(
            profiler.key_averages().table(sort_by='self_cuda_time_total', row_limit=80))
        cpu_profiler.dump_stats(str(out/f'arm-{index}-{mode}.cprofile'))
        with (out/f'arm-{index}-{mode}.cprofile.txt').open('w') as handle:
            pstats.Stats(cpu_profiler, stream=handle).sort_stats('cumulative').print_stats(80)
        assert len(cotangents) == 4*len(ids)*(runner.num_layers+1)
        assert all(row['probe_ids'] == [7000, 7001, 7002, 7003]
                   for rows in payload['costs'].values() for row in rows.values())
        assert any('.experts.' in name for name in payload['costs'])
        if mode == 'exact':
            receipt = record['storage']
            assert receipt['status'] == 'complete'
            assert receipt['telemetry']['peak_resident_tensor_bytes'] <= 5*plane
            assert receipt['telemetry']['live_artifact_bytes'] == 0
            assert receipt['telemetry']['hot_read_misses'] == 0
            assert not list((out/f'artifacts-{index}').rglob('*.pt'))
        (out/f'arm-{index}-{mode}.json').write_text(json.dumps(record, indent=2)+'\n')
        return payload, record
    finally:
        runner.shutdown()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-out', type=Path, required=True)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('native GLM qualification requires CUDA')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
    from test_glm5_next_streamed_forward_parity import _build_tiny_model
    from test_glm_campaign_streaming import write_original_layout_checkpoint
    from transformers.models.glm5_next import modeling_glm5_next as upstream
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    kernels = {}
    for name in ('causal_conv1d_fn', 'causal_conv1d_update',
                 'chunk_kimi_delta_attention', 'recurrent_kimi_delta_attention'):
        reference = getattr(getattr(upstream, name), '__wrapped__', None)
        if reference is None:
            raise RuntimeError(f'missing upstream Torch reference {name}')
        setattr(upstream, name, reference)
        kernels[name] = reference.__module__+'.'+reference.__name__
    with CaptureObserver(args.evidence_out, profile_layers=()) as observer:
        out = args.evidence_out
        model = _build_tiny_model().to(torch.bfloat16)
        source = out/'source'
        write_original_layout_checkpoint(model, source)
        profile = Glm5NextProfile()
        weights = {name: module.weight for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and '.layers.' in name
            and not profile.is_pinned_name(name)}
        weights.update({m.qname: m.weight for m in profile_declared_packed_expert_projections(model, profile)})
        cache = ProductionWeightCache(weights={(name, fmt): weight.detach().clone()+0.03125
            for name, weight in weights.items() for fmt in ('FP8_E4M3', 'NVFP4A16')},
            levers={}, activation_max_abs={name: 1.0 for name in weights})
        del weights, model
        ids = torch.arange(5*17).remainder(126).add(2).reshape(5, 17)
        observer.result.update(schema='prismaquant.joint_boundary_qualification.v1',
            fixture=dict(model='genuine random tiny glm5_next original-layout checkpoint',
                dtype='bfloat16', hidden=64, hc_mult=4, layers=2, routed_experts=4,
                batches=5, sequence=17, probes=[7000,7001,7002,7003],
                token_sha256=tensor_digest(ids), kernels=kernels,
                candidate_weights='synthetic BF16 source +0.03125, not a serving artifact'),
            claims=dict(full_model_fit=False, full_model_throughput=False,
                kernel_qualification=False, scalar_arithmetic_changed=False), arms=[])
        baseline = baseline_record = None
        for index, mode in enumerate(('legacy', 'exact', 'exact', 'legacy')):
            print(f'BEGIN arm={index} mode={mode}', flush=True)
            payload, record = run_arm(source, cache, ids, out, index, mode)
            if baseline is None:
                baseline, baseline_record = payload, record
            else:
                assert payload['costs'] == baseline['costs'], 'cost arithmetic changed'
                assert record['source_calls'] == baseline_record['source_calls'], 'source order changed'
                assert record['cotangents'] == baseline_record['cotangents'], 'cotangent bytes changed'
                for name, row in payload['stats'].items():
                    assert row['h_trace'] == baseline['stats'][name]['h_trace']
                    assert torch.equal(row['fisher_col'], baseline['stats'][name]['fisher_col'])
            record['exact_parity'] = True
            observer.result['arms'].append(record)
            (out/'progress.json').write_text(json.dumps(observer.result, indent=2)+'\n')
            print(f'PASS arm={index} mode={mode} seconds={record["seconds"]:.3f}', flush=True)
            del payload
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
