"""Paired exact capture traversal qualification on a tiny original-layout GLM.

Only the changed capture phase uses Torch profiling; cProfile and the existing
both-host observer cover the whole call. Three source layers exceed the two
source-cache slots, so actual reader calls expose repeated source loading.
"""
from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import pstats
import sys
import time

import torch

from experiments.glm_full_capture_profile import CaptureObserver
from experiments.joint_boundary_profile import process_state, tensor_digest
from prismaquant import aura_cost, streaming_model
from prismaquant.cost_streaming import (
    BOUNDARY_STORAGE_SCHEMA, LAYER_MAJOR_BOUNDARY_STORAGE_SCHEMA,
    build_streamed_causal_lm, build_streamed_model_identity,
)
from prismaquant.model_profiles.glm5_next import Glm5NextProfile
from prismaquant.production_weight_cache import ProductionWeightCache
from prismaquant.routed_experts import profile_declared_packed_expert_projections


def run_arm(source, cache, ids, out, index, order):
    runner = build_streamed_causal_lm(str(source), device=torch.device('cuda'),
        dtype=torch.bfloat16, offload_folder=str(out/f'offload-{index}'),
        profile=Glm5NextProfile(), max_cache_slots=2, prefetch_workers=1,
        prefetch_min_available_gb=0, cache_headroom_gb=0,
        prefetch_lookahead=1, require_prefetched_residency=True,
        attn_implementation='eager')
    identity = build_streamed_model_identity(runner, str(source))
    record = dict(index=index, capture_order=order, identity=identity,
                  source_calls=[], source_reads=[], installs=[], cotangents=[])
    phase = 'setup'
    original_read = streaming_model._read_layer_to_device
    original_install, original_call = runner.context.install, runner._call
    original_capture, original_major = runner.capture_boundaries, runner.capture_layer_major_boundaries
    original_tail, original_reverse = runner.tail_logits, runner.isolated_layer
    tail_calls = reverse_calls = captured_batches = 0
    profile_active = False
    profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA], record_shapes=True, profile_memory=True)

    def read(prefix, *args, **kwargs):
        started = time.perf_counter()
        value = original_read(prefix, *args, **kwargs)
        if prefix.startswith(runner.context.layers_prefix):
            storages = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
                        for t in value.values()}
            record['source_reads'].append(dict(prefix=prefix, phase=phase,
                seconds=time.perf_counter()-started, materialized_storage_bytes=sum(storages.values())))
        return value

    def install(layer, **kwargs):
        value = original_install(layer, **kwargs)
        record['installs'].append(dict(layer=layer, phase=phase, delivery=value))
        return value

    def call(layer, hidden, *, batch, pass_state):
        incoming = tensor_digest(hidden)
        value = original_call(layer, hidden, batch=batch, pass_state=pass_state)
        record['source_calls'].append(dict(layer=layer, batch=int(batch.input_ids[0, 0]),
            grad=torch.is_grad_enabled(), input=incoming, output=tensor_digest(value),
            shape=list(hidden.shape), position_ids=tensor_digest(batch.position_ids)))
        return value

    def begin_capture():
        nonlocal phase, profile_active
        phase = 'capture'
        runner.context.begin_source_initialization_audit()
        record['capture_before'] = process_state()
        profiler.__enter__()
        profile_active = True

    def end_capture():
        nonlocal phase, profile_active
        if profile_active:
            torch.cuda.synchronize()
            record['capture_after'] = process_state()
            record['initialization'] = runner.context.source_initialization_contract()
            profiler.__exit__(None, None, None)
            profile_active = False
        phase = 'reverse'

    def capture(*args, **kwargs):
        nonlocal captured_batches
        if captured_batches == 0:
            begin_capture()
        value = original_capture(*args, **kwargs)
        captured_batches += 1
        if captured_batches == len(ids):
            end_capture()
        return value

    def major(*args, **kwargs):
        begin_capture()
        value = original_major(*args, **kwargs)
        end_capture()
        return value

    def gradient(value, probe, batch, layer):
        record['cotangents'].append(dict(probe=probe, batch=batch, layer=layer,
            dtype=str(value.dtype), shape=list(value.shape), sha256=tensor_digest(value)))

    def tail(batch, hidden):
        nonlocal tail_calls
        b, k = divmod(tail_calls, 4)
        hidden.register_hook(lambda g, b=b, k=k: gradient(g, k, b, runner.num_layers))
        tail_calls += 1
        return original_tail(batch, hidden)

    def reverse(batch, layer, hidden, *, pass_state):
        nonlocal reverse_calls
        k, b = divmod(reverse_calls % (4*len(ids)), len(ids))
        hidden.register_hook(lambda g, k=k, b=b, layer=layer: gradient(g, k, b, layer))
        reverse_calls += 1
        return original_reverse(batch, layer, hidden, pass_state=pass_state)

    streaming_model._read_layer_to_device = read
    runner.context.install, runner._call = install, call
    runner.capture_boundaries, runner.capture_layer_major_boundaries = capture, major
    runner.tail_logits, runner.isolated_layer = tail, reverse
    plane = ids.shape[1]*4*64*2
    policy = dict(schema=BOUNDARY_STORAGE_SCHEMA, directory=str(out/f'artifacts-{index}'),
        max_resident_bytes=5*plane, max_auxiliary_bytes=4*1024**2,
        max_artifact_bytes=32*1024**2, prefetch_batches=2)
    if order == 'layer_major':
        policy.update(schema=LAYER_MAJOR_BOUNDARY_STORAGE_SCHEMA, capture_order=order)
    cpu = cProfile.Profile()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with cpu:
            payload = aura_cost.compute_aura_cost_streamed(runner, ids,
                ['FP8_DYNAMIC', 'NVFP4A16', 'BF16'], n_probes=4,
                probe_microbatch=1, seed_base=7000, min_free_gib=0,
                production_cache=cache, joint_activation=True, collect_col_energy=True,
                include_routed_experts=True, model_identity=identity,
                formats_by_qname={name:['FP8_DYNAMIC','NVFP4A16','BF16'] for name, _ in cache.weights},
                boundary_storage=policy)
        record.update(seconds=time.perf_counter()-started, after=process_state(),
                      storage=payload['provenance']['streamed_boundary_storage'], targets=len(payload['costs']))
        assert record['storage']['status'] == 'complete'
        assert record['storage']['telemetry']['live_artifact_bytes'] == 0
        assert record['storage']['telemetry']['hot_read_misses'] == 0
        captured = [x for x in record['installs'] if x['phase'] == 'capture']
        assert len(captured) == (runner.num_layers if order == 'layer_major' else runner.num_layers*len(ids))
        if order == 'layer_major':
            assert len([x for x in record['source_reads'] if x['phase'] == 'capture']) == runner.num_layers
        assert len(record['cotangents']) == 4*len(ids)*(runner.num_layers+1)
        profiler.export_chrome_trace(str(out/f'arm-{index}.trace.json'))
        (out/f'arm-{index}.profile.txt').write_text(profiler.key_averages().table(sort_by='self_cuda_time_total', row_limit=80))
        cpu.dump_stats(str(out/f'arm-{index}.cprofile'))
        with (out/f'arm-{index}.cprofile.txt').open('w') as handle:
            pstats.Stats(cpu, stream=handle).sort_stats('cumulative').print_stats(80)
        torch.save(payload, out/f'arm-{index}.cost.pt')
        (out/f'arm-{index}.json').write_text(json.dumps(record, indent=2)+'\n')
        return payload, record
    finally:
        if profile_active:
            profiler.__exit__(None, None, None)
        runner.shutdown()
        streaming_model._read_layer_to_device = original_read


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-out', type=Path, required=True)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError('native layer-major qualification requires CUDA')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tests'))
    from test_glm5_next_streamed_forward_parity import _build_model, _tiny_config
    from test_glm_campaign_streaming import write_original_layout_checkpoint
    from transformers.models.glm5_next import modeling_glm5_next as upstream
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    kernels = {}
    for name in ('causal_conv1d_fn','causal_conv1d_update','chunk_kimi_delta_attention','recurrent_kimi_delta_attention'):
        reference = getattr(getattr(upstream, name), '__wrapped__', None)
        if reference is None:
            raise RuntimeError(f'missing upstream Torch reference {name}')
        setattr(upstream, name, reference)
        kernels[name] = reference.__module__+'.'+reference.__name__
    with CaptureObserver(args.evidence_out, profile_layers=()) as observer:
        config = _tiny_config()
        config.text_config.num_hidden_layers = 3
        config.text_config.layer_types = ['linear_attention','deepseek_sparse_attention','deepseek_sparse_attention']
        config.text_config.mlp_layer_types = ['dense','sparse','sparse']
        config.text_config.indexer_types = ['full']*3
        config = type(config).from_dict(config.to_dict())
        torch.manual_seed(20260826)
        model = _build_model(config).to(torch.bfloat16)
        source = args.evidence_out/'source'
        write_original_layout_checkpoint(model, source)
        profile = Glm5NextProfile()
        weights = {name:module.weight for name,module in model.named_modules()
                   if isinstance(module,torch.nn.Linear) and '.layers.' in name and not profile.is_pinned_name(name)}
        weights.update({m.qname:m.weight for m in profile_declared_packed_expert_projections(model,profile)})
        cache = ProductionWeightCache(weights={(name,fmt):tensor.detach().clone()+0.03125
            for name,tensor in weights.items() for fmt in ('FP8_E4M3','NVFP4A16')},
            levers={},activation_max_abs={name:1. for name in weights})
        del model, weights
        ids = torch.arange(5*17).remainder(126).add(2).reshape(5,17)
        observer.result.update(schema='prismaquant.layer_major_boundary_qualification.v1',
            fixture=dict(layers=3,source_cache_slots=2,hidden=64,hc_mult=4,experts=4,
                dtype='bfloat16',batches=5,sequence=17,probe_ids=[7000,7001,7002,7003],
                kernels=kernels,token_sha256=tensor_digest(ids),candidates='synthetic source+0.03125'),
            claims=dict(full_model_fit=False,full_model_throughput=False,fused_kernel_qualification=False),arms=[])
        baseline = baseline_record = None
        for index, order in enumerate(('batch_major','layer_major','layer_major','batch_major')):
            print(f'BEGIN arm={index} capture={order}',flush=True)
            payload, record = run_arm(source,cache,ids,args.evidence_out,index,order)
            if baseline is None:
                baseline, baseline_record = payload, record
            else:
                assert payload['costs'] == baseline['costs']
                assert record['cotangents'] == baseline_record['cotangents']
                for b in ids[:,0].tolist():
                    assert [x for x in record['source_calls'] if x['batch']==b] == [x for x in baseline_record['source_calls'] if x['batch']==b]
                for name, stats in payload['stats'].items():
                    assert stats['h_trace'] == baseline['stats'][name]['h_trace']
                    assert torch.equal(stats['fisher_col'],baseline['stats'][name]['fisher_col'])
            record['exact_parity'] = True
            observer.result['arms'].append(record)
            (args.evidence_out/'progress.json').write_text(json.dumps(observer.result,indent=2)+'\n')
            print(f'PASS arm={index} capture={order}',flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
