"""Actual ordinary AURA measurement and CLI source delivery on small CPU inputs.

CUDA admission, tokenizer and calibration sampling are explicit doubles.
The CLI journal test doubles only the numerical call and uses the real journal.
"""
import math

import pytest
import torch

from prismaquant import aura_checkpoint_journal as journal
from prismaquant import aura_cost as aura
from prismaquant import memory_management
from prismaquant.cost_streaming import build_streamed_causal_lm
from prismaquant.model_profiles import detect_profile
from prismaquant.prepriced_cost import validate_native_cost_source_for_model
from prismaquant.streaming_model import load_resident_source_model, _model_tensor_dtypes
from prismaquant.streamed_model_identity import build_source_model_binding
from tests.native_source_contract_fixtures import (
    producer_cli_inputs, producer_cpu_guard, producer_source,
)


@pytest.mark.parametrize('streaming,dtype', [
    (False, torch.bfloat16), (True, torch.bfloat16), (False, torch.float32),
    pytest.param(True, torch.float32, marks=pytest.mark.xfail(
        strict=True, raises=RuntimeError, reason='base native streamed FP32 delivery keeps decoded FP8 weights in BF16')),
])
def test_ordinary_aura_final_cells_retain_qualified_source(tmp_path, monkeypatch, streaming, dtype):
    source, _expected = producer_source(tmp_path)
    tokens = producer_cli_inputs(monkeypatch)
    binding = build_source_model_binding(source, dtype=dtype)
    profile = detect_profile(str(source))
    kwargs = dict(n_probes=1, min_free_gib=0, unit_filter=r'layers\.0\.self_attn\.o_proj$',
                  allow_packed_expert_omission=True, profile=profile,
                  checkpoint_identity_extra={'source_binding': binding})
    if streaming:
        runner = build_streamed_causal_lm(
            str(source), device=torch.device('cpu'), dtype=dtype, profile=profile,
            offload_folder=str(tmp_path / 'streamed'), max_cache_slots=2,
            prefetch_workers=1, prefetch_min_available_gb=0, cache_headroom_gb=0,
            attn_implementation='eager')
        try:
            payload = aura.compute_aura_cost_streamed(runner, tokens, ['FP8_E4M3', 'BF16'], **kwargs)
        finally:
            runner.shutdown()
    else:
        guard = producer_cpu_guard(tmp_path, monkeypatch)
        model = load_resident_source_model(
            str(source), device=torch.device('cpu'), dtype=dtype, requires_grad=True,
            offload_folder=str(tmp_path / 'resident'), resource_check=guard.check,
            attn_implementation='eager')
        payload = aura.compute_aura_cost(model, tokens, ['FP8_E4M3', 'BF16'], n_linear_chunks=1, **kwargs)
    assert payload['costs']
    for rows in payload['costs'].values():
        assert math.isfinite(rows['FP8_E4M3']['predicted_dloss'])
        for row in rows.values():
            assert row.get('source_binding') == binding
    validate_native_cost_source_for_model(payload, model=str(source), where='ordinary AURA final cells')


@pytest.mark.parametrize('streaming,native,dtype', [
    (False, True, 'bfloat16'), (True, True, 'bfloat16'), (False, True, 'float32'),
    (False, False, 'bfloat16'), (True, False, 'bfloat16'),
    (False, False, 'float32'), (True, False, 'float32'),
])
def test_ordinary_aura_main_delivers_declared_values_before_measurement(
        tmp_path, monkeypatch, streaming, native, dtype):
    source, expected = producer_source(tmp_path, native=native)
    producer_cli_inputs(monkeypatch)
    guard = producer_cpu_guard(tmp_path, monkeypatch)
    monkeypatch.setattr(memory_management, 'CaptureMemoryGuard', lambda *_args: guard)
    requested = getattr(torch, dtype)
    observed = []

    def measurement(delivery, *_args, **kwargs):
        model = delivery.model if streaming else delivery
        if streaming:
            for layer in range(delivery.num_layers):
                delivery.context.schedule_prefetch(layer)
                delivery.context.install(layer, require_prefetched=True, prefetch_following=False)
        plan = _model_tensor_dtypes(model, requested)
        for name, parameter in model.named_parameters():
            wanted = expected[name].to(dtype=plan.get(name, requested))
            assert not parameter.is_meta, name
            assert torch.equal(parameter, wanted), name
            assert parameter.dtype == wanted.dtype, name
        extra = kwargs['checkpoint_identity_extra']
        if native:
            assert extra.get('source_binding') == build_source_model_binding(source, dtype=requested)
        else:
            assert 'source_binding' not in extra
        observed.append(True)
        return {'costs': {}, 'provenance': {'dw_rendered_rows': 0, 'dw_rtn_fallback_rows': 0, 'seed_base': 7000}}

    monkeypatch.setattr(aura, 'compute_aura_cost_streamed' if streaming else 'compute_aura_cost', measurement)
    args = ['--model', str(source), '--output', str(tmp_path / 'cli.pkl'), '--device', 'cpu',
            '--dtype', dtype, '--dataset', 'synthetic-token-draw', '--formats', 'FP8_E4M3']
    if streaming:
        args.append('--streaming')
    assert aura.main(args) == 0
    assert observed == [True]


@pytest.mark.parametrize('mode', ['0', '1'], ids=['certified', 'dev'])
def test_ordinary_aura_main_supplies_binding_before_journal_restore(tmp_path, monkeypatch, mode):
    source, _expected = producer_source(tmp_path)
    producer_cli_inputs(monkeypatch)
    monkeypatch.setenv('PRISMAQUANT_DEV_MODE', mode)
    checkpoint = tmp_path / 'journal'
    old = journal.AuraCheckpointJournal.open(checkpoint, resume=False, identity={'extra': {'streaming': True}}, names=['unit'])
    old.publish_state('unit', {'scalar': [1.25]})
    before = {str(p.relative_to(checkpoint)): p.read_bytes() for p in checkpoint.rglob('*') if p.is_file() and p.name != 'streamed_model_identity.json'}
    reads = []

    def no_unit_read(*_args, **_kwargs):
        reads.append(True)
        pytest.fail('native journal admission must precede unit reads')

    monkeypatch.setattr(journal, 'load_unit', no_unit_read)

    def measurement(_runner, *_args, **kwargs):
        journal.AuraCheckpointJournal.open(
            kwargs['checkpoint_dir'], resume=True,
            identity={'extra': kwargs['checkpoint_identity_extra']}, names=['unit'])
        pytest.fail('unbound native journal was admitted')

    monkeypatch.setattr(aura, 'compute_aura_cost_streamed', measurement)
    with pytest.raises(RuntimeError, match='native source binding.*fresh checkpoint'):
        aura.main(['--model', str(source), '--output', str(tmp_path / 'cli.pkl'), '--device', 'cpu',
                   '--dtype', 'bfloat16', '--dataset', 'synthetic-token-draw', '--streaming',
                   '--checkpoint-dir', str(checkpoint), '--resume'])
    assert reads == []
    assert {str(p.relative_to(checkpoint)): p.read_bytes() for p in checkpoint.rglob('*') if p.is_file() and p.name != 'streamed_model_identity.json'} == before


def test_hub_native_aura_uses_resolved_original_source(tmp_path, monkeypatch):
    import huggingface_hub
    from transformers import AutoModelForCausalLM
    from prismaquant import model_profiles

    source, expected = producer_source(tmp_path)
    producer_cli_inputs(monkeypatch)
    guard = producer_cpu_guard(tmp_path, monkeypatch)
    monkeypatch.setattr(memory_management, 'CaptureMemoryGuard', lambda *_args: guard)
    hub_id = 'contract-fixture/native-model'
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', lambda **_kwargs: str(source))
    actual_profile = model_profiles.detect_profile
    monkeypatch.setattr(model_profiles, 'detect_profile',
                        lambda model, **kwargs: actual_profile(str(source) if model == hub_id else model, **kwargs))

    def no_hf(*_args, **_kwargs):
        pytest.fail('native Hub source must use the qualified resident decoder')

    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', no_hf)
    observed = []

    def measurement(model, *_args, **kwargs):
        for name, parameter in model.named_parameters():
            assert parameter.dtype == expected[name].dtype
            assert torch.equal(parameter, expected[name]), name
        assert kwargs['checkpoint_identity_extra']['source_binding'] == build_source_model_binding(source, dtype=torch.bfloat16)
        observed.append(True)
        return {'costs': {}, 'provenance': {'dw_rendered_rows': 0, 'dw_rtn_fallback_rows': 0, 'seed_base': 7000}}

    monkeypatch.setattr(aura, 'compute_aura_cost', measurement)
    assert aura.main(['--model', hub_id, '--output', str(tmp_path / 'hub.pkl'),
        '--device', 'cpu', '--dtype', 'bfloat16', '--dataset', 'synthetic-token-draw']) == 0
    assert observed == [True]


@pytest.mark.parametrize('dtype', ['float32', 'auto'])
def test_native_streamed_fp32_refuses_before_load(tmp_path, monkeypatch, capsys, dtype):
    from prismaquant import cost_streaming

    source, _expected = producer_source(tmp_path)
    producer_cli_inputs(monkeypatch)
    if dtype == 'auto':
        monkeypatch.setattr(aura, '_resolve_auto_dtype', lambda *_args: 'float32')

    def no_load(*_args, **_kwargs):
        pytest.fail('unsupported native streamed FP32 must refuse before loading')

    monkeypatch.setattr(cost_streaming, 'build_streamed_causal_lm', no_load)
    with pytest.raises(SystemExit) as refused:
        aura.main(['--model', str(source), '--output', str(tmp_path / 'fp32.pkl'),
            '--device', 'cpu', '--dtype', dtype, '--streaming', '--dataset', 'synthetic-token-draw'])
    assert refused.value.code == 2
    assert 'native FP8 streaming requires --dtype bfloat16' in capsys.readouterr().err
    assert not (tmp_path / 'fp32.pkl').exists()


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='native streamed FP32 delivery keeps decoded FP8 weights in BF16')
def test_direct_native_streamed_fp32_delivery_limit(tmp_path):
    source, expected = producer_source(tmp_path)
    runner = build_streamed_causal_lm(
        str(source), device=torch.device('cpu'), dtype=torch.float32,
        profile=detect_profile(str(source)), offload_folder=str(tmp_path / 'direct-fp32'),
        max_cache_slots=2, prefetch_workers=1, prefetch_min_available_gb=0,
        cache_headroom_gb=0, attn_implementation='eager')
    try:
        for layer in range(runner.num_layers):
            runner.context.schedule_prefetch(layer)
            runner.context.install(layer, require_prefetched=True, prefetch_following=False)
        plan = _model_tensor_dtypes(runner.model, torch.float32)
        for name, parameter in runner.model.named_parameters():
            wanted = expected[name].to(dtype=plan.get(name, torch.float32))
            assert torch.equal(parameter.to(torch.float32), wanted.to(torch.float32)), name
            assert parameter.dtype == wanted.dtype, name
    finally:
        runner.shutdown()


def test_ordinary_aura_main_resumes_a_bound_journal(tmp_path, monkeypatch):
    source, _expected = producer_source(tmp_path)
    producer_cli_inputs(monkeypatch)
    checkpoint = tmp_path / 'bound-journal'
    binding = build_source_model_binding(source, dtype=torch.bfloat16)
    seen = []

    class Prices:
        def restore_unit(self, name, state):
            assert name == 'unit' and state == {'scalar': [1.25]}
            seen.append('restored')

    def measurement(_runner, *_args, **kwargs):
        extra = kwargs['checkpoint_identity_extra']
        assert extra['source_binding'] == binding
        current = journal.AuraCheckpointJournal.open(
            kwargs['checkpoint_dir'], resume=kwargs['resume'],
            identity={'extra': extra}, names=['unit'])
        if kwargs['resume']:
            assert current.resume_units(Prices()) == {'unit'}
        else:
            current.publish_state('unit', {'scalar': [1.25]})
        return {'costs': {}, 'provenance': {'dw_rendered_rows': 0, 'dw_rtn_fallback_rows': 0, 'seed_base': 7000}}

    monkeypatch.setattr(aura, 'compute_aura_cost_streamed', measurement)
    args = ['--model', str(source), '--output', str(tmp_path / 'bound.pkl'), '--device', 'cpu',
            '--dtype', 'bfloat16', '--dataset', 'synthetic-token-draw', '--streaming',
            '--checkpoint-dir', str(checkpoint)]
    assert aura.main(args) == 0
    assert aura.main(args + ['--resume']) == 0
    assert seen == ['restored']
