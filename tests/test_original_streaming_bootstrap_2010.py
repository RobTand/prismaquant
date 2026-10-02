"""Internal CPU original bootstrap and existing loader integration, no automatic gate."""
from __future__ import annotations

import gc
import hashlib
import json
import threading
import traceback
from concurrent.futures import CancelledError
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from prismaquant import layer_streaming, streaming_model, tessera_calibration_cache as cc
from prismaquant.residency_map import bind_residency_manifest
from test_capture_original_material import _bound, _owner, _sha, material  # noqa: F401
from test_stage_b_prep_staged_reads_1092 import _stage_manifest
from test_strict_reader_tier_enforcement import MANIFEST, _forget_state  # noqa: F401

pytestmark = pytest.mark.own_process


def _rebind(m, monkeypatch):
    m['raws'] = {name: Path(path).read_bytes() for name, path in m['paths'].items()}
    for row in m['publisher']['siblings']:
        raw = m['raws'][row['rfilename']]
        row['size'] = len(raw)
        row['blobId'] = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if row.get('lfs'):
            row['lfs'].update(size=len(raw), sha256=_sha(raw))
    for row in m['readset']['entries']:
        raw = Path(row['path']).read_bytes()
        row.update(bytes=len(raw), sha256=_sha(raw))
    m['readset']['total_bytes'] = sum(row['bytes'] for row in m['readset']['entries'])
    m['producer']['files'] = {name: _sha(raw) for name, raw in m['raws'].items()
                              if name.endswith('.safetensors')}
    m['producer']['auxiliary_sha256'] = {name: _sha(raw) for name, raw in m['raws'].items()
                                       if not name.endswith('.safetensors') and name != 'config.json'}
    m['producer']['config_sha256'] = _sha(m['raws']['config.json'])
    m['producer']['tensors'] = json.loads(m['raws']['model.safetensors.index.json'])['weight_map']
    m['options']['publisher_input'] = _bound(m['tmp'] / 'model-publisher.json', m['publisher'])
    m['options']['readset_input'] = _bound(m['tmp'] / 'model-readset.json', m['readset'])
    m['options']['max_material_bytes'] = sum(len(raw) for raw in m['raws'].values())
    m['rebinds'] = m.get('rebinds', 0) + 1
    stage = m['tmp'] / f"model-stage-{m['rebinds']}"
    stage.mkdir()
    m['staged'] = _stage_manifest(stage, monkeypatch, m['readset'])
    bind_residency_manifest(MANIFEST)


@pytest.fixture
def original_model(material, monkeypatch):  # noqa: F811
    from transformers import LlamaConfig, LlamaForCausalLM
    m = material
    torch.manual_seed(19)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                         tie_word_embeddings=False)
    model = LlamaForCausalLM(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.1)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    head = {name: value for name, value in expected.items() if not name.startswith('model.layers.')}
    body = {name: value for name, value in expected.items() if name.startswith('model.layers.')}
    save_file(head, m['paths']['one.safetensors'])
    save_file(body, m['paths']['two.safetensors'])
    Path(m['paths']['config.json']).write_text(config.to_json_string())
    mapping = {name: 'one.safetensors' if name in head else 'two.safetensors' for name in expected}
    Path(m['paths']['model.safetensors.index.json']).write_text(json.dumps({'weight_map': mapping}))
    _rebind(m, monkeypatch)
    m['expected'] = expected
    # These mutable logical paths must never choose metadata or tensor bytes.
    (m['root'] / 'config.json').write_text('{"model_type":"untrusted_pool"}')
    (m['root'] / 'model.safetensors.index.json').write_text('{"weight_map":{}}')
    return m


def _context(m, owner, path):
    return streaming_model._build_streaming_context(str(path), device=torch.device('cpu'),
        dtype=torch.float32, offload_folder=str(m['tmp'] / ('owned-offload' if owner else 'legacy-offload')),
        source_authentication=owner, cache_headroom_gb=0, max_cache_slots=2,
        prefetch_workers=1, prefetch_min_available_gb=0, attn_implementation='eager')


def test_owned_bootstrap_and_full_cpu_layers_match_legacy_original_bytes(original_model):
    m = original_model
    legacy = _context(m, None, Path(m['paths']['config.json']).parent)
    owner = _owner(m)
    owned = _context(m, owner, m['root'])
    try:
        assert owned.model.config.model_type == legacy.model.config.model_type == 'llama'
        assert owned.model.config.hidden_size == legacy.model.config.hidden_size == 16
        for context in (legacy, owned):
            for layer in range(context.num_layers):
                context.install(layer)
        del context
        for name, value in owned.model.state_dict().items():
            assert torch.equal(value, m['expected'][name]), name
            assert torch.equal(value, legacy.model.state_dict()[name]), name
        del value
        ids = torch.tensor([[1, 2, 3, 4]])
        assert torch.equal(owned.model(ids).logits, legacy.model(ids).logits)
        assert 0 < owner.material_live_bytes <= m['options']['max_material_bytes']
        with pytest.raises(RuntimeError, match='live.*consumer'):
            owner.close()
        with pytest.raises(RuntimeError, match='qualified immutable source'):
            cc.require_automatic_capture_source_recording()
    finally:
        legacy.shutdown()
        owned.shutdown()
        del legacy, owned
        gc.collect()
        owner.close()


def test_pool_and_stage_config_mutation_after_seal_does_not_change_bootstrap(original_model, monkeypatch):
    m = original_model
    owner = _owner(m)
    real = cc._original_bootstrap_json
    reads = []

    def mutate_after_read(path):
        value = real(path)
        reads.append(path)
        Path(m['paths']['config.json']).write_text('{"model_type":"corrupt_pool"}')
        Path(m['staged'][m['paths']['config.json']]).write_text('{"model_type":"corrupt_stage"}')
        return value

    monkeypatch.setattr(cc, '_original_bootstrap_json', mutate_after_read)
    with owner:
        config, profile, construction, original = streaming_model.load_original_streaming_bootstrap(m['root'], owner)
        assert config.model_type == original['model_type'] == 'llama'
        assert config.hidden_size == 16 and not construction
        assert profile is not None and reads


@pytest.mark.parametrize('operation', ['context', 'layer', 'head', 'fp8'])
def test_original_gpu_routes_refuse_before_any_material_or_cuda_load(original_model, operation):
    m = original_model
    with _owner(m) as owner:
        before = list(m['checks'])
        with pytest.raises(RuntimeError, match='GPU loads/transfers are not qualified'):
            if operation == 'context':
                streaming_model._build_streaming_context(str(m['root']), device=torch.device('cuda'),
                    dtype=torch.float32, offload_folder='unused', source_authentication=owner)
            elif operation == 'layer':
                layer_streaming._read_layer_to_device('model.layers.', {}, {}, torch.float32,
                                                      torch.device('cuda'), source_authentication=owner)
            elif operation == 'head':
                layer_streaming._materialize(torch.nn.Linear(1, 1), [], {}, {}, torch.device('cuda'),
                                             torch.float32, source_authentication=owner)
            else:
                layer_streaming._apply_fp8_dequant_inplace({}, {}, torch.device('cuda'),
                                                          source_authentication=owner)
        assert m['checks'] == before and owner.material_live_bytes == 0


def test_loader_cpu_aliases_hold_material_after_automatic_read_window(original_model):
    m = original_model
    owner = _owner(m)
    key = 'model.layers.0.self_attn.q_proj.weight'
    shard = str(m['root'] / 'two.safetensors')
    values = layer_streaming._read_layer_to_device('model.layers.0.', {key: shard}, {key: key},
        torch.float32, torch.device('cpu'), source_authentication=owner)
    alias = values[key].detach()[1:]
    values.clear()
    assert owner.material_live_bytes == len(m['raws']['two.safetensors'])
    assert torch.equal(alias, m['expected'][key][1:])
    with pytest.raises(RuntimeError, match='live.*consumer'):
        owner.close()
    del alias
    gc.collect()
    assert owner.material_live_bytes == 0
    owner.close()


def test_existing_text_staging_and_owned_derived_config_match(original_model, monkeypatch):
    from prismaquant.sensitivity_probe import stage_text_only
    m = original_model
    raw = json.loads(m['raws']['config.json'])
    raw.update(hidden_size=128, text_config={'hidden_size': 16},
               vision_config={'hidden_size': 8}, architectures=['LlamaForConditionalGeneration'])
    Path(m['paths']['config.json']).write_text(json.dumps(raw))
    _rebind(m, monkeypatch)
    monkeypatch.setenv('PRISMAQUANT_TMPDIR', str(m['tmp'] / 'legacy-text-stage'))
    source = str(Path(m['paths']['config.json']).parent)
    legacy = streaming_model.load_streaming_auto_config(source, stage_text_only(source))
    with _owner(m) as owner:
        owned, _, construction, original = streaming_model.load_original_streaming_bootstrap(m['root'], owner)
        assert not construction and original == raw
        assert type(owned) is type(legacy)
        left, right = owned.to_dict(), legacy.to_dict()
        left.pop('_name_or_path', None)
        right.pop('_name_or_path', None)
        assert left == right
        assert owned.hidden_size == 16 and not hasattr(owned, 'vision_config')


def test_multimodal_construction_does_not_materialize_visual_input(original_model, monkeypatch):
    from prismaquant import model_profiles
    from prismaquant.model_profiles.default import DefaultProfile

    class ConstructionProfile(DefaultProfile):
        def requires_multimodal_skeleton(self):
            return True

    m = original_model
    real = model_profiles.detect_profile
    def detect(path, *, config=None):
        selected = real(path, config=config)
        selected.__class__ = ConstructionProfile
        return selected
    real_visual = streaming_model._find_visual_module
    calls = []
    def visual(*args, **kwargs):
        calls.append(args[0])
        assert len(calls) == 1, 'construction fact authorized visual materialization'
        return real_visual(*args, **kwargs)
    monkeypatch.setattr(model_profiles, 'detect_profile', detect)
    monkeypatch.setattr(streaming_model, '_find_visual_module', visual)
    owner = _owner(m)
    context = _context(m, owner, m['root'])
    assert context.multimodal and context.visual_module is None
    calls.clear()  # The inspection spy must not retain the meta skeleton.
    context.shutdown()
    del context
    gc.collect()
    owner.close()


@pytest.mark.parametrize('field', ['configuration_files', 'auto_map'])
def test_dynamic_config_refuses_before_stock_autoconfig(original_model, monkeypatch, field):
    m = original_model
    raw = json.loads(m['raws']['config.json'])
    raw[field] = ['other.json'] if field == 'configuration_files' else {'AutoConfig': 'dynamic.Config'}
    Path(m['paths']['config.json']).write_text(json.dumps(raw))
    _rebind(m, monkeypatch)
    def unexpected(*args, **kwargs):
        raise AssertionError('dynamic config reached stock adapter')
    monkeypatch.setattr(streaming_model, '_streaming_auto_config_from_bytes', unexpected)
    with pytest.raises(RuntimeError, match='dynamic original bootstrap'):
        _owner(m)


def test_loader_cancellation_precedes_material_window(original_model):
    m = original_model
    with _owner(m) as owner:
        before = list(m['checks'])
        cancel = threading.Event()
        cancel.set()
        with pytest.raises(CancelledError):
            layer_streaming._read_layer_to_device('model.layers.0.', {}, {}, torch.float32,
                torch.device('cpu'), source_authentication=owner, cancel=cancel)
        assert m['checks'] == before and owner.material_live_bytes == 0


def test_loader_window_bound_includes_native_outputs_from_previous_shard(original_model):
    m = original_model
    m['options']['max_material_bytes'] = max(len(raw) for raw in m['raws'].values())
    owner = _owner(m)
    head = 'model.embed_tokens.weight'
    body = 'model.layers.0.self_attn.q_proj.weight'
    first = layer_streaming._read_layer_to_device('model.embed_tokens.',
        {head: str(m['root'] / 'one.safetensors')}, {head: head}, torch.float32,
        torch.device('cpu'), source_authentication=owner)
    before = list(m['checks'])
    with pytest.raises(RuntimeError, match='material byte bound'):
        layer_streaming._read_layer_to_device('model.layers.0.',
            {body: str(m['root'] / 'two.safetensors')}, {body: body}, torch.float32,
            torch.device('cpu'), source_authentication=owner)
    assert m['checks'] == before
    first.clear()
    gc.collect()
    assert owner.material_live_bytes == 0
    second = layer_streaming._read_layer_to_device('model.layers.0.',
        {body: str(m['root'] / 'two.safetensors')}, {body: body}, torch.float32,
        torch.device('cpu'), source_authentication=owner)
    assert torch.equal(second[body], m['expected'][body])
    second.clear()
    gc.collect()
    owner.close()


def test_owned_profile_index_controls_qwen_source_namespace(original_model, monkeypatch):
    m = original_model
    raw = json.loads(m['raws']['config.json'])
    raw.update(model_type='qwen3_5_moe', architectures=['Qwen3_5MoeForCausalLM'])
    Path(m['paths']['config.json']).write_text(json.dumps(raw))
    _rebind(m, monkeypatch)
    with _owner(m) as owner:
        profile = layer_streaming._source_profile(str(m['root']), owner)
        assert profile.source_tensor_name('model.layers.0.self_attn.q_proj.weight') == 'model.layers.0.self_attn.q_proj.weight'
        assert owner.material_live_bytes == 0


def test_mid_layer_reader_failure_retains_aliases_until_error_frame_release(original_model, monkeypatch):
    m = original_model
    owner = _owner(m)
    keys = ['model.layers.0.self_attn.q_proj.weight', 'model.layers.0.self_attn.k_proj.weight']
    shard = str(m['root'] / 'two.safetensors')
    real = cc._CaptureSourceSafeOpen.get_tensor
    reads = []
    def failed(reader, key):
        reads.append(key)
        if len(reads) == 2:
            raise RuntimeError('injected second payload failure')
        return real(reader, key)
    monkeypatch.setattr(cc._CaptureSourceSafeOpen, 'get_tensor', failed)
    monkeypatch.setenv('PRISMAQUANT_LAYER_READ_THREADS', '1')
    with pytest.raises(RuntimeError, match='second payload failure') as failure:
        layer_streaming._read_layer_to_device('model.layers.0.', {key: shard for key in keys},
            {key: key for key in keys}, torch.float32, torch.device('cpu'), source_authentication=owner)
    assert len(reads) == 2
    assert owner.material_live_bytes == len(m['raws']['two.safetensors'])
    with pytest.raises(RuntimeError, match='live.*consumer'):
        owner.close()
    traceback.clear_frames(failure.value.__traceback__)
    failure.value.__traceback__ = None
    del failure
    gc.collect()
    assert owner.material_live_bytes == 0
    owner.close()
