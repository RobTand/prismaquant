"""A selected snapshot uses the shared prefetch cache without unrelated weights."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file


def test_selected_snapshot_reads_only_requested_dense_weights(tmp_path, monkeypatch):
    from prismaquant import layer_streaming as ls, streaming_model as sm
    from prismaquant.cost_streaming import StreamedCausalLM

    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.layers = torch.nn.ModuleList([torch.nn.Module()])
    layer = model.model.layers[0]
    for name in ("gate", "up", "unrelated"):
        setattr(layer, name, torch.nn.Linear(4, 3, bias=False, device="meta"))
    source = {f"model.layers.0.{name}.weight": torch.full((3, 4), float(i))
              for i, name in enumerate(("gate", "up", "unrelated"), 1)}
    path = tmp_path / "weights.safetensors"
    save_file(source, path)
    shards = {name: str(path) for name in source}
    keys = {name: name for name in source}
    read = []
    original = sm._read_layer_to_device

    def observe(prefix, shard_map, *args, **kwargs):
        read.extend(name for name in shard_map if name.startswith(prefix))
        return original(prefix, shard_map, *args, **kwargs)

    monkeypatch.setattr(sm, "_read_layer_to_device", observe)
    profile = SimpleNamespace(per_expert_moe_regex=lambda: None,
                              concat_merge_groups=lambda: ())
    from prismaquant import routed_experts
    monkeypatch.setattr(routed_experts, "profile_declared_packed_expert_projections", lambda *_: [])
    monkeypatch.setattr(routed_experts, "refresh_packed_expert_projections", lambda *_: [])
    pool = ThreadPoolExecutor(max_workers=1)
    context = sm.StreamingContext(
        model=model, base_model=model.model, layers=model.model.layers,
        layers_prefix="model.layers.", num_layers=1,
        install_resolvers=[ls._build_install_resolver(model, "model.layers.0")],
        weight_shard=shards, weight_ckpt=keys,
        layer_cache=sm.LayerCache(max_bytes=1024**2, max_entries=2),
        prefetch_pool=pool, device=torch.device("cpu"), dtype=torch.float32,
        offload_folder=str(tmp_path / "offload"), estimated_layer_bytes=144,
    )
    # Before the implementation this source-only intent is ignored and the
    # existing snapshot reads the unrelated tensor along with the requested two.
    context.source_snapshot_only = True
    runner = StreamedCausalLM(context, profile, prefetch_lookahead=1,
                             require_prefetched_residency=True)
    names = ["model.layers.0.gate", "model.layers.0.up"]
    try:
        weights, receipt = runner.snapshot_selected_weights(names, max_resident_bytes=96)
        assert read == [name + ".weight" for name in names]
        assert receipt["source_forward_count"] == 0
        assert all(torch.equal(weights[name], source[name + ".weight"]) for name in names)
        assert all(p.is_meta for p in model.parameters())
    finally:
        runner.shutdown()


def test_snapshot_context_rejects_forward_install_before_reading(tmp_path):
    from prismaquant.streaming_model import StreamingContext
    context = object.__new__(StreamingContext)
    context.source_snapshot_only = True
    context.ensure_loaded = lambda *_a, **_k: pytest.fail("read source for a forbidden forward")
    with pytest.raises(RuntimeError, match="snapshot"):
        context.install(0)


from test_glm_campaign_streaming import glm_checkpoint


@pytest.mark.parametrize('selection', ['dense', 'expert'])
def test_glm_snapshot_preserves_source_bytes_without_head_or_unrelated_reads(
    glm_checkpoint, tmp_path, monkeypatch, selection,
):
    import hashlib
    from prismaquant import autoscale, streaming_model as sm
    from prismaquant.cost_streaming import build_streamed_causal_lm
    from prismaquant.model_profiles.glm5_next import Glm5NextProfile
    from prismaquant.routed_experts import profile_declared_packed_expert_projections
    from prismaquant.tessera_calibration_cache import CaptureSourceAuthentication

    reference, source = glm_checkpoint
    profile = Glm5NextProfile()
    if selection == 'expert':
        member = next(m for m in profile_declared_packed_expert_projections(reference, profile)
                      if m.qname.endswith('.gate_proj'))
        name, expected = member.qname, member.weight.detach()
    else:
        name, module = next((n, m) for n, m in reference.named_modules()
                            if isinstance(m, torch.nn.Linear) and '.shared_experts.gate_proj' in n)
        expected = module.weight.detach()
    shape = list(expected.shape)
    options = dict(unit_shapes={name: shape}, counts={name: 2}, max_act_rows=2,
                   cache_slots=2, prefetch_workers=1, headroom_gb=0)
    whole = autoscale.selected_anchor_resources(source, **options)
    selected = autoscale.selected_anchor_resources(source, **options,
        source_snapshot_policy='selected-tensors-v1')
    assert selected['source_header_sha256'] == whole['source_header_sha256']
    before, after = (plan['phases']['source_preparation'] for plan in (whole, selected))
    assert before['nonbody_source_bytes'] > after['nonbody_source_bytes'] == 0
    assert after['source_window_bytes'] < before['source_window_bytes']
    assert selected['phases']['resident_anchors']['source_validation_bytes'] == (
        whole['phases']['resident_anchors']['source_validation_bytes'])
    keys = selected['source_tensor_keys']
    if selection == 'dense':
        assert keys == [name+'.weight']
        assert after['loader_transient_bytes'] == 0
    else:
        # A single projected gate owns the full gate/up parent, across experts.
        expert_count = member.module.gate_up_proj.shape[0]
        assert len(keys) == expert_count*2
        assert all(k.endswith(('.gate_proj.weight', '.up_proj.weight')) for k in keys)
        assert after['loader_transient_bytes'] > 0
    digests = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
               for path in source.iterdir() if path.is_file()}
    auth = CaptureSourceAuthentication(source,
        dict(source_files=digests, census_sha256='a'*64), {}, manifest_sha256='b'*64)
    reads = []
    original = sm._read_layer_to_device
    def observe(prefix, shards, *args, **kwargs):
        reads.extend(key for key in shards if key.startswith(prefix))
        return original(prefix, shards, *args, **kwargs)
    monkeypatch.setattr(sm, '_read_layer_to_device', observe)
    monkeypatch.setattr(sm, '_materialize', lambda *_a, **_k: pytest.fail('materialized nonbody source'))
    runner = build_streamed_causal_lm(str(source), device=torch.device('cpu'),
        dtype=torch.bfloat16, offload_folder=str(tmp_path/'offload'), profile=profile,
        max_cache_slots=2, prefetch_workers=1, prefetch_min_available_gb=0,
        cache_headroom_gb=0, prefetch_lookahead=1, require_prefetched_residency=True,
        attn_implementation='eager', source_authentication=auth, source_snapshot_only=True)
    try:
        assert all(p.is_meta for p in runner.model.parameters())
        with pytest.raises(RuntimeError, match='snapshot'):
            runner.context.ensure_loaded(0)
        with pytest.raises(RuntimeError, match='resident byte budget'):
            runner.snapshot_selected_weights([name], max_resident_bytes=1)
        assert reads == []
        weights, receipt = runner.snapshot_selected_weights([name],
            max_resident_bytes=selected['selected_source_weight_bytes'], expected_source_keys=keys)
        assert sorted(reads) == keys
        assert torch.equal(weights[name], expected.to(torch.bfloat16))
        assert receipt['nonbody_materialized'] is False
        assert all(p.is_meta for p in runner.model.parameters())
        with pytest.raises(RuntimeError, match='snapshot'):
            runner._prepare(torch.ones((1, 1), dtype=torch.long))
        with pytest.raises(RuntimeError, match='snapshot'):
            runner.context.begin_source_initialization_audit()
    finally:
        runner.shutdown()
        auth.close()


def test_dependency_closure_refuses_missing_and_ambiguous_concat():
    from prismaquant.layer_streaming import selected_weight_source_keys
    profile = SimpleNamespace(per_expert_moe_regex=lambda: None,
        concat_merge_groups=lambda: (('conv.weight', ('q.weight', 'k.weight', 'v.weight'), 0),))
    names = ['model.layers.0.conv']
    keys = [f'model.layers.0.{label}.weight' for label in ('q', 'k', 'v')]
    assert selected_weight_source_keys(names, profile, keys) == tuple(sorted(keys))
    with pytest.raises(RuntimeError, match='incomplete concat'):
        selected_weight_source_keys(names, profile, keys[:-1])
    with pytest.raises(RuntimeError, match='ambiguous'):
        selected_weight_source_keys(names, profile, keys+[names[0]+'.weight'])
    with pytest.raises(RuntimeError, match='no checkpoint dependency'):
        selected_weight_source_keys(['model.layers.0.missing'], profile, keys)


def test_snapshot_builder_requires_authentication_before_source_io():
    from prismaquant.streaming_model import _build_streaming_context
    with pytest.raises(RuntimeError, match='authenticated'):
        _build_streaming_context('/does-not-exist', device=torch.device('cpu'),
            dtype=torch.bfloat16, offload_folder='/unused', source_snapshot_only=True)


def test_native_packed_checkpoint_dependency_uses_the_complete_parent():
    from prismaquant.layer_streaming import selected_weight_source_keys
    from prismaquant.model_profiles.glm5_next import Glm5NextProfile
    prefix = 'model.language_model.layers.4.mlp.experts.'
    assert selected_weight_source_keys([prefix+'7.gate_proj'], Glm5NextProfile(),
        [prefix+'gate_up_proj', prefix+'down_proj']) == (prefix+'gate_up_proj',)


def test_resource_policy_rejects_unknown_snapshot_policy():
    from prismaquant.autoscale import selected_anchor_resources
    with pytest.raises(ValueError, match='snapshot policy'):
        selected_anchor_resources('/does-not-exist', unit_shapes={'layers.0.a': [2, 2]},
            counts={'layers.0.a': 1}, max_act_rows=1, cache_slots=2,
            prefetch_workers=1, headroom_gb=0, source_snapshot_policy='typo')
