"""Campaign source qualification uses the real GLM skeleton and shard loader."""
import os

import pytest
import torch

from test_glm5_next_streamed_forward_parity import (
    _build_tiny_model, _torch_only_causal_conv1d,
)
from prismaquant.cost_streaming import build_streamed_causal_lm
from prismaquant.model_profiles.glm5_next import Glm5NextProfile


def write_original_layout_checkpoint(model, source):
    model.save_pretrained(source, safe_serialization=True)
    # Exercise the same unpacked/legacy-name checkpoint bridge as the real
    # source, including expert fusion, KDA convolution and mHC renames.
    from safetensors.torch import save_file
    tensors = {}
    for name, value in model.state_dict().items():
        if name.endswith('.mlp.experts.gate_up_proj'):
            prefix = name.removesuffix('gate_up_proj')
            for expert, slab in enumerate(value):
                gate, up = slab.chunk(2, dim=0)
                tensors[f'{prefix}{expert}.gate_proj.weight'] = gate.clone().contiguous()
                tensors[f'{prefix}{expert}.up_proj.weight'] = up.clone().contiguous()
        elif name.endswith('.mlp.experts.down_proj'):
            prefix = name.removesuffix('down_proj')
            for expert, slab in enumerate(value):
                tensors[f'{prefix}{expert}.down_proj.weight'] = slab.clone().contiguous()
        elif name.endswith('.self_attn.conv1d.weight'):
            prefix = name.removesuffix('conv1d.weight')
            for label, slab in zip(('q', 'k', 'v'), value.chunk(3, dim=0)):
                tensors[f'{prefix}{label}_conv1d.weight'] = slab.clone().contiguous()
        else:
            key = name.replace('.self_attn.forget_gate.', '.self_attn.')
            key = key.replace('.attn_hc.', '.hc_attn_').replace('.ffn_hc.', '.hc_ffn_')
            tensors[key] = value.clone().contiguous()
    save_file(tensors, str(source/'model.safetensors'))


@pytest.fixture
def glm_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISMAQUANT_TMPDIR", str(tmp_path / "staging"))
    source = tmp_path / "source"
    write_original_layout_checkpoint(_build_tiny_model(), source)
    # The public checkpoint loader is part of the baseline under test.
    from transformers import Glm5NextForConditionalGeneration
    model = Glm5NextForConditionalGeneration.from_pretrained(
        source, dtype=torch.float32, attn_implementation="eager").eval()
    return model, source


def test_campaign_glm_checkpoint_has_profile_aware_source_route(glm_checkpoint, tmp_path):
    """A GLM campaign must get past model loading through the profile route."""
    reference, source = glm_checkpoint
    from prismaquant import tessera_campaign as campaign
    # Exercise the production entry point up to its first post-load boundary.
    # This sentinel avoids external producer/calibration work, not model loading.
    class SourceLoaded(Exception):
        pass
    def population(model, profile, stride):
        assert profile.name == "glm5_next"
        assert len(model.model.language_model.layers) == 2
        raise SourceLoaded
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(campaign, "_require_campaign_population", population)
        from prismaquant import tessera_render
        patch.setattr(tessera_render, "tessera_encoder_hessian_status", lambda: {"accepted": True})
        with pytest.raises(SourceLoaded):
            campaign.main(["--model", str(source), "--out", str(tmp_path / "unused.pkl"),
                "--cache-dir", str(tmp_path / "cache"), "--census-out", str(tmp_path / "census.json"),
                "--attention-implementation", "eager", "--streaming",
                "--streaming-cache-headroom-gb", "0"])


def test_real_glm_streamed_source_matches_checkpoint(glm_checkpoint, tmp_path):
    reference, source = glm_checkpoint
    runner = build_streamed_causal_lm(str(source), device=torch.device("cpu"),
        dtype=torch.float32, offload_folder=str(tmp_path / "offload"),
        profile=Glm5NextProfile(), max_cache_slots=2, prefetch_workers=1,
        prefetch_min_available_gb=0, cache_headroom_gb=0,
        prefetch_lookahead=1, require_prefetched_residency=True,
        attn_implementation="eager")
    try:
        runner.context.begin_source_initialization_audit()
        with pytest.raises(RuntimeError, match="every source layer"):
            runner.context.source_initialization_contract()
        torch.manual_seed(441)
        ids = torch.randint(0, reference.config.text_config.vocab_size, (1, 257))
        with torch.inference_mode():
            expected = reference(input_ids=ids, use_cache=False).logits
            actual = runner(ids).logits
        assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
        assert all(p.is_meta for layer in runner.layers for p in layer.parameters())
        from prismaquant import validate_source_initialization_contract, pretrained_initialization_contract
        contract = runner.context.source_initialization_contract()
        assert contract['schema'] == 'prismaquant.streaming_initialization.v1'
        assert contract == validate_source_initialization_contract(contract)
        assert contract['num_layers'] == 2 and contract['persistent_tensors'] > 0
        with pytest.raises(ValueError, match='pretrained initialization'):
            pretrained_initialization_contract(runner.model)
    finally:
        runner.shutdown()


def test_layer_visit_preserves_campaign_prefix_and_full_hessian(glm_checkpoint, tmp_path):
    from prismaquant.tessera_campaign import _collect_activations
    from prismaquant.routed_experts import profile_declared_packed_expert_projections
    reference, source = glm_checkpoint
    profile = Glm5NextProfile()
    dense = [name for name, module in reference.named_modules()
             if isinstance(module, torch.nn.Linear) and ".layers." in name
             and ".mlp." in name and not profile.is_pinned_name(name)]
    targets = [*dense, *(m.qname for m in profile_declared_packed_expert_projections(reference, profile))]
    assert targets and any(".experts." in name for name in targets)
    torch.manual_seed(441)
    tokens = [torch.randint(2, 128, (1, 257)), torch.randint(2, 128, (1, 257))]
    baseline = _collect_activations(reference, targets, tokens, 7, "cpu", want_hessian=True, profile=profile)
    runner = build_streamed_causal_lm(str(source), device=torch.device("cpu"),
        dtype=torch.float32, offload_folder=str(tmp_path / "offload"), profile=profile,
        max_cache_slots=2, prefetch_workers=1, prefetch_min_available_gb=0,
        cache_headroom_gb=0, prefetch_lookahead=1, require_prefetched_residency=True,
        attn_implementation="eager")
    captured = [{}, {}, {}, {}]
    visited = []
    def visit(layer, forward_batch):
        names = [n for n in targets if runner.layer_index_for_qname(n) == layer]
        values = _collect_activations(runner.model, names, tokens, 7, "cpu",
            want_hessian=True, profile=profile, forward_batch=forward_batch)
        for actual, local in zip(captured, values):
            actual.update(local)
        visited.append(layer)
    try:
        runner.visit_layer_batches(tokens, visit)
        assert visited == [0, 1]
        assert captured[2:] == list(baseline[2:])
        for actual, expected in zip(captured[:2], baseline[:2]):
            assert actual.keys() == expected.keys()
            for name in actual:
                assert torch.equal(actual[name], expected[name]), name
        assert all(count > 7 for count in captured[2].values())
    finally:
        runner.shutdown()


def test_streamed_resource_plan_prices_layer_capture_and_packer_transients(glm_checkpoint):
    from prismaquant.autoscale import streamed_calibration_resources
    from prismaquant.routed_experts import profile_declared_packed_expert_projections
    reference, source = glm_checkpoint
    profile = Glm5NextProfile()
    shapes = {m.qname:list(m.weight.shape)
              for m in profile_declared_packed_expert_projections(reference, profile)}
    counts = {name: 514 for name in shapes}
    result = streamed_calibration_resources(source, unit_shapes=shapes, counts=counts,
        nsamples=2, seqlen=257, max_act_rows=7, cache_slots=2, prefetch_workers=1,
        headroom_gb=1)
    terms = result['terms']
    assert terms['loader_transient_bytes'] > 0
    assert terms['layer_hessian_bytes'] == sum(shape[1]**2*4 for shape in shapes.values())
    assert terms['layer_prefix_bytes'] == sum(shape[1]*7*4 for shape in shapes.values())
    assert terms['current_boundary_bytes'] == 2*257*64*4*2
    assert terms['microbatch_transition_bytes'] == 257*64*4*2
    assert result['memory_bytes'] == sum(terms.values())
    assert result['disk_bytes'] > result['full_hessian_bytes']+result['full_prefix_bytes']


@pytest.mark.parametrize('capture_policy', ['shared-inputs-release-v1', 'shared-inputs-bounded-v1'])
def test_streamed_campaign_publishes_original_layout_census_and_capture(glm_checkpoint, tmp_path, monkeypatch, capture_policy):
    """The CLI publishes a complete capture from real GLM source forwards."""
    import json
    from pathlib import Path
    from prismaquant import tessera_campaign as campaign
    from prismaquant import tessera_calibration_cache as capture
    from test_glm5_next_streamed_forward_parity import _tiny_config, _build_model
    config = _tiny_config()
    config.text_config.hidden_size = 256
    config.text_config.intermediate_size = 512
    config.text_config.moe_intermediate_size = 256
    config.vision_config.out_hidden_size = 256
    config = type(config).from_dict(config.to_dict())
    source = tmp_path/"campaign-source"
    write_original_layout_checkpoint(_build_model(config).to(torch.bfloat16), source)
    # Exercise the actual pinned producer. Resolve it the way
    # test_tessera_packed_plan_handoff does: from TESSERA_REPO, falling back
    # to the fleet's shared pinned checkout, and skip -- never fail -- where
    # neither exists (GitHub CI has no /mnt/shared).
    pinned = '/mnt/shared/tessera-measurements/first-model-20260907/inputs/tessera-382a1a97'
    producer = Path(os.environ.get('TESSERA_REPO') or pinned)
    if not producer.is_dir():
        pytest.skip('TESSERA_REPO must name the pinned producer checkout '
                    f'(unset, and {pinned} is absent)')
    monkeypatch.setenv('TESSERA_REPO', str(producer))
    tokens = [torch.arange(257).remainder(126).add(2).reshape(1, -1)]
    monkeypatch.setattr(campaign, '_calibration_tokens', lambda *_: (tokens, 'tiny GLM frozen draw'))
    census = tmp_path/'census.json'
    root = tmp_path/'capture'
    common = ['--model', str(source), '--out', str(tmp_path/'unused.pkl'),
        '--menu-mode', 'research', '--nsamples', '1', '--seqlen', '257', '--max-act-rows', '7',
        '--attention-implementation', 'eager', '--streaming',
        '--streaming-cache-headroom-gb', '0']
    campaign.main([*common, '--cache-dir', str(tmp_path/'census-cache'), '--census-out', str(census)])
    result = json.loads(census.read_text())
    assert result['model_load_contract']['schema'] == 'prismaquant.streaming_initialization.v1'
    assert result['counts'] and min(result['counts'].values()) > 7
    campaign.main([*common, '--cache-dir', str(tmp_path/'capture-cache'),
        '--calibration-census', str(census), '--capture-calibration-out', str(root)])
    manifest = root/'capture_manifest.json'
    capture.require_capture_contract(manifest)
    assert json.loads(manifest.read_text())['status'] == 'complete'

    # The explicit policy must traverse the same source and publish the same
    # per-unit tensors, while releasing each exhausted layer before return.
    from prismaquant.streaming_model import StreamingContext
    released = []
    release = StreamingContext.release_completed_layer
    def track_release(context, layer):
        release(context, layer)
        assert all(p.is_meta for p in context.layers[layer].parameters())
        assert layer not in context.layer_cache._cache
        released.append(layer)
    monkeypatch.setattr(StreamingContext, 'release_completed_layer', track_release)
    shared_root = tmp_path/'shared-capture'
    campaign.main([*common, '--cache-dir', str(tmp_path/'shared-capture-cache'),
        '--calibration-census', str(census), '--capture-calibration-out', str(shared_root),
        '--streaming-capture-policy', capture_policy])
    shared_manifest = shared_root/'capture_manifest.json'
    capture.require_capture_contract(shared_manifest)
    baseline = json.loads(manifest.read_text())
    candidate = json.loads(shared_manifest.read_text())
    assert candidate['identity'] == baseline['identity']
    assert candidate['entries'].keys() == baseline['entries'].keys()
    for name, entry in baseline['entries'].items():
        before = torch.load(root/entry['path'], weights_only=True)
        after = torch.load(shared_root/candidate['entries'][name]['path'], weights_only=True)
        assert before.keys() == after.keys()
        for key, value in before.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value.view(torch.uint8), after[key].view(torch.uint8)), (name, key)
            else:
                assert value == after[key], (name, key)
    assert released == list(range(config.text_config.num_hidden_layers))
    telemetry = json.loads((tmp_path/'shared-capture-cache'/'streamed-calibration-telemetry.json').read_text())
    assert all(row['capture_policy'] == capture_policy and
               row['completed_source_released'] for row in telemetry)
    if capture_policy == 'shared-inputs-bounded-v1':
        assert [[entry['layer'] for entry in row['settled_prefetch']] for row in telemetry] == [[1], []]
        for row in telemetry:
            guard = row['physical_memory_guard']
            if torch.cuda.is_available():
                assert guard['peak_conservative_bytes'] <= guard['budget_bytes']-guard['margin_bytes']
                assert guard['min_host_available_bytes'] >= guard['host_floor_bytes']
                assert guard['last_checkpoint']['cuda_reserved_bytes'] > 0
            else:
                assert guard is None

        # Reuse that actual complete capture through the selected-source CLI,
        # including one real pinned-producer anchor and a checkpoint resume.
        import pickle
        dense_group = next(key for key in result['anchor_groups']
                           if key.startswith('u:') and '.experts.' not in key)
        names = result['anchor_groups'][dense_group]
        selection_path = tmp_path/'selected-units.json'
        selection_path.write_text(json.dumps(dict(schema=campaign.UNITS_SCHEMA,
            model=str(source), layer_stride=1,
            groups=[dict(key=dense_group, members=names)])))
        def no_forward(*args, **kwargs):
            pytest.fail('selected anchor reuse repeated calibration')
        monkeypatch.setattr(campaign, '_collect_activations', no_forward)
        original_menus = campaign.expand_menus_for_targets
        def one_rung(weights, targets, **kwargs):
            assert set(weights) == set(names)
            assert all(not value.is_meta for value in weights.values())
            menus = original_menus(weights, targets, **kwargs)
            narrowed = {name: [row for row in rows
                if row.format_name == 'TESSERA_E4M3_K1_R1024'] for name, rows in menus.items()}
            assert all(narrowed.values()), 'native fixture must admit the measured rung'
            return narrowed
        monkeypatch.setattr(campaign, 'expand_menus_for_targets', one_rung)
        selected_out = tmp_path/'selected-cost.pkl'
        selected_argv = [*common, '--out', str(selected_out),
            '--cache-dir', str(tmp_path/'selected-cache'), '--units', str(selection_path),
            '--calibration-census', str(census), '--calibration-cache', str(shared_manifest),
            '--calibration-cache-sha256', capture.sha256(shared_manifest),
            '--max-rounds', '1']
        profile_root = os.environ.get('PRISMAQUANT_SELECTED_SOURCE_PROFILE')
        if profile_root:
            import time
            profile_root = Path(profile_root)
            profile_root.mkdir(parents=True, exist_ok=True)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
                torch.cuda.reset_peak_memory_stats()
            started = time.time()
            with torch.profiler.profile(activities=activities, profile_memory=True,
                                       record_shapes=True) as prof:
                assert campaign.main(selected_argv) == 0
            prof.export_chrome_trace(str(profile_root/'selected-anchor.trace.json'))
            (profile_root/'window.json').write_text(json.dumps(dict(start_unix=started,
                end_unix=time.time(), cuda_max_allocated_bytes=(
                    torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None))))
        else:
            assert campaign.main(selected_argv) == 0
        with selected_out.open('rb') as handle:
            selected = pickle.load(handle)
        assert set(selected['costs']) == set(names)
        receipt = selected['provenance']['selected_source_preparation']
        if profile_root:
            (profile_root/'selected-source-receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        # The guard reads absolute process bytes; the plan states deltas. The
        # gap between them used to be covered by declared headroom, which is
        # zero on this row (--streaming-cache-headroom-gb 0), so nothing but
        # the plan itself can cover the growth the guard observed. A plan that
        # undercharges fails here (RobTand/prismaquant#390).
        selected_guard = receipt['memory_guard']
        baseline = receipt['baseline']
        if torch.cuda.is_available():
            assert baseline is not None and baseline['measured_in_process'] is True
            assert selected_guard['last_checkpoint']['cuda_reserved_bytes'] > 0
            assert receipt['resources']['phases']['source_preparation'][
                'declared_headroom_bytes'] == 0
            growth = selected_guard['peak_conservative_bytes'] - baseline['bytes']
            assert growth <= receipt['resources']['memory_bytes'], dict(
                growth=growth, baseline=baseline, guard=selected_guard,
                plan=receipt['resources'])
        else:
            assert selected_guard is None and baseline is None
        assert receipt['source_forward_count'] == 0
        assert receipt['full_source_initialization_repeated'] is False
        assert all(row['source'] != 'cold' for row in receipt['layers'])
        journal = selected_out.with_suffix('.anchors.json')
        first = json.loads(journal.read_text())
        assert campaign.main(selected_argv) == 0
        second = json.loads(journal.read_text())
        assert first['identity_sha256'] == second['identity_sha256']
        with selected_out.open('rb') as handle:
            resumed = pickle.load(handle)
        assert selected['costs'] == resumed['costs']


def test_streamed_bf16_keeps_hf_strict_fp32_source_slots(glm_checkpoint, tmp_path):
    """BF16 forward weights retain HF-declared strict FP32 recurrence state."""
    from transformers import Glm5NextForConditionalGeneration
    _, source = glm_checkpoint
    reference = Glm5NextForConditionalGeneration.from_pretrained(
        source, dtype=torch.bfloat16, attn_implementation='eager').eval()
    expected = dict([*reference.named_parameters(), *reference.named_buffers()])
    runner = build_streamed_causal_lm(str(source), device=torch.device('cpu'),
        dtype=torch.bfloat16, offload_folder=str(tmp_path/'bf16-offload'),
        profile=Glm5NextProfile(), max_cache_slots=2, prefetch_workers=1,
        prefetch_min_available_gb=0, cache_headroom_gb=0,
        prefetch_lookahead=1, require_prefetched_residency=True,
        attn_implementation='eager')
    try:
        observed = []
        for layer in range(runner.num_layers):
            runner.context.schedule_prefetch(layer)
            runner.context.install(layer, require_prefetched=True)
            for name, value in [*runner.model.named_parameters(), *runner.model.named_buffers()]:
                if not name.startswith(f'{runner.context.layers_prefix}{layer}.'):
                    continue
                assert value.dtype == expected[name].dtype, name
                assert torch.equal(value, expected[name]), name
                if value.dtype == torch.float32:
                    observed.append(name)
            runner.context.unload(layer)
        assert any(name.endswith('conv1d.weight') for name in observed)
        assert any(name.endswith('e_score_correction_bias') for name in observed)
    finally:
        runner.shutdown()


def test_selected_glm_source_copies_dense_and_logical_expert_without_forward(glm_checkpoint, tmp_path, monkeypatch):
    from prismaquant.routed_experts import profile_declared_packed_expert_projections
    from prismaquant.autoscale import selected_anchor_resources
    reference, source = glm_checkpoint
    profile = Glm5NextProfile()
    expert = profile_declared_packed_expert_projections(reference, profile)[0]
    dense = next(name for name, module in reference.named_modules()
                 if isinstance(module, torch.nn.Linear) and '.layers.0.mlp.' in name)
    expected = {dense: dict(reference.named_modules())[dense].weight.to(torch.bfloat16),
                expert.qname: expert.weight.to(torch.bfloat16)}
    shapes = {name: list(value.shape) for name, value in expected.items()}
    resources = selected_anchor_resources(source, unit_shapes=shapes,
        counts={name: 257 for name in shapes}, max_act_rows=7,
        cache_slots=2, prefetch_workers=1, headroom_gb=0)
    runner = build_streamed_causal_lm(str(source), device=torch.device('cpu'),
        dtype=torch.bfloat16, offload_folder=str(tmp_path/'selected-offload'),
        profile=profile, max_cache_slots=2, prefetch_workers=1,
        prefetch_min_available_gb=0, cache_headroom_gb=0,
        prefetch_lookahead=1, require_prefetched_residency=True,
        attn_implementation='eager')
    def no_forward(*args, **kwargs):
        pytest.fail('selected source must not run a calibration forward')
    monkeypatch.setattr(runner.model, 'forward', no_forward)
    try:
        weights, receipt = runner.snapshot_selected_weights(list(expected),
            max_resident_bytes=resources['selected_source_weight_bytes'])
        for name in expected:
            assert torch.equal(weights[name], expected[name])
            assert weights[name].untyped_storage().nbytes() == weights[name].numel()*weights[name].element_size()
        assert receipt['source_forward_count'] == 0
        assert [row['layer'] for row in receipt['layers']] == sorted({runner.layer_index_for_qname(n) for n in expected})
        assert all(p.is_meta for layer in runner.layers for p in layer.parameters())
        assert runner.context.layer_cache._cache == {}
        assert runner.context._inflight == {}
    finally:
        runner.shutdown()


def test_layer_visit_releases_derived_batch_metadata(glm_checkpoint, tmp_path, monkeypatch):
    """A census-size draw must not retain one mask/position table per B1."""
    import weakref
    _, source = glm_checkpoint
    runner = build_streamed_causal_lm(str(source), device=torch.device('cpu'),
        dtype=torch.float32, offload_folder=str(tmp_path/'metadata-offload'),
        profile=Glm5NextProfile(), max_cache_slots=2, prefetch_workers=1,
        prefetch_min_available_gb=0, cache_headroom_gb=0,
        prefetch_lookahead=1, require_prefetched_residency=True,
        attn_implementation='eager')
    refs = []
    original = runner._prepare
    def track(value):
        if isinstance(value, torch.Tensor):
            refs.append(weakref.ref(value))
        elif isinstance(value, dict):
            for entry in value.values():track(entry)
        elif isinstance(value, (list,tuple)):
            for entry in value:track(entry)
    def prepare(ids):
        result = original(ids)
        track(result[1]);track(result[3]);track(result[4])
        return result
    monkeypatch.setattr(runner, '_prepare', prepare)
    tokens = [torch.arange(257).remainder(126).add(2).reshape(1,-1) for _ in range(3)]
    def visit(layer, forward):
        assert refs and all(ref() is None for ref in refs)
        for ids in tokens:
            forward(ids)
            assert all(ref() is None for ref in refs)
    try:
        runner.visit_layer_batches(tokens, visit)
    finally:
        runner.shutdown()


@pytest.mark.parametrize('extra', [[], ['--streaming'],
    ['--capture-calibration-out', '/unused/capture']])
def test_shared_capture_policy_refuses_other_routes_before_loading(extra, tmp_path, capsys):
    from prismaquant.tessera_campaign import main
    with pytest.raises(SystemExit) as caught:
        main(['--model', '/missing/source', '--out', str(tmp_path/'unused.pkl'),
              '--cache-dir', str(tmp_path/'cache'),
              '--streaming-capture-policy', 'shared-inputs-release-v1', *extra])
    assert caught.value.code == 2
    assert '--streaming-capture-policy requires --streaming and --capture-calibration-out' in capsys.readouterr().err
    assert not (tmp_path/'cache').exists()
