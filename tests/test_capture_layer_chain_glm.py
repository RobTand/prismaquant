"""The capture chain over the real GLM skeleton and shard loader (PQ #1885).

Needs transformers' ``glm5_next`` (the ``-tf516`` venv) and, for the campaign
CLI, the pinned Tessera producer.
"""
import json
import os
import re
from pathlib import Path

import pytest
import torch

from test_glm_campaign_streaming import glm_checkpoint, write_original_layout_checkpoint  # noqa: F401
from prismaquant.cost_streaming import build_streamed_causal_lm
from prismaquant.model_profiles.glm5_next import Glm5NextProfile


@pytest.fixture
def _cpu_glm_kernels(monkeypatch):
    # Import inside the fixture: importing this autouse fixture at module scope
    # would also replace the real kernels in the CUDA controls below.
    from test_glm5_next_streamed_forward_parity import _torch_only_causal_conv1d

    _torch_only_causal_conv1d.__wrapped__(monkeypatch)


def _runner(source, offload):
    return build_streamed_causal_lm(str(source), device=torch.device("cpu"),
        dtype=torch.bfloat16, offload_folder=str(offload), profile=Glm5NextProfile(),
        max_cache_slots=2, prefetch_workers=1, prefetch_min_available_gb=0,
        cache_headroom_gb=0, prefetch_lookahead=1, require_prefetched_residency=True,
        attn_implementation="eager")


class _Frontier:
    def __init__(self, layer, hidden):
        self.layer, self.hidden = layer, hidden

    def hidden_batches(self):
        yield from self.hidden


def _visit(runner, tokens, targets, *, start=None, stop_layer=None):
    """Captured X/H/rows/max per unit, the hidden after the last layer, and the visited layers."""
    from prismaquant.tessera_campaign import _collect_activations
    captured, final, visited = [{}, {}, {}, {}], [], []

    def visit(layer, forward_batch):
        names = [name for name in targets if runner.layer_index_for_qname(name) == layer]
        values = _collect_activations(runner.model, names, tokens, 7, "cpu",
            want_hessian=True, profile=runner.profile, forward_batch=forward_batch)
        for actual, local in zip(captured, values):
            actual.update(local)
        visited.append(layer)

    runner.visit_layer_batches(tokens, visit, start=start, stop_layer=stop_layer,
        boundary_consumer=lambda index, hidden: final.append(hidden.detach().cpu().clone()))
    return captured, final, visited


def test_glm_chained_visits_equal_the_monolith_and_their_witnesses_merge(
        glm_checkpoint, tmp_path, _cpu_glm_kernels):
    """Two quanta over the real GLM forward equal one traversal, and so do their witnesses."""
    from prismaquant.routed_experts import profile_declared_packed_expert_projections
    from prismaquant.streaming_model import merge_selected_initialization_witnesses
    reference, source = glm_checkpoint
    profile = Glm5NextProfile()
    dense = [name for name, module in reference.named_modules()
             if isinstance(module, torch.nn.Linear) and ".layers." in name
             and ".mlp." in name and not profile.is_pinned_name(name)]
    targets = [*dense, *(m.qname for m in profile_declared_packed_expert_projections(reference, profile))]
    torch.manual_seed(441)
    tokens = [torch.randint(2, 128, (1, 257)), torch.randint(2, 128, (1, 257))]
    runner = _runner(source, tmp_path / "monolith")
    try:
        runner.context.begin_source_initialization_audit()
        expected, expected_final, visited = _visit(runner, tokens, targets)
        contract = runner.context.source_initialization_contract()
    finally:
        runner.shutdown()
    assert visited == [0, 1]
    captured, hidden, witnesses = [{}, {}, {}, {}], None, []
    for start, stop in ((0, 1), (1, 2)):
        runner = _runner(source, tmp_path / f"quantum-{start}")
        try:
            runner.context.begin_source_initialization_audit()
            values, hidden, visited = _visit(runner, tokens, targets, stop_layer=stop,
                start=None if start == 0 else _Frontier(start, hidden))
            witnesses.append(runner.context.source_selected_initialization_witness(range(start, stop)))
        finally:
            runner.shutdown()
        assert visited == list(range(start, stop))
        if stop < 2:
            # The GLM boundary is stored in the runner's own dtype: no cast.
            assert all(value.dtype == torch.bfloat16 for value in hidden)
        for actual, local in zip(captured, values):
            actual.update(local)
    for actual, wanted in zip(captured[:2], expected[:2]):
        assert actual.keys() == wanted.keys()
        for name in actual:
            assert torch.equal(actual[name], wanted[name]), name
    assert captured[2:] == expected[2:]
    assert all(torch.equal(a, b) for a, b in zip(hidden, expected_final))
    assert merge_selected_initialization_witnesses(witnesses) == contract


# -- the campaign CLI ------------------------------------------------------------

def _three_layer_config():
    from test_glm5_next_streamed_forward_parity import _tiny_config
    config = _tiny_config()
    text = config.text_config
    text.hidden_size = 256
    text.intermediate_size = 512
    text.moe_intermediate_size = 256
    config.vision_config.out_hidden_size = 256
    text.num_hidden_layers = 3
    text.layer_types = ["linear_attention", "deepseek_sparse_attention", "linear_attention"]
    text.mlp_layer_types = ["dense", "sparse", "sparse"]
    text.indexer_types = ["full", "full", "full"]
    return type(config).from_dict(config.to_dict())


_LAYER = re.compile(r"\.layers\.(\d+)\.")


def _write_sharded_checkpoint(model, source):
    """The original-layout checkpoint, one shard per decoder layer plus head and vision."""
    from safetensors.torch import load_file, save_file
    write_original_layout_checkpoint(model, source)
    single = source / "model.safetensors"
    tensors = load_file(str(single))
    single.unlink()
    shards = {}
    for name, value in tensors.items():
        match = _LAYER.search(name)
        shard = ("model-visual.safetensors" if "visual" in name else
                 "model-head.safetensors" if match is None else
                 f"model-layer-{int(match[1]):03d}.safetensors")
        shards.setdefault(shard, {})[name] = value
    weight_map = {}
    for shard, part in shards.items():
        save_file(part, str(source / shard), metadata={"format": "pt"})
        weight_map.update({name: shard for name in part})
    total = sum(value.numel() * value.element_size() for value in tensors.values())
    (source / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2))
    return sorted(shards)


def _manifest(root):
    return json.loads((Path(root) / "capture_manifest.json").read_text())


def _assert_same_capture(expected_root, actual_root, *, what):
    """Two published captures are the same manifest and the same bytes, entry for entry."""
    expected, published = _manifest(expected_root), _manifest(actual_root)
    assert published == expected, what
    for name, entry in expected["entries"].items():
        before = torch.load(Path(expected_root) / entry['path'], weights_only=True)
        after = torch.load(Path(actual_root) / published['entries'][name]['path'], weights_only=True)
        assert before.keys() == after.keys(), (what, name)
        for key, value in before.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value.view(torch.uint8), after[key].view(torch.uint8)), (what, name, key)
            else:
                assert value == after[key], (what, name, key)


def _chain_equals_the_monolith(tmp_path, monkeypatch, policy, ranges, *, device):
    """Prep, quanta and join publish the monolith's capture; each file is hashed by its reader.

    The campaign places its model on CUDA whenever CUDA is available, so the
    CPU case hides it and the CUDA case records the device of every boundary
    tensor a quantum writes and of every one its successor reads.
    """
    from prismaquant import capture_layer_chain as chain
    from prismaquant import cost_streaming
    from prismaquant import tessera_calibration_cache as cache
    from prismaquant import tessera_campaign as campaign
    # Controlled legacy-mechanism fixture, not a qualified immutable provider.
    # The independent automatic-admission matrix exercises the real refusal.
    from prismaquant import tessera_calibration_cache as qualification_store
    monkeypatch.setattr(qualification_store, 'require_automatic_capture_source_recording', lambda: None)
    from test_glm5_next_streamed_forward_parity import _build_model
    from projection_producer_fixture import require_projection_producer
    require_projection_producer(monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_TMPDIR", str(tmp_path / "staging"))
    if device == "cpu":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(1885)
    source = tmp_path / "source"
    shards = _write_sharded_checkpoint(_build_model(_three_layer_config()).to(torch.bfloat16), source)
    assert "model-visual.safetensors" in shards and "model-head.safetensors" in shards
    tokens = [torch.arange(257).remainder(126).add(2).reshape(1, -1),
              torch.arange(257).flip(0).remainder(126).add(2).reshape(1, -1)]
    monkeypatch.setattr(campaign, '_calibration_tokens', lambda *_: (tokens, 'tiny GLM frozen draw'))
    census = tmp_path / 'census.json'
    common = ['--model', str(source), '--out', str(tmp_path / 'unused.pkl'),
              '--menu-mode', 'research', '--nsamples', '2', '--seqlen', '257',
              '--max-act-rows', '7', '--attention-implementation', 'eager', '--streaming',
              '--streaming-cache-headroom-gb', '0']
    capture = [*common, '--calibration-census', str(census)]
    if policy != "legacy":
        capture += ['--streaming-capture-policy', policy]
    assert campaign.main([*common, '--cache-dir', str(tmp_path / 'census-cache'),
                          '--census-out', str(census)]) == 0
    monolith = tmp_path / 'monolith'
    assert campaign.main([*capture, '--cache-dir', str(tmp_path / 'monolith-cache'),
                          '--capture-calibration-out', str(monolith)]) == 0
    if device == "cuda":
        # A control: a chain can only equal a monolith that equals itself.
        again = tmp_path / 'monolith-again'
        assert campaign.main([*capture, '--cache-dir', str(tmp_path / 'monolith-again-cache'),
                              '--capture-calibration-out', str(again)]) == 0
        _assert_same_capture(monolith, again,
                             what="two monolithic CUDA captures differ: the forward is not repeat-exact")

    # Every whole-file source hash in the tree: the capture identity's and the
    # descriptor owner's (``sha256``) and the streamed identity's.
    hashed = []

    def spy(original):
        def hash_file(path, *args, **kwargs):
            resolved = Path(path).resolve()
            if resolved.parent == source.resolve():
                hashed.append(resolved.name)
            return original(path, *args, **kwargs)
        return hash_file
    monkeypatch.setattr(cache, "sha256", spy(cache.sha256))
    monkeypatch.setattr(cost_streaming, "_file_sha256", spy(cost_streaming._file_sha256))

    # Where each boundary tensor lives as a quantum writes it, and as its
    # successor's frontier hands it to the forward (which moves it to the
    # runner's device).
    written, read = [], []
    write = cost_streaming.StreamedBoundaryArtifacts.write

    def recording_write(self, tensor, **kwargs):
        written.append(tensor.device.type)
        return write(self, tensor, **kwargs)
    monkeypatch.setattr(cost_streaming.StreamedBoundaryArtifacts, "write", recording_write)
    hidden_batches = chain._BoundaryFrontier.hidden_batches

    def recording_hidden_batches(self):
        for hidden in hidden_batches(self):
            read.append(hidden.device.type)
            yield hidden
    monkeypatch.setattr(chain._BoundaryFrontier, "hidden_batches", recording_hidden_batches)

    root = tmp_path / 'chain'
    storage = {"schema": "prismaquant.aura.boundary_storage.v2", "capture_order": "layer_major",
               "directory": str(tmp_path / "boundaries"), "max_resident_bytes": 64 << 20,
               "max_auxiliary_bytes": 1 << 20, "max_artifact_bytes": 1 << 30,
               "prefetch_batches": 1}
    chained = [*capture, '--capture-calibration-out', str(root)]
    assert campaign.main([*chained, '--cache-dir', str(tmp_path / 'prep-cache'),
        '--capture-chain', 'prep', '--capture-chain-ranges', ranges,
        '--capture-chain-boundary-storage', json.dumps(storage)]) == 0
    roster = {path.name for path in cache.capture_source_files(source)}
    assert hashed == []  # the prep reads no payload (PQ #1896)
    pairs = chain.parse_layer_ranges(ranges)
    prep = chain.read_prep(root)
    read_by_quanta = set()
    for start, stop in pairs:
        hashed.clear()
        assert campaign.main([*chained, '--cache-dir', str(tmp_path / f'quantum-{start}-cache'),
            '--capture-chain', 'quantum', '--capture-layer-range', f'{start}:{stop}']) == 0
        consumed = {"model-head.safetensors",
                    *(f"model-layer-{layer:03d}.safetensors" for layer in range(start, stop))}
        # Each file the quantum reads is hashed once, by that read; nothing else is.
        assert {name for name in hashed if name.endswith('.safetensors')} == consumed
        assert set(hashed) <= roster and len(hashed) == len(set(hashed))
        fragment = json.loads(chain.fragment_path(root, start, stop).read_text())
        receipt = fragment["source_authentication"]
        assert receipt["schema"] == cache.RECORDING_RECEIPT_SCHEMA
        assert {row["name"] for row in receipt["verified_files"]} == set(hashed)
        assert receipt["payload_bytes_hashed"] == sum(
            (source / name).stat().st_size for name in consumed)
        read_by_quanta |= set(hashed)
        if start:
            # The quantum left the boundary it started from in place.
            before = json.loads(chain.fragment_path(root, *pairs[pairs.index((start, stop)) - 1])
                                .read_text())["boundary"]
            assert before and all(Path(record["path"]).is_file() for record in before)
    # Every interior boundary was written from the runner's device and read
    # back by the next quantum; a silent CPU fallback fails the CUDA case here.
    passed = (len(pairs) - 1) * len(tokens)
    assert written == [device] * passed
    assert len(read) == passed
    hashed.clear()
    assert campaign.main([*chained, '--cache-dir', str(tmp_path / 'join-cache'),
                          '--capture-chain', 'join']) == 0
    # The join hashes only what no quantum read (the vision tower here), once.
    assert sorted(hashed) == sorted(roster - read_by_quanta)
    assert "model-visual.safetensors" in hashed
    record = json.loads(chain.join_path(root).read_text())
    assert record["retired_boundary_entries"] == passed
    entries = list((Path(storage["directory"]) / prep["session"]["generation"] / "entries").iterdir())
    assert entries == []
    _assert_same_capture(monolith, root, what="the chain's capture differs from the monolith's")
    cache.require_capture_contract(root / 'capture_manifest.json')


@pytest.mark.parametrize("policy,ranges", [
    ("legacy", "0:1,1:3"),
    ("legacy", "0:1,1:2,2:3"),
    ("shared-inputs-bounded-v1", "0:1,1:2,2:3"),
])
def test_glm_capture_chain_equals_the_monolith_entry_for_entry(
        tmp_path, monkeypatch, policy, ranges, _cpu_glm_kernels):
    """The chain on CPU, including on a box that has CUDA."""
    _chain_equals_the_monolith(tmp_path, monkeypatch, policy, ranges, device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA; certifies nothing when skipped")
@pytest.mark.parametrize("policy,ranges", [
    ("legacy", "0:1,1:2,2:3"),
    ("shared-inputs-bounded-v1", "0:1,1:2,2:3"),
])
def test_glm_capture_chain_equals_the_monolith_entry_for_entry_on_cuda(
        tmp_path, monkeypatch, request, policy, ranges):
    """The chain with the model on CUDA: boundaries leave and re-enter the device.

    Each quantum writes its boundary from CUDA tensors (``boundary_consumer`` ->
    ``StreamedBoundaryArtifacts.write``) and the next quantum's frontier moves
    the stored tensors back with ``.to(device)``. Both cases tile three
    ranges, so the middle quantum's written boundary is computed from a
    frontier it moved to the device: a CPU fallback on either side leaves a
    CPU tensor at that write. A bounded CUDA capture needs its release policy
    before the process starts, so that case runs in a child pytest (#1096).
    """
    from test_glm_campaign_streaming import bounded_capture_child_needed, run_in_bounded_capture_child
    if bounded_capture_child_needed(policy, cuda=True, environ=os.environ):
        run_in_bounded_capture_child(request, tmp_path)
        return
    _chain_equals_the_monolith(tmp_path, monkeypatch, policy, ranges, device="cuda")


# -- selected fresh captures (--units) -------------------------------------------

def _selected_units_file(tmp_path, source, census, *, layer=None):
    """A v1 --units file for one whole anchor group, optionally of one layer."""
    groups = census["anchor_groups"]
    keys = sorted(groups)
    if layer is not None:
        single_layer = [key for key in keys
                        if {_LAYER.search(name).group(1) for name in groups[key]} == {str(layer)}]
        if single_layer:
            keys = single_layer
    assert keys, "the census names no anchor group"
    key = keys[0]
    path = tmp_path / "units.json"
    path.write_text(json.dumps({
        "schema": "prismaquant.tessera_campaign_units.v1",
        "model": str(source), "layer_stride": 1,
        "groups": [{"key": key, "members": sorted(groups[key])}]}))
    return path, key, sorted(groups[key])


def test_selected_capture_scope_refusals(tmp_path):
    """A fresh capture prices whole groups: sampled, audited, partitioned and
    exact-member selections refuse; a whole v1 group derives its exact units."""
    from prismaquant import tessera_campaign as campaign
    resolved = {"g": ["m.a", "m.b"]}
    args = type("Args", (), {"model": "src", "layer_stride": 1, "units": "units.json",
                             "research_exact_member": None})()
    whole = {"schema": "prismaquant.tessera_campaign_units.v1",
             "groups": [{"key": "g", "members": ["m.a", "m.b"]}]}
    assert campaign.selected_capture_unit_names(whole, args=args, resolved=resolved) == ["m.a", "m.b"]
    sampled = {"schema": "prismaquant.tessera_campaign_units.v2", "groups": [dict(
        whole["groups"][0], sampled=["m.a"],
        inclusion_probability={"m.a": 0.5})]}
    with pytest.raises(RuntimeError, match="samples or audits"):
        campaign.selected_capture_unit_names(sampled, args=args, resolved=resolved)
    partition = {"schema": "prismaquant.tessera_campaign_units.v3", "groups": [dict(
        key="s:packed", members=["m.a", "m.b"],
        partition={"schema": "prismaquant.tessera_campaign_expert_partition.v1",
                   "experts_per_row": 8, "index": 0, "count": 2, "rate_q256": 64,
                   "members": ["m.a"]})]}
    with pytest.raises(RuntimeError, match="partition"):
        campaign.selected_capture_unit_names(partition, args=args, resolved=resolved)
    exact = type(args, (), {**args.__dict__, "research_exact_member": "m.a"})()
    with pytest.raises(RuntimeError, match="estimator"):
        campaign.selected_capture_unit_names(whole, args=exact, resolved=resolved)
    other_model = type(args, (), {**args.__dict__, "model": "other"})()
    with pytest.raises(Exception, match="model/layer_stride"):
        campaign.selected_capture_unit_names(whole, args=other_model, resolved=resolved)
    outside = {"schema": "prismaquant.tessera_campaign_units.v1",
               "groups": [{"key": "elsewhere", "members": ["m.a", "m.b"]}]}
    with pytest.raises(RuntimeError, match="does not contain"):
        campaign.selected_capture_unit_names(outside, args=args, resolved=resolved)


def test_glm_selected_capture_chain_equals_the_monolith_on_its_units(
        tmp_path, monkeypatch, _cpu_glm_kernels):
    """A --units full-group capture chains like the monolith: the quanta
    collect only the selected units but forward every source layer, the join
    still holds the full layer tiling and witness, and the published manifest
    carries exactly the selected units under an explicit selected scope."""
    from prismaquant import capture_layer_chain as chain
    from prismaquant import tessera_calibration_cache as cache
    from prismaquant import tessera_campaign as campaign
    from prismaquant import tessera_calibration_cache as qualification_store
    monkeypatch.setattr(qualification_store, 'require_automatic_capture_source_recording', lambda: None)
    from test_glm5_next_streamed_forward_parity import _build_model
    from projection_producer_fixture import require_projection_producer
    require_projection_producer(monkeypatch)
    monkeypatch.setenv("PRISMAQUANT_TMPDIR", str(tmp_path / "staging"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    torch.manual_seed(20261001)
    source = tmp_path / "source"
    _write_sharded_checkpoint(_build_model(_three_layer_config()).to(torch.bfloat16), source)
    tokens = [torch.arange(257).remainder(126).add(2).reshape(1, -1),
              torch.arange(257).flip(0).remainder(126).add(2).reshape(1, -1)]
    monkeypatch.setattr(campaign, '_calibration_tokens', lambda *_: (tokens, 'tiny GLM frozen draw'))
    census_path = tmp_path / 'census.json'
    common = ['--model', str(source), '--out', str(tmp_path / 'unused.pkl'),
              '--menu-mode', 'research', '--nsamples', '2', '--seqlen', '257',
              '--max-act-rows', '7', '--attention-implementation', 'eager', '--streaming',
              '--streaming-cache-headroom-gb', '0']
    capture = [*common, '--calibration-census', str(census_path)]
    assert campaign.main([*common, '--cache-dir', str(tmp_path / 'census-cache'),
                          '--census-out', str(census_path)]) == 0
    census = json.loads(census_path.read_text())
    units_path, group_key, selected = _selected_units_file(tmp_path, source, census, layer=2)
    capture += ['--units', str(units_path)]

    monolith = tmp_path / 'monolith'
    assert campaign.main([*capture, '--cache-dir', str(tmp_path / 'monolith-cache'),
                          '--capture-calibration-out', str(monolith)]) == 0
    manifest = _manifest(monolith)
    assert set(manifest['entries']) == set(selected)
    assert manifest['identity']['unit_scope'] == 'selected'
    assert set(manifest['identity']['units']) == set(selected)
    # The capture retains the full draw's calibration: each entry's H and
    # count describe every routed row, not the retained scoring prefix.
    values, _receipt = cache.prefetch_capture(monolith / 'capture_manifest.json',
        expected_identity=manifest['identity'], census=census, names=selected, device='cpu')
    assert values[2] == {name: census['counts'][name] for name in selected}
    assert values[3] == {name: census['max_abs'][name] for name in selected}

    storage = {"schema": "prismaquant.aura.boundary_storage.v2", "capture_order": "layer_major",
               "directory": str(tmp_path / "boundaries"), "max_resident_bytes": 64 << 20,
               "max_auxiliary_bytes": 1 << 20, "max_artifact_bytes": 1 << 30,
               "prefetch_batches": 1}
    root = tmp_path / 'chain'
    chained = [*capture, '--capture-calibration-out', str(root)]
    assert campaign.main([*chained, '--cache-dir', str(tmp_path / 'prep-cache'),
        '--capture-chain', 'prep', '--capture-chain-ranges', '0:1,1:2,2:3',
        '--capture-chain-boundary-storage', json.dumps(storage)]) == 0
    prep = chain.read_prep(root)
    # The prep has no model: it derived the same selected scope from the census.
    assert prep['identity']['unit_scope'] == 'selected'
    assert set(prep['identity']['units']) == set(selected)
    for start, stop in [(0, 1), (1, 2), (2, 3)]:
        assert campaign.main([*chained, '--cache-dir', str(tmp_path / f'quantum-{start}-cache'),
            '--capture-chain', 'quantum', '--capture-layer-range', f'{start}:{stop}']) == 0
        fragment = json.loads(chain.fragment_path(root, start, stop).read_text())
        expected_units = {name for name in selected
                          if start <= int(_LAYER.search(name).group(1)) < stop}
        # Each quantum journals only its own selected units; a range with none
        # still forwards every batch and journals an empty map.
        assert set(fragment['units']) == expected_units
        assert fragment['source_authentication']['verified_files'], (
            f"quantum {start}:{stop} read no source; it did not forward its layers")
    assert campaign.main([*chained, '--cache-dir', str(tmp_path / 'join-cache'),
                          '--capture-chain', 'join']) == 0
    _assert_same_capture(monolith, root, what="the selected chain differs from the selected monolith")
    cache.require_capture_contract(root / 'capture_manifest.json')
