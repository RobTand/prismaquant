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


def test_glm_chained_visits_equal_the_monolith_and_their_witnesses_merge(glm_checkpoint, tmp_path):
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


@pytest.mark.parametrize("policy,ranges", [
    ("legacy", "0:1,1:3"),
    ("legacy", "0:1,1:2,2:3"),
    ("shared-inputs-bounded-v1", "0:1,1:2,2:3"),
])
def test_glm_capture_chain_equals_the_monolith_entry_for_entry(tmp_path, monkeypatch, policy, ranges):
    """Prep, quanta and join publish the monolith's capture; each quantum hashes only what it reads."""
    from prismaquant import capture_layer_chain as chain
    from prismaquant import tessera_calibration_cache as cache
    from prismaquant import tessera_campaign as campaign
    from test_glm5_next_streamed_forward_parity import _build_model
    pinned = '/mnt/shared/tessera-measurements/first-model-20260907/inputs/tessera-382a1a97'
    producer = Path(os.environ.get('TESSERA_REPO') or pinned)
    if not producer.is_dir():
        pytest.skip('TESSERA_REPO must name the pinned producer checkout '
                    f'(unset, and {pinned} is absent)')
    monkeypatch.setenv('TESSERA_REPO', str(producer))
    monkeypatch.setenv("PRISMAQUANT_TMPDIR", str(tmp_path / "staging"))
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

    # Every whole-file source hash in the tree: the capture identity's and the
    # descriptor owner's (``sha256``) and the streamed identity's.
    from prismaquant import cost_streaming
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
    assert sorted(hashed) == sorted(roster)  # the prep hashes the whole source once
    metadata = {name for name in roster if not name.endswith('.safetensors')}
    pairs = chain.parse_layer_ranges(ranges)
    prep = chain.read_prep(root)
    for start, stop in pairs:
        hashed.clear()
        assert campaign.main([*chained, '--cache-dir', str(tmp_path / f'quantum-{start}-cache'),
            '--capture-chain', 'quantum', '--capture-layer-range', f'{start}:{stop}']) == 0
        consumed = {"model-head.safetensors",
                    *(f"model-layer-{layer:03d}.safetensors" for layer in range(start, stop))}
        assert {name for name in hashed if name.endswith('.safetensors')} == consumed
        assert set(hashed) == consumed | metadata and len(hashed) == len(set(hashed))
        fragment = json.loads(chain.fragment_path(root, start, stop).read_text())
        receipt = fragment["source_authentication"]
        assert {row["name"] for row in receipt["verified_files"]} == consumed | metadata
        assert receipt["payload_bytes_hashed"] == sum(
            (source / name).stat().st_size for name in consumed)
        if start:
            # The quantum left the boundary it started from in place.
            before = json.loads(chain.fragment_path(root, *pairs[pairs.index((start, stop)) - 1])
                                .read_text())["boundary"]
            assert before and all(Path(record["path"]).is_file() for record in before)
    hashed.clear()
    assert campaign.main([*chained, '--cache-dir', str(tmp_path / 'join-cache'),
                          '--capture-chain', 'join']) == 0
    assert hashed == []  # the join hashes no source file
    record = json.loads(chain.join_path(root).read_text())
    assert record["retired_boundary_entries"] == (len(pairs) - 1) * len(tokens)
    entries = list((Path(storage["directory"]) / prep["session"]["generation"] / "entries").iterdir())
    assert entries == []

    expected, published = _manifest(monolith), _manifest(root)
    assert published == expected
    for name, entry in expected["entries"].items():
        before = torch.load(monolith / entry['path'], weights_only=True)
        after = torch.load(root / published['entries'][name]['path'], weights_only=True)
        assert before.keys() == after.keys()
        for key, value in before.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value.view(torch.uint8), after[key].view(torch.uint8)), (name, key)
            else:
                assert value == after[key], (name, key)
    cache.require_capture_contract(root / 'capture_manifest.json')
