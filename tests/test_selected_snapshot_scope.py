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
