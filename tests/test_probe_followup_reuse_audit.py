"""Probe reuse must preserve valid dense caches and routed populations."""
import pickle

import pytest
import torch

from prismaquant import incremental_probe as probe


def _shard(tmp_path, *, packed=False, routed=False, version=2):
    name = "model.layers.0.mlp.experts.gate_up_proj" if packed else "model.layers.0.self_attn.q_proj"
    row = {"h_trace": 1.0, "n_tokens_seen": 8}
    if packed:
        row.update(_packed_experts_module="model.layers.0.mlp.experts", num_experts=2)
    data = {"stats": {name: row}, "meta": {"model": "source", "router_coverage_version": version},
            "router_counts": {}, "router_totals": {}, "router_active_counts": {}, "expert_info": {}}
    if routed:
        data["router_counts"] = {"model.layers.0.mlp.gate": {"0": 3.0}}
        data["router_totals"] = {"model.layers.0.mlp.gate": 8}
    path = tmp_path / "probe_shard_000.pkl"
    path.write_bytes(pickle.dumps(data))
    expected = {"model": "source", "router_coverage_version": 3, "packed_input_capture": "module_once.v1"}
    return path, expected, data


def test_dense_v2_shard_remains_reusable(tmp_path):
    path, expected, _ = _shard(tmp_path)
    assert probe.probe_shard_is_reusable(path, expected)
    assert probe.scan_cached_linear_stats(tmp_path, expected)


@pytest.mark.parametrize("packed,routed", [(True, False), (False, True)])
def test_affected_v2_shard_still_refuses(tmp_path, packed, routed):
    path, expected, _ = _shard(tmp_path, packed=packed, routed=routed)
    assert not probe.probe_shard_is_reusable(path, expected)
    assert not probe.scan_cached_linear_stats(tmp_path, expected)


@pytest.mark.parametrize("routed", [False, True])
def test_valid_v2_precompute_keeps_saved_maps(tmp_path, routed):
    maps = {"model.layers.0.mlp.gate": {"0": 3.0}} if routed else {}
    pre = probe.GlobalPrecompute(
        activations_cpu=[torch.ones(1, 2, 4)], grad_at_tail=torch.ones(1, 2, 4),
        ids=torch.ones(1, 2, dtype=torch.int64), resident_stats={}, resident_h_full={},
        resident_g2_per_token={}, resident_act_snaps={}, resident_act_row_indices={},
        expert_info={}, router_counts=maps,
        router_totals={"model.layers.0.mlp.gate": 2} if routed else {},
        router_active_counts={}, expert_route_stats={})
    old = {"model": "source", "nsamples": 1, "seqlen": 2, "router_coverage_version": 2}
    path = tmp_path / "precomputed.pt"
    probe._save_precompute_cache(path, pre, old)
    resumed = probe._load_precompute_cache(path, {**old, "router_coverage_version": 3}, torch.device("cpu"))
    assert resumed is not None
    assert resumed.router_counts == maps


def test_routed_pool_cannot_publish_a_shard_with_empty_maps(tmp_path):
    path, expected, data = _shard(tmp_path, packed=True, routed=True, version=3)
    data["meta"].update(expected)
    path.write_bytes(pickle.dumps(data))
    cached = probe.scan_cached_linear_stats(tmp_path, expected)
    output = tmp_path / "regrouped.pkl"
    reused = probe.synthesize_shard_from_linear_cache(r"model\.layers\.0\.", "", cached, expected, output)
    if reused:
        result = pickle.loads(output.read_bytes())
        assert result["router_counts"] == data["router_counts"]
        assert result["router_totals"] == data["router_totals"]
    else:
        assert not output.exists()
