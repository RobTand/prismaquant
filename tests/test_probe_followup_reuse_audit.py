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
    assert reused
    result = pickle.loads(output.read_bytes())
    assert result["router_counts"] == data["router_counts"]
    assert result["router_totals"] == data["router_totals"]


def test_regrouped_route_maps_merge_without_loss_or_duplication(tmp_path):
    originals = []
    for layer in (0, 1):
        inputs = tmp_path / f"input{layer}"
        inputs.mkdir()
        path, expected, data = _shard(inputs, packed=True, routed=True, version=3)
        data["meta"].update(expected)
        name = f"model.layers.{layer}.mlp.experts.gate_up_proj"
        row = next(iter(data["stats"].values()))
        row["_packed_experts_module"] = name.rsplit(".", 1)[0]
        data["stats"] = {name: row}
        router = f"model.layers.{layer}.mlp.gate"
        data.update(router_counts={router: {"0": 3.0}}, router_totals={router: 8},
                    router_active_counts={router: {"0": 5}})
        data["expert_route_stats"] = probe._route_stats_from_counts(
            data["router_counts"], data["router_totals"], data["router_active_counts"])
        target = tmp_path / f"probe_shard_{layer:03d}.pkl"
        target.write_bytes(pickle.dumps(data))
        originals.append(target)
    cached = probe.scan_cached_linear_stats(tmp_path, expected)
    combined = tmp_path / "combined.pkl"
    assert probe.synthesize_shard_from_linear_cache(r"model\.layers\.", "", cached, expected, combined)
    regrouped = pickle.loads(combined.read_bytes())
    original = probe.merge_probe_pickles(originals, tmp_path / "old.pkl", write_output=False)
    for field in probe._ROUTING_FIELDS:
        assert regrouped[field] == original[field]


def test_conflicting_duplicate_router_refuses_pooled_reuse(tmp_path):
    path, expected, data = _shard(tmp_path, packed=True, routed=True, version=3)
    data["meta"].update(expected)
    path.write_bytes(pickle.dumps(data))
    data["router_totals"]["model.layers.0.mlp.gate"] += 1
    (tmp_path / "probe_shard_001.pkl").write_bytes(pickle.dumps(data))
    assert not probe.scan_cached_linear_stats(tmp_path, expected)


def test_old_synthesized_packed_shard_with_empty_maps_refuses(tmp_path):
    path, expected, data = _shard(tmp_path, packed=True, version=3)
    data["meta"].update(expected, synthesized_from_cache=True)
    path.write_bytes(pickle.dumps(data))
    assert not probe.probe_shard_is_reusable(path, expected)
    assert not probe.scan_cached_linear_stats(tmp_path, expected)
