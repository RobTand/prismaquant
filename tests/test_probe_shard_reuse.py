"""Tests for LPS-invariant probe-shard reuse.

Fisher stats are per-Linear numbers. A shard at LAYERS_PER_SHARD=5 contains
stats for 5 layers' Linears; a shard at LAYERS_PER_SHARD=3 contains stats
for 3 layers' Linears. The *numbers* for overlapping Linears are identical
across the two runs — only the grouping changes. So restart across LPS
values should reuse cached Linear stats, not recompute from scratch.

These tests pin the reuse machinery:

  - scan_cached_linear_stats pools Linears across shards and rejects
    shards whose content-level meta (nsamples/seqlen/dtype/etc) differs
    from the anchor.
  - synthesize_shard_from_linear_cache produces a pickle shaped like a
    freshly-computed shard: same keys, merge_probe_pickles-compatible.
  - Partial cache coverage: filter returns the matching subset; caller
    decides to run fresh compute for the gap.
  - End-to-end: writing 2 shards at lps=5 (L0-L4, L5-L9), then
    synthesizing 5 shards at lps=2 (L0-L1, L2-L3, L4-L5, L6-L7, L8-L9),
    then merging → same final stats dict as the lps=5 merge.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from prismaquant.incremental_probe import (
    build_extended_shard_regexes,
    build_shard_schedule,
    merge_probe_pickles,
    scan_cached_linear_stats,
    synthesize_shard_from_linear_cache,
)


def _write_shard(
    path: Path,
    stats: dict,
    *,
    nsamples: int = 32,
    seqlen: int = 1024,
    dtype: str = "bf16",
    model: str = "/tmp/fake/model",
    linear_include: str = "re:.*",
    linear_exclude: str = r"(?:mlp\.gate$|mlp\..*gate$|\.router(?:$|\.)|block_sparse_moe\.gate$)",
    activation_cache_dir: str = "/tmp/act",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "stats": stats,
            "router_counts": {},
            "router_totals": {},
            "expert_info": {},
            "meta": {
                "model": model,
                "dataset": "ultrachat_200k",
                "nsamples": nsamples,
                "seqlen": seqlen,
                "dtype": dtype,
                "requested_device": "cuda",
                "requested_device_map": "None",
                "importance_weighting": True,
                "activation_cache_dir": activation_cache_dir,
                "linear_include": linear_include,
                "linear_exclude": linear_exclude,
                "h_detail_dir": None,
            },
        }, f)


def _anchor(**overrides):
    a = {
        "model": "/tmp/fake/model",
        "dataset": "ultrachat_200k",
        "nsamples": 32,
        "seqlen": 1024,
        "dtype": "bf16",
        "requested_device": "cuda",
        "requested_device_map": "None",
        "importance_weighting": True,
        "activation_cache_dir": "/tmp/act",
        "linear_exclude": r"(?:mlp\.gate$|mlp\..*gate$|\.router(?:$|\.)|block_sparse_moe\.gate$)",
        "h_detail_dir": None,
    }
    a.update(overrides)
    return a


def _stats_for(layer_ids: list[int]) -> dict:
    """Fake per-Linear stats for a list of layer indices. Every layer
    has three Linears (q_proj, k_proj, mlp.down_proj); the dict value
    carries the stat so we can spot-check merges."""
    out = {}
    for L in layer_ids:
        for proj in ("self_attn.q_proj", "self_attn.k_proj", "mlp.down_proj"):
            name = f"model.layers.{L}.{proj}"
            out[name] = {"h_trace": float(L) + 0.1, "n_tokens_seen": 100}
    return out


def test_extended_shards_skip_declared_mtp_when_index_has_no_mtp(tmp_path, capsys):
    (tmp_path / "config.json").write_text(json.dumps({
        "num_hidden_layers": 2,
        "num_nextn_predict_layers": 1,
    }))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00001.safetensors",
        }
    }))

    regexes = build_extended_shard_regexes(
        str(tmp_path),
        1,
        include_visual=False,
        include_lm_head=False,
    )

    assert regexes == [r"model\.layers\.0\.", r"model\.layers\.1\."]
    assert not any("mtp" in regex for regex in regexes)
    assert "skipping MTP shards" in capsys.readouterr().out


def test_extended_shards_include_mtp_when_index_has_mtp_but_config_omits_count(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "num_hidden_layers": 2,
    }))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "model.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
            "model.layers.1.self_attn.q_proj.weight": "model-00001.safetensors",
            "mtp.fc.weight": "model-00001.safetensors",
            "mtp.layers.0.self_attn.q_proj.weight": "model-00001.safetensors",
        }
    }))

    regexes = build_extended_shard_regexes(
        str(tmp_path),
        1,
        include_body=False,
        include_visual=False,
        include_lm_head=False,
    )

    assert regexes == [r"(?:mtp\.fc|mtp\.layers\.0\.)"]


def test_extended_shards_use_profile_visual_prefix(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "gemma4",
        "num_hidden_layers": 2,
        "vision_config": {"num_hidden_layers": 3},
    }))

    regexes = build_extended_shard_regexes(
        str(tmp_path),
        2,
        include_body=False,
        include_mtp=False,
        include_lm_head=False,
    )

    assert regexes == [
        r"model\.vision_tower\.vision_model\.encoder\.layers\.(?:0|1|2)\.",
        r"^(?!model\.vision_tower\.vision_model\.encoder\.layers\.\d+\.)"
        r"(?:model\.vision_tower|model\.embed_vision)\.",
    ]


def test_shard_schedule_uses_profile_body_and_head_names(tmp_path, capsys):
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "deepseek_v4",
        "num_hidden_layers": 2,
        "num_nextn_predict_layers": 1,
    }))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {
            "layers.0.attn.wkv.weight": "model-00001.safetensors",
            "layers.1.attn.wkv.weight": "model-00001.safetensors",
            "mtp.0.attn.wkv.weight": "model-00001.safetensors",
        }
    }))

    schedule = build_shard_schedule(
        model_path=str(tmp_path),
        num_body_layers=2,
        body_layers_per_shard=1,
        body_layer_range=(0, 2),
        include_mtp=True,
        include_visual=False,
        include_lm_head=True,
        unified_body_sweep=False,
    )

    assert schedule.regexes() == [
        r"layers\.0\.",
        r"layers\.1\.",
        r"mtp\.0\.",
        r"^head$",
    ]
    assert schedule.body_layer_indices() == frozenset({0, 1})
    assert "skipping MTP" not in capsys.readouterr().out


# ---------------------------------------------------------------------------
# scan_cached_linear_stats
# ---------------------------------------------------------------------------

def test_scan_pools_compatible_shards(tmp_path):
    """Two compatible shards at lps=5 (L0-L4, L5-L9) should pool into a
    single Linear-keyed dict."""
    _write_shard(tmp_path / "probe_shard_000.pkl",
                 _stats_for([0, 1, 2, 3, 4]),
                 linear_include=r"re:^model\.layers\.(?:0|1|2|3|4)\.")
    _write_shard(tmp_path / "probe_shard_001.pkl",
                 _stats_for([5, 6, 7, 8, 9]),
                 linear_include=r"re:^model\.layers\.(?:5|6|7|8|9)\.")
    pool = scan_cached_linear_stats(tmp_path, _anchor())
    assert len(pool) == 10 * 3, f"expected 30 Linears, got {len(pool)}"
    assert "model.layers.0.self_attn.q_proj" in pool
    assert "model.layers.9.mlp.down_proj" in pool


def test_scan_rejects_incompatible_meta(tmp_path):
    """A shard with different nsamples must NOT be pooled — its Fisher
    values were computed against a different calibration set."""
    _write_shard(tmp_path / "probe_shard_000.pkl",
                 _stats_for([0, 1]),
                 nsamples=32)
    _write_shard(tmp_path / "probe_shard_001.pkl",
                 _stats_for([2, 3]),
                 nsamples=64)  # different!
    pool = scan_cached_linear_stats(tmp_path, _anchor(nsamples=32))
    assert len(pool) == 6, f"expected only the nsamples=32 shard pooled, got {len(pool)}"
    assert "model.layers.0.self_attn.q_proj" in pool
    assert "model.layers.2.self_attn.q_proj" not in pool


def test_scan_empty_dir(tmp_path):
    assert scan_cached_linear_stats(tmp_path, _anchor()) == {}


def test_scan_skips_corrupt_pickle(tmp_path):
    """A truncated pickle must not crash the scan — the other shard still pools."""
    (tmp_path / "probe_shard_000.pkl").write_bytes(b"not a pickle")
    _write_shard(tmp_path / "probe_shard_001.pkl",
                 _stats_for([5, 6]))
    pool = scan_cached_linear_stats(tmp_path, _anchor())
    assert len(pool) == 6
    assert "model.layers.5.self_attn.q_proj" in pool


def test_scan_ignores_non_shard_pickles(tmp_path):
    """Files that don't match `probe_shard_*.pkl` must be ignored."""
    (tmp_path / "precomputed.pt").write_bytes(b"torch save")
    (tmp_path / "notes.txt").write_text("hi")
    _write_shard(tmp_path / "probe_shard_000.pkl",
                 _stats_for([0, 1]))
    pool = scan_cached_linear_stats(tmp_path, _anchor())
    assert len(pool) == 6


# ---------------------------------------------------------------------------
# synthesize_shard_from_linear_cache
# ---------------------------------------------------------------------------

def test_synthesize_filters_by_regex(tmp_path):
    """Given a pool covering L0-L9, synthesizing a shard at lps=3 for
    L0-L2 should select exactly those Linears."""
    pool = _stats_for(list(range(10)))
    out = tmp_path / "probe_shard_000.pkl"
    expected_meta = {
        "model": "/tmp/fake/model",
        "nsamples": 32, "seqlen": 1024, "dtype": "bf16",
        "linear_include": r"re:^model\.layers\.(?:0|1|2)\.",
        "linear_exclude": r"(?:mlp\.gate$|mlp\..*gate$)",
    }
    ok = synthesize_shard_from_linear_cache(
        linear_include=r"re:^model\.layers\.(?:0|1|2)\.",
        linear_exclude=r"(?:mlp\.gate$|mlp\..*gate$)",
        cache=pool,
        expected_meta=expected_meta,
        output_path=out,
    )
    assert ok
    with open(out, "rb") as f:
        data = pickle.load(f)
    stats = data["stats"]
    assert len(stats) == 3 * 3, f"expected L0-L2 × 3 Linears = 9, got {len(stats)}"
    for L in (0, 1, 2):
        assert f"model.layers.{L}.self_attn.q_proj" in stats
    assert "model.layers.3.self_attn.q_proj" not in stats
    assert data["meta"]["synthesized_from_cache"] is True


def test_synthesize_exclude_drops_matches(tmp_path):
    """Linear-exclude regex must suppress matches the include would
    otherwise keep. mlp.gate entries must never make it into a shard."""
    pool = {
        "model.layers.0.mlp.gate": {"h_trace": 1.0},    # routing gate — excluded
        "model.layers.0.self_attn.q_proj": {"h_trace": 2.0},
        "model.layers.0.mlp.gate_up_proj": {"h_trace": 3.0},  # NOT a gate
    }
    out = tmp_path / "probe_shard_000.pkl"
    ok = synthesize_shard_from_linear_cache(
        linear_include=r"re:^model\.layers\.0\.",
        linear_exclude=r"(?:mlp\.gate$)",
        cache=pool,
        expected_meta={"linear_include": "re:^model\\.layers\\.0\\."},
        output_path=out,
    )
    assert ok
    data = pickle.loads(out.read_bytes())
    assert "model.layers.0.mlp.gate" not in data["stats"]
    assert "model.layers.0.self_attn.q_proj" in data["stats"]
    assert "model.layers.0.mlp.gate_up_proj" in data["stats"]


def test_synthesize_returns_false_when_nothing_matches(tmp_path):
    """No regex match → caller falls through to compute."""
    pool = _stats_for([0, 1])
    out = tmp_path / "probe_shard_000.pkl"
    ok = synthesize_shard_from_linear_cache(
        linear_include=r"re:^model\.layers\.(?:50|51)\.",   # nothing in pool
        linear_exclude="",
        cache=pool,
        expected_meta={},
        output_path=out,
    )
    assert not ok
    assert not out.exists(), "no pickle should be written on empty match"


def test_synthesize_handles_bare_regex_without_re_prefix(tmp_path):
    """Callers sometimes pass regexes without the compressed-tensors
    `re:` prefix. Both forms must compile identically."""
    pool = _stats_for([0])
    out = tmp_path / "probe_shard_000.pkl"
    ok = synthesize_shard_from_linear_cache(
        linear_include=r"^model\.layers\.0\.",   # no re:
        linear_exclude="",
        cache=pool,
        expected_meta={},
        output_path=out,
    )
    assert ok
    data = pickle.loads(out.read_bytes())
    assert len(data["stats"]) == 3


# ---------------------------------------------------------------------------
# End-to-end: LPS change via synthesize + merge produces identical stats
# ---------------------------------------------------------------------------

def test_lps_change_produces_identical_merged_stats(tmp_path):
    """Write 2 shards at lps=5 covering L0-L9. Then synthesize the
    equivalent 5 shards at lps=2 (L0-L1, L2-L3, L4-L5, L6-L7, L8-L9)
    from the pool and merge them. Merged stats dicts must match
    element-for-element."""
    # Phase A: the 'original' lps=5 run
    lps5_dir = tmp_path / "lps5"
    _write_shard(lps5_dir / "probe_shard_000.pkl",
                 _stats_for([0, 1, 2, 3, 4]),
                 linear_include=r"re:^model\.layers\.(?:0|1|2|3|4)\.")
    _write_shard(lps5_dir / "probe_shard_001.pkl",
                 _stats_for([5, 6, 7, 8, 9]),
                 linear_include=r"re:^model\.layers\.(?:5|6|7|8|9)\.")

    # Merge lps=5 to get the reference stats dict
    lps5_merged_path = tmp_path / "merged_lps5.pkl"
    merge_probe_pickles(
        sorted(lps5_dir.glob("probe_shard_*.pkl")),
        lps5_merged_path,
    )
    ref_stats = pickle.loads(lps5_merged_path.read_bytes())["stats"]

    # Phase B: resume with lps=2, synthesizing each new shard from pool
    pool = scan_cached_linear_stats(lps5_dir, _anchor())
    lps2_dir = tmp_path / "lps2"
    shard_ranges = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
    for i, (lo, hi) in enumerate(shard_ranges):
        regex = f"re:^model\\.layers\\.(?:{lo}|{hi})\\."
        ok = synthesize_shard_from_linear_cache(
            linear_include=regex,
            linear_exclude=_anchor()["linear_exclude"],
            cache=pool,
            expected_meta={"linear_include": regex},
            output_path=lps2_dir / f"probe_shard_{i:03d}.pkl",
        )
        assert ok, f"shard {i} ({lo}-{hi}) should have synthesized"

    # Merge lps=2 and compare
    lps2_merged_path = tmp_path / "merged_lps2.pkl"
    merge_probe_pickles(
        sorted(lps2_dir.glob("probe_shard_*.pkl")),
        lps2_merged_path,
    )
    new_stats = pickle.loads(lps2_merged_path.read_bytes())["stats"]
    assert set(new_stats) == set(ref_stats), \
        "Linear coverage differs between LPS values"
    for name in ref_stats:
        assert new_stats[name] == ref_stats[name], \
            f"{name}: stats differ after LPS round-trip"


def test_merge_rejects_overlap_when_synthesis_double_covers(tmp_path):
    """Safety: if two synthesized shards somehow both claim the same
    Linear (bad regex set), merge_probe_pickles must still raise —
    idempotency requires disjoint shards in the final merge."""
    pool = _stats_for([0, 1])
    out_dir = tmp_path / "shards"
    out_dir.mkdir()
    synthesize_shard_from_linear_cache(
        linear_include=r"re:^model\.layers\.(?:0|1)\.",
        linear_exclude="",
        cache=pool,
        expected_meta={},
        output_path=out_dir / "probe_shard_000.pkl",
    )
    # Second shard overlaps L0 — should not happen in real pipeline but
    # pins the merger's overlap detection.
    synthesize_shard_from_linear_cache(
        linear_include=r"re:^model\.layers\.0\.",
        linear_exclude="",
        cache=pool,
        expected_meta={},
        output_path=out_dir / "probe_shard_001.pkl",
    )
    with pytest.raises(ValueError, match="overlap"):
        merge_probe_pickles(
            sorted(out_dir.glob("probe_shard_*.pkl")),
            tmp_path / "merged.pkl",
        )
