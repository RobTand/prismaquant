"""The per-expert checkpoint bridge must compute the packed A-side, not a cousin.

`build_weight_resolver` maps a card unit to ONE checkpoint key, which fits a
packed `[E, M, N]` routed-expert parameter. GLM-5.3-Flash stores every expert as
its own 2-D Linear weight and splits the fused pair in two, so its 84 packed
units -- 97% of the parameters -- resolved to nothing and silently priced their
A-side at 0.0 on a lane whose own attested contract says NVFP4 is "Real A4 on
BOTH the dense and the packed-expert route".

`packed_act_dloss_per_expert` streams those per-expert tensors instead of
stacking 19 GiB of gate_up. The risk in reordering a reduction is that it stops
being the same reduction, so the load-bearing test here is the agreement one:
streamed must equal `_activation_dloss_packed` on the stacked equivalent.
"""
import numpy as np
import pytest
import torch

from prismaquant.aqua_activation_cost import (
    packed_act_dloss_per_expert, per_expert_weight_keys,
)
from prismaquant.format_cost_protocol import _activation_dloss_packed
from prismaquant.sensitivity_card import SensitivityUnit, UnitTopology

E, M, N = 4, 6, 8          # experts, out_features (gate+up), in_features


def _unit(name="model.layers.0.mlp.experts.gate_up_proj"):
    rng = np.random.default_rng(0)
    return SensitivityUnit(
        topology=UnitTopology(name=name, layer_index=0, role="gate_up"),
        out_features=M, in_features=N, n_params=E * M * N, n_tokens=512,
        h_trace_raw=1.0, h_w2_sum_raw=0.0, w_norm_sq=1.0, w_max_abs=1.0,
        expert_g_sq_sum=rng.random((E, M)).astype(np.float32) + 0.1,
        expert_act_sq_sum=rng.random((E, N)).astype(np.float32) + 0.1,
        expert_tokens=np.array([128, 64, 32, 0], dtype=np.int64),
    )


def test_streamed_equals_the_stacked_packed_reduction():
    unit = _unit()
    rng = np.random.default_rng(7)
    per_expert = [torch.tensor(rng.standard_normal((M, N)), dtype=torch.float32)
                  for _ in range(E)]
    stacked = torch.stack(per_expert).numpy()
    var = rng.random((E, N)) + 0.05

    keys = [[f"e{e}"] for e in range(E)]
    lookup = {f"e{e}": per_expert[e] for e in range(E)}
    streamed = packed_act_dloss_per_expert(
        unit, keys, "unused", var, handles=lookup.__getitem__)
    reference = _activation_dloss_packed(unit, stacked, var)
    assert streamed == pytest.approx(reference, rel=1e-9)


def test_fused_siblings_concatenate_along_the_output_axis():
    """gate then up -- the order `expert_g_sq_sum`'s rows are indexed in.

    Concatenating the other way silently pairs each expert's up-projection rows
    with the gradient statistics of its gate rows, which is a wrong number that
    still has the right shape.
    """
    unit = _unit()
    rng = np.random.default_rng(11)
    gate = [torch.tensor(rng.standard_normal((M // 2, N)), dtype=torch.float32)
            for _ in range(E)]
    up = [torch.tensor(rng.standard_normal((M // 2, N)), dtype=torch.float32)
          for _ in range(E)]
    var = rng.random((E, N)) + 0.05
    lookup = {}
    keys = []
    for e in range(E):
        lookup[f"g{e}"], lookup[f"u{e}"] = gate[e], up[e]
        keys.append([f"g{e}", f"u{e}"])
    streamed = packed_act_dloss_per_expert(
        unit, keys, "unused", var, handles=lookup.__getitem__)
    stacked = torch.stack([torch.cat([gate[e], up[e]], dim=0)
                           for e in range(E)]).numpy()
    assert streamed == pytest.approx(_activation_dloss_packed(unit, stacked, var),
                                     rel=1e-9)


def test_a_zero_token_expert_contributes_no_activation_cost():
    """Expert 3 saw no calibration tokens; its sigma row is zero.

    `expert_act_sigma` returns a zero row rather than a fitted distribution, and
    the streamed reduction must carry that through as 0.0 -- inventing an A-side
    for the least-evidenced expert is worse than reporting none.
    """
    unit = _unit()
    rng = np.random.default_rng(3)
    per_expert = [torch.tensor(rng.standard_normal((M, N)), dtype=torch.float32)
                  for _ in range(E)]
    var = rng.random((E, N)) + 0.05
    lookup = {f"e{e}": per_expert[e] for e in range(E)}
    keys = [[f"e{e}"] for e in range(E)]
    full = packed_act_dloss_per_expert(unit, keys, "u", var,
                                       handles=lookup.__getitem__)
    var_zeroed = var.copy()
    var_zeroed[3] = 0.0
    zeroed = packed_act_dloss_per_expert(unit, keys, "u", var_zeroed,
                                         handles=lookup.__getitem__)
    assert zeroed < full


def test_per_expert_keys_are_found_and_ordered():
    wm = {}
    for e in range(E):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            wm[f"model.layers.0.mlp.experts.{e}.{proj}.weight"] = "s.safetensors"
    keys = per_expert_weight_keys("model.layers.0.mlp.experts.gate_up_proj",
                                  wm, n_experts=E)
    assert keys == [[f"model.layers.0.mlp.experts.{e}.gate_proj.weight",
                     f"model.layers.0.mlp.experts.{e}.up_proj.weight"]
                    for e in range(E)]
    down = per_expert_weight_keys("model.layers.0.mlp.experts.down_proj",
                                  wm, n_experts=E)
    assert down == [[f"model.layers.0.mlp.experts.{e}.down_proj.weight"]
                    for e in range(E)]


def test_a_packed_layout_declines_rather_than_guessing():
    """No per-expert keys -> None, so the caller falls back to the single-key
    resolver instead of taking a special case that would price nothing."""
    assert per_expert_weight_keys(
        "model.layers.0.mlp.experts.gate_up_proj",
        {"model.layers.0.mlp.experts.gate_up_proj": "s.safetensors"},
        n_experts=E) is None
    assert per_expert_weight_keys("model.layers.0.mlp.down_proj", {},
                                  n_experts=E) is None


# ---------------------------------------------------------------------------
# The PRODUCTION entry, end to end
# ---------------------------------------------------------------------------
# The helper-level agreement above is necessary and not sufficient. PQ-R2 was
# that these helpers existed, were tested, and were never CALLED: the resolver
# returned one key per card unit, so a GLM-style checkpoint -- one 2-D tensor
# per routed expert, fused gate/up split in two -- resolved NONE of its packed
# units, and they were dropped before pricing AND before the hole report. The
# failure mode is silent by construction: `cost_entry_act_dloss` defaults to
# 0.0, so a unit missing from `table` is a unit the DP reads as FREE.
#
# So the load-bearing regression is the one that goes through
# `activation_dloss_table`. Everything below builds a REAL checkpoint on disk
# (index + safetensors shard) and prices it the way the stage does.

#: Per-expert output rows of EACH sibling. The card's `out_features` is the
#: CONCATENATED gate+up size -- the axis `_activation_dloss_packed` indexes.
_ROWS = 8
_IN = 16


def _dense_unit(name="model.layers.0.mlp.down_proj"):
    rng = np.random.default_rng(3)
    return SensitivityUnit(
        topology=UnitTopology(name=name, layer_index=0, role="down"),
        out_features=_ROWS, in_features=_IN, n_params=_ROWS * _IN, n_tokens=256,
        h_trace_raw=1.0, h_w2_sum_raw=1e-3, w_norm_sq=1.0, w_max_abs=0.1,
        g_sq_sum=np.full(_ROWS, 1e-3, dtype=np.float64),
        act_sq_sum=rng.uniform(0.5, 1.5, _IN).astype(np.float64),
        act_absmax=rng.uniform(2.0, 6.0, _IN).astype(np.float64),
    )


def _packed_unit(name="model.layers.0.mlp.experts.gate_up_proj", *,
                 experts=E):
    rng = np.random.default_rng(11)
    return SensitivityUnit(
        topology=UnitTopology(name=name, layer_index=0, role="gate_up"),
        out_features=2 * _ROWS,          # gate rows then up rows, per expert
        in_features=_IN, n_params=experts * 2 * _ROWS * _IN,
        n_tokens=512, h_trace_raw=1.0, h_w2_sum_raw=0.0, w_norm_sq=1.0,
        w_max_abs=1.0,
        expert_g_sq_sum=rng.random((experts, 2 * _ROWS)).astype(np.float32) + 0.1,
        expert_act_sq_sum=rng.random((experts, _IN)).astype(np.float32) + 0.1,
        expert_tokens=np.full(experts, 64, dtype=np.int64),
    )


def _per_expert_checkpoint(tmp_path, *, drop=None):
    """A real checkpoint: one dense key + one per-expert packed unit.

    ``drop=(expert, leaf)`` omits one sibling so an incomplete roster can be
    exercised without inventing a second layout.
    """
    from safetensors.torch import save_file
    import json

    rng = np.random.default_rng(5)
    tensors: dict = {}
    weight_map: dict = {}

    def _add(key, shape):
        tensors[key] = torch.tensor(rng.standard_normal(shape),
                                    dtype=torch.bfloat16)
        weight_map[key] = "shard-00001.safetensors"

    _add("model.layers.0.mlp.down_proj.weight", (_ROWS, _IN))
    for e in range(E):
        for leaf in ("gate_proj", "up_proj"):
            if drop is not None and (e, leaf) == drop:
                continue
            _add(f"model.layers.0.mlp.experts.{e}.{leaf}.weight", (_ROWS, _IN))

    save_file(tensors, str(tmp_path / "shard-00001.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}))
    return tmp_path, tensors


def _card(units):
    card_mod = pytest.importorskip("prismaquant.sensitivity_card")
    return card_mod.SensitivityCard(
        card_mod.CardProvenance(model_id="unit-test", calib_hash="c0ffee",
                                n_calib_samples=1, seq_len=8,
                                probe_commit="0" * 40),
        units)


def test_the_production_table_prices_a_per_expert_packed_unit(tmp_path):
    """PQ-R2's regression: the bridge must be REACHED from the stage itself."""
    from prismaquant.aqua_activation_cost import activation_dloss_table
    from prismaquant.format_cost_protocol import _activation_dloss_packed
    from prismaquant.format_cost_registry import RegistryFormatPlugin

    model, tensors = _per_expert_checkpoint(tmp_path)
    packed_name = "model.layers.0.mlp.experts.gate_up_proj"
    dense_name = "model.layers.0.mlp.down_proj"
    packed, dense = _packed_unit(packed_name), _dense_unit(dense_name)

    table, holes, meta = activation_dloss_table(
        _card([dense, packed]), str(model), ["NVFP4"], device="cpu",
        names=[dense_name, packed_name], executed_activation_formats="all")

    assert dense_name in table, "the dense unit must still price"
    assert packed_name in table, (
        "the per-expert packed unit was dropped before pricing: it has no "
        "A-side, and `cost_entry_act_dloss` reads that absence as 0.0 (free) "
        "on a lane whose packed-expert route runs 4-bit activations")
    assert not holes.get("NVFP4"), holes
    assert meta["per_expert_units_priced"] == 1
    assert meta["units_unresolved"] == 0

    # And the number is the PRODUCTION quantity, not a cousin: same reduction,
    # same float64 accumulation, same variance authority as the stacked path.
    # The variance is taken from the plugin directly -- not from the stage's own
    # resolver -- so this regression depends on nothing the fix introduced and
    # fails on BEHAVIOUR before it, rather than at import.
    plugin = RegistryFormatPlugin.build(
        "NVFP4", shape=(packed.out_features, packed.in_features), device="cpu")
    var = plugin.expert_activation_error_variance(packed)
    assert var is not None
    stacked = torch.stack(
        [torch.cat([tensors[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"],
                    tensors[f"model.layers.0.mlp.experts.{e}.up_proj.weight"]],
                   dim=0).to(torch.float32) for e in range(E)]).numpy()
    assert table[packed_name]["NVFP4"] == pytest.approx(
        _activation_dloss_packed(packed, stacked, var), rel=1e-9)


def test_an_incomplete_expert_roster_is_a_hole_never_a_price(tmp_path):
    """A missing sibling must not silently redefine the group.

    `per_expert_weight_keys` returns None unless EVERY expert's keys exist, and
    the fallback is "unresolved" -- which the stage now REPORTS per executed
    format. Pricing the experts that happen to be present would put a smaller,
    wrong E into a quantity normalized by the global token count.
    """
    from prismaquant.aqua_activation_cost import activation_dloss_table

    model, _ = _per_expert_checkpoint(tmp_path, drop=(E - 1, "up_proj"))
    packed_name = "model.layers.0.mlp.experts.gate_up_proj"
    dense_name = "model.layers.0.mlp.down_proj"
    packed, dense = _packed_unit(packed_name), _dense_unit(dense_name)

    table, holes, meta = activation_dloss_table(
        _card([dense, packed]), str(model), ["NVFP4"], device="cpu",
        names=[dense_name, packed_name], executed_activation_formats="all")

    assert dense_name in table
    assert packed_name not in table, "an incomplete roster cannot be priced"
    assert packed_name in holes["NVFP4"], (
        "the unpriced unit must be NAMED: silence here is the defect")
    assert meta["units_unresolved"] == 1


def test_the_shard_handle_cache_releases_an_evicted_shard(tmp_path, monkeypatch):
    """The bridge must keep the dense loop's resident-mmap promise.

    The dense path reads one shard at a time by construction; the per-expert
    reduction is expert-major, so it reads keys out of shard order and has to
    CLOSE a shard to keep resident mapped pages bounded (holding every handle
    cost 48.2 GiB on this model, measured). The bound is therefore the mechanism
    under test: over capacity, the least recently used shard is released, and
    `close()` releases the rest.
    """
    from safetensors.torch import save_file
    import json

    from prismaquant import aqua_activation_cost as aqc

    save_file({"a.weight": torch.zeros(2, 2, dtype=torch.bfloat16)},
              str(tmp_path / "shard-a.safetensors"))
    save_file({"b.weight": torch.zeros(2, 2, dtype=torch.bfloat16)},
              str(tmp_path / "shard-b.safetensors"))
    weight_map = {"a.weight": "shard-a.safetensors",
                  "b.weight": "shard-b.safetensors"}

    released = []
    real = aqc._close_safetensor_handle

    def _spy(handle):
        released.append(handle)
        return real(handle)

    monkeypatch.setattr(aqc, "_close_safetensor_handle", _spy)
    cache = aqc._CheckpointShardHandles(str(tmp_path), weight_map, None,
                                        capacity=1)
    try:
        cache("a.weight")
        assert released == [], "one handle is within capacity"
        cache("b.weight")
        assert len(released) == 1, (
            "the second shard must evict the first rather than hold both open")
    finally:
        cache.close()
    assert len(released) == 2, "close() must release every remaining shard"
    assert cache._open == {}


def test_a_joint_row_is_never_stamped_with_the_aqua_a_side(monkeypatch):
    """The trace root asked for: joint rows are exempt, not double-counted.

    A joint row's one signed residual already carries the weight, activation and
    mixed terms under a single downstream Fisher, and
    `validate_joint_aura_entry` REFUSES a row carrying `act_dloss` because that
    would apply the activation term twice. `cost_entry_predicted_dloss` returns
    before the A-side branch for such a row; this pins that the merge stage does
    not stamp one on behind it, which would not double-count but INVALIDATE the
    row and stop the allocation.
    """
    from prismaquant import aqua_activation_cost as aqc
    from prismaquant.allocator_candidates import ACT_DLOSS_KEY

    joint: dict = {}                      # sentinel: the predicate is the subject
    costs = {"a": {"NVFP4": joint}}
    monkeypatch.setattr(aqc, "cost_entry_is_joint_aura", lambda e: e is joint,
                        raising=False)
    report = aqc.merge_act_dloss(costs, {"a": {"NVFP4": 0.1}})

    assert ACT_DLOSS_KEY not in joint, (
        "a joint row already carries the activation term and REFUSES a second "
        "one; stamping `act_dloss` on it invalidates the row")
    assert report["entries_merged"] == 0
    assert report["joint_rows_skipped"] == 1
