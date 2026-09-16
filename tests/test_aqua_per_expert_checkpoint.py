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


def _per_expert_checkpoint(tmp_path, *, drop=None, extra_dense=()):
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
    for extra in extra_dense:
        _add(f"{extra}.weight", (_ROWS, _IN))
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


# ---------------------------------------------------------------------------
# The CLI contract around cells that are ALREADY joint-priced
# ---------------------------------------------------------------------------
# Root QA on the first version of this fix: `main` still refused when
# `entries_merged == 0`, so a VALID artifact whose every owed cell is a joint
# AURA row was rejected as a no-op -- and the stage still priced the A-side for
# those cells before the merge threw that work away. Both are regressions of the
# "eliminate unnecessary work" rule, and they are what these tests pin.
#
# The joint rows here are built by `make_joint_aura_entry`, which VALIDATES its
# own row, so this exercises the real predicate rather than a monkeypatched one.

def _joint_row(name, fmt, *, loss=1.0, shape=(_ROWS, _IN)):
    """A real, validating joint AURA row for one (unit, format) cell."""
    import math

    from prismaquant.cost_stage_checkpoint import canonical_json_sha256
    from prismaquant.cost_streaming import STREAMED_MODEL_IDENTITY_SCHEMA
    from prismaquant.joint_aura import (
        arithmetic_identity, identity_sha256, make_joint_aura_entry,
    )

    content = {"config": {"fixture": True},
               "weight_map": {"fixture.weight": "fixture.weight"},
               "shards": [{"path": "/fixture/synthetic.safetensors", "size": 1,
                           "sha256": "a" * 64}]}
    source_model = {"schema": STREAMED_MODEL_IDENTITY_SCHEMA,
                    "source": "synthetic", "resolved_commit": None,
                    "content_sha256": canonical_json_sha256(
                        content, where="pqr2 joint fixture"),
                    **content}
    arithmetic = arithmetic_identity(torch.float32)
    bytes_ = shape[0] * shape[1] * 4
    probe = {"schema": "prismaquant.joint_aura.probes.v2", "seed_base": 0,
             "n_probes": 3, "calibration_sha256": "c" * 64,
             "producer_source_sha256": "d" * 64, "source_model": source_model,
             "distribution": "rademacher", "normalization": "global_kl_fisher",
             "temperature": 1.0, "arithmetic": arithmetic}
    operator = {"schema": "prismaquant.joint_aura.operator.v2", "qname": name,
                "format": fmt, "probe_identity_sha256": identity_sha256(probe),
                "source_weight": {"content_sha256": "a" * 64,
                                  "shape": list(shape),
                                  "dtype": "torch.float32",
                                  "logical_bytes": bytes_},
                "rendered_weight": {"content_sha256": "b" * 64,
                                    "shape": list(shape),
                                    "dtype": "torch.float32",
                                    "logical_bytes": bytes_},
                "activation": {"schema": "prismaquant.joint_aura.activation.v1",
                               "quantizes_input": True,
                               "activation_max_abs": None,
                               "input_global_scale": None},
                "arithmetic": arithmetic}
    total = math.sqrt(2.0 * loss)
    return make_joint_aura_entry(
        operator_identity=operator, probe_identity=probe,
        signed_components=[{"weight": total, "activation": 0.0, "mixed": 0.0,
                            "total": total} for _ in range(3)])


def _cli_fixture(tmp_path, costs, units, *, extra_dense=(), extra_argv=()):
    """A card .npz, a --cost-in pkl and the argv `main` needs, on CPU."""
    import pickle

    model, _ = _per_expert_checkpoint(tmp_path, extra_dense=extra_dense)
    card_path = tmp_path / "card.npz"
    _card(units).to_npz(str(card_path))
    cost_in = tmp_path / "cost-in.pkl"
    cost_in.write_bytes(pickle.dumps({"costs": costs, "provenance": {}}))
    cost_out = tmp_path / "cost-out.pkl"
    argv = ["aqua-activation-cost", "--card", str(card_path),
            "--model-path", str(model), "--cost-in", str(cost_in),
            "--cost-out", str(cost_out), "--device", "cpu",
            "--lane-executes-all-activation-grids", *extra_argv]
    return cost_out, argv


def test_the_cli_accepts_an_all_joint_artifact_without_pricing_anything(
        tmp_path, monkeypatch):
    """A fulfilled artifact is accepted, and it costs no weight read.

    Refusing it as a "no-op" was wrong: nothing was computed because nothing was
    OWED. The `materialize_source_weight` trap is the proof that no A-side was
    computed for it -- not a timing claim, and not a claim that the checkpoint is
    untouched: `activation_dloss_table` still reads the small
    `model.safetensors.index.json` metadata index (and the scale map built from
    it) to resolve names. What is avoided is opening a shard, building a plugin
    and reading a weight.
    """
    import pickle
    import sys

    from prismaquant import aqua_activation_cost as aqc

    name = "model.layers.0.mlp.down_proj"
    cost_out, argv = _cli_fixture(
        tmp_path, {name: {"NVFP4": _joint_row(name, "NVFP4")}},
        [_dense_unit(name)])

    def _must_not_read(*_a, **_k):
        raise AssertionError("an all-joint artifact must not read a weight")

    monkeypatch.setattr(aqc, "materialize_source_weight", _must_not_read)
    monkeypatch.setattr(sys, "argv", argv)
    assert aqc.main() == 0

    out = pickle.loads(cost_out.read_bytes())
    assert "act_dloss" not in out["costs"][name]["NVFP4"]
    prov = out["provenance"]["aqua_activation_cost"]
    assert prov["joint_cells_already_priced"] == [[name, "NVFP4"]]
    assert prov["units_fully_joint_priced"] == 1
    assert prov["requested_cells"] == 1
    assert prov["priced_cells"] == 0
    assert prov["cells_without_act_price"] == 0
    assert prov["merge_report"]["joint_rows_skipped"] == 1
    assert prov["merge_report"]["entries_merged"] == 0


def test_a_requested_format_the_artifact_lacks_is_not_a_cell(tmp_path,
                                                            monkeypatch):
    """QA on dc371: the requested set is PER UNIT, not the global list.

    `--formats NVFP4,FP8_E4M3` on an artifact whose only cell for this unit is
    the joint NVFP4 row must not manufacture an FP8 cell for it. The first
    version formed `formats - joint` globally, so it opened the shard, built a
    plugin and computed an A-side for a cell the artifact cannot carry -- and
    then reported the unit as needing work. The trap proves no weight is read.
    """
    import sys

    from prismaquant import aqua_activation_cost as aqc

    name = "model.layers.0.mlp.down_proj"
    cost_out, argv = _cli_fixture(
        tmp_path, {name: {"NVFP4": _joint_row(name, "NVFP4")}},
        [_dense_unit(name)], extra_argv=("--formats", "NVFP4,FP8_E4M3"))

    def _must_not_read(*_a, **_k):
        raise AssertionError(
            "FP8_E4M3 is not a cell of this unit in this artifact")

    monkeypatch.setattr(aqc, "materialize_source_weight", _must_not_read)
    monkeypatch.setattr(sys, "argv", argv)
    assert aqc.main() == 0
    assert cost_out.exists()


def test_an_unrelated_joint_cell_does_not_licence_a_zero_priced_run(
        tmp_path, monkeypatch):
    """QA on dc371: acceptance is per REQUESTED cell, not per artifact.

    Unit A's NVFP4 row is joint; unit B's is an ordinary legacy row whose card
    carries no `g_sq_sum`, so this run prices nothing. B is a requested cell
    with no A-side, and A's joint rung says nothing about it -- the run must
    still refuse rather than write a copy wearing the AQUA name.
    """
    import dataclasses
    import sys

    from prismaquant import aqua_activation_cost as aqc

    joint_name = "model.layers.0.mlp.down_proj"
    legacy_name = "model.layers.0.mlp.o_proj"
    cost_out, argv = _cli_fixture(
        tmp_path,
        {joint_name: {"NVFP4": _joint_row(joint_name, "NVFP4")},
         legacy_name: {"NVFP4": {"predicted_dloss": 1.0}}},
        [_dense_unit(joint_name),
         dataclasses.replace(_dense_unit(legacy_name), g_sq_sum=None)],
        extra_dense=(legacy_name,))

    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="requested"):
        aqc.main()
    assert not cost_out.exists()


def test_a_mixed_artifact_prices_exactly_its_legacy_cells(tmp_path,
                                                          monkeypatch):
    """Per-CELL selection, not per-unit: the joint rung is left alone."""
    import pickle
    import sys

    from prismaquant import aqua_activation_cost as aqc
    from prismaquant.allocator_candidates import ACT_DLOSS_KEY

    name = "model.layers.0.mlp.down_proj"
    joint = _joint_row(name, "NVFP4")
    cost_out, argv = _cli_fixture(
        tmp_path,
        {name: {"NVFP4": joint, "FP8_E4M3": {"predicted_dloss": 1.0}}},
        [_dense_unit(name)])

    monkeypatch.setattr(sys, "argv", argv)
    assert aqc.main() == 0

    out = pickle.loads(cost_out.read_bytes())["costs"][name]
    assert ACT_DLOSS_KEY not in out["NVFP4"], (
        "the joint rung already carries its activation term")
    assert out["FP8_E4M3"][ACT_DLOSS_KEY] > 0.0, (
        "the legacy cell still needs its A-side")
    assert out["NVFP4"]["predicted_dloss"] == joint["predicted_dloss"]


def test_a_malformed_joint_claim_refuses_before_any_pricing(tmp_path,
                                                            monkeypatch):
    """Joint evidence is VALIDATED, not trusted: a claim without a row raises."""
    import pickle
    import sys

    from prismaquant import aqua_activation_cost as aqc
    from prismaquant.joint_aura import JOINT_CURRENCY

    name = "model.layers.0.mlp.down_proj"
    cost_out, argv = _cli_fixture(
        tmp_path, {name: {"NVFP4": {"cost_currency": JOINT_CURRENCY}}},
        [_dense_unit(name)])

    def _must_not_read(*_a, **_k):
        raise AssertionError("refusal must precede pricing")

    monkeypatch.setattr(aqc, "materialize_source_weight", _must_not_read)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="joint AURA"):
        aqc.main()
    assert not cost_out.exists()


def test_the_silent_no_op_refusal_still_fires(tmp_path, monkeypatch):
    """The case the refusal exists for: nothing priced and no joint coverage.

    A card whose unit carries no `g_sq_sum` prices nothing, and this artifact
    has no joint row either -- so `--cost-out` would be a byte-equivalent copy
    of a weight-only table wearing the AQUA name.
    """
    import dataclasses
    import sys

    from prismaquant import aqua_activation_cost as aqc

    name = "model.layers.0.mlp.down_proj"
    unit = dataclasses.replace(_dense_unit(name), g_sq_sum=None)
    cost_out, argv = _cli_fixture(
        tmp_path, {name: {"NVFP4": {"predicted_dloss": 1.0}}}, [unit])

    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="byte-equivalent copy"):
        aqc.main()
    assert not cost_out.exists()
