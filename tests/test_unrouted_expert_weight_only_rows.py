"""Weight-only cost rows for the declared never-routed expert class.

The allocator's CB coverage gate is load-bearing and stays a gate: it may only
stop refusing for a class that has been declared, counted and stamped.
"""
from __future__ import annotations

import json

import pytest

from prismaquant.measure_quant_cost import (
    UNROUTED_EXPERT_COST_SOURCE,
    _load_unrouted_expert_declaration,
)
from prismaquant.model_profiles.deepseek_v4 import DeepseekV4Profile

RULE = "unrouted_expert_neutral_prior:layer_routed_mean"


def _sidecar(tmp_path, names, rule=RULE):
    p = tmp_path / "cw.pkl.provenance.json"
    p.write_text(json.dumps({"names": list(names), "rule": rule,
                             "basis": "probe n_tokens_seen == 0"}))
    return p


def test_no_declaration_leaves_the_gate_exactly_as_it_was(monkeypatch) -> None:
    monkeypatch.delenv("PRISMAQUANT_UNROUTED_EXPERT_PROVENANCE", raising=False)
    assert _load_unrouted_expert_declaration() == frozenset()


def test_scope_is_exactly_the_declared_names(tmp_path, monkeypatch) -> None:
    declared = ["model.layers.3.mlp.experts.7.gate_proj"]
    monkeypatch.setenv("PRISMAQUANT_UNROUTED_EXPERT_PROVENANCE",
                       str(_sidecar(tmp_path, declared)))
    got = _load_unrouted_expert_declaration()

    assert got == frozenset(declared)
    # A ROUTED expert that happens to be missing a row is NOT in the set, so it
    # still reaches the allocator's refusal.
    assert "model.layers.3.mlp.experts.8.gate_proj" not in got


def test_an_unimplemented_rule_refuses_to_widen_the_gate(
        tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("PRISMAQUANT_UNROUTED_EXPERT_PROVENANCE",
                       str(_sidecar(tmp_path, ["a"], rule="hand_wave_v9")))
    with pytest.raises(ValueError, match="refusing to widen"):
        _load_unrouted_expert_declaration()


def test_cost_source_is_a_distinct_named_value() -> None:
    # Must not collide with a measured row's vocabulary.
    assert UNROUTED_EXPERT_COST_SOURCE == "unrouted_expert_weight_only"


def test_never_routed_expert_cannot_take_a_rung_of_its_own() -> None:
    """The defusing mechanism, pinned.

    h_trace == 0 for a never-routed expert is a coverage artifact, not proof it
    is unimportant, so it must not be free to take the cheapest rung on its own.
    It cannot: every projection of every expert in a MoE layer keys to ONE
    packed serving group, which the DP decides as a single multi-choice item.
    """
    profile = DeepseekV4Profile()
    routed = profile.packed_expert_format_group(
        "model.layers.23.mlp.experts.5.gate_proj")
    never_routed = profile.packed_expert_format_group(
        "model.layers.23.mlp.experts.27.gate_proj")
    other_proj = profile.packed_expert_format_group(
        "model.layers.23.mlp.experts.27.down_proj")

    assert routed is not None
    assert never_routed == routed
    assert other_proj == routed
    # ...and a different layer is a different unit, so this is not a global
    # collapse that would hide the per-layer K14/K15 decision.
    assert profile.packed_expert_format_group(
        "model.layers.24.mlp.experts.5.gate_proj") != routed


def test_passthrough_rows_are_not_stamped_with_a_cost_source() -> None:
    """A structurally-zero passthrough row must stay bit-exact.

    An explicit cost_source makes cost_entry_is_bit_exact return False, and the
    row would then read as BRANCH_MEASURED off its output_mse=0.0 — claiming a
    measurement that never happened.
    """
    from prismaquant.priced_cell import cost_entry_is_bit_exact

    passthrough = {"weight_mse": 0.0, "output_mse": 0.0, "rel_output_mse": 0.0}
    assert cost_entry_is_bit_exact(passthrough, "BF16")

    stamped = dict(passthrough, cost_source=UNROUTED_EXPERT_COST_SOURCE)
    assert not cost_entry_is_bit_exact(stamped, "BF16")


def test_weight_only_row_is_not_read_as_a_measured_row() -> None:
    from prismaquant.priced_cell import _has_measured_output_mse

    row = {"weight_mse": 1e-4, "output_mse": 0.0, "rel_output_mse": 0.0,
           "output_mse_measured": False,
           "cost_source": UNROUTED_EXPERT_COST_SOURCE}
    assert not _has_measured_output_mse({}, row)
