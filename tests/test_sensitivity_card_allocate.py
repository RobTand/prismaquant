"""End-to-end: a Sensitivity Card + an arbitrary format menu drives the real DP.

These tests pass supplied format menus through the actual
`allocator_solver.solve_allocation`. They check allocation plumbing, budget
feasibility and cost ordering; they do not independently prove optimality.

Synthetic weights are used deliberately. The point under test is the plumbing
and the ordering the cost induces, not any particular model's numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from prismaquant.format_cost_protocol import CostModel, FormatDescriptor
from prismaquant.sensitivity_card import (
    CardProvenance,
    RenderBasis,
    SensitivityCard,
    SensitivityUnit,
    UnitTopology,
)
from prismaquant.sensitivity_card_allocate import (
    allocate_from_card,
    assignment_summary,
    candidates_from_card,
)

OUT, IN, TOKENS, N_UNITS = 16, 24, 64, 12


class StubPlugin:
    """A format whose weight error is a fixed function of its bit width.

    Real quantizers live in `format_cost_registry`; this keeps the allocation
    tests deterministic and CPU-only so they can run beside GPU work.
    """

    def __init__(self, name, weight_bits, act_bits=None, passthrough=False,
                 requires_source_dtype=None, err_scale=None):
        self.descriptor = FormatDescriptor(
            name=name, weight_bits=weight_bits, act_bits=act_bits,
            quantizes_activations=act_bits is not None,
            passthrough=passthrough, requires_source_dtype=requires_source_dtype)
        # Error falls ~4x per extra bit, the standard quantizer scaling.
        self._err = err_scale if err_scale is not None else 4.0 ** (-weight_bits)

    def weight_error(self, unit, weight):
        return np.full((unit.out_features, unit.in_features), self._err)


def _card(n_units=N_UNITS, seed=0, source_dtype="bfloat16"):
    rng = np.random.default_rng(seed)
    units = []
    for i in range(n_units):
        g_sq = rng.random((TOKENS, OUT)) + 0.1
        x_sq = rng.random((TOKENS, IN)) + 0.1
        # Give units very different sensitivities so the DP has real choices.
        g_sq *= (1.0 + 10.0 * (i / max(1, n_units - 1)))
        H = g_sq.T @ x_sq
        units.append(SensitivityUnit(
            topology=UnitTopology(name=f"model.layers.{i}.mlp.down_proj",
                                  layer_index=i, role="down",
                                  source_dtype=source_dtype),
            out_features=OUT, in_features=IN, n_params=OUT * IN, n_tokens=TOKENS,
            h_trace_raw=float(H.sum()), h_w2_sum_raw=0.0,
            w_norm_sq=1.0, w_max_abs=1.0,
            fisher_row=H.sum(axis=1), fisher_col=H.sum(axis=0),
            act_sq_sum=x_sq.sum(axis=0), g_sq_sum=g_sq.sum(axis=0),
            act_absmax=np.sqrt(x_sq).max(axis=0)))
    prov = CardProvenance(model_id="test/m", calib_hash="a" * 64,
                          n_calib_samples=8, seq_len=512, probe_commit="abc",
                          render_basis=RenderBasis.RTN)
    card = SensitivityCard(prov, units)
    card.validate()
    return card


def _weights(seed=1):
    rng = np.random.default_rng(seed)
    table = {}

    def lookup(name):
        if name not in table:
            table[name] = rng.standard_normal((OUT, IN)).astype(np.float32)
        return table[name]
    return lookup


def test_arbitrary_menu_produces_a_valid_allocation():
    card, w = _card(), _weights()
    menu = [StubPlugin("W4", 4.0), StubPlugin("W8", 8.0), StubPlugin("W16", 16.0)]
    result = allocate_from_card(card, w, menu, target_bits=6.0)
    assert result is not None
    assignment, chosen = result
    assert set(assignment) == set(card.names())
    assert set(assignment.values()) <= {"W4", "W8", "W16"}
    achieved = sum(chosen[n].bits_per_param for n in assignment) / len(assignment)
    assert achieved <= 6.0 + 1e-6


def test_adding_a_format_to_the_menu_changes_the_allocation():
    """The menu is genuinely arbitrary: a new rung is used when it helps."""
    card, w = _card(), _weights()
    two_rung = [StubPlugin("W4", 4.0), StubPlugin("W16", 16.0)]
    three_rung = two_rung + [StubPlugin("W6", 6.0)]

    a2 = allocate_from_card(card, w, two_rung, target_bits=7.0)[0]
    a3 = allocate_from_card(card, w, three_rung, target_bits=7.0)[0]
    assert "W6" in set(a3.values()), "the new rung should be selected"
    assert a2 != a3


def test_the_budget_binds():
    card, w = _card(), _weights()
    menu = [StubPlugin("W4", 4.0), StubPlugin("W8", 8.0), StubPlugin("W16", 16.0)]
    tight = allocate_from_card(card, w, menu, target_bits=4.5)[1]
    loose = allocate_from_card(card, w, menu, target_bits=12.0)[1]
    bits = lambda ch: sum(c.bits_per_param for c in ch.values()) / len(ch)
    assert bits(tight) < bits(loose)


def test_infeasible_budget_returns_none():
    card, w = _card(), _weights()
    menu = [StubPlugin("W8", 8.0), StubPlugin("W16", 16.0)]
    assert allocate_from_card(card, w, menu, target_bits=4.0) is None


def test_sensitive_units_get_more_bits():
    """The card's whole job: spend the budget where the Fisher says it matters."""
    card, w = _card(), _weights()
    menu = [StubPlugin("W4", 4.0), StubPlugin("W8", 8.0), StubPlugin("W16", 16.0)]
    assignment, chosen = allocate_from_card(card, w, menu, target_bits=7.0)

    bits = {n: chosen[n].bits_per_param for n in assignment}
    sens = {n: card[n].h_trace for n in assignment}
    hi = sorted(sens, key=lambda n: sens[n])[-3:]
    lo = sorted(sens, key=lambda n: sens[n])[:3]
    assert np.mean([bits[n] for n in hi]) > np.mean([bits[n] for n in lo])


def test_passthrough_illegality_removes_the_format_not_the_unit():
    """An FP8-source unit may not take BF16 passthrough, but is still allocated."""
    card, w = _card(source_dtype="float8_e4m3fn"), _weights()
    menu = [StubPlugin("W4", 4.0),
            StubPlugin("BF16", 16.0, passthrough=True,
                       requires_source_dtype="bfloat16")]
    stats, cands = candidates_from_card(card, w, menu)
    assert set(stats) == set(card.names()), "units must survive"
    for options in cands.values():
        assert {c.fmt for c in options} == {"W4"}


def test_aqua_demotes_activation_quantized_formats():
    """AQUA-AURA changes the ANSWER, not just the number.

    Two formats at the same weight bits and the same weight error, differing
    only in whether the kernel quantizes activations. Under the weight-only
    model they tie and the DP is indifferent; under AQUA the A4 variant carries
    a strictly larger predicted loss and must lose.
    """
    card, w = _card(), _weights()
    a4 = StubPlugin("W4A4", 4.0, act_bits=4, err_scale=4.0 ** -4)
    a16 = StubPlugin("W4A16", 4.0, act_bits=None, err_scale=4.0 ** -4)
    menu = [a4, a16, StubPlugin("W16", 16.0)]

    _, cands_marginal = candidates_from_card(card, w, menu,
                                             model=CostModel.MARGINAL)
    _, cands_aqua = candidates_from_card(card, w, menu, model=CostModel.AQUA)

    name = card.names()[0]
    by_fmt = lambda cs: {c.fmt: c.predicted_dloss for c in cs}
    m, a = by_fmt(cands_marginal[name]), by_fmt(cands_aqua[name])

    # Weight-only: the two 4-bit formats are indistinguishable.
    assert m["W4A4"] == m["W4A16"]
    # AQUA: the activation term separates them, and only the A4 side moves.
    assert a["W4A4"] > a["W4A16"]
    assert a["W4A16"] == m["W4A16"]

    # And the DP acts on it: at a budget where 4-bit is affordable, the
    # activation-quantized rung is not the one chosen.
    assignment = allocate_from_card(card, w, menu, target_bits=4.0,
                                    model=CostModel.AQUA)[0]
    assert "W4A4" not in set(assignment.values())


def test_summary_is_a_histogram():
    card, w = _card(), _weights()
    menu = [StubPlugin("W4", 4.0), StubPlugin("W16", 16.0)]
    assignment, _ = allocate_from_card(card, w, menu, target_bits=8.0)
    hist = assignment_summary(assignment)
    assert sum(hist.values()) == len(assignment)
