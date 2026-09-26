"""Allocator admission semantics for anchored-AURA cost rows (P0).

The campaign in ``anchored_cost`` prices a whole menu from one production-arm
render per unit. Those rows are a distinct provenance, and
these tests pin the four behaviours the allocator owes them — plus, more
importantly, the four things the admission must NOT do:

  * it must not be claimable by writing one string into a hand-made table;
  * it must not weaken the unmeasured-activation zero guard for anything else;
  * it must not be described as making AURA activation-quantization-aware;
  * it must not let the currency contract exist in two spellings.
"""

import pathlib

import pytest

from prismaquant import allocator_candidates as candidates
from prismaquant import anchored_cost
from prismaquant import format_registry as fr


_REPO = pathlib.Path(__file__).resolve().parents[1]


def _anchored_entry(**overrides) -> dict:
    entry = {
        "predicted_dloss": 4.0e-6,
        "cost_currency": candidates.ANCHORED_AURA_COST_CURRENCY,
        "cost_source": candidates.ANCHORED_AURA_COST_SOURCE,
        "fisher_application_count": 1,
    }
    entry.update(overrides)
    return entry


def test_anchored_row_needs_all_three_stamps_and_refuses_near_misses():
    assert candidates.cost_entry_is_anchored_aura_supersurrogate(
        _anchored_entry())

    # Each stamp alone is insufficient: drop any one and the claim fails.
    for field in ("cost_currency", "cost_source", "fisher_application_count"):
        partial = _anchored_entry()
        partial.pop(field)
        assert not candidates.cost_entry_is_anchored_aura_supersurrogate(
            partial), field

    # A plausible-looking near-miss string must not be accepted. This is the
    # forgery the three-stamp design exists to refuse.
    for forged in (
        {"cost_currency": "predicted_dloss"},
        {"cost_currency": "aura"},
        {"cost_source": "render"},
        {"cost_source": "aura_render"},
        {"cost_source": candidates.SOURCE_PASSTHROUGH_COST_SOURCE},
    ):
        assert not candidates.cost_entry_is_anchored_aura_supersurrogate(
            _anchored_entry(**forged)), forged

    # The h^2 guard: predicted_dloss already contains the KL-Fisher, so a row
    # claiming a second application is not an anchored row.
    for count in (0, 2, "1", None):
        assert not candidates.cost_entry_is_anchored_aura_supersurrogate(
            _anchored_entry(fisher_application_count=count)), count

    assert not candidates.cost_entry_is_anchored_aura_supersurrogate({})
    assert not candidates.cost_entry_is_anchored_aura_supersurrogate(None)


def test_anchored_zero_is_retained_but_the_guard_keeps_full_strength():
    """The zero-admission bypass is scoped, not a global weakening."""
    stats = {"h_trace": 1.0}
    fmt = "NVFP4"
    assert fr.get_format(fmt).act_quant_changes_input, (
        "test premise: the rung quantizes activations")

    anchored_zero = _anchored_entry(predicted_dloss=0.0)
    assert not candidates.cost_entry_prices_unmeasured_activation_at_zero(
        stats, anchored_zero, 0.0, fmt)

    # Every other zero-priced row on an activation-quantizing format is still
    # removed. A one-stamp-short row is NOT anchored and must not inherit the
    # bypass — this is the regression that would silently reopen the free-lunch
    # cell the guard exists for.
    for entry in (
        {"predicted_dloss": 0.0},
        {"predicted_dloss": 0.0, "cost_source": "band_interpolated"},
        _anchored_entry(predicted_dloss=0.0, fisher_application_count=2),
        {k: v for k, v in _anchored_entry(predicted_dloss=0.0).items()
         if k != "cost_currency"},
    ):
        assert candidates.cost_entry_prices_unmeasured_activation_at_zero(
            stats, entry, 0.0, fmt), entry

    # And the guard's own positive-sensitivity exemption still governs: a
    # measured-dead unit stays free to take the cheapest format.
    assert not candidates.cost_entry_prices_unmeasured_activation_at_zero(
        {"h_trace": 0.0}, {"predicted_dloss": 0.0}, 0.0, fmt)


def test_anchored_row_is_priced_directly_and_stamped_with_its_own_branch():
    stats = {"h_trace": 2.0}
    fmt = "FP8_E4M3"
    entry = _anchored_entry(predicted_dloss=7.5e-6)

    assert candidates.cost_entry_activation_pricing_branch(
        stats, entry, fmt) == candidates.ANCHORED_AURA_BRANCH
    # Distinguishable from "we looked for a sample and found none".
    assert candidates.ANCHORED_AURA_BRANCH != candidates.BRANCH_UNCALIBRATED

    priced = candidates.cost_entry_predicted_dloss(
        stats, entry, format_name=fmt)
    weight_only = candidates.cost_entry_weight_only_dloss(stats, entry)
    assert priced == weight_only, "anchored rows are read directly"


def test_anchored_rows_never_enter_the_p5a_calibration_sample():
    fmt = fr.get_format("FP8_E4M3")
    stats = {"unit.0": {"h_trace": 1.0}}
    costs = {"unit.0": {fmt.name: _anchored_entry()}}

    rows, measured, weight_only = (
        candidates.collect_activation_calibration_rows(stats, costs, [fmt]))
    assert rows == []
    # Neither census counts it: inflating weight_only_by_family would make
    # ``calibrate`` fail-close over a population the penalty never touches.
    assert measured.get(str(fmt.family), 0) == 0
    assert weight_only.get(str(fmt.family), 0) == 0


def test_the_currency_contract_has_exactly_one_definition():
    """Two spellings is how a near-miss silently routes down the wrong branch."""
    assert anchored_cost.AURA_CURRENCY is (
        candidates.ANCHORED_AURA_COST_CURRENCY)
    assert anchored_cost.PRODUCTION_RENDER_SOURCE is (
        candidates.ANCHORED_AURA_COST_SOURCE)


def test_the_flag_is_not_documented_as_an_activation_error_model():
    """AURA is activation-WEIGHTED, not activation-quantization-aware.

    ``aura_cost`` runs its adjoint on unquantized boundary activations and
    ``dW`` is a weight delta. A future edit that re-describes this admission as
    "activation-inclusive" would be certifying something the code does not do,
    which is the exact claim this project retired on 2026-08-11.
    """
    assert candidates.AURA_SUPERSURROGATE_ALLOCATOR_SEMANTICS is True

    source = (_REPO / "prismaquant/allocator_candidates.py").read_text()
    assert "activation-quantization-BLIND" in source
    assert "NOT a claim that AURA models activation-QUANTIZATION error" in source
    predicate = source.split(
        "def cost_entry_is_anchored_aura_supersurrogate")[1].split(
            "\ndef ")[0]
    assert "weights-only" in predicate
    assert "activation-quantization-blind" in predicate

    aura = (_REPO / "prismaquant/aura_cost.py").read_text()
    assert "activation_quantize_dequantize" not in aura, (
        "aura_cost now quantizes activations; the blindness limitation "
        "documented on the admission predicate must be re-derived")


# One Tessera family per activation route, at several rates each. Until
# 2026-09-25 this bound was asserted over the codebook families' K ladders;
# that lane was archived (#1304), and Tessera's rate ladders are the live
# within-family ladders an anchored row prices.
_TESSERA_LADDERS = {
    "E2M1_K2": ("TESSERA_E2M1_K2_R512", "TESSERA_E2M1_K2_R896"),
    "E2M1_K1": ("TESSERA_E2M1_K1_R256", "TESSERA_E2M1_K1_R512"),
    "E4M3_K1": (
        "TESSERA_E4M3_K1_R512", "TESSERA_E4M3_K1_R896", "TESSERA_E4M3_K1_R1024",
    ),
}


@pytest.mark.parametrize("family", sorted(_TESSERA_LADDERS))
def test_activation_path_is_constant_across_rate_within_each_family(family):
    """The bound on the standing limitation, asserted rather than assumed.

    Blindness that is constant within a family cannot reorder that family's
    rungs; it can only move the family-choice margin. If a future rung breaks
    this, the limitation stops being merely a family-margin question.
    """
    paths = {
        bool(fr.get_format(name).act_quant_changes_input)
        for name in _TESSERA_LADDERS[family]
    }
    assert len(paths) == 1, (family, paths)


def test_terminals_keep_the_passthrough_contract_not_the_aura_branch():
    """A byte-verbatim terminal is exact by construction, not an anchored row."""
    terminal = {
        "predicted_dloss": 0.0,
        "cost_currency": candidates.ANCHORED_AURA_COST_CURRENCY,
        "cost_source": candidates.SOURCE_PASSTHROUGH_COST_SOURCE,
        "fisher_application_count": 1,
    }
    assert not candidates.cost_entry_is_anchored_aura_supersurrogate(terminal)

    fmt = "MXFP4_SOURCE"
    assert not fr.get_format(fmt).act_quant_changes_input
    assert candidates.cost_entry_is_source_passthrough(terminal, fmt)
    assert candidates.cost_entry_activation_pricing_branch(
        {"h_trace": 1.0}, terminal, fmt) == candidates.BRANCH_SOURCE_PASSTHROUGH

    block = "FP8_BLOCK_UE8M0_SOURCE"
    assert not fr.get_format(block).act_quant_changes_input
    assert block in candidates.SOURCE_PASSTHROUGH_FORMATS
    assert candidates.cost_entry_is_source_passthrough(terminal, block)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
