"""The A4 ``input_global_scale`` POLICY travels with the artifact (#624).

The export gate already binds the activation-scale VALUE: the F32 scalar the
exporter writes must be the F32 scalar the costs were priced against.  It could
not bind the FORMULA.  ``legacy_6_over_calibration_amax.v1`` (``6/amax``) and
``full_e4m3_range_448x6_over_calibration_amax.v1`` (``448*6/amax``) produce the
same tensor shape and are 448x apart in serve-time block-scale headroom, and the
pinned Tessera runtime executes whichever scalar it is handed: its attested
``activation_quantizers`` table publishes the arithmetic AROUND ``G`` -- the
UE4M3 block scale ties to zero at ``block_amax/6*G = 2**-10`` and saturates at
448 -- and publishes no policy for ``G`` itself.  ``G`` is a producer choice.

So the label is the producer's own claim, and until this change it was written
twice by the campaign (the payload identity and the scale file's
``__metadata__``) and then dropped: the allocation block carried only values and
the export report carried no policy at all, so a reader holding the checkpoint
had to chase an absolute path on a shared mount to learn which of the two
policies priced and serves it.

These tests drive the real producers (``tessera_campaign.write_export_inputs``,
``tessera_menu.priced_static_scales``) and the real gate
(``tessera_export_lane.require_priced_export_inputs``) on the A4 rung the
GLM-5.3-Flash campaign builds on, ``TESSERA_E2M1_K2_R896``.
"""
import json

import pytest

pytest.importorskip("torch")

from prismaquant import nvfp4_activation_contract as nac
from prismaquant import tessera_export_lane as export

A4_FMT = "TESSERA_E2M1_K2_R896"
DENSE = "model.layers.0.self_attn.o_proj"
LEGACY = nac.LEGACY_INPUT_GLOBAL_SCALE_POLICY
FULL = nac.FULL_E4M3_INPUT_GLOBAL_SCALE_POLICY

#: A scalar produced by the legacy formula from a calibration maximum, rounded
#: exactly as an exported F32 tensor carries it.
LEGACY_SCALE = nac.input_global_scale_from_max_abs(37.5, policy=LEGACY)


def _units():
    return {DENSE: LEGACY_SCALE}


def _layer_config(tmp_path, *, units, policy, fmt=A4_FMT):
    """A dense A4 allocation.  ``policy=None`` is a pre-#624 allocation."""
    payload = {name: {"data_type": "tessera", "bits": 4,
                      "tessera_format": fmt} for name in units}
    scales_block = {"schema": export.PRICED_STATIC_SCALES_SCHEMA,
                    "units": dict(units)}
    if policy is not None:
        scales_block["input_global_scale_policy"] = policy
    payload["__prismaquant__"] = {
        "tessera_hessian": {"supplied": False, "text_sha256": "a" * 64,
                            "fit_ids_sha256": "b" * 64, "fit_tokens": 4096},
        "tessera_activation_static_scales": scales_block,
    }
    path = tmp_path / "layer_config.json"
    path.write_text(json.dumps(payload))
    return path


def _scales_file(tmp_path, units, *, policy):
    """The campaign's OWN writer, so the label under test is the shipped one."""
    from prismaquant.tessera_campaign import write_export_inputs

    _capture, scales, _digest = write_export_inputs(
        tmp_path, hessians=None, hessian_rows={}, hessian_identity={},
        static_scales=dict(units), static_scale_policy=policy)
    return scales


def _unlabelled_scales_file(tmp_path, units):
    """A hand-built scale file: right values, no policy label at all.

    This is the shape of the campaign-external stub files that get built from a
    census ``max_abs`` dump, and the file the gate must not read as "legacy".
    """
    from safetensors.torch import save_file

    path = tmp_path / "stub-input-scales.safetensors"
    save_file({f"{name}.input_global_scale":
               nac.input_global_scale_tensor(value)
               for name, value in units.items()}, str(path))
    return path


# ---------------------------------------------------------------------------
# The producer: the allocation states the formula its numbers came out of
# ---------------------------------------------------------------------------

def test_the_allocation_block_carries_the_cost_table_s_own_policy():
    from prismaquant.tessera_menu import priced_static_scales

    block = priced_static_scales(
        {DENSE: A4_FMT},
        {DENSE: {A4_FMT: {"input_global_scale": LEGACY_SCALE}}},
        policy=LEGACY)
    assert block["input_global_scale_policy"] == LEGACY
    assert block["units"] == {DENSE: LEGACY_SCALE}


def test_a_cost_table_with_no_policy_is_refused_not_defaulted():
    """A table that never said is not a table that said 'legacy'."""
    from prismaquant.tessera_menu import priced_static_scales

    with pytest.raises(ValueError, match="carries no activation_static_scales"):
        priced_static_scales(
            {DENSE: A4_FMT},
            {DENSE: {A4_FMT: {"input_global_scale": LEGACY_SCALE}}},
            policy=None)


def test_a_selection_that_priced_no_static_scale_names_no_policy():
    """No priced scalar, no formula to name -- and nothing invented either.

    The export gate refuses such a unit as unbound (its row never said what
    priced it) before it would ever ask for a policy, so requiring one here
    would refuse a dynamic-contract selection that is perfectly exportable.
    """
    from prismaquant.tessera_menu import priced_static_scales

    block = priced_static_scales(
        {DENSE: "TESSERA_E4M3_K1_R1024"},
        {DENSE: {"TESSERA_E4M3_K1_R1024": {"output_mse": 0.1}}},
        policy=None)
    assert block == {"schema": export.PRICED_STATIC_SCALES_SCHEMA,
                     "units": {}}


def test_an_alias_is_canonicalised_so_two_spellings_cannot_disagree():
    from prismaquant.tessera_menu import priced_static_scales

    block = priced_static_scales(
        {DENSE: A4_FMT},
        {DENSE: {A4_FMT: {"input_global_scale": LEGACY_SCALE}}},
        policy="legacy")
    assert block["input_global_scale_policy"] == LEGACY


# ---------------------------------------------------------------------------
# The gate: the file's label and the allocation's label are one object
# ---------------------------------------------------------------------------

def test_the_matched_pair_exports_and_the_report_states_the_policy(tmp_path):
    units = _units()
    config = _layer_config(tmp_path, units=units, policy=LEGACY)
    scales = _scales_file(tmp_path, units, policy=LEGACY)
    report = export.require_priced_export_inputs(
        config, input_scales_path=scales)
    assert report["input_global_scale_policy"] == LEGACY
    assert report["input_scales_bound_units"] == 1


def test_a_scale_file_labelled_the_other_policy_is_refused(tmp_path):
    """The values match; the formulas do not.  One record is wrong.

    Before #624 the gate compared only the scalars, so this pair exported and
    the ship record would have named a policy that did not price these costs.
    """
    units = _units()
    config = _layer_config(tmp_path, units=units, policy=LEGACY)
    scales = _scales_file(tmp_path, units, policy=FULL)
    with pytest.raises(export.TesseraExportLaneError,
                       match="but the allocation priced under"):
        export.require_priced_export_inputs(config, input_scales_path=scales)


def test_an_unlabelled_scale_file_is_refused(tmp_path):
    units = _units()
    config = _layer_config(tmp_path, units=units, policy=LEGACY)
    scales = _unlabelled_scales_file(tmp_path, units)
    with pytest.raises(export.TesseraExportLaneError,
                       match="carries no __metadata__"):
        export.require_priced_export_inputs(config, input_scales_path=scales)


def test_an_allocation_that_declares_no_policy_is_refused(tmp_path):
    """A pre-#624 allocation: the file says, and there is nothing to check."""
    units = _units()
    config = _layer_config(tmp_path, units=units, policy=None)
    scales = _scales_file(tmp_path, units, policy=LEGACY)
    with pytest.raises(export.TesseraExportLaneError,
                       match="declares no input_global_scale_policy"):
        export.require_priced_export_inputs(config, input_scales_path=scales)


def test_an_unknown_policy_name_is_refused(tmp_path):
    units = _units()
    config = _layer_config(tmp_path, units=units, policy=LEGACY)
    scales = _scales_file(tmp_path, units, policy="six_over_amax_probably")
    with pytest.raises(export.TesseraExportLaneError,
                       match="not one of"):
        export.require_priced_export_inputs(config, input_scales_path=scales)


def test_the_policy_reaches_the_build_anchor(tmp_path, monkeypatch):
    """The ship record is where a reader of the bytes finds the policy."""
    units = _units()
    config = _layer_config(tmp_path, units=units, policy=LEGACY)
    scales = _scales_file(tmp_path, units, policy=LEGACY)
    report = export.require_priced_export_inputs(
        config, input_scales_path=scales)
    build = {}
    if report.get("input_global_scale_policy") is not None:
        build["tessera_activation_input_global_scale_policy"] = report[
            "input_global_scale_policy"]
    assert build == {"tessera_activation_input_global_scale_policy": LEGACY}


# ---------------------------------------------------------------------------
# What the pinned runtime does and does not attest about G
# ---------------------------------------------------------------------------

def test_the_two_policies_are_exactly_448x_apart():
    """The headroom question #624 records, stated as arithmetic.

    Nothing here is a preference between them: the legacy scalar puts every
    stored block scale in (0, 1] (underflow at 1024x below the calibration
    maximum, 448x of clip headroom above it), the full-E4M3 scalar puts them in
    (0, 448] (no clip headroom above the calibration maximum).  Which is better
    is a served measurement, not a constant.
    """
    legacy = nac.input_global_scale_from_max_abs(37.5, policy=LEGACY)
    full = nac.input_global_scale_from_max_abs(37.5, policy=FULL)
    assert full == pytest.approx(legacy * nac.FP8_E4M3_MAX, rel=1e-6)
