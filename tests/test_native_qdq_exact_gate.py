"""The activation gate has no tolerance, and the GEMM bound has a derivation.

Measured 2026-09-13 on ``TESSERA_E2M1_K2_R896`` (issue #567): ``o_proj`` prefill
0.095703125, ``v_proj`` prefill 0.1435546875, ``v_proj`` decode 0.0078125.  The
first two failed and the third PASSED, and the only thing between them was a
constant that was a CLI default.  All three are code flips.
"""
import pytest

from prismaquant.native_operator_panel import (derive_gemm_numerics,
                                               require_attested_activation_oracle,
                                               require_panel_activation_attestation,
                                               validate_native_numerics)

MEASURED = {"o_proj_prefill": 0.095703125, "v_proj_prefill": 0.1435546875,
            "v_proj_decode": 0.0078125}


def _error(max_abs, numerics):
    return {"status": "passed", "finite": True, "max_normalized_error": 0.5,
            "max_abs_error": max_abs, **numerics}


def test_the_activation_gate_admits_exact_agreement_only():
    numerics = {"atol": 0.25, "rtol": 0.0}
    validate_native_numerics(_error(0.0, numerics), numerics,
                             phase="prefill", kind="qdq_numerics", exact=True)


@pytest.mark.parametrize("label", sorted(MEASURED))
def test_every_measured_fp4_divergence_is_refused(label):
    """Including the one that passed."""
    numerics = {"atol": 0.25, "rtol": 0.0}
    with pytest.raises(ValueError, match="E2M1 code flipped"):
        validate_native_numerics(_error(MEASURED[label], numerics), numerics,
                                 phase="prefill", kind="qdq_numerics", exact=True)


def test_the_gemm_gate_keeps_its_tolerance():
    numerics = {"atol": 0.25, "rtol": 0.0}
    validate_native_numerics(_error(0.1, numerics), numerics,
                             phase="prefill", kind="numerics")


def test_an_absent_absolute_error_cannot_satisfy_the_exact_gate():
    numerics = {"atol": 0.25, "rtol": 0.0}
    with pytest.raises(ValueError):
        validate_native_numerics(
            {"status": "passed", "finite": True, "max_normalized_error": 0.0,
             "max_abs_error": None, **numerics},
            numerics, phase="decode", kind="qdq_numerics", exact=True)


def test_the_derived_bound_is_the_dtypes_and_the_operands():
    numerics, derivation = derive_gemm_numerics(8.0, k=1024)
    coefficient = 4.0 * 2.0 ** -8 + 2.0 * 1024 * 2.0 ** -24
    assert numerics == {"atol": coefficient * 8.0, "rtol": 0.0}
    assert derivation["coefficient"] == coefficient
    assert derivation["k"] == 1024
    assert derivation["scope"] == "gemm_output_only"


def test_the_derived_bound_moves_with_the_operands_not_with_a_default():
    small, _ = derive_gemm_numerics(1.0, k=64)
    large, _ = derive_gemm_numerics(100.0, k=64)
    assert large["atol"] == pytest.approx(100.0 * small["atol"])


def test_a_zero_operand_magnitude_bounds_at_zero():
    numerics, _ = derive_gemm_numerics(0.0, k=16)
    assert numerics == {"atol": 0.0, "rtol": 0.0}


def test_the_contraction_length_is_required():
    with pytest.raises(ValueError, match="contraction length"):
        derive_gemm_numerics(1.0, k=0)


def test_a_panel_that_priced_an_activation_carries_what_attested_it():
    stamp = {"schema": "prismaquant.activation_quantizer_attestation.v1"}
    require_panel_activation_attestation({"activation_quantizer_attestation": stamp}, True)
    with pytest.raises(ValueError, match="carries no attestation"):
        require_panel_activation_attestation({"activation_quantizer_attestation": None}, True)
    with pytest.raises(ValueError, match="carries no attestation"):
        require_panel_activation_attestation({}, True)


def test_a_panel_that_priced_nothing_must_not_carry_one():
    require_panel_activation_attestation({"activation_quantizer_attestation": None}, False)
    with pytest.raises(ValueError, match="disagree about what was priced"):
        require_panel_activation_attestation(
            {"activation_quantizer_attestation": {"schema": "x"}}, False)


def test_a_static_contract_with_no_published_table_refuses_at_freeze():
    from prismaquant.tessera_runtime_contract import TesseraContractError

    activation = {"quantizes_input": True, "quantizer": "pq.oracle",
                  "static_contract": {"execution": "e2m1_group16_ue4m3_static"}}
    with pytest.raises(TesseraContractError, match="publishes no quantiser"):
        require_attested_activation_oracle(
            activation, platform="sm_121", table={})


def test_an_attestation_is_addressed_by_platform():
    activation = {"quantizes_input": True, "quantizer": "pq.oracle",
                  "static_contract": {"execution": "e2m1_group16_ue4m3_static"}}
    with pytest.raises(ValueError, match="addressed by platform"):
        require_attested_activation_oracle(activation, platform="", table={})


def test_a_dynamic_scale_quantizer_records_that_it_is_unattested():
    """Named, not hidden.

    fp8's scale is derived from ``x``, so both sides compute the same function
    of the same tensor and every measured cell agreed at exactly 0.0.  That is
    evidence, not attestation, and the stamp says which of the two it is rather
    than leaving it to be inferred from a missing field.
    """
    stamp = require_attested_activation_oracle(
        {"quantizes_input": True, "quantizer": "pq.fp8", "static_contract": None},
        platform="sm_121", table={})
    assert stamp["status"] == "unattested_dynamic_scale"
    assert stamp["attests"] is None


def test_a_format_that_does_not_quantize_attests_nothing():
    assert require_attested_activation_oracle(
        {"quantizes_input": False}, platform="sm_121") is None


def test_the_september_18_qkv_refusals_are_one_code_flips_that_stay_refused():
    """RobTand/prismaquant#717: 3 of 7 TESSERA_E2M1_K2 cells refuse.

    Re-measured 2026-09-18 under the attested image
    (vllm/vllm-openai@sha256:61fc8a89, receipts
    frontier-qwen3-0.6b-20260918-panels/consume-native-receipts.json: 53
    admitted, 3 refused): q/k/v_proj prefill disagree by exactly 0.015625 --
    one E2M1 code at the block scale -- while o/gate/up/down_proj are
    bit-exact. The executing image is refuted as the variable, so the gate
    must keep refusing the flip rather than absorb it into a tolerance.
    """
    numerics = {"atol": 0.25, "rtol": 0.0}
    for unit in ("q_proj", "k_proj", "v_proj"):
        with pytest.raises(ValueError, match="E2M1 code flipped"):
            validate_native_numerics(_error(0.015625, numerics), numerics,
                                     phase="prefill", kind="qdq_numerics", exact=True)
    for unit in ("o_proj", "gate_proj", "up_proj", "down_proj"):
        validate_native_numerics(_error(0.0, numerics), numerics,
                                 phase="prefill", kind="qdq_numerics", exact=True)
