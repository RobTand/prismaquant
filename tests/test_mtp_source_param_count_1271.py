"""BF16 draft floors must agree with the validated source tensor identity."""
import pytest

from prismaquant.glm_mtp_selection import select_mtp_rungs
from prismaquant.joint_aura import identity_sha256
from test_glm_mtp_selection import CONSTANTS, PARAMS, ROUTED, R832, SHARED, _payload


def test_undersized_positive_params_cannot_manufacture_a_bf16_budget_fit():
    payload = _payload()
    payload["params"] = {unit: 1 for unit in payload["params"]}
    # Every validated source operator still names [64,128]. Metadata of one
    # param would invent an 18-byte full BF16 draft under this 100-byte cap.
    with pytest.raises(ValueError, match="params.*source.*shape"):
        select_mtp_rungs(payload, byte_budget=100, constants=CONSTANTS)


@pytest.mark.parametrize("unit", [ROUTED[0], SHARED[0]])
@pytest.mark.parametrize("count", [PARAMS - 1, PARAMS + 1])
def test_parameter_count_must_match_source_identity_for_every_unit(unit, count):
    payload = _payload()
    payload["params"][unit] = count
    with pytest.raises(ValueError, match="params.*source.*shape"):
        select_mtp_rungs(payload, byte_budget=10**12, constants=CONSTANTS)


def test_ineligible_second_rung_cannot_hide_a_different_source_count():
    payload = _payload()
    row = payload["costs"][ROUTED[0]][R832]
    operator = row["joint_operator_identity"]
    # Keep a complete, correctly hashed row, but price a different source
    # shape on the second rung. Eligibility must not conceal that mismatch.
    for field in ("source_weight", "rendered_weight"):
        operator[field]["shape"] = [32, 128]
        operator[field]["logical_bytes"] //= 2
    row["joint_operator_identity_sha256"] = identity_sha256(operator)
    with pytest.raises(ValueError, match="params.*source.*shape"):
        select_mtp_rungs(payload, byte_budget=10**12, constants=CONSTANTS,
                         eligible=lambda _unit, rung: rung != R832)


@pytest.mark.parametrize("unit", [ROUTED[0], SHARED[0]])
@pytest.mark.parametrize("count", [1, PARAMS])
def test_empty_unit_rungs_cannot_inherit_another_units_source_identity(unit, count):
    payload = _payload()
    payload["costs"][unit] = {}
    payload["wire_bytes"][unit] = {}
    payload["params"][unit] = count
    with pytest.raises(ValueError, match="MTP unit .*no priced source"):
        select_mtp_rungs(payload, byte_budget=10**12, constants=CONSTANTS)


def test_priced_but_ineligible_rungs_still_bind_a_valid_bf16_fallback():
    payload = _payload()
    expected = 2 * PARAMS * len(payload["params"])
    result = select_mtp_rungs(payload, byte_budget=expected, constants=CONSTANTS,
                              eligible=lambda _unit, _rung: False)
    assert set(result["assignment"].values()) == {"BF16"}
    assert result["resident_bytes"] == expected
