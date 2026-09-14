"""The runtime's quantiser table, read and refused.

``reference_qdq`` is PrismaQuant's own re-implementation of the rounding rule a
Tessera serve executes, and contract v24 published only the rule's NAME
(RobTand/prismaquant#567). Contract v25 publishes ``activation_quantizers``:
eleven probe groups the kernel was actually run on
(RobTand/tessera#484, RobTand/tessera#485).

These cover the reader, the numeric coverage requirement and every refusal,
against the REAL published table -- and the two that matter most mutate OUR
arithmetic, not the fixture, because a passing attestation otherwise proves only
that the comparison ran.
"""
import copy
import json
from pathlib import Path

import pytest
import torch

from prismaquant import tessera_runtime_contract as trc
from prismaquant.nvfp4_activation_contract import E2M1_MIDPOINTS

CONTRACT = "e2m1_group16_ue4m3_static"
PLATFORM = "sm_121"
FIXTURE = Path(__file__).parent / "fixtures" / "tessera_activation_quantizers_v25.json"


def _payload():
    """The block as Tessera publishes it, before this reader touches it."""
    return copy.deepcopy(json.loads(FIXTURE.read_text()))


def _table(payload=None):
    return trc._parse_activation_quantizers(
        payload if payload is not None else _payload(), "<fixture>")


def _vectors(payload):
    return payload["activation_quantizers"]["platforms"][PLATFORM][
        "contracts"][CONTRACT]["vectors"]


def _contract_block(payload):
    return payload["activation_quantizers"]["platforms"][PLATFORM][
        "contracts"][CONTRACT]


def _attest(payload=None, **kwargs):
    return trc.require_activation_quantizer_attested(
        CONTRACT, platform=PLATFORM, table=_table(payload), **kwargs)


# --- the agreeing case, against the table the runtime actually published ----

def test_prismaquant_reproduces_the_published_table_exactly():
    """Both halves: the stored UE4M3 byte and every element's code.

    Eleven groups, 176 elements, over the seven E2M1 midpoints at dyadic and
    non-dyadic used scales, midpoints plus and minus one bf16 ulp, every code
    and both signs, element saturation, and the block-scale underflow tie,
    the value above it and the 448 overflow.
    """
    stamp = _attest(contract_sha256="a" * 64)
    assert stamp["vectors"] == 11 and stamp["elements"] == 176
    assert stamp["attests"] == ["amax_to_ue4m3_stored_scale",
                                "value_to_code_rounding"]
    assert stamp["does_not_attest"] == ["non_dyadic_used_scale"]
    assert stamp["contract_sha256"] == "a" * 64
    assert stamp["op"] == "torch.ops._C.scaled_fp4_quant"
    json.dumps(stamp, allow_nan=False)


# --- absence and addressing -------------------------------------------------

def test_an_absent_block_is_refused_not_skipped():
    with pytest.raises(trc.TesseraContractError, match="publishes no quantiser"):
        trc.require_activation_quantizer_attested(
            CONTRACT, platform=PLATFORM, table={})


def test_an_installed_table_cannot_bypass_the_reviewed_runtime_pin(monkeypatch):
    """The v25 fixture is not an authorized v24 pricing table merely by being installed."""
    from prismaquant.tessera_serving_runtime_pin import TesseraServingRuntimePinError

    monkeypatch.setattr(trc, "contract_path", lambda: FIXTURE)
    with pytest.raises(TesseraServingRuntimePinError, match="not the pinned Tessera"):
        trc.require_activation_quantizer_attested(CONTRACT, platform=PLATFORM)


def test_a_table_for_another_platform_does_not_attest_this_one():
    with pytest.raises(trc.TesseraContractError, match="publishes no quantiser"):
        trc.require_activation_quantizer_attested(
            CONTRACT, platform="gfx1201", table=_table())


def test_a_table_for_another_contract_does_not_attest_this_one():
    with pytest.raises(trc.TesseraContractError, match="publishes no quantiser"):
        trc.require_activation_quantizer_attested(
            "fp8_per_token_dynamic", platform=PLATFORM, table=_table())


def test_a_contract_without_the_block_still_parses():
    assert trc._parse_activation_quantizers({"formats": []}, "<fixture>") == {}


# --- the vocabulary is compared, never defaulted ----------------------------

def test_an_unknown_grammar_is_refused_rather_than_read_with_this_one():
    payload = _payload()
    payload["activation_quantizers"]["schema"] = "tessera.activation-quantizer.v2"
    with pytest.raises(trc.TesseraContractError, match="this reader implements only"):
        _table(payload)


@pytest.mark.parametrize("field,value", [
    ("op", "torch.ops._C.another_fp4_quant"),
    ("unit", "tensor"), ("grid", "E3M0"), ("block_scale", "E8M0"),
    ("global_scale", "dynamic_per_token"),
    ("unit_length", 32)])
def test_another_vocabulary_attests_a_different_quantizer(field, value):
    payload = _payload()
    _contract_block(payload)[field] = value
    with pytest.raises(trc.TesseraContractError, match="another vocabulary|unit_length"):
        _attest(payload)


@pytest.mark.parametrize("length", ["16", 16.5, True])
def test_unit_length_does_not_coerce_an_invalid_contract(length):
    payload = _payload()
    _contract_block(payload)["unit_length"] = length
    with pytest.raises(trc.TesseraContractError, match="positive integer"):
        _table(payload)


def test_an_unknown_member_is_a_review_not_a_skip():
    payload = _payload()
    _contract_block(payload)["detail"] = "the kernel rounds to nearest even"
    with pytest.raises(trc.TesseraContractError, match="does not know"):
        _table(payload)


# --- coverage: numeric, so a mislabelled boundary buys nothing --------------

@pytest.mark.parametrize("midpoint", E2M1_MIDPOINTS)
def test_every_missing_midpoint_is_a_refusal(midpoint):
    """Drop every vector that REACHES this midpoint, not just the one named
    after it: several probes cover the same tie at different used scales, and a
    test that removed only one of them would pass while proving nothing."""
    payload = _payload()
    block = _contract_block(payload)
    block["vectors"] = [v for v in block["vectors"]
                        if not _reaches(v, midpoint)]
    assert len(block["vectors"]) < len(_vectors(_payload()))
    with pytest.raises(trc.TesseraContractError, match="where the rounding RULE"):
        _attest(payload)


def test_a_mislabelled_boundary_does_not_buy_coverage():
    """Rename every probe to the midpoint boundary and drop the ones that
    reach the ties; the numeric check still refuses."""
    payload = _payload()
    block = _contract_block(payload)
    block["vectors"] = [dict(v, boundary="e2m1_midpoint_dyadic")
                        for v in block["vectors"]
                        if not _reaches(v, E2M1_MIDPOINTS[0])]
    with pytest.raises(trc.TesseraContractError, match="midpoint t=0.25"):
        _attest(payload)


def test_a_table_with_no_negative_input_leaves_the_sign_unattested():
    """Every published probe carries both signs, so this builds the table that
    does not: the dyadic midpoint group with its negative half folded onto the
    positive one, inputs and codes together so the group stays self-consistent
    and the refusal is the coverage gap, not a disagreement."""
    payload = _payload()
    block = _contract_block(payload)
    vector = copy.deepcopy(
        next(v for v in block["vectors"] if v["id"] == "midpoint_dyadic"))
    for i, text in enumerate(vector["input"]):
        bits = int(text, 16)
        if bits >> 15:
            vector["input"][i] = f"0x{bits & 0x7fff:04x}"
            vector["codes"][i] -= 8
    block["vectors"] = [vector]
    with pytest.raises(trc.TesseraContractError) as raised:
        _attest(payload)
    assert "negative input" in str(raised.value)


def test_a_table_that_never_saturates_leaves_the_top_of_the_lattice_open():
    """Drop every vector with an element above the top code, not only the one
    named after saturation: the block-scale overflow probe saturates too."""
    payload = _payload()
    block = _contract_block(payload)
    block["vectors"] = [v for v in block["vectors"] if not _saturates(v)]
    assert len(block["vectors"]) < len(_vectors(_payload()))
    with pytest.raises(trc.TesseraContractError, match="above the top code"):
        _attest(payload)


@pytest.mark.parametrize("byte", (0, 1))
def test_the_block_scale_underflow_boundary_is_required(byte):
    payload = _payload()
    block = _contract_block(payload)
    block["vectors"] = [v for v in block["vectors"]
                        if v["stored_scale"] != byte]
    with pytest.raises(trc.TesseraContractError, match="underflow boundary"):
        _attest(payload)


# --- disagreement, in both halves -------------------------------------------

def test_one_flipped_published_code_is_refused_and_named():
    payload = _payload()
    vector = _vectors(payload)[0]
    vector["codes"][1] = (vector["codes"][1] + 1) % 8
    with pytest.raises(trc.TesseraContractError) as raised:
        _attest(payload)
    assert "midpoint_dyadic" in str(raised.value)
    assert "element 1" in str(raised.value)
    assert "Do NOT widen a tolerance" in str(raised.value)


def test_one_flipped_published_stored_scale_is_refused_and_named():
    payload = _payload()
    vector = _vectors(payload)[0]
    vector["stored_scale"] += 1
    with pytest.raises(trc.TesseraContractError) as raised:
        _attest(payload)
    assert "the runtime stored" in str(raised.value)
    assert "midpoint_dyadic" in str(raised.value)


# --- mutate the DRIVER, not the fixture -------------------------------------

def test_the_check_bites_when_OUR_tie_break_moves(monkeypatch):
    """Swap PrismaQuant's tie-break for ties-to-lower -- the rule the test-only
    stub in ``tests/test_serving_nvfp4_route.py`` uses, and the live alternative
    candidate in #567 -- and the same published table must refuse."""
    from prismaquant import nvfp4_activation_contract as nac

    _attest()

    def ties_to_lower(normalized):
        positive = nac._e2m1_positive_table(normalized.device)
        magnitude = normalized.abs().contiguous()
        upper = torch.bucketize(magnitude, positive).clamp_max(
            positive.numel() - 1)
        lower = (upper - 1).clamp_min(0)
        choose = (positive[upper] - magnitude) < (magnitude - positive[lower])
        index = torch.where(choose, upper, lower)
        return index | (torch.signbit(normalized).to(index.dtype) << 3)

    monkeypatch.setattr(nac, "nvfp4_e2m1_code", ties_to_lower)
    with pytest.raises(trc.TesseraContractError, match="not the one the pinned"):
        _attest()


def test_the_check_bites_when_OUR_scale_saturation_moves(monkeypatch):
    """The other half. Move the UE4M3 ceiling in the stored-scale derivation and
    the ``block_scale_overflow`` probe, whose group maximum implies 512, must
    refuse.

    The mutation moves the ceiling rather than removing the clamp, and that is
    deliberate.  ``.to(torch.float8_e4m3fn)`` does not agree across torch builds
    about what happens to a value above the format's largest finite number:
    some saturate to 448, some do not.  A "drop the clamp" mutation therefore
    tests the installed torch, not this module -- it bit on the dl380g10 CPU
    lane and did NOT bite on the hosted runner, from the same source.  Moving
    the ceiling is a different rounding RULE on every build, which is what the
    attestation is supposed to catch.  (The driver's own explicit
    ``clamp(max=FP8_E4M3_MAX)`` is why that divergence changes nothing about
    what PrismaQuant computes: it never relies on the cast to saturate.)
    """
    from prismaquant import nvfp4_activation_contract as nac

    _attest()

    def half_ceiling(grouped, g):
        amax = grouped.abs().amax(dim=-1, keepdim=True)
        return (amax / nac.FP4_E2M1_MAX * g).clamp(
            max=nac.FP8_E4M3_MAX / 2).to(torch.float8_e4m3fn)

    monkeypatch.setattr(nac, "nvfp4_group_stored_scale", half_ceiling)
    with pytest.raises(trc.TesseraContractError) as raised:
        _attest()
    assert "block_scale_overflow" in str(raised.value)
    assert "the runtime stored 0x7e" in str(raised.value)


# --- malformed input ---------------------------------------------------------

@pytest.mark.parametrize("bad,match", [
    ("0x3F800000", "lowercase hex"), ("0x3f80", "8 lowercase hex")])
def test_a_malformed_global_scale_is_refused(bad, match):
    payload = _payload()
    _vectors(payload)[0]["global_scale"] = bad
    with pytest.raises(trc.TesseraContractError, match=match):
        _table(payload)


def test_a_malformed_input_bit_pattern_is_refused():
    payload = _payload()
    _vectors(payload)[0]["input"][0] = "0x40C0"
    with pytest.raises(trc.TesseraContractError, match="4 lowercase hex"):
        _table(payload)


def test_a_short_group_is_refused():
    payload = _payload()
    _vectors(payload)[0]["codes"].pop()
    with pytest.raises(trc.TesseraContractError, match="exactly unit_length"):
        _table(payload)


@pytest.mark.parametrize("value", (-1, 16, 2.0, "7"))
def test_a_code_outside_the_grid_is_refused(value):
    payload = _payload()
    _vectors(payload)[0]["codes"][0] = value
    with pytest.raises(trc.TesseraContractError, match="integers 0..15"):
        _table(payload)


def test_a_stored_scale_outside_a_byte_is_refused():
    payload = _payload()
    _vectors(payload)[0]["stored_scale"] = 256
    with pytest.raises(trc.TesseraContractError, match="integer byte"):
        _table(payload)


def test_a_missing_vector_member_is_refused():
    payload = _payload()
    del _vectors(payload)[0]["boundary"]
    with pytest.raises(trc.TesseraContractError, match="must publish exactly"):
        _table(payload)


# --- one mechanism -----------------------------------------------------------

def test_the_encoder_and_the_qdq_are_one_mechanism():
    """The codes the preflight attests decode to the values the panel prices."""
    from prismaquant.nvfp4_activation_contract import (
        FP4_GROUP_SIZE, _E2M1_POSITIVE, nvfp4_activation_qdq_served,
        nvfp4_e2m1_code, nvfp4_e2m1_normalize, nvfp4_group_stored_scale)

    torch.manual_seed(574)
    x = torch.randn(4, 2 * FP4_GROUP_SIZE, dtype=torch.float32) * 3.0
    g = 2.0
    qdq = nvfp4_activation_qdq_served(x, g)

    grouped = x.reshape(-1, x.shape[-1] // FP4_GROUP_SIZE, FP4_GROUP_SIZE)
    used = nvfp4_group_stored_scale(grouped, g).float() / g
    codes = nvfp4_e2m1_code(nvfp4_e2m1_normalize(grouped, used))
    magnitudes = torch.tensor(_E2M1_POSITIVE, dtype=torch.float32)
    decoded = (magnitudes[codes & 7]
               * torch.where(codes >= 8, -1.0, 1.0) * used)
    assert torch.equal(decoded.reshape(x.shape), qdq)


def _saturates(vector):
    return any(magnitude > 6.0 for magnitude in _magnitudes(vector))


def _reaches(vector, midpoint):
    return any(magnitude == midpoint for magnitude in _magnitudes(vector))


def _magnitudes(vector):
    """|x| / used_scale for each element, or nothing for a zeroed group."""
    stored = float(torch.tensor([vector["stored_scale"]], dtype=torch.uint8)
                   .view(torch.float8_e4m3fn).float()[0])
    if stored == 0.0:
        return
    bits = int(vector["global_scale"], 16)
    g = float(torch.tensor([bits - (1 << 32) if bits >> 31 else bits],
                           dtype=torch.int32).view(torch.float32)[0])
    used = stored / g
    for text in vector["input"]:
        bits = int(text, 16)
        value = float(torch.tensor([bits - (1 << 16) if bits >> 15 else bits],
                                   dtype=torch.int16).view(torch.bfloat16)[0])
        yield abs(value) / used
