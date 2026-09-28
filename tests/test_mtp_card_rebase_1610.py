"""A whole-artifact card prices the GLM MTP layer at its selected bytes (PQ #1610).

The MTP layer is chosen outside the body knapsack, so its units never enter
the body assignment and the footprint floor used to keep them at SOURCE
bytes. On GLM-5.3 that over-charged the card by 10,837,919,808 B, measured
from safetensors headers of the uniform-R1024 export
``glm53-pact-uniform-arms-20260927/a8/body-mtp-v39/exported-r2``. The pins
below are those header bytes; the allocator tests use a synthetic checkpoint.
"""
from __future__ import annotations

import json
import sys

import pytest

from prismaquant import footprint as fp

# --- exported-r2 and its source, safetensors headers only ------------------
#: GLM-5.3 MTP layer 45: 288 routed experts x 3 projections + 3 shared
#: projections, every one (2048 x 4096) or (4096 x 2048), BF16 source.
GLM_MTP_UNITS = 867
GLM_MTP_UNIT_PARAMS = 4096 * 2048
#: Source header spans of those 867 units (the checkpoint manifest resolves
#: them to exactly params x 2).
GLM_MTP_UNIT_SOURCE_BYTES = 14_545_846_272
#: The whole of layer 45 in the source checkpoint, and in exported-r2.
GLM_MTP_LAYER_SOURCE_BYTES = 14_865_185_408
GLM_MTP_LAYER_EXPORT_BYTES = 4_027_265_600
#: The release's MTP selection: routed TESSERA_BF16_K1_R1024, shared BF16.
GLM_MTP_SELECTED_BYTES = 3_707_898_528
#: The a8 body allocation priced by the pre-fix card, and the export's
#: tensor-data bytes.
A8_CARD_PAYLOAD_BYTES = 184_568_814_456
EXPORT_TENSOR_BYTES = 173_759_669_344
#: What the export carries that no card prices yet (PQ #1609): the
#: TSRFUSE1 framing and the Tessera manifest. Routed, shared and dense-MLP
#: body buckets, then the MTP experts' framing alone (10 + 14 + len(role)
#: per expert projection, since the MTP wires already carry the manifest).
UNPRICED_SIDE_BYTES = {"routed": 28_667_520, "shared": 99_136, "dense_mlp": 8_040,
                       "mtp": 288 * (33 + 33 + 31)}


def _glm_payload():
    units = [f"model.language_model.layers.45.mlp.experts.{e}.{p}"
             for e in range(288) for p in ("gate_proj", "up_proj", "down_proj")]
    units += [f"model.language_model.layers.45.mlp.shared_experts.{p}"
              for p in ("gate_proj", "up_proj", "down_proj")]
    return {"params": {u: GLM_MTP_UNIT_PARAMS for u in units},
            "source_dtype": {u: "bfloat16" for u in units}}


def test_the_card_matches_exported_r2_except_the_unpriced_side_bytes():
    payload = _glm_payload()
    assert len(payload["params"]) == GLM_MTP_UNITS
    rebase = fp.mtp_selection_rebased_bytes(
        A8_CARD_PAYLOAD_BYTES, payload, GLM_MTP_SELECTED_BYTES, context="exported-r2")
    assert rebase["mtp_source_bytes"] == GLM_MTP_UNIT_SOURCE_BYTES
    # The release's stamped selection payload (tools/reselect_mtp_fixed.py).
    assert rebase["total_bytes"] == 173_730_866_712
    # MTP bucket: the non-unit layer-45 tensors ship verbatim, the units ship
    # the selection; only the MTP framing is left.
    mtp_card = (GLM_MTP_LAYER_SOURCE_BYTES - GLM_MTP_UNIT_SOURCE_BYTES
                + GLM_MTP_SELECTED_BYTES)
    assert GLM_MTP_LAYER_EXPORT_BYTES - mtp_card == UNPRICED_SIDE_BYTES["mtp"]
    # Whole artifact: every remaining byte is side bytes #1609 names.
    assert EXPORT_TENSOR_BYTES - rebase["total_bytes"] == sum(UNPRICED_SIDE_BYTES.values())
    # Before the fix the card was 10.8 GB heavy on MTP alone.
    assert (A8_CARD_PAYLOAD_BYTES - EXPORT_TENSOR_BYTES
            == GLM_MTP_UNIT_SOURCE_BYTES - GLM_MTP_SELECTED_BYTES
            - sum(UNPRICED_SIDE_BYTES.values()))


def test_a_manifest_that_disagrees_with_the_payload_is_refused():
    payload = {"params": {"a": 10, "b": 20}, "source_dtype": {"a": "bfloat16", "b": "bfloat16"}}
    assert fp.mtp_selection_rebased_bytes(
        1000, payload, 7, context="t", manifest={"a": 20, "b": 40})["total_bytes"] == 947
    with pytest.raises(ValueError, match="checkpoint holds 62 B"):
        fp.mtp_selection_rebased_bytes(1000, payload, 7, context="t",
                                       manifest={"a": 22, "b": 40})
    with pytest.raises(ValueError, match="not resolvable"):
        fp.mtp_selection_rebased_bytes(1000, payload, 7, context="t", manifest={"a": 20})
    with pytest.raises(ValueError, match="body-assignable"):
        fp.mtp_selection_rebased_bytes(1000, payload, 7, context="t", assigned_names={"b"})
    with pytest.raises(ValueError, match="no per-parameter width"):
        fp.mtp_selection_rebased_bytes(
            1000, {**payload, "source_dtype": {"a": "float8_e4m3fn", "b": "bfloat16"}},
            7, context="t")


# --- through the real allocator ------------------------------------------------

@pytest.fixture
def mtp_case(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from test_allocator_byte_budget_selection import _fixture
    from test_glm_mtp_selection import PARAMS, ROUTED, SHARED, _payload, _write_payload

    from prismaquant import format_registry

    monkeypatch.setattr(format_registry, "format_is_producer_eligible", lambda name, **_: True)
    draft = {f"{name}.weight": ("BF16", (64, 128)) for name in ROUTED + SHARED}
    assert 64 * 128 == PARAMS
    _, probe_p, cost_p, stats = _fixture(tmp_path, nvfp4_dloss=1.0, fp8_dloss=0.5,
                                         extra_tensors=draft)
    payload = _payload()
    cost, constants = _write_payload(tmp_path, payload)
    return tmp_path, probe_p, cost_p, payload, cost, constants


def _run_mtp(monkeypatch, case, *extra):
    from test_allocator_byte_budget_selection import _run

    tmp_path, probe_p, cost_p = case[:3]
    selection, layer_cfg = _run(monkeypatch, tmp_path, probe_p, cost_p, disk_gb=1.0,
                                fmt_for_target=lambda t: "FP8_E4M3", extra_argv=extra)
    return selection, layer_cfg["__prismaquant__"]


def test_the_card_prices_the_selected_mtp_bytes_not_the_source(mtp_case, monkeypatch):
    from test_glm_mtp_selection import R1024, ROUTED, SHARED

    payload, cost, constants = mtp_case[3:]
    plain, plain_meta = _run_mtp(monkeypatch, mtp_case)
    budget = sum(payload["wire_bytes"][n][R1024] for n in ROUTED) + \
        sum(2 * payload["params"][n] for n in SHARED)
    rebased, meta = _run_mtp(
        monkeypatch, mtp_case, "--mtp-joint-cost", str(cost),
        "--mtp-byte-budget", str(budget), "--mtp-serve-constants", str(constants))

    assert meta["mtp_selection"]["resident_bytes"] == budget
    source = sum(2 * payload["params"][n] for n in ROUTED + SHARED)
    assert source > budget
    # The same FP8 body, so the card moves by exactly the MTP swap.
    assert plain["source_total_bytes"] - rebased["source_total_bytes"] == source - budget
    assert (plain_meta["whole_artifact_budget"]["selection_tensor_payload_bytes"]
            - meta["whole_artifact_budget"]["selection_tensor_payload_bytes"]
            == source - budget)


def test_fixed_formats_pin_the_mtp_rung_the_card_charges(mtp_case, monkeypatch):
    from test_glm_mtp_selection import R832, ROUTED, SHARED

    tmp_path, _probe, _cost_p, payload, cost, constants = mtp_case
    fixed = tmp_path / "fixed.json"
    fixed.write_text(json.dumps({"routed": R832, "shared": "BF16"}))
    plain, _ = _run_mtp(monkeypatch, mtp_case)
    got, layer_cfg = _run_mtp(
        monkeypatch, mtp_case, "--mtp-joint-cost", str(cost),
        "--mtp-byte-budget", str(10**9), "--mtp-serve-constants", str(constants),
        "--mtp-fixed-formats", str(fixed))
    record = layer_cfg["__prismaquant__"]["mtp_selection"]
    assert record["rung"] == f"routed={R832}|shared=BF16"
    assert record["fixed_formats"] == {"routed": R832, "shared": "BF16"}
    selected = sum(payload["wire_bytes"][n][R832] for n in ROUTED) + \
        sum(2 * payload["params"][n] for n in SHARED)
    assert record["resident_bytes"] == selected
    source = sum(2 * payload["params"][n] for n in ROUTED + SHARED)
    assert plain["source_total_bytes"] - got["source_total_bytes"] == source - selected
