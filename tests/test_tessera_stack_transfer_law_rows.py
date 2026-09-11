"""Provenance on a two-tier stack: what a model-predicted row may and may not say.

RobTand/prismaquant#495 part 3.  Under the schedule
``docs/results/glm_tessera_probe_reduction_regret_2026-09-10.md`` section 5
proposes, a routed stack is censused at one rung and sampled at the other two;
the sampled rungs are then PREDICTED for every expert by the transfer law.

Three claims are pinned here, and the third is the one that has teeth:

* the censused rung is a measurement and says so
  (``cost_source: tessera_campaign_measured``);
* a predicted rung says it is a prediction
  (``PROVENANCE_INTERPOLATED`` + ``cost_source: tessera_campaign_interpolated``)
  and carries the whole law it was predicted by;
* a predicted rung carries NEITHER ``dloss_stderr`` NOR
  ``estimator: horvitz_thompson``.  Those describe the variance of a sampling
  design over repeated draws.  A regression's output has no such variance, and
  stamping one on it would let a reader -- or a UCB hedge that later learns to
  read the field -- treat a model error as a measured spread.  The old code
  built an interpolated row's ``sampled_experts`` block by subtracting
  ``members`` from the measured row's, which carried ``estimator`` straight
  onto it; the block is now built from an allowlist.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

MODULE_A = "model.layers.7.feed_forward.experts"
MODULE_B = "model.layers.8.feed_forward.experts"
#: The campaign's three rungs.  960 is censused; 832 and 1088 are predicted.
REFERENCE = 960
TARGETS = (832, 1088)
#: The transfer-law draw: the experts that were also encoded at 832 and 1088.
LAW_EXPERTS = (0, 2, 5, 7)
EXPERTS = 12


def _campaign():
    return pytest.importorskip("prismaquant.tessera_campaign")


def _profile():
    from prismaquant.model_profiles.lfm2_moe import Lfm2MoeProfile
    return Lfm2MoeProfile()


def _probe_row(module, per_expert):
    return {
        "h_trace": float(math.fsum(per_expert)),
        "h_trace_per_expert": [float(v) for v in per_expert],
        "num_experts": len(per_expert),
        "_packed_experts_module": module,
        "_packed_param": "gate_up_proj",
        "out_features": 8, "in_features": 4,
        "n_params": len(per_expert) * 32,
        "router_path": None, "expert_id": None,
    }


def _anchor(campaign, qname, q256, dloss):
    return campaign.CampaignAnchor(
        qname=qname, family="TESSERA_E2M1_K1",
        format_name=f"TESSERA_E2M1_K1_R{q256}", body_rate_q256=q256,
        dloss=float(dloss), dloss_stderr=0.0, memory_bytes=1024,
        bits_per_param=4.0, activation_contract="bfloat16",
        activation_quantized=False, wire_bytes=1100, seconds=1.0,
        hessian_applied=True, input_global_scale=None)


#: The law the fixture's experts obey exactly, per (rung, projection).  Exact
#: rather than noisy so the assertions are about the WIRING -- which rows get
#: written, with which provenance -- and not about a fit's luck; the fit's own
#: statistics are tested in ``test_tessera_stack_transfer_law.py``.
PLANTED = {832: {"w1": (1.05, 1.30), "w3": (0.95, 1.45)},
           1088: {"w1": (0.96, -1.20), "w3": (1.04, -1.35)}}


def _stack(campaign, module, *, offset, anchors):
    """One two-tier stack: a 960 census, plus 832/1088 on the law draw only."""
    h = [1.0 + 0.3 * expert for expert in range(EXPERTS)]
    reference = {}
    for expert in range(EXPERTS):
        for index, role in enumerate(("w1", "w3")):
            qname = f"{module}.{expert}.{role}"
            value = 2.0 ** (-8.0 + offset + 0.21 * expert + 0.37 * index)
            reference[(expert, role)] = value
            anchors[qname] = {"TESSERA_E2M1_K1": [
                _anchor(campaign, qname, REFERENCE, value)]}
    for expert in LAW_EXPERTS:
        for role in ("w1", "w3"):
            qname = f"{module}.{expert}.{role}"
            for rung in TARGETS:
                slope, intercept = PLANTED[rung][role]
                anchors[qname]["TESSERA_E2M1_K1"].append(_anchor(
                    campaign, qname, rung,
                    2.0 ** (intercept
                            + slope * math.log2(reference[(expert, role)]))))
    sample = campaign.stack_sample_from_probe(
        f"{module}.gate_up_proj", _probe_row(module, h), _profile(),
        sampled_experts=range(EXPERTS),
        inclusion_prob={e: 1.0 for e in range(EXPERTS)}, seed=11,
        design="census", transfer_law_experts=LAW_EXPERTS)
    return sample, reference


@pytest.fixture
def payload():
    """Two two-tier stacks, so each can be predicted by a law that held it out."""
    campaign = _campaign()
    anchors: dict = {}
    samples = {}
    for index, module in enumerate((MODULE_A, MODULE_B)):
        sample, _ = _stack(campaign, module, offset=1.4 * index, anchors=anchors)
        samples[sample.packed_qname] = sample
    built = campaign.campaign_cost_payload(
        anchors, {}, loo={},
        provenance={"provenance": {"hessian": {"supplied": True}}},
        stack_samples=samples)
    return built


def _rows(payload, module):
    return payload["costs"][f"{module}.gate_up_proj"]


def test_the_censused_rung_is_a_measurement(payload):
    """960 covers every expert with certainty, so it is a census, not a sample."""
    row = _rows(payload, MODULE_A)[f"TESSERA_E2M1_K1_R{REFERENCE}"]
    assert row["cost_source"] == "tessera_campaign_measured"
    assert row["output_mse_measured"] is True
    assert row["tessera_provenance"] == "measured"
    assert row["tessera_body_rate_q256"] == REFERENCE
    # A census has an exactly-zero sampling error, and says which estimator
    # produced it -- that is legitimate here, and only here.
    assert row["dloss_stderr"] == 0.0
    assert row["sampled_experts"]["estimator"] == "horvitz_thompson"
    assert "transfer_law" not in row


def test_a_drawn_rung_keeps_the_sampled_cost_source():
    """A one-tier draw is an estimate and must not borrow the census spelling."""
    campaign = _campaign()
    h = [1.0 + expert for expert in range(4)]
    anchors = {}
    for expert in range(4):
        for role in ("w1", "w3"):
            qname = f"{MODULE_A}.{expert}.{role}"
            anchors[qname] = {"TESSERA_E2M1_K1": [
                _anchor(campaign, qname, REFERENCE, 0.01 * (expert + 1))]}
    sample = campaign.stack_sample_from_probe(
        f"{MODULE_A}.gate_up_proj", _probe_row(MODULE_A, h), _profile(),
        sampled_experts=[0, 1, 2, 3],
        inclusion_prob={0: 1.0, 1: 0.5, 2: 0.5, 3: 1.0}, seed=3)
    built = campaign.campaign_cost_payload(
        anchors, {}, loo={},
        provenance={"provenance": {"hessian": {"supplied": True}}},
        stack_samples={sample.packed_qname: sample})
    row = built["costs"][f"{MODULE_A}.gate_up_proj"][
        f"TESSERA_E2M1_K1_R{REFERENCE}"]
    assert row["cost_source"] == "tessera_campaign_measured_stack_sample"


@pytest.mark.parametrize("rung", TARGETS)
def test_a_predicted_rung_declares_the_law_it_was_predicted_by(payload, rung):
    """The whole law travels on the row: slopes, intercepts, ids, n, spread."""
    from prismaquant.allocator_candidates import TESSERA_INTERPOLATED_COST_SOURCE
    from prismaquant.tessera_rate_surface import PROVENANCE_INTERPOLATED

    row = _rows(payload, MODULE_A)[f"TESSERA_E2M1_K1_R{rung}"]
    assert row["cost_source"] == TESSERA_INTERPOLATED_COST_SOURCE
    assert row["tessera_provenance"] == PROVENANCE_INTERPOLATED
    assert row["output_mse_measured"] is False

    law = row["transfer_law"]
    assert law["predicted_q256"] == rung
    assert law["reference_q256"] == REFERENCE
    assert law["held_out"] == f"{MODULE_A}.gate_up_proj"
    assert law["pooled_stacks"] == [f"{MODULE_B}.gate_up_proj"]
    assert law["n"] == len(LAW_EXPERTS)
    assert law["sample_experts"] == list(LAW_EXPERTS)
    assert set(law["slope"][str(rung)]) == {"w1", "w3"}
    assert set(law["intercept"]) == {"w1", "w3"}
    assert set(law["residual_sd_log2"][str(rung)]) == {"w1", "w3"}
    assert law["model_error"]["kind"] == "transfer_law_model_error_log2"
    assert law["model_error"]["is_sampling_error"] is False
    # The fixture's experts obey the planted law exactly, so the pooled slope
    # is that law's and the prediction is the truth to floating point.
    for role in ("w1", "w3"):
        assert law["slope"][str(rung)][role] == pytest.approx(
            PLANTED[rung][role][0], abs=1e-9)


@pytest.mark.parametrize("rung", TARGETS)
def test_a_predicted_rung_carries_no_sampling_estimator(payload, rung):
    """The refusal this whole part exists for.

    ``dloss_stderr`` and ``estimator: horvitz_thompson`` describe a draw.  This
    value came from a regression over every expert's own censused reference,
    not from a draw standing in for the stack, so neither field may appear --
    at the top level or anywhere inside the block that describes the draw.
    """
    row = _rows(payload, MODULE_A)[f"TESSERA_E2M1_K1_R{rung}"]
    assert "dloss_stderr" not in row
    assert "dloss_stderr_currency" not in row
    assert "estimator" not in row
    block = row["sampled_experts"]
    assert "estimator" not in block
    assert "variance_estimator" not in block
    assert "dloss_stderr" not in block
    assert "n_random_stratum" not in block
    # The draw is still described -- which experts, under what probability --
    # because the intercept was fitted on them.
    assert block["transfer_law_experts"] == list(LAW_EXPERTS)
    assert block["design"] == "census"
    assert set(block["inclusion_prob"]) == set(range(EXPERTS))


def test_the_interpolated_surface_row_also_carries_no_estimator():
    """The pre-existing stack interpolation path had the same leak.

    Two measured rungs over one draw give a ``TesseraRateSurface``, and its
    interpolated rows used to inherit ``estimator`` by subtraction.  Same rule:
    a value no estimator produced may not name one.
    """
    campaign = _campaign()
    h = [1.0 + expert for expert in range(4)]
    anchors = {}
    for expert in range(4):
        for role in ("w1", "w3"):
            qname = f"{MODULE_A}.{expert}.{role}"
            anchors[qname] = {"TESSERA_E2M1_K1": [
                _anchor(campaign, qname, 832, 0.02 * (expert + 1)),
                _anchor(campaign, qname, 1088, 0.01 * (expert + 1))]}
    sample = campaign.stack_sample_from_probe(
        f"{MODULE_A}.gate_up_proj", _probe_row(MODULE_A, h), _profile(),
        sampled_experts=[0, 1, 2, 3],
        inclusion_prob={0: 1.0, 1: 0.5, 2: 0.5, 3: 1.0}, seed=3)
    # The three fields ``_stack_menu`` and the interpolation loop read; a real
    # ``MenuRung`` would drag a route admission and a TP gate into a test about
    # provenance.
    rungs = [SimpleNamespace(
        family="TESSERA_E2M1_K1", format_name=f"TESSERA_E2M1_K1_R{REFERENCE}",
        body_rate_q256=REFERENCE,
        admission=SimpleNamespace(activation_contract="bfloat16"))]
    built = campaign.campaign_cost_payload(
        anchors, {f"{MODULE_A}.gate_up_proj": rungs}, loo={},
        provenance={"provenance": {"hessian": {"supplied": True}}},
        stack_samples={sample.packed_qname: sample})
    row = built["costs"][f"{MODULE_A}.gate_up_proj"].get(
        f"TESSERA_E2M1_K1_R{REFERENCE}")
    assert row is not None, "the surface must price the bracketed rung"
    assert row["cost_source"] == "tessera_campaign_interpolated"
    assert "dloss_stderr" not in row
    assert "estimator" not in row["sampled_experts"]
    assert "variance_estimator" not in row["sampled_experts"]
