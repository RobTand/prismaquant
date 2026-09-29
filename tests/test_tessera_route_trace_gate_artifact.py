"""The route.trace gate judged per ARTIFACT, against the union of its serve phases.

A non-speculative serve never dispatches an MTP draft layer, so judging each
phase against the whole price can never pass an MTP artifact.  The artifact
gate takes the phases the caller CLAIMS, requires every priced module to be
served in at least one of them and every served module to be priced with the
same contract in every phase that dispatches it.  Fixtures are cut from the
real A8 traces (`tests/fixtures/tessera_route_trace_union/PROVENANCE.md`).
"""
import copy
import json
import pathlib

import pytest

from prismaquant import tessera_route_trace_gate as gate
from prismaquant.lane_spec import load_lane_spec

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "tessera_route_trace_union"
GRIDS = {"TESSERA_E2M1_K2": "E2M1x2", "TESSERA_E4M3_K1": "E4M3", "TESSERA_BF16_K1": "BF16"}
# The attested draft namespace: glm5_next.served_module_name maps the MTP layer here.
MTP_SERVED = "model.layers.45.mlp.experts"
LAYER10_SERVED = "language_model.model.layers.10.mlp.experts"


def _contract():
    spec = load_lane_spec("tessera")
    executes = {p: dict(e) for p, e in spec.served_activation_quantization.executes_by_platform.items()}
    return executes, {f: {"family": f, "grid": g} for f, g in GRIDS.items()}


def _load(name):
    return json.loads((FIXTURE / name).read_text())


def _phase(stem, mutate=None):
    """Both ranks of one phase; ``mutate(trace)`` edits each rank's copy."""
    traces = []
    for rank in (0, 1):
        trace = _load(f"{stem}-rank{rank}.json")
        if mutate is not None:
            mutate(trace)
        traces.append((f"rank{rank}", trace))
    return traces


def _judge(phases, *, config=None, expected_ranks=2):
    executes, formats = _contract()
    return gate.compare_artifact_route_traces(
        phases, expected_ranks=expected_ranks, config=config or _load("config.json"),
        platform="sm_121", executes_by_platform=executes, formats=formats)


def _rename_layer45_contract(trace, contract):
    """Serve the draft layer on the layer-10 routed-MoE FP8 contract.

    Folds the draft module into the FP8 routed-MoE entry of the same token
    count (a second entry with that counter key would be a duplicate) and drops
    the draft layer's own BF16 entry.
    """
    for entry in [e for e in trace["entries"] if MTP_SERVED in e["module_names"]]:
        trace["entries"].remove(entry)
        host = next(e for e in trace["entries"]
                    if e["kind"] == "moe" and e["shape"] == entry["shape"]
                    and LAYER10_SERVED in e["module_names"])
        assert host["contract"] == contract
        host["module_names"].append(MTP_SERVED)
        host["modules"] = len(host["module_names"])


def _serve_layer10_as_bf16(trace):
    """The routed experts of layer 10 dispatched on the BF16 contract."""
    for entry in trace["entries"]:
        if LAYER10_SERVED in entry["module_names"]:
            entry["contract"] = "bf16_unquantized"


def _add_unpriced_module(trace):
    extra = "language_model.model.layers.46.mlp.experts"
    for entry in [e for e in trace["entries"] if MTP_SERVED in e["module_names"]]:
        entry["module_names"] = [extra]


# ---------------------------------------------------------------------------
# The three outcomes the MTP artifact needs
# ---------------------------------------------------------------------------
def test_the_mtp_artifact_agrees_when_the_speculative_phase_serves_the_draft_layer():
    verdict = _judge({"tr3": _phase("nonspec"), "latency": _phase("nonspec"), "2c": _phase("spec")})
    assert verdict["status"] == gate.AGREE, verdict["detail"]
    assert verdict["exact_module_qualified"] is True
    assert set(verdict["phase_names"]) == {"tr3", "latency", "2c"}
    # The draft layer is the one module a non-speculative phase cannot serve.
    # module -> the claimed phases that did not dispatch it.
    assert verdict["not_served_in_every_phase"] == {MTP_SERVED: ["latency", "tr3"]}


def test_a_claimed_speculative_phase_with_an_empty_trace_is_not_verified():
    verdict = _judge({"tr3": _phase("nonspec"), "2c": _phase("empty")})
    assert verdict["status"] == gate.NOT_VERIFIED
    assert "2c" in verdict["detail"]
    assert verdict["exact_module_qualified"] is False


def test_the_draft_layer_served_on_another_contract_is_refused():
    wrong = _phase("spec", lambda t: _rename_layer45_contract(t, "fp8_per_token_dynamic"))
    verdict = _judge({"tr3": _phase("nonspec"), "2c": wrong})
    assert verdict["status"] == gate.REFUSED
    assert "2c" in verdict["detail"]
    assert "priced TESSERA_BF16/moe/bf16_unquantized" in verdict["detail"]
    assert "served TESSERA_FP8/moe/fp8_per_token_dynamic" in verdict["detail"]


# ---------------------------------------------------------------------------
# The union rule, both directions
# ---------------------------------------------------------------------------
def test_a_priced_module_served_in_no_claimed_phase_is_refused():
    verdict = _judge({"tr3": _phase("nonspec"), "latency": _phase("nonspec")})
    assert verdict["status"] == gate.REFUSED
    assert "layers.45.mlp.experts" in verdict["detail"]
    assert "served by no phase" in verdict["detail"]


def test_a_served_module_the_price_does_not_name_is_refused():
    verdict = _judge({"tr3": _phase("nonspec"), "2c": _phase("spec", _add_unpriced_module)})
    assert verdict["status"] == gate.REFUSED
    assert "layers.46.mlp.experts" in verdict["detail"]
    assert "the price names no such module" in verdict["detail"]


def test_a_contract_mismatch_in_one_of_two_phases_is_refused_and_names_the_phase():
    verdict = _judge({
        "tr3": _phase("nonspec"),
        "latency": _phase("nonspec", _serve_layer10_as_bf16),
        "2c": _phase("spec"),
    })
    assert verdict["status"] == gate.REFUSED
    assert "latency" in verdict["detail"]
    assert "layers.10.mlp.experts" in verdict["detail"]


def test_a_module_served_on_two_contracts_in_two_phases_is_refused():
    # Each phase alone is self-consistent for layer 10; across phases they differ.
    verdict = _judge({
        "tr3": _phase("nonspec"),
        "2c": _phase("spec", _serve_layer10_as_bf16),
    })
    assert verdict["status"] == gate.REFUSED
    assert "2c" in verdict["detail"]


def test_a_refusal_outranks_a_missing_phase():
    wrong = _phase("nonspec", _serve_layer10_as_bf16)
    verdict = _judge({"tr3": wrong, "2c": _phase("empty")})
    assert verdict["status"] == gate.REFUSED


# ---------------------------------------------------------------------------
# Missing input is not verified
# ---------------------------------------------------------------------------
def test_no_claimed_phase_is_not_verified():
    assert _judge({})["status"] == gate.NOT_VERIFIED


def test_a_claimed_phase_with_a_missing_rank_payload_is_not_verified():
    phases = {"tr3": _phase("nonspec"), "2c": [("rank0", None), ("rank1", None)]}
    verdict = _judge(phases)
    assert verdict["status"] == gate.NOT_VERIFIED
    assert "2c" in verdict["detail"]


def test_a_claimed_phase_with_one_rank_short_is_not_verified():
    phases = {"tr3": _phase("nonspec"), "2c": _phase("spec")[:1]}
    assert _judge(phases)["status"] == gate.NOT_VERIFIED


def test_phases_are_never_inferred_from_a_bare_list_of_traces():
    with pytest.raises(gate.TesseraRouteTraceError, match="phases"):
        _judge(_phase("spec"))


def test_a_phase_with_no_traces_at_all_is_not_verified():
    assert _judge({"tr3": _phase("nonspec"), "2c": []})["status"] == gate.NOT_VERIFIED


# ---------------------------------------------------------------------------
# Per-phase verdicts stay, labelled diagnostic
# ---------------------------------------------------------------------------
def test_per_phase_verdicts_are_kept_and_labelled_diagnostic():
    verdict = _judge({"tr3": _phase("nonspec"), "2c": _phase("spec")})
    assert verdict["status"] == gate.AGREE
    phases = verdict["phases"]
    assert set(phases) == {"tr3", "2c"}
    assert all(p["diagnostic"] is True for p in phases.values())
    # tr3 alone cannot pass the whole price (the draft layer is never served
    # there): that is exactly why the phase verdict is diagnostic, not the gate.
    assert phases["tr3"]["status"] == gate.REFUSED
    assert phases["2c"]["status"] == gate.AGREE
    assert verdict["granularity"] == gate.EXACT_GRANULARITY


def test_the_artifact_verdict_carries_the_union_it_judged():
    verdict = _judge({"tr3": _phase("nonspec"), "2c": _phase("spec")})
    assert MTP_SERVED in verdict["served_union"]
    assert len(verdict["served_union"]) == 6
    assert verdict["schema"] == gate.ARTIFACT_VERDICT_SCHEMA


def test_a_single_phase_covering_the_whole_price_is_the_old_verdict():
    """One phase claimed, serving everything priced: same outcome as the per-serve gate."""
    verdict = _judge({"2c": _phase("spec")})
    executes, formats = _contract()
    single = gate.compare_route_traces(
        _phase("spec"), expected_ranks=2, config=_load("config.json"),
        platform="sm_121", executes_by_platform=executes, formats=formats)
    assert single["status"] == gate.AGREE
    assert verdict["status"] == gate.AGREE


# ---------------------------------------------------------------------------
# The sample helper's (shown, rest) shape at the artifact gate (#1712)
# ---------------------------------------------------------------------------
def test_a_fully_observed_agreement_names_the_late_module_in_its_detail():
    verdict = _judge({"tr3": _phase("nonspec"), "2c": _phase("spec")})
    assert verdict["status"] == gate.AGREE, verdict["detail"]
    assert verdict["detail"].endswith(
        f"; 1 module(s) not dispatched in every phase: {MTP_SERVED}")


def test_an_overflowing_unserved_list_counts_the_rest_and_looks_up_no_module(monkeypatch):
    real = gate._sample
    monkeypatch.setattr(gate, "_sample", lambda names, *, limit=8: real(names, limit=0))
    verdict = _judge({"tr3": _phase("nonspec")})
    assert verdict["status"] == gate.REFUSED
    assert verdict["detail"].endswith("(1 further module(s) not shown)")
    assert "served by no phase" not in verdict["detail"]


def test_the_inputs_are_not_mutated():
    phases = {"tr3": _phase("nonspec"), "2c": _phase("spec")}
    before = copy.deepcopy(phases)
    _judge(phases)
    assert phases == before
