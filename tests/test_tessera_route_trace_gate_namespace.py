"""The route.trace gate on GLM-5.3's served module names (RobTand/prismaquant#1490).

A GLM artifact's ``config_groups`` targets are checkpoint names
(``model.language_model.layers.N.…``). Tessera's trace records each layer's
``prefix``, which is the name vLLM built the module under: the body is
``language_model.model.layers.N.…`` and the MTP draft layer is the bare
``model.layers.45.…``. The gate maps every priced target through the model
profile (``ModelProfile.served_module_name``) before the per-module
comparison.

Fixtures (``tests/fixtures/tessera_route_trace_1490``, see ``SOURCE.md``):
``measured/`` is what the U4 BAL TP2 serves wrote, header-trimmed; ``named/``
is SYNTHETIC -- the same traces with the 29 NVFP4 routed stacks named as
Tessera #680 names them, since these serves predate #680 and traced them
unnamed.

Fail-before on ``0ff167c1fad`` (origin/main): the gate compared the names
verbatim, so ``named/mtp-rank*.json`` against the BAL config was REFUSED with
"the price names no such module" for every served module
(``test_without_a_profile_map_the_names_are_compared_verbatim`` keeps that
behaviour visible for a config that resolves no profile map).
"""
from __future__ import annotations

import copy
import json
import pathlib

import pytest

from prismaquant import tessera_route_trace_gate as gate
from prismaquant.model_profiles.glm5_next import Glm5NextProfile

from test_tessera_route_trace_gate import _contract, _exact_config, _exact_traces

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "tessera_route_trace_1490"

NVFP4 = "e2m1_group16_ue4m3_static"
BF16 = "bf16_unquantized"
NVFP4_MOE = f"TESSERA_NVFP4/moe/{NVFP4}"
BF16_MOE = f"TESSERA_BF16/moe/{BF16}"

DRAFT_TARGET = "model.language_model.layers.45.mlp.experts"
#: What the measured MTP serve traced for the draft experts (not synthetic).
DRAFT_SERVED = "model.layers.45.mlp.experts"


def _config():
    return json.loads((FIXTURE / "config.json").read_text())


def _traces(kind, serve):
    return [(f"rank{rank}", json.loads((FIXTURE / kind / f"{serve}-rank{rank}.json").read_text()))
            for rank in range(2)]


def _compare(traces, config=None):
    executes, formats = _contract()
    return gate.compare_route_traces(
        traces, expected_ranks=2, config=_config() if config is None else config,
        platform="sm_121", executes_by_platform=executes, formats=formats)


def _body_scope(config):
    """The priced side minus the MTP draft's targets, which a non-speculative
    serve never dispatches. This mirrors the U4 wrapper's ``body_scope``
    diagnostic (``pact/u4/u4_route_trace.py --unserved``); the gate itself has
    no such scope, so the full config against a non-spec serve stays refused."""
    scoped = copy.deepcopy(config)
    draft = Glm5NextProfile.mtp_draft_layer_range(config)
    groups = scoped["quantization_config"]["config_groups"]
    for name in list(groups):
        keep = [target for target in groups[name]["targets"]
                if not (target.startswith("model.language_model.layers.")
                        and int(target.split(".")[3]) in draft)]
        if keep:
            groups[name]["targets"] = keep
        else:
            del groups[name]
    return scoped


def _names(trace):
    return {name for entry in trace["entries"] for name in entry["module_names"]}


def _rewrite(traces, change):
    """Apply ``change(module_names) -> module_names`` to every entry of every rank."""
    out = []
    for label, trace in traces:
        trace = copy.deepcopy(trace)
        for entry in trace["entries"]:
            names = change(list(entry["module_names"]))
            entry["modules"] -= len(entry["module_names"]) - len(names)
            entry["module_names"] = sorted(names)
        out.append((label, trace))
    return out


# ---------------------------------------------------------------------------
# The fixture is the measured serve plus names, and nothing else
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("serve", ["mtp", "tr3"])
def test_the_named_fixture_differs_from_the_measured_serve_only_in_the_filled_names(serve):
    for (_, measured), (_, named) in zip(_traces("measured", serve), _traces("named", serve)):
        assert {k: v for k, v in measured.items() if k != "entries"} == {
            k: v for k, v in named.items() if k != "entries"}
        assert len(measured["entries"]) == len(named["entries"])
        filled = set()
        for before, after in zip(measured["entries"], named["entries"]):
            if not before["unnamed_modules"]:
                assert after == before
                continue
            # The only unnamed entries the serves wrote are the NVFP4 routed stacks.
            assert (before["policy"], before["kind"], before["contract"]) == (
                "TESSERA_NVFP4:resident", "moe", NVFP4)
            assert before["module_names"] == [] and before["unnamed_modules"] == 29
            identity = ("module_names", "unnamed_modules", "dispatches_without_prefix")
            assert {k: v for k, v in after.items() if k not in identity} == {
                k: v for k, v in before.items() if k not in identity}
            assert after["unnamed_modules"] == after["dispatches_without_prefix"] == 0
            assert len(after["module_names"]) == after["modules"] == 29
            filled.update(after["module_names"])
        assert len(filled) == 29


def test_every_name_the_serve_wrote_is_a_priced_target_in_the_profiles_namespace():
    """The real evidence for the map: the measured, non-synthetic names."""
    profile = Glm5NextProfile()
    config = _config()
    executes, formats = _contract()
    owners = gate.priced_histogram(config, platform="sm_121",
                                   executes_by_platform=executes, formats=formats)["owners"]
    mapped = {profile.served_module_name(target, config) for target in owners}
    for _, trace in _traces("measured", "mtp"):
        served = _names(trace)
        assert DRAFT_SERVED in served
        assert served <= mapped
        # 133 priced; the 29 NVFP4 routed stacks are the ones the serve left unnamed.
        assert len(mapped - served) == 29


# ---------------------------------------------------------------------------
# (a) (b) the MTP k=1 serve
# ---------------------------------------------------------------------------
def test_the_named_mtp_serve_agrees_per_module_on_both_ranks():
    verdict = _compare(_traces("named", "mtp"))
    assert verdict["status"] == gate.AGREE, verdict["detail"]
    assert verdict["exact_module_qualified"] is True
    assert verdict["granularity"] == gate.EXACT_GRANULARITY
    renamed = verdict["served_namespace"]["renamed"]
    assert verdict["served_namespace"]["profile"] == "glm5_next"
    assert len(verdict["priced_owners"]) == len(renamed) == 133
    assert renamed[DRAFT_TARGET] == DRAFT_SERVED
    in_serve = {renamed[target]: key for target, key in verdict["priced_owners"].items()}
    assert verdict["served_modules"]["rank0"] == verdict["served_modules"]["rank1"] == in_serve
    assert in_serve[DRAFT_SERVED] == BF16_MOE
    assert verdict["served"][NVFP4_MOE] == 29
    assert "named in the serve's namespace by the glm5_next profile" in verdict["detail"]


def test_the_measured_mtp_serve_with_unnamed_nvfp4_stacks_stays_not_verified():
    verdict = _compare(_traces("measured", "mtp"))
    assert verdict["status"] == gate.NOT_VERIFIED, verdict["detail"]
    assert "no stable prefix" in verdict["detail"]
    assert "a count is not a name" in verdict["detail"]
    assert verdict["exact_module_qualified"] is False


# ---------------------------------------------------------------------------
# (c) refusals keep naming the module
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dropped,target", [
    pytest.param("language_model.model.layers.10.mlp.experts",
                 "model.language_model.layers.10.mlp.experts", id="body"),
    pytest.param(DRAFT_SERVED, DRAFT_TARGET, id="draft"),
])
def test_a_served_module_that_is_missing_is_refused_by_name(dropped, target):
    traces = _rewrite(_traces("named", "mtp"),
                      lambda names: [name for name in names if name != dropped])
    verdict = _compare(traces)
    assert verdict["status"] == gate.REFUSED, verdict["detail"]
    assert (f"{dropped} (priced as {target}): priced {BF16_MOE}, served by no module"
            in verdict["detail"]), verdict["detail"]
    assert verdict["exact_module_qualified"] is False


def test_the_draft_module_traced_in_the_body_namespace_is_refused():
    wrong = "language_model.model.layers.45.mlp.experts"
    traces = _rewrite(_traces("named", "mtp"),
                      lambda names: [wrong if name == DRAFT_SERVED else name for name in names])
    verdict = _compare(traces)
    assert verdict["status"] == gate.REFUSED, verdict["detail"]
    assert (f"{wrong}: served {BF16_MOE} but the price names no such module"
            in verdict["detail"]), verdict["detail"]
    assert (f"{DRAFT_SERVED} (priced as {DRAFT_TARGET}): priced {BF16_MOE}, "
            "served by no module" in verdict["detail"]), verdict["detail"]


# ---------------------------------------------------------------------------
# (d) the non-speculative serve
# ---------------------------------------------------------------------------
def test_the_named_tr3_serve_agrees_on_the_body_scope_without_the_draft():
    config = _config()
    scoped = _body_scope(config)
    assert DRAFT_TARGET not in json.dumps(scoped["quantization_config"])
    verdict = _compare(_traces("named", "tr3"), config=scoped)
    assert verdict["status"] == gate.AGREE, verdict["detail"]
    assert verdict["exact_module_qualified"] is True
    assert len(verdict["priced_owners"]) == 132
    assert DRAFT_SERVED not in verdict["served_modules"]["rank0"]


def test_the_full_price_against_a_non_speculative_serve_is_still_refused():
    """The draft is priced and a non-spec serve never dispatches it: not a pass."""
    verdict = _compare(_traces("named", "tr3"))
    assert verdict["status"] == gate.REFUSED, verdict["detail"]
    assert (f"{DRAFT_SERVED} (priced as {DRAFT_TARGET}): priced {BF16_MOE}, "
            "served by no module" in verdict["detail"]), verdict["detail"]


def test_the_measured_tr3_serve_stays_not_verified():
    verdict = _compare(_traces("measured", "tr3"), config=_body_scope(_config()))
    assert verdict["status"] == gate.NOT_VERIFIED, verdict["detail"]
    assert "no stable prefix" in verdict["detail"]


# ---------------------------------------------------------------------------
# A profile with no map, and a map that cannot be applied
# ---------------------------------------------------------------------------
def test_without_a_profile_map_the_names_are_compared_verbatim():
    """The fail-before behaviour, still what a config that resolves no map gets."""
    config = _config()
    del config["model_type"], config["architectures"]
    verdict = _compare(_traces("named", "mtp"), config=config)
    assert verdict["status"] == gate.REFUSED, verdict["detail"]
    assert "served_namespace" not in verdict
    assert "but the price names no such module" in verdict["detail"]


def test_a_profile_with_no_map_leaves_the_verdict_unchanged():
    verdict = gate.compare_route_traces(
        _exact_traces(), expected_ranks=2, config=_exact_config(), platform="sm_121",
        executes_by_platform=_contract()[0], formats=_contract()[1])
    assert verdict["status"] == gate.AGREE, verdict["detail"]
    assert "served_namespace" not in verdict
    assert verdict["served_modules"]["rank0"] == verdict["priced_owners"]
    assert "profile" not in verdict["detail"]


def test_a_draft_range_the_config_does_not_state_is_refused_on_the_price():
    config = _config()
    del config["text_config"]["num_nextn_predict_layers"]
    with pytest.raises(gate.TesseraRouteTraceError, match="num_nextn_predict_layers"):
        _compare(_traces("named", "mtp"), config=config)


def test_two_targets_on_one_served_name_are_refused(monkeypatch):
    monkeypatch.setattr(Glm5NextProfile, "served_module_name",
                        lambda self, name, config: "language_model.model.layers.0.mlp.experts")
    with pytest.raises(gate.TesseraRouteTraceError, match="both map to the served module"):
        _compare(_traces("named", "mtp"))
