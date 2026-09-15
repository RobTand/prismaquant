"""Principle 14's serve-side leg on the Tessera lane (RobTand/prismaquant#575).

The priced activation contract and the served one must be the same object.
Tessera's plugin counts every served dispatch when a serve runs with
``TESSERA_ROUTE_TRACE``; ``prismaquant.tessera_route_trace_gate`` compares
those counts with the contracts the artifact's ``config.json`` prices on its
platform, and the lane's ``route.trace`` shipcard slot carries the verdict.

The fixtures are real: ``tests/fixtures/tessera_route_trace_m44e1`` holds the
``config.json`` of ``glm53-4layer-a4-e2m1x2-q896-l2`` and the two rank traces
its TP2 eager serve (m44e1, sparky rank 0 and sparklina rank 1) wrote.

Fail-before on ``667b35a2d4``: ``prismaquant.tessera_route_trace_gate`` did not
exist, a Tessera card owed no ``route.trace`` slot, and ``shipcard_cli`` had no
``fill-route-trace``, so a Tessera card verified with no served route read.
"""
from __future__ import annotations

import copy
import json
import pathlib

import pytest

from prismaquant import shipcard
from prismaquant import tessera_route_trace_gate as gate
from prismaquant.lane_shipcard import open_lane_shipcard
from prismaquant.lane_spec import load_lane_spec
from prismaquant.shipcard_cli import EXIT_NOT_VERIFIED, main as shipcard_cli

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "tessera_route_trace_m44e1"
RANK_FILES = ("routes-rank0-sparky.json", "routes-rank1-sparklina.json")

NVFP4 = "e2m1_group16_ue4m3_static"
FP8 = "fp8_per_token_dynamic"
BF16 = "bf16_unquantized"

#: What the m44e1 serve dispatched on each rank, and what its config prices:
#: one BF16 and one FP8 dense module in layer 0, the two NVFP4 shared-expert
#: projections and the routed experts in layer 1.
EXPECTED = {
    f"TESSERA_BF16/dense/{BF16}": 1,
    f"TESSERA_FP8/dense/{FP8}": 1,
    f"TESSERA_NVFP4/dense/{NVFP4}": 2,
    f"TESSERA_NVFP4/moe/{NVFP4}": 1,
}

#: The pinned contract's ``formats[]`` grids. The unit tests below stand the
#: packaged contract in with these plus the lane spec's contract-derived
#: ``executes_by_platform``; `test_the_packaged_contract_prices_m44e1_as_served`
#: reads the real contract and checks this transcription against it.
GRIDS = {"TESSERA_E2M1_K2": "E2M1x2", "TESSERA_E4M3_K1": "E4M3", "TESSERA_BF16_K1": "BF16"}


def _contract():
    spec = load_lane_spec("tessera")
    executes = {
        platform: dict(entry)
        for platform, entry in spec.served_activation_quantization.executes_by_platform.items()
    }
    formats = {family: {"family": family, "grid": grid} for family, grid in GRIDS.items()}
    return executes, formats


@pytest.fixture
def contract(monkeypatch):
    monkeypatch.setattr(gate, "load_trace_contract", _contract)


def _config():
    return json.loads((FIXTURE / "config.json").read_text())


def _traces():
    return [(f"rank{index}", json.loads((FIXTURE / name).read_text()))
            for index, name in enumerate(RANK_FILES)]


def _compare(traces, *, expected_ranks=2, platform="sm_121", config=None):
    executes, formats = _contract()
    return gate.compare_route_traces(
        traces, expected_ranks=expected_ranks, config=config or _config(),
        platform=platform, executes_by_platform=executes, formats=formats)


def _served_moe_as_fp8(trace):
    """The routed experts served on the FP8 contract instead of NVFP4's."""
    mutated = copy.deepcopy(trace)
    for entry in mutated["entries"]:
        if entry["kind"] == "moe":
            entry["contract"] = FP8
    return mutated


# ---------------------------------------------------------------------------
# The gate, on the real m44e1 traces
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", RANK_FILES)
def test_the_real_m44e1_traces_parse_to_the_served_histogram(name):
    entries = gate.parse_route_trace(
        (FIXTURE / name).read_text(), where=name)
    assert len(entries) == 40
    served = gate.served_histogram(entries, where=name)
    assert served["histogram"] == EXPECTED
    assert served["token_counts"] == [1, 2, 4, 5, 6, 178, 356, 2048]
    assert served["decoders"] == ["native_span2", "torch_materialize_stock", "torch_window"]
    assert "vllm.fused_moe.modular_kernel:FLASHINFER_CUTLASS" in served["symbols"]
    assert served["residencies"] == ["resident"]


def test_the_real_artifact_prices_the_same_histogram():
    executes, formats = _contract()
    priced = gate.priced_histogram(
        _config(), platform="sm_121", executes_by_platform=executes, formats=formats)
    assert priced["histogram"] == EXPECTED
    assert len(priced["owners"]) == 5


def test_agreeing_traces_are_accepted():
    verdict = _compare(_traces())
    assert verdict["status"] == gate.AGREE, verdict["detail"]
    assert verdict["served"] == verdict["priced"] == EXPECTED
    assert verdict["granularity"] == gate.GRANULARITY


def test_a_served_contract_that_differs_from_the_price_is_refused_naming_both():
    traces = [(label, _served_moe_as_fp8(trace)) for label, trace in _traces()]
    verdict = _compare(traces)
    assert verdict["status"] == gate.REFUSED
    assert f"TESSERA_NVFP4/moe/{NVFP4}: priced 1 module(s), served 0" in verdict["detail"]
    assert f"TESSERA_NVFP4/moe/{FP8}: priced 0 module(s), served 1" in verdict["detail"]


def test_one_rank_serving_another_contract_is_refused():
    traces = _traces()
    traces[1] = (traces[1][0], _served_moe_as_fp8(traces[1][1]))
    verdict = _compare(traces)
    assert verdict["status"] == gate.REFUSED
    assert "ranks served different module histograms" in verdict["detail"]


def test_a_module_absent_from_every_token_count_is_refused():
    """A fallback or error dispatch is not counted, so the module vanishes."""
    traces = []
    for label, trace in _traces():
        trace = copy.deepcopy(trace)
        trace["entries"] = [e for e in trace["entries"] if not e["shape"].endswith(":N4096:K1024")]
        traces.append((label, trace))
    verdict = _compare(traces)
    assert verdict["status"] == gate.REFUSED
    assert f"TESSERA_NVFP4/dense/{NVFP4}: priced 2 module(s), served 1" in verdict["detail"]


def test_a_module_absent_at_one_token_count_is_refused():
    traces = []
    for label, trace in _traces():
        trace = copy.deepcopy(trace)
        trace["entries"] = [e for e in trace["entries"] if e["shape"] != "M178:N4096:K1024"]
        traces.append((label, trace))
    verdict = _compare(traces)
    assert verdict["status"] == gate.REFUSED
    assert "differs between token counts" in verdict["detail"]


def _compiled(trace):
    trace = copy.deepcopy(trace)
    trace["entries"][0]["shape"] = "M*:N4096:K6144"
    return trace


def _empty(trace):
    return {**copy.deepcopy(trace), "entries": []}


def _other_schema(trace):
    return {**copy.deepcopy(trace), "schema": "tessera.route_trace/0"}


@pytest.mark.parametrize("damage", [
    pytest.param(lambda traces: [traces[0], (traces[1][0], None)], id="missing-rank-file"),
    pytest.param(lambda traces: traces[:1], id="fewer-traces-than-ranks"),
    pytest.param(lambda traces: [traces[0], (traces[1][0], _compiled(traces[1][1]))], id="compiled"),
    pytest.param(lambda traces: [traces[0], (traces[1][0], _empty(traces[1][1]))], id="empty"),
    pytest.param(lambda traces: [traces[0], (traces[1][0], _other_schema(traces[1][1]))], id="schema"),
    pytest.param(lambda traces: [traces[0], (traces[1][0], json.dumps(traces[1][1])[:500])], id="truncated"),
])
def test_an_unusable_rank_trace_is_not_verified_never_a_pass(damage):
    verdict = _compare(damage(_traces()))
    assert verdict["status"] == gate.NOT_VERIFIED
    assert verdict["detail"].startswith("NOT VERIFIED")


def test_a_platform_with_no_native_route_for_the_price_refuses():
    with pytest.raises(gate.TesseraRouteTraceError, match="no native route"):
        _compare(_traces(), platform="gfx1151")


def test_the_platform_comes_from_the_card_and_must_agree_with_the_caller():
    build = {"tessera_serving_scope": {"target": {"platform": "sm_121"}}}
    assert gate.resolve_platform(build, None) == "sm_121"
    assert gate.resolve_platform(None, "sm_121") == "sm_121"
    with pytest.raises(gate.TesseraRouteTraceError, match="differs"):
        gate.resolve_platform(build, "gfx1151")
    with pytest.raises(gate.TesseraRouteTraceError, match="no serving platform"):
        gate.resolve_platform({}, None)


def test_the_packaged_contract_prices_m44e1_as_served():
    """The real contract, no stand-in: grids and platform map as transcribed."""
    pytest.importorskip("tessera.serving")
    executes, formats = gate.load_trace_contract()
    for family, grid in GRIDS.items():
        assert formats[family]["grid"] == grid
    assert executes == _contract()[0]
    verdict = gate.compare_route_traces(
        _traces(), expected_ranks=2, config=_config(), platform="sm_121",
        executes_by_platform=executes, formats=formats)
    assert verdict["status"] == gate.AGREE, verdict["detail"]


# ---------------------------------------------------------------------------
# The shipcard slot: opened by the lane, closed by the traces, replayed
# ---------------------------------------------------------------------------
def _artifact(tmp_path):
    model_dir = tmp_path / "exported"
    model_dir.mkdir()
    (model_dir / "config.json").write_bytes((FIXTURE / "config.json").read_bytes())
    (model_dir / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    path = open_lane_shipcard(model_dir, "tessera")
    return model_dir, path


def _trace_files(tmp_path, traces=None):
    directory = tmp_path / "traces"
    directory.mkdir()
    paths = []
    for (label, trace), name in zip(traces or _traces(), RANK_FILES):
        path = directory / name
        path.write_text(json.dumps(trace))
        paths.append(str(path))
    return paths


def _fill(path, model_dir, paths, *extra):
    args = ["fill-route-trace", str(path), "--model-dir", str(model_dir),
            "--expected-ranks", "2", "--platform", "sm_121"]
    for trace in paths:
        args += ["--trace", trace]
    return shipcard_cli(args + list(extra))


def _trace_problems(path, model_dir):
    card = shipcard.load_shipcard(path)
    return [p for p in shipcard.verify(card, model_dir=model_dir)
            if p.startswith("route.trace")]


def test_the_lane_declares_the_trace_gate_and_a_tessera_card_owes_it(tmp_path):
    assert load_lane_spec("tessera").gate("route.trace").shipcard_slot == "route.trace"
    assert "route.trace" in shipcard.lane_gate_slots("tessera")
    assert "route.trace" not in shipcard.lane_gate_slots("gguf")
    model_dir, path = _artifact(tmp_path)
    assert _trace_problems(path, model_dir) == ["route.trace: UNFILLED"]


def test_agreeing_traces_close_the_slot_and_verify_replays_them(tmp_path, contract):
    model_dir, path = _artifact(tmp_path)
    assert _fill(path, model_dir, _trace_files(tmp_path)) == 0
    record = shipcard.load_shipcard(path)["slots"]["route.trace"]
    assert record["passed"] is True
    assert record["trace_verdict"]["served"] == EXPECTED
    assert _trace_problems(path, model_dir) == []


def test_a_mutated_served_contract_refuses_at_fill_and_leaves_the_slot_open(
        tmp_path, contract, capsys):
    model_dir, path = _artifact(tmp_path)
    traces = [(label, _served_moe_as_fp8(trace)) for label, trace in _traces()]
    assert _fill(path, model_dir, _trace_files(tmp_path, traces)) == 1
    err = capsys.readouterr().err
    assert "REFUSED" in err and NVFP4 in err and FP8 in err
    assert shipcard.load_shipcard(path)["slots"]["route.trace"] is None
    assert _trace_problems(path, model_dir) == ["route.trace: UNFILLED"]


def test_a_missing_rank_trace_is_not_verified_and_leaves_the_slot_open(
        tmp_path, contract, capsys):
    model_dir, path = _artifact(tmp_path)
    paths = _trace_files(tmp_path)
    paths[1] = str(tmp_path / "traces" / "routes-rank1-never-written.json")
    assert _fill(path, model_dir, paths) == EXIT_NOT_VERIFIED
    assert "NOT VERIFIED" in capsys.readouterr().err
    assert shipcard.load_shipcard(path)["slots"]["route.trace"] is None
    assert _trace_problems(path, model_dir) == ["route.trace: UNFILLED"]


def test_verify_replays_the_carried_traces_not_the_passed_flag(tmp_path, contract):
    model_dir, path = _artifact(tmp_path)
    assert _fill(path, model_dir, _trace_files(tmp_path)) == 0
    card = shipcard.load_shipcard(path)
    record = card["slots"]["route.trace"]
    record["route_traces"][1]["trace"] = _served_moe_as_fp8(record["route_traces"][1]["trace"])
    assert record["passed"] is True
    problems = [p for p in shipcard.verify(card, model_dir=model_dir)
                if p.startswith("route.trace")]
    assert any("ranks served different module histograms" in p for p in problems), problems
    assert any("differs from the replay" in p for p in problems), problems


def test_verify_refuses_traces_compared_against_another_config(tmp_path, contract):
    model_dir, path = _artifact(tmp_path)
    assert _fill(path, model_dir, _trace_files(tmp_path)) == 0
    card = shipcard.load_shipcard(path)
    config = json.loads(card["slots"]["route.trace"]["config_json"])
    config["quantization_config"]["config_groups"].popitem()
    card["slots"]["route.trace"]["config_json"] = json.dumps(config)
    problems = [p for p in shipcard.verify(card, model_dir=model_dir)
                if p.startswith("route.trace")]
    assert any("differs from the artifact's config.json" in p for p in problems), problems


def test_verify_refuses_when_no_contract_can_be_read(tmp_path, contract, monkeypatch):
    model_dir, path = _artifact(tmp_path)
    assert _fill(path, model_dir, _trace_files(tmp_path)) == 0

    def _absent():
        raise gate.TesseraRouteTraceError("no packaged Tessera runtime contract")

    monkeypatch.setattr(gate, "load_trace_contract", _absent)
    problems = _trace_problems(path, model_dir)
    assert problems and "REFUSED on replay" in problems[0]


def test_the_tessera_arm_prints_the_trace_step():
    driver = (ROOT / "prismaquant" / "run-pipeline.sh").read_text(encoding="utf-8")
    arm = driver.split('if [[ "$EXPORT_CONTAINER" == "tessera" ]]; then')[-1].split("\nfi\n")[0]
    assert "TESSERA_ROUTE_TRACE=" in arm
    assert "fill-route-trace" in arm
    assert "--expected-ranks" in arm
