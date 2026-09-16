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
def _artifact(tmp_path, config=None):
    model_dir = tmp_path / "exported"
    model_dir.mkdir()
    if config is None:
        (model_dir / "config.json").write_bytes((FIXTURE / "config.json").read_bytes())
    else:
        (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
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


# ---------------------------------------------------------------------------
# Tessera #509: the exact per-module grade
#
# A #509 trace stamps ``identity_version: 1`` in its header, ``rank``,
# ``world_size`` and ``platform`` beside it, and names the modules that
# dispatched under each entry. Only then can a consumer compare the contract
# each module rode; the histogram cannot see a swap between two modules.
#
# ``tests/fixtures/tessera_route_trace_509`` is what Tessera's OWN telemetry
# wrote at producer commit e72d581 (see that directory's ``PROVENANCE.md`` and
# ``generate.py``), so the schema under test is the producer's and not one
# invented here. The synthetic documents below exist only for inputs a real
# producer never writes -- a malformed one, a half-stamped one -- and they
# carry the producer's header shape so a schema change cannot hide behind them.
# ---------------------------------------------------------------------------
FIXTURE_509 = ROOT / "tests" / "fixtures" / "tessera_route_trace_509"
EXACT_RANK_FILES = ("routes-rank0.json", "routes-rank1.json")
SWAPPED_RANK_FILES = ("routes-swapped-rank0.json", "routes-swapped-rank1.json")

#: ``module prefix -> (policy family, structure, activation contract, N:K,
#: symbol, decoder)``. Two layer-0 modules ride different families, so a swap
#: between them leaves every count in the histogram identical.
EXACT_LEDGER = {
    "model.language_model.layers.0.mlp.down_proj":
        ("TESSERA_BF16", "dense", BF16, "N4096:K6144", "torch.mm", "torch_window"),
    "model.language_model.layers.0.mlp.gate_up_proj":
        ("TESSERA_FP8", "dense", FP8, "N8192:K6144", "torch.mm", "torch_window"),
    "model.language_model.layers.1.mlp.experts":
        ("TESSERA_NVFP4", "routed_moe", NVFP4, "N2048:K512",
         "vllm.fused_moe.modular_kernel:FLASHINFER_CUTLASS", "native_span2"),
    "model.language_model.layers.1.mlp.shared_experts.down_proj":
        ("TESSERA_NVFP4", "dense", NVFP4, "N4096:K1024", "torch.mm", "torch_window"),
    "model.language_model.layers.1.mlp.shared_experts.gate_up_proj":
        ("TESSERA_NVFP4", "dense", NVFP4, "N8192:K1024", "torch.mm", "torch_window"),
}

#: The payload ``formats[]`` grid each policy family's modules are priced on.
_GRID_FOR_FAMILY = {"TESSERA_BF16": "BF16", "TESSERA_FP8": "E4M3",
                    "TESSERA_NVFP4": "E2M1x2"}

#: The one identity schema version a consumer may read, pinned here so a gate
#: that changes it has to say so in this file rather than inherit the change.
IDENTITY_VERSION = 1


def _exact_config(ledger=None):
    """The price for ``ledger``: one config group per (family, structure)."""
    ledger = EXACT_LEDGER if ledger is None else ledger
    grouped = {}
    for name, (family, structure, *_rest) in ledger.items():
        grouped.setdefault((family, structure), []).append(name)
    groups = {
        f"tessera_{family.lower()}_{structure}": {
            "scheme": {"family": family, "structure": structure,
                       "grid": _GRID_FOR_FAMILY[family]},
            "targets": sorted(names),
        }
        for (family, structure), names in grouped.items()
    }
    return {"quantization_config": {"quant_method": "tessera",
                                    "config_groups": groups}}


def _entry(family, structure, contract, shape, names, **overrides):
    entry = {
        "policy": f"{family}:resident",
        "shape": shape,
        "symbol": "torch.mm",
        "decoder": "torch_window",
        "contract": contract,
        "kind": gate.TRACE_KIND_FOR_STRUCTURE[structure],
        "launches": 1,
        "modules": len(names),
        "module_names": list(names),
        "unnamed_modules": 0,
        "dispatches_without_prefix": 0,
    }
    entry.update(overrides)
    return entry


def _document(entries, *, rank=0, world_size=2, platform="sm_121",
              identity_version=IDENTITY_VERSION, rank_source="torch.distributed",
              rank_conflict=None):
    """A synthetic document carrying the producer's own header shape."""
    document = {"schema": "tessera.route_trace/1", "pid": 7, "entries": entries,
                "rank_conflict": rank_conflict}
    if rank is not None:
        document["rank"] = rank
    if world_size is not None:
        document["world_size"] = world_size
    if rank_source is not None:
        document["rank_source"] = rank_source
    if platform is not None:
        document["platform"] = platform
    if identity_version is not None:
        document["identity_version"] = identity_version
    return document


def _exact_trace(ledger, *, rank, token_counts=(1, 4), **header):
    """One rank's exact-grade trace for ``ledger``."""
    entries = []
    for m in token_counts:
        grouped = {}
        for name, (family, structure, contract, shape, symbol, decoder) in ledger.items():
            key = (family, structure, contract, f"M{m}:{shape}", symbol, decoder)
            grouped.setdefault(key, []).append(name)
        for (family, structure, contract, shape, symbol, decoder), names in sorted(grouped.items()):
            entries.append(_entry(family, structure, contract, shape, sorted(names),
                                  symbol=symbol, decoder=decoder))
    return _document(entries, rank=rank, **header)


def _exact_traces(ledger=None, **header):
    ledger = EXACT_LEDGER if ledger is None else ledger
    return [(f"rank{rank}", _exact_trace(ledger, rank=rank, **header))
            for rank in range(2)]


def _contract_parts():
    return _contract()


def _compare_exact(traces, *, priced=None, expected_ranks=2, platform="sm_121"):
    executes, formats = _contract_parts()
    return gate.compare_route_traces(
        traces, expected_ranks=expected_ranks, config=_exact_config(priced),
        platform=platform, executes_by_platform=executes, formats=formats)


def _fixture_traces(names=EXACT_RANK_FILES, *, priced=None):
    """``[(rank label, payload), ...]`` read from the producer's written trace."""
    traces = []
    for index, name in enumerate(names):
        payload = json.loads((FIXTURE_509 / name).read_text())
        traces.append((f"rank{index}", payload))
    return traces


def _compare_fixture_509(names=EXACT_RANK_FILES, **kwargs):
    """The gate on a real producer trace, priced by the m44e1 artifact."""
    return _compare(_fixture_traces(names), **kwargs)


def _served_owners():
    """What the m44e1 artifact prices, keyed exactly as the gate keys it."""
    executes, formats = _contract_parts()
    return gate.priced_histogram(
        _config(), platform="sm_121", executes_by_platform=executes,
        formats=formats)["owners"]


def test_the_fixture_is_a_written_trace_and_not_a_local_transcription():
    """Tessera wrote these; the schema under test is the producer's (#509)."""
    assert gate.IDENTITY_VERSION == IDENTITY_VERSION
    for name in EXACT_RANK_FILES + SWAPPED_RANK_FILES:
        payload = json.loads((FIXTURE_509 / name).read_text())
        assert payload["schema"] == gate.ROUTE_TRACE_SCHEMA
        assert payload["identity_version"] == IDENTITY_VERSION
        assert payload["rank_source"] == "torch.distributed"
        assert payload["rank_conflict"] is None
        assert payload["platform"] == "sm_121"
        assert payload["world_size"] == 2
        assert payload["rank"] in (0, 1)
        assert payload["entries"]
        for entry in payload["entries"]:
            assert entry["module_names"] == sorted(entry["module_names"])
            assert entry["modules"] == (
                len(entry["module_names"]) + entry["unnamed_modules"])
            assert entry["unnamed_modules"] == 0


def test_a_written_producer_trace_is_compared_module_by_module_and_says_so():
    verdict = _compare_fixture_509()
    assert verdict["status"] == gate.AGREE, verdict["detail"]
    assert verdict["granularity"] == gate.EXACT_GRANULARITY
    assert verdict["exact_module_qualified"] is True
    assert verdict["served_modules"]["rank0"] == verdict["priced_owners"]
    assert verdict["served_modules"]["rank1"] == verdict["priced_owners"]
    assert verdict["priced_owners"] == _served_owners()
    assert len(verdict["priced_owners"]) == len(EXACT_LEDGER)
    header = verdict["header"]["rank0"]
    assert (header["rank"], header["world_size"], header["platform"]) == (0, 2, "sm_121")
    assert header["identity_version"] == IDENTITY_VERSION
    assert header["rank_conflict"] is None


def test_the_written_swapped_serve_is_refused_though_its_counts_are_identical():
    """Two serves the producer wrote; only the module names can tell them apart."""
    for (label, agreed), (other_label, swapped) in zip(
            _fixture_traces(), _fixture_traces(SWAPPED_RANK_FILES)):
        as_served = gate.served_histogram(
            gate.parse_route_trace_document(agreed, where=label)[0], where=label)
        when_swapped = gate.served_histogram(
            gate.parse_route_trace_document(swapped, where=other_label)[0],
            where=other_label)
        assert when_swapped["histogram"] == as_served["histogram"], (
            "the swapped fixture must not change a count, or it tests nothing")

    verdict = _compare_fixture_509(SWAPPED_RANK_FILES)
    assert verdict["status"] == gate.REFUSED
    assert "differ per module" in verdict["detail"]
    assert (
        "model.language_model.layers.0.mlp.down_proj: "
        f"priced TESSERA_BF16/dense/{BF16} but served "
        f"TESSERA_NVFP4/dense/{NVFP4}" in verdict["detail"]
    ), verdict["detail"]
    assert (
        "model.language_model.layers.1.mlp.shared_experts.down_proj: "
        f"priced TESSERA_NVFP4/dense/{NVFP4} but served "
        f"TESSERA_BF16/dense/{BF16}" in verdict["detail"]
    ), verdict["detail"]


def test_a_module_named_under_two_contracts_in_one_forward_is_refused():
    """One module counted twice looks exactly like two modules to the counts."""
    priced = {
        "m.a": ("TESSERA_NVFP4", "dense", NVFP4, "N2048:K512", "torch.mm", "torch_window"),
        "m.b": ("TESSERA_BF16", "dense", BF16, "N2048:K512", "torch.mm", "torch_window"),
    }
    entries = [
        _entry("TESSERA_NVFP4", "dense", NVFP4, "M1:N2048:K512", ["m.a"]),
        _entry("TESSERA_BF16", "dense", BF16, "M1:N2048:K512", ["m.a"]),
    ]
    documents = [_document(entries, rank=rank) for rank in range(2)]
    traces = [(f"rank{rank}", document) for rank, document in enumerate(documents)]
    parsed, _header = gate.parse_route_trace_document(documents[0], where="dup")
    assert gate.served_histogram(parsed, where="dup")["histogram"] == {
        f"TESSERA_NVFP4/dense/{NVFP4}": 1, f"TESSERA_BF16/dense/{BF16}": 1}

    verdict = _compare_exact(traces, priced=priced)
    assert verdict["status"] == gate.REFUSED
    assert "'m.a' dispatched under" in verdict["detail"], verdict["detail"]


def test_the_named_modules_must_add_up_to_the_reported_count():
    priced = {
        "m.a": ("TESSERA_NVFP4", "dense", NVFP4, "N2048:K512", "torch.mm", "torch_window"),
        "m.b": ("TESSERA_NVFP4", "dense", NVFP4, "N2048:K512", "torch.mm", "torch_window"),
    }
    entries = [_entry("TESSERA_NVFP4", "dense", NVFP4, "M1:N2048:K512", ["m.a"],
                      modules=2)]
    traces = [(f"rank{rank}", _document(entries, rank=rank)) for rank in range(2)]
    verdict = _compare_exact(traces, priced=priced)
    assert verdict["status"] == gate.REFUSED
    assert "modules=2" in verdict["detail"], verdict["detail"]


def test_a_module_the_serve_could_not_name_is_refused_not_counted():
    """A count is not a name: the unnamed count never passes on the histogram."""
    priced = {
        "m.a": ("TESSERA_NVFP4", "dense", NVFP4, "N2048:K512", "torch.mm", "torch_window"),
        "m.b": ("TESSERA_NVFP4", "dense", NVFP4, "N4096:K1024", "torch.mm", "torch_window"),
    }
    entries = [_entry("TESSERA_NVFP4", "dense", NVFP4, "M1:N2048:K512", ["m.a"],
                      modules=2, unnamed_modules=1, dispatches_without_prefix=1)]
    documents = [_document(entries, rank=rank) for rank in range(2)]
    traces = [(f"rank{rank}", document) for rank, document in enumerate(documents)]
    executes, formats = _contract_parts()
    served = gate.served_histogram(
        gate.parse_route_trace_document(documents[0], where="unnamed")[0], where="unnamed")
    assert served["histogram"] == gate.priced_histogram(
        _exact_config(priced), platform="sm_121", executes_by_platform=executes,
        formats=formats)["histogram"]

    verdict = _compare_exact(traces, priced=priced)
    assert verdict["status"] == gate.REFUSED
    assert "no stable prefix" in verdict["detail"], verdict["detail"]


@pytest.mark.parametrize("damage", [
    pytest.param(lambda trace: {k: v for k, v in trace.items()
                                if k != "identity_version"}, id="names-without-version"),
    pytest.param(lambda trace: {**trace, "identity_version": IDENTITY_VERSION + 1},
                 id="future-version"),
    pytest.param(lambda trace: {**trace, "entries": [
        {k: v for k, v in entry.items() if k != "unnamed_modules"}
        for entry in trace["entries"]]}, id="partial-entry-identity"),
    pytest.param(lambda trace: {**trace, "identity_version": None}, id="version-absent"),
])
def test_half_stamped_identity_is_refused_never_read_as_v1(damage):
    traces = [(label, damage(trace)) for label, trace in _exact_traces()]
    verdict = _compare_exact(traces)
    assert verdict["status"] == gate.REFUSED, verdict["detail"]
    assert verdict["exact_module_qualified"] is False


def test_identity_version_without_names_is_refused():
    traces = []
    for rank, trace in _exact_traces():
        entries = [{k: v for k, v in entry.items()
                    if k not in ("module_names", "unnamed_modules",
                                 "dispatches_without_prefix")}
                   for entry in trace["entries"]]
        for entry in entries:
            entry["modules"] = 1
        traces.append((rank, {**trace, "entries": entries}))
    verdict = _compare_exact(traces)
    assert verdict["status"] == gate.REFUSED, verdict["detail"]
    assert "names no modules" in verdict["detail"] or "name their modules" in verdict["detail"]


def test_a_legacy_trace_stays_histogram_grade_and_is_never_exact_qualified():
    verdict = _compare(_traces())
    assert verdict["status"] == gate.AGREE, verdict["detail"]
    assert verdict["granularity"] == gate.GRANULARITY
    assert verdict["exact_module_qualified"] is False
    assert verdict["served_modules"] is None
    assert verdict["header"]["rank0"]["stamped"] is False
    assert "histogram grade" in verdict["detail"]


def _without_module_identity(trace):
    """The same rank's file without the #509 entry identity, header intact."""
    stripped = {key: value for key, value in trace.items() if key != "identity_version"}
    stripped["entries"] = [
        {key: value for key, value in entry.items()
         if key not in ("module_names", "unnamed_modules", "dispatches_without_prefix")}
        for entry in trace["entries"]]
    return stripped


def test_ranks_that_disagree_on_the_grade_are_refused():
    """One rank stamps its rank identity and names no module; the other names them."""
    exact = _exact_traces()
    traces = [exact[0], (exact[1][0], _without_module_identity(exact[1][1]))]
    verdict = _compare_exact(traces)
    assert verdict["status"] == gate.REFUSED
    assert "observation grade" in verdict["detail"], verdict["detail"]


def test_a_legacy_file_among_exact_ones_is_refused_as_a_half_stamped_serve():
    exact = _exact_traces()
    legacy = json.loads((FIXTURE / RANK_FILES[1]).read_text())
    verdict = _compare_exact([exact[0], (exact[1][0], legacy)])
    assert verdict["status"] == gate.REFUSED
    assert "stamps its ranks together" in verdict["detail"], verdict["detail"]


@pytest.mark.parametrize("damage", [
    pytest.param(lambda traces: [(label, {**trace, "world_size": 4})
                                 for label, trace in traces], id="world-size"),
    pytest.param(lambda traces: [(label, {**trace, "rank": 0})
                                 for label, trace in traces], id="duplicate-rank"),
    pytest.param(lambda traces: [(traces[0][0], {**traces[0][1], "platform": "gfx1201"}),
                                 traces[1]], id="platform"),
    pytest.param(lambda traces: [(traces[0][0], traces[0][1]),
                                 (traces[1][0], {k: v for k, v in traces[1][1].items()
                                                 if k not in ("rank", "world_size")})],
                 id="one-rank-unstamped"),
])
def test_a_header_that_does_not_match_the_serve_is_refused(damage):
    verdict = _compare_exact(damage(_exact_traces()))
    assert verdict["status"] == gate.REFUSED, verdict["detail"]


def test_a_label_bound_to_another_rank_than_the_file_stamps_is_refused():
    traces = _exact_traces()
    swapped = [("rank1", traces[0][1]), ("rank0", traces[1][1])]
    verdict = _compare_exact(swapped)
    assert verdict["status"] == gate.REFUSED
    assert "bound to rank" in verdict["detail"], verdict["detail"]


def test_exact_grade_traces_close_the_slot_and_verify_replays_them(tmp_path, contract):
    model_dir, path = _artifact(tmp_path)
    assert _fill(path, model_dir, _trace_files(tmp_path, _fixture_traces())) == 0
    record = shipcard.load_shipcard(path)["slots"]["route.trace"]
    assert record["passed"] is True
    assert record["trace_verdict"]["exact_module_qualified"] is True
    assert record["trace_verdict"]["granularity"] == gate.EXACT_GRANULARITY
    assert _trace_problems(path, model_dir) == []


def test_exact_grade_swapped_contracts_refuse_at_fill_and_leave_the_slot_open(
        tmp_path, contract, capsys):
    model_dir, path = _artifact(tmp_path)
    traces = _fixture_traces(SWAPPED_RANK_FILES)
    assert _fill(path, model_dir, _trace_files(tmp_path, traces)) == 1
    err = capsys.readouterr().err
    assert "REFUSED" in err and "per module" in err
    assert shipcard.load_shipcard(path)["slots"]["route.trace"] is None
    assert _trace_problems(path, model_dir) == ["route.trace: UNFILLED"]


def test_a_recorded_rank_conflict_is_reported_and_never_gates_the_verdict():
    """The producer reports a later disagreeing observation; we do not gate it."""
    conflict = {"rank": 1, "world_size": 2, "source": "torch.distributed"}
    traces = []
    for label, payload in _fixture_traces():
        traces.append((label, {**payload, "rank_conflict": conflict}))
    verdict = _compare(traces)
    assert verdict["status"] == gate.AGREE, verdict["detail"]
    assert verdict["header"]["rank0"]["rank_conflict"] == conflict
    assert verdict["header"]["rank1"]["rank_conflict"] == conflict


def test_a_rank_conflict_field_that_is_absent_is_not_a_defect():
    traces = [(label, {k: v for k, v in payload.items() if k != "rank_conflict"})
              for label, payload in _fixture_traces()]
    verdict = _compare(traces)
    assert verdict["status"] == gate.AGREE, verdict["detail"]
    assert verdict["header"]["rank0"]["rank_conflict"] is None


def test_the_producers_unknown_platform_is_reported_and_not_compared():
    """``""`` is "never latched a token": an unknown platform is not another one."""
    traces = [(label, {**payload, "platform": ""}) for label, payload in _fixture_traces()]
    verdict = _compare(traces)
    assert verdict["status"] == gate.AGREE, verdict["detail"]
    assert verdict["header"]["rank0"]["platform"] == ""
