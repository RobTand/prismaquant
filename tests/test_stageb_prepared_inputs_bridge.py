"""Prepared-input derivation, dispatch and native immutable PWC route (PQ #917).

The three completion requirements execute here against a tiny fixture
with an ACTUAL prepared pickle (a real ProductionWeightCache with real
render files and sealed verified cells, loaded once by the generator):

1. ``tools/regenerate_joint_quanta.py --executable-readsets`` derives
   each layer's prepared contract from the pickle through the existing
   retained planners (no forged annotations); the emitted manifests
   carry one render phase per retained window.
2. The production ``before_window`` body
   (``prepare_retained_window_read``) enters the render phase and then
   awaits the window's exact sealed entries through the existing
   strict readiness API before the loading pool reads its immutable inputs.
3. The integrated fixture runs normal generator CLI -> normal
   dispatcher CLI (FakeGateway submission seam only, no private
   validator) -> immutable staged inputs -> the production readiness
   body -> strict PWC reads, with exact tensor bytes, reader closure and
   zero pool bytes. The separate native bridge proof delays second-window
   warming and uses stock afterok to exclude the consumer until publication.

CPU/meta scope: model kernels are tiny synthetic tensors; no GPU
campaign claim. Progress transport remains a separate migration obligation.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from prismaquant import io_engine
from prismaquant import joint_layer_quanta as jl
from prismaquant.production_weight_cache import ProductionWeightCache
from test_quantum_executable_readset import (
    N_PROBES, N_BATCHES, STRIDED, _tiny_parent, _tiny_receipt, _write_boundary,
)
from prismaquant.staged_tier_policy import staged_tier_policy_test_context
from staged_native_fixtures import bind, native_staged_state, publish_inputs

from prepared_input_fixtures import (LAYERS, FMT, SHAPE, STATISTICS_CAP, WINDOWS_PER_LAYER, _qnames, _render_files, _production_pkl, _execution)


@pytest.fixture(autouse=True)
def _native_policy(native_staged_state):
    with staged_tier_policy_test_context("ssd"):
        yield

def _campaign_files(tmp_path, *, execution=None, model_config=None):
    """Plan/prepared/parent/derivation/partition/receipt files + digests.

    ``execution`` adds fields to the plan's execution block; ``model_config``
    writes the source ``config.json``.
    """
    files = _render_files(tmp_path)
    pkl_path = _production_pkl(tmp_path, files)
    pkl_sha = hashlib.sha256(pkl_path.read_bytes()).hexdigest()
    out_root = tmp_path / "run"
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_config is not None:
        (model_dir / "config.json").write_text(json.dumps(model_config))
    calib = tmp_path / "calib.pt"
    torch.save(torch.arange(16, dtype=torch.float32), calib)
    plan = {
        "output_root": str(out_root),
        "model": str(model_dir),
        "execution": {**_execution(), **(execution or {})},
        "calibration_input": {
            "path": str(calib),
            "sha256": hashlib.sha256(calib.read_bytes()).hexdigest()},
        "distributed_campaign": {},
    }
    parent = _tiny_parent()
    parent["mount_prefix"] = str(tmp_path)
    for entry in parent["entries"]:
        entry["path"] = entry["path"].replace("/fixture/model", str(model_dir))
    prepared = {
        "formats_by_qname": {
            name: [FMT] for layer in LAYERS for name in _qnames(layer)},
        "production_cache": {"path": str(pkl_path), "sha256": pkl_sha},
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    prepared_path = tmp_path / "prepared.json"
    prepared_path.write_text(json.dumps(prepared))
    parent_path = tmp_path / "parent.json"
    parent_path.write_text(json.dumps(parent))
    derivation = {"chunk_target_bytes": 200, "stride": 2,
                  "ram_window_gib": 160, "max_resident_consumers": 2}
    derivation_path = tmp_path / "derivation.json"
    derivation_path.write_text(json.dumps(derivation))
    partition = {"windows_by_layer": {str(layer): WINDOWS_PER_LAYER
                                      for layer in LAYERS}}
    partition_path = tmp_path / "partition.json"
    partition_path.write_text(json.dumps(partition))
    campaign = {
        "plan_sha256": hashlib.sha256(
            plan_path.read_bytes()).hexdigest(),
        "prepared_sha256": hashlib.sha256(
            prepared_path.read_bytes()).hexdigest(),
        "campaign_scope": parent["annotations"]["campaign_scope"],
    }
    receipt = _tiny_receipt(tmp_path, campaign)
    receipt["boundary_entries"]["0"] = [
        _write_boundary(tmp_path / "adjoint", 0, batch)
        for batch in range(N_BATCHES)]
    receipt["plan_sha256"] = campaign["plan_sha256"]
    receipt["prepared_sha256"] = campaign["prepared_sha256"]
    receipt_path = tmp_path / "adjoint-capture.json"
    receipt_path.write_text(json.dumps(receipt))
    return {
        "files": files, "plan": plan, "prepared": prepared,
        "plan_path": plan_path, "prepared_path": prepared_path,
        "parent_path": parent_path, "derivation_path": derivation_path,
        "partition_path": partition_path, "receipt_path": receipt_path,
        "campaign": campaign,
        "parent_sha": hashlib.sha256(
            parent_path.read_bytes()).hexdigest(),
    }


def _run_regen(tmp_path, *extra, execution=None, model_config=None):
    """The normal generator CLI; returns (records_dir, out_root)."""
    import regenerate_joint_quanta as regen
    layout = _campaign_files(tmp_path, execution=execution,
                             model_config=model_config)
    records_out = tmp_path / "regen" / "records"
    code = regen.main([
        "--plan", str(layout["plan_path"]),
        "--plan-sha256", layout["campaign"]["plan_sha256"],
        "--prepared", str(layout["prepared_path"]),
        "--prepared-sha256", layout["campaign"]["prepared_sha256"],
        "--parent-manifest", str(layout["parent_path"]),
        "--parent-manifest-sha256", layout["parent_sha"],
        "--derivation", str(layout["derivation_path"]),
        "--partition", str(layout["partition_path"]),
        "--records-out", str(records_out),
        "--output-root", str(tmp_path / "run"),
        "--adjoint-receipt", str(layout["receipt_path"]),
        "--executable-readsets",
        *extra,
    ])
    assert code == 0, f"generator CLI refuses the bridge fixture: {code}"
    return layout, records_out


# -- requirement 1: real derivation -----------------------------------------

def test_regen_cli_derives_prepared_contracts(tmp_path):
    layout, records_out = _run_regen(tmp_path)
    paths = sorted(records_out.glob("layer-*.json"))
    assert len(paths) == len(LAYERS)
    for record_path in paths:
        record = json.loads(record_path.read_text())
        block = record["executable_readset"]
        prepared = block["prepared_input"]
        assert prepared["schema"] == jl.PREPARED_INPUT_SCHEMA
        assert prepared["production_pkl_sha256"] == layout[
            "prepared"]["production_cache"]["sha256"]
        assert prepared["prepared_sha256"] == layout["campaign"][
            "prepared_sha256"]
        assert [w["window_index"] for w in prepared["windows"]] == list(
            range(WINDOWS_PER_LAYER))
        manifest = json.loads(
            __import__("gzip").decompress(
                Path(block["manifest_path"]).read_bytes()).decode("utf-8"))
        names = [p["name"] for p in manifest["read_plan"]["phases"]]
        for window in range(WINDOWS_PER_LAYER):
            assert names.index(f"render-{window:02d}") + 1 == names.index(
                f"replay-{window:02d}-p0")
        for window in prepared["windows"]:
            assert len(window["members"]) == 1
            assert len(window["entries"]) == 1
            entry = window["entries"][0]
            assert entry["offset"] == 0
            assert entry["bytes"] > 0
            assert Path(entry["path"]).stat().st_size == entry["bytes"]
            assert hashlib.sha256(
                Path(entry["path"]).read_bytes()).hexdigest() == entry[
                "sha256"]


# The spill bound (WS-SB4): 3 calibration rows of 8 tokens, one row per batch,
# captured 2 batches per pass -> groups of 16 and 8 tokens, 24 in all.
SPILL_EXECUTION = {"n_calib_samples": 3, "calib_seqlen": 8, "probe_microbatch": 1}
QWEN3_CONFIG = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}


@pytest.mark.parametrize("grid, reservation", [(None, 61440), (512, 18432)])
def test_regen_spill_mode_seals_the_spill_bound(tmp_path, grid, reservation):
    """``--replay-mode spill`` seals every record's spill bound.

    Each layer has two windows of one dense 16x32 target, so a token spills
    64 input and 32 gradient elements. In bf16 over 24 tokens that is
    3072 + 2 probes x 1536 = 6144 bytes, in (2 + 1) x 2 targets x 2 groups
    = 12 parts. The reservation adds 12 x (512 + grid) and rounds to the
    grid: 61440 on the default 4096-byte grid, 18432 on a 512-byte one.
    """
    from prismaquant.joint_replay_spill import check_spill_bound
    extra = ["--replay-mode", "spill", "--replay-regime", "capture_batch=2"]
    if grid is not None:
        extra += ["--spill-block-bytes", str(grid)]
    _layout, records_out = _run_regen(
        tmp_path, *extra, execution=SPILL_EXECUTION, model_config=QWEN3_CONFIG)
    paths = sorted(records_out.glob("layer-*.json"))
    assert len(paths) == len(LAYERS)
    for record_path in paths:
        bound = json.loads(record_path.read_text())[
            "executable_readset"]["spill_bound"]
        assert check_spill_bound(bound) == reservation
        assert bound["block"] == (grid or 4096)
        assert bound["capture_batch"] == 2
        assert bound["element_dtype"] == "bfloat16"
        geometry = bound["geometry"]
        assert geometry["tokens"] == 24
        assert geometry["max_batch_tokens"] == 16
        assert geometry["total_bytes"] == 6144
        assert geometry["max_parts"] == 12


def test_regen_windowed_mode_seals_no_spill_bound(tmp_path):
    _layout, records_out = _run_regen(tmp_path)
    for record_path in sorted(records_out.glob("layer-*.json")):
        assert "spill_bound" not in json.loads(
            record_path.read_text())["executable_readset"]


def test_regen_refuses_unadmittable_budget(tmp_path, capsys, monkeypatch):
    import regenerate_joint_quanta as regen

    # PQ #1049: the Stage A receipt and its run header bind the plan they
    # were captured under, so tightening a copy of the plan after the fact
    # refuses on the header's plan digest before the budget is ever
    # checked. Build the whole campaign under the tight plan instead.
    base_execution = _execution

    def tight_execution():
        execution = base_execution()
        execution["retained_operator_windows"]["budget"][
            "statistics_cap_bytes"] = 2048
        return execution

    monkeypatch.setattr(sys.modules[__name__], "_execution", tight_execution)
    layout = _campaign_files(tmp_path)
    plan_sha = layout["campaign"]["plan_sha256"]
    records_out = tmp_path / "regen-tight" / "records"
    capsys.readouterr()
    code = regen.main([
        "--plan", str(layout["plan_path"]), "--plan-sha256", plan_sha,
        "--prepared", str(layout["prepared_path"]),
        "--prepared-sha256", layout["campaign"]["prepared_sha256"],
        "--parent-manifest", str(layout["parent_path"]),
        "--parent-manifest-sha256", layout["parent_sha"],
        "--derivation", str(layout["derivation_path"]),
        "--partition", str(layout["partition_path"]),
        "--records-out", str(records_out),
        "--output-root", str(tmp_path / "run-tight"),
        "--adjoint-receipt", str(layout["receipt_path"]),
        "--executable-readsets",
    ])
    err = capsys.readouterr().err
    assert code != 0
    assert (f"plan {plan_sha} seals an operator-declared retained budget "
            "that does not admit layer") in err, err
    assert "derive_policy" in err, err
    assert list(records_out.glob("layer-*.json")) == []


def test_regen_refuses_tampered_production_pickle(tmp_path):
    import regenerate_joint_quanta as regen
    layout = _campaign_files(tmp_path)
    pkl_path = Path(layout["prepared"]["production_cache"]["path"])
    raw = bytearray(pkl_path.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    pkl_path.write_bytes(bytes(raw))
    records_out = tmp_path / "regen-tampered" / "records"
    code = regen.main([
        "--plan", str(layout["plan_path"]),
        "--plan-sha256", layout["campaign"]["plan_sha256"],
        "--prepared", str(layout["prepared_path"]),
        "--prepared-sha256", layout["campaign"]["prepared_sha256"],
        "--parent-manifest", str(layout["parent_path"]),
        "--parent-manifest-sha256", layout["parent_sha"],
        "--derivation", str(layout["derivation_path"]),
        "--partition", str(layout["partition_path"]),
        "--records-out", str(records_out),
        "--output-root", str(tmp_path / "run-tampered"),
        "--adjoint-receipt", str(layout["receipt_path"]),
        "--executable-readsets",
    ])
    assert code != 0
    assert list(records_out.glob("layer-*.json")) == []


def test_derive_refuses_missing_verified_cell(tmp_path):
    from prismaquant.joint_cost_quantum import derive_layer_prepared_inputs
    files = _render_files(tmp_path)
    pkl_path = _production_pkl(tmp_path, files)
    cache = pickle.loads(pkl_path.read_bytes())
    victim = (_qnames(2)[0], FMT)
    del cache.metadata["verified_cells"][victim]
    formats = {name: [FMT] for name in _qnames(2)}
    record = {"layer": 2, "quantum_id": "layer-002",
              "windows": [{"window_index": 0}, {"window_index": 1}],
              "campaign": {"unit_roster_sha256": "0" * 64}}
    with pytest.raises(ValueError, match="no verified cell"):
        derive_layer_prepared_inputs(
            record, execution=_execution(),
            formats_by_qname=formats, production_cache=cache,
            prepared_sha256="1" * 64, production_pkl_sha256="8" * 64,
            unit_roster_sha256="0" * 64)


def test_check_rejects_nonzero_offset(tmp_path):
    record, receipt, parent, _kwargs = _tiny_parent_and_bound(tmp_path)
    files = _offset_files(tmp_path)
    prepared = {
        "schema": jl.PREPARED_INPUT_SCHEMA,
        "production_pkl_sha256": "8" * 64,
        "unit_roster_sha256": "7" * 64,
        "prepared_sha256": record["campaign"]["prepared_sha256"],
        "windows": [{"window_index": 0,
                     "members": [["unit-x", "FMT-A"]],
                     "entries": [files[0]]},
                    {"window_index": 1,
                     "members": [["unit-y", "FMT-A"]],
                     "entries": [files[1]]}],
    }
    with pytest.raises(ValueError, match="whole-file"):
        jl.check_prepared_input_windows(
            prepared, window_indices=range(2), where="prepared inputs")


def _tiny_parent_and_bound(tmp_path):
    from test_quantum_executable_readset import _bound_inputs
    record, receipt, parent, kwargs = _bound_inputs(tmp_path)
    return record, receipt, parent, kwargs


def _offset_files(tmp_path):
    import torch as _torch
    out = []
    for index, name in enumerate(("unit-x", "unit-y")):
        path = tmp_path / "renders" / f"{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        _torch.save(_torch.arange(8, dtype=_torch.float32), path)
        raw = path.read_bytes()
        out.append({"qname": name, "fmt": "FMT-A", "path": str(path),
                    "offset": 8 if index == 0 else 0, "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest()})
    return out


def _dispatch_bridge(tmp_path, monkeypatch):
    """Normal generator/dispatcher CLI, replacing only native submission."""
    import dispatch_joint_quanta as dispatch

    layout, records = _run_regen(tmp_path)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"container": {"image": "sha256:" + "0" * 64},
                                "env": {}}))
    monkeypatch.setattr(dispatch, "SPEC_PATH", spec)
    gateway = dispatch.FakeGateway()
    # Four fake quanta and no pilot run: the explicit, recorded override
    # (production pilot admission is unchanged; PQ #1811 did the same).
    assert dispatch.main([
        "--records", str(records), "--output-root", str(tmp_path / "dispatch"),
        "--adjoint-receipt", str(layout["receipt_path"]),
        "--force-unverified-pilot"], _gateway=gateway,
        # No checkpoint behind the fixture; coverage is tested on its own
        # (test_readset_coverage_1095).
        _coverage=lambda rows: []) == 0
    assert len(gateway.submitted) == len(LAYERS)
    record = json.loads((records / "layer-002.json").read_text())
    block = record["executable_readset"]
    wire = Path(block["manifest_path"]).read_bytes()
    digest = hashlib.sha256(wire).hexdigest()
    submitted = next(row for row in gateway.submitted
                     if block["manifest_path"] in row["row"]["argv"])
    application_row = submitted["row"]
    argv = application_row["argv"]
    assert digest in " ".join(argv)
    phases = [phase.split("=", 1)[0] for phase in application_row["progress_phases"]]
    assert phases == block["phases"]
    return layout, record, phases


def _consume_bridge(layout, record, phases, resolver, monkeypatch):
    """Actual readiness and PWC byte/lifetime checks, also used by Flux proof."""
    from prismaquant.joint_cost_quantum import (
        ChunkFrontier, QuantumProgress, prepare_retained_window_read)
    from test_native_strict_reader_contract import _forbid_source

    windows = record["executable_readset"]["prepared_input"]["windows"]
    order = []
    sources = [_forbid_source(monkeypatch, Path(entry["path"]))
               for window in windows for entry in window["entries"]]
    assert resolver.declared_readset()["state"] == "bound", resolver.declared_readset()
    progress = QuantumProgress(frontier=ChunkFrontier(chunks=[], windows=[]))
    # Local phase data is the actual dispatched roster. Progress transport is
    # deliberately not injected; its independent migration stays in slice4.
    progress._phases = tuple(phases)
    real_enter = progress.enter_read_phase

    def enter(phase):
        real_enter(phase)
        assert progress._phase == phase
        order.append(f"phase:{phase}")

    monkeypatch.setattr(progress, "enter_read_phase", enter)
    cache = pickle.loads(Path(layout["prepared"]["production_cache"][
        "path"]).read_bytes())
    cache.enable_lru(1 << 20)
    cache.require_file_load_sha256(
        {pair: row["sha256"] for pair, row in layout["files"].items()},
        max_file_bytes=max(Path(row["path"]).stat().st_size
                           for row in layout["files"].values()))
    real_load = io_engine.load_file

    def load_file(path, limit, **kwargs):
        # A retained window's renders are read by the IO engine (PQ #1294),
        # on its pool threads: record the phase each read opens in.
        order.append(f"open:{progress._phase}")
        return real_load(path, limit, **kwargs)

    monkeypatch.setattr(io_engine, "load_file", load_file)
    for window in windows:
        index = window["window_index"]
        keys = [tuple(pair) for pair in window["members"]]
        assert prepare_retained_window_read(
            index, record=record, progress=progress) == "ready"
        order.append(f"ready:{index}")
        with cache.retained_window(keys, max_resident_bytes=1 << 20,
                                   max_workers=1, max_load_buffer_bytes=1 << 20):
            for key in keys:
                actual = cache.get(*key)
                name_index = _qnames(2).index(key[0])
                expected = torch.arange(512, dtype=torch.float32).reshape(
                    *SHAPE) + 200 + name_index
                assert torch.equal(actual.view(torch.uint8),
                                   expected.view(torch.uint8))
                assert cache.file_load_receipt(key, actual)["serving_tier"] == "stage"
        for probe in range(N_PROBES):
            progress.enter_read_phase(f"replay-{index:02d}-p{probe}")
    for index in range(2):
        assert order.index(f"phase:render-{index:02d}") < order.index(f"ready:{index}")
        assert order.index(f"ready:{index}") < order.index(f"open:render-{index:02d}")
        assert order.index(f"open:render-{index:02d}") < order.index(
            f"phase:replay-{index:02d}-p0")
    report = resolver.report()
    assert report["bytes_from_pool"] == 0
    assert report["bytes_from_stage"] > 0
    assert report["serving_tiers"] and all(
        row["pin_id"] for row in report["serving_tiers"])
    assert all(not opened for opened in sources)
    assert resolver._windows == 0
    descriptors = [fd for fd, _signature in resolver._fds.values()]
    assert descriptors
    resolver.close()
    assert not resolver._fds and not resolver._pins
    for fd in descriptors:
        with pytest.raises(OSError):
            os.fstat(fd)
    return {"order": order, "reads": report, "descriptors_closed": len(descriptors),
            "known_tensor_checks": 2, "canonical_payload_opens": 0}


def test_generator_dispatch_waits_for_delayed_leased_render(tmp_path, monkeypatch):
    """Generated declarations reach native readiness before strict pool loads.

    Native afterok/delayed-warm execution is proved separately with these same
    helpers; this test owns the application CLI, tensor and reader predicates.
    """
    layout, record, phases = _dispatch_bridge(tmp_path, monkeypatch)
    block = record["executable_readset"]
    entries = [entry for window in block["prepared_input"]["windows"]
               for entry in window["entries"]]
    binding = publish_inputs(tmp_path, [Path(entry["path"]) for entry in entries],
                             read_contract=Path(block["manifest_path"]).read_bytes())
    resolver = bind(monkeypatch, binding)
    _consume_bridge(layout, record, phases, resolver, monkeypatch)
