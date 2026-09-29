"""Row startup under a GPU reservation (PQ #1654).

A PACT gamut row sat about 203 s with its GPU idle before its first encode.
Two of the causes are PrismaQuant's, and these tests pin their fixes:

* **Identity.** A streaming row dispatched without a source-identity proof
  hashes every shard it reads, whole, on its GPU reservation. The dispatcher
  used to warn and plan the row anyway; it now refuses, both when it plans and
  when it is asked to submit a manifest that carries such a row, and it
  refuses a proof the row itself would refuse at adoption.
* **Source weights.** The stream head installed the whole layer (13.8 GB) to
  snapshot its projected expert views, then re-read each one from its shard
  in one serial pass to compare the two byte for byte. It now snapshots only
  dense units: a projected unit is a ``meta`` placeholder whose reader thread
  reads the producer's source tensor -- the tensor the exporter re-reads and
  the serial check compares against -- and the row prices exactly that.
"""
from __future__ import annotations

import json
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))

from test_tessera_row_stream import (COLUMNS, LOAD_POLICY, UNITS,  # noqa: E402
                                     stream_fixture)


# ---------------------------------------------------------------------------
# Identity: the dispatcher refuses a streaming row with no source proof
# ---------------------------------------------------------------------------

def _proofless_streaming_workspace(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    workspace = tmp_path / "campaign"
    workspace.mkdir()
    (workspace / "census.json").write_text(json.dumps({
        "model": str(model), "anchor_groups": {"u:a": ["a"]}, "layer_stride": 1,
        "unit_shapes": {"a": [8, 16]}}))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "model": str(model), "campaign_argv": ["--streaming"], "cwd": str(tmp_path),
        "python": "python3", "env": {}, "headroom_gb": 0}))
    return spec, workspace


def _streaming_plan_args(spec, workspace, **overrides):
    args = dict(spec=spec, workspace=workspace, calibration_cache="capture.json",
                groups_per_row=1, rows_per_box=1, timeout_s=300,
                stack_sample=None, stack_sample_seed=0, audit_rate=10,
                probe=None, seed_checkpoint=None, seed_wire_dir=None)
    args.update(overrides)
    return SimpleNamespace(**args)


def test_a_streaming_plan_without_a_source_proof_is_refused(tmp_path, monkeypatch):
    import dispatch_tessera_campaign as dispatch

    spec, workspace = _proofless_streaming_workspace(tmp_path)
    monkeypatch.setattr(dispatch, "_calibration_cache_binding",
                        lambda path, census: {"path": str(tmp_path / path),
                                              "sha256": "c" * 64})
    with pytest.raises(RuntimeError, match="require --source-identity-cache"):
        dispatch.cmd_plan(_streaming_plan_args(spec, workspace))
    # Nothing was planned, so nothing can be submitted.
    assert not (workspace / "manifest.json").exists()
    assert not (workspace / "plan.json").exists()


CAPTURE = {"path": "capture.json", "sha256": "c" * 64}


def _proved_model(tmp_path, monkeypatch, *, roster_sha=None):
    """A two-shard model, a real proof of it, and a capture roster.

    The proof is built by the production builder (``_write_proof``), so the
    planner checks the bytes a row would adopt. Only the capture manifest's
    roster is stood in: ``roster_sha`` replaces one shard's declared SHA.
    """
    from safetensors.torch import save_file
    from test_stage_a_identity_proof_adoption import _write_proof
    from prismaquant import tessera_calibration_cache as cc

    model = tmp_path / "model"
    model.mkdir()
    save_file({"a": torch.ones(2, 2)}, str(model / "a.safetensors"))
    save_file({"b": torch.zeros(2, 2)}, str(model / "b.safetensors"))
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "a.safetensors", "b": "b.safetensors"}}))
    proof, digest = _write_proof(model, tmp_path / "source-identity.json")
    roster = {name: cc.sha256(model / name) for name in ("a.safetensors", "b.safetensors")}
    if roster_sha is not None:
        roster["b.safetensors"] = roster_sha
    roster["config.json"] = "f" * 64
    monkeypatch.setattr(cc, "require_capture_contract", lambda path, expected_sha256=None: (
        {"identity": {"source_files": roster}} if (path, expected_sha256) ==
        (CAPTURE["path"], CAPTURE["sha256"]) else pytest.fail("another capture")))
    return model, proof, digest


def test_an_adoptable_proof_is_bound_by_its_digest(tmp_path, monkeypatch):
    import dispatch_tessera_campaign as dispatch

    model, proof, digest = _proved_model(tmp_path, monkeypatch)
    assert dispatch._source_identity_cache_binding(proof, model, CAPTURE) == {
        "path": str(proof.resolve()), "sha256": digest}


def test_a_proof_that_differs_from_the_capture_roster_is_refused_at_plan(
        tmp_path, monkeypatch):
    import dispatch_tessera_campaign as dispatch

    model, proof, _digest = _proved_model(tmp_path, monkeypatch, roster_sha="b" * 64)
    with pytest.raises(RuntimeError, match="not a source proof a row can adopt.*"
                       "b.safetensors: streamed source SHA differs"):
        dispatch._source_identity_cache_binding(proof, model, CAPTURE)


def test_a_stale_proof_is_refused_at_plan_not_hashed_around_on_the_row(
        tmp_path, monkeypatch):
    """A restamped shard: the row's adoption would refuse and hash it fresh."""
    import dispatch_tessera_campaign as dispatch
    from test_stage_a_identity_proof_adoption import _restamp

    model, proof, _digest = _proved_model(tmp_path, monkeypatch)
    _restamp(model / "a.safetensors")
    with pytest.raises(RuntimeError, match="a.safetensors: streamed source proof names "
                                           "another object"):
        dispatch._source_identity_cache_binding(proof, model, CAPTURE)


def test_the_spec_can_name_the_proof_a_replan_binds(tmp_path):
    import dispatch_tessera_campaign as dispatch

    assert dispatch._planned_source_identity_cache(None, "p.json") == "p.json"
    assert dispatch._planned_source_identity_cache("q.json", None) == "q.json"
    assert dispatch._planned_source_identity_cache(
        str(tmp_path / "p.json"), str(tmp_path / "." / "p.json")) == str(tmp_path / "p.json")
    with pytest.raises(RuntimeError, match="differs from the spec's source_identity_cache"):
        dispatch._planned_source_identity_cache("q.json", "p.json")
    with pytest.raises(RuntimeError, match="nonempty path string"):
        dispatch._planned_source_identity_cache(None, "")


def test_plan_checks_the_specs_proof_as_it_checks_the_flags(tmp_path, monkeypatch):
    """No flag, a spec-declared proof: plan binds and checks that proof."""
    import dispatch_tessera_campaign as dispatch

    model, proof, _digest = _proved_model(tmp_path, monkeypatch, roster_sha="b" * 64)
    workspace = tmp_path / "campaign"
    workspace.mkdir()
    (workspace / "census.json").write_text(json.dumps({
        "model": str(model), "anchor_groups": {"u:a": ["a"]}, "layer_stride": 1,
        "unit_shapes": {"a": [8, 16]}}))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({
        "model": str(model), "campaign_argv": ["--streaming"], "cwd": str(tmp_path),
        "python": "python3", "env": {}, "headroom_gb": 0,
        "source_identity_cache": str(proof)}))
    monkeypatch.setattr(dispatch, "_calibration_cache_binding",
                        lambda path, census: dict(CAPTURE))
    with pytest.raises(RuntimeError, match="not a source proof a row can adopt"):
        dispatch.cmd_plan(_streaming_plan_args(spec, workspace))
    assert not (workspace / "manifest.json").exists()


def _campaign_row(*flags, units="units/row-0003.json"):
    return {"argv": ["python3", "-u", "-m", "prismaquant.tessera_campaign",
                     "--model", "m", *(["--units", units] if units else []), *flags],
            "demand": {"mem_gb": 1}}


SELECTED = ("--streaming", "--calibration-cache", "c.json",
            "--calibration-cache-sha256", "c" * 64)


def test_submit_refuses_a_planned_row_that_carries_no_source_proof():
    import dispatch_tessera_campaign as dispatch

    rows = [_campaign_row(*SELECTED, "--source-identity-cache", "p.json",
                          "--source-identity-cache-sha256", "d" * 64,
                          units="units/row-0001.json"),
            _campaign_row(*SELECTED, units="units/row-0003.json"),
            # A census row reads the whole source and is not a selected row.
            _campaign_row("--streaming", "--census-out", "census.json", units=None)]
    with pytest.raises(dispatch.DemandRefused, match="row-0003") as refused:
        dispatch.require_source_identity_proofs(rows)
    assert "row-0001" not in str(refused.value)
    dispatch.require_source_identity_proofs([rows[0], rows[2]])


def test_the_manifest_check_refuses_before_it_derives_any_demand(tmp_path, monkeypatch):
    import dispatch_tessera_campaign as dispatch

    workspace = tmp_path / "campaign"
    workspace.mkdir()
    manifest = workspace / "manifest.json"
    manifest.write_text(json.dumps([_campaign_row(*SELECTED)]))
    (tmp_path / "spec").mkdir()
    spec, _ = _proofless_streaming_workspace(tmp_path / "spec")
    (workspace / "census.json").write_text("{}")
    monkeypatch.setattr(dispatch, "verify_manifest_demands",
                        lambda *_a, **_k: pytest.fail("demands derived for an unproved row"))
    with pytest.raises(dispatch.DemandRefused, match="source-identity-cache"):
        dispatch._checked_manifest(SimpleNamespace(workspace=str(workspace), spec=str(spec)),
                                   manifest=manifest)


# ---------------------------------------------------------------------------
# Projection: the byte check rides the row stream's readers
# ---------------------------------------------------------------------------

def _source_shard(tmp_path, tensor):
    from safetensors.torch import save_file

    root = tmp_path / "source"
    root.mkdir()
    save_file({"t": tensor}, str(root / "shard.safetensors"))
    return root, {"tensors": {"t": "shard.safetensors"}}, {"source_tensor": "t", "rows": 2, "cols": 8}


@pytest.mark.parametrize("release", [False, True])
def test_the_per_unit_check_is_the_serial_checks_comparison(tmp_path, release):
    from prismaquant import tessera_campaign as campaign

    tensor = torch.arange(16, dtype=torch.float32).reshape(2, 8).to(torch.bfloat16)
    root, source, unit = _source_shard(tmp_path, tensor)
    assert campaign._check_projected_unit(
        "u", unit, live=tensor.clone(), model_path=root, source=source,
        release_source_pages=release) is None
    changed = tensor.clone()
    changed[1, 3] += 1
    mismatch = campaign._check_projected_unit(
        "u", unit, live=changed, model_path=root, source=source,
        release_source_pages=release)
    assert mismatch.startswith("u (live (2, 8) torch.bfloat16 vs source t")
    with pytest.raises(RuntimeError) as serial:
        campaign._checked_projected_units({"s": {"u": unit}}, weights={"u": changed},
                                          model_path=root, source=source,
                                          release_source_pages=release)
    # The stream head's refusal names the unit exactly as the serial pass does.
    assert str(serial.value) == campaign.PROJECTED_BYTES_REFUSAL.format(units=mismatch)
    other_dtype = tensor.to(torch.float32)
    assert campaign._check_projected_unit(
        "u", unit, live=other_dtype, model_path=root, source=source) is not None


def test_the_stream_head_binds_the_projection_without_the_serial_pass(
        tmp_path, monkeypatch):
    from prismaquant import tessera_campaign as campaign
    from prismaquant import tessera_expert_projection as tep

    unit = {"source_tensor": "t", "rows": 2, "cols": 8}
    bound = {"s0": {"u1": dict(unit), "u2": dict(unit)}, "s1": {"u3": dict(unit)}}
    monkeypatch.setattr(tep, "bind_expert_projection", lambda producer, declared: bound)
    serial = []
    monkeypatch.setattr(campaign, "_checked_projected_units",
                        lambda *a, **k: serial.append(k.get("measured")) or {})
    population = SimpleNamespace(declared={"s0": ["u1", "u2"], "s1": ["u3"]})
    projection = {"schema": "carried", "producer": {"source": {"tensors": {}}}}
    kwargs = dict(weights={}, menus={}, model_path=str(tmp_path), cache_dir=tmp_path,
                  measured={"u1", "u3"}, projection=projection)
    carried, records = campaign._project_expert_population(population, check_units=False,
                                                           **kwargs)
    assert serial == []
    assert records == {"u1": unit, "u3": unit}
    assert carried == projection
    # The records are the unit set the serial pass would have read.
    assert records == campaign._measured_projected_units(bound, {"u1", "u3"})
    campaign._project_expert_population(population, **kwargs)
    assert serial == [{"u1", "u3"}]


def test_the_stream_heads_read_is_the_serial_checks_source_tensor(tmp_path):
    """What a stream-head reader prices is the tensor the serial check compares
    a snapshot against, so a snapshot that passed that check is these bytes."""
    from prismaquant import tessera_campaign as campaign
    from prismaquant.tessera_expert_projection import source_unit_weight

    tensor = torch.arange(16, dtype=torch.float32).reshape(2, 8).to(torch.bfloat16)
    root, source, unit = _source_shard(tmp_path, tensor)
    for release in (False, True):
        weight, done = campaign._read_projected_unit(
            "u", unit, model_path=root, source=source, release_source_pages=release)
        done()
        # Advising the pages away leaves the returned tensor intact: it is a copy.
        assert weight.dtype == tensor.dtype and torch.equal(weight, tensor)
        assert torch.equal(weight, source_unit_weight(root, source, unit))
        assert weight.is_contiguous() and weight.device.type == "cpu"
    # A snapshot view the serial check accepts is equal to that read.
    assert campaign._check_projected_unit("u", unit, live=tensor.clone(), model_path=root,
                                          source=source) is None


REAL = dict(zip(UNITS, (torch.full((32, COLUMNS), float(i + 1), dtype=torch.bfloat16)
                        for i in range(len(UNITS)))))


def _loading_stream(state, weights, load_unit, *, bind=None, threads=2, batch_size=1):
    from prismaquant import tessera_calibration_cache as cc
    from prismaquant.tessera_row_stream import RowStream

    manifest = state["manifest"]
    return RowStream(capture_path=manifest, expected_sha256=cc.sha256(manifest),
                     expected_identity=state["canonical"], census=state["census"],
                     names=UNITS, policy=LOAD_POLICY, weights=weights,
                     hessian_identity=state["calibration"],
                     bind=bind or (lambda name, *, weight, **_tensors: (None, dict(
                         weight=bytes(weight.view(torch.uint16).numpy()).hex()))),
                     threads=threads, batch_size=batch_size, device="cpu", memo_capacity=1,
                     load_unit=load_unit)


def _placeholders(names):
    return {name: (torch.empty(REAL[name].shape, dtype=REAL[name].dtype, device="meta")
                   if name in names else REAL[name].clone()) for name in UNITS}


def test_a_placeholder_unit_is_read_once_on_a_reader_and_installed(monkeypatch, tmp_path):
    import threading

    from prismaquant import tessera_calibration_cache as cc

    _campaign, _argv, state = stream_fixture(monkeypatch, tmp_path)
    streamed = set(UNITS[1:])
    weights = _placeholders(streamed)
    events, consumer = [], threading.get_ident()
    original_entry = cc._verified_capture_entry

    def entry(path, name, **kwargs):
        events.append(("entry", name))
        return original_entry(path, name, **kwargs)

    def load(name):
        assert threading.get_ident() != consumer, "the source read ran on the consumer thread"
        assert name in streamed
        events.append(("load", name))
        return REAL[name].clone()

    monkeypatch.setattr(cc, "_verified_capture_entry", entry)
    stream = _loading_stream(state, weights, load)
    stream.plan([[UNITS[1]], [UNITS[0]], [UNITS[2]], [UNITS[1]]])
    stream.admit(0)
    assert events.index(("load", UNITS[1])) < events.index(("entry", UNITS[1]))
    # Installed by the consumer before the encoder can ask for it.
    assert not weights[UNITS[1]].is_meta and torch.equal(weights[UNITS[1]], REAL[UNITS[1]])
    for index in (1, 2, 3):
        # Batch 3 is not adjacent to batch 0, so it re-reads the entry, and
        # never the source weight again.
        stream.admit(index)
    stream.finish()
    assert [event for event in events if event[0] == "load"] == [
        ("load", name) for name in sorted(streamed)]
    assert stream.stats["rereads"] == 1
    assert all(not value.is_meta and torch.equal(value, REAL[name])
               for name, value in weights.items())
    record = stream.execution_record()
    assert record["source_weight_reads"] == 2 and record["source_weight_read_seconds"] >= 0


def test_placeholder_units_bind_the_receipts_of_the_snapshot_they_replace(monkeypatch, tmp_path):
    """The A/B at the row stream: one arm snapshots every weight, the other
    streams two of them; both bind the same bytes into the same receipts and
    leave the encoder the same tensors (PQ #1654)."""
    from prismaquant import tessera_campaign as campaign

    _campaign, _argv, state = stream_fixture(monkeypatch, tmp_path)

    def bind(name, *, weight, inputs, hessian, source):
        return campaign._stream_unit_identity(
            name, weight=weight, inputs=inputs, hessian=hessian, source=source,
            menu=[], projected_unit=None, static_scales={})

    arms = {}
    for arm, streamed in (("snapshot", set()), ("stream", set(UNITS[1:]))):
        weights = _placeholders(streamed)
        stream = _loading_stream(state, weights, lambda name: REAL[name].clone(), bind=bind)
        stream.plan([[name] for name in UNITS])
        for index in range(len(UNITS)):
            stream.admit(index)
        stream.finish()
        arms[arm] = (stream.unit_identities(), stream.load_execution(), weights)
    assert arms["snapshot"][0] == arms["stream"][0]
    assert arms["snapshot"][1] == arms["stream"][1]
    for name in UNITS:
        left, right = arms["snapshot"][2][name], arms["stream"][2][name]
        assert left.dtype == right.dtype and left.stride() == right.stride()
        assert torch.equal(left.view(torch.uint16), right.view(torch.uint16))


@pytest.mark.parametrize("wrong", ["shape", "dtype", "meta"])
def test_a_source_read_that_is_not_the_planned_tensor_is_refused(monkeypatch, tmp_path, wrong):
    _campaign, _argv, state = stream_fixture(monkeypatch, tmp_path)
    weights = _placeholders({UNITS[0]})
    real = REAL[UNITS[0]]
    bad = {"shape": real[:16].clone(), "dtype": real.to(torch.float32),
           "meta": torch.empty(real.shape, dtype=real.dtype, device="meta")}[wrong]
    stream = _loading_stream(state, weights, lambda _name: bad)
    stream.plan([[UNITS[0]]])
    with pytest.raises(RuntimeError, match="not the planned host tensor"):
        stream.admit(0)
    assert weights[UNITS[0]].is_meta
    stream.close()


def test_a_placeholder_without_a_reader_is_refused(monkeypatch, tmp_path):
    _campaign, _argv, state = stream_fixture(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="require a load_unit reader"):
        _loading_stream(state, _placeholders({UNITS[2]}), None)


def test_the_reader_reserves_a_placeholders_source_bytes(monkeypatch, tmp_path):
    _campaign, _argv, state = stream_fixture(monkeypatch, tmp_path)
    weights = _placeholders({UNITS[2]})
    stream = _loading_stream(state, weights, lambda name: REAL[name].clone())
    nbytes = REAL[UNITS[2]].numel() * REAL[UNITS[2]].element_size()
    assert stream.reader_reserve_bytes(UNITS[2]) == stream.reader_reserve_bytes(UNITS[0]) + nbytes
    stream.plan([[UNITS[2]]])
    stream.admit(0)
    # Once installed, a re-read reserves the entry only.
    assert stream.reader_reserve_bytes(UNITS[2]) == stream.reader_reserve_bytes(UNITS[0])
    stream.close()


def test_placeholders_batch_as_the_host_weights_they_stand_for():
    from prismaquant import tessera_campaign as campaign

    pending = [(name, "TESSERA_E4M3_K1", rung) for name in UNITS for rung in (1024, 2048)]
    host = {name: REAL[name] for name in UNITS}
    meta = _placeholders(set(UNITS))
    mixed = _placeholders({UNITS[1]})
    for batch_size in (1, 2, 4):
        expected = campaign._anchor_batches(pending, weights=host, batch_size=batch_size)
        assert campaign._anchor_batches(pending, weights=meta, batch_size=batch_size) == expected
        assert campaign._anchor_batches(pending, weights=mixed, batch_size=batch_size) == expected


def test_the_preflight_lstats_on_threads_and_fails_in_name_order(monkeypatch, tmp_path):
    from prismaquant import tessera_calibration_cache as cc

    _campaign, _argv, state = stream_fixture(monkeypatch, tmp_path)
    manifest = json.loads(state["manifest"].read_text())
    root, entries = state["manifest"].parent, manifest["entries"]

    def preflight(entries, threads):
        return cc.preflight_verified_capture_entries(
            root, entries, names=UNITS, policy=LOAD_POLICY, census=state["census"],
            max_rows=state["canonical"]["max_act_rows"], threads=threads)

    serial = preflight(entries, 1)
    assert serial["max_file_bytes"] > 0
    assert all(preflight(entries, threads) == serial for threads in (2, 8))
    # Two failures: the earlier unit's is raised, whatever the thread count.
    broken = json.loads(json.dumps(entries))
    broken[UNITS[1]]["path"] = "elsewhere.pt"
    (root / entries[UNITS[2]]["path"]).unlink()
    for threads in (1, 8):
        with pytest.raises(RuntimeError, match=f"{UNITS[1]}: noncanonical"):
            preflight(broken, threads)
        with pytest.raises(FileNotFoundError):
            preflight(entries, threads)
    with pytest.raises(ValueError, match="positive thread count"):
        preflight(entries, 0)


# ---------------------------------------------------------------------------
# The stream head's weights: dense snapshots, projected placeholders
# ---------------------------------------------------------------------------

class _Runner:
    """``selected_weight_specs`` and ``snapshot_selected_weights`` over REAL."""

    def __init__(self):
        self.snapshots = []
        self.context = SimpleNamespace(weight_ckpt={name + ".weight": "shard" for name in UNITS})

    def selected_weight_specs(self, names):
        return {name: (tuple(REAL[name].shape), REAL[name].dtype,
                       REAL[name].numel() * REAL[name].element_size()) for name in names}

    def snapshot_selected_weights(self, names, **options):
        self.snapshots.append((list(names), options))
        return ({name: REAL[name].clone() for name in names},
                dict(schema="prismaquant.selected_source_weights.v1", units=sorted(names),
                     layers=[dict(layer=0, units=sorted(names), source="prefetched")],
                     resident_bytes=0, source_forward_count=0,
                     packed_parent_storage_retained=False,
                     source_snapshot_policy="selected-tensors-v1",
                     source_tensor_keys=[name + ".weight" for name in sorted(names)],
                     nonbody_materialized=False))


def _resources(budget=None, keys=None):
    return dict(selected_source_weight_bytes=(sum(
        value.numel() * value.element_size() for value in REAL.values())
        if budget is None else budget),
        source_tensor_keys=keys if keys is not None else sorted(
            name + ".weight" for name in UNITS))


def test_projected_units_are_placeholders_and_dense_units_are_snapshotted():
    from prismaquant import tessera_campaign as campaign
    from prismaquant.model_profiles import DefaultProfile

    runner = _Runner()
    weights, record = campaign._streamed_source_weights(
        runner, profile=DefaultProfile(), dense_targets=[UNITS[0]],
        expert_targets=UNITS[1:], resources=_resources(),
        snapshot_policy="selected-tensors-v1")
    assert [names for names, _options in runner.snapshots] == [[UNITS[0]]]
    options = runner.snapshots[0][1]
    assert options["host"] is True and options["expected_source_keys"] == (UNITS[0] + ".weight",)
    assert torch.equal(weights[UNITS[0]], REAL[UNITS[0]])
    for name in UNITS[1:]:
        assert weights[name].is_meta
        assert (weights[name].shape, weights[name].dtype) == (REAL[name].shape, REAL[name].dtype)
    assert record["units"] == sorted(UNITS)
    assert record["resident_bytes"] == _resources()["selected_source_weight_bytes"]
    assert record["source_tensor_keys"] == [UNITS[0] + ".weight"]
    assert record["streamed_source_units"] == sorted(UNITS[1:])
    assert record["streamed_source_policy"] == campaign.STREAMED_SOURCE_POLICY
    assert record["source_forward_count"] == 0


def test_an_all_projected_row_installs_no_layer():
    from prismaquant import tessera_campaign as campaign
    from prismaquant.model_profiles import DefaultProfile

    runner = _Runner()
    weights, record = campaign._streamed_source_weights(
        runner, profile=DefaultProfile(), dense_targets=[], expert_targets=list(UNITS),
        resources=_resources(), snapshot_policy="selected-tensors-v1")
    assert runner.snapshots == []
    assert all(value.is_meta for value in weights.values())
    assert record["layers"] == [] and record["source_tensor_keys"] == []


def test_the_stream_heads_weights_keep_the_snapshots_refusals():
    from prismaquant import tessera_campaign as campaign
    from prismaquant.model_profiles import DefaultProfile

    with pytest.raises(RuntimeError, match="exceed their resident byte budget"):
        campaign._streamed_source_weights(
            _Runner(), profile=DefaultProfile(), dense_targets=[], expert_targets=list(UNITS),
            resources=_resources(budget=1), snapshot_policy="selected-tensors-v1")
    with pytest.raises(RuntimeError, match="differ from the admitted plan"):
        campaign._streamed_source_weights(
            _Runner(), profile=DefaultProfile(), dense_targets=[UNITS[0]],
            expert_targets=UNITS[1:], resources=_resources(keys=[UNITS[1] + ".weight"]),
            snapshot_policy="selected-tensors-v1")


def test_a_streamed_unit_needs_a_rostered_admitted_producer_tensor():
    from prismaquant import tessera_campaign as campaign

    units = {name: {"source_tensor": name + ".weight"} for name in UNITS}
    source = {"tensors": {name + ".weight": "shard" for name in UNITS}}
    admitted = [name + ".weight" for name in UNITS]
    campaign._require_streamed_projection(UNITS, units, source, admitted)
    campaign._require_streamed_projection(UNITS, units, source, None)
    with pytest.raises(RuntimeError, match="no producer projection: " + UNITS[2]):
        campaign._require_streamed_projection(UNITS, {n: units[n] for n in UNITS[:2]},
                                              source, admitted)
    with pytest.raises(RuntimeError, match="not in the producer's roster"):
        campaign._require_streamed_projection(UNITS, units, {"tensors": {}}, admitted)
    with pytest.raises(RuntimeError, match="not admitted source keys: " + UNITS[0]):
        campaign._require_streamed_projection(UNITS, units, source, admitted[1:])
