"""Row startup under a GPU reservation (PQ #1654).

A PACT gamut row sat about 203 s with its GPU idle before its first encode.
Two of the causes are PrismaQuant's, and these tests pin their fixes:

* **Identity.** A streaming row dispatched without a source-identity proof
  hashes every shard it reads, whole, on its GPU reservation. The dispatcher
  used to warn and plan the row anyway; it now refuses, both when it plans and
  when it is asked to submit a manifest that carries such a row, and it
  refuses a proof the row itself would refuse at adoption.
* **Projection.** The producer projection's byte check re-read every priced
  unit's source tensor in one serial pass (13.8 GB, 29.5 s) before the first
  encode. The stream head now runs the same per-unit check on its reader
  threads, before each unit's entry is read.
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


def _proof_fixture(tmp_path, monkeypatch, *, proved_sha, validator=None):
    """A model with one shard, a capture roster and a proof, the validators faked.

    ``validate_cached_streamed_model_identity`` is the one fail-closed proof
    validator (stat fingerprints, coverage, weight map, live config) and has
    its own tests; here it stands in so the dispatcher's use of it is what is
    under test.
    """
    from prismaquant import cost_streaming
    from prismaquant import tessera_calibration_cache as cc

    model = tmp_path / "model"
    model.mkdir()
    shard = model / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"shard")
    proof = tmp_path / "source-identity.json"
    proof.write_text("{}")
    identity = {"shards": [{"path": str(shard), "sha256": proved_sha}]}
    record = {"identity": identity, "fingerprints": []}
    calls = []

    def validate(source, path, *, require_complete_checkpoint):
        calls.append((source, str(path), require_complete_checkpoint))
        if validator is not None:
            validator()
        return identity

    monkeypatch.setattr(cost_streaming, "validate_cached_streamed_model_identity", validate)
    monkeypatch.setattr(cost_streaming, "_read_streamed_model_identity_cache",
                        lambda *_a, **_k: (record, identity))
    monkeypatch.setattr(cc, "require_capture_contract", lambda path, expected_sha256=None: {
        "identity": {"source_files": {shard.name: "a" * 64, "config.json": "f" * 64}}})
    return model, proof, calls


def test_a_proof_that_differs_from_the_capture_roster_is_refused_at_plan(
        tmp_path, monkeypatch):
    import dispatch_tessera_campaign as dispatch

    model, proof, _calls = _proof_fixture(tmp_path, monkeypatch, proved_sha="b" * 64)
    with pytest.raises(RuntimeError, match="does not prove the capture's source"):
        dispatch._source_identity_cache_binding(
            proof, model, {"path": "capture.json", "sha256": "c" * 64})


def test_a_stale_proof_is_refused_at_plan_not_hashed_around_on_the_row(
        tmp_path, monkeypatch):
    import dispatch_tessera_campaign as dispatch

    def drifted():
        raise RuntimeError("streamed model identity source shard stat drifted")

    model, proof, _calls = _proof_fixture(tmp_path, monkeypatch, proved_sha="a" * 64,
                                          validator=drifted)
    with pytest.raises(RuntimeError, match="stat drifted"):
        dispatch._source_identity_cache_binding(
            proof, model, {"path": "capture.json", "sha256": "c" * 64})


def test_an_adoptable_proof_is_bound_by_its_digest(tmp_path, monkeypatch):
    import dispatch_tessera_campaign as dispatch
    from prismaquant.tessera_calibration_cache import sha256

    model, proof, calls = _proof_fixture(tmp_path, monkeypatch, proved_sha="a" * 64)
    binding = dispatch._source_identity_cache_binding(
        proof, model, {"path": "capture.json", "sha256": "c" * 64})
    assert binding == {"path": str(proof.resolve()), "sha256": sha256(proof)}
    assert calls == [(str(model), str(proof.resolve()), True)]


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


def _checked_stream(state, weights, check_unit, *, threads=2):
    from prismaquant import tessera_calibration_cache as cc
    from prismaquant.tessera_row_stream import RowStream

    manifest = state["manifest"]
    return RowStream(capture_path=manifest, expected_sha256=cc.sha256(manifest),
                     expected_identity=state["canonical"], census=state["census"],
                     names=UNITS, policy=LOAD_POLICY, weights=weights,
                     hessian_identity=state["calibration"],
                     bind=lambda name, **_tensors: (None, dict(weight=name)),
                     threads=threads, batch_size=1, device="cpu", memo_capacity=1,
                     check_unit=check_unit)


def test_every_unit_is_checked_on_a_reader_before_its_entry_is_read(monkeypatch, tmp_path):
    import threading

    from prismaquant import tessera_calibration_cache as cc

    _campaign, _argv, state = stream_fixture(monkeypatch, tmp_path)
    weights = {name: torch.ones(32, COLUMNS, dtype=torch.bfloat16) for name in UNITS}
    events, consumer = [], threading.get_ident()
    original_entry = cc._verified_capture_entry

    def entry(path, name, **kwargs):
        events.append(("entry", name))
        return original_entry(path, name, **kwargs)

    def check(name, weight):
        assert threading.get_ident() != consumer, "the check ran on the consumer thread"
        assert weight is weights[name]
        events.append(("check", name))
        # A dense unit has no projection, so its check reads nothing.
        return name != UNITS[1]

    monkeypatch.setattr(cc, "_verified_capture_entry", entry)
    stream = _checked_stream(state, weights, check)
    # Only the first unit is ever admitted; finish reads the other two.
    stream.plan([[UNITS[0]]])
    stream.admit(0)
    assert events.index(("check", UNITS[0])) < events.index(("entry", UNITS[0]))
    stream.finish()
    for name in UNITS:
        assert events.count(("check", name)) == 1
        assert events.index(("check", name)) < events.index(("entry", name))
    assert stream.stats["projection_checked_reads"] == 2
    record = stream.execution_record()
    assert record["projection_checked_reads"] == 2
    assert record["projection_check_seconds"] >= 0


def test_a_unit_whose_source_bytes_differ_never_reaches_the_encoder(monkeypatch, tmp_path):
    from prismaquant import tessera_campaign as campaign

    _campaign, _argv, state = stream_fixture(monkeypatch, tmp_path)
    weights = {name: torch.ones(32, COLUMNS, dtype=torch.bfloat16) for name in UNITS}

    def check(name, weight):
        if name == UNITS[1]:
            raise RuntimeError(campaign.PROJECTED_BYTES_REFUSAL.format(units=name))
        return True

    stream = _checked_stream(state, weights, check)
    stream.plan([[name] for name in UNITS])
    stream.admit(0)
    with pytest.raises(RuntimeError, match=f"byte-for-byte .*{UNITS[1]}"):
        stream.admit(1)
    assert UNITS[1] not in stream._live
    with pytest.raises(RuntimeError, match="not resident"):
        stream.entry(UNITS[1])
    stream.close()
