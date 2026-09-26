"""Sealing is off unless a run asks for it (PQ #1147).

Rob, 2026-09-24: "All sealing should be disabled until further notice." Dev
mode is the default: ``PRISMAQUANT_DEV_MODE`` unset (or anything but ``0``)
prints a ``[DEV-MODE]`` line at every run-gate seal and continues. ``0`` is
certified mode, where every seal refuses exactly as before.

``tests/conftest.py`` sets ``0`` for every test, so each test here unsets the
variable explicitly: the unset default is what these tests pin.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from prismaquant import dev_mode
from prismaquant.cost_stage_checkpoint import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

ENV = dev_mode.DEV_MODE_ENV
ONE, TWO = "1" * 64, "2" * 64


@pytest.fixture
def unset(monkeypatch, capsys):
    """The default: the variable unset. Returns a reader of captured stdout."""
    monkeypatch.delenv(ENV, raising=False)
    capsys.readouterr()
    return lambda: capsys.readouterr().out


# -- the switch --------------------------------------------------------------

@pytest.mark.parametrize("value,dev", [
    (None, True), ("", True), ("1", True), ("yes", True), ("0", False)])
def test_dev_mode_is_on_unless_the_variable_is_zero(value, dev):
    environ = {} if value is None else {ENV: value}
    assert dev_mode.dev_mode_enabled(environ) is dev


def test_the_unset_default_is_dev_mode(unset):
    assert dev_mode.dev_mode_enabled() is True


def test_the_switch_has_one_definition():
    from prismaquant import joint_adjoint_checkpoints

    assert joint_adjoint_checkpoints.dev_mode_enabled is dev_mode.dev_mode_enabled
    for path in (ROOT / "tools" / "dispatch_tessera_campaign.py",
                 ROOT / "prismaquant" / "joint_adjoint_checkpoints.py"):
        text = path.read_text()
        assert "environ.get(DEV_MODE_ENV" not in text
        assert 'environ.get("PRISMAQUANT_DEV_MODE"' not in text


# -- the helper --------------------------------------------------------------

class _Refused(ValueError):
    pass


def test_seal_check_agrees_silently(unset):
    assert dev_mode.seal_check("plan", "a", "a", where="here") is True
    assert unset() == ""


@pytest.mark.parametrize("refusal", [
    _Refused("the original message"),
    lambda: _Refused("the original message"),
])
def test_certified_mode_raises_the_sites_own_refusal(monkeypatch, refusal):
    monkeypatch.setenv(ENV, "0")
    with pytest.raises(_Refused, match="^the original message$"):
        dev_mode.seal_check("plan", "a", "b", where="here", refusal=refusal)


def test_certified_mode_builds_a_message_from_a_class(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    with pytest.raises(_Refused, match="here: plan differs"):
        dev_mode.seal_check("plan", "a", "b", where="here", refusal=_Refused)


def test_dev_mode_prints_both_values_and_continues(unset):
    assert dev_mode.seal_check(
        "policy", {"budget": {"workspace_reserve_bytes": 1}},
        {"budget": {"workspace_reserve_bytes": 2}}, where="here",
        refusal=_Refused("never")) is False
    out = unset()
    assert out.startswith("[DEV-MODE] seal policy differs at "
                          "budget.workspace_reserve_bytes (here): expected 1, actual 2")


def test_dev_mode_names_an_input_it_did_not_compute(unset):
    assert dev_mode.seal_check("source", {"schema": "x"}, dev_mode.NOT_COMPUTED,
                               where="here") is False
    assert "[DEV-MODE] seal source not computed (here)" in unset()


def test_same_overrides_equality(unset, monkeypatch):
    assert dev_mode.seal_check("status", "running", "complete", where="here",
                               same=True) is True
    monkeypatch.setenv(ENV, "0")
    with pytest.raises(RuntimeError, match="status differs"):
        dev_mode.seal_check("status", "running", "running", where="here", same=False)


# -- the campaign join --------------------------------------------------------

from tests.test_joint_quanta_join import campaign, probe  # noqa: E402,F401 (fixtures)


def test_join_admits_a_quantum_sealed_for_another_scope(tmp_path, campaign, unset):
    from prismaquant.joint_quanta_join import join_joint_quanta
    from tests.test_joint_quanta_allocator_bridge import _generated_outputs

    root, campaign, _, _, _ = _generated_outputs(tmp_path, campaign)
    other = copy.deepcopy(campaign)
    other["scope"] = {**campaign["scope"], "campaign": "another"}
    result = join_joint_quanta(receipts=None, campaign=other, input_root=root,
                               output_dir=tmp_path / "joined")
    assert result["status"] == "complete"
    out = unset()
    assert "[DEV-MODE] seal campaign_scope differs" in out
    assert "[DEV-MODE] seal payload campaign_scope differs" in out


def test_join_admits_rows_whose_probe_identity_differs(tmp_path, campaign, unset):
    """A replay regime is arithmetic: stamped for a human (the numerics case)."""
    from prismaquant.joint_quanta_join import join_joint_quanta
    from prismaquant.joint_replay_regime import stamp_replay_regime
    from tests.test_joint_quanta_allocator_bridge import _generated_outputs
    from tests.test_stageb_replay_regime import _quanta, _restamp

    root, campaign, _, _, _ = _generated_outputs(tmp_path, campaign)
    quanta = _quanta(root)
    _restamp(root, quanta[1], lambda arithmetic: stamp_replay_regime(
        arithmetic, "capture_batch=2"))
    result = join_joint_quanta(receipts=None, campaign=campaign, input_root=root,
                               output_dir=tmp_path / "joined")
    assert result["status"] == "complete"
    assert "[DEV-MODE] seal probe identity differs" in unset()


def _restamp_probe(root, quantum, field, value, *, first_only=False):
    """Rewrite a quantum's joint rows with ``probe_identity[field] = value``.

    Every digest over the probe identity is recomputed, so each row still
    validates on its own. Returns ``(rewritten, joint_rows)``.
    """
    import pickle

    from prismaquant.joint_aura import identity_sha256

    path = root / "layer-quanta" / quantum / "cost.pkl"
    payload = pickle.loads(path.read_bytes())
    rewritten = total = 0
    for rows in payload["costs"].values():
        for fmt, entry in list(rows.items()):
            if not isinstance(entry, dict) or "error" in entry:
                continue
            total += 1
            if first_only and rewritten:
                continue
            entry = copy.deepcopy(entry)
            entry["probe_identity"][field] = value
            operator = entry["joint_operator_identity"]
            digest = identity_sha256(entry["probe_identity"])
            entry["probe_identity_sha256"] = operator["probe_identity_sha256"] = digest
            entry["joint_operator_identity_sha256"] = identity_sha256(operator)
            rows[fmt] = entry
            rewritten += 1
    path.write_bytes(pickle.dumps(payload))
    return rewritten, total


_PROBE_IDENTITY = {
    "schema": "prismaquant.joint_aura.probes.v2", "calibration_sha256": "1" * 64,
    "calibration_shape": [2, 4], "calibration_dtype": "torch.int64", "n_probes": 2,
    "seed_base": 1, "token_scope": "all", "temperature": 1.0,
    "distribution": "rademacher", "normalization": "global_kl_fisher",
    "source_model": {"content_sha256": "2" * 64}, "source_execution": {"schema": "v1"},
    "producer_source_sha256": "3" * 64,
    "arithmetic": {"dtype": "torch.bfloat16", "execution_partition": {"rows": 2}},
}


@pytest.mark.parametrize("field,value,wall", [
    ("calibration_sha256", "9" * 64, True), ("calibration_shape", [1, 4], True),
    ("n_probes", 3, True), ("seed_base", 7, True), ("token_scope", "tail", True),
    ("noise_layout", {"rows": 2}, True), ("source_model", {"content_sha256": "8" * 64}, True),
    ("source_execution", {"schema": "v2"}, True),
    ("producer_source_sha256", "9" * 64, False),
    ("arithmetic", {"dtype": "torch.float16", "execution_partition": {"rows": 2}}, False),
    ("arithmetic", {"dtype": "torch.bfloat16", "execution_partition": {"rows": 4}}, True),
])
def test_a_probe_identity_splits_what_was_measured_from_how(field, value, wall):
    """One split for the cost table, the join and Stage B's restored rows."""
    from prismaquant.cost_currency import probe_identity_walls_differ

    assert probe_identity_walls_differ(_PROBE_IDENTITY, {**_PROBE_IDENTITY, field: value}) is wall


def test_join_refuses_a_quantum_of_another_calibration_draw(tmp_path, campaign, unset):
    """The calibration draw is what was measured: a wall in dev mode too."""
    from prismaquant.joint_quanta_join import JoinRefused, join_joint_quanta
    from tests.test_joint_quanta_allocator_bridge import _generated_outputs
    from tests.test_stageb_replay_regime import _quanta

    root, campaign, _, _, _ = _generated_outputs(tmp_path, campaign)
    _restamp_probe(root, _quanta(root)[1], "calibration_sha256", "9" * 64)
    with pytest.raises(JoinRefused, match="probe or measurement identity differs"):
        join_joint_quanta(receipts=None, campaign=campaign, input_root=root,
                          output_dir=tmp_path / "joined")
    assert "[DEV-MODE] seal probe identity" not in unset()


def test_rows_of_another_calibration_draw_refuse_in_one_cost_table(tmp_path, campaign, unset):
    from prismaquant.joint_quanta_join import JoinRefused, join_joint_quanta
    from tests.test_joint_quanta_allocator_bridge import _generated_outputs
    from tests.test_stageb_replay_regime import _quanta

    root, campaign, _, _, _ = _generated_outputs(tmp_path, campaign)
    rewritten, total = _restamp_probe(root, _quanta(root)[1], "calibration_sha256",
                                      "9" * 64, first_only=True)
    assert rewritten == 1 < total
    with pytest.raises(JoinRefused, match="rows do not share one probe/calibration identity"):
        join_joint_quanta(receipts=None, campaign=campaign, input_root=root,
                          output_dir=tmp_path / "joined")
    assert "[DEV-MODE] seal probe identity" not in unset()


def test_rows_of_another_producer_source_are_ranked_with_a_stamp(
        tmp_path, campaign, unset, monkeypatch):
    """The producer source is how a cost was computed, not what was measured."""
    from prismaquant.joint_quanta_join import JoinRefused, join_joint_quanta
    from tests.test_joint_quanta_allocator_bridge import _generated_outputs
    from tests.test_stageb_replay_regime import _quanta

    root, campaign, _, _, _ = _generated_outputs(tmp_path, campaign)
    rewritten, total = _restamp_probe(root, _quanta(root)[1], "producer_source_sha256",
                                      "9" * 64, first_only=True)
    assert rewritten == 1 < total
    result = join_joint_quanta(receipts=None, campaign=campaign, input_root=root,
                               output_dir=tmp_path / "joined")
    assert result["status"] == "complete"
    assert ("[DEV-MODE] seal probe identity differs at producer_source_sha256"
            in unset())
    monkeypatch.setenv(ENV, "0")
    with pytest.raises(JoinRefused, match="rows do not share one probe/calibration identity"):
        join_joint_quanta(receipts=None, campaign=campaign, input_root=root,
                          output_dir=tmp_path / "certified")


# -- the quantum record and the stage-A header ---------------------------------

def _record(**campaign_fields):
    from prismaquant.joint_layer_quanta import LAYER_QUANTUM_SCHEMA, canonical_sha256

    record = {"schema": LAYER_QUANTUM_SCHEMA, "quantum_id": "layer-001", "layer": 1,
              "campaign": {"plan_sha256": "a" * 64, "prepared_sha256": "b" * 64,
                           "read_manifest_sha256": "c" * 64,
                           "unit_roster_sha256": "d" * 64,
                           "campaign_scope": {"campaign": "sealed"},
                           **campaign_fields},
              "adjoint": {}}
    record["identity_sha256"] = canonical_sha256(record, where="quantum record")
    return record


CAMPAIGN = {"plan_sha256": "a" * 64, "prepared_sha256": "b" * 64,
            "read_manifest_sha256": "c" * 64, "unit_roster_sha256": "d" * 64,
            "campaign_scope": {"campaign": "sealed"}}


@pytest.mark.parametrize("field,value", [
    ("plan_sha256", "e" * 64), ("prepared_sha256", "e" * 64),
    ("read_manifest_sha256", "e" * 64), ("campaign_scope", {"campaign": "another"}),
    ("campaign_scope", None)])
def test_a_quantum_sealed_for_another_campaign_revision_is_admitted(
        field, value, unset, monkeypatch):
    from prismaquant.joint_layer_quanta import check_quantum_for_campaign

    campaign = {**CAMPAIGN, field: value}
    check_quantum_for_campaign(_record(), campaign)
    assert f"[DEV-MODE] seal {field}" in unset()
    monkeypatch.setenv(ENV, "0")
    with pytest.raises(ValueError, match="refusing"):
        check_quantum_for_campaign(_record(), campaign)


def test_the_unit_roster_stays_a_wall(unset):
    from prismaquant.joint_layer_quanta import check_quantum_for_campaign

    with pytest.raises(ValueError, match="another unit_roster_sha256"):
        check_quantum_for_campaign(_record(), {**CAMPAIGN, "unit_roster_sha256": "e" * 64})


def test_an_edited_record_stays_refused(unset):
    from prismaquant.joint_layer_quanta import check_quantum_for_campaign

    record = _record()
    record["layer"] = 2
    with pytest.raises(ValueError, match="edited record"):
        check_quantum_for_campaign(record, CAMPAIGN)


def test_a_stage_a_header_of_another_plan_or_a_null_scope_is_admitted(unset, monkeypatch):
    from prismaquant.joint_layer_quanta import check_adjoint_run_identity

    header = {"run_identity": {"plan_sha256": "a" * 64, "prepared_sha256": "b" * 64,
                               "campaign_scope": None}}
    for scope in (None, {"campaign": "sealed"}):
        check_adjoint_run_identity(header, plan_sha256="e" * 64,
                                   prepared_sha256="b" * 64, scope=scope)
    out = unset()
    assert "[DEV-MODE] seal campaign_scope" in out
    assert "[DEV-MODE] seal plan_sha256 differs" in out
    monkeypatch.setenv(ENV, "0")
    with pytest.raises(ValueError, match="scope is unset"):
        check_adjoint_run_identity(header, plan_sha256="a" * 64,
                                   prepared_sha256="b" * 64, scope=None)


# -- the catalog extension ------------------------------------------------------

def test_a_catalog_extension_over_a_null_scope_run_keeps_the_null(unset, monkeypatch):
    from prismaquant import joint_catalog_extension as extension

    header = {"run_identity": {"plan_sha256": "a" * 64, "campaign_scope": None}}
    document = {"schema": "prismaquant.joint_catalog_extension.v2"}
    identity = extension._effective_run_identity(document, header, {}, {})
    assert identity == header["run_identity"]
    assert "[DEV-MODE] seal the original run sealed no campaign scope" in unset()
    monkeypatch.setenv(ENV, "0")
    with pytest.raises(ValueError, match="derives none"):
        extension._effective_run_identity(document, header, {}, {})


# -- the joint projection qualification ------------------------------------------

def test_a_runtime_that_differs_from_the_qualification_runs(unset, monkeypatch):
    from prismaquant.joint_projection_backend import _require_runtime

    expected = {"image": "sha256:" + "a" * 64, "torch": "2.11.0+cu130"}
    actual = {"image": "sha256:" + "b" * 64, "torch": "2.11.0+cu130"}
    _require_runtime(actual, expected)
    assert "[DEV-MODE] seal joint projection runtime identity differs at image" in unset()
    monkeypatch.setenv(ENV, "0")
    with pytest.raises(RuntimeError, match="unqualified runtime identity: image"):
        _require_runtime(actual, expected)


# -- the dispatchers and the prepared completion ---------------------------------

def test_a_declared_submission_digest_is_a_record(tmp_path, unset, monkeypatch):
    import dispatch_tessera_campaign as dispatch

    path = tmp_path / "plan.json"
    path.write_bytes(b"plan")
    actual = hashlib.sha256(b"plan").hexdigest()
    assert dispatch._bound_sha256(path, "e" * 64, label="plan") == actual
    assert "[DEV-MODE] seal plan declared digest differs" in unset()
    monkeypatch.setenv(ENV, "0")
    with pytest.raises(RuntimeError, match="not the declared"):
        dispatch._bound_sha256(path, "e" * 64, label="plan")


def test_a_prepared_completion_of_another_plan_is_recorded(unset, monkeypatch):
    from prismaquant.tessera_joint_aura import require_prepared_digests

    completion = {"plan_sha256": "a" * 64, "implementation_sha256": ONE}
    require_prepared_digests(completion, plan_sha256="e" * 64, implementation_sha256=TWO)
    out = unset()
    assert "[DEV-MODE] seal prepared plan_sha256 differs" in out
    assert "[DEV-MODE] seal prepared implementation_sha256 differs" in out
    monkeypatch.setenv(ENV, "0")
    with pytest.raises(ValueError, match="prepared plan_sha256: identity mismatch"):
        require_prepared_digests(completion, plan_sha256="e" * 64,
                                 implementation_sha256=ONE)


# -- forward recovery -----------------------------------------------------------

def _forward_document():
    from prismaquant.joint_forward_resume import SCHEMA

    old = {"source_model": {"schema": "fixture", "content_sha256": "5" * 64},
           "producer_source_sha256": ONE, "calibration_sha256": "6" * 64,
           "execution_partition": {"partition_count": 2}}
    return {"schema": SCHEMA, "original_bind_identity": old,
            "implementation_compatibility": {
                "original": ONE, "recovery": ONE, "scope": "forward-identical-memory-only"},
            "campaign_identity": {"plan_sha256": "a" * 64, "unit_roster_sha256": "d" * 64},
            "session": {"run_identity_sha256": canonical_json_sha256(
                old, where="exact boundary source")},
            "frontier": 1, "n_batches": 2}


def test_forward_recovery_admits_another_implementation_source_and_plan(unset, monkeypatch):
    from prismaquant.joint_forward_resume import (
        ForwardRecoveryRefused, validate_forward_state)

    document = _forward_document()
    bind = {**document["original_bind_identity"], "producer_source_sha256": TWO,
            "source_model": {"schema": "fixture", "content_sha256": "7" * 64}}
    campaign = {**document["campaign_identity"], "plan_sha256": "e" * 64}
    assert validate_forward_state(document, bind_identity=bind,
                                  campaign_identity=campaign) == (1, 2)
    out = unset()
    assert "[DEV-MODE] seal forward recovery implementation differs" in out
    assert "[DEV-MODE] seal forward source and campaign identity differs" in out
    monkeypatch.setenv(ENV, "0")
    with pytest.raises(ForwardRecoveryRefused, match="not explicitly bound"):
        validate_forward_state(document, bind_identity=bind, campaign_identity=campaign)


@pytest.mark.parametrize("wall", ["calibration", "roster"])
def test_forward_recovery_still_refuses_another_draw_or_roster(unset, wall):
    from prismaquant.joint_forward_resume import (
        ForwardRecoveryRefused, validate_forward_state)

    document = _forward_document()
    bind = dict(document["original_bind_identity"])
    campaign = dict(document["campaign_identity"])
    if wall == "calibration":
        bind["calibration_sha256"] = "8" * 64
    else:
        campaign["unit_roster_sha256"] = "8" * 64
    with pytest.raises(ForwardRecoveryRefused, match="identity differs"):
        validate_forward_state(document, bind_identity=bind, campaign_identity=campaign)


# -- AURA checkpoints --------------------------------------------------------------

def test_an_aura_lineage_of_another_producer_is_reused_by_default(tmp_path, monkeypatch, capsys):
    from tests.test_dev_mode_provenance_gates import _aura_run

    _, first = _aura_run(tmp_path, monkeypatch, source_sha="a" * 64, resume=False)
    monkeypatch.delenv(ENV, raising=False)
    capsys.readouterr()
    context, payload = _aura_run(tmp_path, monkeypatch, source_sha="b" * 64, resume=True)
    assert "[DEV-MODE] seal AURA checkpoint identity differs" in capsys.readouterr().out
    assert context.install_calls == 0
    assert repr(payload["costs"]) == repr(first["costs"])
    assert not sorted(tmp_path.glob("checkpoints.dev-archived-*"))


def test_aura_units_without_a_manifest_are_archived_and_recomputed_by_default(
        tmp_path, monkeypatch, capsys):
    """Units with no manifest have no recorded identity to be reused under.
    Dev mode archives the lineage whole and recomputes, as it did before
    PQ #1147; certified mode refuses, as before."""
    from tests.test_dev_mode_provenance_gates import _aura_run

    _, first = _aura_run(tmp_path, monkeypatch, source_sha="a" * 64, resume=False)
    root = tmp_path / "checkpoints"
    (root / "manifest.json").unlink()
    units = sorted(path.name for path in (root / "units").glob("*.pkl"))
    assert units
    with pytest.raises(RuntimeError, match="without a manifest"):
        _aura_run(tmp_path, monkeypatch, source_sha="a" * 64, resume=True)
    assert not sorted(tmp_path.glob("checkpoints.dev-archived-*"))
    monkeypatch.delenv(ENV, raising=False)
    capsys.readouterr()
    context, payload = _aura_run(tmp_path, monkeypatch, source_sha="a" * 64, resume=True)
    assert "[DEV-MODE] archived AURA checkpoint lineage" in capsys.readouterr().out
    archived = sorted(tmp_path.glob("checkpoints.dev-archived-*"))
    assert len(archived) == 1
    assert sorted(path.name for path in (archived[0] / "units").glob("*.pkl")) == units
    assert (root / "manifest.json").is_file()
    assert context.install_calls > 0
    assert repr(payload["costs"]) == repr(first["costs"])


# -- the boundary generation a chain resume reopens -----------------------------

def _bound_generation(tmp_path):
    """An interrupted run's boundary generation: status ``running``."""
    from prismaquant.cost_streaming import BOUNDARY_STORAGE_SCHEMA, StreamedBoundaryArtifacts

    config = {"schema": BOUNDARY_STORAGE_SCHEMA, "directory": str(tmp_path / "exact"),
              "max_resident_bytes": 1 << 20, "max_auxiliary_bytes": 1 << 20,
              "max_artifact_bytes": 1 << 20, "prefetch_batches": 4}
    owner = StreamedBoundaryArtifacts(config)
    owner.bind({"source_model": "fixture"}, n_probes=2, published=True)
    (owner.directory / "entries").mkdir(exist_ok=True)
    return config, dict(owner.session)


@pytest.mark.parametrize("field,value", [
    ("max_resident_bytes", 1 << 21), ("max_auxiliary_bytes", 1 << 21),
    ("max_artifact_bytes", 1 << 21)])
def test_rebind_stamps_a_byte_ceiling_difference(tmp_path, unset, monkeypatch, field, value):
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts

    config, session = _bound_generation(tmp_path / "dev")
    owner = StreamedBoundaryArtifacts({**config, field: value})
    owner.rebind(session, identity={"source_model": "fixture"}, n_probes=2)
    assert f"[DEV-MODE] seal boundary storage policy differs at {field}" in unset()
    assert owner.session == session
    # The dev rebind republished the generation under its own ceilings, so
    # certified mode is shown on a generation of its own.
    config, session = _bound_generation(tmp_path / "certified")
    monkeypatch.setenv(ENV, "0")
    with pytest.raises(RuntimeError, match="another boundary storage policy"):
        StreamedBoundaryArtifacts({**config, field: value}).rebind(
            session, identity={"source_model": "fixture"}, n_probes=2)


@pytest.mark.parametrize("field,value", [
    ("prefetch_batches", 8), ("schema", "prismaquant.aura.boundary_storage.v2")])
def test_rebind_refuses_a_boundary_layout_difference(tmp_path, unset, field, value):
    """The schema, the capture order and the read window lay out the stored
    entries: a wall in dev mode too (PQ #1147)."""
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts

    config, session = _bound_generation(tmp_path)
    running = {**config, field: value}
    if value == "prismaquant.aura.boundary_storage.v2":
        running["capture_order"] = "layer_major"
    with pytest.raises(RuntimeError, match="another boundary storage policy"):
        StreamedBoundaryArtifacts(running).rebind(
            session, identity={"source_model": "fixture"}, n_probes=2)
    assert "[DEV-MODE]" not in unset()


@pytest.mark.parametrize("status", ["complete", "attached", "retained"])
def test_rebind_refuses_a_generation_that_did_not_stop_partway(tmp_path, unset, status):
    """Another owner or reader holds it: a wall in dev mode too (PQ #1147)."""
    from prismaquant.cost_streaming import StreamedBoundaryArtifacts

    config, session = _bound_generation(tmp_path)
    path = Path(config["directory"]) / session["generation"] / "generation.json"
    document = json.loads(path.read_text())
    document["status"] = status
    path.write_text(json.dumps(document))
    with pytest.raises(RuntimeError, match=f"status is '{status}'"):
        StreamedBoundaryArtifacts(config).rebind(
            session, identity={"source_model": "fixture"}, n_probes=2)
    assert "[DEV-MODE]" not in unset()
