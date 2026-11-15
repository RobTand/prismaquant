"""Run-gate provenance in certified and development modes (#771, #1147).

Development mode is the default. ``PRISMAQUANT_DEV_MODE=0`` selects certified
mode, which is also this suite's default fixture:

* certified mode refuses the incomplete provenance fixtures below;
* development mode accepts and records a ``[DEV-MODE]`` warning and a
  top-level ``dev_uncertified`` stamp in results.json, so a dev result can
  never masquerade as a certified one.

CPU-only; no external checkpoint is touched.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from cpu_staged_fixtures import empty_native_staged_inputs, native_staged_state
import torch

from prismaquant import aura_provenance

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from container_inspection_fixture import inspection
from prismaquant import dev_mode
from prismaquant import joint_aura_run_transition as transition
from prismaquant.digests import DIRECT_UTF8_STRICT, file_sha256hex
from prismaquant import joint_aura_transitions as transitions
from prismaquant import tessera_joint_aura as bridge
from prismaquant.production_weight_cache import _production_cache_source_sha256

from test_joint_aura_streamed import _fixture, _run as _streamed_run
from test_streamed_cost_checkpoints import (
    _anchored_streamed_aura, _dense_runner, _DenseTinyLM,
)

DEV_ENV = dev_mode.DEV_MODE_ENV


def _write_json(path, value):
    path.write_bytes(DIRECT_UTF8_STRICT.encoded(value) + b"\n")
    return {"path": str(path), "sha256": file_sha256hex(path)}


def _stub_device_envelope(monkeypatch):
    monkeypatch.setattr(bridge, "_apply_device_envelope",
                        lambda device, device_bytes, **kw: {
                            "enforced": False, "stub": True, "device": str(device),
                            "device_envelope_bytes": int(device_bytes)})


@pytest.fixture
def sealed(tmp_path, monkeypatch):
    """The run-transition fixture of ``test_joint_aura_run_transition``, minimally.

    A tiny sealed package: an old tree hashed into the contract, a fixed tree
    that reconstructs it, and a receipt created against both. Dev-mode tests
    mutate the package the way an iteration does.
    """
    new_git = "3" * 40
    package = tmp_path / "package"
    package.mkdir()
    (package / "aura_cost.py").write_bytes(b"new-math\nnew-glue\n")
    (package / "tessera_joint_aura.py").write_bytes(b"new-import\n")
    (package / "dependency.py").write_bytes(b"unchanged\n")
    (package / "joint_aura_run_transition.py").write_bytes(b"verifier\n")
    (package / "joint_aura_transitions.py").write_bytes(b"dispatcher\n")
    old = tmp_path / "original"
    old.mkdir()
    (old / "aura_cost.py").write_bytes(b"old-math\nold-glue\n")
    (old / "tessera_joint_aura.py").write_bytes(b"old-import\n")
    (old / "dependency.py").write_bytes(b"unchanged\n")
    old_source = _production_cache_source_sha256(old)
    for path in old.iterdir():
        path.unlink()
    old.rmdir()
    monkeypatch.setattr(transition, "_SOURCE_REWRITES", {
        "aura_cost.py": [("old-math", "new-math"), ("old-glue", "new-glue")],
        "tessera_joint_aura.py": [("old-import", "new-import")]})
    plan_config = {"output_root": str(tmp_path), "calibration": "fixed", "backend": "fixed"}
    plan = _write_json(tmp_path / "plan.json", plan_config)
    cache_file = tmp_path / "production.pkl"
    cache_file.write_bytes(b"prepared production cache fixture")
    prepared = _write_json(tmp_path / "prepared.json", {
        "schema": transition.PREPARED_SCHEMA, "status": "complete", "measured_cells": 6,
        "implementation_sha256": old_source, "plan_sha256": plan["sha256"],
        "production_cache": {"path": str(cache_file), "sha256": file_sha256hex(cache_file)}})
    identity = _write_json(tmp_path / "identity.json", {"schema": "campaign identity fixture"})
    contract = {"source_sha256": old_source, "git_commit": "1" * 40, "plan_sha256": plan["sha256"],
                "prepared_sha256": prepared["sha256"],
                "production_cache_sha256": file_sha256hex(cache_file),
                "campaign_identity_sha256": identity["sha256"], "measured_cells": 6}
    monkeypatch.setattr(transition, "_CONTRACT", contract)
    state = {"commit": new_git, "package": package}

    def actual_execution():
        return {"git_commit": state["commit"], **transition.source_proof(state["package"])}
    monkeypatch.setattr(transition, "_actual_execution", actual_execution)
    monkeypatch.setattr(transition, "_committed_package",
                        lambda root: {"git_parent_commit": "5" * 40})
    monkeypatch.setattr(aura_provenance, "checkpoint_git_commit", lambda: state["commit"])
    monkeypatch.setattr(aura_provenance, "producer_package_sha256",
                        lambda: transition.source_proof(state["package"])["producer_source_sha256"])
    receipt = transition.create_transition(
        bindings={"plan": plan, "prepared": prepared, "campaign_identity": identity},
        output=tmp_path / "transition.json")
    checkpoints = tmp_path / "checkpoints"

    def load(**overrides):
        arguments = {"config": plan_config, "plan_sha256": plan["sha256"], "prepared": prepared,
                     "checkpoint_dir": checkpoints, **overrides}
        return transitions.load_transition(receipt, **arguments)
    return {"receipt": receipt, "load": load, "state": state, "config": plan_config,
            "checkpoints": checkpoints, "old_source": old_source, "package": package,
            "plan": plan, "prepared": prepared}


# ----------------------------------------------------------------- gate 1: the
# source_proof / loader identity check accepts any executing package

@pytest.mark.parametrize("mutation", ["other_file", "math"])
def test_source_proof_still_refuses_a_changed_package_without_dev_mode(sealed, mutation):
    package = sealed["package"]
    if mutation == "other_file":
        (package / "dependency.py").write_bytes(b"unchanged\n ")
    else:
        (package / "aura_cost.py").write_bytes(b"wrong-math\nnew-glue\n")
    with pytest.raises(ValueError, match="source|package"):
        transition.source_proof(package)


def test_source_proof_admits_any_package_and_records_the_actual_tree_digest(
        sealed, monkeypatch, capsys):
    (sealed["package"] / "dependency.py").write_bytes(b"unchanged\n ")  # an iteration
    monkeypatch.setenv(DEV_ENV, "1")
    proof = transition.source_proof(sealed["package"])
    actual = _production_cache_source_sha256(sealed["package"])
    # Even a dev run records what ran: the ACTUAL tree digest, never the contract's.
    assert proof["producer_source_sha256"] == actual
    assert proof["reconstructed_source_sha256"] == actual
    assert proof["producer_source_sha256"] != sealed["old_source"]
    assert proof["dev_uncertified"] is True
    assert "DEV-MODE" in capsys.readouterr().out


def test_the_loader_admits_a_changed_package_under_dev_mode(sealed, monkeypatch, capsys):
    cap = sealed["load"]()  # certified: the package still matches the receipt
    (sealed["package"] / "dependency.py").write_bytes(b"unchanged\n ")
    with pytest.raises(ValueError, match="package"):
        sealed["load"]()
    monkeypatch.setenv(DEV_ENV, "1")
    dev_cap = sealed["load"]()  # accepted, stamped, loudly
    assert isinstance(dev_cap, transition.VerifiedRunTransition)
    out = capsys.readouterr().out
    assert "DEV-MODE" in out
    assert "receipt execution source" in out
    # The hot-path re-check refuses a post-admission change certified, records it dev.
    (sealed["package"] / "dependency.py").write_bytes(b"unchanged\n  ")
    monkeypatch.setenv(DEV_ENV, "0")  # certified is "0" since PQ #1147
    with pytest.raises(ValueError, match="package"):
        transition.require_verified_transition(
            cap, checkpoint_dir=sealed["checkpoints"], resume=True, joint_activation=True)
    monkeypatch.setenv(DEV_ENV, "1")
    assert transition.require_verified_transition(
        dev_cap, checkpoint_dir=sealed["checkpoints"], resume=True,
        joint_activation=True) is dev_cap


def test_a_foreign_checkpoint_manifest_is_recorded_under_dev_mode(
        sealed, monkeypatch, capsys):
    _, _, runner, cache = _fixture()
    _streamed_run(runner, cache, checkpoint_dir=sealed["checkpoints"])
    # Certified: a manifest carrying another measurement source is a wall.
    with pytest.raises(ValueError, match="another measurement source"):
        sealed["load"]()
    # Dev: recorded, not gated -- and the provenance is stamped with the
    # ACTUAL package digest. The stamp is equality-stable (no timestamp)
    # because unit checkpoints compare it across a resume.
    monkeypatch.setenv(DEV_ENV, "1")
    fixture_digest = _production_cache_source_sha256(sealed["package"])
    monkeypatch.setattr(transition, "source_proof", lambda root=None: {
        "producer_source_sha256": fixture_digest,
        "reconstructed_source_sha256": fixture_digest,
        "transition_module_sha256": "f" * 64, "dev_uncertified": True})
    cap = sealed["load"]()
    out = capsys.readouterr().out
    assert "DEV-MODE" in out and "another measurement source" in out
    provenance = cap.execution_provenance
    assert provenance[dev_mode.DEV_UNCERTIFIED_KEY] is True
    assert provenance[dev_mode.DEV_MODE_KEY]["producer_source_sha256"] == fixture_digest
    assert "timestamp" not in provenance[dev_mode.DEV_MODE_KEY]


def test_certified_execution_provenance_carries_no_dev_stamp(sealed):
    cap = sealed["load"]()
    provenance = cap.execution_provenance
    assert dev_mode.DEV_UNCERTIFIED_KEY not in provenance
    assert dev_mode.DEV_MODE_KEY not in provenance


# ----------------------------------------------------------------- gate 2: the
# prepared-record digest comparisons in the run path

def _prepared_binding(tmp_path, *, plan_sha256, implementation_sha256):
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps({
        "schema": bridge.PREPARED_SCHEMA, "status": "complete",
        "plan_sha256": plan_sha256, "implementation_sha256": implementation_sha256}))
    return {"path": str(prepared), "sha256": file_sha256hex(prepared)}


def test_prepared_digest_mismatch_still_refuses_without_dev_mode(tmp_path):
    binding = _prepared_binding(tmp_path, plan_sha256="a" * 64,
                                implementation_sha256="b" * 64)
    with pytest.raises(ValueError, match="prepared plan_sha256"):
        bridge._preflight_run_prepared(binding, plan_sha256="c" * 64,
                                       implementation_sha256="b" * 64,
                                       reader_identity=None, projection_backend=None)
    with pytest.raises(ValueError, match="prepared implementation_sha256"):
        bridge._preflight_run_prepared(binding, plan_sha256="a" * 64,
                                       implementation_sha256="d" * 64,
                                       reader_identity=None, projection_backend=None)


def test_prepared_digest_mismatch_is_recorded_under_dev_mode(tmp_path, monkeypatch, capsys):
    binding = _prepared_binding(tmp_path, plan_sha256="a" * 64,
                                implementation_sha256="b" * 64)
    monkeypatch.setenv(DEV_ENV, "1")
    completion = bridge._preflight_run_prepared(binding, plan_sha256="c" * 64,
                                                implementation_sha256="d" * 64,
                                                reader_identity=None,
                                                projection_backend=None)
    assert completion["plan_sha256"] == "a" * 64
    out = capsys.readouterr().out
    # The loud warning names BOTH digests, for both digest bindings.
    for stored, running in (("a" * 64, "c" * 64), ("b" * 64, "d" * 64)):
        assert stored in out and running in out
    assert "DEV-MODE" in out


def test_prepared_non_digest_fields_still_wall_under_dev_mode(tmp_path, monkeypatch):
    binding = _prepared_binding(tmp_path, plan_sha256="a" * 64,
                                implementation_sha256="b" * 64)
    monkeypatch.setenv(DEV_ENV, "1")
    with pytest.raises(ValueError, match="prepared reader_identity"):
        bridge._preflight_run_prepared(binding, plan_sha256="a" * 64,
                                       implementation_sha256="b" * 64,
                                       reader_identity={"reader": "other"},
                                       projection_backend=None)


# ----------------------------------------------------------------- gate 3: a
# checkpoint-lineage identity mismatch is reused under dev (PQ #1147)

def _aura_run(tmp_path, monkeypatch, *, source_sha, resume):
    """One durable streamed AURA run; returns ``(streaming context, payload)``.

    The context's ``install_calls`` counts layer installs, so zero means every
    unit came from the checkpoint lineage. (Until 2026-09-25 this ran the
    resident path against a codebook cache identity; that lane was archived,
    #1304, and durable AURA checkpoints are streamed-only.)
    """
    monkeypatch.setattr(aura_provenance, "checkpoint_git_commit", lambda: "7" * 40)
    monkeypatch.setattr(aura_provenance, "producer_package_sha256", lambda: source_sha)
    torch.manual_seed(2026)
    state = {name: value.detach().clone()
             for name, value in _DenseTinyLM().state_dict().items()}
    _model, context, runner = _dense_runner(state)
    payload = _anchored_streamed_aura(
        runner, torch.tensor([[1, 2, 3, 4]]), tmp_path / "checkpoints",
        resume=resume)
    return context, payload


def test_checkpoint_identity_mismatch_still_refuses_without_dev_mode(tmp_path, monkeypatch):
    _aura_run(tmp_path, monkeypatch, source_sha="a" * 64, resume=False)
    # The iteration changes the producer package; tonight's 752 s failure.
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _aura_run(tmp_path, monkeypatch, source_sha="b" * 64, resume=True)


def test_checkpoint_identity_mismatch_reuses_the_lineage_under_dev_mode(
        tmp_path, monkeypatch, capsys):
    # PQ #1147: dev mode reuses a mismatched lineage with a stamp; it never
    # archives and never recomputes (the 2026-09-19 archive did the work twice).
    _, first = _aura_run(tmp_path, monkeypatch, source_sha="a" * 64, resume=False)
    root = tmp_path / "checkpoints"
    old_manifest = json.loads((root / "manifest.json").read_text())
    monkeypatch.setenv(DEV_ENV, "1")
    context, payload = _aura_run(tmp_path, monkeypatch, source_sha="b" * 64, resume=True)
    out = capsys.readouterr().out
    assert "[DEV-MODE]" in out and "AURA checkpoint identity" in out
    assert "a" * 64 in out and "b" * 64 in out  # both producer digests are named
    assert not sorted(root.parent.glob("checkpoints.dev-archived-*"))
    assert json.loads((root / "manifest.json").read_text()) == old_manifest
    assert context.install_calls == 0  # every unit was reused, none recomputed
    assert repr(payload["costs"]) == repr(first["costs"])


# ----------------------------------------------------------------- gates 4+5:
# dispatch digests become records; PRISMAQUANT_DEV_MODE reaches the container


def _launch_container(monkeypatch, launcher_args):
    from tools import tessera_campaign_container as launcher
    monkeypatch.setattr(launcher, "inspect_or_load", lambda _: [inspection()])
    monkeypatch.setattr(launcher, "verify_pinned_import", lambda *args, **kwargs: {})
    monkeypatch.setattr(launcher, "checkout_commit", lambda cwd: None)
    monkeypatch.setattr(launcher, "gpu_attachment", lambda *args, **kwargs: (False, "CPU test"))
    executed = []
    from container_admission import stub_launch
    stub_launch(monkeypatch, launcher, executed)
    launcher.main(launcher_args)
    docker_env = dict(item.split("=", 1) for index, item in enumerate(executed[0])
                      if index and executed[0][index - 1] == "--env")
    return docker_env


def test_a_hand_sealed_spec_carries_dev_mode_to_the_container(monkeypatch):
    # The LAYER_READ_THREADS mirror: an env the spec itself declares travels
    # to the container without any dispatch-side merge. The launcher takes
    # the spec as an inline JSON document, exactly as the submitter seals it.
    from tools import tessera_campaign_container as launcher
    spec = json.dumps({"container": {"image": "x"},
                       "env": {dev_mode.DEV_MODE_ENV: "1"}})
    docker_env = _launch_container(
        monkeypatch, ["--spec", spec, "--", "python", "-c", "pass"])
    assert docker_env[dev_mode.DEV_MODE_ENV] == "1"


# ----------------------------------------------------------------- gate 6: the
# dev stamp is top-level in results.json (progress records: opt-in)

def _refusing_prepare(tmp_path, monkeypatch):
    from prismaquant import calibration_data, cost_streaming, gpu_guard
    draw = dict(fit_ids_sha256="a" * 64, text_sha256="b" * 64, nsamples=512,
                seqlen=512, seed=0)
    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *_args: None)
    monkeypatch.setattr(bridge, "load_measured_anchor_input", lambda _inputs, **_kwargs:
                        SimpleNamespace(census={"model": "fixture",
                                                "attention_implementation": "eager"},
                                        cells={}, unit_scope=None, render_mirror_root=None,
                                        synthesized_now=0, encoder_source_reuse=None,
                                        head_walk_workers=None, head_walk_resumed_units=0,
                                        payload={"provenance": {"hessian": {
                                            "calibration_identity": draw}}}))
    monkeypatch.setattr(calibration_data, "load_calibration_input", lambda *_args, **_kwargs:
                        (torch.zeros((1, 512), dtype=torch.int64),
                         {"provenance": {**draw, "nsamples": 1}}))
    monkeypatch.setattr(cost_streaming, "build_streamed_causal_lm", lambda *_a, **_k:
                        pytest.fail("subset draw reached model construction"))
    _stub_device_envelope(monkeypatch)
    config = {"model": "fixture", "inputs": {}, "output_root": str(tmp_path),
              "calibration_input": {"path": "fixture", "sha256": "a" * 64},
              "profile_tool": "cprofile", "max_gpu_bytes": 2048,
              "execution": {"production_act_scales": "0", "n_calib_samples": 1,
                            "calib_seqlen": 512}}
    with pytest.raises(ValueError, match="original full draw nsamples"):
        bridge.execute("prepare", config, plan_sha256="b" * 64)
    return json.loads((tmp_path / "prepare/results.json").read_text())


@pytest.mark.usefixtures("empty_native_staged_inputs")
def test_results_json_has_no_dev_keys_without_the_flag(tmp_path, monkeypatch):
    result = _refusing_prepare(tmp_path, monkeypatch)
    assert dev_mode.DEV_UNCERTIFIED_KEY not in result
    assert dev_mode.DEV_MODE_KEY not in result


@pytest.mark.usefixtures("empty_native_staged_inputs")
def test_results_json_carries_the_top_level_dev_stamp(tmp_path, monkeypatch):
    monkeypatch.setenv(DEV_ENV, "1")
    result = _refusing_prepare(tmp_path, monkeypatch)
    # Top level, unmistakable, and carrying the ACTUAL tree digest of what ran.
    assert result[dev_mode.DEV_UNCERTIFIED_KEY] is True
    stamp = result[dev_mode.DEV_MODE_KEY]
    assert stamp[DEV_ENV] == "1"
    assert len(stamp["producer_source_sha256"]) == 64
    assert stamp["producer_source_sha256"] == _production_cache_source_sha256(
        ROOT / "prismaquant")
    assert "timestamp" in stamp
