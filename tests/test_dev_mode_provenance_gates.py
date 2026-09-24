"""PRISMAQUANT_DEV_MODE turns run-gate provenance into recorded stamps (#771).

Rob's decision (2026-09-19): rapid iteration must not pay provenance tax --
the seal returns at the artifact gate, not the run gate. One environment
variable, ``PRISMAQUANT_DEV_MODE=1``, read at the gates:

* without it, every gate below refuses under the same fixtures it refused
  under before the switch existed -- certified behavior is byte-identical;
* with it, the gate accepts and RECORDS: a loud ``[DEV-MODE]`` warning and a
  top-level ``dev_uncertified`` stamp in results.json, so a dev result can
  never masquerade as a certified one. Progress records carry the stamp only
  when ``PRISMAQUANT_DEV_PROGRESS_STAMP=1`` opts in (PR #828): the default
  per-unit commit is the certified six-field record and pays no hash.

CPU-only; no external checkpoint is touched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

from prismaquant import aura_cost as aura
from prismaquant import dev_mode
from prismaquant import joint_aura_run_transition as transition
from prismaquant import joint_aura_transitions as transitions
from prismaquant import tessera_joint_aura as bridge
from prismaquant.production_weight_cache import _production_cache_source_sha256
import prismaquant.production_weight_cache as pwc

from test_joint_aura_streamed import _fixture, _run as _streamed_run
from test_aura_checkpoint_resume_identity import (
    _cb_provenance, _run as _aura_cost_run, _TinyCache, _TinyLM,
)
from test_glm_joint_data_manifest_at_submit import (
    _prepared_fixture, _scope_args, _workspace,
)

DEV_ENV = dev_mode.DEV_MODE_ENV


@pytest.fixture()
def scratch(request):
    """A per-test directory that is never under ``/tmp`` (house rule, copied)."""
    root = ROOT / ".dev-mode-scratch" / request.node.name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def shared_mount(scratch, monkeypatch):
    """The fixture's files are not on /mnt/shared, so the prefix moves to it."""
    from experiments import glm_data_manifests
    monkeypatch.setattr(glm_data_manifests, "SHARED_MOUNT", str(scratch))


def _write_json(path, value):
    path.write_bytes(transition._canonical(value) + b"\n")
    return {"path": str(path), "sha256": transition._sha(path)}


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
        "production_cache": {"path": str(cache_file), "sha256": transition._sha(cache_file)}})
    identity = _write_json(tmp_path / "identity.json", {"schema": "campaign identity fixture"})
    contract = {"source_sha256": old_source, "git_commit": "1" * 40, "plan_sha256": plan["sha256"],
                "prepared_sha256": prepared["sha256"],
                "production_cache_sha256": transition._sha(cache_file),
                "campaign_identity_sha256": identity["sha256"], "measured_cells": 6}
    monkeypatch.setattr(transition, "_CONTRACT", contract)
    state = {"commit": new_git, "package": package}

    def actual_execution():
        return {"git_commit": state["commit"], **transition.source_proof(state["package"])}
    monkeypatch.setattr(transition, "_actual_execution", actual_execution)
    monkeypatch.setattr(transition, "_committed_package",
                        lambda root: {"git_parent_commit": "5" * 40})
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: state["commit"])
    monkeypatch.setattr(aura, "_aura_source_sha256",
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
    return {"path": str(prepared), "sha256": transition._sha(prepared)}


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
    monkeypatch.setattr(aura, "_checkpoint_git_commit", lambda: "7" * 40)
    monkeypatch.setattr(aura, "_aura_source_sha256", lambda: source_sha)
    monkeypatch.setattr(pwc, "production_cache_cb_render_provenance",
                        lambda *_args, **_kwargs: _cb_provenance())
    torch.manual_seed(2026)
    model = _TinyLM()
    payload = _aura_cost_run(model, _TinyCache(model), tmp_path / "checkpoints",
                             resume=resume)
    return model, payload


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
    model, payload = _aura_run(tmp_path, monkeypatch, source_sha="b" * 64, resume=True)
    out = capsys.readouterr().out
    assert "[DEV-MODE]" in out and "AURA checkpoint identity" in out
    assert "a" * 64 in out and "b" * 64 in out  # both producer digests are named
    assert not sorted(root.parent.glob("checkpoints.dev-archived-*"))
    assert json.loads((root / "manifest.json").read_text()) == old_manifest
    assert model.forward_calls == 0  # every unit was reused, none recomputed
    assert repr(payload["costs"]) == repr(first["costs"])


# ----------------------------------------------------------------- gates 4+5:
# dispatch digests become records; PRISMAQUANT_DEV_MODE reaches the container

def _joint_submission(scratch, *, prepared=True, extra=()):
    import dispatch_tessera_campaign as dispatch
    fixture = _workspace(scratch)
    arguments = ["submit-joint", "run", "--plan", str(fixture["plan"]),
                 *_scope_args(fixture)]
    if prepared:
        prepared_path, _cache = _prepared_fixture(scratch)
        arguments += ["--prepared", str(prepared_path)]
    spec = scratch / "spec.joint.json"
    spec.write_text(json.dumps({"container": {"image": "x"}}))
    arguments += ["--spec", str(spec), "--demand", "gpu=1,mem_gb=104",
                  "--manifest-dir", str(scratch / "manifests"), "--dry-run"]
    return dispatch, fixture, spec, [*arguments, *extra]


def test_submit_joint_wrong_digest_still_refuses_without_dev_mode(
        scratch, shared_mount, monkeypatch):
    dispatch, _fixture_, _spec, argv = _joint_submission(
        scratch, extra=["--resume", "--plan-sha256", "0" * 64])
    from experiments import glm_data_manifests
    monkeypatch.setattr(dispatch, "_manifest_producer", lambda: glm_data_manifests)
    with pytest.raises(RuntimeError, match="hashes to"):
        dispatch.main(argv)


def test_submit_joint_digest_arguments_are_records_under_dev_mode(
        scratch, shared_mount, monkeypatch, capsys):
    import shlex
    dispatch, fixture, _spec, argv = _joint_submission(
        scratch, extra=["--resume", "--plan-sha256", "0" * 64])
    from experiments import glm_data_manifests
    monkeypatch.setattr(dispatch, "_manifest_producer", lambda: glm_data_manifests)
    monkeypatch.setenv(DEV_ENV, "1")
    assert dispatch.main(argv) == 0
    out = capsys.readouterr().out
    assert "DEV-MODE" in out
    line = next(line for line in out.splitlines() if line.startswith("[dry-run] "))
    submitted = shlex.split(line[len("[dry-run] "):])
    actual = hashlib.sha256(Path(fixture["plan"]).read_bytes()).hexdigest()
    inner = submitted[submitted.index("prismaquant.tessera_joint_aura"):]
    assert inner[inner.index("--plan-sha256") + 1] == actual  # the record is the ACTUAL digest


def test_submit_joint_run_resume_needs_no_receipt_under_dev_mode(
        scratch, shared_mount, monkeypatch, capsys):
    import shlex
    dispatch, _fixture_, _spec, argv = _joint_submission(
        scratch, extra=["--resume"])  # no --source-transition: no receipt wiring
    from experiments import glm_data_manifests
    monkeypatch.setattr(dispatch, "_manifest_producer", lambda: glm_data_manifests)
    # Certified mode refuses this shape at the pass gates (gates 1-3 above);
    # the submission itself is admitted only under dev mode's contract.
    monkeypatch.setenv(DEV_ENV, "1")
    assert dispatch.main(argv) == 0
    out = capsys.readouterr().out
    line = next(line for line in out.splitlines() if line.startswith("[dry-run] "))
    submitted = shlex.split(line[len("[dry-run] "):])
    inner = submitted[submitted.index("prismaquant.tessera_joint_aura"):]
    assert "--resume" in inner
    assert "--source-transition" not in inner


def _launch_container(monkeypatch, launcher_args):
    from tools import tessera_campaign_container as launcher
    monkeypatch.setattr(launcher, "inspect_or_load", lambda _: [{"Id": "sha256:" + "a" * 64}])
    monkeypatch.setattr(launcher, "image_content_sha256", lambda _: "b" * 64)
    monkeypatch.setattr(launcher, "verify_pinned_import", lambda *args, **kwargs: {})
    monkeypatch.setattr(launcher, "checkout_commit", lambda cwd: None)
    monkeypatch.setattr(launcher, "gpu_attachment", lambda *args, **kwargs: (False, "CPU test"))
    executed = []
    monkeypatch.setattr(launcher.os, "execvp", lambda binary, argv: executed.append(argv))
    launcher.main(launcher_args)
    docker_env = dict(item.split("=", 1) for index, item in enumerate(executed[0])
                      if index and executed[0][index - 1] == "--env")
    return docker_env


def _submitted_launcher_argv(capsys):
    import shlex
    out = capsys.readouterr().out
    line = next(line for line in out.splitlines() if line.startswith("[dry-run] "))
    submitted = shlex.split(line[len("[dry-run] "):])
    return submitted[submitted.index("tools.tessera_campaign_container") + 1:]


def test_dev_mode_env_reaches_the_container_and_the_spec_file_stays_sealed(
        scratch, shared_mount, monkeypatch, capsys):
    dispatch, _fixture_, spec, argv = _joint_submission(scratch)
    from experiments import glm_data_manifests
    monkeypatch.setattr(dispatch, "_manifest_producer", lambda: glm_data_manifests)
    # Certified is "0" since PQ #1147 (unset is dev mode), and the certified
    # submitter seals "0" so the container, which does not inherit this
    # environment, stays certified.
    monkeypatch.setenv(DEV_ENV, "0")
    assert dispatch.main(argv) == 0
    certified = _submitted_launcher_argv(capsys)
    assert "PRISMAQUANT_DEV_MODE=1" not in certified
    docker_env = _launch_container(monkeypatch, certified)
    assert docker_env[dev_mode.DEV_MODE_ENV] == "0"
    original = spec.read_text()

    monkeypatch.setenv(DEV_ENV, "1")
    assert dispatch.main(argv) == 0
    dev = _submitted_launcher_argv(capsys)
    assert spec.read_text() == original  # the sealed spec file is never rewritten
    spec_json = json.loads(dev[dev.index("--spec") + 1])
    assert spec_json["env"][dev_mode.DEV_MODE_ENV] == "1"
    docker_env = _launch_container(monkeypatch, dev)
    assert docker_env[dev_mode.DEV_MODE_ENV] == "1"  # PRISMAQUANT_LAYER_READ_THREADS's road


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


def test_results_json_has_no_dev_keys_without_the_flag(tmp_path, monkeypatch):
    result = _refusing_prepare(tmp_path, monkeypatch)
    assert dev_mode.DEV_UNCERTIFIED_KEY not in result
    assert dev_mode.DEV_MODE_KEY not in result


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


def _progress_record(tmp_path, monkeypatch):
    progress = tmp_path / "progress.json"
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_PATH", str(progress))
    monkeypatch.setenv("PRISMABUILD_ACTION_PROGRESS_TOKEN", "token-fixture")
    assert bridge._pb_commit(3, "head", unit="layers.0.mlp") is True
    return json.loads(progress.read_text())


def test_progress_records_have_no_dev_keys_without_the_flag(tmp_path, monkeypatch):
    record = _progress_record(tmp_path, monkeypatch)
    # Byte-identical certified shape: exactly the six fields the worker reads.
    assert set(record) == {"schema", "token", "phase", "units_completed", "unit",
                           "reported_unix"}


def test_progress_records_carry_the_dev_stamp_only_when_opted_in(tmp_path, monkeypatch):
    monkeypatch.setenv(DEV_ENV, "1")
    # Default: Rob's campaign directive -- no sealing ceremony on progress
    # lines; the certified six-field shape stands even in dev mode.
    record = _progress_record(tmp_path, monkeypatch)
    assert set(record) == {"schema", "token", "phase", "units_completed", "unit",
                           "reported_unix"}
    # Opt-in: the stamp rides along for whoever wants per-line provenance.
    monkeypatch.setenv("PRISMAQUANT_DEV_PROGRESS_STAMP", "1")
    record = _progress_record(tmp_path, monkeypatch)
    assert record[dev_mode.DEV_UNCERTIFIED_KEY] is True
    stamp = record[dev_mode.DEV_MODE_KEY]
    assert stamp[DEV_ENV] == "1"
    assert len(stamp["producer_source_sha256"]) == 64
    assert "timestamp" in stamp
