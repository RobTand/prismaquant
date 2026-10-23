"""The derived-capture CLI preserves certified and dev source policy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.glm5_next")
from safetensors.torch import save_file

from tests import test_glm_mtp_capture as fixture
from tests.test_glm_mtp_capture import _torch_only_causal_conv1d  # noqa: F401


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value))
    return path


def _body_inputs(env):
    reference = fixture._runner(env, name="audit-reference")
    try:
        with torch.inference_mode():
            boundaries = reference.capture_boundaries(env.ids)
    finally:
        reference.shutdown()
    draw = {
        "fit_ids_sha256": hashlib.sha256(env.ids.to(torch.int32).numpy().tobytes()).hexdigest(),
        "fit_tokens": env.ids.numel(), "nsamples": fixture.N_SEQUENCES,
        "seqlen": fixture.SEQ_LEN,
    }
    tokens = env.root / "calibration_tokens.safetensors"
    save_file({"calibration_ids": env.ids}, str(tokens),
              metadata={"calibration_provenance": json.dumps(draw)})
    boundary = fixture._write_boundaries(env, boundaries.activations_cpu[fixture.LAST])
    boundary["calibration_sha256"] = hashlib.sha256(env.ids.contiguous().numpy().tobytes()).hexdigest()
    boundary_path = _write_json(env.root / "boundaries.json", boundary)
    plan = {
        "model": str(env.source),
        "calibration_input": {"path": str(tokens), "sha256": _sha(tokens)},
        "canonical_capture": {key: env.capture[key] for key in ("path", "sha256")},
        "inputs": {"census": {"path": str(env.census_path), "sha256": _sha(env.census_path)}},
        "execution": {"n_calib_samples": fixture.N_SEQUENCES, "calib_seqlen": fixture.SEQ_LEN},
    }
    plan_path = _write_json(env.root / "plan.json", plan)
    prepared = _write_json(env.root / "prepared.json", {
        "source_execution": {"modules": {
            f"model.language_model.layers.{fixture.LAST}.mlp.experts": {"experts": "eager"},
        }},
    })
    return SimpleNamespace(env=env, plan=plan, plan_path=plan_path,
                           boundary_path=boundary_path, prepared=prepared)


def _publish(cli, inputs):
    env = inputs.env
    final_dir = env.root / "cli-final-hidden"
    assert cli.main([
        "--phase", "final-hidden", "--plan", str(inputs.plan_path),
        "--plan-sha256", _sha(inputs.plan_path),
        "--boundaries", str(inputs.boundary_path),
        "--boundaries-sha256", _sha(inputs.boundary_path),
        "--out", str(final_dir), "--offload-folder", str(env.root / "cli-offload"),
        "--read-ahead-mb", "1",
    ]) == 0
    projection_dir = env.root / "cli-projection"
    assert cli.main([
        "--phase", "projection", "--plan", str(inputs.plan_path),
        "--plan-sha256", _sha(inputs.plan_path), "--out", str(projection_dir),
    ]) == 0
    inputs.final = final_dir / "manifest.json"
    inputs.projection = projection_dir / "mtp-projection.json"
    return inputs


@pytest.fixture
def source_pair(tmp_path, monkeypatch):
    from prismaquant import joint_cost_quantum
    from tools import glm_mtp_capture as cli

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    request = SimpleNamespace(param=torch.bfloat16)
    a_root, b_root = tmp_path / "a", tmp_path / "b"
    a_root.mkdir()
    b_root.mkdir()
    a = fixture.mtp_source.__wrapped__(request, a_root, monkeypatch)
    original = fixture._build_tiny_model

    def changed_source():
        model = original()
        with torch.no_grad():
            model.model.language_model.embed_tokens.weight[:, 0].add_(0.25)
        return model

    with monkeypatch.context() as patch:
        patch.setattr(fixture, "_build_tiny_model", changed_source)
        b = fixture.mtp_source.__wrapped__(request, b_root, patch)
    assert torch.equal(a.ids, b.ids)
    assert _sha(a.source / "model.safetensors") != _sha(b.source / "model.safetensors")
    a_manifest = json.loads(Path(a.capture["path"]).read_text())
    b_manifest = json.loads(Path(b.capture["path"]).read_text())
    assert a_manifest["identity"]["source_files"] != b_manifest["identity"]["source_files"]
    inputs = [_body_inputs(env) for env in (a, b)]
    by_source = {str(item.env.source): item for item in inputs}

    def actual_tiny_runner(config, *, offload_folder, source_authentication):
        item = by_source[config["model"]]
        assert config == item.plan
        assert source_authentication is not None
        return fixture._runner(item.env, source_authentication=source_authentication,
                               name="audit-cli-source")

    monkeypatch.setattr(joint_cost_quantum, "build_quantum_source_runner", actual_tiny_runner)
    inputs = [_publish(cli, item) for item in inputs]
    finals = [json.loads(item.final.read_text()) for item in inputs]
    assert finals[0]["inputs"]["calibration_sha256"] == finals[1]["inputs"]["calibration_sha256"]
    assert finals[0]["inputs"]["canonical_capture"] != finals[1]["inputs"]["canonical_capture"]
    owned = {path: _sha(path) for item in inputs for path in item.env.root.rglob("*")
             if path.is_file() and "offload" not in str(path.relative_to(item.env.root))}
    return SimpleNamespace(a=inputs[0], b=inputs[1], cli=cli, owned=owned)


@pytest.mark.parametrize("dev", [False, True], ids=["certified", "dev"])
@pytest.mark.parametrize("different", [False, True], ids=["same_source", "other_source"])
def test_real_cli_final_hidden_source_policy(source_pair, monkeypatch, capsys, dev, different):
    pair = source_pair
    current = pair.b
    selected = pair.a if different else pair.b
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1" if dev else "0")
    capsys.readouterr()
    out = current.env.root / "audit-mtp-capture"
    census = current.env.root / "audit-mtp-census.json"
    argv = [
        "--phase", "capture", "--plan", str(current.plan_path),
        "--plan-sha256", _sha(current.plan_path),
        "--prepared", str(current.prepared), "--prepared-sha256", _sha(current.prepared),
        "--final-hidden", str(selected.final), "--final-hidden-sha256", _sha(selected.final),
        "--expert-projection", str(current.projection),
        "--expert-projection-sha256", _sha(current.projection),
        "--out", str(out), "--census-out", str(census), "--device", "cpu",
        "--read-ahead-mb", "1",
    ]
    try:
        if different and not dev:
            with pytest.raises(RuntimeError, match="(?i)(source|model|capture)"):
                pair.cli.main(argv)
            assert not census.exists()
            return
        assert pair.cli.main(argv) == 0
        report = json.loads((out / "capture-run.json").read_text())
        published = json.loads(census.read_text())
        assert report["units"] > 0
        assert report["source_authentication"]["census_sha256"] == _sha(current.env.census_path)
        assert published["model"] == str(current.env.source)
        assert published["mtp_extension"]["final_hidden"]["sha256"] == _sha(selected.final)
        if different:
            lines = capsys.readouterr().out.splitlines()
            assert any("[DEV-MODE]" in line and "hidden" in line.lower()
                       and ("source" in line.lower() or "model" in line.lower()) for line in lines), lines
            assert report["source_authentication"]["dev_uncertified"] is True
    finally:
        assert all(_sha(path) == digest for path, digest in pair.owned.items())


class _AdmissionReached(Exception):
    pass


def _pricing_intake(current, selected, prefetch):
    final = json.loads(selected.final.read_text())
    layer = int(current.env.mtp.layer_idx)
    runner = SimpleNamespace(source_layers=(layer,),
                             context=SimpleNamespace(schedule_prefetch=prefetch))
    data = SimpleNamespace(census={"mtp_extension": {
        "layer": layer,
        "final_hidden": {"path": str(selected.final), "sha256": _sha(selected.final)},
    }})
    config = {"canonical_capture": current.plan["canonical_capture"]}
    calibration = {"calibration_sha256": final["inputs"]["calibration_sha256"]}
    return runner, data, config, calibration


@pytest.mark.parametrize("dev", [False, True], ids=["certified", "dev"])
@pytest.mark.parametrize("different", [False, True], ids=["same_source", "other_source"])
def test_real_pricing_admits_source_before_prefetch(
        source_pair, monkeypatch, capsys, dev, different):
    from prismaquant import glm_mtp_quantum

    pair = source_pair
    current = pair.b
    selected = pair.a if different else pair.b
    prefetched = []

    def after_admission(layer):
        prefetched.append(layer)
        raise _AdmissionReached

    runner, data, config, calibration = _pricing_intake(current, selected, after_admission)
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1" if dev else "0")
    capsys.readouterr()
    expected = RuntimeError if different and not dev else _AdmissionReached
    try:
        with pytest.raises(expected) as error:
            glm_mtp_quantum.run_mtp_scope(
                runner, config=config, data=data, production_cache=None,
                calibration_ids=current.env.ids, calibration=calibration,
                source_model={}, projection_backend=None, operator_windows={}, device_bytes=1)
        if different and not dev:
            assert "source model" in str(error.value)
            assert not prefetched
        else:
            assert prefetched == [int(current.env.mtp.layer_idx)]
        if different and dev:
            lines = capsys.readouterr().out.splitlines()
            assert any("[DEV-MODE]" in line and "final-hidden source" in line for line in lines), lines
    finally:
        assert all(_sha(path) == digest for path, digest in pair.owned.items())


@pytest.mark.parametrize("dev", [False, True], ids=["certified", "dev"])
def test_pricing_calibration_stays_strict(source_pair, monkeypatch, dev):
    from prismaquant import glm_mtp_quantum

    def unexpected_prefetch(layer):
        pytest.fail(f"prefetched layer {layer} before calibration admission")

    current = source_pair.b
    runner, data, config, calibration = _pricing_intake(current, current, unexpected_prefetch)
    calibration["calibration_sha256"] = "0" * 64
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1" if dev else "0")
    with pytest.raises(RuntimeError, match="another calibration draw"):
        glm_mtp_quantum.run_mtp_scope(
            runner, config=config, data=data, production_cache=None,
            calibration_ids=current.env.ids, calibration=calibration,
            source_model={}, projection_backend=None, operator_windows={}, device_bytes=1)


def test_same_source_can_have_an_independent_canonical_capture(source_pair):
    from prismaquant import glm_mtp_capture as cap
    from prismaquant import tessera_calibration_cache as cc

    current = source_pair.b
    env = current.env
    name = fixture.DENSE_UNIT
    identity = dict(env.canonical, max_act_rows=env.canonical["max_act_rows"] + 1)
    count = int(env.census["counts"][name])
    rows = torch.randn(min(count, identity["max_act_rows"]), env.census["unit_shapes"][name][1],
                       generator=torch.Generator().manual_seed(43))
    capture = cc.publish_capture(
        env.root / "independent-canonical", census_path=env.census_path, identity=identity,
        acts={name: rows}, hessians={name: rows.T @ rows},
        counts=env.census["counts"], maxima=env.census["max_abs"])
    assert capture["sha256"] != current.plan["canonical_capture"]["sha256"]
    final, _ = cap.read_bound_json(current.final, _sha(current.final))
    assert cap.require_final_hidden_source(
        final, capture, where="independent-capture source control") is True
    assert all(_sha(path) == digest for path, digest in source_pair.owned.items())
