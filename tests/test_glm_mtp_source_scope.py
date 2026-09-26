"""GLM MTP layer 45: the selected-source campaign prices it (PQ #1316, M3).

The MTP layer lives outside the decoder body, so the body's streamed loader
drops its keys. A profile-declared source scope (``ModelProfile.source_scope``)
lets the same loader -- weight map, packer, authentication, layer cache and
prefetch pool -- read that layer for a one-shot selected snapshot, and a
campaign row with ``--source-scope mtp`` prices its units from the MTP capture
both capture phases publish.

Each test runs on the tiny checkpoint of ``test_glm_mtp_capture``: a two-layer
body plus its MTP layer, captured through the real CLI phases.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip(
    "transformers.models.glm5_next",
    reason="glm5_next requires transformers >= 5.16",
)

from safetensors.torch import save_file  # noqa: E402

from prismaquant import glm_mtp  # noqa: E402
from tests.test_glm5_next_streamed_forward_parity import _torch_only_causal_conv1d  # noqa: E402,F401
from tests.test_glm_mtp_capture import (  # noqa: E402,F401
    BACKBONE, CALIBRATION_TEXT, LAST, MAX_ROWS, N_SEQUENCES, PREFIX, PROFILE, SEQ_LEN,
    _calibration_rows, _mtp_routed, _runner, _sha, _write_boundaries, mtp_source,
)

SHARED_DOWN = f"{PREFIX}mlp.shared_experts.down_proj"
STACK = f"{PREFIX}mlp.experts"


def _published_mtp_capture(env, monkeypatch):
    """Both capture phases and the projection, through ``tools/glm_mtp_capture``."""
    from prismaquant import joint_cost_quantum
    from tools import glm_mtp_capture as cli

    reference = _runner(env, name="offload-reference")
    try:
        with torch.inference_mode():
            boundaries = reference.capture_boundaries(env.ids)
    finally:
        reference.shutdown()
    draw = {"fit_ids_sha256": hashlib.sha256(env.ids.to(torch.int32).numpy().tobytes()).hexdigest(),
            "fit_tokens": env.ids.numel(), "nsamples": N_SEQUENCES, "seqlen": SEQ_LEN}
    tokens = env.root / "calibration_tokens.safetensors"
    save_file({"calibration_ids": env.ids}, str(tokens),
              metadata={"calibration_provenance": json.dumps(draw)})
    manifest = _write_boundaries(env, boundaries.activations_cpu[LAST])
    manifest["calibration_sha256"] = hashlib.sha256(env.ids.contiguous().numpy().tobytes()).hexdigest()
    boundary_path = env.root / "boundaries.json"
    boundary_path.write_text(json.dumps(manifest))
    plan = {"model": str(env.source),
            "calibration_input": {"path": str(tokens), "sha256": _sha(tokens)},
            "canonical_capture": {"path": env.capture["path"], "sha256": env.capture["sha256"]},
            "inputs": {"census": {"path": str(env.census_path), "sha256": _sha(env.census_path)}},
            "execution": {"n_calib_samples": N_SEQUENCES, "calib_seqlen": SEQ_LEN}}
    plan_path = env.root / "plan.json"
    plan_path.write_text(json.dumps(plan))
    prepared = {"source_execution": {"modules": {
        f"model.language_model.layers.{LAST}.mlp.experts": {"experts": "eager"}}}}
    prepared_path = env.root / "prepared.json"
    prepared_path.write_text(json.dumps(prepared))
    monkeypatch.setattr(
        joint_cost_quantum, "build_quantum_source_runner",
        lambda config, *, offload_folder, source_authentication: _runner(
            env, source_authentication=source_authentication, name="offload-cli"))
    bound = ["--plan", str(plan_path), "--plan-sha256", _sha(plan_path)]
    final = env.root / "final-hidden"
    assert cli.main(["--phase", "final-hidden", *bound, "--boundaries", str(boundary_path),
                     "--boundaries-sha256", _sha(boundary_path), "--out", str(final),
                     "--offload-folder", str(env.root / "unused"), "--read-ahead-mb", "1"]) == 0
    projection = env.root / "projection"
    assert cli.main(["--phase", "projection", *bound, "--out", str(projection)]) == 0
    census_path = env.root / "mtp-census.json"
    capture = env.root / "mtp-capture"
    assert cli.main(["--phase", "capture", *bound, "--prepared", str(prepared_path),
                     "--prepared-sha256", _sha(prepared_path),
                     "--final-hidden", str(final / "manifest.json"),
                     "--final-hidden-sha256", _sha(final / "manifest.json"),
                     "--expert-projection", str(projection / "mtp-projection.json"),
                     "--expert-projection-sha256", _sha(projection / "mtp-projection.json"),
                     "--out", str(capture), "--census-out", str(census_path),
                     "--device", "cpu", "--read-ahead-mb", "1"]) == 0
    report = json.loads((capture / "capture-run.json").read_text())
    return SimpleNamespace(census_path=census_path, census=json.loads(census_path.read_text()),
                           capture=report["capture"])


def _loaded_mtp_weights(env, dtype):
    """Every priced MTP unit's weight, from the MTP layer's own loader."""
    from prismaquant.routed_experts import profile_declared_packed_expert_projections

    layer, _ = glm_mtp.load_mtp_layer(env.source, env.text_config, profile=PROFILE, dtype=dtype,
                                      experts_implementation="eager")
    wrapper = glm_mtp.MtpCheckpointModel(layer)
    modules = dict(wrapper.named_modules())
    weights = {member.qname: member.weight.detach().clone()
               for member in profile_declared_packed_expert_projections(wrapper, PROFILE)}
    for name in glm_mtp.mtp_priced_units(wrapper, PROFILE):
        if name not in weights:
            weights[name] = modules[name].weight.detach().clone()
    return weights


@pytest.mark.parametrize("mtp_source", [torch.bfloat16], indirect=True)
def test_scoped_source_snapshots_the_mtp_layer_through_the_body_loader(mtp_source, monkeypatch):
    """The ``mtp`` scope reads layer 45's selected tensors with the body's
    loader, under their checkpoint names, and nothing else."""
    from prismaquant import tessera_calibration_cache as cc
    from prismaquant.autoscale import selected_anchor_resources
    from prismaquant.cost_streaming import build_streamed_causal_lm

    env = mtp_source
    published = _published_mtp_capture(env, monkeypatch)
    units = [SHARED_DOWN, f"{STACK}.1.gate_proj", f"{STACK}.4.down_proj"]
    census = published.census
    resources = selected_anchor_resources(
        str(env.source), unit_shapes={name: census["unit_shapes"][name] for name in units},
        counts=census["counts"], max_act_rows=MAX_ROWS, cache_slots=2, prefetch_workers=1,
        headroom_gb=0, source_snapshot_policy="selected-tensors-v1", source_scope="mtp")
    assert resources["source_tensor_keys"]
    assert all(key.startswith(PREFIX) for key in resources["source_tensor_keys"])

    owner = cc.authenticate_selected_capture_source(
        published.census_path, published.capture["path"],
        expected_sha256=published.capture["sha256"], model=str(env.source),
        max_act_rows=MAX_ROWS, attention_implementation="eager")
    try:
        runner = build_streamed_causal_lm(
            str(env.source), device=torch.device("cpu"), dtype=torch.bfloat16,
            offload_folder=str(env.root / "scoped-offload"), profile=PROFILE, max_cache_slots=2,
            prefetch_workers=1, prefetch_min_available_gb=0, cache_headroom_gb=0,
            prefetch_lookahead=1, require_prefetched_residency=True,
            attn_implementation="eager", source_authentication=owner,
            source_snapshot_only=True, source_scope="mtp")
        try:
            assert runner.num_layers == BACKBONE + 1
            assert runner.source_layers == (BACKBONE,)
            assert all(runner.layer_index_for_qname(name) == BACKBONE for name in units)
            with pytest.raises(RuntimeError, match="cannot execute a forward"):
                runner(env.ids)
            weights, receipt = runner.snapshot_selected_weights(
                units, max_resident_bytes=resources["selected_source_weight_bytes"],
                expected_source_keys=resources["source_tensor_keys"])
        finally:
            runner.shutdown()
        authentication = owner.receipt()
    finally:
        owner.close()

    expected = _loaded_mtp_weights(env, torch.bfloat16)
    assert set(weights) == set(units)
    for name in units:
        assert torch.equal(weights[name], expected[name]), name
    assert receipt["source_forward_count"] == 0
    assert [row["layer"] for row in receipt["layers"]] == [BACKBONE]
    assert receipt["source_tensor_keys"] == list(resources["source_tensor_keys"])
    assert receipt["nonbody_materialized"] is False
    assert {row["name"] for row in authentication["verified_files"]} == {"model-mtp.safetensors"}


#: ``runner.`` uses of the whole-scope streamed calibration, which never runs
#: on a scoped source (its runner cannot execute a forward).
WHOLE_SCOPE_RUNNER_USES = {"device", "model", "prefetch_lookahead", "visit_layer_batches"}


@pytest.mark.parametrize("mtp_source", [torch.bfloat16], indirect=True)
def test_both_runners_satisfy_the_selected_source_protocol(mtp_source, monkeypatch):
    """``SelectedSource`` names what selected-source consumers may use: the
    body runner and the scoped runner both satisfy it, and every other
    ``runner.`` use in the campaign belongs to the whole-scope forward."""
    import ast
    import inspect

    from prismaquant import tessera_calibration_cache as cc
    from prismaquant import tessera_campaign
    from prismaquant.cost_streaming import (SELECTED_SOURCE_MEMBERS, SelectedSource,
                                            build_streamed_causal_lm)

    assert set(SELECTED_SOURCE_MEMBERS) == {
        "source_layers", "num_layers", "layer_index_for_qname", "snapshot_selected_weights",
        "shutdown", "context"}
    env = mtp_source
    body = _runner(env, name="offload-body")
    try:
        assert isinstance(body, SelectedSource)
        assert body.source_layers == tuple(range(body.num_layers))
    finally:
        body.shutdown()
    published = _published_mtp_capture(env, monkeypatch)
    owner = cc.authenticate_selected_capture_source(
        published.census_path, published.capture["path"],
        expected_sha256=published.capture["sha256"], model=str(env.source),
        max_act_rows=MAX_ROWS, attention_implementation="eager")
    try:
        scoped = build_streamed_causal_lm(
            str(env.source), device=torch.device("cpu"), dtype=torch.bfloat16,
            offload_folder=str(env.root / "scoped"), profile=PROFILE, max_cache_slots=2,
            prefetch_workers=1, prefetch_min_available_gb=0, cache_headroom_gb=0,
            prefetch_lookahead=1, attn_implementation="eager", source_authentication=owner,
            source_snapshot_only=True, source_scope="mtp")
        try:
            assert isinstance(scoped, SelectedSource)
        finally:
            scoped.shutdown()
    finally:
        owner.close()

    tree = ast.parse(inspect.getsource(tessera_campaign))
    used = {node.attr for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "runner"}
    assert used - WHOLE_SCOPE_RUNNER_USES <= set(SELECTED_SOURCE_MEMBERS), used


@pytest.mark.parametrize("mtp_source", [torch.bfloat16], indirect=True)
def test_campaign_row_prices_mtp_units_from_the_mtp_capture(mtp_source, monkeypatch, tmp_path):
    """A ``--source-scope mtp`` row prices the shared expert and the routed
    stack from the MTP capture; its recomputed capture identity is the
    manifest's, and it reads no body layer."""
    from prismaquant import tessera_campaign as campaign

    env = mtp_source
    published = _published_mtp_capture(env, monkeypatch)
    census = published.census
    groups = [f"u:{SHARED_DOWN}", f"s:{STACK}"]
    selection = tmp_path / "units.json"
    selection.write_text(json.dumps(dict(
        schema=campaign.UNITS_SCHEMA, model=str(env.source), layer_stride=1,
        groups=[dict(key=key, members=census["anchor_groups"][key]) for key in groups])))
    monkeypatch.setattr(campaign, "_calibration_tokens",
                        lambda *_args, **_kwargs: (_calibration_rows(env.ids), CALIBRATION_TEXT))

    def no_forward(*_args, **_kwargs):
        pytest.fail("a selected MTP row must not run a calibration forward")

    monkeypatch.setattr(campaign, "_collect_activations", no_forward)
    original_menus = campaign.expand_menus_for_targets

    def one_rung(weights, targets, **kwargs):
        menus = original_menus(weights, targets, **kwargs)
        return {name: [row for row in rows if row.format_name == "TESSERA_E4M3_K1_R1024"]
                for name, rows in menus.items()}

    monkeypatch.setattr(campaign, "expand_menus_for_targets", one_rung)
    out = tmp_path / "mtp-cost.pkl"
    assert campaign.main([
        "--model", str(env.source), "--out", str(out), "--cache-dir", str(tmp_path / "cache"),
        "--menu-mode", "research", "--nsamples", str(N_SEQUENCES), "--seqlen", str(SEQ_LEN),
        "--max-act-rows", str(MAX_ROWS), "--attention-implementation", "eager",
        "--streaming", "--streaming-cache-headroom-gb", "0", "--units", str(selection),
        "--calibration-census", str(published.census_path),
        "--calibration-cache", published.capture["path"],
        "--calibration-cache-sha256", published.capture["sha256"],
        "--source-snapshot-policy", "selected-tensors-v1", "--source-scope", "mtp",
        "--max-rounds", "1"]) == 0
    with out.open("rb") as handle:
        payload = pickle.load(handle)
    priced = {name for key in groups for name in census["anchor_groups"][key]}
    assert set(payload["costs"]) == priced
    assert {f"{STACK}.{e}.{p}" for e in range(int(env.text_config.n_routed_experts))
            for p in ("gate_proj", "up_proj", "down_proj")} <= priced
    receipt = payload["provenance"]["selected_source_preparation"]
    assert receipt["source_forward_count"] == 0
    assert [row["layer"] for row in receipt["layers"]] == [BACKBONE]
    assert all(key.startswith(PREFIX) for key in receipt["source_tensor_keys"])
