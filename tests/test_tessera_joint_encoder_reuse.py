"""A historical encoder seal is reusable only where a plan names it.

``tessera.cached_unit.encoder_source_sha256`` hashes every python/cuda/cpp file
of the installed Tessera package, the branches a campaign never measured
included. A campaign priced under an older package therefore cannot pass a
newer package's own seal even when every byte its encoder read -- source
weight, H, calibration, recipe, unit, fixture and wire blob -- is identical.
The full-512 joint panel failed exactly there: the first planned cell
``model.language_model.layers.0.mlp.down_proj@TESSERA_BF16_K1_R1024`` carries
``0833671b...`` while the pinned package derives ``f7cba441...``, and
``verify_cached_unit`` refused the pair.

These regressions hold the bounded answer's two halves. The intake refuses an
unnamed seal before it resolves a single render origin, and a named seal is
admitted visibly: only that one identity field is substituted, the historical
wire record keeps its own bytes, and the receipt carries both observed digests
and the explicit unverified-reuse status.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from prismaquant.cost_stage_checkpoint import canonical_json_sha256, write_unit
from test_tessera_joint_aura import (
    _unlink_render, bind, fixture, installed_encoder_source_sha256,
)

#: The seal the joint panel's own checkpoint carries. The plan must name this
#: value before any of its 197,990 cells may be qualified.
PRODUCTION_HISTORICAL_SEAL = "0833671bbddbc3fb7186bdbed0a905ef248c23ffdf1aeee0c083322680d20f6b"


def _entry(seal, **overrides):
    entry = {"encoder_source_sha256": seal,
             "reason": "the campaign was priced under the historical Tessera package",
             "evidence": "docs/experiments/glm_full_anchor_preparation_20260908/README.md",
             "recorded_unix": 1789621000.0, "recorded_by": "fixture"}
    entry.update(overrides)
    return entry


def _policy(*seals, entries=None):
    from prismaquant.tessera_joint_aura import HISTORICAL_ENCODER_REUSE_SCHEMA

    return {"schema": HISTORICAL_ENCODER_REUSE_SCHEMA,
            "allowlist": list(entries if entries is not None else [_entry(seal) for seal in seals])}


def _reseal(config, states, seal):
    """Re-price one fixture campaign's checkpoint under another encoder seal."""
    path = Path(config["merged_checkpoint"]["path"])
    manifest = json.loads(path.read_text())
    manifest["identity"]["encoder_source_sha256"] = seal
    manifest["identity_sha256"] = canonical_json_sha256(manifest["identity"], where="fixture reseal")
    path.write_text(json.dumps(manifest))
    parts = path.with_name(path.name + ".parts")
    for name, state in states.items():
        for record in state["wire_records"].values():
            record["identity"]["encoder_source_sha256"] = seal
        write_unit(parts, stage="Tessera campaign", qname=name,
                   identity_sha256=manifest["identity_sha256"], state=state)
    config["merged_checkpoint"] = bind(path)
    return manifest["identity_sha256"]


def _tensor_sha256(tensor):
    """A digest of the shard's own bytes, which is what the cell carries."""
    import torch

    return hashlib.sha256(tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def test_the_production_historical_seal_is_not_the_installed_packages_own():
    """The premise of the whole reuse path, held against the live package."""
    assert installed_encoder_source_sha256() != PRODUCTION_HISTORICAL_SEAL


def test_intake_refuses_an_unnamed_seal_before_it_resolves_a_render_origin(tmp_path, monkeypatch):
    """The cheap refusal: no plan entry, no wire decode, no origin walk."""
    from tessera import unit_artifact

    from prismaquant import tessera_joint_aura as bridge

    config, names, fmt, _payload, states = fixture(tmp_path)
    _reseal(config, states, PRODUCTION_HISTORICAL_SEAL)
    _unlink_render(tmp_path, names[0], fmt)
    decoded = []
    monkeypatch.setattr(unit_artifact, "read_unit_artifact",
                        lambda blob, device="cpu": decoded.append(device))

    with pytest.raises(ValueError, match="refusing unverified encoder reuse") as refusal:
        bridge.load_measured_anchor_input(config, verify_payloads=False)

    assert PRODUCTION_HISTORICAL_SEAL in str(refusal.value)
    assert installed_encoder_source_sha256() in str(refusal.value)
    assert decoded == [], "the refusal must precede the per-cell wire decode"


def test_intake_admits_a_named_seal_and_carries_the_reuse_record(tmp_path):
    from prismaquant import tessera_joint_aura as bridge

    config, names, fmt, _payload, states = fixture(tmp_path)
    _reseal(config, states, PRODUCTION_HISTORICAL_SEAL)
    data = bridge.load_measured_anchor_input(
        config, verify_payloads=False,
        historical_encoder_reuse=_policy(PRODUCTION_HISTORICAL_SEAL))

    reuse = data.encoder_source_reuse
    assert reuse["schema"] == bridge.HISTORICAL_ENCODER_REUSE_SCHEMA
    assert reuse["status"] == bridge.ENCODER_REUSE_STATUS
    assert reuse["recorded_encoder_source_sha256"] == PRODUCTION_HISTORICAL_SEAL
    assert reuse["observed_current_encoder_source_sha256"] == installed_encoder_source_sha256()
    assert reuse["allowlist_entry"] == _entry(PRODUCTION_HISTORICAL_SEAL)
    assert set(data.cells) == {(name, fmt) for name in names}
    # A checkpoint that already carries this package's own seal reuses nothing.
    strict, _names, _fmt, _payload, _states = fixture(tmp_path / "own")
    assert bridge.load_measured_anchor_input(strict, verify_payloads=False).encoder_source_reuse is None


def _real_encoded_cell(tmp_path, *, seal, recipe=None, seed=237):
    """One real encoded wire, its real identity, and the bytes it decodes to.

    Through ``encode_tessera_unit`` and the producer's own identity and receipt
    grammar, so the mismatch under test is the production one and not a second
    spelling of it.
    """
    import torch
    from tessera import cached_unit
    from tessera.control import grid_for_name

    from prismaquant.tessera_render import encode_tessera_unit

    name, fmt = "model.language_model.layers.0.mlp.down_proj", "TESSERA_E4M3_K1_R1024"
    weight = torch.randn(64, 64, generator=torch.Generator().manual_seed(seed)).bfloat16()
    render, blob = encode_tessera_unit(weight, fmt, hessian_required=False, verify=False)
    identity = cached_unit.encoding_input_identity(weight, name, grid_for_name("E4M3"), 1024)
    if recipe is not None:
        identity = {**identity, "recipe": recipe}
    record = cached_unit.make_unit_record(
        blob, {**identity, "encoder_source_sha256": seal}, filename=name + ".tessera")
    wire = tmp_path / (name + ".tessera")
    wire.write_bytes(blob)
    anchor = {"qname": name, "format_name": fmt, "family": "TESSERA_E4M3_K1",
              "body_rate_q256": 1024, "dloss": 0.25, "dloss_stderr": 0.0,
              "memory_bytes": len(blob), "bits_per_param": 8.0,
              "activation_contract": "fp8_e4m3", "activation_quantized": True,
              "wire_bytes": len(blob), "seconds": 0.1, "hessian_applied": False,
              "input_global_scale": None}
    cell = {"anchor": anchor, "record": record, "wire": str(wire),
            "render_file_sha256": _tensor_sha256(render),
            "render_origin": "encoded"}
    return cell, weight, render, identity


def _reuse_for(seal):
    from prismaquant.tessera_joint_aura import (
        normalize_historical_encoder_reuse, resolve_encoder_source_reuse)

    return resolve_encoder_source_reuse(
        seal, installed_encoder_source_sha256(),
        normalize_historical_encoder_reuse(_policy(seal)), where="test cell")


def test_the_production_mismatch_refuses_and_the_named_seal_admits(tmp_path):
    """Red first: the real pair fails on the encoder seal alone. Green: named."""
    from prismaquant.tessera_joint_aura import verify_anchor_render

    cell, weight, render, identity = _real_encoded_cell(
        tmp_path, seal=PRODUCTION_HISTORICAL_SEAL)
    kwargs = dict(calibration_source=None, projected_unit=None, static_scales={})

    with pytest.raises(ValueError, match="cached unit encoder_source_sha256 identity mismatch"):
        verify_anchor_render(cell, weight, render, **kwargs)

    reuse = _reuse_for(PRODUCTION_HISTORICAL_SEAL)
    receipt = verify_anchor_render(cell, weight, render, encoder_source_reuse=reuse, **kwargs)
    assert receipt["encoder_source_reuse"] == reuse
    assert receipt["render_comparison"] == "independent_render_vs_wire"
    # Two identities, named separately: the one this package derives, and the
    # one the comparison was actually made against.
    assert receipt["encoding_identity_sha256"] == canonical_json_sha256(
        {**identity, "encoder_source_sha256": PRODUCTION_HISTORICAL_SEAL}, where="historical")
    assert receipt["current_encoding_identity_sha256"] == canonical_json_sha256(
        identity, where="current")
    assert receipt["encoding_identity_sha256"] != receipt["current_encoding_identity_sha256"]
    # The historical record keeps its own bytes and its own seal.
    assert cell["record"]["identity"]["encoder_source_sha256"] == PRODUCTION_HISTORICAL_SEAL
    assert cell["record"]["identity"] == {
        **identity, "encoder_source_sha256": PRODUCTION_HISTORICAL_SEAL}


def test_only_the_encoder_seal_is_substituted(tmp_path):
    """Every other identity field stays mandatory, including the recipe."""
    from prismaquant.tessera_joint_aura import verify_anchor_render

    cell, weight, render, identity = _real_encoded_cell(
        tmp_path, seal=PRODUCTION_HISTORICAL_SEAL)
    other = {**cell["record"], "identity": {
        **cell["record"]["identity"],
        "recipe": {**identity["recipe"], "q256": 960}}}
    with pytest.raises(ValueError, match="cached unit recipe identity mismatch"):
        verify_anchor_render({**cell, "record": other}, weight, render,
                             calibration_source=None, projected_unit=None, static_scales={},
                             encoder_source_reuse=_reuse_for(PRODUCTION_HISTORICAL_SEAL))


def test_a_reuse_record_from_another_installed_package_is_refused(tmp_path):
    """The substitution names the package it was observed on, and no other."""
    from prismaquant.tessera_joint_aura import verify_anchor_render

    cell, weight, render, _identity = _real_encoded_cell(
        tmp_path, seal=PRODUCTION_HISTORICAL_SEAL)
    stale = dict(_reuse_for(PRODUCTION_HISTORICAL_SEAL),
                 observed_current_encoder_source_sha256="0" * 64)
    with pytest.raises(ValueError, match="installed encoder source seal"):
        verify_anchor_render(cell, weight, render, calibration_source=None,
                             projected_unit=None, static_scales={},
                             encoder_source_reuse=stale)


@pytest.mark.parametrize("entry,pattern", [
    (_entry(PRODUCTION_HISTORICAL_SEAL, encoder_source_sha256="not-a-digest"), "64-hex"),
    (_entry(PRODUCTION_HISTORICAL_SEAL, encoder_source_sha256="A" * 64), "64-hex"),
    (_entry(PRODUCTION_HISTORICAL_SEAL, reason="   "), "non-empty reason"),
    (_entry(PRODUCTION_HISTORICAL_SEAL, evidence=""), "non-empty evidence"),
    (_entry(PRODUCTION_HISTORICAL_SEAL, recorded_unix=float("inf")), "finite non-negative"),
    (_entry(PRODUCTION_HISTORICAL_SEAL, recorded_unix=float("nan")), "finite non-negative"),
    (_entry(PRODUCTION_HISTORICAL_SEAL, recorded_unix=-1), "finite non-negative"),
    (_entry(PRODUCTION_HISTORICAL_SEAL, recorded_unix="1789621000"), "finite non-negative"),
    (_entry(PRODUCTION_HISTORICAL_SEAL, recorded_by=None), "non-empty recorded_by"),
    ({**_entry(PRODUCTION_HISTORICAL_SEAL), "extra": 1}, "requires exactly"),
])
def test_the_reuse_grammar_refuses_every_open_spelling(entry, pattern):
    from prismaquant.tessera_joint_aura import (
        HISTORICAL_ENCODER_REUSE_SCHEMA, normalize_historical_encoder_reuse)

    with pytest.raises(ValueError, match=pattern):
        normalize_historical_encoder_reuse(
            {"schema": HISTORICAL_ENCODER_REUSE_SCHEMA, "allowlist": [entry]})


def test_the_reuse_grammar_has_no_wildcard_or_duplicate_spelling():
    from prismaquant.tessera_joint_aura import (
        HISTORICAL_ENCODER_REUSE_SCHEMA, normalize_historical_encoder_reuse)

    assert normalize_historical_encoder_reuse(None) is None
    for block in ({"schema": HISTORICAL_ENCODER_REUSE_SCHEMA},
                  {"schema": HISTORICAL_ENCODER_REUSE_SCHEMA, "allowlist": []},
                  {"schema": "other", "allowlist": [_entry(PRODUCTION_HISTORICAL_SEAL)]},
                  {"schema": HISTORICAL_ENCODER_REUSE_SCHEMA,
                   "allowlist": [_entry(PRODUCTION_HISTORICAL_SEAL), "all"]},
                  {"schema": HISTORICAL_ENCODER_REUSE_SCHEMA,
                   "allowlist": [_entry(PRODUCTION_HISTORICAL_SEAL),
                                 _entry(PRODUCTION_HISTORICAL_SEAL)]},
                  {"schema": HISTORICAL_ENCODER_REUSE_SCHEMA,
                   "allowlist": [_entry(PRODUCTION_HISTORICAL_SEAL)], "strict": False}):
        with pytest.raises(ValueError):
            normalize_historical_encoder_reuse(block)


def test_a_plan_that_names_a_malformed_reuse_block_is_refused_at_load(tmp_path):
    """The plan grammar is where a reuse is admitted, so it is checked there."""
    from prismaquant.tessera_joint_aura import SCHEMA, _load_plan

    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"schema": SCHEMA,
                                "historical_encoder_reuse": {"schema": "other", "allowlist": []}}))
    with pytest.raises(ValueError, match="historical encoder reuse schema"):
        _load_plan(path, hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("command", ["prepare", "run"])
def test_execute_passes_the_plan_policy_to_the_intake(tmp_path, monkeypatch, command):
    """Strict default unchanged; a named policy reaches the one intake call."""
    import torch
    from types import SimpleNamespace

    from prismaquant import (
        aura_cost, calibration_data, cost_streaming, gpu_guard, memory_management)
    from prismaquant import model_profiles
    from prismaquant import tessera_joint_aura as bridge

    monkeypatch.setattr(model_profiles, "detect_profile", lambda _path: object())
    monkeypatch.setattr(gpu_guard, "require_cuda_hot_path", lambda *_args: None)
    # ``execute`` reads the plan's envelope before it calls the allocator; this
    # config is not an admitted plan, so the field is supplied and the CUDA
    # half is stubbed rather than exercised on a CPU-only box. The stub is
    # declared only where the envelope exists: a tree whose ``memory_management``
    # has no device envelope runs this path unarmed, and the encoder reuse this
    # file gates does not depend on it either way.
    monkeypatch.setattr(memory_management, "enforce_device_envelope",
                        lambda *_args, **_kwargs: {"enforced": False, "reason": "cpu fixture"},
                        raising=False)
    draw = dict(fit_ids_sha256="a" * 64, text_sha256="b" * 64, nsamples=512, seqlen=512, seed=0)
    seen = []

    def intake(_inputs, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(census={"model": "fixture", "attention_implementation": "eager"},
            cells={}, unit_scope=None, render_mirror_root=None, synthesized_now=0,
            encoder_source_reuse=None, head_walk_workers=None, head_walk_resumed_units=0,
            payload={"provenance": {"hessian": {"calibration_identity": draw}}})

    monkeypatch.setattr(bridge, "load_measured_anchor_input", intake)
    if command == "run":
        monkeypatch.setattr(bridge, "_preflight_run_prepared", lambda *_a, **_k: None)
    monkeypatch.setattr(calibration_data, "load_calibration_input", lambda *_a, **_k:
        (torch.zeros((512, 512), dtype=torch.int64), {"provenance": draw}))
    monkeypatch.setattr(aura_cost, "_aura_source_sha256", lambda: "i" * 64)
    monkeypatch.setattr(bridge, "_prepare_source_owner",
                        lambda *_a, **_k: SimpleNamespace(close=lambda: None))

    class Reached(Exception):
        pass

    def reached(*_args, **_kwargs):
        raise Reached

    monkeypatch.setattr(cost_streaming, "build_streamed_causal_lm", reached)
    base = {"model": "fixture", "inputs": {}, "output_root": str(tmp_path),
            "calibration_input": {"path": "fixture", "sha256": "a" * 64},
            "source_prefetch": dict(max_cache_slots=24, prefetch_workers=4,
                prefetch_lookahead=4, cache_headroom_gb=4.0,
                prefetch_min_available_gb=2.0, require_prefetched_residency=True),
            "max_gpu_bytes": 1 << 30,
            "execution": {"production_act_scales": "0", "n_calib_samples": 512,
                          "calib_seqlen": 512}}
    policy = _policy(PRODUCTION_HISTORICAL_SEAL)
    for config in (base, {**base, "historical_encoder_reuse": policy}):
        seen.clear()
        with pytest.raises(Reached):
            bridge.execute(command, config, plan_sha256="b" * 64)
        assert seen[0].get("historical_encoder_reuse") == (
            None if config is base else policy)


def test_synthesize_reports_the_same_reuse_it_read(tmp_path, monkeypatch):
    """The standalone decode stage cannot qualify cells under an unstated seal."""
    from types import SimpleNamespace

    from prismaquant import tessera_joint_aura as bridge

    config, _names, _fmt, _payload, states = fixture(tmp_path)
    _reseal(config, states, PRODUCTION_HISTORICAL_SEAL)
    policy = _policy(PRODUCTION_HISTORICAL_SEAL)
    seen = []

    def intake(_inputs, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(formats_by_qname={}, cells={}, synthesized_now=0,
                               render_mirror_root=None,
                               encoder_source_reuse=bridge.resolve_encoder_source_reuse(
                                   PRODUCTION_HISTORICAL_SEAL, installed_encoder_source_sha256(),
                                   bridge.normalize_historical_encoder_reuse(policy),
                                   where="fixture"))

    monkeypatch.setattr(bridge, "load_measured_anchor_input", intake)
    record = bridge.synthesize_renders(
        {"inputs": config, "historical_encoder_reuse": policy},
        plan_sha256="p" * 64, mirror_root=tmp_path / "mirror")
    assert seen[0]["historical_encoder_reuse"] == policy
    assert record["encoder_source_reuse"]["status"] == bridge.ENCODER_REUSE_STATUS
