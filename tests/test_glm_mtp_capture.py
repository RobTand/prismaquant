"""GLM MTP layer 45: the capture's model, weight loader and inputs (PQ #1290).

The capture reuses the body campaign's collector (`_collect_activations`)
unchanged, so what needs checking is everything that feeds it:

* the wrapper that puts the MTP layer under its checkpoint name, so the
  profile declares its packed experts exactly as it declares a body layer's;
* the loader that turns the checkpoint's per-expert tensors into the packed
  parameters the layer runs, through the profile's own bridge;
* the target's post-final-norm hidden state, computed from a layer boundary
  on the streamed runner the body used;
* the feed that pairs each calibration row with its own hidden state.

A wrong pairing, a swapped gate/up pack or a missing final norm all produce
plausible tensors, so each is checked against an independent computation on a
tiny random config, on CPU.

The last tests run both capture phases end to end on a tiny checkpoint that
carries its MTP layer, through the real streamed loader and the canonical
capture writer, and check what a later consumer relies on: the capture's
identity inherits the canonical source roster, names the MTP layer's own
load contract, and authenticates as a selected-source capture.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from contextlib import closing
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip(
    "transformers.models.glm5_next",
    reason="glm5_next requires transformers >= 5.16",
)

from safetensors.torch import save_file  # noqa: E402

from prismaquant import genuine_weight_initialization  # noqa: E402
from prismaquant import glm_mtp, glm_mtp_capture  # noqa: E402
from prismaquant.export_native_compressed import _split_packed_expert_tensor  # noqa: E402
from prismaquant.model_profiles.glm5_next import Glm5NextProfile  # noqa: E402
from prismaquant.routed_experts import profile_declared_packed_expert_projections  # noqa: E402
from prismaquant.tessera_campaign import _collect_activations  # noqa: E402
from tests.test_glm5_next_streamed_forward_parity import (  # noqa: E402
    _build_tiny_model, _streamed_runner, _torch_only_causal_conv1d,  # noqa: F401
)
from tests.test_glm_mtp_layer import (  # noqa: E402
    BACKBONE, VOCAB, _randomize, _text_config,
)
from tests.test_glm_campaign_streaming import write_original_layout_checkpoint  # noqa: E402

PROFILE = Glm5NextProfile()
PREFIX = f"model.language_model.layers.{BACKBONE}."
EXPERTS = 6
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


@pytest.fixture(scope="module")
def layer():
    with genuine_weight_initialization():
        built = glm_mtp.Glm5NextMtpLayer(_text_config())
    _randomize(built, 20260926)
    with torch.no_grad():
        # A random correction bias of the router's own scale starves some of
        # six experts on a few dozen rows, and the collector rightly refuses
        # an expert no row reached. Keep it, but small beside the scores.
        built.mlp.gate.e_score_correction_bias.mul_(0.05)
    built = built.to(torch.float32).eval()
    for parameter in built.parameters():
        parameter.requires_grad_(False)
    return built


def _routed_names():
    return {f"{PREFIX}mlp.experts.{e}.{p}" for e in range(EXPERTS) for p in PROJECTIONS}


def _shared_names():
    return {f"{PREFIX}mlp.shared_experts.{p}" for p in PROJECTIONS}


def test_wrapper_declares_the_layer_like_a_body_moe_layer(layer):
    wrapper = glm_mtp.MtpCheckpointModel(layer)
    modules = dict(wrapper.named_modules())
    assert modules[f"{PREFIX}mlp.experts"] is layer.mlp.experts
    for name in _shared_names():
        assert isinstance(modules[name], torch.nn.Linear)

    projections = profile_declared_packed_expert_projections(wrapper, PROFILE)
    assert {member.qname for member in projections} == _routed_names()

    units = glm_mtp.mtp_priced_units(wrapper, PROFILE)
    hidden, inter = layer.config.hidden_size, layer.config.moe_intermediate_size
    assert set(units) == _routed_names() | _shared_names()
    for name, shape in units.items():
        want = (hidden, inter) if name.endswith("down_proj") else (inter, hidden)
        assert shape == want, name


def _per_expert_checkpoint(layer):
    """The layer as the checkpoint stores it: one tensor per expert and
    projection. Gate rows come first in the packed tensor because
    Transformers' `_apply_gate` chunks `gate, up`; that is written out here by
    hand rather than read from the profile under test."""
    state = {PREFIX + name: tensor.detach().clone()
             for name, tensor in layer.state_dict().items()
             if not name.startswith("mlp.experts.")}
    inter = layer.config.moe_intermediate_size
    gate_up = layer.mlp.experts.gate_up_proj.detach()
    down = layer.mlp.experts.down_proj.detach()
    for expert in range(gate_up.shape[0]):
        state[f"{PREFIX}mlp.experts.{expert}.gate_proj.weight"] = gate_up[expert, :inter].clone()
        state[f"{PREFIX}mlp.experts.{expert}.up_proj.weight"] = gate_up[expert, inter:].clone()
        state[f"{PREFIX}mlp.experts.{expert}.down_proj.weight"] = down[expert].clone()
    return state


def _write_checkpoint(directory, state):
    """Two shards and an index, plus a backbone tensor the loader must skip."""
    state = dict(state)
    state["model.language_model.layers.1.mlp.gate.weight"] = torch.zeros(EXPERTS, 64)
    names = sorted(state)
    shards = {"model-00001-of-00002.safetensors": names[::2],
              "model-00002-of-00002.safetensors": names[1::2]}
    weight_map = {}
    for shard, keys in shards.items():
        save_file({key: state[key].contiguous() for key in keys}, str(directory / shard))
        weight_map.update({key: shard for key in keys})
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}))


def test_loader_reads_the_layer_and_packs_its_experts(layer, tmp_path):
    _write_checkpoint(tmp_path, _per_expert_checkpoint(layer))

    loaded, receipt = glm_mtp.load_mtp_layer(
        tmp_path, _text_config(), profile=PROFILE, dtype=torch.float32,
        experts_implementation="eager")

    want = layer.state_dict()
    got = loaded.state_dict()
    assert set(got) == set(want)
    for name, tensor in want.items():
        assert torch.equal(got[name], tensor), name
    assert receipt["shards"] == ["model-00001-of-00002.safetensors",
                                 "model-00002-of-00002.safetensors"]
    assert receipt["tensors"] == len(_per_expert_checkpoint(layer))
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())["weight_map"]
    assert receipt["source_map"] == {key: shard for key, shard in index.items()
                                     if key.startswith(PREFIX)}

    split = dict(_split_packed_expert_tensor(
        loaded.mlp.experts.gate_up_proj, "gate_up_proj", PROFILE))
    inter = layer.config.moe_intermediate_size
    assert torch.equal(split["gate_proj"], layer.mlp.experts.gate_up_proj[:, :inter])
    assert torch.equal(split["up_proj"], layer.mlp.experts.gate_up_proj[:, inter:])


def test_loader_keeps_the_router_bias_in_fp32(layer, tmp_path):
    _write_checkpoint(tmp_path, _per_expert_checkpoint(layer))
    loaded, _ = glm_mtp.load_mtp_layer(
        tmp_path, _text_config(), profile=PROFILE, dtype=torch.bfloat16,
        experts_implementation="eager")
    for name, tensor in loaded.state_dict().items():
        want = torch.float32 if name.endswith("e_score_correction_bias") else torch.bfloat16
        assert tensor.dtype == want, name
        assert tensor.device.type == "cpu", name


def test_loader_refuses_a_missing_expert(layer, tmp_path):
    state = _per_expert_checkpoint(layer)
    del state[f"{PREFIX}mlp.experts.3.up_proj.weight"]
    _write_checkpoint(tmp_path, state)
    with pytest.raises(ValueError, match="missing expert"):
        glm_mtp.load_mtp_layer(tmp_path, _text_config(), profile=PROFILE, dtype=torch.float32,
                               experts_implementation="eager")


def test_loader_refuses_a_tensor_the_layer_does_not_have(layer, tmp_path):
    state = _per_expert_checkpoint(layer)
    state[f"{PREFIX}mlp.gate.bias"] = torch.zeros(EXPERTS)
    _write_checkpoint(tmp_path, state)
    with pytest.raises(RuntimeError, match="unexpected"):
        glm_mtp.load_mtp_layer(tmp_path, _text_config(), profile=PROFILE, dtype=torch.float32,
                               experts_implementation="eager")


def _swiglu(config, gate, up):
    gate = gate.clamp(max=config.swiglu_limit)
    up = up.clamp(min=-config.swiglu_limit, max=config.swiglu_limit)
    return torch.nn.functional.silu(gate) * up


def _expected_capture(layer, moe_inputs):
    """Rows, counts, XᵀX and max|x| per unit, from the router and the weights."""
    config = layer.config
    inter = config.moe_intermediate_size
    x = torch.cat([value.reshape(-1, value.shape[-1]) for value in moe_inputs])
    _, _, top = layer.mlp.gate(x)
    gate_up = layer.mlp.experts.gate_up_proj
    rows = {}
    for expert in range(EXPERTS):
        routed = x[(top == expert).any(-1)]
        hidden = _swiglu(config, routed @ gate_up[expert, :inter].T, routed @ gate_up[expert, inter:].T)
        rows[f"{PREFIX}mlp.experts.{expert}.gate_proj"] = routed
        rows[f"{PREFIX}mlp.experts.{expert}.up_proj"] = routed
        rows[f"{PREFIX}mlp.experts.{expert}.down_proj"] = hidden
    shared = layer.mlp.shared_experts
    rows[f"{PREFIX}mlp.shared_experts.gate_proj"] = x
    rows[f"{PREFIX}mlp.shared_experts.up_proj"] = x
    rows[f"{PREFIX}mlp.shared_experts.down_proj"] = _swiglu(
        config, shared.gate_proj(x), shared.up_proj(x))
    return rows


def test_capture_collects_the_rows_the_router_routes(layer):
    """The body's collector, unchanged, on the wrapper: per-expert rows,
    counts, Hessians and maxima as the router and SwiGLU produce them."""
    generator = torch.Generator().manual_seed(4)
    ids = torch.randint(0, VOCAB, (4, 17), generator=generator)
    hidden = torch.randn(4, 17, layer.config.hidden_size, generator=generator)
    embed = torch.randn(VOCAB, layer.config.hidden_size, generator=generator)
    feed = glm_mtp_capture.MtpCaptureFeed(
        glm_mtp.MtpCheckpointModel(layer), lambda t: embed[t], ids,
        read_final_hidden=lambda index: hidden[index:index + 1])

    moe_inputs = []
    handle = layer.post_attention_layernorm.register_forward_hook(
        lambda _m, _a, out: moe_inputs.append(out.detach().clone()))
    try:
        targets = sorted(_routed_names() | _shared_names())
        rows, hessians, counts, maxima = _collect_activations(
            feed.model, targets, feed.batches(), 5, "cpu", want_hessian=True,
            profile=PROFILE, forward_batch=feed)
    finally:
        handle.remove()
    feed.require_complete()

    expected = _expected_capture(layer, moe_inputs)
    assert set(rows) == set(expected)
    for name, x in expected.items():
        assert counts[name] == x.shape[0] > 0, name
        torch.testing.assert_close(hessians[name], x.T @ x, rtol=1e-5, atol=1e-5)
        assert maxima[name] == pytest.approx(float(x.abs().max()), rel=1e-6), name
        if ".shared_experts." in name:
            torch.testing.assert_close(rows[name], x[:5], rtol=1e-6, atol=1e-6)
        else:
            # The collector orders an expert's rows by top-k slot, then token
            # (`torch.where` over the slot-major routing mask), so its scoring
            # prefix is some five of the expert's rows, not the first five.
            assert rows[name].shape == (min(5, x.shape[0]), x.shape[1]), name
            for row in rows[name]:
                assert float((x - row).abs().amax(dim=-1).min()) < 1e-6, name
    assert counts[f"{PREFIX}mlp.shared_experts.gate_proj"] == 4 * 16


def test_feed_refuses_a_row_out_of_order(layer):
    ids = torch.arange(18).view(2, 9) % VOCAB
    hidden = torch.zeros(2, 9, layer.config.hidden_size)
    feed = glm_mtp_capture.MtpCaptureFeed(
        glm_mtp.MtpCheckpointModel(layer), lambda t: torch.zeros(*t.shape, layer.config.hidden_size),
        ids, read_final_hidden=lambda index: hidden[index:index + 1])
    with pytest.raises(RuntimeError, match="calibration row 0"):
        feed(ids[1:2])


def test_feed_refuses_an_incomplete_pass(layer):
    ids = torch.arange(18).view(2, 9) % VOCAB
    hidden = torch.zeros(2, 9, layer.config.hidden_size)
    feed = glm_mtp_capture.MtpCaptureFeed(
        glm_mtp.MtpCheckpointModel(layer), lambda t: torch.zeros(*t.shape, layer.config.hidden_size),
        ids, read_final_hidden=lambda index: hidden[index:index + 1])
    with torch.no_grad():
        feed(ids[0:1])
    with pytest.raises(RuntimeError, match="1 of 2"):
        feed.require_complete()


@pytest.fixture(scope="module")
def tiny_model():
    return _build_tiny_model()


def test_final_hidden_from_the_last_boundary_reproduces_the_logits(tiny_model):
    """Layer boundary → last layer → mHC collapse → final norm is the hidden
    state the target's `lm_head` reads: the logits must match the model's."""
    runner = _streamed_runner(tiny_model)
    ids = torch.randint(0, tiny_model.config.text_config.vocab_size, (1, 12),
                        generator=torch.Generator().manual_seed(9))
    with torch.inference_mode():
        boundaries = runner.capture_boundaries(ids)
        last = runner.num_layers - 1
        final = glm_mtp_capture.final_hidden(
            runner, ids, boundaries.activations_cpu[last], layer=last)
        reference = tiny_model(input_ids=ids).logits
    assert final.shape == (1, 12, tiny_model.config.text_config.hidden_size)
    torch.testing.assert_close(tiny_model.lm_head(final), reference, rtol=1e-4, atol=1e-4)


def test_final_hidden_refuses_a_layer_that_is_not_the_last(tiny_model):
    runner = _streamed_runner(tiny_model)
    ids = torch.zeros(1, 4, dtype=torch.long)
    with pytest.raises(ValueError, match="last backbone layer"):
        glm_mtp_capture.final_hidden(runner, ids, torch.zeros(1, 4, 4, 64), layer=0)


def test_loader_binds_the_named_experts_dispatch(layer, tmp_path):
    """A default dispatch is whatever Transformers falls back to; the capture
    must run the dispatch the body recorded, so the loader takes it by name."""
    _write_checkpoint(tmp_path, _per_expert_checkpoint(layer))
    loaded, receipt = glm_mtp.load_mtp_layer(
        tmp_path, _text_config(), profile=PROFILE, dtype=torch.float32,
        experts_implementation="batched_mm")
    assert loaded.mlp.experts.config._experts_implementation == "batched_mm"
    assert receipt["experts_implementation"] == "batched_mm"
    with pytest.raises(TypeError):
        glm_mtp.load_mtp_layer(tmp_path, _text_config(), profile=PROFILE, dtype=torch.float32)
    with pytest.raises(ValueError, match="must be named"):
        glm_mtp.load_mtp_layer(tmp_path, _text_config(), profile=PROFILE, dtype=torch.float32,
                               experts_implementation="")


def _input_ref():
    return {"schema": glm_mtp_capture.FINAL_HIDDEN_SCHEMA, "path": "/final/manifest.json",
            "sha256": "c" * 64}


def test_load_contract_binds_every_tensor_and_the_input(layer, tmp_path):
    """The MTP capture's initialization contract is its own load: a changed
    tensor, a tampered digest or an unbound input is a different contract or
    none, and the shared validator routes it without admitting it elsewhere."""
    from prismaquant import validate_source_initialization_contract
    from prismaquant.streaming_model import validate_mtp_layer_initialization_contract

    _write_checkpoint(tmp_path, _per_expert_checkpoint(layer))
    loaded, receipt = glm_mtp.load_mtp_layer(
        tmp_path, _text_config(), profile=PROFILE, dtype=torch.float32,
        experts_implementation="eager")
    contract = glm_mtp.mtp_layer_initialization_contract(loaded, receipt, input_manifest=_input_ref())
    assert contract["schema"] == "prismaquant.mtp_layer_initialization.v1"
    assert contract["layer_prefix"] == PREFIX
    assert contract["checkpoint_tensors"] == receipt["tensors"]
    assert contract["experts_implementation"] == "eager"
    assert set(contract["state"]) == set(loaded.state_dict())
    assert validate_source_initialization_contract(contract) == contract
    assert glm_mtp.mtp_layer_initialization_contract(
        loaded, receipt, input_manifest=_input_ref()) == contract

    with torch.no_grad():
        loaded.mlp.experts.down_proj[1, 2, 3] += 1.0
    changed = glm_mtp.mtp_layer_initialization_contract(loaded, receipt, input_manifest=_input_ref())
    assert changed["state_sha256"] != contract["state_sha256"]
    assert changed["state"]["mlp.experts.down_proj"] != contract["state"]["mlp.experts.down_proj"]

    tampered = json.loads(json.dumps(contract))
    tampered["state"]["eh_proj.weight"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="state digest"):
        validate_mtp_layer_initialization_contract(tampered)
    unbound = json.loads(json.dumps(contract))
    unbound["input"]["sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="hash-bound input"):
        validate_source_initialization_contract(unbound)
    extra = dict(contract, note="x")
    with pytest.raises(ValueError, match="MTP layer initialization"):
        validate_source_initialization_contract(extra)


def test_boundary_records_are_checked_before_any_read():
    """Sequence i must get sequence i's boundary: a record whose own
    coordinates name another sequence refuses before its file is opened."""
    session = {"generation": "g", "run_identity_sha256": "r" * 64}

    def record(index, batch):
        return {"name": f"boundary-{index}-44-at-44", "path": "/does/not/exist.pt",
                "metadata": {"identity": {"session": session, "kind": "boundary",
                                          "coordinates": {"batch": batch, "boundary": 44}}}}

    good = {"session": session, "records": [record(0, 0), record(1, 1)]}
    assert [r["name"] for r in glm_mtp_capture.ordered_boundary_records(good, 2, layer=44)] == [
        "boundary-0-44-at-44", "boundary-1-44-at-44"]
    swapped = {"session": session, "records": [record(0, 0), record(1, 0)]}
    with pytest.raises(RuntimeError, match="not sequence 1"):
        glm_mtp_capture.ordered_boundary_records(swapped, 2, layer=44)
    short = {"session": session, "records": [record(0, 0)]}
    with pytest.raises(RuntimeError, match="sequence 1"):
        glm_mtp_capture.ordered_boundary_records(short, 2, layer=44)


# --------------------------------------------------------------------------
# Both phases, end to end, on a tiny checkpoint that carries its MTP layer.
# --------------------------------------------------------------------------

N_SEQUENCES, SEQ_LEN, MAX_ROWS = 4, 13, 4
LAST = BACKBONE - 1
DENSE_UNIT = "model.language_model.layers.0.mlp.down_proj"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime():
    return dict(torch=torch.__version__, cuda=torch.version.cuda,
                transformers=importlib.metadata.version("transformers"))


@pytest.fixture
def mtp_source(tmp_path, monkeypatch):
    """A two-layer GLM checkpoint plus its MTP layer (index 2), a body census
    and a complete canonical capture over one body unit."""
    import prismaquant.model_profiles.glm5_next as glm5_profile
    from safetensors import safe_open
    from transformers import AutoConfig

    from prismaquant import tessera_calibration_cache as cc
    from prismaquant.tessera_campaign import calibration_census

    monkeypatch.setenv("PRISMAQUANT_TMPDIR", str(tmp_path / "staging"))
    # The profile names GLM-5.3-Flash's MTP layer by its index, 45, and the
    # streamed loader drops it by that name. Here the MTP layer is 2.
    monkeypatch.setattr(glm5_profile, "_MTP_LAYER_RE",
                        re.compile(r"^model\.language_model\.layers\.2\."))
    source = tmp_path / "source"
    write_original_layout_checkpoint(_build_tiny_model(), source)
    config = json.loads((source / "config.json").read_text())
    # One MTP layer, and an indexer that keeps every key of a 12-row draft.
    config["text_config"].update(num_nextn_predict_layers=1, index_topk=16,
                                 index_kpool_always_select_tail=True)
    (source / "config.json").write_text(json.dumps(config))
    text_config = AutoConfig.from_pretrained(source).text_config
    text_config._attn_implementation = "eager"
    with genuine_weight_initialization():
        mtp = glm_mtp.Glm5NextMtpLayer(text_config)
    _randomize(mtp, 20260927)
    with torch.no_grad():
        mtp.mlp.gate.e_score_correction_bias.mul_(0.05)
    mtp = mtp.to(torch.float32).eval()
    mtp_state = _per_expert_checkpoint(mtp)
    save_file({key: value.contiguous() for key, value in mtp_state.items()},
              str(source / "model-mtp.safetensors"))
    with safe_open(str(source / "model.safetensors"), framework="pt") as handle:
        weight_map = dict.fromkeys(handle.keys(), "model.safetensors")
    weight_map.update(dict.fromkeys(mtp_state, "model-mtp.safetensors"))
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}))

    version = importlib.metadata.version("transformers")
    contract = dict(schema="prismaquant.streaming_initialization.v1",
        scope="streamed_text_source_forward", status="completed",
        transformers_version=version, model_class="SyntheticSource",
        dtype="torch.float32", layers_prefix="model.language_model.layers.",
        num_layers=BACKBONE, persistent_tensors=2, derived_buffers=0,
        state_sha256="a" * 64, source_map_sha256="b" * 64)
    calibration = {"fit_ids_sha256": "d" * 64, "fit_tokens": N_SEQUENCES * SEQ_LEN,
                   "nsamples": N_SEQUENCES, "seqlen": SEQ_LEN, "seed": 0,
                   "source": "synthetic", "split_role": "calibration",
                   "text_sha256": "e" * 64}
    producer = {"schema": "tessera.expert_projection.v1",
                "source": {"config_sha256": _sha(source / "config.json")}}
    census = calibration_census(
        {DENSE_UNIT: 4}, {DENSE_UNIT: 3.0},
        args=SimpleNamespace(model=str(source), nsamples=N_SEQUENCES, seqlen=SEQ_LEN,
                             seed=0, layer_stride=1),
        groups={"u:" + DENSE_UNIT: [DENSE_UNIT]}, dense_targets=[DENSE_UNIT],
        expert_targets=[], shapes={DENSE_UNIT: [64, 128]},
        identity={"text_sha256": "e" * 64, "fit_ids_sha256": "d" * 64},
        expert_projection={"producer": producer}, model_load_contract=contract,
        attention_implementation="eager", capture_runtime=_runtime())
    census_path = tmp_path / "census.json"
    census_path.write_text(json.dumps(census))
    canonical = cc.capture_identity(census_path, calibration=calibration, max_act_rows=MAX_ROWS,
        model_load_contract=contract, attention_implementation="eager")
    rows = torch.randn(4, 128)
    capture = cc.publish_capture(tmp_path / "canonical", census_path=census_path,
        identity=canonical, acts={DENSE_UNIT: rows}, hessians={DENSE_UNIT: rows.T @ rows},
        counts=census["counts"], maxima=census["max_abs"])
    ids = torch.randint(2, 128, (N_SEQUENCES, SEQ_LEN), generator=torch.Generator().manual_seed(31))
    return SimpleNamespace(source=source, text_config=text_config, mtp=mtp, census=census,
                           census_path=census_path, canonical=canonical, capture=capture,
                           ids=ids, root=tmp_path)


def _runner(env, *, source_authentication=None, name="offload"):
    from prismaquant.cost_streaming import build_streamed_causal_lm

    return build_streamed_causal_lm(
        str(env.source), device=torch.device("cpu"), dtype=torch.float32,
        offload_folder=str(env.root / name), profile=PROFILE, max_cache_slots=2,
        prefetch_workers=1, prefetch_min_available_gb=0, cache_headroom_gb=0,
        prefetch_lookahead=1, require_prefetched_residency=True,
        attn_implementation="eager",
        **({"source_authentication": source_authentication}
           if source_authentication is not None else {}))


def _write_boundaries(env, activations):
    """Stage A's layer-input entries, as the shared boundary manifest lists them."""
    from prismaquant.joint_adjoint_checkpoints import exact_entry_record
    from prismaquant.perturbed_x_cache import write_exact_activation_cache_entry

    session = {"generation": "stage-a", "run_identity_sha256": "f" * 64}
    records = []
    for index in range(N_SEQUENCES):
        value = activations[index:index + 1].contiguous()
        nbytes = value.numel() * value.element_size()
        reference = write_exact_activation_cache_entry(
            env.root / "boundaries", f"boundary-{index}-{LAST}-at-{LAST}", value,
            identity={"coordinates": {"batch": index, "boundary": LAST, "probe": None},
                      "kind": "boundary", "session": session, "slot": f"boundary-{index}-{LAST}"},
            max_tensor_bytes=nbytes, max_file_bytes=nbytes + (1 << 16))
        records.append(exact_entry_record(reference))
    return {"session": session, "records": records}


def _phase_one(env, owner):
    runner = _runner(env, source_authentication=owner, name="offload-phase1")
    try:
        runner.context.begin_source_initialization_audit()
        runner.context.schedule_prefetch(LAST)
        runner.context.install(LAST, require_prefetched=True, prefetch_following=False)
        boundary_manifest = env.boundary_manifest
        session = glm_mtp_capture.final_hidden_session({"test": "phase-1"})
        entry = SEQ_LEN * 4 * 64 * 4
        records, head_check = glm_mtp_capture.write_final_hidden(
            runner, env.ids,
            glm_mtp_capture.ordered_boundary_records(boundary_manifest, N_SEQUENCES, layer=LAST),
            boundary_session=boundary_manifest["session"], layer=LAST,
            out_dir=env.root / "final-hidden", session=session,
            read_ahead_bytes=2 * entry, head=runner._head())
        witness = runner.context.source_selected_initialization_witness([LAST])
        return glm_mtp_capture.publish_final_hidden(
            env.root / "final-hidden", session=session, records=records, layer=LAST,
            inputs={"test": True}, source_witness=witness,
            source_authentication=owner.receipt(), head_check=head_check), witness
    finally:
        runner.shutdown()


def test_both_phases_publish_a_capture_a_selected_consumer_accepts(mtp_source):
    from prismaquant import tessera_calibration_cache as cc
    from prismaquant.streaming_model import validate_streaming_selected_initialization_witness

    env = mtp_source
    # Stage A's role: the last layer's input, from a complete streamed pass
    # whose own witness names the checkpoint's source map.
    reference = _runner(env, name="offload-reference")
    try:
        reference.context.begin_source_initialization_audit()
        with torch.inference_mode():
            boundaries = reference.capture_boundaries(env.ids)
            logits = reference(env.ids).logits
        full_contract = reference.context.source_initialization_contract()
    finally:
        reference.shutdown()
    env.boundary_manifest = _write_boundaries(env, boundaries.activations_cpu[LAST])

    owner = cc.authenticate_selected_capture_source(
        env.census_path, env.capture["path"], expected_sha256=env.capture["sha256"],
        model=str(env.source), max_act_rows=MAX_ROWS, attention_implementation="eager")
    try:
        final_ref, witness = _phase_one(env, owner)
        assert validate_streaming_selected_initialization_witness(witness) == witness
        assert witness["observed_layers"] == [LAST]
        assert witness["source_map_sha256"] == full_contract["source_map_sha256"]
        final_manifest, _ = glm_mtp_capture.read_bound_json(final_ref["path"], final_ref["sha256"])
        assert len(final_manifest["head_check"]) == N_SEQUENCES
        assert all(0.0 <= row["top1"] <= 1.0 for row in final_manifest["head_check"])

        # Phase 2, through the same source owner.
        layer, receipt = glm_mtp.load_mtp_layer(
            env.source, env.text_config, profile=PROFILE, dtype=torch.float32,
            experts_implementation="eager", source_authentication=owner)
        embed_weight, embed_shard = glm_mtp.read_checkpoint_tensor(
            env.source, "model.language_model.embed_tokens.weight", source_authentication=owner)
        assert embed_shard == "model.safetensors"
        embed = torch.nn.Embedding.from_pretrained(embed_weight.to(torch.float32))
        contract = glm_mtp.mtp_layer_initialization_contract(layer, receipt, input_manifest=final_ref)
        wrapper = glm_mtp.MtpCheckpointModel(layer)
        units = glm_mtp.mtp_priced_units(wrapper, PROFILE)
        read, stream = glm_mtp_capture.final_hidden_stream(
            final_manifest, N_SEQUENCES, read_ahead_bytes=2 * SEQ_LEN * 64 * 4)
        moe_inputs, finals = [], []

        def recording_read(index):
            value = read(index)
            finals.append(value)
            return value

        handle = layer.post_attention_layernorm.register_forward_hook(
            lambda _m, _a, out: moe_inputs.append(out.detach().clone()))
        try:
            with closing(stream):
                rows, hessians, counts, maxima = glm_mtp_capture.capture_mtp_layer(
                    wrapper, embed, env.ids, recording_read, units=units, profile=PROFILE,
                    device="cpu", max_act_rows=MAX_ROWS)
        finally:
            handle.remove()
        groups = glm_mtp_capture.mtp_anchor_groups(wrapper, units, PROFILE)
        census = glm_mtp_capture.mtp_census(
            base_census=env.census,
            base_census_ref={"path": str(env.census_path), "sha256": _sha(env.census_path)},
            canonical_capture_ref=env.capture, final_hidden_ref=final_ref, layer=BACKBONE,
            units=units, counts=counts, max_abs=maxima, groups=groups,
            model_load_contract=contract,
            attention_implementation=layer.config._attn_implementation,
            capture_runtime=_runtime())
        census_path = env.root / "mtp-census.json"
        identity, census_sha256, sealed = glm_mtp_capture.publish_mtp_capture(
            env.root / "mtp-capture", census=census, census_path=census_path,
            source_authentication=owner, calibration=env.canonical["calibration"],
            max_act_rows=MAX_ROWS, rows=rows, hessians=hessians, counts=counts,
            max_abs=maxima,
            completed_contract=glm_mtp.mtp_layer_initialization_contract(
                layer, receipt, input_manifest=final_ref))
        authentication = owner.receipt()
    finally:
        owner.close()

    # The hidden states are the ones the target's head reads.
    head = torch.nn.Linear(64, 128, bias=False)
    from safetensors import safe_open
    with safe_open(str(env.source / "model.safetensors"), framework="pt") as handle:
        head.weight.data.copy_(handle.get_tensor("lm_head.weight"))
    for index, value in enumerate(finals):
        torch.testing.assert_close(head(value)[0], logits[index], rtol=1e-4, atol=1e-4)

    # The capture's identity: the canonical roster, the MTP census and units,
    # the MTP layer's own load contract, the body's draw.
    assert identity["source_files"] == env.canonical["source_files"]
    assert identity["census_sha256"] == census_sha256 == _sha(census_path)
    assert identity["units"] == {name: list(shape) for name, shape in units.items()}
    assert identity["calibration"] == env.canonical["calibration"]
    assert identity["model_load_contract"] == contract
    assert cc.require_capture_contract(sealed["path"], sealed["sha256"])["identity"] == identity
    assert census["mtp_extension"]["canonical_capture"] == env.capture
    assert census["mtp_extension"]["final_hidden"] == final_ref
    prefix = f"model.language_model.layers.{BACKBONE}."
    assert set(census["anchor_groups"]) == {
        f"s:{prefix}mlp.experts", f"g:{prefix}mlp.shared_experts.gate_up_proj",
        f"u:{prefix}mlp.shared_experts.down_proj"}
    assert census["expert_targets"] == sorted(n for n in units if ".mlp.experts." in n)

    # Every shard both phases read was authenticated, and the derived census
    # is the one the owner admitted.
    verified = {row["name"] for row in authentication["verified_files"]}
    assert {"model.safetensors", "model-mtp.safetensors"} <= verified
    assert authentication["derived_census_sha256"] == [census_sha256]

    # The shared expert saw every row: n·(T-1), and its XᵀX is the sum over them.
    x = torch.cat([value.reshape(-1, value.shape[-1]) for value in moe_inputs])
    gate = f"{prefix}mlp.shared_experts.gate_proj"
    assert counts[gate] == N_SEQUENCES * (SEQ_LEN - 1) == x.shape[0]
    torch.testing.assert_close(hessians[gate], x.T @ x, rtol=1e-5, atol=1e-5)

    # A later consumer binds the published capture like any selected source.
    consumer = cc.authenticate_selected_capture_source(
        census_path, sealed["path"], expected_sha256=sealed["sha256"], model=str(env.source),
        max_act_rows=MAX_ROWS, attention_implementation="eager")
    consumer.close()


def test_derived_census_must_name_the_canonical_source(mtp_source, tmp_path):
    """Only a census over the same model and producer roster may inherit the
    canonical capture's source roster, and only once it is admitted."""
    from prismaquant import tessera_calibration_cache as cc

    env = mtp_source
    owner = cc.authenticate_selected_capture_source(
        env.census_path, env.capture["path"], expected_sha256=env.capture["sha256"],
        model=str(env.source), max_act_rows=MAX_ROWS, attention_implementation="eager")
    try:
        derived = json.loads(json.dumps(env.census))
        derived["counts"] = dict(derived["counts"])
        derived_path = tmp_path / "derived.json"
        derived_path.write_text(json.dumps(derived, sort_keys=True))
        with pytest.raises(RuntimeError, match="census or complete source roster"):
            cc.capture_identity(derived_path, calibration=env.canonical["calibration"],
                max_act_rows=MAX_ROWS, model_load_contract=derived["model_load_contract"],
                attention_implementation="eager", source_authentication=owner)
        foreign = json.loads(json.dumps(env.census))
        foreign["expert_projection"]["producer"]["source"]["config_sha256"] = "0" * 64
        foreign_path = tmp_path / "foreign.json"
        foreign_path.write_text(json.dumps(foreign))
        with pytest.raises(RuntimeError, match="producer roster"):
            owner.admit_derived_census(foreign_path)
        elsewhere = dict(env.census, model=str(tmp_path / "other"))
        elsewhere_path = tmp_path / "elsewhere.json"
        elsewhere_path.write_text(json.dumps(elsewhere))
        with pytest.raises(RuntimeError, match="another source model"):
            owner.admit_derived_census(elsewhere_path)
        assert owner.admit_derived_census(derived_path) == _sha(derived_path)
        identity = cc.capture_identity(derived_path, calibration=env.canonical["calibration"],
            max_act_rows=MAX_ROWS, model_load_contract=derived["model_load_contract"],
            attention_implementation="eager", source_authentication=owner)
        assert identity["census_sha256"] == _sha(derived_path)
        assert identity["source_files"] == env.canonical["source_files"]
    finally:
        owner.close()


def test_cli_runs_both_phases_from_the_body_plan(mtp_source, monkeypatch):
    """``tools/glm_mtp_capture`` reads the body's plan, its boundary entries
    and its prepared dispatch, and publishes a capture a consumer accepts."""
    from prismaquant import joint_cost_quantum
    from prismaquant import tessera_calibration_cache as cc
    from tools import glm_mtp_capture as cli

    env = mtp_source
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

    built = []

    def runner(config, *, offload_folder, source_authentication):
        assert config == plan and source_authentication is not None
        built.append(offload_folder)
        return _runner(env, source_authentication=source_authentication, name="offload-cli")

    monkeypatch.setattr(joint_cost_quantum, "build_quantum_source_runner", runner)
    phase1 = env.root / "cli-final-hidden"
    assert cli.main(["--phase", "final-hidden", "--plan", str(plan_path),
                     "--plan-sha256", _sha(plan_path), "--boundaries", str(boundary_path),
                     "--boundaries-sha256", _sha(boundary_path), "--out", str(phase1),
                     "--offload-folder", str(env.root / "unused"), "--read-ahead-mb", "1"]) == 0
    assert built == [str(env.root / "unused")]
    report = json.loads((phase1 / "final-hidden-run.json").read_text())
    final_path = phase1 / "manifest.json"
    assert report["manifest"]["sha256"] == _sha(final_path)
    assert report["sequences"] == N_SEQUENCES
    final = json.loads(final_path.read_text())
    assert final["layer"] == LAST
    assert final["inputs"]["plan"]["sha256"] == _sha(plan_path)
    assert final["inputs"]["boundaries"]["sha256"] == _sha(boundary_path)

    phase2 = env.root / "cli-capture"
    census_path = env.root / "cli-mtp-census.json"
    capture_args = ["--phase", "capture", "--plan", str(plan_path),
                    "--plan-sha256", _sha(plan_path), "--prepared", str(prepared_path),
                    "--prepared-sha256", _sha(prepared_path), "--final-hidden", str(final_path),
                    "--final-hidden-sha256", _sha(final_path), "--out", str(phase2),
                    "--census-out", str(census_path), "--device", "cpu",
                    "--read-ahead-mb", "1"]
    assert cli.main(capture_args) == 0
    report = json.loads((phase2 / "capture-run.json").read_text())
    assert report["experts_implementation"] == "eager"
    assert report["census"]["sha256"] == _sha(census_path)
    census = json.loads(census_path.read_text())
    assert census["model_load_contract"]["schema"] == "prismaquant.mtp_layer_initialization.v1"
    assert census["model_load_contract"]["experts_implementation"] == "eager"
    assert census["mtp_extension"]["final_hidden"]["sha256"] == _sha(final_path)
    consumer = cc.authenticate_selected_capture_source(
        census_path, report["capture"]["path"], expected_sha256=report["capture"]["sha256"],
        model=str(env.source), max_act_rows=MAX_ROWS, attention_implementation="eager")
    consumer.close()

    # Another draw's final hidden states are refused before the layer loads.
    manifest["calibration_sha256"] = "0" * 64
    boundary_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="another calibration draw"):
        cli.main(["--phase", "final-hidden", "--plan", str(plan_path),
                  "--plan-sha256", _sha(plan_path), "--boundaries", str(boundary_path),
                  "--boundaries-sha256", _sha(boundary_path), "--out", str(env.root / "again"),
                  "--offload-folder", str(env.root / "unused")])
