"""Materializing the visual tower is a run-modality decision, not a skeleton one.

Measured failure this pins (PrismaBuild action
`dae1474a406ec02ef7a953f8a5d3fb0e1f4a30f99292b00aabc66f424b2c8471`, stage A of
the GLM-5.3-Flash joint-AURA campaign, 2026-09-20, rc 1):

    [cost-streaming] profile glm5_next has no text-only skeleton route;
      using the multimodal construction
    [cost-streaming] base_prefix='model.language_model'  layers=45
      multimodal=True  visual_prefix=model.visual
    [cost-streaming] head materialized (3 tensors, rotary re-init) in 2.8s
    File "prismaquant/streaming_model.py", line 1667, in _build_streaming_context
      tensors = _read_layer_to_device(
    prismaquant.staged_tier_policy.TierPolicyRefused: staged-tier-forbidden:
      readset-not-staged: .../GLM-5.3-Flash-BF16/model-00120-of-00120.safetensors

The staged-tier refusal was correct: shard 120 carries `model.visual.*` and the
run's readset is text-only.  The defect is that the read was issued at all.

Stage A feeds `StreamedCausalLM` calibrated token IDs.  `_prepare`
(`cost_streaming.py:1250`) turns them into hidden states through
`base_model.embed_tokens` and nothing else, so nothing on that path can execute
the visual tower.  It was loaded anyway because one `multimodal` flag inside
`_build_streaming_context` carried two independent questions:

  * can this family be CONSTRUCTED text-only?  A property of the pinned
    transformers, declared by `ModelProfile.requires_multimodal_skeleton()` and
    consumed at `streaming_model.py:1530`.  `glm5_next` answers "no": there is
    no `Glm5NextForCausalLM` auto-route, so the skeleton must come from the
    declared `...ForConditionalGeneration` class.
  * will THIS RUN drive visual inputs?  The caller's own declaration, consumed
    at `streaming_model.py:1661` to materialize the tower onto the device.

The first was allowed to imply the second, so a construction-compatibility fact
put the `model.visual.*` namespace into a text-only run's readset.  What the
campaign actually measured is one `_read_layer_to_device("model.visual.", ...)`
call, whose first tensor range fell in a shard the run had not staged; the
shard's name carries the checkpoint's total shard count (120) and says nothing
about how much vision was read.  This is the same contract the streamed
*exporter* already states correctly: its own flip on
`requires_multimodal_skeleton()` builds the multimodal skeleton and leaves "the
visual tower ... on meta" (`docs/ARCHITECTURE.md`, "Multimodal-forced export
skeleton").

What must NOT change, and is pinned here too: a caller that explicitly declares
`multimodal=True` still gets a fully resident tower, and
`visual_requires_grad=True` still flips grad on its Linears.  That is the visual
Fisher probe's contract (`sensitivity_probe.run_multimodal_visual_probe_pass`)
and the visual cost shard's (`incremental_measure_quant_cost`).

Everything here is synthetic and CPU-only: a tiny real Qwen2-VL checkpoint (one
text layer, hidden 32, one vision block) written to `tmp_path`.  Qwen2-VL is
used because transformers builds it in exactly the v5 umbrella layout the
campaign hit -- `model.model.visual` beside `model.model.language_model`, base
prefix `model.language_model`, root `lm_head` -- so the fixture reproduces the
failing shape without a glm5_next checkpoint, container or vendored modelling.
Nothing below is Qwen2-VL-specific.
"""
from __future__ import annotations

import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
save_file = pytest.importorskip("safetensors.torch").save_file


HIDDEN = 32
VOCAB = 64
TEXT_LAYERS = 1
VISION_EMBED = 16
VISUAL_PREFIX = "model.visual"
LAYERS_PREFIX = "model.language_model.layers."


# --------------------------------------------------------------------------
# Fixture: a tiny real multimodal-umbrella checkpoint
# --------------------------------------------------------------------------
def _vl_config():
    config_cls = getattr(transformers, "Qwen2VLConfig", None)
    if config_cls is None or not hasattr(
            transformers, "Qwen2VLForConditionalGeneration"):
        pytest.skip("pinned transformers ships no Qwen2-VL umbrella arch")
    config = config_cls(
        text_config={
            "hidden_size": HIDDEN,
            "num_hidden_layers": TEXT_LAYERS,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 64,
            "vocab_size": VOCAB,
            "max_position_embeddings": 128,
            "tie_word_embeddings": False,
        },
        vision_config={
            "depth": 1,
            "embed_dim": VISION_EMBED,
            "hidden_size": HIDDEN,
            "num_heads": 2,
            "mlp_ratio": 2,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "in_channels": 3,
        },
        tie_word_embeddings=False,
    )
    # `_resolve_declared_model_cls` reads architectures[0]; without it the
    # multimodal construction would silently fall back to the auto class.
    config.architectures = ["Qwen2VLForConditionalGeneration"]
    return config


def _write_vl_ckpt(tmp_path, name: str) -> str:
    """Write config + one real safetensors shard holding every tensor the
    declared architecture asks for.

    The tensor list is enumerated from the meta skeleton the loader itself
    builds, so the fixture cannot drift from transformers' own module tree.
    """
    from transformers import AutoConfig

    from prismaquant.streaming_model import build_streaming_skeleton

    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    _vl_config().save_pretrained(str(directory))

    skeleton = build_streaming_skeleton(
        AutoConfig.from_pretrained(str(directory), trust_remote_code=True),
        multimodal=True, log_prefix="[test]")
    tensors = {
        key: torch.zeros(tuple(param.shape), dtype=torch.bfloat16)
        for key, param in skeleton.named_parameters()
    }
    assert any(key.startswith(VISUAL_PREFIX + ".") for key in tensors), (
        "fixture has no visual tensors — the umbrella layout changed")
    assert any(key.startswith(LAYERS_PREFIX) for key in tensors), (
        "fixture has no decoder tensors under the umbrella layer prefix")

    shard = "model-00001-of-00001.safetensors"
    save_file(tensors, str(directory / shard))
    (directory / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(
            t.numel() * t.element_size() for t in tensors.values())},
        "weight_map": {key: shard for key in tensors},
    }, indent=2))
    return str(directory)


# --------------------------------------------------------------------------
# Driver: build the real context, recording which prefixes were read
# --------------------------------------------------------------------------
def _build(tmp_path, monkeypatch, path, *, force_multimodal_skeleton, **kwargs):
    from prismaquant import streaming_model as sm
    from prismaquant.model_profiles import detect_profile

    monkeypatch.setenv("PRISMAQUANT_TMPDIR", str(tmp_path / "stage"))
    (tmp_path / "stage").mkdir(parents=True, exist_ok=True)

    if force_multimodal_skeleton:
        # Exactly what `Glm5NextProfile` declares. Patching the detected
        # profile's own class keeps every consumer (`detect_profile` in
        # `_build_streaming_context` and `_build_weight_map`,
        # `profile_from_model` in `_head_prefixes`) on one answer.
        monkeypatch.setattr(type(detect_profile(path)),
                            "requires_multimodal_skeleton",
                            lambda self: True, raising=True)

    reads: list[str] = []
    real_read = sm._read_layer_to_device

    def recording_read(prefix, *args, **kw):
        reads.append(prefix)
        return real_read(prefix, *args, **kw)

    monkeypatch.setattr(sm, "_read_layer_to_device", recording_read)

    context = sm._build_streaming_context(
        path,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        offload_folder=str(tmp_path / "offload"),
        log_prefix="[test]",
        **kwargs,
    )
    return context, reads


def _visual_reads(reads: list[str]) -> list[str]:
    return [p for p in reads if p.startswith(VISUAL_PREFIX + ".")]


# --------------------------------------------------------------------------
# The reported failure
# --------------------------------------------------------------------------
def test_forced_multimodal_skeleton_reads_no_visual_tensors(
        tmp_path, monkeypatch):
    """A profile that can only be CONSTRUCTED multimodally must not, by that
    fact alone, pull the vision namespace into the run's readset."""
    path = _write_vl_ckpt(tmp_path, "vl_forced")
    context, reads = _build(tmp_path, monkeypatch, path,
                            force_multimodal_skeleton=True)
    try:
        assert _visual_reads(reads) == [], (
            "the text-only caller read visual tensors: "
            f"{_visual_reads(reads)} — this is the campaign's "
            "readset-not-staged refusal, one layer down")
        assert context.visual_module is None
    finally:
        context.shutdown()


def test_forced_multimodal_skeleton_still_builds_the_umbrella_skeleton(
        tmp_path, monkeypatch):
    """The half that must NOT change: construction still takes the declared
    multimodal class, so the body/head names resolve as the campaign's do."""
    path = _write_vl_ckpt(tmp_path, "vl_shape")
    context, _reads = _build(tmp_path, monkeypatch, path,
                             force_multimodal_skeleton=True)
    try:
        assert type(context.model).__name__ == "Qwen2VLForConditionalGeneration"
        assert context.layers_prefix == LAYERS_PREFIX
        assert context.num_layers == TEXT_LAYERS
        # The tower exists on the skeleton and is simply left where the
        # exporter already leaves it.
        tower = context.model.model.visual
        assert all(p.is_meta for p in tower.parameters())
        # Head pieces are resident, as before.
        head = context.model.get_output_embeddings()
        assert head is not None and not head.weight.is_meta
        assert not context.base_model.embed_tokens.weight.is_meta
    finally:
        context.shutdown()


# --------------------------------------------------------------------------
# Guards: the genuine visual paths are untouched
# --------------------------------------------------------------------------
def test_explicit_multimodal_caller_still_materializes_the_tower(
        tmp_path, monkeypatch):
    """`incremental_measure_quant_cost._run_visual_cost_shard` passes
    `multimodal=True` and then requires `ctx.visual_module`."""
    path = _write_vl_ckpt(tmp_path, "vl_explicit")
    context, reads = _build(tmp_path, monkeypatch, path,
                            force_multimodal_skeleton=False, multimodal=True)
    try:
        assert _visual_reads(reads) == [VISUAL_PREFIX + "."]
        assert context.visual_module is not None
        assert context.visual_prefix == VISUAL_PREFIX
        assert not any(p.is_meta for p in context.visual_module.parameters())
    finally:
        context.shutdown()


def test_explicit_multimodal_still_materializes_under_a_forced_profile(
        tmp_path, monkeypatch):
    """The forced-skeleton flip must not suppress an explicit request either."""
    path = _write_vl_ckpt(tmp_path, "vl_explicit_forced")
    context, reads = _build(tmp_path, monkeypatch, path,
                            force_multimodal_skeleton=True, multimodal=True)
    try:
        assert _visual_reads(reads) == [VISUAL_PREFIX + "."]
        assert context.visual_module is not None
        assert not any(p.is_meta for p in context.visual_module.parameters())
    finally:
        context.shutdown()


def test_visual_requires_grad_default_still_reaches_the_visual_linears(
        tmp_path, monkeypatch):
    """`sensitivity_probe.run_multimodal_visual_probe_pass` passes
    `multimodal=True, visual_requires_grad=True` and needs grad on every
    visual Linear so its Fisher backward hooks fire."""
    path = _write_vl_ckpt(tmp_path, "vl_grad")
    context, _reads = _build(tmp_path, monkeypatch, path,
                             force_multimodal_skeleton=False,
                             multimodal=True, visual_requires_grad=True)
    try:
        linears = [m for _n, m in context.visual_module.named_modules()
                   if isinstance(m, torch.nn.Linear)]
        assert linears, "fixture vision tower has no Linear to track"
        assert all(p.requires_grad
                   for m in linears for p in m.parameters(recurse=False))
        # Body/head stay frozen — only the tower is tracked.
        assert not context.base_model.embed_tokens.weight.requires_grad
    finally:
        context.shutdown()


def test_the_construction_half_of_the_flip_is_load_bearing(tmp_path, monkeypatch):
    """Control on the OTHER axis: without the profile flip this very
    checkpoint cannot be constructed at all at the pinned transformers.

    So the fix cannot be "stop taking the multimodal construction" -- only the
    materialization is separable. Measured on the fixture by the RED run of
    this file (PB action `2964064574d8`): `Qwen2VLTextConfig` is absent from
    `AutoModelForCausalLM`'s mapping and no `Qwen2VLForCausalLM` is importable,
    which is the same shape as glm5_next's declared
    `requires_multimodal_skeleton()`.
    """
    path = _write_vl_ckpt(tmp_path, "vl_textonly")
    with pytest.raises(RuntimeError, match="cannot build a text-only skeleton"):
        _build(tmp_path, monkeypatch, path, force_multimodal_skeleton=False)
