"""An FP8 source checkpoint's config and index are read off the stage (PQ #1219).

#1139 took profile detection off the pool under the Stage B preparation's
strict reads. For a source checkpoint with FP8 weights the source plan's
``_build_fp8_scale_inv_map`` still opened ``config.json`` at its pool path for
the dequant block size and the MXFP4 declarations
(``_declared_weight_block_size``, ``autoscale.declared_fp4_expert_dtype``,
``_check_declared_mxfp4_scale_fmt``), and DeepSeek-V4's ``fp8_scale_pairs``
opened the checkpoint index there. These tests use a real FP8 checkpoint
(``float8_e4m3fn`` weights, ``weight_scale_inv`` block scales, a
``quantization_config.weight_block_size``, an index) and check that:

* the source plan built under a stage double opens neither the config nor the
  index at its pool path, and its FP8 map takes the block size and the MXFP4
  declaration from the staged bytes;
* DeepSeek-V4's ``.scale`` pairing scans the caller's index, not its own open
  (the source plan refuses DeepSeek-V4 before it builds the map, since its
  head extras come from the live tree, so this is checked on the map itself);
* the generator, run with ``--data-manifest-sha256`` and a spill replay over
  the FP8 source, seals the bounds the plain run seals and opens no source
  file at its pool path, for a Qwen3 source and for a Qwen3.5-MoE one (whose
  profile's own index read, for ``source_tensor_name``, the run never
  reaches).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from prismaquant import layer_streaming as ls  # noqa: E402
from prismaquant import stage_b_prep_io as io  # noqa: E402
from test_quantum_executable_readset import (  # noqa: E402
    _exec_campaign, _exec_receipt)
from test_stage_b_prep_io_1070 import (  # noqa: E402
    _recorded_reads, generator_args, write_submission)
from test_stage_b_spill_bound_strict_reads_1139 import (  # noqa: E402
    SPILL, _bind_stage, _bounds, _opened, _sha)
from test_stageb_prepared_inputs_bridge import (  # noqa: E402
    QWEN3_CONFIG, SPILL_EXECUTION)
from test_stageb_readset_source_coverage import (  # noqa: E402
    CKPT_PREFIX, PREFIX, _with_source_paths)

INDEX = "model.safetensors.index.json"
FP8_QUANTIZATION = {"quant_method": "fp8", "weight_block_size": [128, 128]}
FP8_CONFIG = {**QWEN3_CONFIG, "quantization_config": FP8_QUANTIZATION}
#: Qwen3.5-MoE's profile reads the index itself for ``source_tensor_name``
#: (``_checkpoint_source_layout``); the strict run checks it is not reached.
QWEN35_MOE_FP8_CONFIG = {
    "model_type": "qwen3_5_moe",
    "architectures": ["Qwen3_5MoeForConditionalGeneration"],
    "quantization_config": FP8_QUANTIZATION}


@pytest.fixture(autouse=True)
def _unbind_staged_reads():
    io.reset_staged_reads_for_tests()
    yield
    io.reset_staged_reads_for_tests()


def _fp8(rows, cols):
    return torch.randn(rows, cols).clamp(-4, 4).to(torch.float8_e4m3fn)


def _fp8_model(root: Path, config: dict, *, experts: bool = False) -> Path:
    """A 4-layer FP8 checkpoint: each layer's ``mlp.weight`` (and, with
    ``experts``, a routed expert) is ``float8_e4m3fn`` with a one-block
    ``weight_scale_inv``; the names are ``_source_model``'s."""
    model = root / "model"
    model.mkdir()
    weight_map = {}
    for layer in range(4):
        tensors = {
            f"{CKPT_PREFIX}{layer}.norm.weight": torch.ones(32, dtype=torch.bfloat16),
            f"{CKPT_PREFIX}{layer}.mlp.weight": _fp8(64, 128),
            f"{CKPT_PREFIX}{layer}.mlp.weight_scale_inv": torch.ones(1, 1),
        }
        if experts:
            tensors[f"{CKPT_PREFIX}{layer}.mlp.experts.0.down_proj.weight"] = (
                torch.zeros(64, 64, dtype=torch.uint8))
            tensors[f"{CKPT_PREFIX}{layer}.mlp.experts.0.down_proj.weight_scale_inv"] = (
                torch.zeros(64, 4, dtype=torch.uint8))
        if layer == 3:
            tensors[f"{CKPT_PREFIX}2.extra.weight"] = torch.ones(16, dtype=torch.bfloat16)
            tensors["model.language_model.norm.weight"] = torch.ones(64, dtype=torch.bfloat16)
            tensors["lm_head.weight"] = torch.ones(32, 64, dtype=torch.bfloat16)
        if layer == 0:
            tensors["model.language_model.embed_tokens.weight"] = torch.ones(
                32, 64, dtype=torch.bfloat16)
        shard = f"layer-{layer}.safetensors"
        save_file(tensors, str(model / shard))
        weight_map.update({name: shard for name in tensors})
    (model / INDEX).write_text(json.dumps({"weight_map": weight_map}))
    (model / "config.json").write_text(json.dumps(config))
    return model


class _StageReads:
    """``source_reads`` over stage copies of the model's files.

    ``whole`` and ``prefix`` answer from the copies, as
    ``StagedPreparationReads`` answers from the residency map; ``replace``
    swaps a copy's bytes to show which bytes a reader consumed.
    """

    def __init__(self, model: Path, stage: Path):
        stage.mkdir()
        self.copies = {}
        for path in sorted(model.iterdir()):
            copy = stage / path.name
            shutil.copyfile(path, copy)
            self.copies[os.path.normpath(str(path))] = copy

    def replace(self, path, raw: bytes):
        self.copies[os.path.normpath(str(path))].write_bytes(raw)

    def whole(self, path, *, sha256=None, where):
        return self.copies[os.path.normpath(str(path))].read_bytes()

    def prefix(self, path, *, nbytes, where):
        return self.copies[os.path.normpath(str(path))].read_bytes()


def _spy_fp8_map(monkeypatch):
    built = []
    real = ls._build_fp8_scale_inv_map

    def spy(*args, **kwargs):
        built.append(real(*args, **kwargs))
        return built[-1]

    monkeypatch.setattr(ls, "_build_fp8_scale_inv_map", spy)
    return built


@pytest.mark.parametrize("experts", [False, True])
def test_the_source_plan_reads_the_fp8_declarations_off_the_stage(
        tmp_path, monkeypatch, experts):
    config = dict(FP8_CONFIG)
    if experts:
        config["expert_dtype"] = "fp4"
        config["quantization_config"] = {**config["quantization_config"],
                                          "scale_fmt": "ue8m0"}
    model = _fp8_model(tmp_path, config, experts=experts)
    reads = _StageReads(model, tmp_path / "stage")
    # The stage holds a different block size than the pool: the map's
    # block shows which copy the plan read.
    reads.replace(model / "config.json", json.dumps({
        **config, "quantization_config": {**config["quantization_config"],
                                          "weight_block_size": [64, 128]}}).encode())
    built = _spy_fp8_map(monkeypatch)

    plan, opened = _recorded_reads(lambda: ls.streaming_source_plan(
        str(model), layers_prefix=PREFIX, layers=range(4), source_reads=reads))
    opened = _opened(opened)
    for name in ("config.json", INDEX):
        assert str(model / name) not in opened, f"{name} was read at its pool path"
        assert str(reads.copies[str(model / name)]) in opened
    (fp8,) = built
    assert fp8.block == (64, 128), "the FP8 block size came from the pool config"
    assert f"{PREFIX}0.mlp.weight" in fp8
    assert f"{CKPT_PREFIX}0.mlp.weight_scale_inv" in set(plan["span_tensors"].values())
    expert = f"{PREFIX}0.mlp.experts.0.down_proj.weight"
    assert (expert in fp8.mxfp4_names) is experts


def test_a_staged_mxfp4_scale_format_is_checked_from_the_stage(tmp_path):
    """A staged config declaring a non-E8M0 scale format refuses, although
    the pool copy declares ``ue8m0``: the check read the staged bytes."""
    config = {**FP8_CONFIG, "expert_dtype": "fp4",
              "quantization_config": {**FP8_CONFIG["quantization_config"],
                                      "scale_fmt": "ue8m0"}}
    model = _fp8_model(tmp_path, config, experts=True)
    reads = _StageReads(model, tmp_path / "stage")
    reads.replace(model / "config.json", json.dumps({
        **config, "quantization_config": {**config["quantization_config"],
                                          "scale_fmt": "e4m3"}}).encode())
    with pytest.raises(ValueError, match="scale_fmt='e4m3'"):
        ls.streaming_source_plan(str(model), layers_prefix=PREFIX,
                                 layers=range(4), source_reads=reads)


def _dsv4_model(root: Path) -> tuple[Path, dict, dict]:
    model = root / "dsv4"
    model.mkdir()
    tensors = {
        "layers.0.attn.wq.weight": _fp8(128, 128),
        "layers.0.attn.wq.scale": torch.ones(1, 1),
        "layers.0.ffn.experts.0.w1.weight": torch.zeros(128, 64, dtype=torch.int8),
        "layers.0.ffn.experts.0.w1.scale": torch.zeros(128, 4, dtype=torch.uint8),
    }
    save_file(tensors, str(model / "model-0.safetensors"))
    raw = {name: "model-0.safetensors" for name in tensors}
    (model / INDEX).write_text(json.dumps({"weight_map": raw}))
    config = {"model_type": "deepseek_v4", "expert_dtype": "fp4",
              "quantization_config": {"weight_block_size": [128, 128],
                                      "scale_fmt": "ue8m0"}}
    (model / "config.json").write_text(json.dumps(config))
    return model, raw, config


def test_deepseek_v4_pairs_its_scales_from_the_callers_index(tmp_path):
    from prismaquant.model_profiles import detect_profile

    model, raw, config = _dsv4_model(tmp_path)
    profile = detect_profile(str(model), config=config)
    assert profile.name == "deepseek_v4"
    # The source plan never reaches the map for DeepSeek-V4: its head extras
    # come from the live module tree, and it refuses before any FP8 read.
    with pytest.raises(ValueError, match="live"):
        ls.streaming_source_plan(str(model), layers_prefix="model.layers.",
                                 layers=[0], source_reads=_StageReads(
                                     model, tmp_path / "stage"))
    fp8, opened = _recorded_reads(lambda: ls._build_fp8_scale_inv_map(
        str(model), raw_weight_map=raw, profile=profile, config=config))
    opened = _opened(opened)
    assert str(model / INDEX) not in opened, "the index was read at its pool path"
    assert str(model / "config.json") not in opened
    assert fp8.block == (128, 128)
    assert fp8["model.layers.0.self_attn.wq.weight"] == (
        str(model / "model-0.safetensors"), "layers.0.attn.wq.scale")
    assert "model.layers.0.mlp.experts.0.gate_proj.weight" in fp8.mxfp4_names


# -- the generator, strict, over an FP8 source -------------------------------

def _fp8_layout(tmp_path, config):
    """``test_stage_b_spill_bound_strict_reads_1139._layout`` over an FP8
    source checkpoint declaring ``config``."""
    pool = tmp_path / "pool"
    pool.mkdir()
    campaign = _exec_campaign(pool)
    model = _fp8_model(pool, config)
    plan = json.loads(campaign["plan_path"].read_text())
    plan["model"] = str(model)
    plan["execution"] = {**plan["execution"], **SPILL_EXECUTION}
    campaign["plan_path"].write_text(json.dumps(plan))
    campaign["plan_sha"] = _sha(campaign["plan_path"])
    parent = _with_source_paths(json.loads(campaign["parent_path"].read_text()), model)
    head = model / "head.bin"
    head.write_bytes(b"\x01" * 100)
    parent["entries"][0]["path"] = str(head)
    campaign["parent_path"].write_text(json.dumps(parent))
    campaign["parent_sha"] = _sha(campaign["parent_path"])
    receipt, space = _exec_receipt(pool, campaign)
    receipt_path = space / "adjoint-capture.json"
    receipt_path.write_text(json.dumps(receipt))
    metadata = pool / "metadata-generation"
    write_submission(pool, campaign, receipt_path, metadata)
    wire = (pool / "submission" / "read-manifest.json.gz").read_bytes()
    manifest = json.loads(gzip.decompress(wire))
    argv = generator_args(pool, campaign, receipt_path, metadata) + [
        *SPILL, "--data-manifest-sha256", hashlib.sha256(wire).hexdigest()]
    return pool, campaign, receipt_path, model, manifest, argv


@pytest.mark.parametrize("config, profile", [
    (FP8_CONFIG, "qwen3"), (QWEN35_MOE_FP8_CONFIG, "qwen3_5")])
def test_a_strict_run_over_an_fp8_source_never_opens_its_config_or_index_in_the_pool(
        tmp_path, monkeypatch, capsys, config, profile):
    import regenerate_joint_quanta as regen

    pool, campaign, receipt_path, model, manifest, argv = _fp8_layout(
        tmp_path, config)
    config_path, index = str(model / "config.json"), str(model / INDEX)
    declared = {os.path.normpath(entry["path"]) for entry in manifest["entries"]}
    assert {config_path, index} <= declared
    plain = pool / "plain-generation"
    assert regen.main(generator_args(pool, campaign, receipt_path, plain)
                      + list(SPILL)) == 0
    expected = _bounds(plain / "records")
    assert len(expected) == 4

    built = _spy_fp8_map(monkeypatch)
    copies = _bind_stage(tmp_path, monkeypatch, manifest,
                         argv[argv.index("--data-manifest-sha256") + 1])
    capsys.readouterr()
    code, opened = _recorded_reads(lambda: regen.main(argv))
    assert code == 0
    assert f"source plan: profile {profile} " in capsys.readouterr().out
    assert _bounds(pool / "metadata-generation" / "records") == expected
    assert built and all(fp8.block == (128, 128) and fp8 for fp8 in built), \
        "the source plan built no FP8 map"
    opened = _opened(opened)
    for path in (config_path, index):
        assert path not in opened, f"{path} was read at its pool path"
        assert copies[path] in opened
    in_pool = sorted(path for path in opened
                     if path.startswith(str(model) + os.sep))
    assert not in_pool, f"read at their pool paths: {in_pool}"
