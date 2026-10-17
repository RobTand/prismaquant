"""Fused-sibling NVFP4 joint globals: resident fill == streamed fill == export.

vLLM serves q/k/v (and gate/up) from one fused parameter with one
``weight_global_scale``, so every builder renders those siblings against a
joint per-tensor global, and the exporter re-derives the shipped NVFP4 codes
and FP8 block scales from the cached dequant under the joint global it
recomputes.  Principle 8 needs those to be one rendering: a builder that
derives the joint global under a different recipe from the one export uses
ships bytes the validated render never produced.

These cases drive the real algorithms on a tiny CPU fixture with the JSO
recipe (``gptq + static_act_order + joint_scale_opt`` records
``nvfp4_scale_rule=joint_mse``): the resident fill, the streamed fill, both
exporter pre-passes and the exporter's cached re-pack.  The fixture makes the
``joint_mse`` and ``static_6`` globals differ, so a builder on the wrong recipe
cannot pass by coincidence.

A cache's recipe record cannot show which global its fused siblings actually
used: streamed caches built before the fix record the same ``joint_mse``
levers.  Each fill therefore witnesses, per entry, the exact joint global it
rendered on and the shard it published, and export refuses a fused re-pack
unless the bytes it read are that shard and the global is the witnessed one.
The witness cases cover legacy refusal, the pickle, resident resume, a shard
another build or writer changed after its fill witnessed it (review N3-01),
moved and copied directories, and the cases a refusal must leave alone.
"""
from __future__ import annotations

import copy
import json
import math
import os
import pickle
import re
import shutil
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import prismaquant.export_native_compressed as enc
import prismaquant.native_render_codec as codec
import prismaquant.streaming_production_cache as streaming
from prismaquant.measure_quant_cost import ActivationIndex
from prismaquant.model_profiles import DefaultProfile
from prismaquant.native_render_levers import NativeRenderLevers
from prismaquant.nvfp4_activation_contract import _E2M1_POSITIVE
from prismaquant.file_identity import file_stat_signature
from prismaquant.production_weight_cache import (
    MTP_APPEND_SIDECAR_KEY,
    NVFP4_JOINT_GLOBAL_WITNESS_KEY,
    NVFP4_JOINT_GLOBAL_WITNESS_SIDECAR_FILENAME,
    RENDER_IDENTITY_SIDECAR_FILENAME,
    STREAMED_FILL_IDENTITY_SCHEMA,
    STREAMED_FILL_SIDECAR_KEY,
    NVFP4JointGlobalWitness,
    ProductionWeightCache,
    PublishedShard,
    _check_and_record_append_identity,
    _LinearActivationCollector,
    build_mtp_append_identity,
    fill_production_weight_cache,
    nvfp4_joint_global_witness_problem,
    nvfp4_joint_global_witness_section,
    nvfp4_joint_global_witnesses,
)
from prismaquant.render_cache_paths import cache_weight_filename
from prismaquant.streaming_production_cache import run_streaming_render

HIDDEN = 32
INTER = 32
VOCAB = 64
LAYER = "model.layers.0"
JSO = {"gptq": True, "static_act_order": True, "joint_scale_opt": True}
FUSED = {
    f"{LAYER}.self_attn.{leaf}" for leaf in ("q_proj", "k_proj", "v_proj")
} | {f"{LAYER}.mlp.{leaf}" for leaf in ("gate_proj", "up_proj")}


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.k_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.v_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, h):
        return self.o_proj(self.q_proj(h) + self.k_proj(h) + self.v_proj(h))


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(HIDDEN, INTER, bias=False)
        self.up_proj = nn.Linear(HIDDEN, INTER, bias=False)
        self.down_proj = nn.Linear(INTER, HIDDEN, bias=False)

    def forward(self, h):
        return self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = _Mlp()

    def forward(self, h):
        # Every Linear sees its input; the residual stays the embedding so the
        # activation cache and the resident fill's own forward are identical.
        self.self_attn(h)
        self.mlp(h)
        return h


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, HIDDEN)
        self.layers = nn.ModuleList([_Layer()])

    def forward(self, input_ids):
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h)
        return h


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()

    def forward(self, input_ids, use_cache=False, **_kwargs):
        return self.model(input_ids)


def _make_model() -> _Model:
    torch.manual_seed(20261016)
    model = _Model()
    with torch.no_grad():
        # One group per fused group whose largest magnitudes sit on the
        # max-to-4 codebook levels: max-to-4 renders it almost exactly and
        # wins the joint_mse choice, so the joint_mse global is 1.5x the
        # static_6 one. The two planted peaks differ by a factor that is not
        # a power of two: the FP8 block scales absorb a power-of-two change of
        # global exactly, so only distinct, non-power-of-two globals make a
        # group mix-up or a stale global visible in the re-packed bytes.
        pattern = torch.tensor([1.0, 0.75, 0.5, 0.25] * 4)
        for linear, multiple in (
            (model.model.layers[0].self_attn.q_proj, 2.6),
            (model.model.layers[0].mlp.gate_proj, 2.0),
        ):
            peak = multiple * float(linear.weight.abs().max())
            linear.weight[0, :16] = pattern * peak
    return model.to(torch.bfloat16).eval()


def _power_of_two_apart(a: float, b: float) -> bool:
    exponent = math.log2(a / b)
    return abs(exponent - round(exponent)) < 1e-6


def _dense_qnames(model) -> list[str]:
    return sorted(
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    )


def _write_activation_cache(model, calib_ids, qnames, act_dir: Path) -> None:
    collector = _LinearActivationCollector(
        model, set(qnames), max_rows=4096, store_qnames=set(qnames),
        store_device="cpu", store_dtype=torch.float32,
    )
    collector.install()
    try:
        with torch.no_grad():
            for row in range(calib_ids.size(0)):
                model(calib_ids[row:row + 1], use_cache=False)
    finally:
        collector.remove()
    for name, inputs in collector.collected().items():
        stem = re.sub(r"[^A-Za-z0-9_-]", "__", name)
        torch.save({"inputs": inputs.detach().cpu(), "name": name},
                   act_dir / f"{stem}.pt")


def _joint_global_spy(monkeypatch):
    """Record every fused joint global the builders compute (real algorithm)."""
    seen: list[dict[str, torch.Tensor]] = []
    real = enc._compute_nvfp4_joint_global

    def spy(*args, **kwargs):
        result = real(*args, **kwargs)
        seen.append({name: value.clone() for name, value in result.items()})
        return result

    monkeypatch.setattr(enc, "_compute_nvfp4_joint_global", spy)
    return seen


def _decode(packed: dict[str, torch.Tensor], global_real: torch.Tensor):
    """Dequantize an exported NVFP4 triple with the codec's own arithmetic."""
    weight_packed = packed["weight_packed"]
    rows, cols = weight_packed.shape[0], weight_packed.shape[1] * 2
    low = (weight_packed & 0xF).long()
    high = ((weight_packed >> 4) & 0xF).long()
    index = torch.stack([low, high], dim=-1).reshape(rows, cols)
    sign = -((index >> 3).to(torch.float32) * 2 - 1)
    levels = torch.tensor(_E2M1_POSITIVE, dtype=torch.float32)[index & 0x7]
    eff = (packed["weight_scale"].to(torch.float32)
           * global_real.to(torch.float32)).clamp_min(1e-12)
    eff = eff.unsqueeze(-1).expand(-1, -1, 16).reshape(rows, cols)
    return sign * levels * eff


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    model = _make_model()
    profile = DefaultProfile()
    qnames = _dense_qnames(model)
    assignment = {name: "NVFP4" for name in qnames}
    calib_ids = torch.randint(
        0, VOCAB, (2, 8), generator=torch.Generator().manual_seed(7),
    )
    act_dir = tmp_path_factory.mktemp("act")
    _write_activation_cache(model, calib_ids, qnames, act_dir)

    mp = pytest.MonkeyPatch()
    try:
        # The JSO record must choose joint_mse itself, not an ambient rule.
        mp.delenv("PRISMAQUANT_NVFP4_SCALE_RULE", raising=False)
        seen = _joint_global_spy(mp)
        resident = fill_production_weight_cache(
            model, calib_ids, qnames,
            formats=["NVFP4"], render_assignment=assignment,
            levers=dict(JSO), max_act_rows=4096,
            recache_profile=profile, progress=False,
        )
        resident_globals = seen[-1]
        streamed = run_streaming_render(
            model,
            layers_prefix="model.layers.", num_layers=1,
            render_assignment=assignment,
            act_index=ActivationIndex(act_dir, []),
            formats=["NVFP4"], levers=dict(JSO),
            cache_dir_path=None, profile=profile, skip_tokens=[],
            device=torch.device("cpu"), max_act_rows=4096, progress=False,
        )
        streamed_globals = seen[-1]
    finally:
        mp.undo()
    return {
        "model": model,
        "profile": profile,
        "assignment": assignment,
        "calib_ids": calib_ids,
        "qnames": qnames,
        "act_dir": act_dir,
        "resident": resident,
        "streamed": streamed,
        "resident_globals": resident_globals,
        "streamed_globals": streamed_globals,
    }


def _export_globals(rendered, cache, monkeypatch):
    monkeypatch.setattr(enc, "_PRODUCTION_WEIGHT_CACHE", cache)
    monkeypatch.setattr(enc, "_RENDER_LEVERS", NativeRenderLevers())
    monkeypatch.setattr(enc, "_INPUT_GLOBAL_SCALES", None)
    model, profile = rendered["model"], rendered["profile"]
    options = enc._export_render_options()
    layer = enc._compute_layer_joint_nvfp4(
        model.get_submodule(LAYER), LAYER, rendered["assignment"], profile,
        levers=options,
    )
    whole = enc._compute_nvfp4_joint_global(
        model, rendered["assignment"], profile=profile, levers=options,
    )
    return options, layer, whole


def test_fixture_separates_the_joint_mse_and_static_6_globals(rendered):
    weight = rendered["model"].get_submodule(f"{LAYER}.self_attn.q_proj").weight
    weight = weight.detach().float()
    static = codec.compute_nvfp4_global_real(
        weight, levers=NativeRenderLevers(nvfp4_scale_rule="static_6"))
    joint = codec.compute_nvfp4_global_real(
        weight, levers=NativeRenderLevers(nvfp4_scale_rule="joint_mse"))
    assert float(joint) > float(static)
    assert rendered["resident"].levers["nvfp4_scale_rule"] == "joint_mse"


def test_fixture_separates_the_two_fused_groups(rendered, monkeypatch):
    # Each group shares one joint global, and the two differ by a factor that
    # is not a power of two, so a witness keyed or valued from the other group
    # is a different value and re-packs to different bytes.
    _options, layer, _whole = _export_globals(
        rendered, rendered["resident"], monkeypatch)
    attention = {float(layer[f"{LAYER}.self_attn.{leaf}"])
                 for leaf in ("q_proj", "k_proj", "v_proj")}
    mlp = {float(layer[f"{LAYER}.mlp.{leaf}"]) for leaf in ("gate_proj", "up_proj")}
    assert len(attention) == 1 and len(mlp) == 1
    assert not _power_of_two_apart(attention.pop(), mlp.pop())


@pytest.mark.parametrize("builder", ["resident", "streamed"])
def test_builders_render_fused_siblings_on_the_export_joint_global(
    rendered, builder, monkeypatch,
):
    cache = rendered[builder]
    options, layer, whole = _export_globals(rendered, cache, monkeypatch)
    assert options == NativeRenderLevers.from_record(cache.levers)
    assert set(layer) == FUSED and set(whole) == FUSED
    fill_globals = rendered[f"{builder}_globals"]
    assert set(fill_globals) == FUSED
    for name in sorted(FUSED):
        assert torch.equal(layer[name], whole[name]), name
        assert torch.equal(fill_globals[name], layer[name]), (
            f"{builder} fill rendered {name} on joint global "
            f"{float(fill_globals[name])!r}; export re-derives under "
            f"{float(layer[name])!r}"
        )


@pytest.mark.parametrize("builder", ["resident", "streamed"])
def test_export_repack_reproduces_the_cached_fused_render(
    rendered, builder, monkeypatch,
):
    cache = rendered[builder]
    _options, layer, _whole = _export_globals(rendered, cache, monkeypatch)
    for name in sorted(FUSED):
        cached = cache.get(name, "NVFP4")
        packed = enc._pack_production_cached_2d(
            name, "NVFP4", nvfp4_global_real_override=layer[name],
            device=torch.device("cpu"),
        )
        assert torch.equal(
            packed["weight_global_scale"],
            (1.0 / layer[name]).to(torch.float32).reshape(1),
        )
        shipped = _decode(packed, layer[name]).to(cached.dtype)
        differing = int((shipped != cached).sum())
        assert differing == 0, (
            f"{builder}: exported {name} differs from the cached render in "
            f"{differing}/{cached.numel()} elements (max |d|="
            f"{float((shipped.float() - cached.float()).abs().max()):.4g})"
        )


def test_streamed_and_resident_fused_renders_are_bit_identical(rendered):
    for name in sorted(FUSED):
        resident = rendered["resident"].get(name, "NVFP4")
        streamed = rendered["streamed"].get(name, "NVFP4")
        assert torch.equal(resident, streamed), name


UNWITNESSED = "carries no fused-sibling joint-global witness.*fresh --cache-dir"
UNBOUND = "is not the file its producer published.*fresh --cache-dir"


def _without_witness(cache):
    """The same cache as a fill before the witness existed would leave it.

    Its recipe record, render scores and bytes are unchanged; only the
    per-entry joint-global witness is absent.
    """
    legacy = copy.copy(cache)
    legacy.metadata = {
        key: value for key, value in cache.metadata.items()
        if key != NVFP4_JOINT_GLOBAL_WITNESS_KEY
    }
    assert legacy.levers["nvfp4_scale_rule"] == "joint_mse"
    return legacy


def _pack_fused(name, joint_global):
    return enc._pack_production_cached_2d(
        name, "NVFP4", nvfp4_global_real_override=joint_global,
        device=torch.device("cpu"))


def _ungated_repack(cache, name, joint_global):
    """The bytes the default re-pack would ship if nothing checked the witness."""
    packed, scale, _global = codec.quantize_dequantize_nvfp4(
        cache.get(name, "NVFP4").to(torch.float32), group_size=16,
        global_real_override=joint_global,
        levers=enc._export_render_options(cache))
    return _decode({"weight_packed": packed, "weight_scale": scale}, joint_global)


def _assert_exports_bit_exact(cache, layer, names):
    for name in sorted(names):
        cached = cache.get(name, "NVFP4")
        packed = _pack_fused(name, layer[name])
        assert torch.equal(_decode(packed, layer[name]).to(cached.dtype), cached), name


def _resident_fill(rendered, cache_dir):
    return fill_production_weight_cache(
        rendered["model"], rendered["calib_ids"], rendered["qnames"],
        formats=["NVFP4"], render_assignment=rendered["assignment"],
        levers=dict(JSO), max_act_rows=4096,
        recache_profile=rendered["profile"], progress=False,
        cache_dir=cache_dir,
    )


def _streamed_fill(rendered, cache_dir, *, levers=JSO, include_qnames=None):
    return run_streaming_render(
        rendered["model"],
        layers_prefix="model.layers.", num_layers=1,
        render_assignment=rendered["assignment"],
        act_index=ActivationIndex(rendered["act_dir"], []),
        formats=["NVFP4"], levers=dict(levers),
        cache_dir_path=cache_dir, profile=rendered["profile"], skip_tokens=[],
        device=torch.device("cpu"), max_act_rows=4096,
        include_qnames=include_qnames, progress=False,
    )


def _overwrite_with_streamed(rendered, cache_dir, monkeypatch, **kwargs):
    """A streamed build writing into an existing cache directory.

    The streamed fill's directory claim refuses this now; a build from before
    the claim, or any other writer, still can. Bypassing the claim keeps the
    shard binding under test on its own.
    """
    with monkeypatch.context() as patch:
        patch.setattr(
            streaming, "claim_streamed_cache_directory",
            lambda *_args, **_kwargs: None)
        return _streamed_fill(rendered, cache_dir, **kwargs)


def _replace_by_copy(path: Path) -> None:
    """Replace a file by a byte-identical copy: the same bytes, another file."""
    copied = path.with_name(path.name + ".copy")
    shutil.copy2(path, copied)
    os.replace(copied, path)


@pytest.mark.parametrize("builder", ["resident", "streamed"])
def test_each_fill_witnesses_exactly_its_fused_nvfp4_globals(
    rendered, builder, monkeypatch,
):
    cache = rendered[builder]
    _options, layer, _whole = _export_globals(rendered, cache, monkeypatch)
    witnesses = nvfp4_joint_global_witnesses(cache.metadata, where="test")
    # Unfused NVFP4 entries (o_proj, down_proj) render on their own global
    # and carry no witness. These fills keep their renders in memory, so the
    # witnesses name no shard.
    assert witnesses == {
        (name, "NVFP4"): NVFP4JointGlobalWitness(float(layer[name]), None)
        for name in FUSED
    }


@pytest.mark.parametrize("builder", ["resident", "streamed"])
def test_disk_fills_bind_each_witness_to_the_shard_they_published(
    rendered, builder, tmp_path, monkeypatch,
):
    monkeypatch.delenv("PRISMAQUANT_NVFP4_SCALE_RULE", raising=False)
    cache_dir = tmp_path / "pwc"
    cache_dir.mkdir()
    fill = _resident_fill if builder == "resident" else _streamed_fill
    cache = pickle.loads(pickle.dumps(fill(rendered, cache_dir)))
    witnesses = nvfp4_joint_global_witnesses(cache.metadata, where="test")
    assert set(witnesses) == {(name, "NVFP4") for name in FUSED}
    for (name, fmt), witness in witnesses.items():
        shard = cache_weight_filename(name, fmt)
        assert witness.shard == PublishedShard(
            shard, file_stat_signature((cache_dir / shard).lstat()))
    _options, layer, _whole = _export_globals(rendered, cache, monkeypatch)
    _assert_exports_bit_exact(cache, layer, FUSED)


def test_export_refuses_legacy_streamed_joint_mse_caches(rendered, monkeypatch):
    cache = _without_witness(rendered["streamed"])
    _options, globals_by_name, _whole = _export_globals(rendered, cache, monkeypatch)
    assert cache.metadata["streaming"] is True
    for name in sorted(FUSED):
        with pytest.raises(RuntimeError, match=UNWITNESSED):
            _pack_fused(name, globals_by_name[name])


def test_export_refuses_legacy_resident_joint_mse_caches(rendered, monkeypatch):
    # The original resident fill also derived these globals from the ambient
    # recipe; its caches carry no recipe record this exporter accepts, and a
    # resident record alone does not show which producer rendered an entry.
    cache = _without_witness(rendered["resident"])
    _options, globals_by_name, _whole = _export_globals(rendered, cache, monkeypatch)
    name = sorted(FUSED)[0]
    with pytest.raises(RuntimeError, match=UNWITNESSED):
        _pack_fused(name, globals_by_name[name])


def test_export_refuses_a_witness_of_another_global(rendered, monkeypatch):
    cache = copy.copy(rendered["streamed"])
    witnesses = nvfp4_joint_global_witnesses(cache.metadata, where="test")
    name = sorted(FUSED)[0]
    key = (name, "NVFP4")
    witnesses[key] = witnesses[key]._replace(
        joint_global=witnesses[key].joint_global * 1.5)
    cache.metadata = {
        **cache.metadata,
        NVFP4_JOINT_GLOBAL_WITNESS_KEY: (
            nvfp4_joint_global_witness_section(witnesses)
        ),
    }
    _options, globals_by_name, _whole = _export_globals(rendered, cache, monkeypatch)
    with pytest.raises(RuntimeError, match="rendered on fused joint global"):
        _pack_fused(name, globals_by_name[name])


def test_export_refuses_a_v1_witness_that_names_no_shard(rendered, monkeypatch):
    cache = copy.copy(rendered["streamed"])
    witnesses = nvfp4_joint_global_witnesses(cache.metadata, where="test")
    cache.metadata = {
        **cache.metadata,
        NVFP4_JOINT_GLOBAL_WITNESS_KEY: {
            "schema": (
                "prismaquant.production_weight_cache.nvfp4_joint_global_witness.v1"
            ),
            "records": {
                f"{qname}|{fmt}": witness.joint_global
                for (qname, fmt), witness in witnesses.items()
            },
        },
    }
    _options, globals_by_name, _whole = _export_globals(rendered, cache, monkeypatch)
    name = sorted(FUSED)[0]
    with pytest.raises(ValueError, match="schema v1.*fresh --cache-dir"):
        _pack_fused(name, globals_by_name[name])


def test_witness_binding_names_each_way_bytes_can_differ(tmp_path):
    shard = tmp_path / "entry__NVFP4.pt"
    torch.save(torch.ones(2), shard)
    published = PublishedShard(shard.name, file_stat_signature(shard.lstat()))
    bound = NVFP4JointGlobalWitness(0.5, published)
    in_memory = NVFP4JointGlobalWitness(0.5, None)

    assert nvfp4_joint_global_witness_problem(
        bound, served=shard.name, path=shard) is None
    assert nvfp4_joint_global_witness_problem(
        in_memory, served=None, path=None) is None
    assert "in-memory render" in nvfp4_joint_global_witness_problem(
        in_memory, served=shard.name, path=shard)
    assert "holds the entry in memory" in nvfp4_joint_global_witness_problem(
        bound, served=None, path=None)
    assert "serves it from other" in nvfp4_joint_global_witness_problem(
        bound, served="other__NVFP4.pt", path=tmp_path / "other__NVFP4.pt")
    _replace_by_copy(shard)
    assert "is not the file its producer published" in (
        nvfp4_joint_global_witness_problem(bound, served=shard.name, path=shard))
    shard.unlink()
    assert "cannot be checked" in nvfp4_joint_global_witness_problem(
        bound, served=shard.name, path=shard)


@pytest.mark.parametrize("filesystem", ["nfs", "nfs4"])
@pytest.mark.parametrize("builder", ["resident", "streamed"])
def test_fused_export_and_resume_accept_the_same_shard_on_another_nfs_mount(
    rendered, filesystem, builder, tmp_path, monkeypatch,
):
    from prismaquant import streamed_model_identity as identity

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    monkeypatch.delenv("PRISMAQUANT_NVFP4_SCALE_RULE", raising=False)
    monkeypatch.setattr(identity, "_mount_filesystem_type", lambda _path: filesystem)
    cache_dir = tmp_path / "pwc"
    fill = _resident_fill if builder == "resident" else _streamed_fill
    cache = fill(rendered, cache_dir)
    witnesses = nvfp4_joint_global_witnesses(cache.metadata, where="test")
    # The persisted device is from the writer's mount. All four file fields
    # still identify the same server inode on the reader's NFS mount.
    witnesses = {
        key: witness._replace(shard=PublishedShard(
            witness.shard.name,
            (witness.shard.stat[0] + 1, *witness.shard.stat[1:])))
        for key, witness in witnesses.items()
    }
    section = nvfp4_joint_global_witness_section(witnesses)
    cache.metadata[NVFP4_JOINT_GLOBAL_WITNESS_KEY] = section
    if builder == "resident":
        (cache_dir / NVFP4_JOINT_GLOBAL_WITNESS_SIDECAR_FILENAME).write_text(
            json.dumps(section))
        cache = _resident_fill(rendered, cache_dir)
        assert nvfp4_joint_global_witnesses(cache.metadata, where="test") == witnesses
    _options, globals_by_name, _whole = _export_globals(rendered, cache, monkeypatch)
    _assert_exports_bit_exact(cache, globals_by_name, FUSED)


@pytest.mark.parametrize("filesystem", ["ext4", "zfs", None])
@pytest.mark.parametrize("dev_mode", ["0", "1"])
def test_fused_witness_refuses_other_local_devices_even_in_dev_mode(
    filesystem, dev_mode, tmp_path, monkeypatch,
):
    from prismaquant import streamed_model_identity as identity

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", dev_mode)
    monkeypatch.setattr(identity, "_mount_filesystem_type", lambda _path: filesystem)
    shard = tmp_path / "entry__NVFP4.pt"
    shard.write_bytes(b"rendered bytes")
    observed = file_stat_signature(shard.lstat())
    witness = NVFP4JointGlobalWitness(0.5, PublishedShard(
        shard.name, (observed[0] + 1, *observed[1:])))
    assert "is not the file its producer published" in (
        nvfp4_joint_global_witness_problem(
            witness, served=shard.name, path=shard))


@pytest.mark.parametrize("field", [1, 2, 3, 4], ids=["inode", "size", "mtime", "ctime"])
def test_fused_witness_refuses_file_identity_drift_across_nfs_mounts(
    field, tmp_path, monkeypatch,
):
    from prismaquant import streamed_model_identity as identity

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "1")
    monkeypatch.setattr(identity, "_mount_filesystem_type", lambda _path: "nfs4")
    shard = tmp_path / "entry__NVFP4.pt"
    shard.write_bytes(b"rendered bytes")
    changed = list(file_stat_signature(shard.lstat()))
    changed[0] += 1
    changed[field] += 1
    witness = NVFP4JointGlobalWitness(0.5, PublishedShard(shard.name, tuple(changed)))
    assert "is not the file its producer published" in (
        nvfp4_joint_global_witness_problem(
            witness, served=shard.name, path=shard))


def test_legacy_caches_still_pack_unfused_entries(rendered, monkeypatch):
    cache = _without_witness(rendered["streamed"])
    _export_globals(rendered, cache, monkeypatch)
    for name in (f"{LAYER}.self_attn.o_proj", f"{LAYER}.mlp.down_proj"):
        packed = enc._pack_production_cached_2d(
            name, "NVFP4", device=torch.device("cpu"))
        assert set(packed) >= {"weight_packed", "weight_scale",
                               "weight_global_scale"}


def test_match_render_scale_opt_out_does_not_consult_the_witness(
    rendered, monkeypatch,
):
    # PRISMAQUANT_NVFP4_EXPORT_MATCH_RENDER_SCALE=0 re-derives under the
    # exporter's own recipe by design and claims no render parity.
    cache = _without_witness(rendered["streamed"])
    _options, globals_by_name, _whole = _export_globals(rendered, cache, monkeypatch)
    monkeypatch.setattr(
        enc, "_RENDER_LEVERS",
        NativeRenderLevers(nvfp4_export_match_render_scale=False))
    name = sorted(FUSED)[0]
    assert _pack_fused(name, globals_by_name[name]) is not None


@pytest.mark.parametrize("builder", ["resident", "streamed"])
def test_witness_survives_the_cache_pickle(rendered, builder, monkeypatch):
    cache = pickle.loads(pickle.dumps(rendered[builder]))
    _options, layer, _whole = _export_globals(rendered, cache, monkeypatch)
    _assert_exports_bit_exact(cache, layer, FUSED)
    legacy = pickle.loads(pickle.dumps(_without_witness(rendered[builder])))
    _export_globals(rendered, legacy, monkeypatch)
    with pytest.raises(RuntimeError, match=UNWITNESSED):
        _pack_fused(sorted(FUSED)[0], layer[sorted(FUSED)[0]])


def test_resident_resume_keeps_the_witnesses_its_fill_published(
    rendered, tmp_path, monkeypatch,
):
    monkeypatch.delenv("PRISMAQUANT_NVFP4_SCALE_RULE", raising=False)
    cache_dir = tmp_path / "pwc"
    first = _resident_fill(rendered, cache_dir)
    assert (cache_dir / NVFP4_JOINT_GLOBAL_WITNESS_SIDECAR_FILENAME).is_file()
    resumed = pickle.loads(pickle.dumps(_resident_fill(rendered, cache_dir)))
    witnesses = nvfp4_joint_global_witnesses(resumed.metadata, where="test")
    assert witnesses == nvfp4_joint_global_witnesses(first.metadata, where="test")
    assert set(witnesses) == {(name, "NVFP4") for name in FUSED}
    _options, layer, _whole = _export_globals(rendered, resumed, monkeypatch)
    _assert_exports_bit_exact(resumed, layer, FUSED)


def test_resident_resume_warns_about_a_resumed_shard_without_a_witness(
    rendered, tmp_path, monkeypatch, capsys,
):
    monkeypatch.delenv("PRISMAQUANT_NVFP4_SCALE_RULE", raising=False)
    cache_dir = tmp_path / "pwc"
    _resident_fill(rendered, cache_dir)
    missing_name = sorted(FUSED)[0]
    missing_shard = cache_dir / cache_weight_filename(missing_name, "NVFP4")
    sidecar = cache_dir / NVFP4_JOINT_GLOBAL_WITNESS_SIDECAR_FILENAME
    record = json.loads(sidecar.read_text())
    del record["records"][f"{missing_name}|NVFP4"]
    sidecar.write_text(json.dumps(record))
    capsys.readouterr()

    def no_joint_recalculation(*args, **kwargs):
        raise AssertionError("a fully resumed cache recalculated joint globals")

    with monkeypatch.context() as fully_resumed:
        fully_resumed.setattr(enc, "_compute_nvfp4_joint_global", no_joint_recalculation)
        resumed = _resident_fill(rendered, cache_dir)
    warning = capsys.readouterr().out
    assert "resume adopts no fused NVFP4 joint-global witness for 1" in warning
    assert missing_shard.name in warning
    assert "Remove only the affected shards" in warning
    assert set(nvfp4_joint_global_witnesses(resumed.metadata, where="test")) == {
        (name, "NVFP4") for name in FUSED - {missing_name}}
    _options, layer, _whole = _export_globals(rendered, resumed, monkeypatch)
    with pytest.raises(RuntimeError, match=UNWITNESSED):
        _pack_fused(missing_name, layer[missing_name])

    missing_shard.unlink()
    repaired = _resident_fill(rendered, cache_dir)
    _options, layer, _whole = _export_globals(rendered, repaired, monkeypatch)
    _assert_exports_bit_exact(repaired, layer, FUSED)


def test_resident_resume_witnesses_only_the_shard_it_renders(
    rendered, tmp_path, monkeypatch,
):
    monkeypatch.delenv("PRISMAQUANT_NVFP4_SCALE_RULE", raising=False)
    cache_dir = tmp_path / "pwc"
    _resident_fill(rendered, cache_dir)
    # A directory a pre-witness fill left: the same shards, no witnesses.
    (cache_dir / NVFP4_JOINT_GLOBAL_WITNESS_SIDECAR_FILENAME).unlink()
    renewed = sorted(FUSED)[0]
    (cache_dir / cache_weight_filename(renewed, "NVFP4")).unlink()
    resumed = _resident_fill(rendered, cache_dir)
    assert set(nvfp4_joint_global_witnesses(resumed.metadata, where="test")) == {
        (renewed, "NVFP4")}
    _options, layer, _whole = _export_globals(rendered, resumed, monkeypatch)
    assert _pack_fused(renewed, layer[renewed]) is not None
    for name in sorted(FUSED - {renewed}):
        with pytest.raises(RuntimeError, match=UNWITNESSED):
            _pack_fused(name, layer[name])


@pytest.mark.parametrize("variant", ["recipe", "scope"])
def test_resident_resume_adopts_no_witness_for_shards_another_build_overwrote(
    rendered, variant, tmp_path, monkeypatch, capsys,
):
    """Review N3-01, inverted: a streamed build overwrote fused shards of a
    resident directory, under another recipe or a narrower scope, and the
    resident fill then resumed the directory with its original settings.
    The streamed fill's own directory claim now refuses that overwrite
    (``test_streamed_fill_refuses_a_directory_another_rendering_holds``);
    here it is bypassed so the shard binding is shown on its own."""
    monkeypatch.delenv("PRISMAQUANT_NVFP4_SCALE_RULE", raising=False)
    cache_dir = tmp_path / "pwc"
    first = _resident_fill(rendered, cache_dir)
    if variant == "recipe":
        # gptq without joint_scale_opt re-renders every fused shard on its
        # static_6 joint global.
        overwritten = set(FUSED)
        streamed = _overwrite_with_streamed(
            rendered, cache_dir, monkeypatch, levers={"gptq": True})
    else:
        # The same JSO recipe narrowed to k_proj: its joint global spans k_proj
        # alone.
        overwritten = {f"{LAYER}.self_attn.k_proj"}
        streamed = _overwrite_with_streamed(
            rendered, cache_dir, monkeypatch, include_qnames=sorted(overwritten))
    first_witnesses = nvfp4_joint_global_witnesses(first.metadata, where="first")
    streamed_witnesses = nvfp4_joint_global_witnesses(
        streamed.metadata, where="streamed")
    for name in overwritten:
        # The overwrite's global is neither the resident one nor a power of
        # two away from it, so stale bytes would show in the re-pack.
        assert not _power_of_two_apart(
            streamed_witnesses[(name, "NVFP4")].joint_global,
            first_witnesses[(name, "NVFP4")].joint_global)
    capsys.readouterr()

    resumed = pickle.loads(pickle.dumps(_resident_fill(rendered, cache_dir)))

    assert "resume adopts no fused NVFP4 joint-global witness for " + str(
        len(overwritten)) in capsys.readouterr().out
    assert set(nvfp4_joint_global_witnesses(resumed.metadata, where="resumed")) == {
        (name, "NVFP4") for name in FUSED - overwritten}
    _options, layer, _whole = _export_globals(rendered, resumed, monkeypatch)
    for name in sorted(overwritten):
        assert float(layer[name]) == first_witnesses[(name, "NVFP4")].joint_global
        with pytest.raises(RuntimeError, match=UNWITNESSED):
            _pack_fused(name, layer[name])
        # Without the binding the export would ship bytes that are neither
        # this cached render nor the resident render the witness named.
        cached = resumed.get(name, "NVFP4")
        stale = _ungated_repack(resumed, name, layer[name]).to(cached.dtype)
        assert not torch.equal(stale, cached), name
    _assert_exports_bit_exact(resumed, layer, FUSED - overwritten)


@pytest.mark.parametrize("change", ["replaced", "rewritten"])
def test_export_refuses_a_shard_changed_after_its_fill_witnessed_it(
    rendered, change, tmp_path, monkeypatch,
):
    """The binding alone: no other producer runs, only the shard file changes."""
    monkeypatch.delenv("PRISMAQUANT_NVFP4_SCALE_RULE", raising=False)
    cache_dir = tmp_path / "pwc"
    cache = pickle.loads(pickle.dumps(_resident_fill(rendered, cache_dir)))
    name = f"{LAYER}.mlp.gate_proj"
    shard = cache_dir / cache_weight_filename(name, "NVFP4")
    if change == "replaced":
        # A byte-identical copy renamed over the shard is another file.
        _replace_by_copy(shard)
    else:
        # The same file rewritten in place. The file identity resolves time to
        # one kernel timestamp tick, so let one pass after the publish.
        time.sleep(0.05)
        torch.save(cache.get(f"{LAYER}.mlp.up_proj", "NVFP4"), shard)
    _options, layer, _whole = _export_globals(rendered, cache, monkeypatch)
    with pytest.raises(RuntimeError, match=UNBOUND):
        _pack_fused(name, layer[name])
    _assert_exports_bit_exact(cache, layer, FUSED - {name})


def test_export_checks_the_witness_after_reading_the_entry(
    rendered, tmp_path, monkeypatch,
):
    monkeypatch.delenv("PRISMAQUANT_NVFP4_SCALE_RULE", raising=False)
    cache_dir = tmp_path / "pwc"
    cache = pickle.loads(pickle.dumps(_resident_fill(rendered, cache_dir)))
    name = f"{LAYER}.self_attn.q_proj"
    assert cache.prefetch([(name, "NVFP4")]) == 1
    # The resident bytes are still the witnessed render, but once the shard is
    # replaced the cache can no longer show it, so the export refuses.
    _replace_by_copy(cache_dir / cache_weight_filename(name, "NVFP4"))
    _options, layer, _whole = _export_globals(rendered, cache, monkeypatch)
    with pytest.raises(RuntimeError, match=UNBOUND):
        _pack_fused(name, layer[name])


def test_export_refuses_a_shard_replaced_between_check_and_load(
    rendered, tmp_path, monkeypatch,
):
    """Another writer replaces the shard just before the cache reads it. A
    check before the read would pass on the witnessed file and then re-pack
    the replacement; the check after the read sees the replacement."""
    monkeypatch.delenv("PRISMAQUANT_NVFP4_SCALE_RULE", raising=False)
    cache_dir = tmp_path / "pwc"
    cache = pickle.loads(pickle.dumps(_resident_fill(rendered, cache_dir)))
    name = f"{LAYER}.mlp.gate_proj"
    shard = cache_dir / cache_weight_filename(name, "NVFP4")
    real_load = ProductionWeightCache._load_file_tensor

    def replaced_just_before_the_read(self, value, key=None):
        if key == (name, "NVFP4"):
            staged = shard.with_name(shard.name + ".other")
            torch.save(torch.zeros((INTER, HIDDEN), dtype=torch.bfloat16), staged)
            os.replace(staged, shard)
        return real_load(self, value, key)

    _options, layer, _whole = _export_globals(rendered, cache, monkeypatch)
    monkeypatch.setattr(
        ProductionWeightCache, "_load_file_tensor", replaced_just_before_the_read)
    with pytest.raises(RuntimeError, match=UNBOUND):
        _pack_fused(name, layer[name])


def test_witnesses_follow_a_moved_cache_directory_but_not_a_copy(
    rendered, tmp_path, monkeypatch, capsys,
):
    monkeypatch.delenv("PRISMAQUANT_NVFP4_SCALE_RULE", raising=False)
    built = tmp_path / "pwc"
    pickled = pickle.dumps(_resident_fill(rendered, built))

    # Renaming the directory keeps every file object, so witnesses still bind.
    moved = tmp_path / "moved"
    built.rename(moved)
    cache = pickle.loads(pickled)
    cache.relocate(moved)
    _options, layer, _whole = _export_globals(rendered, cache, monkeypatch)
    _assert_exports_bit_exact(cache, layer, FUSED)

    # A copy has the same bytes and preserved mtimes but other files, so no
    # witness binds it: export and a resume of the copy both refuse them.
    copied = tmp_path / "copied"
    shutil.copytree(moved, copied)
    cache = pickle.loads(pickled)
    cache.relocate(copied)
    _export_globals(rendered, cache, monkeypatch)
    for name in sorted(FUSED):
        with pytest.raises(RuntimeError, match=UNBOUND):
            _pack_fused(name, layer[name])
    capsys.readouterr()
    resumed = _resident_fill(rendered, copied)
    assert "resume adopts no fused NVFP4 joint-global witness for 5" in (
        capsys.readouterr().out)
    assert nvfp4_joint_global_witnesses(resumed.metadata, where="resumed") == {}
    _export_globals(rendered, resumed, monkeypatch)
    for name in sorted(FUSED):
        with pytest.raises(RuntimeError, match=UNWITNESSED):
            _pack_fused(name, layer[name])


def _file_identities(directory: Path) -> dict[str, tuple]:
    return {
        path.name: file_stat_signature(path.lstat())
        for path in sorted(directory.iterdir())
    }


@pytest.mark.parametrize(
    "holder", ["resident", "streamed", "pre_claim_shards", "witness_sidecar"])
def test_streamed_fill_refuses_a_directory_another_rendering_holds(
    rendered, holder, tmp_path, monkeypatch,
):
    """#146 for the streamed fill: it never resumes and overwrites every shard
    in its scope, so it claims only a directory no rendering holds. The
    ``resident`` case is review N3-01's step 2."""
    monkeypatch.delenv("PRISMAQUANT_NVFP4_SCALE_RULE", raising=False)
    cache_dir = tmp_path / "pwc"
    if holder == "resident":
        _resident_fill(rendered, cache_dir)
    elif holder == "streamed":
        _streamed_fill(rendered, cache_dir)
    elif holder == "pre_claim_shards":
        # Shards with no sidecar: a streamed build from before the claim.
        cache_dir.mkdir()
        _overwrite_with_streamed(rendered, cache_dir, monkeypatch)
        assert not (cache_dir / RENDER_IDENTITY_SIDECAR_FILENAME).exists()
    else:
        cache_dir.mkdir()
        (cache_dir / NVFP4_JOINT_GLOBAL_WITNESS_SIDECAR_FILENAME).write_text("{}")
    before = _file_identities(cache_dir)

    with pytest.raises(
        ValueError, match="already holds a rendering.*fresh --cache-dir"):
        _streamed_fill(rendered, cache_dir, levers={"gptq": True})

    assert _file_identities(cache_dir) == before


def test_resident_fill_refuses_a_directory_a_streamed_fill_claimed(
    rendered, tmp_path, monkeypatch,
):
    monkeypatch.delenv("PRISMAQUANT_NVFP4_SCALE_RULE", raising=False)
    cache_dir = tmp_path / "pwc"
    _streamed_fill(rendered, cache_dir)
    before = _file_identities(cache_dir)

    with pytest.raises(
        ValueError, match="holds a streamed fill's renders.*fresh --cache-dir"):
        _resident_fill(rendered, cache_dir)

    assert _file_identities(cache_dir) == before


def test_streamed_fill_claims_an_unclaimed_directory_for_its_appends(
    rendered, tmp_path, monkeypatch,
):
    monkeypatch.delenv("PRISMAQUANT_NVFP4_SCALE_RULE", raising=False)
    cache_dir = tmp_path / "pwc"
    # fill_production_weight_cache_streaming creates its offload folder here
    # before the render starts; that is not a rendering.
    (cache_dir / "streaming_offload").mkdir(parents=True)
    _streamed_fill(rendered, cache_dir)

    sidecar = cache_dir / RENDER_IDENTITY_SIDECAR_FILENAME
    claim = json.loads(sidecar.read_text())
    assert set(claim) == {STREAMED_FILL_SIDECAR_KEY}
    section = claim[STREAMED_FILL_SIDECAR_KEY]
    assert section["schema"] == STREAMED_FILL_IDENTITY_SCHEMA
    assert section["rendered_pairs"] == sorted(
        f"{name}|NVFP4" for name in rendered["qnames"])
    assert section["levers"]["joint_scale_opt"] is True
    assert section["max_act_rows"] == 4096

    # An append of the same streamed build joins the claimed directory.
    _check_and_record_append_identity(
        cache_dir,
        MTP_APPEND_SIDECAR_KEY,
        build_mtp_append_identity(
            max_act_rows=4096,
            activation_source_hash="0" * 64,
            activation_rows={"mtp.fc": 8},
            source_prefix="mtp.",
            source_tensor_count=1,
            profile_name="test",
        ),
        progress=False,
    )
    joined = json.loads(sidecar.read_text())
    assert set(joined) == {STREAMED_FILL_SIDECAR_KEY, MTP_APPEND_SIDECAR_KEY}
    assert joined[STREAMED_FILL_SIDECAR_KEY] == section


def _streaming_export(rendered, source: Path, device: str, monkeypatch) -> dict:
    """The production streaming exporter over ``source``: real source reads,
    joint-global pre-pass, production cache, codec and witness gate. Only the
    skeleton construction is replaced, as in the export-resume tests."""
    from types import SimpleNamespace

    from transformers import AutoConfig

    import prismaquant.sensitivity_probe as sensitivity_probe
    import prismaquant.streaming_model as streaming_model

    class Skeleton:
        @staticmethod
        def _from_config(_config):
            return _Model()

    emitted: dict[str, torch.Tensor] = {}
    with monkeypatch.context() as patch:
        patch.setattr(
            AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace())
        patch.setattr(sensitivity_probe, "stage_text_only", lambda path: path)
        patch.setattr(
            streaming_model, "_skeleton_config_and_class",
            lambda config, **kw: (config, Skeleton))
        patch.setattr(streaming_model, "_init_rotary_inplace", lambda *a: None)
        enc.materialize_tensors_streaming(
            str(source), dict(rendered["assignment"]),
            profile=rendered["profile"], bf16_passthrough=set(),
            device=torch.device(device), tensor_sink=emitted.update)
    return emitted


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="the CUDA leg runs on a GPU worker")),
])
def test_streaming_exporter_ships_witnessed_fused_renders_and_refuses_a_replaced_one(
    rendered, device, tmp_path, monkeypatch,
):
    """A resident fill and the streaming exporter on one device: every fused
    entry ships its cached render exactly, under the witnessed global, and a
    shard replaced after the fill refuses on the same path."""
    from safetensors.torch import save_file

    monkeypatch.delenv("PRISMAQUANT_NVFP4_SCALE_RULE", raising=False)
    source = tmp_path / "source"
    source.mkdir()
    save_file(
        {name: tensor.contiguous()
         for name, tensor in rendered["model"].state_dict().items()},
        str(source / "model.safetensors"),
    )
    cache_dir = tmp_path / "pwc"
    pickled = pickle.dumps(fill_production_weight_cache(
        copy.deepcopy(rendered["model"]).to(device), rendered["calib_ids"],
        rendered["qnames"], formats=["NVFP4"],
        render_assignment=rendered["assignment"], levers=dict(JSO),
        max_act_rows=4096, recache_profile=rendered["profile"],
        progress=False, cache_dir=cache_dir,
    ))
    monkeypatch.setattr(enc, "_RENDER_LEVERS", NativeRenderLevers())
    monkeypatch.setattr(enc, "_INPUT_GLOBAL_SCALES", None)

    cache = pickle.loads(pickled)
    monkeypatch.setattr(enc, "_PRODUCTION_WEIGHT_CACHE", cache)
    emitted = _streaming_export(rendered, source, device, monkeypatch)
    witnesses = nvfp4_joint_global_witnesses(cache.metadata, where="test")
    for name in sorted(FUSED):
        joint_global = torch.tensor(witnesses[(name, "NVFP4")].joint_global)
        assert torch.equal(
            emitted[f"{name}.weight_global_scale"],
            (1.0 / joint_global).to(torch.float32).reshape(1))
        cached = cache.get(name, "NVFP4").cpu()
        shipped = _decode(
            {"weight_packed": emitted[f"{name}.weight_packed"],
             "weight_scale": emitted[f"{name}.weight_scale"]},
            joint_global)
        assert torch.equal(shipped.to(cached.dtype), cached), name

    _replace_by_copy(cache_dir / cache_weight_filename(f"{LAYER}.mlp.up_proj", "NVFP4"))
    monkeypatch.setattr(enc, "_PRODUCTION_WEIGHT_CACHE", pickle.loads(pickled))
    with pytest.raises(RuntimeError, match=UNBOUND):
        _streaming_export(rendered, source, device, monkeypatch)
