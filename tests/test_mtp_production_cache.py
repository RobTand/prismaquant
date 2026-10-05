from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from prismaquant.build_production_cache import (
    _filter_assignment_to_include_qnames,
    validate_render_assignment_cache_coverage,
)
from prismaquant.mtp_production_cache import (
    MTP_RENDER_METADATA_SCHEMA,
    fill_profile_mtp_production_cache,
)
from prismaquant.production_weight_cache import ProductionWeightCache


class _TinyMtp(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)
        self.proj = nn.Linear(4, 4, bias=False)


class _TinyMtpProfile:
    name = "tiny_mtp"

    def __init__(self, *, empty_source: bool = False):
        self.empty_source = bool(empty_source)
        self.source = {
            "fc.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4)
            / 32.0,
            "proj.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4)
            .flip(0)
            / 24.0,
        }

    def has_mtp(self):
        return True

    def mtp_source_prefix(self):
        return "mtp."

    def build_mtp_module(self, _text_config):
        return _TinyMtp()

    def read_mtp_source_state_dict(self, _model_path):
        if self.empty_source:
            return {}
        return {key: value.clone() for key, value in self.source.items()}

    def load_mtp_state_dict(self, module, raw):
        result = module.load_state_dict(raw, strict=False)
        return list(result.unexpected_keys), list(result.missing_keys)


def _patch_auto_config(monkeypatch):
    from transformers import AutoConfig

    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )


def _write_activation(cache_dir, qname: str, inputs: torch.Tensor):
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = qname.replace(".", "__") + ".pt"
    torch.save({"inputs": inputs, "name": qname}, cache_dir / filename)


def _empty_cache(cache_dir=None):
    return ProductionWeightCache(
        weights={
            ("model.body", "FP8_E4M3"): torch.zeros((4, 4)),
        },
        levers={
            "gptq": False,
            "scale_sweep": False,
            "static_act_order": False,
            "joint_scale_opt": False,
            "fisher_gptq": False,
        },
        activation_max_abs={},
        failed={},
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        metadata={
            "requested_entries": 1,
            "render_scores": {
                "schema": "prismaquant.production_render_scores.v1",
                "entries": 1,
                "records": {
                    "model.body|FP8_E4M3": {
                        "qname": "model.body",
                        "format": "FP8_E4M3",
                    },
                },
            },
        },
    )


def test_mtp_menu_uses_shared_renderer_and_appends_disk_cache(
    tmp_path, monkeypatch,
):
    _patch_auto_config(monkeypatch)
    activation_dir = tmp_path / "activations"
    weight_dir = tmp_path / "weights"
    rows = torch.tensor(
        [
            [1.0, -0.5, 0.25, 0.75],
            [-0.25, 0.5, 1.0, -1.0],
            [0.125, 0.25, -0.5, 0.75],
        ],
        dtype=torch.float32,
    )
    _write_activation(activation_dir, "mtp.fc", rows)
    _write_activation(activation_dir, "mtp.proj", rows.flip(0))
    cache = _empty_cache(weight_dir)

    rendered = fill_profile_mtp_production_cache(
        cache,
        "/fake/model",
        profile=_TinyMtpProfile(),
        activation_cache_dir=activation_dir,
        formats=("FP8_E4M3",),
        cache_dir=weight_dir,
        device="cpu",
        dtype=torch.float32,
        max_act_rows=2,
        progress=False,
    )

    assert rendered == 2
    assert cache.resolve_key("mtp.fc", "FP8_E4M3") is not None
    assert cache.resolve_key("mtp.proj", "FP8_E4M3") is not None
    assert all(
        isinstance(cache.weights[(qname, "FP8_E4M3")], str)
        for qname in ("mtp.fc", "mtp.proj")
    )
    assert cache.get("mtp.fc", "FP8_E4M3").shape == (4, 4)
    assert cache.metadata["requested_entries"] == 3
    mtp_meta = cache.metadata["mtp_render"]
    assert mtp_meta["schema"] == MTP_RENDER_METADATA_SCHEMA
    assert mtp_meta["entries"] == 2
    assert mtp_meta["formats"] == ["FP8_E4M3"]
    assert mtp_meta["activation_rows"] == {"mtp.fc": 3, "mtp.proj": 3}
    scores = cache.metadata["render_scores"]
    assert scores["entries"] == 3
    assert "model.body|FP8_E4M3" in scores["records"]
    assert "mtp.fc|FP8_E4M3" in scores["records"]
    assert (weight_dir / "render_scores.json").is_file()


def test_mtp_include_qnames_keeps_stripes_disjoint(tmp_path, monkeypatch):
    _patch_auto_config(monkeypatch)
    activation_dir = tmp_path / "activations"
    _write_activation(activation_dir, "mtp.fc", torch.randn(3, 4))
    cache = _empty_cache()

    rendered = fill_profile_mtp_production_cache(
        cache,
        "/fake/model",
        profile=_TinyMtpProfile(),
        activation_cache_dir=activation_dir,
        formats=("FP8_E4M3",),
        include_qnames=("model.body", "mtp.fc"),
        device="cpu",
        dtype=torch.float32,
        progress=False,
    )

    assert rendered == 1
    assert cache.resolve_key("mtp.fc", "FP8_E4M3") is not None
    assert cache.resolve_key("mtp.proj", "FP8_E4M3") is None
    assert cache.metadata["mtp_render"]["qnames"] == ["mtp.fc"]


def test_assignment_coverage_is_projected_to_include_stripe():
    assignment = {
        "model.body": "FP8_E4M3",
        "mtp.fc": "FP8_E4M3",
        "mtp.proj": "FP8_E4M3",
    }
    cache = ProductionWeightCache(
        weights={
            ("mtp.fc", "FP8_E4M3"): torch.zeros((4, 4)),
        },
        levers={},
        failed={},
    )

    stripe_assignment = _filter_assignment_to_include_qnames(
        assignment,
        ("mtp.fc",),
    )
    validate_render_assignment_cache_coverage(cache, stripe_assignment)

    assert stripe_assignment == {"mtp.fc": "FP8_E4M3"}


def test_mtp_append_replaces_prior_scope_without_double_counting(
    tmp_path, monkeypatch,
):
    _patch_auto_config(monkeypatch)
    activation_dir = tmp_path / "activations"
    _write_activation(activation_dir, "mtp.fc", torch.randn(3, 4))
    _write_activation(activation_dir, "mtp.proj", torch.randn(3, 4))
    cache = _empty_cache()

    fill_profile_mtp_production_cache(
        cache,
        "/fake/model",
        profile=_TinyMtpProfile(),
        activation_cache_dir=activation_dir,
        formats=("FP8_E4M3",),
        device="cpu",
        dtype=torch.float32,
        progress=False,
    )
    assert cache.metadata["requested_entries"] == 3

    fill_profile_mtp_production_cache(
        cache,
        "/fake/model",
        profile=_TinyMtpProfile(),
        activation_cache_dir=activation_dir,
        formats=("FP8_E4M3",),
        include_qnames=("mtp.fc",),
        device="cpu",
        dtype=torch.float32,
        progress=False,
    )

    assert cache.metadata["requested_entries"] == 2
    assert cache.resolve_key("mtp.fc", "FP8_E4M3") is not None
    assert cache.resolve_key("mtp.proj", "FP8_E4M3") is None
    assert "mtp.proj|FP8_E4M3" not in (
        cache.metadata["render_scores"]["records"]
    )


def test_body_only_include_qnames_skips_mtp_before_profile_load(
    tmp_path, monkeypatch,
):
    from transformers import AutoConfig

    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("MTP module should not be built"),
    )
    cache = _empty_cache()

    rendered = fill_profile_mtp_production_cache(
        cache,
        "/fake/model",
        profile=_TinyMtpProfile(),
        activation_cache_dir=tmp_path,
        formats=("FP8_E4M3",),
        include_qnames=("model.layers.0.mlp.down_proj",),
        device="cpu",
        dtype=torch.float32,
        progress=False,
    )

    assert rendered == 0
    assert all(not key[0].startswith("mtp.") for key in cache.weights)


def test_explicit_quantized_mtp_requires_activation_cache():
    cache = _empty_cache()

    with pytest.raises(RuntimeError, match="requires --activation-cache-dir"):
        fill_profile_mtp_production_cache(
            cache,
            "/fake/model",
            profile=_TinyMtpProfile(),
            activation_cache_dir=None,
            render_assignment={"mtp.fc": "FP8_E4M3"},
            device="cpu",
            dtype=torch.float32,
            progress=False,
        )

    assert all(not key[0].startswith("mtp.") for key in cache.weights)


def test_missing_mtp_activation_fails_before_any_render(tmp_path, monkeypatch):
    _patch_auto_config(monkeypatch)
    activation_dir = tmp_path / "activations"
    _write_activation(activation_dir, "mtp.fc", torch.randn(3, 4))
    cache = _empty_cache()

    with pytest.raises(RuntimeError, match="activation-cache coverage failure"):
        fill_profile_mtp_production_cache(
            cache,
            "/fake/model",
            profile=_TinyMtpProfile(),
            activation_cache_dir=activation_dir,
            formats=("FP8_E4M3",),
            device="cpu",
            dtype=torch.float32,
            progress=False,
        )

    assert all(not key[0].startswith("mtp.") for key in cache.weights)
    assert "mtp_render" not in cache.metadata


def test_missing_mtp_source_fails_closed(tmp_path, monkeypatch):
    _patch_auto_config(monkeypatch)
    activation_dir = tmp_path / "activations"
    _write_activation(activation_dir, "mtp.fc", torch.randn(3, 4))
    cache = _empty_cache()

    with pytest.raises(RuntimeError, match="no source tensors"):
        fill_profile_mtp_production_cache(
            cache,
            "/fake/model",
            profile=_TinyMtpProfile(empty_source=True),
            activation_cache_dir=activation_dir,
            render_assignment={"mtp.fc": "FP8_E4M3"},
            device="cpu",
            dtype=torch.float32,
            progress=False,
        )

    assert all(not key[0].startswith("mtp.") for key in cache.weights)


def test_unknown_allowlisted_mtp_qname_fails_closed(tmp_path, monkeypatch):
    _patch_auto_config(monkeypatch)
    cache = _empty_cache()

    with pytest.raises(RuntimeError, match="include-qnames/module coverage"):
        fill_profile_mtp_production_cache(
            cache,
            "/fake/model",
            profile=_TinyMtpProfile(),
            activation_cache_dir=tmp_path,
            formats=("FP8_E4M3",),
            include_qnames=("mtp.does_not_exist",),
            device="cpu",
            dtype=torch.float32,
            progress=False,
        )

    assert all(not key[0].startswith("mtp.") for key in cache.weights)


@pytest.mark.parametrize("fmt", ["NVFP4_CB_K16", "FP8_CB_K28"])
def test_retired_codebook_mtp_rung_refuses_before_any_render(
    tmp_path, monkeypatch, fmt,
):
    """A retired codebook rung (#1304) on the MTP menu refuses, naming the
    archive, before anything is rendered or appended (#1345)."""
    import traceback

    from prismaquant.format_registry import RETIRED_CODEBOOK_ARCHIVE, RetiredFormatError

    _patch_auto_config(monkeypatch)
    activation_dir = tmp_path / "activations"
    _write_activation(activation_dir, "mtp.fc", torch.randn(3, 4))
    _write_activation(activation_dir, "mtp.proj", torch.randn(3, 4))
    cache = _empty_cache()

    with pytest.raises(RetiredFormatError) as info:
        fill_profile_mtp_production_cache(
            cache,
            "/fake/model",
            profile=_TinyMtpProfile(),
            activation_cache_dir=activation_dir,
            formats=(fmt,),
            device="cpu",
            dtype=torch.float32,
            progress=False,
        )

    assert RETIRED_CODEBOOK_ARCHIVE in str(info.value)
    frames = [frame.name for frame in traceback.extract_tb(info.tb)]
    assert "_preflight_weighted_rows" in frames, frames
    assert all(not key[0].startswith("mtp.") for key in cache.weights)
    assert "mtp_render" not in cache.metadata


# ---------------------------------------------------------------------------
# #2231: the MTP append writes through the same non-injective mangled leaf
# as every other producer, so a colliding ``mtp.*`` roster must refuse at
# the append open -- before any shard is written -- naming both qnames, and
# the refusal must span the union with the manifest keys already in the
# cache when the append opens a real directory.
# ---------------------------------------------------------------------------


class _CollidingMtp(nn.Module):
    """An MTP sidecar whose roster aliases one shard leaf.

    ``a.b`` and ``a_b`` are distinct Linears (a nested container plus a
    sibling attribute -- exactly how a real synthesized sidecar nests), but
    the cache leaf mangles both to ``a_b``.
    """

    def __init__(self):
        super().__init__()
        self.a = nn.Module()
        self.a.b = nn.Linear(4, 4, bias=False)
        self.a_b = nn.Linear(4, 4, bias=False)


class _CollidingMtpProfile(_TinyMtpProfile):
    name = "colliding_mtp"

    def __init__(self):
        super().__init__()
        self.source = {
            "a.b.weight": (
                torch.arange(16, dtype=torch.float32).reshape(4, 4) / 32.0
            ),
            "a_b.weight": (
                torch.arange(16, dtype=torch.float32).flip(0).reshape(4, 4)
                / 24.0
            ),
        }

    def build_mtp_module(self, _text_config):
        return _CollidingMtp()


def test_mtp_append_refuses_a_colliding_pair_before_any_write(
    tmp_path, monkeypatch,
):
    _patch_auto_config(monkeypatch)
    activation_dir = tmp_path / "activations"
    rows = torch.randn(3, 4)
    _write_activation(activation_dir, "mtp.a.b", rows)
    _write_activation(activation_dir, "mtp.a_b", rows.flip(0))
    weight_dir = tmp_path / "weights"
    cache = _empty_cache(weight_dir)

    with pytest.raises(ValueError) as exc_info:
        fill_profile_mtp_production_cache(
            cache,
            "/fake/model",
            profile=_CollidingMtpProfile(),
            activation_cache_dir=activation_dir,
            render_assignment={"mtp.a.b": "FP8_E4M3", "mtp.a_b": "FP8_E4M3"},
            cache_dir=weight_dir,
            device="cpu",
            dtype=torch.float32,
            progress=False,
        )

    message = str(exc_info.value)
    assert "mtp.a.b" in message
    assert "mtp.a_b" in message
    assert "mtp_a_b__FP8_E4M3.pt" in message
    # The refusal precedes every write: the append directory holds no shard
    # and no sidecar, and the manifest gained no mtp key.
    assert not any(weight_dir.iterdir())
    assert all(not key[0].startswith("mtp.") for key in cache.weights)


def test_mtp_stripe_refuses_a_pair_split_across_stripes(
    tmp_path, monkeypatch,
):
    # Stripe 1 renders "mtp.a.b" into a real cache directory; stripe 2 asks
    # for "mtp.a_b", whose shard leaf is the same file. The append open must
    # refuse over the union of the new names and the manifest keys already
    # in the cache, and it must refuse BEFORE the #170 append-identity
    # sidecar runs: on the unfixed code stripe 2 reaches the sidecar and
    # refuses with the generic rebuild-the-directory mismatch, naming
    # neither colliding qname.
    _patch_auto_config(monkeypatch)
    activation_dir = tmp_path / "activations"
    _write_activation(activation_dir, "mtp.a.b", torch.randn(3, 4))
    _write_activation(activation_dir, "mtp.a_b", torch.randn(3, 4))
    weight_dir = tmp_path / "weights"
    cache = _empty_cache(weight_dir)

    rendered = fill_profile_mtp_production_cache(
        cache,
        "/fake/model",
        profile=_CollidingMtpProfile(),
        activation_cache_dir=activation_dir,
        render_assignment={"mtp.a.b": "FP8_E4M3"},
        cache_dir=weight_dir,
        device="cpu",
        dtype=torch.float32,
        progress=False,
    )
    assert rendered == 1

    with pytest.raises(ValueError) as exc_info:
        fill_profile_mtp_production_cache(
            cache,
            "/fake/model",
            profile=_CollidingMtpProfile(),
            activation_cache_dir=activation_dir,
            render_assignment={"mtp.a_b": "FP8_E4M3"},
            cache_dir=weight_dir,
            device="cpu",
            dtype=torch.float32,
            progress=False,
        )

    message = str(exc_info.value)
    assert "mtp.a.b" in message
    assert "mtp.a_b" in message
    assert "mtp_a_b__FP8_E4M3.pt" in message
    assert cache.resolve_key("mtp.a.b", "FP8_E4M3") is not None


class _CrossFormatMtp(nn.Module):
    """The alias pair again, at NVFP4's 16-aligned group size."""

    def __init__(self):
        super().__init__()
        self.a = nn.Module()
        self.a.b = nn.Linear(16, 16, bias=False)
        self.a_b = nn.Linear(16, 16, bias=False)


class _CrossFormatMtpProfile(_CollidingMtpProfile):
    name = "cross_format_mtp"

    def __init__(self):
        super().__init__()
        self.source = {
            "a.b.weight": (
                torch.arange(256, dtype=torch.float32).reshape(16, 16) / 512.0
            ),
            "a_b.weight": (
                torch.arange(256, dtype=torch.float32).flip(0).reshape(16, 16)
                / 384.0
            ),
        }

    def build_mtp_module(self, _text_config):
        return _CrossFormatMtp()


def test_mtp_append_admits_a_cross_format_alias_pair(
    tmp_path, monkeypatch,
):
    # Injectivity is filename-level over (qname, canonical format)
    # coordinates (#2219): "mtp.a.b" and "mtp.a_b" mangle to one leaf, but
    # at two formats they name two different shard files, so the append
    # proceeds and lands both.
    _patch_auto_config(monkeypatch)
    activation_dir = tmp_path / "activations"
    _write_activation(activation_dir, "mtp.a.b", torch.randn(3, 16))
    _write_activation(activation_dir, "mtp.a_b", torch.randn(3, 16))
    weight_dir = tmp_path / "weights"
    cache = _empty_cache(weight_dir)

    rendered = fill_profile_mtp_production_cache(
        cache,
        "/fake/model",
        profile=_CrossFormatMtpProfile(),
        activation_cache_dir=activation_dir,
        render_assignment={"mtp.a.b": "FP8_E4M3", "mtp.a_b": "NVFP4"},
        cache_dir=weight_dir,
        device="cpu",
        dtype=torch.float32,
        progress=False,
    )

    assert rendered == 2
    assert cache.resolve_key("mtp.a.b", "FP8_E4M3") is not None
    assert cache.resolve_key("mtp.a_b", "NVFP4") is not None
    assert (weight_dir / "mtp_a_b__FP8_E4M3.pt").is_file()
    assert (weight_dir / "mtp_a_b__NVFP4.pt").is_file()
