"""Write opens must refuse colliding coordinates before publishing any bytes."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from prismaquant import production_weight_cache as pwc
from prismaquant import tessera_campaign as campaign
from prismaquant import tessera_render
from prismaquant.weight_session import WeightSession
from test_packed_expert_cross_domain_gate import TinyLM


FORMAT = "TESSERA_E4M3_K1_R1024"
OTHER_FORMAT = "TESSERA_E4M3_K1_R1280"


@pytest.fixture
def anchor_inputs(monkeypatch, tmp_path):
    # Only encoding is substituted. Scoring, both file writes and the cache
    # manifest updates still use their real campaign entry points.
    spec = SimpleNamespace(
        bits_for_shape=lambda shape: 8 * shape[0] * shape[1],
        memory_bytes_for_shape=lambda shape: shape[0] * shape[1],
        act_dtype_name="a16",
    )

    def prepare(**kwargs):
        return dict(
            spec=spec, family=SimpleNamespace(name="TESSERA_E4M3_K1"),
            rung=int(kwargs["format_name"].rsplit("_R", 1)[1]), wire="recipe",
            activation_qdq=lambda value: value, input_scale=None,
            activation_kwargs=None, hessian_required=False,
        )

    monkeypatch.setattr(campaign, "_prepare_anchor", prepare)
    monkeypatch.setattr(campaign, "_encode_and_render",
                        lambda weight, *_args, **_kwargs: (weight * 0.75, b"wire"))
    monkeypatch.setattr(tessera_render, "encode_tessera_units",
                        lambda weights, *_args, **_kwargs:
                        [(weight * 0.75, b"wire") for weight in weights])
    wire_dir = tmp_path / "wire"
    wire_dir.mkdir()
    cache = pwc.ProductionWeightCache(
        weights={}, levers={}, cache_dir=str(tmp_path), metadata={})
    return cache, wire_dir, torch.ones(2, 2), prepare


def assert_refusal(error, first, second, filename):
    message = str(error.value)
    assert first in message
    assert second in message
    assert filename in message


def assert_no_files(tmp_path):
    assert not [path for path in tmp_path.rglob("*") if path.is_file()]


@pytest.mark.parametrize("first,second,filename", [
    ("layer.a", "layer_a", f"layer_a__{FORMAT}.pt"),
    ("layer.a", "layer__a", f"layer__a__{FORMAT}.tessera"),
])
def test_campaign_publication_checks_existing_manifest(
        anchor_inputs, tmp_path, first, second, filename):
    cache, wire_dir, weight, prepare = anchor_inputs
    cache.weights[(first, FORMAT)] = weight
    before = dict(cache.weights)

    with pytest.raises(ValueError) as error:
        campaign._finish_anchor(
            qname=second, weight=weight, activations=weight, format_name=FORMAT,
            cache=cache, wire_dir=wire_dir, prepared=prepare(format_name=FORMAT),
            render=weight * 0.75, blob=b"wire", elapsed=0.0,
        )

    assert_refusal(error, first, second, filename)
    assert cache.weights == before
    assert_no_files(tmp_path)


@pytest.mark.parametrize("first,second,filename", [
    ("layer.a", "layer_a", f"layer_a__{FORMAT}.pt"),
    ("layer.a", "layer__a", f"layer__a__{FORMAT}.tessera"),
])
def test_campaign_batch_refuses_before_first_publication(
        anchor_inputs, tmp_path, first, second, filename):
    cache, wire_dir, weight, _prepare = anchor_inputs

    with pytest.raises(ValueError) as error:
        campaign._measure_anchor_batch(
            qnames=[first, second], weights=[weight, weight],
            activations=[weight, weight], format_name=FORMAT, cache=cache,
            wire_dir=wire_dir, hessian_required=False,
        )

    assert_refusal(error, first, second, filename)
    assert cache.weights == {}
    assert_no_files(tmp_path)


def test_campaign_admits_cross_format_name_aliases(anchor_inputs, tmp_path):
    cache, wire_dir, weight, _prepare = anchor_inputs
    for name, fmt in (("layer.a", FORMAT), ("layer_a", OTHER_FORMAT)):
        campaign._measure_anchor(
            qname=name, weight=weight, activations=weight, format_name=fmt,
            cache=cache, wire_dir=wire_dir, hessian_required=False,
        )

    assert set(cache.weights) == {("layer.a", FORMAT), ("layer_a", OTHER_FORMAT)}
    assert {path.name for path in tmp_path.glob("*.pt")} == {
        f"layer_a__{FORMAT}.pt", f"layer_a__{OTHER_FORMAT}.pt",
    }
    assert {path.name for path in wire_dir.iterdir()} == {
        f"layer__a__{FORMAT}.tessera", f"layer_a__{OTHER_FORMAT}.tessera",
    }
    assert all(path.read_bytes() == b"wire" for path in wire_dir.iterdir())


@pytest.mark.parametrize("operation", ["format_weight", "initialize"])
def test_snapshot_open_refuses_before_capture_or_record(tmp_path, operation):
    model = nn.Module()
    model.layer = nn.Module()
    model.layer.a = nn.Linear(32, 32, bias=False)
    model.layer_a = nn.Linear(32, 32, bias=False)
    cache = pwc.ProductionWeightCache(
        weights={(name, "FP8_E4M3"): torch.ones(32, 32)
                 for name in ("layer.a", "layer_a")}, levers={},
    )

    with pytest.raises(ValueError) as error:
        session = WeightSession(model, production_weight_cache=cache,
                                snapshot_dir=str(tmp_path))
        if operation == "format_weight":
            session.format_weight("layer.a", "BF16")
            session.format_weight("layer_a", "BF16")
        else:
            session.initialize({"layer.a": "FP8_E4M3", "layer_a": "FP8_E4M3"}, units=[])

    assert_refusal(error, "layer.a", "layer_a", "layer_a__bf16src.pt")
    assert_no_files(tmp_path)


def test_packed_append_checks_dense_manifest_before_sidecar(tmp_path, monkeypatch):
    dense = "mlp_experts_gate_up_proj"
    packed = "mlp.experts.gate_up_proj"
    filename = pwc._cache_weight_filename(dense, "NVFP4")
    shard = tmp_path / filename
    shard.write_bytes(b"existing dense shard")
    cache = pwc.ProductionWeightCache(
        weights={(dense, "NVFP4"): filename}, levers={}, cache_dir=str(tmp_path),
    )

    def sidecar_write(*_args, **_kwargs):
        pytest.fail("packed append reached sidecar write before checking existing manifest")

    monkeypatch.setattr(pwc, "_check_and_record_append_identity", sidecar_write)
    with pytest.raises(ValueError) as error:
        pwc.fill_packed_expert_cache_entries(
            cache, TinyLM().eval(), None,
            render_assignment={packed: "NVFP4"}, cache_dir=tmp_path,
            levers={}, profile=None,
            module_acts_override={"mlp.experts": torch.ones(2, 16)}, progress=False,
        )

    assert_refusal(error, dense, packed, filename)
    assert cache.weights == {(dense, "NVFP4"): filename}
    assert shard.read_bytes() == b"existing dense shard"
    assert list(tmp_path.iterdir()) == [shard]
