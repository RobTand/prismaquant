"""PQ #1384: tensor digest sites keep their bytes under one owner.

Each site below hashed a tensor's raw host bytes with its own copy of one
recipe. Each now binds or calls ``prismaquant.tensor_digests``. The outcomes
of the pre-move code on every input here (the returned digest or identity,
key order included, or the exception type, ``str()`` and chained cause) are
frozen in ``fixtures/tensor_digests_1384.json`` (``tests/golden_table.py``),
and every call goes through the old site's name.

Inputs cover every dtype the sites stamp (float32, bf16, fp16, fp8, int64,
int32, uint8, bool), empty, zero-dimensional and non-contiguous tensors, a
tensor that requires grad, CUDA tensors, and values that are not tensors.
"""
from __future__ import annotations

import hashlib
import importlib

import pytest
import torch

from tests.golden_table import GoldenTable

GOLDEN = GoldenTable("tensor_digests_1384")

_CUDA = pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"))
DEVICES = ["cpu", _CUDA]


def _site(ref):
    """Resolve ``package.module.name[.attr]`` through the old site's path."""
    parts = ref.split(".")
    for split in range(len(parts) - 1, 0, -1):
        try:
            owner = importlib.import_module(".".join(parts[:split]))
        except ModuleNotFoundError:
            continue
        for part in parts[split:]:
            owner = getattr(owner, part)
        return owner
    raise ModuleNotFoundError(ref)


def _tensors(device):
    generator = torch.Generator().manual_seed(1384)
    base = torch.randn(6, 10, generator=generator)
    values = [
        base,
        base.to(torch.bfloat16),
        base.to(torch.float16),
        base.to(torch.float8_e4m3fn),
        (base * 1000).to(torch.int64),
        (base * 1000).to(torch.int32),
        (base.abs() * 50).to(torch.uint8),
        base > 0,
        base.t(),
        base[:, ::3],
        base[2:4],
        torch.zeros(0, 7),
        torch.zeros(3, 0, dtype=torch.bfloat16),
        torch.tensor(1.5),
        torch.tensor(7, dtype=torch.uint8),
        torch.arange(12, dtype=torch.int64).reshape(3, 4),
    ]
    out = [value.to(device) for value in values]
    grad = base.clone().to(device).requires_grad_(True)
    out.append(grad * 2)
    return out


NOT_TENSORS = [None, [1.0, 2.0], 3, "tensor"]


# ---------------------------------------------------------------------------
# Identity dicts: ``{dtype, shape, sha256}`` and ``{shape, dtype, sha256}``.
# ---------------------------------------------------------------------------
IDENTITY_SITES = [
    "prismaquant.sample_parallel_probe_merge._tensor_identity",
    "prismaquant.expert_empirical_cost._tensor_value_stamp",
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("site", IDENTITY_SITES)
def test_identity_sites_keep_their_bytes_and_key_order(site, device):
    fn = _site(site)
    for value in _tensors(device):
        GOLDEN.call(lambda: fn(value))


@pytest.mark.parametrize("site", IDENTITY_SITES)
def test_identity_sites_keep_their_refusals(site):
    fn = _site(site)
    for value in NOT_TENSORS:
        GOLDEN.call(lambda: fn(value))


# ---------------------------------------------------------------------------
# The calibration hash: shape text, dtype text, then bytes.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", DEVICES)
def test_tensor_hash_update_keeps_its_stream(device):
    fn = _site("prismaquant.perturbed_x_cache._tensor_hash_update")
    for value in _tensors(device):
        for make in (hashlib.sha256, lambda: hashlib.blake2b(digest_size=16)):
            def run():
                digest = make()
                fn(digest, value)
                return digest.hexdigest()
            GOLDEN.call(run)


@pytest.mark.parametrize("device", DEVICES)
def test_calibration_data_hash_keeps_its_value(device):
    fn = _site("prismaquant.perturbed_x_cache.calibration_data_hash")
    tensors = _tensors(device)
    GOLDEN.call(lambda: fn(tensors[0]))
    GOLDEN.call(lambda: fn(tensors[1]))
    GOLDEN.call(lambda: fn({"input_ids": tensors[4], "mask": tensors[7]}))
    GOLDEN.call(lambda: fn([tensors[4], tensors[5]]))


# ---------------------------------------------------------------------------
# Tessera's capture digest: a per-unit header, NUL, then bytes.
# ---------------------------------------------------------------------------
PROVENANCE = {"text_sha256": "a" * 64, "fit_ids_sha256": "b" * 64,
              "fit_tokens": 4096, "hessian_role": "fit"}


@pytest.mark.parametrize("device", DEVICES)
def test_hessian_capture_digest_keeps_its_value(device):
    fn = _site("prismaquant.tessera_export_lane.hessian_capture_sha256")
    tensors = _tensors(device)
    GOLDEN.call(lambda: fn({"model.layers.0.mlp.up_proj": tensors[0]}, PROVENANCE))
    GOLDEN.call(lambda: fn({
        "model.layers.1.self_attn.o_proj": tensors[1],
        "model.layers.0.mlp.up_proj": tensors[8],
        "model.layers.0.mlp.down_proj": tensors[11],
    }, PROVENANCE))
    GOLDEN.call(lambda: fn({"model.layers.0.mlp.up_proj": tensors[13]}, PROVENANCE))
    GOLDEN.call(lambda: fn({}, {}))
