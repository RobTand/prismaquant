"""PQ #1394 (#1303 part 1): the five numeric near-duplicate pairs keep their bits.

Each pair below computed one quantity with two copies of one recipe. Each site
now binds its owner. The outcomes of the pre-move code on every input here are
frozen in ``fixtures/numerics_pairs_1394.json`` (``tests/golden_table.py``),
and every call goes through the old site's name.

A tensor outcome is its dtype, shape, device type and the SHA-256 of its raw
bytes, so "the same" means bit-identical, not close. Sites that run on the GPU
in production (the probe's marginals and the MX scale rounding) are also
checked on CUDA in the dtypes the probe and the render feed them.
"""
from __future__ import annotations

import hashlib
import importlib
import math

import pytest
import torch

from tests.golden_table import GoldenTable

GOLDEN = GoldenTable("numerics_pairs_1394")

_CUDA = pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA"))
DEVICES = ["cpu", _CUDA]


@pytest.fixture(autouse=True)
def _one_thread():
    # A CPU reduction's partial sums can depend on the intra-op thread count;
    # pin it so a row recorded on one worker replays on another.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


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


def _bits(value):
    """A tensor's exact identity; containers are walked in order."""
    if isinstance(value, torch.Tensor):
        data = value.detach().to("cpu").contiguous()
        raw = data.reshape(-1).view(torch.uint8)
        return ("tensor", str(data.dtype), tuple(data.shape), value.device.type,
                hashlib.sha256(raw.numpy().tobytes()).hexdigest())
    if isinstance(value, dict):
        return {key: _bits(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_bits(item) for item in value)
    return value


def _check(call, *, tmp=None):
    return GOLDEN.call(lambda: _bits(call()), tmp=tmp)


# ---------------------------------------------------------------------------
# 1. The kneedle's log-distortion axis.
# ---------------------------------------------------------------------------
LOG_ERROR_SITES = [
    "prismaquant.allocator._log_error_values",
    "prismaquant.select_validated_frontier._log_error_values",
]
LOG_ERROR_INPUTS = [
    [],
    [1.0],
    [1.0, 0.1, 0.0],
    [0.5, -0.25, 0.0, 1e-300, 5e-324],
    [math.inf, 1.0, math.nan, -math.inf, 2.0],
    [math.nan, math.inf, -1.0],
    [3, 2, 1],
    (0.0301, 0.0151, 0.0103, 0.0087),
    [1e308, 1e-308, 7.0],
    ["0.5", 0.25],
    ["x", 1.0],
    [None],
]


@pytest.mark.parametrize("site", LOG_ERROR_SITES)
def test_log_error_values_keep_their_floats(site):
    fn = _site(site)
    for values in LOG_ERROR_INPUTS:
        _check(lambda: fn(values))


# ---------------------------------------------------------------------------
# 2. The MX E8M0 scale's power-of-two rounding.
# ---------------------------------------------------------------------------
MX_SITES = [
    "prismaquant.export_native_compressed._mx_rounded_amax_power2",
    "prismaquant.format_registry._mx_rounded_amax_power2",
]


def _mx_inputs(device):
    generator = torch.Generator().manual_seed(1394)
    # Every exponent, with mantissas on both sides of the rounding bit.
    bits = torch.randint(0, 2 ** 31 - 1, (1 << 16,), generator=generator, dtype=torch.int64)
    sampled = bits.to(torch.int32).view(torch.float32)
    specials = torch.tensor(
        [0.0, -0.0, 1.0, 1.5, 1.75, 2.0, 3.0, 6.0, 448.0, 57344.0,
         torch.finfo(torch.float32).tiny, torch.finfo(torch.float32).tiny / 2,
         1e-45, torch.finfo(torch.float32).max, math.inf, -math.inf, math.nan,
         -1.0, -6.0], dtype=torch.float32)
    # Real block amaxes: |w| maxima over 32-wide blocks of a scaled weight.
    weight = torch.randn(256, 512, generator=generator) * 0.02
    amax = weight.abs().reshape(256, -1, 32).amax(dim=-1)
    return [
        sampled.to(device),
        specials.to(device),
        amax.to(device),
        amax.to(torch.bfloat16).to(device),
        amax.to(torch.float16).to(device),
        amax.unsqueeze(-1).to(device),
        torch.zeros(0, dtype=torch.float32, device=device),
        torch.tensor(0.75, device=device),
    ]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("site", MX_SITES)
def test_mx_power2_rounding_keeps_its_bits(site, device):
    fn = _site(site)
    for amax in _mx_inputs(device):
        _check(lambda: fn(amax))


# ---------------------------------------------------------------------------
# 3. The GGUF imatrix from the activation cache.
# ---------------------------------------------------------------------------
IMATRIX_SITES = [
    "prismaquant.export_gguf.build_imatrix_from_act_cache",
    "prismaquant.export_gguf_direct.build_direct_imatrix",
]


@pytest.fixture(scope="module")
def act_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("act")
    generator = torch.Generator().manual_seed(1303)

    def rows(t, n, dtype):
        return (torch.randn(t, n, generator=generator) * 3).to(dtype)

    blobs = {
        # The probe's own schema, in the dtypes it caches.
        "model__layers__0__self_attn__q_proj": {
            "name": "model.layers.0.self_attn.q_proj", "inputs": rows(64, 96, torch.bfloat16),
            "row_indices": torch.arange(64)},
        "model__layers__0__mlp__experts": {
            "name": "model.layers.0.mlp.experts", "inputs": rows(33, 48, torch.float16)},
        "model__layers__1__mlp__down_proj": {"inputs": rows(17, 40, torch.float32)},
        "model__layers__2__mlp__up_proj": {"name": "", "inputs": rows(1, 8, torch.bfloat16)},
        "model__layers__3__mlp__gate_proj": {"name": "g", "inputs": rows(0, 8, torch.float32)},
        # Skipped: not two-dimensional, no inputs, not a dict.
        "model__layers__4__mlp__up_proj": {"name": "three_d", "inputs": rows(4, 8, torch.float32).reshape(2, 2, 8)},
        "model__layers__5__mlp__up_proj": {"name": "none"},
        "model__layers__6__mlp__up_proj": [rows(2, 2, torch.float32)],
        # Two files naming one module: the later file wins.
        "zz_duplicate": {"name": "model.layers.0.self_attn.q_proj", "inputs": rows(5, 96, torch.float32)},
    }
    for stem, blob in blobs.items():
        torch.save(blob, root / f"{stem}.pt")
    (root / "ignored.txt").write_text("not an activation")
    return root


@pytest.mark.parametrize("site", IMATRIX_SITES)
def test_imatrix_keeps_its_bits(site, act_dir, tmp_path):
    fn = _site(site)
    _check(lambda: fn(act_dir), tmp=act_dir)
    _check(lambda: fn(str(act_dir)), tmp=act_dir)
    _check(lambda: fn(tmp_path / "missing"), tmp=tmp_path)


# ---------------------------------------------------------------------------
# 4. The probe's per-channel marginals.
# ---------------------------------------------------------------------------
MARGINAL_SITES = [
    "prismaquant.incremental_probe._marginal_chunk",
    "prismaquant.sensitivity_probe.FisherAccumulator._dense_marginal_chunk",
]


def _marginal_inputs(device):
    generator = torch.Generator().manual_seed(1295)
    cases = []
    for tokens, out_features, in_features, dtype in (
            (128, 96, 64, torch.bfloat16),
            (1, 16, 24, torch.bfloat16),
            (0, 16, 24, torch.bfloat16),
            (37, 40, 56, torch.float16),
            (19, 24, 32, torch.float32),
            (512, 256, 384, torch.bfloat16)):
        gy2 = (torch.randn(tokens, out_features, generator=generator) * 1e-3).to(dtype)
        x2 = (torch.randn(tokens, in_features, generator=generator) * 4).to(dtype)
        if tokens > 2:
            x2[1, 0] = torch.finfo(dtype).max
            x2[2, 1] = -torch.finfo(dtype).max
        gy2, x2 = gy2.to(device), x2.to(device)
        gy2_sq = gy2.float().pow(2).to(dtype)
        x2_sq = x2.float().pow(2)
        # Any (out, in) fp32 matrix: the sites only reduce it. Drawn rather
        # than multiplied so the input itself has no kernel-choice noise.
        chunk_h = (torch.rand(out_features, in_features, generator=generator) * 1e-4).to(device)
        if tokens > 2:
            chunk_h[0, 0] = math.inf
        cases.append((gy2_sq, x2_sq, x2, chunk_h))
    nan_x = torch.tensor([[1.0, math.nan], [-2.0, 3.0]], device=device)
    cases.append((torch.ones(2, 3, device=device), nan_x.pow(2), nan_x, torch.ones(3, 2, device=device)))
    return cases


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("site", MARGINAL_SITES)
def test_probe_marginals_keep_their_bits(site, device):
    fn = _site(site)
    for gy2_sq, x2_sq, x2, chunk_h in _marginal_inputs(device):
        _check(lambda: fn(gy2_sq, x2_sq, x2, chunk_h))


# ---------------------------------------------------------------------------
# 5. The layer-shard regexes.
# ---------------------------------------------------------------------------
REGEX_CASES = [
    (0, 1), (1, 1), (5, 2), (8, 4), (7, 3), (48, 5), (3, 10), (4, 0), (4, -1), (-3, 2),
]
PREFIXES = ["model.layers", "model.language_model.layers", "mtp.layers", "model.visual.blocks",
            "a+b(c)[d]"]


def test_incremental_probe_shard_regexes_keep_their_text():
    fn = _site("prismaquant.incremental_probe.build_layer_shard_regexes")
    for layers, per_shard in REGEX_CASES:
        _check(lambda: fn(layers, per_shard))
        for prefix in PREFIXES:
            _check(lambda: fn(layers, per_shard, prefix))
            _check(lambda: fn(layers, per_shard, layer_prefix=prefix))


def test_profile_shard_regexes_keep_their_text():
    fn = _site("prismaquant.model_profiles.base._build_layer_shard_regexes")
    for layers, per_shard in REGEX_CASES:
        for prefix in PREFIXES:
            _check(lambda: fn(layers, per_shard, layer_prefix=prefix))
    _check(lambda: fn(5, 2))
