"""PQ #1394 (#1303 part 1): the five numeric near-duplicate pairs keep their bits.

Each pair below computed one quantity with two copies of one recipe. Each site
now binds its owner. The outcomes of the pre-move code on every input here are
frozen in ``fixtures/numerics_pairs_1394.json`` (``tests/golden_table.py``),
and every call goes through the old site's name.

A tensor outcome is its dtype, shape, device type and the SHA-256 of its raw
bytes, so "the same" means bit-identical, not close. Sites that run on the GPU
in production (the probe's marginals and the MX scale rounding) are also
checked on CUDA in the dtypes the probe and the render feed them.

The CPU rows introduced by #1414 use exact integer-derived inputs and were
recorded from the pre-consolidation source at ``d6a490cd756``. The one
exception to raw-byte identity is the all-NaN imatrix result of an empty
activation file: CPU reduction kernels vary its NaN payload bits. The CUDA
inputs and their historical golden rows remain unchanged.
"""
from __future__ import annotations

import hashlib
import importlib
import math

import pytest
import torch

from tests.golden_table import GoldenTable

GOLDEN = GoldenTable("numerics_pairs_1394")
# CPU rows use inputs made from integer arithmetic.  The original table remains
# intact as the pre-#1414 record (and still checks the CUDA cases).
CPU_GOLDEN = GoldenTable("numerics_pairs_1414_cpu")

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


def _check(call, *, tmp=None, table=GOLDEN, bits=_bits):
    return table.call(lambda: bits(call()), tmp=tmp)


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
    # The old CPU torch.randn fixture produced different bytes on ARM and x86.
    # Build finite positive fp32 bit patterns directly, covering each exponent
    # and both sides of the mantissa rounding thresholds.  Specials below
    # independently cover infinities, NaN, signed zero and negatives.
    if device == "cpu":
        mantissas = (0, 1, (1 << 21) - 1, 1 << 21, (1 << 21) + 1,
                     (1 << 22) - 1, 1 << 22, (1 << 22) + 1,
                     (1 << 23) - 1)
        bits = torch.tensor(
            [((i % 255) << 23) | mantissas[(i // 255) % len(mantissas)]
             for i in range(1 << 16)], dtype=torch.int32)
    else:
        # Retain the historical GPU fixture and its frozen CUDA outcomes.
        bits = torch.randint(0, 2 ** 31 - 1, (1 << 16,), generator=generator,
                             dtype=torch.int64).to(torch.int32)
    sampled = bits.view(torch.float32)
    specials = torch.tensor(
        [0.0, -0.0, 1.0, 1.5, 1.75, 2.0, 3.0, 6.0, 448.0, 57344.0,
         torch.finfo(torch.float32).tiny, torch.finfo(torch.float32).tiny / 2,
         1e-45, torch.finfo(torch.float32).max, math.inf, -math.inf, math.nan,
         -1.0, -6.0], dtype=torch.float32)
    # Real block amaxes: |w| maxima over 32-wide blocks of a scaled weight.
    if device == "cpu":
        # Every operand is exactly representable; amax sees varied blocks
        # without depending on a platform's normal-distribution kernel.
        index = torch.arange(256 * 512, dtype=torch.int32)
        weight = (((index * 73) % 4093) - 2046).to(torch.float32).reshape(256, 512) / 131072
    else:
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
    table = CPU_GOLDEN if device == "cpu" else GOLDEN
    for amax in _mx_inputs(device):
        _check(lambda: fn(amax), table=table)


# ---------------------------------------------------------------------------
# 3. The GGUF imatrix from the activation cache.
# ---------------------------------------------------------------------------
from prismaquant.moe_imatrix import build_imatrix_from_act_cache


@pytest.fixture(scope="module")
def act_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("act")

    def rows(t, n, dtype):
        # Rotate a 0/1/2 pattern per column. Each magnitude-2 row contributes
        # four to the squared sum; t % 4 rows of magnitude 1 complete it.
        # The squared sum is exactly t, even when t is 5, 17 or 33.
        columns = torch.arange(n, dtype=torch.int32)
        if t == 0:
            return torch.empty((0, n), dtype=dtype)
        positions = (torch.arange(t, dtype=torch.int32)[:, None]
                     + columns[None, :]) % t
        magnitudes = 2 * (positions < t // 4).to(torch.int32)
        magnitudes += ((positions >= t // 4) &
                       (positions < t // 4 + t % 4)).to(torch.int32)
        scales = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0])[columns % 5]
        row_signs = 1 - 2 * ((torch.arange(t, dtype=torch.int32)[:, None]
                              + columns[None, :]) % 2)
        return (row_signs.to(torch.float32) * magnitudes * scales).to(dtype)

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


def test_imatrix_fixture_requires_averaging(act_dir):
    """A first-row shortcut or dropped first row changes the intended output."""
    for stem in ("model__layers__0__self_attn__q_proj",
                 "model__layers__0__mlp__experts",
                 "model__layers__1__mlp__down_proj", "zz_duplicate"):
        inputs = torch.load(act_dir / f"{stem}.pt", weights_only=False)["inputs"].float()
        squared = inputs.square()
        whole = squared.mean(dim=0)
        assert not torch.equal(whole, squared[0])
        assert not torch.equal(whole, squared[1:].mean(dim=0))


def _imatrix_bits(value):
    # mean(empty) yields NaNs whose payload bits vary by CPU reduction kernel.
    # Freeze the empty-file result as all-NaN, including its dtype and shape;
    # every nonempty imatrix value still has an exact raw-byte digest.
    result = _bits(value)
    empty = value.get("g")
    if isinstance(empty, torch.Tensor) and torch.isnan(empty).all():
        result["g"] = ("tensor", str(empty.dtype), tuple(empty.shape),
                       empty.device.type, "all NaN")
    return result


def test_imatrix_keeps_its_bits(act_dir, tmp_path):
    # Each nonempty fixture column has squared sum t * scale**2 over t rows.
    # Assert that independent closed-form result, not a copy of the reducer.
    squared_scales = torch.tensor([1/16, 1/4, 1., 4., 16.])
    expected = {name: squared_scales[torch.arange(width) % 5]
                for name, width in (("model.layers.0.mlp.experts", 48),
                                    ("model.layers.0.self_attn.q_proj", 96),
                                    ("model.layers.1.mlp.down_proj", 40),
                                    ("model.layers.2.mlp.up_proj", 8))}
    expected['g'] = torch.full((8,), float('nan'))
    for path in (act_dir, str(act_dir)):
        actual = build_imatrix_from_act_cache(path)
        assert _imatrix_bits(actual) == _imatrix_bits(expected)
    assert build_imatrix_from_act_cache(tmp_path / 'missing') == {}


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
        if device == "cpu":
            gy_index = torch.arange(tokens * out_features, dtype=torch.int32)
            x_index = torch.arange(tokens * in_features, dtype=torch.int32)
            gy2 = ((((gy_index * 37) % 17) - 8).to(torch.float32)
                   / 8192).reshape(tokens, out_features).to(dtype)
            x2 = ((((x_index * 53) % 47) - 23).to(torch.float32)
                  / 8).reshape(tokens, in_features).to(dtype)
        else:
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
        if device == "cpu":
            h_index = torch.arange(out_features * in_features, dtype=torch.int32)
            chunk_h = (((h_index * 19) % 29).to(torch.float32)
                       / 1048576).reshape(out_features, in_features)
        else:
            chunk_h = torch.rand(out_features, in_features, generator=generator) * 1e-4
        chunk_h = chunk_h.to(device)
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
    table = CPU_GOLDEN if device == "cpu" else GOLDEN
    for gy2_sq, x2_sq, x2, chunk_h in _marginal_inputs(device):
        _check(lambda: fn(gy2_sq, x2_sq, x2, chunk_h), table=table)


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
