"""Bit-identity pins for the CB encode fast paths.

The encoder chooses the codewords that ship in the artifact, so every
performance change to it is only legal if the emitted fields are byte-for-byte
what the slow path emits. These pin the three optimizations that exploit
structure rather than change math:

  * ``wq_period`` — a per-input-column imatrix repeats every ``in_f/VEC_DIM``
    weight vectors, so the col-weight moment ``A`` (and ``_vq_assign``'s
    ``wq @ cb_sq^T``) is built from one base block and broadcast;
  * the fused ``_vq_dist_argmin`` epilogue, which never materializes the
    (m, K) distance plane;
  * the carried-state WLS refit in the two-tier moment encoder, which drops
    the redundant re-evaluation at the merged scale.

Plus the property the cost driver's ``chunk_size`` depends on: the encode of a
tensor must not depend on what it is batched with.
"""
from __future__ import annotations

import pytest
import torch

from prismaquant import nvfp4_cb_formats as cb
from prismaquant.cb_layout import parse_format_name

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
# Production menu of the DSv4-Flash 92 GB cost run.
_FORMATS = ["NVFP4_CB_K14", "NVFP4_CB_K15", "FP8_CB_K36"]
_TIERS = ["balanced", "max"]


# The producer context the DSv4-Flash production cost run binds.
_PROD_CB_ENV = {
    "CB_CODEBOOK_SOURCE": "lattice",
    "CB_SCALE_CODING": "two_tier",
    "CB_SCALE_SWEEP": "1",
    "PRISMAQUANT_CB_LDLQ": "0",
    "PRISMAQUANT_CB_ENCODE_TIER": "balanced",
}


def _prod_cb_env(mp):
    for k, v in _PROD_CB_ENV.items():
        mp.setenv(k, v)


def _case(rows: int, in_f: int, device: str, seed: int = 7):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = (torch.randn(rows, in_f, generator=g) * 0.05).to(
        device=device, dtype=torch.bfloat16)
    # The production imatrix: ONE per-input-column vector, strictly positive.
    cw = (torch.rand(in_f, generator=g) + 0.05).to(device)
    return w, cw


def _fields(w, cw, fmt, tier, *, coding, force_dense_cw=False):
    grid, mode, k = _cb_info(fmt)
    col = cw
    if force_dense_cw:
        # Materialize the broadcast so the encoder cannot see the row
        # periodicity and must take the general path.
        col = cw.reshape(1, -1).expand(w.shape[0], -1).contiguous()
    return cb.nvfp4_cb_fields(
        w, k, grid=grid, mode=mode, col_weights=col, codebook=None,
        scale_sweep=True, scale_coding=coding, encode_tier=tier)


def _cb_info(fmt):
    fam, k = parse_format_name(fmt)
    return fam.grid, fam.mode, k


def _coding_for(fmt):
    grid, _, _ = _cb_info(fmt)
    return (cb.SCALE_CODING_TWO_TIER if grid == "fp4"
            else cb.SCALE_CODING_V1)


def _assert_same_value(av, bv, tag):
    """Compare a field value, which may be a tensor, or a tuple of tensors.

    ``codebook`` is a tuple of product subtables.  Comparing it with ``==``
    dispatches to elementwise tensor comparison and raises "Boolean value of
    Tensor with more than one value is ambiguous", so every parametrisation
    of these tests errored out before it could check anything.  The
    determinism they assert was therefore never actually exercised.
    """
    if torch.is_tensor(av):
        assert torch.is_tensor(bv), tag
        assert av.dtype == bv.dtype and av.shape == bv.shape, tag
        assert torch.equal(av, bv), f"{tag} is not bit-identical"
        return
    if isinstance(av, (tuple, list)):
        assert isinstance(bv, (tuple, list)) and len(av) == len(bv), tag
        for index, (ai, bi) in enumerate(zip(av, bv)):
            _assert_same_value(ai, bi, f"{tag}[{index}]")
        return
    assert av == bv, tag


def _assert_same_fields(a, b, tag):
    assert set(a) == set(b), tag
    for key, av in a.items():
        _assert_same_value(av, b[key], f"{tag}: field {key!r}")


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("fmt", _FORMATS)
@pytest.mark.parametrize("tier", _TIERS)
def test_periodic_col_weight_moments_are_bit_identical(device, fmt, tier):
    """Broadcast-A must emit exactly what the dense-col-weights path emits."""
    w, cw = _case(64, 512, device)
    coding = _coding_for(fmt)
    fast = _fields(w, cw, fmt, tier, coding=coding)
    dense = _fields(w, cw, fmt, tier, coding=coding, force_dense_cw=True)
    _assert_same_fields(fast, dense, f"{fmt}/{tier}/{device}")


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("fmt", _FORMATS)
def test_encode_is_chunk_size_independent(device, fmt, monkeypatch):
    """A Linear's render must not depend on the cost driver's --chunk-size.

    ``measure_batched_gpu`` groups Linears into chunks of ``--chunk-size`` and
    hands the stack to ``_batched_quantize``, which renders CB families ONE
    SLICE AT A TIME precisely so the shipped bytes cannot depend on the
    grouping. Pin that: the same Linear rendered standalone, and as a member of
    stacks of two different sizes, must come out byte-identical.

    This is load-bearing, not decorative. The encoder itself is NOT row-block
    invariant — the balanced tier calibrates ``m2_used`` from a pilot over the
    leading ``_PILOT_ROWS`` rows, so concatenating two Linears into a single
    encode call WOULD change both of their codewords. Chunk size is only free
    because the per-slice loop never does that.
    """
    from prismaquant import format_registry as fr
    from prismaquant.measure_quant_cost import _batched_quantize

    spec = fr.get_format(fmt)
    _prod_cb_env(monkeypatch)
    w, cw = _case(48, 512, device)
    stack4 = torch.stack([w, w.flip(0), w * 0.5, w + 0.01])
    cw4 = cw.reshape(1, 1, -1).expand(4, 1, -1).contiguous()

    solo = _batched_quantize(spec, stack4[:1], col_weights=cw4[:1])
    pair = _batched_quantize(spec, stack4[:2], col_weights=cw4[:2])
    quad = _batched_quantize(spec, stack4, col_weights=cw4)
    assert torch.equal(solo[0], pair[0]), f"{fmt}/{device}: N=1 vs N=2"
    assert torch.equal(solo[0], quad[0]), f"{fmt}/{device}: N=1 vs N=4"
    assert torch.equal(pair[1], quad[1]), f"{fmt}/{device}: slice 1 N=2 vs N=4"


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("fmt", _FORMATS)
def test_encoder_pilot_makes_row_blocks_coupled(device, fmt):
    """Documents WHY Linears may never be concatenated into one encode call.

    ``_calibrate_m2_used`` reads the leading ``_PILOT_ROWS`` rows, so a tensor's
    codewords depend on the whole row block it is encoded with. If a future
    change makes this invariant (a per-row-block pilot), this test should be
    retired deliberately — not silently.
    """
    w, cw = _case(96, 512, device)
    coding = _coding_for(fmt)
    head = _fields(w[:32], cw, fmt, "balanced", coding=coding)
    whole = _fields(w, cw, fmt, "balanced", coding=coding)
    m2_head = cb._calibrate_m2_used(
        w[:32].float(), cw.reshape(1, -1).expand(32, -1).contiguous(),
        _cb_info(fmt)[0], _cb_info(fmt)[1],
        cb._resolve_codebook(_cb_info(fmt)[2], _cb_info(fmt)[0],
                             _cb_info(fmt)[1], None, w.device))
    m2_whole = cb._calibrate_m2_used(
        w.float(), cw.reshape(1, -1).expand(96, -1).contiguous(),
        _cb_info(fmt)[0], _cb_info(fmt)[1],
        cb._resolve_codebook(_cb_info(fmt)[2], _cb_info(fmt)[0],
                             _cb_info(fmt)[1], None, w.device))
    assert m2_head != m2_whole, (
        "pilot no longer depends on the row block — retire this test and "
        "revisit test_encode_is_chunk_size_independent's rationale")
    assert head["indices"].shape[0] == 32 and whole["indices"].shape[0] == 96


@pytest.mark.parametrize("device", _DEVICES)
def test_vq_dist_argmin_fused_matches_eager(device):
    """The fused distance+argmin keeps values AND the first-min tie rule."""
    g = torch.Generator(device="cpu").manual_seed(11)
    term1 = torch.randn(2048, 96, generator=g).to(device)
    term2 = torch.randn(2048, 96, generator=g).to(device)
    ref = cb._vq_dist_argmin_eager(term2, term1)
    assert torch.equal(cb._vq_dist_argmin(term2, term1), ref)
    # Exact ties: an all-equal distance row must take the FIRST index, which
    # is the rule the two-tier zero/degenerate group rule depends on.
    flat1 = torch.zeros(8, 16, device=device)
    flat2 = torch.zeros(8, 16, device=device)
    assert torch.equal(cb._vq_dist_argmin(flat2, flat1),
                       torch.zeros(8, dtype=torch.long, device=device))
    # Broadcast (periodic) form == dense form.
    P, r, K = 64, 4, 96
    t1 = torch.randn(r * P, K, generator=g).to(device)
    t2b = torch.randn(P, K, generator=g).to(device)
    dense = cb._vq_dist_argmin(t2b.repeat(r, 1), t1)
    bcast = cb._vq_dist_argmin(t2b.reshape(1, P, K),
                               t1.reshape(r, P, K)).reshape(-1)
    assert torch.equal(dense, bcast)


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("periodic", [False, True])
def test_compiled_score_indices_are_true_argmins(device, periodic):
    """Check the serialized INDEX, not the minimum value returned beside it.

    A broken min reduction can return a correct-looking value paired with the
    wrong index.  Both index-producing moment reducers are checked directly
    against the score plane they claim to minimize, including the compact
    periodic-A layout used by production imatrix weights.
    """
    g = torch.Generator(device="cpu").manual_seed(29)
    m, p, k, scales = 48, 12, 37, 5
    B = torch.randn(m, k, generator=g).to(device)
    A_dense = torch.rand(m, k, generator=g).to(device) + 0.1
    A = A_dense[:p].clone() if periodic else A_dense
    if periodic:
        A_dense = A.repeat(m // p, 1)
    s = (torch.rand(m, scales, generator=g) + 0.05).to(device)

    direct = ((s * s).unsqueeze(1) * A_dense.unsqueeze(-1)
              - (2.0 * s).unsqueeze(1) * B.unsqueeze(-1))
    true_batched = direct.argmin(dim=1)
    _, got_batched = cb._score_minargmin_batched(A, B, s)
    assert torch.equal(got_batched, true_batched)

    true_single = direct[:, :, 0].argmin(dim=-1)
    _, got_single = cb._score_argmin(A, B, s[:, :1])
    assert torch.equal(got_single, true_single)


def test_index_compilers_are_never_called_on_cpu(monkeypatch):
    """Contain the known torch.compile CPU wrong-index lowering."""
    monkeypatch.setattr(cb, "_encode_compile_on", lambda: True)

    def forbidden():
        raise AssertionError("index-producing compiled route reached on CPU")

    monkeypatch.setattr(cb, "_vq_dist_argmin_compiled", forbidden)
    monkeypatch.setattr(cb, "_score_argmin_compiled", forbidden)
    monkeypatch.setattr(cb, "_score_minargmin_batched_compiled", forbidden)

    term1 = torch.randn(8, 11)
    term2 = torch.randn(8, 11)
    assert torch.equal(cb._vq_dist_argmin(term2, term1),
                       cb._vq_dist_argmin_eager(term2, term1))

    A = torch.rand(8, 11)
    B = torch.randn(8, 11)
    s = torch.rand(8, 3)
    _, single = cb._score_argmin(A, B, s[:, :1])
    _, single_ref = cb._score_argmin_eager(A, B, s[:, :1])
    assert torch.equal(single, single_ref)
    _, batched = cb._score_minargmin_batched(A, B, s)
    direct = ((s * s).unsqueeze(1) * A.unsqueeze(-1)
              - (2.0 * s).unsqueeze(1) * B.unsqueeze(-1))
    assert torch.equal(batched, direct.argmin(dim=1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_compiled_score_indices_survive_interleaved_k_specializations():
    """A mixed-rung campaign must not reuse a wrong K specialization."""
    g = torch.Generator(device="cpu").manual_seed(41)
    cases = []
    for k in (128, 256, 512, 128):
        A = (torch.rand(64, k, generator=g) + 0.1).cuda()
        B = torch.randn(64, k, generator=g).cuda()
        s = (torch.rand(64, 4, generator=g) + 0.05).cuda()
        cases.append((A, B, s))
    for A, B, s in cases:
        direct = ((s * s).unsqueeze(1) * A.unsqueeze(-1)
                  - (2.0 * s).unsqueeze(1) * B.unsqueeze(-1))
        _, got = cb._score_minargmin_batched(A, B, s)
        assert torch.equal(got, direct.argmin(dim=1))


def test_k28_product_uses_measured_score_chunk_cap():
    tables = tuple(torch.empty(128, 2) for _ in range(4))
    assert cb._moment_rows_step(tables, 512) == 512
    mixed = (torch.empty(512, 2),) + tables[1:]
    assert cb._moment_rows_step(mixed, 512) == 256


def test_matching_pilot_and_first_chunk_reuse_identical_moments(
        monkeypatch):
    """The calibration pilot must not rebuild an identical first chunk."""
    monkeypatch.setenv("PRISMAQUANT_CB_ENCODE_COMPILE", "0")
    g = torch.Generator(device="cpu").manual_seed(53)
    rows, in_f = 8, 256
    w = torch.randn(rows, in_f, generator=g)
    cw = torch.rand(in_f, generator=g) + 0.1
    wq = cw.reshape(1, -1).expand_as(w).contiguous()
    tables = tuple(torch.randn(2, 2, generator=g) for _ in range(4))
    period = in_f // cb.VEC_DIM
    wvec = w.reshape(-1, cb.VEC_DIM)
    wqvec = wq.reshape(-1, cb.VEC_DIM)
    moments = cb._chunk_moments(
        wvec, wqvec, "product", tables, period)

    cold = cb._calibrate_m2_used(
        w, wq, "fp8", "product", tables, period)
    reused = cb._calibrate_m2_used(
        w, wq, "fp8", "product", tables, period, moments)
    assert cold == reused

    original = cb._chunk_moments
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cb, "_chunk_moments", counted)
    cb._sweep_encode_moment(
        w, "fp8", "product", tables, wq, "fast", period)
    assert calls == 1


@pytest.mark.parametrize("device", _DEVICES)
def test_periodic_split_only_fires_on_true_periodicity(device):
    """The compact-A shortcut must never claim a period it does not have."""
    B = torch.zeros(64, 8, device=device)
    assert cb._periodic_split(torch.zeros(8, device=device), B) == (None, None)
    assert cb._periodic_split(torch.zeros(64, 8, device=device), B) == (
        None, None)
    assert cb._periodic_split(torch.zeros(16, 8, device=device), B) == (16, 4)
    assert cb._periodic_split(torch.zeros(24, 8, device=device), B) == (
        None, None)


@pytest.mark.parametrize("device", _DEVICES)
def test_two_tier_legal_e_range_matches_uncached(device):
    _, _, legal = cb._two_tier_tables(device)
    any_legal = legal.any(dim=-1)
    nz = torch.nonzero(any_legal)
    assert cb._two_tier_legal_e_range(device) == (int(nz[0]), int(nz[-1]))


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("fmt", _FORMATS)
def test_encode_is_run_to_run_deterministic(device, fmt):
    w, cw = _case(64, 512, device, seed=3)
    coding = _coding_for(fmt)
    a = _fields(w, cw, fmt, "balanced", coding=coding)
    b = _fields(w, cw, fmt, "balanced", coding=coding)
    _assert_same_fields(a, b, f"{fmt}/{device}/repeat")
