"""Triton kernels for GLM-5.3's chunked KDA delta rule (PQ #1199).

This module computes the function that the GLM derivative image's pinned
Torch fallback ``chunk_kimi_delta_attention`` computes: the Kimi Delta
Attention chunked delta rule with the ``glm_kda_causal_exp_v1`` correction,
which zeroes the strictly upper decay exponents before ``exp``. It rounds
differently. It serves the Stage B spill capture's target-layer passes
(``glm_kda_capture_kernel``); nothing else dispatches to it.

The fallback is limited by memory bandwidth. It materializes FP32
``[B, H, N, 64, 64, 128]`` tensors for the intra-chunk decay products, 4.3 GB
each at capture batch 4, and autograd zero-fills and adds one of them for
every chunk. Here those products are three Triton kernels that recompute
``exp(G_i - G_j)`` per element instead of storing it:

* ``_gram_fwd_kernel``: ``A[i, j] = sum_d a[i, d] b[j, d] exp(G[i, d] - G[j, d])``
  over ``j < i`` (strict) or ``j <= i`` (inclusive), with every other entry 0.
* ``_gram_bwd_rows_kernel`` and ``_gram_bwd_cols_kernel``: the gradients of
  ``a``, ``b`` and ``G`` from ``dA``, reading only the kept entries of ``dA``.

The rest is Torch in FP64, forward and backward: the L2 norms, the beta
products, the in-chunk cumulative gate sums ``G``, the unit lower-triangular
solve, the decay factors and the eight-chunk recurrence.

Inputs and outputs have the fallback's dtypes, and the fallback computes in
FP32 throughout. Every stage here is at least as precise as the fallback's
stage:

* The Torch stages are FP64. Each input gradient rounds once, to its input's
  dtype, and the output rounds once, to the caller's dtype. The fallback
  rounds every intermediate to FP32; in particular ``G_i - G_j`` inherits the
  rounding of two FP32 sums that reach about 320 under GLM's gate bound of -5.
* The Gram kernels read FP32 operands, each one rounding of its FP64 value,
  and read ``G`` as an FP32 pair (``hi``, ``lo``), so each ``G_i - G_j`` is
  within one FP32 rounding of the exact difference.
* Each Gram entry and each Gram gradient forms its products in FP32, sums
  them in FP32 over one 16-wide block, and accumulates the blocks in FP64,
  rounding to FP32 once when it is stored. The fallback sums all 128 (or 64)
  products in FP32. The ``G`` gradient leaves the Gram kernels in FP64.
* The solve ``(I + A)^-1 [v_beta | k_beta exp(G)]`` is one FP64 triangular
  solve. The fallback builds the inverse in FP32 over 63 dependent steps and
  then multiplies.
* A kept diagonal term enters the ``G`` gradient twice with opposite signs
  and cancels exactly. The kernels leave it out of both sums instead of
  cancelling it in rounded arithmetic.

The first version kept the L2 norms, the beta products and the recurrence in
FP32, as the fallback computes them. Where those shared stages dominated the
error (weak or no decay, nearly parallel keys), the two implementations had
errors of one size, and at the production shape one gradient's largest
element error came out above the fallback's by more than the final roundings
explain (PQ #1199). Those stages are FP64 now, so no stage rounds more than
the fallback's does.

The Triton kernels use no atomics and no ``tl.dot``, so they are
deterministic, and no product is TF32.
``experiments/kda_kernel_numerics.py`` compares both implementations with a
float64 run of the executed fallback source.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import struct

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

NAME = "kda_gram_v1"
#: The derivative semantics this kernel computes (``glm_source_derivative``).
IMPLEMENTS = "glm_kda_causal_exp_v1"
CHUNK = 64
HEAD_DIM = 128
_TILE = 16
_DEPTH = 16
_NUM_WARPS = 4
#: Every compiled kernel, one specialization each: (kernel, inclusive).
KERNEL_KEYS = tuple((kernel, inclusive) for kernel in ("gram_fwd", "gram_bwd_rows", "gram_bwd_cols")
                    for inclusive in (False, True))

_COUNTS = {"calls": 0, "gram_forward": 0, "gram_backward": 0}
_COMPILED: dict = {}


class KdaKernelRefused(RuntimeError):
    """A call outside the kernel's declared scope, or a second specialization."""


@triton.jit
def _gram_fwd_kernel(a_ptr, b_ptr, hi_ptr, lo_ptr, out_ptr,
                     C: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
                     INCLUSIVE: tl.constexpr):
    m = tl.program_id(0).to(tl.int64)
    it = tl.program_id(1)
    base = m * C * K
    obase = m * C * C
    ri = it * BT + tl.arange(0, BT)
    for jt in range(0, it + 1):
        cj = jt * BT + tl.arange(0, BT)
        if INCLUSIVE:
            keep = cj[None, :] <= ri[:, None]
        else:
            keep = cj[None, :] < ri[:, None]
        keep3 = keep[:, :, None]
        acc = tl.zeros([BT, BT], dtype=tl.float64)
        for d0 in range(0, K, BD):
            dd = d0 + tl.arange(0, BD)
            offs_i = base + ri[:, None] * K + dd[None, :]
            offs_j = base + cj[:, None] * K + dd[None, :]
            a_i = tl.load(a_ptr + offs_i)
            h_i = tl.load(hi_ptr + offs_i)
            l_i = tl.load(lo_ptr + offs_i)
            b_j = tl.load(b_ptr + offs_j)
            h_j = tl.load(hi_ptr + offs_j)
            l_j = tl.load(lo_ptr + offs_j)
            diff = (h_i[:, None, :] - h_j[None, :, :]) + (l_i[:, None, :] - l_j[None, :, :])
            diff = tl.where(keep3, diff, 0.0)
            term = a_i[:, None, :] * b_j[None, :, :] * libdevice.exp(diff)
            acc += tl.sum(tl.where(keep3, term, 0.0), axis=2).to(tl.float64)
        tl.store(out_ptr + obase + ri[:, None] * C + cj[None, :], acc.to(tl.float32))


@triton.jit
def _gram_bwd_rows_kernel(da_ptr, dg_row_ptr, dA_ptr, a_ptr, b_ptr, hi_ptr, lo_ptr,
                          C: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
                          INCLUSIVE: tl.constexpr):
    m = tl.program_id(0).to(tl.int64)
    it = tl.program_id(1)
    base = m * C * K
    obase = m * C * C
    ri = it * BT + tl.arange(0, BT)
    diag = tl.load(dA_ptr + obase + ri * C + ri)
    for d0 in range(0, K, BD):
        dd = d0 + tl.arange(0, BD)
        offs_i = base + ri[:, None] * K + dd[None, :]
        a_i = tl.load(a_ptr + offs_i)
        b_i = tl.load(b_ptr + offs_i)
        h_i = tl.load(hi_ptr + offs_i)
        l_i = tl.load(lo_ptr + offs_i)
        acc = tl.zeros([BT, BD], dtype=tl.float64)
        for jt in range(0, it + 1):
            cj = jt * BT + tl.arange(0, BT)
            keep3 = (cj[None, :] < ri[:, None])[:, :, None]
            grad = tl.load(dA_ptr + obase + ri[:, None] * C + cj[None, :])
            offs_j = base + cj[:, None] * K + dd[None, :]
            b_j = tl.load(b_ptr + offs_j)
            h_j = tl.load(hi_ptr + offs_j)
            l_j = tl.load(lo_ptr + offs_j)
            diff = (h_i[:, None, :] - h_j[None, :, :]) + (l_i[:, None, :] - l_j[None, :, :])
            diff = tl.where(keep3, diff, 0.0)
            term = grad[:, :, None] * b_j[None, :, :] * libdevice.exp(diff)
            acc += tl.sum(tl.where(keep3, term, 0.0), axis=1).to(tl.float64)
        if INCLUSIVE:
            tl.store(da_ptr + offs_i, (acc + (diag[:, None] * b_i).to(tl.float64)).to(tl.float32))
        else:
            tl.store(da_ptr + offs_i, acc.to(tl.float32))
        tl.store(dg_row_ptr + offs_i, a_i.to(tl.float64) * acc)


@triton.jit
def _gram_bwd_cols_kernel(db_ptr, dg_ptr, dg_row_ptr, dA_ptr, a_ptr, b_ptr, hi_ptr, lo_ptr,
                          C: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr,
                          INCLUSIVE: tl.constexpr):
    m = tl.program_id(0).to(tl.int64)
    jt = tl.program_id(1)
    base = m * C * K
    obase = m * C * C
    cj = jt * BT + tl.arange(0, BT)
    diag = tl.load(dA_ptr + obase + cj * C + cj)
    for d0 in range(0, K, BD):
        dd = d0 + tl.arange(0, BD)
        offs_j = base + cj[:, None] * K + dd[None, :]
        a_j = tl.load(a_ptr + offs_j)
        b_j = tl.load(b_ptr + offs_j)
        h_j = tl.load(hi_ptr + offs_j)
        l_j = tl.load(lo_ptr + offs_j)
        acc = tl.zeros([BT, BD], dtype=tl.float64)
        for it in range(jt, C // BT):
            ri = it * BT + tl.arange(0, BT)
            keep3 = (cj[None, :] < ri[:, None])[:, :, None]
            grad = tl.load(dA_ptr + obase + ri[:, None] * C + cj[None, :])
            offs_i = base + ri[:, None] * K + dd[None, :]
            a_i = tl.load(a_ptr + offs_i)
            h_i = tl.load(hi_ptr + offs_i)
            l_i = tl.load(lo_ptr + offs_i)
            diff = (h_i[:, None, :] - h_j[None, :, :]) + (l_i[:, None, :] - l_j[None, :, :])
            diff = tl.where(keep3, diff, 0.0)
            term = grad[:, :, None] * a_i[:, None, :] * libdevice.exp(diff)
            acc += tl.sum(tl.where(keep3, term, 0.0), axis=0).to(tl.float64)
        if INCLUSIVE:
            tl.store(db_ptr + offs_j, (acc + (diag[:, None] * a_j).to(tl.float64)).to(tl.float32))
        else:
            tl.store(db_ptr + offs_j, acc.to(tl.float32))
        tl.store(dg_ptr + offs_j, tl.load(dg_row_ptr + offs_j) - b_j.to(tl.float64) * acc)


def _launch(name, kernel, inclusive, grid, *args):
    """Launch one kernel and hold the process to one compiled specialization."""
    compiled = kernel[grid](*args, C=CHUNK, K=HEAD_DIM, BT=_TILE, BD=_DEPTH,
                            INCLUSIVE=inclusive, num_warps=_NUM_WARPS)
    key = (name, bool(inclusive))
    known = _COMPILED.get(key)
    if known is None:
        _COMPILED[key] = compiled
    elif known.hash != compiled.hash:
        raise KdaKernelRefused(
            f"KDA kernel {name} (inclusive={inclusive}) compiled a second specialization "
            f"{compiled.hash} after {known.hash}; the kernel identity names one")


def _check_operand(tensor, name):
    if tensor.dtype != torch.float32 or not tensor.is_contiguous() or tensor.data_ptr() % 16:
        raise KdaKernelRefused(f"KDA Gram operand {name} must be contiguous, 16-byte aligned FP32")


class _KdaGram(torch.autograd.Function):
    """``A[m, i, j] = sum_d a[m, i, d] b[m, j, d] exp(G[m, i, d] - G[m, j, d])``.

    ``a`` and ``b`` are FP32 ``[M, 64, 128]``, ``gcum`` is FP64 ``[M, 64, 128]``.
    ``j < i`` for the strict form and ``j <= i`` for the inclusive form; every
    other entry of ``A`` is 0 and its gradient is not read.
    """

    @staticmethod
    def forward(ctx, a, b, gcum, inclusive):
        a, b = a.contiguous(), b.contiguous()
        hi = gcum.to(torch.float32)
        lo = (gcum - hi.to(torch.float64)).to(torch.float32)
        for tensor, name in ((a, "a"), (b, "b"), (hi, "hi"), (lo, "lo")):
            _check_operand(tensor, name)
        count = a.shape[0]
        out = torch.zeros((count, CHUNK, CHUNK), device=a.device, dtype=torch.float32)
        _launch("gram_fwd", _gram_fwd_kernel, inclusive, (count, CHUNK // _TILE),
                a, b, hi, lo, out)
        _COUNTS["gram_forward"] += 1
        ctx.save_for_backward(a, b, hi, lo)
        ctx.inclusive = bool(inclusive)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        a, b, hi, lo = ctx.saved_tensors
        grad = grad.to(torch.float32).contiguous()
        _check_operand(grad, "dA")
        da, db = torch.empty_like(a), torch.empty_like(b)
        dg_row = torch.empty(a.shape, device=a.device, dtype=torch.float64)
        dg = torch.empty(a.shape, device=a.device, dtype=torch.float64)
        grid = (a.shape[0], CHUNK // _TILE)
        _launch("gram_bwd_rows", _gram_bwd_rows_kernel, ctx.inclusive, grid,
                da, dg_row, grad, a, b, hi, lo)
        _launch("gram_bwd_cols", _gram_bwd_cols_kernel, ctx.inclusive, grid,
                db, dg, dg_row, grad, a, b, hi, lo)
        _COUNTS["gram_backward"] += 1
        return da, db, dg, None


def _l2norm(x, dim=-1, eps=1e-6):
    # The fallback's l2norm, operation for operation, in the caller's dtype.
    inv_norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / inv_norm


def chunk_kimi_delta_attention(query, key, value, g, beta, chunk_size=64, initial_state=None,
                               output_final_state=False, use_qk_l2norm_in_kernel=False, **kwargs):
    """The fallback's signature and result for the Stage B capture's calls.

    The capture calls this with the layer's arguments: chunk size 64, no
    initial state, no final state, and L2-normalized ``q`` and ``k``. Any
    other call is refused, never delegated. Extra keyword arguments are
    ignored, as the fallback's dispatch wrapper drops them.
    """
    del kwargs
    if chunk_size != CHUNK or initial_state is not None or output_final_state \
            or use_qk_l2norm_in_kernel is not True:
        raise KdaKernelRefused(
            "the KDA kernel serves only chunk_size=64, no initial or final state and "
            "use_qk_l2norm_in_kernel=True: the Stage B capture's call")
    if query.device.type != "cuda":
        raise KdaKernelRefused("the KDA kernel runs on CUDA only")
    if query.shape[-1] != HEAD_DIM or key.shape[-1] != HEAD_DIM or g.shape != key.shape:
        raise KdaKernelRefused(
            f"the KDA kernel is compiled for key head dim {HEAD_DIM}, got "
            f"q {tuple(query.shape)} k {tuple(key.shape)} g {tuple(g.shape)}")
    _COUNTS["calls"] += 1
    initial_dtype = query.dtype
    # Everything outside the three Gram kernels is FP64, forward and backward:
    # the L2 norms, the beta products, the solve, the decay factors and the
    # eight-chunk recurrence. Each input gradient rounds to its input's dtype
    # once, where the input entered FP64, and the output rounds once, to the
    # caller's dtype.
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float64) for x in (query, key, value, beta, g)]
    query = _l2norm(query, dim=-1, eps=1e-6)
    key = _l2norm(key, dim=-1, eps=1e-6)
    batch, heads, length, _ = key.shape
    v_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    pad = (CHUNK - length % CHUNK) % CHUNK
    chunks = (length + pad) // CHUNK
    query = F.pad(query, (0, 0, 0, pad)) * scale
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    g = F.pad(g, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, g, k_beta, v_beta = [
        x.reshape(batch, heads, chunks, CHUNK, x.shape[-1]) for x in (query, key, g, k_beta, v_beta)]
    gcum = g.cumsum(dim=-2)
    rows = batch * heads * chunks

    def flat32(x):
        # The Gram kernels read FP32 operands: one rounding of each FP64 value.
        return x.reshape(rows, CHUNK, x.shape[-1]).to(torch.float32)

    a_kk = _KdaGram.apply(flat32(k_beta), flat32(key), gcum.reshape(rows, CHUNK, HEAD_DIM), False)
    a_qk = _KdaGram.apply(flat32(query), flat32(key), gcum.reshape(rows, CHUNK, HEAD_DIM), True)
    decay = gcum.exp()
    # (I + A_kk)^-1 [v_beta | k_beta exp(G)]: the fallback's loop builds the
    # unit lower-triangular inverse in FP32 and multiplies; one FP64 solve
    # does both, and its backward is FP64 too.
    solved = torch.linalg.solve_triangular(
        a_kk.reshape(batch, heads, chunks, CHUNK, CHUNK).to(torch.float64),
        torch.cat([v_beta, k_beta * decay], dim=-1),
        upper=False, unitriangular=True)
    u, w = solved.split([v_dim, HEAD_DIM], dim=-1)
    last = gcum[..., -1, :].exp()
    tail = (gcum[..., -1:, :] - gcum).exp()
    per_chunk = [x.unbind(2) for x in (
        u, w, query * decay,
        a_qk.reshape(batch, heads, chunks, CHUNK, CHUNK).to(torch.float64), key * tail, last)]
    state = torch.zeros(batch, heads, HEAD_DIM, v_dim, dtype=torch.float64, device=value.device)
    outputs = []
    for u_n, w_n, q_n, a_n, k_n, last_n in zip(*per_chunk):
        v_new = u_n - w_n @ state
        outputs.append(q_n @ state + a_n @ v_new)
        state = state * last_n.unsqueeze(-1) + k_n.transpose(-1, -2) @ v_new
    out = torch.stack(outputs, dim=2).reshape(batch, heads, chunks * CHUNK, v_dim)[:, :, :length]
    return out.transpose(1, 2).contiguous().to(initial_dtype), None


def counts() -> dict:
    """Calls and Gram passes so far in this process."""
    return dict(_COUNTS)


def source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _code_sha256(cubin: bytes) -> str:
    """sha256 of a cubin's sections, leaving out its debug sections.

    ptxas writes each source file's modification time into the cubin's line
    tables, so two checkouts of one source compile to cubins whose bytes
    differ there and nowhere else. Every other section (machine code,
    constants, kernel attributes, symbols, the toolkit note) is hashed by
    name, type, flags and contents; offsets are not, because a debug section
    of another length would move them.
    """
    if cubin[:6] != b"\x7fELF\x02\x01":
        raise KdaKernelRefused("a compiled KDA kernel is not a 64-bit little-endian ELF")
    section_offset, = struct.unpack_from("<Q", cubin, 0x28)
    entry_size, count, names_index = struct.unpack_from("<HHH", cubin, 0x3A)

    def section(index):
        return struct.unpack_from("<IIQQQQIIQQ", cubin, section_offset + index * entry_size)

    names_offset = section(names_index)[4]
    digest = hashlib.sha256(cubin[16:24] + cubin[0x30:0x34])  # type, machine, version, flags
    for index in range(count):
        name_at, kind, flags, _address, offset, size = section(index)[:6]
        name = cubin[names_offset + name_at:cubin.index(b"\0", names_offset + name_at)]
        if b"debug" in name:
            continue
        digest.update(name + b"\0" + struct.pack("<IQQ", kind, flags, size))
        if kind != 8:  # SHT_NOBITS occupies no bytes in the file
            digest.update(cubin[offset:offset + size])
    return digest.hexdigest()


def compiled_kernels() -> dict:
    """``{"<kernel>_<strict|inclusive>": identity}`` for every kernel compiled so far.

    ``code_sha256`` is the cubin without its debug sections (``_code_sha256``);
    ``triton_hash`` is Triton's key for the compile inputs, which covers the
    source, the specialization, the options and the ptxas version.
    """
    result = {}
    for (name, inclusive), compiled in sorted(_COMPILED.items()):
        result[f"{name}_{'inclusive' if inclusive else 'strict'}"] = {
            "code_sha256": _code_sha256(compiled.kernel),
            "triton_hash": compiled.hash}
    return result


def probe_digest(device) -> dict:
    """Run a fixed small problem forward and backward and hash every result.

    Compiles all six kernels on first use. The inputs come from a seeded CPU
    generator, so the global Torch RNG is untouched; the digest names the
    output and gradient bits the whole path produced on this device.
    """
    generator = torch.Generator().manual_seed(1199)
    shape = (1, 2 * CHUNK, 2, HEAD_DIM)

    def draw(*size, low=None):
        value = torch.randn(*size, generator=generator)
        return value if low is None else low * torch.sigmoid(value)

    inputs = [draw(*shape).to(torch.bfloat16), draw(*shape).to(torch.bfloat16),
              draw(*shape).to(torch.bfloat16), draw(*shape, low=-5.0),
              torch.sigmoid(draw(*shape[:3])).to(torch.bfloat16)]
    stimulus = draw(*shape).to(torch.bfloat16).to(device)
    inputs = [x.to(device).requires_grad_(True) for x in inputs]
    out, _ = chunk_kimi_delta_attention(*inputs, use_qk_l2norm_in_kernel=True)
    torch.autograd.backward([out], [stimulus])
    digest = hashlib.sha256()
    for tensor in (out, *(x.grad for x in inputs)):
        digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return {"shape": list(shape), "sha256": digest.hexdigest()}
