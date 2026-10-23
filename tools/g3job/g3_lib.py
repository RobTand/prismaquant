"""G3 offline decoded-forward evaluator: the pieces the GPU run and the CPU tests share.

The helpers reuse the accepted G3 arithmetic and tensor digest owners.
They import no Tessera or Transformers package at module scope.
Range I/O resolves the admitted PB reader lazily. CPU tests need no CUDA.

What lives here:
  * the accepted served activation quantizer, fp8_per_token_dynamic;
  * the TSRFUSE1 container framing (tessera/fused.py:37 FUSED_MAGIC, :105 parse_fused);
  * the shared tensor digest owner, prismaquant/tensor_digests.py;
  * the map from a priced Linear (payload row name) to the tensor slice the streamed
    transformers model executes, and the in-place substitution with its hash gate;
  * the W+A hooks that put the served activation contract on a streamed MLP.

THE ACTIVATION CONTRACT, AND WHERE ITS ARITHMETIC COMES FROM (principle 14).
Tessera b40c93cb, src/tessera/serving/runtime_contract.json:489: on sm_121
TESSERA_E4M3_K1 executes "fp8_per_token_dynamic".  scheme.py:138 names the same string;
serving/native_ops.py:179/200 implements it as vLLM's registered op
torch.ops._C.dynamic_per_token_scaled_fp8_quant; routed_fused.py:828-830 and
serving/fp8_route.py:446 call it on every Tessera E4M3 GEMM's A side.  The contract's
activation_quantizers block does NOT attest this contract's arithmetic (it names only
e2m1_group16_ue4m3_static). The shared arithmetic retains the original G3 operation order.
The reference is Tessera's pinned tests/test_native_fp8_quant.py.
Its `_kernel_arithmetic` function restates the pinned kernel source.
The historical native check uses vLLM 0.28.1rc1.dev397+gfd4a15126 on 2026-09-28:
    scale = max(fl32(amax / 448), fl32(1 / (448 * 512)))
    code  = e4m3_rne_sat(clamp(fl32(x / scale), -448, 448))
The CPU test compares the shared arithmetic with that independent pinned reference.
This source import does not repeat the historical native GPU qualification.

TENSOR PARALLELISM.  The served T8R ran TP2 (U4 engine_kwargs.tensor_parallel_size = 2, no
expert parallelism).  A column-parallel GEMM (gate/up) sees the whole hidden row, so its
per-token scale spans all of it.  A row-parallel GEMM (down) sees only its rank's contiguous
half of the intermediate row, and each rank quantizes its own half with its own scale
(routed_fused.py: `act` is [routes, inter] with inter = the rank-local down.cols).  So the down
input is quantized per row per contiguous 1/tp slice.
"""
from __future__ import annotations

import struct

import torch
from g3_pq_policy.tensor_digests import tensor_sha256
from g3_pq_policy.g3_numerics import FP8_MAX, MIN_SCALE, fp8_per_token_dynamic, qdq_rows

TP_SERVED = 2




# ---------------------------------------------------------------- wire framing and identity
_FUSED_HEAD = struct.Struct("<8sBB")
_FUSED_MEMBER = struct.Struct("<HIQ")
FUSED_MAGIC = b"TSRFUSE1"


def unwrap_members(data: bytes) -> dict:
    """{member name: blob} of a TSRFUSE1 container, or {None: data} for a bare unit blob.

    Framing restated from tessera/fused.py (b40c93cb) pack_fused/parse_fused: header <8sBB
    (magic, version, count); then, per member, <HIQ (name_len, rows, blob_len) immediately
    followed by that member's UTF-8 name; then the blobs in member order.  Refuses trailing or
    missing bytes.
    """
    if len(data) < _FUSED_HEAD.size or data[:8] != FUSED_MAGIC:
        return {None: bytes(data)}
    hlen, heads = fused_header(data)
    cur, out = hlen, {}
    for name, _rows, blob_len in heads:
        if name in out:
            raise ValueError(f"duplicate fused member {name!r}")
        blob = bytes(data[cur:cur + blob_len])
        if len(blob) != blob_len:
            raise ValueError(f"fused member {name!r}: truncated blob")
        out[name] = blob
        cur += blob_len
    if cur != len(data):
        raise ValueError(f"fused container framing leaves {len(data) - cur} bytes unaccounted")
    return out


def fused_header(prefix: bytes):
    """(header_bytes, [(name, rows, blob_len), ...]) of a TSRFUSE1 container from its first
    bytes, or None for a bare unit blob.  Each member's <HIQ record is followed at once by its
    name (tessera/fused.py pack_fused).  Raises if ``prefix`` is too short for the header."""
    if len(prefix) < 8 or prefix[:8] != FUSED_MAGIC:
        return None
    if len(prefix) < _FUSED_HEAD.size:
        raise ValueError("prefix too short for the fused header")
    _magic, version, count = _FUSED_HEAD.unpack_from(prefix)
    if version != 1 or count == 0:
        raise ValueError(f"unsupported fused container version {version} / count {count}")
    cur, out = _FUSED_HEAD.size, []
    for _ in range(count):
        if cur + _FUSED_MEMBER.size > len(prefix):
            raise ValueError("prefix too short for the fused header")
        name_len, rows, blob_len = _FUSED_MEMBER.unpack_from(prefix, cur)
        cur += _FUSED_MEMBER.size
        if cur + name_len > len(prefix):
            raise ValueError("prefix too short for the fused header")
        out.append((bytes(prefix[cur:cur + name_len]).decode("utf-8"), rows, blob_len))
        cur += name_len
        if rows <= 0 or blob_len <= 0:
            raise ValueError(f"fused member {out[-1][0]!r}: rows {rows}, blob {blob_len}")
    return cur, out




class HashGateError(RuntimeError):
    """A tensor's identity differs from the identity the payload priced."""


def check_identity(t: torch.Tensor, expected_sha: str, what: str) -> str:
    got = tensor_sha256(t)
    if got != expected_sha:
        raise HashGateError(f"{what}: sha256 {got[:16]} != priced {expected_sha[:16]}")
    return got


# ---------------------------------------------------------------- priced Linear -> executed slice
def unit_view(layer_module: torch.nn.Module, row: dict) -> torch.Tensor:
    """The writable tensor the streamed transformers layer executes for one priced Linear.

    row: {"kind": routed|shared|dense, "role": gate_proj|up_proj|down_proj, "expert": int|None}.
    Routed experts live packed (transformers integrations/moe.py, is_concatenated=True,
    is_transposed=False): gate_up_proj [E, 2I, H] with gate rows [:I] and up rows [I:],
    down_proj [E, H, I].
    """
    kind, role = row["kind"], row["role"]
    mlp = layer_module.mlp
    if kind == "routed":
        experts = mlp.experts
        e = int(row["expert"])
        if role == "down_proj":
            return experts.down_proj.data[e]
        inter = experts.gate_up_proj.shape[1] // 2
        return experts.gate_up_proj.data[e, :inter] if role == "gate_proj" else experts.gate_up_proj.data[e, inter:]
    if kind == "shared":
        return getattr(mlp.shared_experts, role).weight.data
    if kind == "dense":
        return getattr(mlp, role).weight.data
    raise ValueError(f"unknown unit kind {kind!r}")


def substitute(layer_module, rows, tensors, *, check_source=True, check_rendered=True, hasher=None):
    """Swap each priced Linear's executed slice for its decoded weight, hash-gated both ways.

    For each row: (1) the slice currently installed must hash to the payload's source identity
    (this is what proves the name -> slice map); (2) the replacement must hash to the priced
    rendered identity for the arm's format; (3) it is copied in place.  ``tensors`` maps the
    row's qname to the replacement (None = keep the source slice: the null arm).  ``hasher``
    optionally maps a list of (tensor, expected, what) to identities in parallel.
    Returns per-row records.  Raises HashGateError on the first mismatch, before any copy of
    that layer: all checks run first, then all copies.
    """
    views = [unit_view(layer_module, r) for r in rows]
    jobs = []
    for r, v in zip(rows, views):
        if check_source:
            jobs.append((v, r["source_sha256"], r["qname"] + " [installed source]"))
        rep = tensors.get(r["qname"])
        if rep is not None:
            if tuple(rep.shape) != tuple(v.shape) or rep.dtype != v.dtype:
                raise HashGateError(f"{r['qname']}: replacement {tuple(rep.shape)}/{rep.dtype} "
                                    f"!= executed {tuple(v.shape)}/{v.dtype}")
            if check_rendered:
                jobs.append((rep, r["rendered_sha256"], r["qname"] + " [decoded]"))
    if hasher is None:
        for t, exp, what in jobs:
            check_identity(t, exp, what)
    else:
        hasher(jobs)
    copied = 0
    for r, v in zip(rows, views):
        rep = tensors.get(r["qname"])
        if rep is not None:
            v.copy_(rep.to(device=v.device))
            copied += 1
    return {"checked": len(jobs), "copied": copied}


# ---------------------------------------------------------------- the served A side (W+A arms)
def install_wa_hooks(layer_module: torch.nn.Module, tp: int = TP_SERVED, skip=frozenset()):
    """Put fp8_per_token_dynamic on every Tessera E4M3 GEMM input of one streamed MLP.

    Column-parallel inputs (gate/up; the routed experts' hidden rows) are quantized per token
    over the whole row; row-parallel inputs (down; the routed experts' per-route activation) per
    token per contiguous 1/tp slice.  The router and the attention are not Tessera units and
    stay unquantized.  Returns a remover.

    ``skip``: what serving runs on 16-bit activations, so gets no hook: a unit kind ("shared",
    "dense": source BF16 passthrough), a single (kind, role) of a shared or dense MLP (a
    passthrough or TESSERA_BF16_K1 projection), or "routed" (the layer's whole routed stack on
    TESSERA_BF16_K1, whose executed contract is bf16_unquantized).
    """
    skip = frozenset(skip)
    roles = ("gate_proj", "up_proj", "down_proj")
    bad = [s for s in skip if not (s in ("shared", "dense", "routed") or (
        isinstance(s, tuple) and len(s) == 2 and s[0] in ("shared", "dense") and s[1] in roles))]
    if bad:
        raise ValueError(f"cannot skip {sorted(map(str, bad))}: only routed, shared, dense or (shared|dense, role)")
    mlp = layer_module.mlp
    handles, restore = [], []

    def pre_whole(_m, args):
        return (qdq_rows(args[0], 1),) + tuple(args[1:])

    def pre_tp(_m, args):
        return (qdq_rows(args[0], tp),) + tuple(args[1:])

    def mlp_hooks(m, kind):
        for role, hook in (("gate_proj", pre_whole), ("up_proj", pre_whole), ("down_proj", pre_tp)):
            if kind in skip or (kind, role) in skip:
                continue
            handles.append(getattr(m, role).register_forward_pre_hook(hook))

    if hasattr(mlp, "experts"):
        if "routed" not in skip:
            experts = mlp.experts
            handles.append(experts.register_forward_pre_hook(pre_whole))
            original = experts._apply_gate  # bound method of the class

            def gated(gate_up):
                return qdq_rows(original(gate_up), tp)

            experts._apply_gate = gated
            restore.append(lambda: delattr(experts, "_apply_gate"))
        mlp_hooks(mlp.shared_experts, "shared")
    else:
        mlp_hooks(mlp, "dense")

    def remove():
        for h in handles:
            h.remove()
        for f in restore:
            f()
    return remove


# ---------------------------------------------------------------- wire bytes -> gated tensor
def read_range(path: str, offset: int, length: int) -> bytes:
    """``length`` bytes of ``path`` at ``offset``; refuses a short read."""
    from g3_residency import staged_range
    staged = staged_range(path, offset, length)
    if staged is not None:
        return bytes(staged)
    import os
    fd = os.open(path, os.O_RDONLY)
    try:
        out = bytearray(length)
        view, got = memoryview(out), 0
        while got < length:
            n = os.preadv(fd, [view[got:]], offset + got)
            if n <= 0:
                raise IOError(f"short read of {path} at {offset + got}: wanted {length - got} more bytes")
            got += n
        return bytes(out)
    finally:
        os.close(fd)


def member_location(loc: dict):
    """(absolute offset, length) of one Linear's unit blob inside an export tensor, from a
    manifest wire location (g3_manifest.py): the member of a TSRFUSE1 container, or the whole
    tensor for a bare blob."""
    if loc.get("member") is None and "member_offset" not in loc:
        return loc["offset"], loc["length"]
    return loc["member_offset"], loc["member_bytes"]


def decode_gated(blob: bytes, expected_sha: str, what: str, *, decoder, device="cpu"):
    """Decode one unit blob, cast to the source dtype (bf16), and refuse unless it hashes to
    the priced rendered identity.  ``decoder`` is tessera.unit_artifact.read_unit_artifact.
    Decoder errors propagate unchanged (a blob the decoder refuses is refused)."""
    dec = decoder(blob, device=device).to(torch.bfloat16)
    check_identity(dec, expected_sha, what)
    return dec
