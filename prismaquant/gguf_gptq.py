"""GPTQ-into-k-quant: OBS error-propagated rounding under frozen
two-tier GGUF scales.

The two-tier scale structure (fp16 super-scale + quantized per-sub-block
scales/mins) is decided by the imatrix-weighted reference search in
:mod:`prismaquant.gguf_formats` on the ORIGINAL weights and then frozen;
the GPTQ pass (Frantar et al. 2022) re-decides only the integer ``q`` per
element, propagating each column's quantization error through the
activation Hessian's Cholesky inverse. This strictly generalizes the
diagonal imatrix weighting llama.cpp uses (the Hessian's diagonal IS the
imatrix; GPTQ adds the off-diagonal error compensation), and mirrors the
NVFP4 lane's static-scale GPTQ (`gptq_obs_rounding_nvfp4`), whose
Hessian preparation and damp conventions are reused verbatim:
``H = X^T X`` with per-token activation clipping, dead activation
channels excluded from the damp reference and NOT zeroed (serving-safe),
``diag += damp * mean_alive_diag`` with the promoted fixed damp 1.0
(``PRISMAQUANT_GPTQ_DAMP`` overrides).

Output is a fields dict whose ``q`` was rewritten under the frozen
scales — `gguf_pack_fields` packs it and `reconstruct_fields` gives the
bit-identical dequant, so the one-math-path invariant holds for GPTQ
renders exactly as for RTN.
"""
from __future__ import annotations

import torch

from prismaquant.gguf_formats import QK_K, compute_fields

# (sub_block, qmin, qmax, asymmetric) per format.
_GRID = {
    "Q2_K": (16, 0, 3, True),
    "Q3_K": (16, -4, 3, False),
    "Q4_K": (32, 0, 15, True),
    "Q5_K": (32, 0, 31, True),
    "Q6_K": (16, -32, 31, False),
    "Q8_0": (32, -128, 127, False),
}

# Formats whose fields carry a uniform integer ``q`` that GPTQ can re-round.
# The IQ family stores grid/codebook indices, not a uniform integer level, so
# it has no GPTQ rounder here and renders via the imatrix-RTN path instead.
GPTQ_SUPPORTED = frozenset(_GRID)


def _per_element_scales(
    fields: dict[str, torch.Tensor], fmt: str, shape: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand frozen superblock fields to per-element (dl, ml), (out, in)."""
    sub, _, _, asym = _GRID[fmt]
    block = 32 if fmt == "Q8_0" else QK_K
    n = fields["q"].shape[0]
    if fmt == "Q8_0":
        dl = fields["d"].expand(n, block)
        ml = torch.zeros_like(dl)
    else:
        dl = (fields["d"] * fields["sc"].float()).repeat_interleave(sub, dim=1)
        if asym:
            ml = (fields["dmin"] * fields["m"].float()).repeat_interleave(
                sub, dim=1)
        else:
            ml = torch.zeros_like(dl)
    return (dl.reshape(shape).to(device), ml.reshape(shape).to(device))


def _hessian_inverse(
    weight: torch.Tensor, activations: torch.Tensor, damp: float | None,
    *, levers,
) -> torch.Tensor:
    """Upper-triangular Cholesky of H^-1, with the NVFP4 lane's activation
    clipping, dead-channel, and damp conventions."""
    from prismaquant.native_render_codec import (
        activation_matrix_for_gptq,
        resolve_gptq_fixed_damp,
    )

    if damp is None:
        damp = resolve_gptq_fixed_damp(levers=levers)
    cols = weight.shape[1]
    X = activation_matrix_for_gptq(
        activations, cols, device=weight.device, levers=levers)
    H = (X.t() @ X).to(torch.float32)
    diag0 = torch.diagonal(H)
    dead = diag0 <= 0
    alive = ~dead
    diag_mean = (
        diag0[alive].mean() if bool(alive.any()) else diag0.new_ones(())
    ).clamp_min(1e-12)
    if bool(dead.any()):
        H[dead, dead] = 1.0
    H.diagonal().add_(damp * diag_mean)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    return torch.linalg.cholesky(Hinv, upper=True)


def gptq_fields(
    w: torch.Tensor,
    fmt: str,
    activations: torch.Tensor,
    col_weights: torch.Tensor | None = None,
    damp: float | None = None,
    block_cols: int = 128,
    *,
    levers,
) -> dict[str, torch.Tensor]:
    """GPTQ-rounded fields for a 2-D weight under frozen k-quant scales.

    ``levers`` is the ``NativeRenderLevers`` whose damp and activation-clip
    controls the Hessian borrows from the native NVFP4 lane.
    """
    if w.ndim != 2:
        raise ValueError(f"gptq_fields expects a 2-D weight, got {w.shape}")
    sub, qmin, qmax, asym = _GRID[fmt]
    fields = compute_fields(w, fmt, col_weights=col_weights)

    rows, cols = int(w.shape[0]), int(w.shape[1])
    device = w.device
    W = w.detach().to(torch.float32).clone()
    dl, ml = _per_element_scales(fields, fmt, (rows, cols), device)
    dl_safe = torch.where(dl != 0, dl, torch.ones_like(dl))
    live = dl != 0

    Hinv = _hessian_inverse(W, activations, damp, levers=levers)
    Q = torch.zeros(rows, cols, dtype=torch.float32, device=device)

    for i1 in range(0, cols, block_cols):
        i2 = min(i1 + block_cols, cols)
        count = i2 - i1
        W1 = W[:, i1:i2].clone()
        Err = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for j in range(count):
            i = i1 + j
            w_col = W1[:, j]
            q = torch.round((w_col + ml[:, i]) / dl_safe[:, i])
            q = torch.where(live[:, i], q.clamp(qmin, qmax),
                            torch.zeros_like(q))
            dq = dl[:, i] * q - ml[:, i]
            Q[:, i] = q
            err = (w_col - dq) / Hinv1[j, j]
            if j + 1 < count:
                W1[:, j + 1:] -= err.unsqueeze(1) * Hinv1[j, j + 1:].unsqueeze(0)
            Err[:, j] = err
        if i2 < cols:
            W[:, i2:] -= Err @ Hinv[i1:i2, i2:]

    q_dtype = torch.uint8 if asym else torch.int8
    block = 32 if fmt == "Q8_0" else QK_K
    fields["q"] = Q.reshape(-1, block).to(q_dtype)
    return fields
