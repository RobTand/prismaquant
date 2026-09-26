"""Research helpers for the DSv4 fused-rotation and CB-feedback pilot.

Nothing in this module is wired into production.  The feedback prototype
keeps the production encoder's codebook and scales fixed, reassigns complete
8-wide CB vectors (and their product-codebook subtables), and adds only
block-sequential off-diagonal Hessian feedback.
"""
from __future__ import annotations

from collections.abc import Callable

import torch

from .cb_layout import FP4_GROUP, VEC_DIM
from .row_dispersion import per_row_error


def activation_row_split_indices(
    row_count: int,
    *,
    cal_rows: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return deterministic, disjoint CAL and HOLDOUT row indices.

    The permutation is generated on CPU so a split is identical regardless of
    the device used for the experiment.  Asking for CAL16 and CAL32 with the
    same ``row_count`` and ``seed`` makes CAL16 a prefix (and therefore a
    subset) of CAL32, which supports a controlled calibration-size probe.
    """
    row_count = int(row_count)
    cal_rows = int(cal_rows)
    if row_count < 2:
        raise ValueError(f"row_count must be at least 2, got {row_count}")
    if not 0 < cal_rows < row_count:
        raise ValueError(
            f"cal_rows must lie strictly between 0 and {row_count}, got {cal_rows}"
        )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(row_count, generator=generator)
    return permutation[:cal_rows].clone(), permutation[cal_rows:].clone()


def compare_output_sse(
    activations: torch.Tensor,
    weight: torch.Tensor,
    plain: torch.Tensor,
    feedback: torch.Tensor,
) -> dict[str, float]:
    """Evaluate plain and feedback reconstructions on one activation split."""
    plain_sse = float(per_row_error(activations, weight, plain).sum())
    feedback_sse = float(per_row_error(activations, weight, feedback).sum())
    if plain_sse <= 0.0:
        raise ValueError(f"plain output SSE must be positive, got {plain_sse}")
    ratio = feedback_sse / plain_sse
    return {
        "plain_sse": plain_sse,
        "feedback_sse": feedback_sse,
        "feedback_over_plain_ratio": ratio,
        "reduction": 1.0 - ratio,
    }


def random_hadamard_signs(
    width: int,
    *,
    seed: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return the float64 Rademacher diagonal for a randomized Hadamard."""
    width = int(width)
    if width <= 0 or width & (width - 1):
        raise ValueError(f"Hadamard width must be a positive power of two, got {width}")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    bits = torch.randint(0, 2, (width,), generator=generator, dtype=torch.int64)
    signs = bits.to(torch.float64).mul_(2.0).sub_(1.0)
    return signs.to(device=device)


def randomized_hadamard_right(
    tensor: torch.Tensor,
    signs: torch.Tensor,
    *,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Apply ``D @ Hadamard / sqrt(n)`` on the right without forming it.

    ``signs`` is constructed in float64.  Arithmetic uses float64 when the
    input is float64 and float32 otherwise; production simulations request a
    bf16 output to model the folded stored weight and transformed activation.
    """
    if tensor.ndim < 1:
        raise ValueError("Hadamard input must have at least one dimension")
    width = int(tensor.shape[-1])
    if width <= 0 or width & (width - 1):
        raise ValueError(f"Hadamard width must be a positive power of two, got {width}")
    if signs.shape != (width,):
        raise ValueError(f"signs must have shape ({width},), got {tuple(signs.shape)}")
    work_dtype = torch.float64 if tensor.dtype == torch.float64 else torch.float32
    work = tensor.to(work_dtype) * signs.to(tensor.device, work_dtype)
    leading_shape = tuple(work.shape[:-1])
    step = 1
    while step < width:
        paired = work.reshape(*leading_shape, -1, 2 * step)
        left = paired[..., :step]
        right = paired[..., step:]
        work = torch.cat((left + right, left - right), dim=-1).reshape(
            *leading_shape, width
        )
        step *= 2
    work = work * (width ** -0.5)
    return work.to(output_dtype or tensor.dtype)


def relative_frobenius_gap(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Return ``||reference-candidate||_F / ||reference||_F`` in float64."""
    if reference.shape != candidate.shape:
        raise ValueError(
            f"gap operands differ: {tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    ref = reference.to(torch.float64)
    cand = candidate.to(torch.float64)
    denominator = torch.linalg.vector_norm(ref)
    if float(denominator) == 0.0:
        return 0.0 if bool(torch.equal(ref, cand)) else float("inf")
    return float(torch.linalg.vector_norm(ref - cand) / denominator)


def inverse_hessian_cholesky(
    hessian: torch.Tensor,
    *,
    damping_fraction: float = 0.0,
) -> tuple[torch.Tensor, float]:
    """Build GPTQ's upper Cholesky factor of the damped inverse Hessian.

    The returned ``U`` satisfies ``U.T @ U == inverse(H_damped)`` up to
    floating-point error.  Damping is ``fraction * mean(diag(H))``.
    """
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise ValueError(f"hessian must be square, got {tuple(hessian.shape)}")
    if damping_fraction < 0:
        raise ValueError("damping_fraction must be non-negative")
    h = hessian.to(torch.float32).clone()
    damping = float(h.diagonal().mean()) * float(damping_fraction)
    if damping:
        h.diagonal().add_(damping)
    lower = torch.linalg.cholesky(h)
    inverse = torch.cholesky_inverse(lower)
    upper = torch.linalg.cholesky(inverse, upper=True)
    return upper, damping


def block_error_feedback(
    weight: torch.Tensor,
    upper_inverse_cholesky: torch.Tensor,
    quantize_block: Callable[[torch.Tensor, int, int], torch.Tensor],
    *,
    block_size: int,
) -> torch.Tensor:
    """Quantize columns blockwise with GPTQ/LDLQ-style residual feedback.

    For block ``A`` the callback quantizes the current compensated weights.
    With residual ``R_A`` and GPTQ factor ``U``, ``E_A U_AA = R_A`` is solved
    and ``E_A U_A,*`` is subtracted from all current/future columns.  Width-1
    blocks reduce exactly to the standard GPTQ update ``e / U[i,i]``.
    """
    if weight.ndim != 2:
        raise ValueError(f"weight must be 2-D, got {tuple(weight.shape)}")
    columns = int(weight.shape[1])
    if upper_inverse_cholesky.shape != (columns, columns):
        raise ValueError(
            "inverse-Hessian factor shape does not match weight columns: "
            f"{tuple(upper_inverse_cholesky.shape)} vs {columns}"
        )
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    work = weight.to(torch.float32).clone()
    quantized = torch.empty_like(work)
    upper = upper_inverse_cholesky.to(work.device, torch.float32)
    for start in range(0, columns, block_size):
        end = min(columns, start + block_size)
        block = work[:, start:end]
        qblock = quantize_block(block, start, end).to(work.device, torch.float32)
        if qblock.shape != block.shape:
            raise ValueError(
                f"quantize_block returned {tuple(qblock.shape)}, expected {tuple(block.shape)}"
            )
        quantized[:, start:end] = qblock
        residual = block - qblock
        diagonal_block = upper[start:end, start:end]
        # Solve E @ U_AA = R without explicitly inverting the triangular tile.
        scaled_error = torch.linalg.solve_triangular(
            diagonal_block.T,
            residual.T,
            upper=False,
        ).T
        work[:, start:] -= scaled_error @ upper[start:end, start:]
    return quantized.to(weight.dtype)


def product_cb_quantize_block(
    block: torch.Tensor,
    *,
    start: int,
    end: int,
    fields: dict,
    col_weights: torch.Tensor,
) -> torch.Tensor:
    """Nearest production-metric assignment for a fixed FP4 product CB tile."""
    from .nvfp4_cb_formats import _vq_assign

    if start % VEC_DIM or end % VEC_DIM:
        raise ValueError("CB reassignment blocks must be aligned to 8-wide vectors")
    if start % FP4_GROUP or end % FP4_GROUP:
        raise ValueError("CB reassignment blocks must preserve group-16 scales")
    rows = int(block.shape[0])
    scales = fields["scales"].to(block.device, torch.float32)
    per_element_scale = scales.repeat_interleave(FP4_GROUP, dim=1)[:, start:end]
    normalized = (block.to(torch.float32) / per_element_scale).reshape(-1, VEC_DIM)
    weights = torch.broadcast_to(
        col_weights[start:end].to(block.device, torch.float32), block.shape
    ).reshape(-1, VEC_DIM)
    codebooks = tuple(table.to(block.device, torch.float32) for table in fields["codebook"])
    if len(codebooks) != 2:
        raise ValueError(f"expected two FP4 product subtables, got {len(codebooks)}")
    sub_width = VEC_DIM // len(codebooks)
    vectors_per_row = (end - start) // VEC_DIM
    decoded_parts = []
    for index, table in enumerate(codebooks):
        lo, hi = index * sub_width, (index + 1) * sub_width
        assignment = _vq_assign(
            normalized[:, lo:hi],
            table,
            weights[:, lo:hi],
            vectors_per_row if rows > 1 else None,
        )
        decoded_parts.append(table[assignment])
    decoded = torch.cat(decoded_parts, dim=-1).reshape(rows, end - start)
    return decoded * per_element_scale


def reassign_product_cb(
    weight: torch.Tensor,
    fields: dict,
    col_weights: torch.Tensor,
    *,
    block_size: int,
    upper_inverse_cholesky: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reassign fixed FP4 product-CB codebooks/scales, optionally with feedback."""
    columns = int(weight.shape[1])
    if columns % block_size or block_size % FP4_GROUP:
        raise ValueError(
            f"block_size={block_size} must divide {columns} and be a multiple of 16"
        )

    def quantize(block: torch.Tensor, start: int, end: int) -> torch.Tensor:
        return product_cb_quantize_block(
            block,
            start=start,
            end=end,
            fields=fields,
            col_weights=col_weights,
        )

    if upper_inverse_cholesky is None:
        pieces = [
            quantize(weight[:, start:start + block_size], start, start + block_size)
            for start in range(0, columns, block_size)
        ]
        return torch.cat(pieces, dim=1).to(weight.dtype)
    return block_error_feedback(
        weight,
        upper_inverse_cholesky,
        quantize,
        block_size=block_size,
    )


__all__ = [
    "activation_row_split_indices",
    "block_error_feedback",
    "compare_output_sse",
    "inverse_hessian_cholesky",
    "product_cb_quantize_block",
    "random_hadamard_signs",
    "randomized_hadamard_right",
    "reassign_product_cb",
    "relative_frobenius_gap",
]
