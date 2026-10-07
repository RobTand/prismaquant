"""Accepted G3 arithmetic. Keep casts and operation order unchanged."""
import torch
FP8_MAX = 448.0
MIN_SCALE = float(torch.tensor(1.0, dtype=torch.float32) / (torch.tensor(FP8_MAX, dtype=torch.float32) * 512.0))

def fp8_per_token_dynamic(x2d: torch.Tensor):
    """Codes (float8_e4m3fn) and fp32 per-row scales [rows, 1] of the served quantizer."""
    if x2d.dim() != 2:
        raise ValueError("per-token quantization takes a 2-D [rows, K] tensor")
    xf = x2d.float()
    amax = xf.abs().amax(dim=1, keepdim=True)
    scale = (amax.double() / FP8_MAX).float().clamp_min(MIN_SCALE)
    codes = (xf.double() / scale.double()).float().clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return codes, scale


def qdq_rows(x: torch.Tensor, groups: int = 1) -> torch.Tensor:
    """Quantize-dequantize every row of ``x`` (last dim K) under fp8_per_token_dynamic, with
    one scale per row per contiguous K/groups slice; returns x's dtype and shape.

    The dequantized value code * scale is formed in fp32 and rounded once to x's dtype (bf16
    here): the served GEMM keeps it unrounded in its fp32 epilogue, so this is the offline
    evaluator's one extra rounding on the A side (relative 2^-9, against the 2^-4 of E4M3).
    """
    shape, dtype = x.shape, x.dtype
    k = shape[-1]
    if groups < 1 or k % groups:
        raise ValueError(f"cannot split K={k} into {groups} equal tensor-parallel slices")
    rows = x.reshape(-1, groups, k // groups).reshape(-1, k // groups)
    codes, scale = fp8_per_token_dynamic(rows)
    deq = (codes.float() * scale).to(dtype)
    return deq.reshape(-1, groups, k // groups).reshape(shape)



def token_kl(teacher, candidate, *, tile_rows=32, require_cuda=True):
    """Match upstream token_kld_chunk on raw logits without CPU vocabulary work."""
    if (teacher.ndim != 2 or teacher.shape != candidate.shape or teacher.shape[1] <= 1
            or teacher.shape[0] <= 0 or type(tile_rows) is not int or tile_rows <= 0):
        raise ValueError("teacher/candidate logit geometry or tile size mismatch")
    if teacher.device != candidate.device or (require_cuda and teacher.device.type != "cuda"):
        raise ValueError("KL requires co-resident CUDA tensors")
    result = torch.empty(teacher.shape[0], dtype=torch.float64, device=teacher.device)
    for start in range(0, teacher.shape[0], tile_rows):
        end = min(start + tile_rows, teacher.shape[0])
        t, c = teacher[start:end].to(torch.float64), candidate[start:end].to(torch.float64)
        if not bool(torch.isfinite(t).all() & torch.isfinite(c).all()):
            raise ValueError("teacher/candidate logits must be finite")
        t = torch.log_softmax(t, dim=-1)
        c = torch.log_softmax(c, dim=-1)
        result[start:end] = (t.exp() * (t - c)).sum(dim=-1, dtype=torch.float64)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("nonfinite full-vocabulary KL")
    return result

def g3_summary(kl_rows, agreement_rows):
    """Preserve the producer's Torch window means and ordered NumPy reductions."""
    import numpy as np
    rows = [torch.as_tensor(row) for row in kl_rows]
    agreements = [torch.as_tensor(row) for row in agreement_rows]
    if not rows or len(rows) != len(agreements) or rows[0].ndim != 1 or not rows[0].numel():
        raise ValueError("G3 summary requires a nonempty paired window population")
    shape = rows[0].shape
    if any(row.dtype != torch.float64 or row.shape != shape for row in rows):
        raise ValueError("G3 KL array geometry or float64 dtype differs")
    if any(row.dtype != torch.bool or row.shape != shape for row in agreements):
        raise ValueError("G3 agreement array geometry or Boolean dtype differs")
    values = np.stack([row.detach().cpu().numpy() for row in rows])
    if not np.isfinite(values).all():
        raise ValueError("G3 owned KL data must be finite")
    agreement_values = np.stack([row.detach().cpu().numpy() for row in agreements])
    window_means = [float(row.mean()) for row in rows]
    window_agreements = [float(row.double().mean()) for row in agreements]
    allk = np.concatenate(list(values)).astype(np.float64)
    metrics = {"mean_kl": float(allk.mean()), "p99_kl": float(np.quantile(allk, 0.99)),
               "max_kl": float(allk.max()), "top1_agreement": float(np.mean(window_agreements))}
    return metrics, values, agreement_values, window_means, window_agreements

