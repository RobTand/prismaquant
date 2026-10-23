"""Canonical T16 weight components: values and row scales kept separate.

The T16 production path runs the stock BF16 GEMM on the code tile and
applies the per-output-row scale in FP32 as an output epilogue, because a
CHANNEL scale commutes with the matmul. Folding the scale into one BF16
tile adds a rounding of ``s_i * t_ik`` that the encoder never scored.
This module is the reader for that path: it returns the pair that the
existing Tessera parse/decode owners produce and projects rows through
the same FP32 dot plus scale epilogue. It owns no codec.
"""
from __future__ import annotations


class CanonicalWeight:
    """Decoded T16 pair: BF16 tile plus FP32 per-row scales.

    ``values`` is ``[out, in]`` BF16, ``row_scales`` is ``[out]`` FP32.
    The two stay separate until :meth:`project` combines them in FP32.
    """

    def __init__(self, values, row_scales):
        import torch
        if not isinstance(values, torch.Tensor) or not isinstance(row_scales, torch.Tensor):
            raise ValueError("Canonical T16 components must be tensors")
        if values.ndim != 2 or values.dtype != torch.bfloat16:
            raise ValueError("Canonical values must be a BF16 [out, in] plane")
        if row_scales.ndim != 1 or row_scales.dtype != torch.float32:
            raise ValueError("Canonical row scales must be an FP32 [out] plane")
        if values.shape[0] != row_scales.shape[0]:
            raise ValueError("Canonical values and row scales disagree on rows")
        if values.device != row_scales.device:
            raise ValueError("Canonical values and row scales must share one device")
        self.values = values
        self.row_scales = row_scales

    @property
    def shape(self):
        return tuple(self.values.shape)

    @property
    def device(self):
        return self.values.device

    def storage_evidence(self):
        """Tensor storage facts: shapes, dtypes, device, resident bytes."""
        values = self.values
        scales = self.row_scales
        values_bytes = values.numel() * values.element_size()
        scales_bytes = scales.numel() * scales.element_size()
        return {
            "values_shape": [int(n) for n in values.shape],
            "values_dtype": str(values.dtype),
            "row_scales_shape": [int(n) for n in scales.shape],
            "row_scales_dtype": str(scales.dtype),
            "device": str(values.device),
            "values_bytes": int(values_bytes),
            "row_scales_bytes": int(scales_bytes),
            "resident_bytes": int(values_bytes + scales_bytes),
        }

    def project(self, rows):
        """Project live rows to FP32 outputs through the served epilogue.

        ``rows`` is ``[n, in]`` on this pair's device. Returns FP32
        ``[n, out]``: the FP32 dot against the BF16 tile, times the
        per-row scale on the output. No BF16 fold occurs anywhere.
        """
        import torch
        if not isinstance(rows, torch.Tensor) or rows.ndim != 2:
            raise ValueError("Live rows must be a two-dimensional tensor")
        if rows.shape[1] != self.values.shape[1]:
            raise ValueError("Live input width differs from canonical weight width")
        if rows.device != self.values.device:
            raise ValueError("Live rows must share the canonical weight device")
        with torch.inference_mode():
            dot = rows.float() @ self.values.float().T
            return dot * self.row_scales[None, :]


def read_canonical_weight(raw, device, expected_shape):
    """Read artifact bytes to a :class:`CanonicalWeight` on ``device``.

    Uses the existing Tessera parse/decode owners
    (``parse_unit_artifact`` plus ``materialize_bf16``) and adds no
    codec. Refuses unknown geometry, misshaped planes, and nonfinite
    values or scales. The finite checks run here, before any empty
    route can hide a bad plane behind zero rows.
    """
    import torch
    from tessera.unit_artifact import parse_unit_artifact
    from tessera.decode import materialize_bf16
    shape = tuple(expected_shape)
    if len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("Expected canonical geometry must be positive integral [out, in]")
    if isinstance(device, torch.device):
        target = device
    else:
        target = torch.device(str(device))
    parsed = parse_unit_artifact(bytes(raw), device=target)
    actual = (int(parsed.manifest.geometry.rows), int(parsed.manifest.geometry.columns))
    if actual != shape:
        raise ValueError("Actual unit geometry differs from requested Linear geometry: "
                         + str(actual) + " versus " + str(shape))
    values, scales = materialize_bf16(parsed.unit, parsed.forests, parsed.code)
    values = values.to(target)
    scales = scales.to(target, torch.float32).reshape(-1)
    if tuple(values.shape) != shape or tuple(scales.shape) != (shape[0],):
        raise ValueError("Canonical planes disagree with the requested Linear geometry")
    if not bool(torch.isfinite(values.float()).all()):
        raise ValueError("The canonical value plane is not finite")
    if not bool(torch.isfinite(scales).all()):
        raise ValueError("The canonical row-scale plane is not finite")
    return CanonicalWeight(values, scales)
