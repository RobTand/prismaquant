"""Synchronous output-error energy; no activation retention and no normalization.

Reuse the G3 instrument's served quantizer owner. Products are FP32; sums
are FP64. An amplitude scales real residuals, not the quantizer input.
For dW and dX, joint residual(a) = a X dW^T + a dX W^T + a^2 dX dW^T.
The amplitude-two and amplitude-four joint energies are measured, not
assumed from amplitude one. Decoded weight references use BF16 before
local energy, except a canonical T16 pair, which keeps its BF16 values
and FP32 row scales separate through the FP32 dot and scale epilogue.
Activation dequantization stays FP32. Offline injection can add BF16
rounding after residual amplification.
"""
from __future__ import annotations

from canonical_weight import CanonicalWeight


def _freeze(value):
    import torch
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().reshape(-1)
        return ("tensor", str(value.dtype), tuple(value.shape), tuple(flat.tolist()))
    if isinstance(value, dict):
        return ("dict", tuple(sorted((key, _freeze(item)) for key, item in value.items())))
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_freeze(item) for item in value))
    try:
        hash(value)
    except TypeError:
        return ("repr", repr(value))
    return ("scalar", value)


def _scale_key(scale):
    if scale is None:
        return ("none", 0.0)
    return _freeze(scale)


def batched_option_sums(x, source_weight, entries, *, chunk_rows=256, resource_check=None):
    """Batched output-error energy for every actual option of one unit.

    Entries are (name, decoded_weight, contract, input_global_scale, tp_splits).
    Captured rows are read once per chunk. Each real A8/A4 perturbation is
    formed once per unit and activation contract. E_A is computed once per
    contract group and its actual value is reused across weight anchors.
    Every decoded plane still comes from its existing render owner. Products
    stay FP32 and sums stay FP64. The per-name result matches one
    output_error_sums call for that entry.
    """
    import torch
    from mixed_injection import quantized_rows_fp32
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("Row chunk must be a positive integer")
    entries = list(entries)
    if not entries:
        raise ValueError("Batched energy needs at least one actual option")
    names = [entry[0] for entry in entries]
    if len(set(names)) != len(names):
        raise ValueError("Batched energy repeats an option")
    if x.ndim != 2 or source_weight.ndim != 2:
        raise ValueError("Input/source geometry must agree as Linear planes")
    if x.shape[1] != source_weight.shape[1]:
        raise ValueError("Live input width differs from actual source/decoded weight")
    groups, canonicals = {}, []
    for name, decoded, contract, scale, tp in entries:
        canonical = decoded if isinstance(decoded, CanonicalWeight) else None
        if canonical is not None:
            if contract != "bf16_unquantized":
                raise ValueError("Canonical T16 components serve bf16_unquantized only")
            if tuple(canonical.shape) != tuple(source_weight.shape):
                raise ValueError("Input/source/decoded geometry must agree as Linear planes")
            if x.device != source_weight.device or x.device != canonical.device:
                raise ValueError("Live rows and prefetched source/decoded weights must share the measured device")
            if not bool(torch.isfinite(canonical.values.float()).all()):
                raise ValueError("The canonical value plane is not finite")
            if not bool(torch.isfinite(canonical.row_scales).all()):
                raise ValueError("The canonical row-scale plane is not finite")
            canonicals.append((name, decoded, contract, scale, tp))
            continue
        if decoded.shape != source_weight.shape:
            raise ValueError("Input/source/decoded geometry must agree as Linear planes")
        if x.device != source_weight.device or x.device != decoded.device:
            raise ValueError("Live rows and prefetched source/decoded weights must share the measured device")
        quantized_rows_fp32(x[:0], contract, input_global_scale=scale, tp_splits=tp)
        groups.setdefault((contract, _scale_key(scale), tp), []).append((name, decoded, contract, scale, tp))
    for name, decoded, contract, scale, tp in canonicals:
        quantized_rows_fp32(x[:0], contract, input_global_scale=scale, tp_splits=tp)
    totals = {name: torch.zeros((3, 3), dtype=torch.float64, device=x.device) for name in names}
    if x.shape[0]:
        source = source_weight.float()
        deltas = {}
        for members in groups.values():
            for name, decoded, _contract, _scale, _tp in members:
                deltas[name] = decoded.float() - source
        tiles = {name: (decoded.values.float(), decoded.row_scales.float()) for name, decoded, _c, _s, _t in canonicals}
        with torch.inference_mode():
            for start in range(0, len(x), chunk_rows):
                if resource_check is not None:
                    resource_check("energy row chunk")
                rows = x[start:start + chunk_rows].float()
                for (contract, _key, tp), members in groups.items():
                    scale = members[0][3]
                    if contract in ("bf16_unquantized", "source_bf16"):
                        for name, _decoded, _c, _s, _t in members:
                            w = rows @ deltas[name].T
                            w2, w4 = 2 * w, 4 * w
                            ew = w.double().square().sum()
                            totals[name][0, 0] += ew
                            totals[name][0, 2] += ew
                            totals[name][1, 0] += w2.double().square().sum()
                            totals[name][1, 2] += w2.double().square().sum()
                            totals[name][2, 0] += w4.double().square().sum()
                            totals[name][2, 2] += w4.double().square().sum()
                        continue
                    q = quantized_rows_fp32(rows, contract, input_global_scale=scale, tp_splits=tp)
                    dx = q - rows
                    a = dx @ source.T
                    a_amps = (a, 2 * a, 4 * a)
                    e_a = [value.double().square().sum() for value in a_amps]
                    for name, _decoded, _c, _s, _t in members:
                        w = rows @ deltas[name].T
                        interaction = dx @ deltas[name].T
                        joint1 = w + a + interaction
                        joint2 = 2 * w + 2 * a + 4 * interaction
                        joint4 = 4 * w + 4 * a + 16 * interaction
                        totals[name][0, 0] += w.double().square().sum()
                        totals[name][1, 0] += (2 * w).double().square().sum()
                        totals[name][2, 0] += (4 * w).double().square().sum()
                        for index in range(3):
                            totals[name][index, 1] += e_a[index]
                        for index, value in enumerate((joint1, joint2, joint4)):
                            totals[name][index, 2] += value.double().square().sum()
                for name, _decoded, _contract, _scale, _tp in canonicals:
                    tile, row_scale = tiles[name]
                    decoded_out = (rows @ tile.T) * row_scale[None, :]
                    base = rows @ source.T
                    w = decoded_out - base
                    w2, w4 = 2 * w, 4 * w
                    totals[name][0, 0] += w.double().square().sum()
                    totals[name][0, 2] += w.double().square().sum()
                    totals[name][1, 0] += w2.double().square().sum()
                    totals[name][1, 2] += w2.double().square().sum()
                    totals[name][2, 0] += w4.double().square().sum()
                    totals[name][2, 2] += w4.double().square().sum()
    results = {}
    for name, decoded, contract, _scale, _tp in entries:
        sums = totals[name]
        if not bool(torch.isfinite(sums).all()):
            raise ValueError("Nonfinite measured output energy")
        values_out = sums.cpu().tolist()
        if isinstance(decoded, CanonicalWeight):
            rounding = ("Canonical T16 pair: BF16 values and FP32 row scales stay separate "
                        "through the FP32 dot and scale epilogue; no BF16 fold. Activation "
                        "dequantization and local products use FP32.")
        else:
            rounding = "The decoded weight reference uses BF16. Activation dequantization and local products use FP32. Offline residual amplification can add BF16 rounding."
        results[name] = {"E_W_sum": values_out[0][0], "E_A_sum": values_out[0][1], "E_WA_sum": values_out[0][2],
            "E_W_sum_amp2": values_out[1][0], "E_A_sum_amp2": values_out[1][1], "E_WA_sum_amp2": values_out[1][2],
            "E_W_sum_amp4": values_out[2][0], "E_A_sum_amp4": values_out[2][1], "E_WA_sum_amp4": values_out[2][2],
            "amp2_measured": True, "amp4_measured": True, "routed_rows": len(x),
            "precision": "FP32 matmul (TF32 disabled); FP64 energy sums",
            "activation_backend": ("none" if contract in ("bf16_unquantized", "source_bf16")
                                   else "served native operator" if x.is_cuda else "CPU attested numerical oracle"),
            "rounding_note": rounding}
    return results


def output_error_sums(x, source_weight, decoded_weight, contract, *,
                      input_global_scale=None, tp_splits=1, chunk_rows=256,
                      resource_check=None):
    import torch
    from mixed_injection import quantized_rows_fp32
    canonical = decoded_weight if isinstance(decoded_weight, CanonicalWeight) else None
    if canonical is not None:
        if contract != "bf16_unquantized":
            raise ValueError("Canonical T16 components serve bf16_unquantized only")
        if x.ndim != 2 or source_weight.ndim != 2:
            raise ValueError("Input/source geometry must agree as Linear planes")
        if tuple(canonical.shape) != tuple(source_weight.shape):
            raise ValueError("Input/source/decoded geometry must agree as Linear planes")
        if x.shape[1] != source_weight.shape[1]:
            raise ValueError("Live input width differs from actual source/decoded weight")
        if x.device != source_weight.device or x.device != canonical.device:
            raise ValueError("Live rows and prefetched source/decoded weights must share the measured device")
        values_t = canonical.values
        scales_t = canonical.row_scales
        if not bool(torch.isfinite(values_t.float()).all()):
            raise ValueError("The canonical value plane is not finite")
        if not bool(torch.isfinite(scales_t).all()):
            raise ValueError("The canonical row-scale plane is not finite")
    else:
        if x.ndim != 2 or source_weight.ndim != 2 or decoded_weight.shape != source_weight.shape:
            raise ValueError("Input/source/decoded geometry must agree as Linear planes")
        if x.shape[1] != source_weight.shape[1]:
            raise ValueError("Live input width differs from actual source/decoded weight")
        if x.device != source_weight.device or x.device != decoded_weight.device:
            raise ValueError("Live rows and prefetched source/decoded weights must share the measured device")
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError("Row chunk must be a positive integer")
    quantized_rows_fp32(x[:0], contract, input_global_scale=input_global_scale, tp_splits=tp_splits)
    return batched_option_sums(x, source_weight,
        [("option", decoded_weight, contract, input_global_scale, tp_splits)],
        chunk_rows=chunk_rows, resource_check=resource_check)["option"]
