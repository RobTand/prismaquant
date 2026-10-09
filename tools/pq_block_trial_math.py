"""Numerical core for the opt-in per-block research trial.

Only FIT moments enter encoding or prices. HELDOUT moments enter the final
assembled-weight error. This module neither captures activations nor implements
an encoder: parent rungs use Tessera's existing ActivationSource/export path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from prismaquant.digests import bytes_sha256hex


def write_trial_json(path: Path, document: dict) -> None:
    """Write strict ASCII JSON in insertion order with one final newline."""
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")


def validate_sample_split(fit_samples, heldout_samples) -> dict:
    """Validate the supplied original sample indices; actual moment counts are separate."""
    fit = np.asarray(fit_samples)
    held = np.asarray(heldout_samples)
    for name, values, lo, hi in (("FIT", fit, 0, 384), ("HELDOUT", held, 384, 512)):
        if values.ndim != 1 or values.dtype.kind not in "iu" or values.size == 0:
            raise ValueError(f"{name}: nonempty integer sample-index vector required")
        if np.any(values < lo) or np.any(values >= hi):
            raise ValueError(f"{name}: sample index outside approved [{lo},{hi}) range")
    if np.intersect1d(fit, held).size:
        raise ValueError("FIT and HELDOUT original samples overlap")
    return {"fit_index_entries": int(fit.size), "heldout_index_entries": int(held.size),
            "fit_distinct_samples": int(np.unique(fit).size),
            "heldout_distinct_samples": int(np.unique(held).size)}


def validate_moment(h: torch.Tensor, count: int, columns: int, role: str) -> None:
    if type(count) is not int or count <= 0:
        raise ValueError(f"{role}: positive actual unit row count required")
    if h.shape != (columns, columns) or not h.dtype.is_floating_point:
        raise ValueError(f"{role}: moment does not match the input-column basis")
    if not bool(torch.isfinite(h).all()):
        raise ValueError(f"{role}: nonfinite moment")
    if not torch.allclose(h, h.t(), rtol=1e-5, atol=1e-6):
        raise ValueError(f"{role}: asymmetric second moment")
    if bool((h.diagonal() < 0).any()):
        raise ValueError(f"{role}: negative moment diagonal")


def mean_output_error(source: torch.Tensor, candidate: torch.Tensor,
                      h: torch.Tensor, count: int) -> float:
    """SSE summed over output features, averaged over actual calibration rows."""
    if source.shape != candidate.shape or source.ndim != 2:
        raise ValueError("source and candidate weight geometry differs")
    validate_moment(h, count, source.shape[1], "scoring")
    error = (candidate - source).double()
    metric = h.to(device=error.device, dtype=torch.float64)
    value = (error @ metric * error).sum() / count
    if not bool(torch.isfinite(value)) or value < 0:
        raise ValueError("invalid assembled output-error quadratic")
    return float(value)


def conditional_fit_prices(source: torch.Tensor, baseline: torch.Tensor,
                           candidates: Sequence[torch.Tensor], h_fit: torch.Tensor,
                           fit_count: int, block_rows: int, block_cols: int) -> np.ndarray:
    """Exact isolated replacement deltas around the complete baseline residual.

    These conditional FIT prices contain the baseline cross-block term. Their
    sum is still NOT the complete loss of a jointly changed assignment.
    """
    if source.ndim != 2 or source.shape != baseline.shape or not candidates:
        raise ValueError("nonempty same-shape source, baseline and candidate bank required")
    rows, cols = source.shape
    if block_rows <= 0 or block_cols <= 0 or rows % block_rows or cols % block_cols:
        raise ValueError("block geometry does not tile the projection")
    validate_moment(h_fit, fit_count, cols, "FIT")
    nr, nc = rows // block_rows, cols // block_cols
    e0 = (baseline - source).double()
    h = h_fit.to(device=e0.device, dtype=torch.float64)
    gradient = e0 @ h
    local_h = torch.stack([h[c:c + block_cols, c:c + block_cols]
                           for c in range(0, cols, block_cols)])
    prices = []
    for candidate in candidates:
        if candidate.shape != source.shape or candidate.device != baseline.device:
            raise ValueError("candidate bank has inconsistent shape or device")
        delta = (candidate - baseline).double()
        linear = 2 * (delta * gradient).reshape(nr, block_rows, nc, block_cols).sum(dim=(1, 3))
        blocks = delta.reshape(nr, block_rows, nc, block_cols).permute(0, 2, 1, 3).contiguous()
        quadratic = (blocks @ local_h * blocks).sum(dim=(-1, -2))
        price = (linear + quadratic) / fit_count
        if not bool(torch.isfinite(price).all()):
            raise ValueError("nonfinite conditional FIT price")
        prices.append(price.reshape(-1).cpu().numpy())
    return np.stack(prices, axis=1)


def block_groups(rows: int, cols: int, block_rows: int, block_cols: int,
                 input_group_cols: int = 32) -> np.ndarray:
    """One group per output block and input decode-chunk width."""
    if rows % block_rows or cols % block_cols or cols % input_group_cols \
            or input_group_cols % block_cols:
        raise ValueError("price-group geometry does not tile the projection")
    nr, nc = rows // block_rows, cols // block_cols
    per = input_group_cols // block_cols
    return (np.arange(nr)[:, None] * (nc // per) + np.arange(nc)[None, :] // per).reshape(-1)


def assemble_weight(candidates: Sequence[torch.Tensor], selection,
                    block_rows: int, block_cols: int) -> torch.Tensor:
    if not candidates or candidates[0].ndim != 2:
        raise ValueError("nonempty candidate weight bank required")
    rows, cols = candidates[0].shape
    if rows % block_rows or cols % block_cols:
        raise ValueError("assembly block geometry does not tile the projection")
    nr, nc = rows // block_rows, cols // block_cols
    for weight in candidates:
        if weight.shape != (rows, cols) or weight.device != candidates[0].device \
                or weight.dtype != candidates[0].dtype:
            raise ValueError("candidate bank has inconsistent shape, dtype or device")
    picked = np.asarray(selection)
    if picked.dtype.kind not in "iu" or picked.size != nr * nc \
            or np.any(picked < 0) or np.any(picked >= len(candidates)):
        raise ValueError("invalid candidate selection")
    picked = torch.as_tensor(picked.reshape(nr, nc), device=candidates[0].device, dtype=torch.long)
    bank = torch.stack(candidates).reshape(len(candidates), nr, block_rows, nc, block_cols)
    bank = bank.permute(1, 3, 0, 2, 4)
    ri = torch.arange(nr, device=picked.device)[:, None]
    ci = torch.arange(nc, device=picked.device)[None, :]
    blocks = bank[ri, ci, picked]
    return blocks.permute(0, 2, 1, 3).contiguous().reshape(rows, cols)


def encode_parent_bank(source: torch.Tensor, h_fit: torch.Tensor, fit_count: int,
                       fit_identity: Mapping, qname: str, rungs: Sequence[int],
                       admit: Callable[[int], Mapping], structure: str,
                       out_dir: str | Path, device: str = "cuda") -> dict:
    """Use the existing producer recipe; canonical admission precedes encoding."""
    from tessera.alphabet import E4M3_GRID
    from tessera.export import ActivationSource, encode_linear_planes, served_recipe
    from tessera.manifest import RotationState

    if source.ndim != 2 or source.dtype != torch.bfloat16:
        raise ValueError("parent encoding requires actual BF16 source weights")
    validate_moment(h_fit, fit_count, source.shape[1], "FIT")
    if fit_identity.get("hessian_role") != "fit":
        raise ValueError("only an explicitly FIT capture may shape parent encodings")
    if len(set(rungs)) != len(rungs) or 1024 not in rungs:
        raise ValueError("unique parent bank must include the uniform A8S control")
    decisions = []
    for rung in rungs:
        if type(rung) is not int:
            raise ValueError("integer q256 rung required")
        decision = dict(admit(rung))
        if decision.get("status") != "allow" or decision.get("rung") != rung:
            raise ValueError(f"canonical admission refuses rung {rung}: {decision}")
        decisions.append(decision)
    weight = source.to(device=device)
    h = h_fit.to(device=device, dtype=torch.float32)
    activation = ActivationSource(hessians={qname: h}, provenance=dict(fit_identity))
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    parents = []
    kwargs = None
    for rung, decision in zip(rungs, decisions):
        recipe = served_recipe(E4M3_GRID, rung, structure)
        if kwargs is None:
            kwargs = activation.for_unit(qname + ".weight", weight.shape[1], device,
                scale_plane=recipe.scale_plane, rotation=RotationState.NONE,
                with_diagonals=False, weight=weight)
        exported, unit, forests = encode_linear_planes(weight, grid=E4M3_GRID,
            q256=rung, name=f"TESSERA_E4M3_K1_R{rung}", verify=True,
            body=recipe.body, span=recipe.span, scale_plane=recipe.scale_plane,
            window_bits=recipe.window_bits, window_seed=recipe.window_seed,
            window_sigma=recipe.window_sigma, channel_sigma=recipe.channel_sigma,
            rotation=RotationState.NONE, with_diagonals=False, **kwargs)
        path = root / f"parent-R{rung}.tessera"
        path.write_bytes(exported.blob)
        parents.append({"rung": rung, "path": str(path), "bytes": len(exported.blob),
                        "sha256": bytes_sha256hex(exported.blob),
                        "admission": decision})
        del unit, forests, exported
    result = {"schema": "prismaquant.block_trial_parent_bank.v1", "qname": qname,
              "shape": list(source.shape), "fit_count": fit_count,
              "fit_identity": dict(fit_identity), "parents": parents,
              "uniform_control_rung": 1024,
              "heldout_consumed": False, "production_admission_claimed": False}
    write_trial_json(root / "parent-bank.json", result)
    return result


def evaluate_packed_candidate(source: torch.Tensor, baseline: torch.Tensor,
                              assembled: torch.Tensor, decoded: torch.Tensor,
                              h_fit: torch.Tensor, fit_count: int,
                              h_heldout: torch.Tensor, heldout_count: int,
                              candidate_bytes: int, baseline_bytes: int) -> dict:
    if candidate_bytes != baseline_bytes:
        raise ValueError("candidate and uniform A8S serialized bytes differ")
    if assembled.shape != decoded.shape or not torch.equal(assembled.cpu(), decoded.cpu()):
        raise ValueError("reference wire does not reconstruct the chosen fragments exactly")
    fit_base = mean_output_error(source, baseline, h_fit, fit_count)
    fit_mix = mean_output_error(source, decoded, h_fit, fit_count)
    held_base = mean_output_error(source, baseline, h_heldout, heldout_count)
    held_mix = mean_output_error(source, decoded, h_heldout, heldout_count)
    if fit_base <= 0 or held_base <= 0:
        raise ValueError("zero baseline error cannot supply a relative reduction gate")
    reduction = 1 - held_mix / held_base
    return {"fit_baseline_error": fit_base, "fit_candidate_error": fit_mix,
            "fit_reduction": 1 - fit_mix / fit_base,
            "heldout_baseline_error": held_base, "heldout_candidate_error": held_mix,
            "heldout_reduction": reduction,
            "baseline_bytes": baseline_bytes, "candidate_bytes": candidate_bytes,
            "fit_count": fit_count, "heldout_count": heldout_count,
            "heldout_used_for_selection": False, "gate_passes_10pct": reduction >= 0.10,
            "serialized_reference_roundtrip_exact": True,
            "not_a_served_G3_or_speed_qualification": True}
