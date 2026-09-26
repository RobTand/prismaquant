"""Pilot-scoped monotone min-chain helpers for product CB encodes.

This module deliberately is not wired into the production encoder.  Calling
the candidate builders requires ``PRISMAQUANT_CB_MINCHAIN_PILOT=1``; the DSV4
campaign pilot is the only current caller.  A selected rung remains an
ordinary flat product-CB payload, so the decoder and serving ABI do not gain a
chain dependency.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from typing import Any

import torch

from .cb_layout import VEC_DIM, codebook_subtable_shapes, family_for
from .nvfp4_cb_formats import (
    FP8_ELEMENT_MAX,
    _snap_to_grid,
    ldlq_reassign_cb_fields,
    nvfp4_cb_fields,
)


MINCHAIN_SCHEMA = "prismaquant.cb_minchain.v1"
MINCHAIN_CONTEXT_VERSION = "minchain-pilot-v1"
MINCHAIN_FLAG = "PRISMAQUANT_CB_MINCHAIN_PILOT"


def require_pilot_enabled() -> None:
    raw = os.environ.get(MINCHAIN_FLAG, "0").strip().lower()
    if raw not in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            f"pilot min-chain execution requires {MINCHAIN_FLAG}=1"
        )


def _tensor_digest(hasher: Any, name: str, value: torch.Tensor) -> None:
    tensor = value.detach().contiguous().cpu()
    header = {
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }
    hasher.update(json.dumps(header, sort_keys=True).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(tensor.view(torch.uint8).numpy().tobytes())
    hasher.update(b"\0")


def solution_digest(fields: Mapping[str, Any]) -> str:
    """Digest the serialized solution planes and resolved flat codebook."""
    hasher = hashlib.sha256()
    hasher.update(MINCHAIN_SCHEMA.encode("ascii"))
    shape = fields.get("shape")
    if shape is not None:
        hasher.update(json.dumps({
            "logical_shape": [int(dim) for dim in shape]
        }, sort_keys=True).encode("utf-8"))
        hasher.update(b"\0")
    for key in ("indices", "scales", "signs", "scale_super", "scale_sub"):
        value = fields.get(key)
        if isinstance(value, torch.Tensor):
            _tensor_digest(hasher, key, value)
    codebook = fields.get("codebook")
    if isinstance(codebook, torch.Tensor):
        _tensor_digest(hasher, "codebook", codebook)
    elif codebook is not None:
        for index, table in enumerate(codebook):
            _tensor_digest(hasher, f"codebook.{index}", table)
    return hasher.hexdigest()


def chain_identity(
    *, winning_arm: str, solution: Mapping[str, Any],
    predecessor_digest: str | None,
) -> dict[str, Any]:
    if winning_arm not in {"free", "embed", "refine"}:
        raise ValueError(f"unknown min-chain arm {winning_arm!r}")
    if winning_arm in {"embed", "refine"} and not predecessor_digest:
        raise ValueError(f"{winning_arm} requires a predecessor digest")
    if winning_arm == "free" and predecessor_digest is not None:
        raise ValueError("free arm cannot claim a predecessor digest")
    return {
        "schema": MINCHAIN_SCHEMA,
        "chain_version": MINCHAIN_CONTEXT_VERSION,
        "winning_arm": winning_arm,
        "predecessor_digest": predecessor_digest,
        "solution_digest": solution_digest(solution),
    }


def recipe_solution_digest(recipe: Mapping[str, Any]) -> str:
    """Digest a deterministic encode recipe without rereading tensor planes.

    Large campaigns use this identity after source/imatrix/activation content
    has already been fail-closed at load.  It keeps bookkeeping off the hot
    path while still changing on every input, context, arm, predecessor, rung,
    or algorithm-version change.
    """
    payload = {
        "schema": MINCHAIN_SCHEMA,
        "chain_version": MINCHAIN_CONTEXT_VERSION,
        "deterministic_recipe": dict(recipe),
    }
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def chain_identity_from_digest(
    *, winning_arm: str, solution_digest_value: str,
    predecessor_digest: str | None,
) -> dict[str, Any]:
    if winning_arm not in {"free", "embed", "refine"}:
        raise ValueError(f"unknown min-chain arm {winning_arm!r}")
    if winning_arm in {"embed", "refine"} and not predecessor_digest:
        raise ValueError(f"{winning_arm} requires a predecessor digest")
    if winning_arm == "free" and predecessor_digest is not None:
        raise ValueError("free arm cannot claim a predecessor digest")
    digest = str(solution_digest_value).lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError("solution digest must be lowercase sha256 hex")
    return {
        "schema": MINCHAIN_SCHEMA,
        "chain_version": MINCHAIN_CONTEXT_VERSION,
        "winning_arm": winning_arm,
        "predecessor_digest": predecessor_digest,
        "solution_digest": digest,
        "digest_basis": "deterministic_content_gated_recipe",
    }


def _expanded_codebook(
    predecessor: Mapping[str, Any], target_k: int,
    *, grid: str, mode: str,
) -> tuple[torch.Tensor, ...]:
    if mode != "product":
        raise ValueError("the min-chain pilot supports product CB only")
    old = tuple(predecessor["codebook"])
    n_sub = family_for(grid, mode).n_sub
    expected = codebook_subtable_shapes(target_k, mode, n_sub)
    if len(old) != len(expected):
        raise ValueError("predecessor subtable count differs from target")
    expanded = []
    for table, shape in zip(old, expected):
        if table.ndim != 2 or table.shape[1] != shape[1]:
            raise ValueError("predecessor subtable dimension differs")
        if table.shape[0] > shape[0]:
            raise ValueError("target rung is smaller than predecessor")
        if table.shape[0] == shape[0]:
            expanded.append(table.clone())
            continue
        padding = table[-1:].expand(shape[0] - table.shape[0], -1).clone()
        expanded.append(torch.cat((table, padding), dim=0))
    return tuple(expanded)


def embed_predecessor(
    predecessor: Mapping[str, Any], target_k: int,
    *, grid: str = "fp8", mode: str = "product",
) -> dict[str, Any]:
    """Embed a finished solution verbatim, padding only the flat tables."""
    require_pilot_enabled()
    out = dict(predecessor)
    out["codebook"] = _expanded_codebook(
        predecessor, target_k, grid=grid, mode=mode
    )
    return out


def _scaled_vectors(
    weight: torch.Tensor, scales: torch.Tensor, *, grid: str,
) -> torch.Tensor:
    if weight.ndim != 2:
        raise ValueError("refine_one_entry expects one 2-D slice")
    if grid != "fp8":
        raise ValueError("the registered pilot refines FP8-CB only")
    per_element = scales.to(torch.float32)
    if per_element.shape[-1] == 1:
        per_element = per_element.expand_as(weight)
    return (weight.float() / per_element.clamp_min(1e-12)).reshape(-1, VEC_DIM)


def _refine_new_centroid(
    vectors: torch.Tensor,
    weights: torch.Tensor,
    old_table: torch.Tensor,
    *, iterations: int,
) -> torch.Tensor:
    # The candidate starts at the currently worst represented vector, then
    # competes against the frozen predecessor entries for at most three Lloyd
    # updates.  All old entries remain byte-verbatim.
    # Padding duplicates are semantically one competitor.  Deduplicating only
    # the distance evaluator keeps the emitted predecessor prefix untouched
    # while preventing repeated add-one steps from becoming exponential work.
    old_table = torch.unique(old_table, dim=0)
    chunk_rows = 1 << 15
    worst_values = []
    worst_vectors = []
    for start in range(0, vectors.shape[0], chunk_rows):
        stop = min(start + chunk_rows, vectors.shape[0])
        local_vectors = vectors[start:stop]
        local_weights = weights[start:stop]
        old_best = (
            (local_vectors[:, None, :] - old_table[None, :, :]).square()
            * local_weights[:, None, :]
        ).sum(-1).min(-1).values
        value, index = old_best.max(0)
        worst_values.append(value)
        worst_vectors.append(local_vectors[index])
    worst_chunk = torch.stack(worst_values).argmax()
    centroid = torch.stack(worst_vectors)[worst_chunk].clone()
    centroid = _snap_to_grid(centroid, "fp8").clamp(
        -FP8_ELEMENT_MAX, FP8_ELEMENT_MAX
    )
    for _ in range(iterations):
        numerator = torch.zeros_like(centroid)
        denominator = torch.zeros_like(centroid)
        selected_count = torch.zeros((), device=vectors.device, dtype=torch.int64)
        for start in range(0, vectors.shape[0], chunk_rows):
            stop = min(start + chunk_rows, vectors.shape[0])
            local_vectors = vectors[start:stop]
            local_weights = weights[start:stop]
            old_best = (
                (local_vectors[:, None, :] - old_table[None, :, :]).square()
                * local_weights[:, None, :]
            ).sum(-1).min(-1).values
            new_distance = (
                (local_vectors - centroid).square() * local_weights
            ).sum(-1)
            selected = new_distance < old_best
            chosen_weights = local_weights[selected]
            numerator += (local_vectors[selected] * chosen_weights).sum(0)
            denominator += chosen_weights.sum(0)
            selected_count += selected.sum()
        if int(selected_count.item()) == 0:
            break
        updated = _snap_to_grid(
            numerator / denominator.clamp_min(1e-30), "fp8"
        ).clamp(
            -FP8_ELEMENT_MAX, FP8_ELEMENT_MAX
        )
        if torch.equal(updated, centroid):
            break
        centroid = updated
    return centroid


def refine_one_entry(
    weight: torch.Tensor,
    predecessor: Mapping[str, Any],
    target_k: int,
    *,
    col_weights: torch.Tensor,
    activation_rows: torch.Tensor,
    grid: str = "fp8",
    mode: str = "product",
    iterations: int = 3,
) -> dict[str, Any]:
    """Cheap add-one-entry arm with frozen old tables and shared scales."""
    require_pilot_enabled()
    if not 0 <= int(iterations) <= 3:
        raise ValueError("min-chain refinement permits at most 3 iterations")
    embedded = embed_predecessor(
        predecessor, target_k, grid=grid, mode=mode
    )
    vectors = _scaled_vectors(weight, predecessor["scales"], grid=grid)
    weights = torch.broadcast_to(
        col_weights.to(device=weight.device, dtype=torch.float32), weight.shape
    )
    weights = weights.reshape(-1, VEC_DIM)
    sub_dim = VEC_DIM // len(embedded["codebook"])
    refined_tables = []
    for sub_index, (old_table, padded) in enumerate(zip(
        predecessor["codebook"], embedded["codebook"]
    )):
        old_count = old_table.shape[0]
        if old_count == padded.shape[0]:
            refined_tables.append(padded)
            continue
        sl = slice(sub_index * sub_dim, (sub_index + 1) * sub_dim)
        centroid = _refine_new_centroid(
            vectors[:, sl], weights[:, sl], old_table.float(),
            iterations=int(iterations),
        )
        tail = centroid.reshape(1, -1).expand(
            padded.shape[0] - old_count, -1
        ).clone()
        refined_tables.append(torch.cat((old_table, tail), dim=0))
    warm = {"scales": predecessor["scales"]}
    fields = nvfp4_cb_fields(
        weight,
        target_k,
        grid=grid,
        mode=mode,
        col_weights=col_weights,
        codebook=tuple(refined_tables),
        scale_sweep=True,
        warm_scale_state=warm,
        encode_tier="balanced",
    )
    fields = ldlq_reassign_cb_fields(
        weight,
        fields,
        col_weights,
        activation_rows,
        grid=grid,
        mode=mode,
    )
    if not torch.equal(fields["scales"], predecessor["scales"]):
        raise AssertionError("min-chain shared-scale refinement changed scales")
    return fields


def relative_epsilon(a: float, b: float, *, rtol: float = 1e-12) -> float:
    """Registered symmetric relative epsilon for min-chain comparisons."""
    return float(rtol) * max(abs(float(a)), abs(float(b)))


def epsilon_le(a: float, b: float, *, rtol: float = 1e-12) -> bool:
    """Return ``a <= b`` under the campaign's registered epsilon semantics."""
    return float(a) <= float(b) + relative_epsilon(a, b, rtol=rtol)


def select_arm(
    errors: Mapping[str, float], *, rtol: float = 1e-12,
) -> tuple[str, float]:
    """Select deterministically, treating epsilon-close values as ties.

    ``free`` has identity-stability priority, followed by ``embed`` and the
    legacy pilot-only ``refine`` arm.  The optimized A-FAST path supplies only
    ``free`` and ``embed``; keeping the optional third arm here preserves the
    immutable pilot-1 reader without putting refine back into the campaign.
    """
    if "free" not in errors:
        raise ValueError("min-chain selection requires the free arm")
    unknown = set(errors).difference({"free", "embed", "refine"})
    if unknown:
        raise ValueError(f"unknown min-chain arms: {sorted(unknown)}")
    order = tuple(arm for arm in ("free", "embed", "refine") if arm in errors)
    winner = "free"
    best = float(errors[winner])
    for arm in order[1:]:
        value = float(errors[arm])
        # A difference inside the registered epsilon is a tie.  Earlier arms
        # win ties, making exact and near-exact free identities stable.
        if value < best - relative_epsilon(value, best, rtol=rtol):
            winner, best = arm, value
    return winner, best


__all__ = [
    "MINCHAIN_CONTEXT_VERSION",
    "MINCHAIN_FLAG",
    "MINCHAIN_SCHEMA",
    "chain_identity",
    "chain_identity_from_digest",
    "epsilon_le",
    "embed_predecessor",
    "refine_one_entry",
    "recipe_solution_digest",
    "relative_epsilon",
    "require_pilot_enabled",
    "select_arm",
    "solution_digest",
]
