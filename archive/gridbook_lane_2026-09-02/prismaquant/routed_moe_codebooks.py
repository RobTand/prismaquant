"""Producer-side book plan for learned routed-MoE codebooks.

Gridbook keeps the physical expert checkpoint tensors fused as ``w13``
(``gate_up_proj``).  Two producer spellings are legal against it, and which one
an artifact uses is decided by how its books were *burned*:

* **stack keying (campaign rule R1, the default).**  One book per
  ``(layer, stack, rung)`` — gate and up pooled — so the fused weight names a
  single ``codebook_ref`` on the packed target, the same spelling a lattice
  layer uses.
* **role keying (the pre-R1 form, opt-in).**  One book per
  ``(layer, projection, rung)``.  A packed target binds exactly one
  ``codebook_ref``, so a per-role layer *must* name the ``gate_proj`` and
  ``up_proj`` halves separately; Gridbook 0.8.4+ resolves those independently.

``down_proj`` is a one-projection stack, so the two keyings give it the same
book name either way.

This module owns that small producer naming bridge plus the structural
split-book predicate the exporter gates on; it contains no Gridbook runtime
code and asserts nothing about what a runtime does with either spelling.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch


# Immutable coverage of the completed DSv4 pooled-Lloyd burn.  Candidate
# legality remains the allocator/source-bpp policy's responsibility.
ROUTED_MOE_CBL_BANK_RUNGS = frozenset(range(28, 34))
_FORMAT_GROUP_PREFIX = "format_group_"

ROUTED_BOOK_KEYING_STACK = "stack"
ROUTED_BOOK_KEYING_ROLE = "role"
ROUTED_BOOK_KEYINGS = (ROUTED_BOOK_KEYING_STACK, ROUTED_BOOK_KEYING_ROLE)
# Campaign rule R1: routed learned books are burned per (layer, stack, rung).
DEFAULT_ROUTED_BOOK_KEYING = ROUTED_BOOK_KEYING_STACK

ROUTED_ROLE_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
ROUTED_STACK_KEYS = ("gate_up_proj", "down_proj")
# The vocabulary a burn selection / bank shard may use in its ``projection``
# member.  Under role keying it names one projection; under stack keying it
# names the packed parent the book was pooled over.
ROUTED_BOOK_KEY_NAMES = frozenset(ROUTED_ROLE_PROJECTIONS) | frozenset(
    ROUTED_STACK_KEYS
)


def normalize_routed_book_keying(value: object) -> str:
    """Return a canonical routed-book keying, refusing anything else."""

    keying = str(value).strip().lower()
    if keying not in ROUTED_BOOK_KEYINGS:
        raise ValueError(
            f"routed book keying must be one of {list(ROUTED_BOOK_KEYINGS)}, "
            f"got {value!r}"
        )
    return keying


def routed_book_key(
    packed_leaf: str,
    projection: str,
    *,
    keying: str,
) -> str:
    """Selection/bank ``projection`` spelling for one book under *keying*.

    Under stack keying that is the packed parent leaf (``gate_up_proj``); under
    role keying it is the projection itself.
    """

    keying = normalize_routed_book_keying(keying)
    projection = str(projection)
    if projection not in ROUTED_ROLE_PROJECTIONS:
        raise ValueError(f"unsupported routed expert projection {projection!r}")
    if keying == ROUTED_BOOK_KEYING_ROLE:
        return projection
    leaf = str(packed_leaf)
    if leaf not in ROUTED_STACK_KEYS:
        raise ValueError(
            f"{leaf!r} is not a routed packed-expert stack leaf "
            f"{list(ROUTED_STACK_KEYS)}"
        )
    return leaf


@dataclass(frozen=True)
class RoutedMoECodebookRole:
    """One logical book applied to a contiguous physical expert-row segment."""

    projection: str
    qname: str
    ref: str
    format_name: str
    codebook: Any
    col_weights: torch.Tensor
    output_rows: int
    member_qnames: tuple[str, ...]


def _packed_parent_and_suffix(packed_qname: str) -> tuple[str, str | None]:
    """Split an optional per-expert format-group discriminator.

    The split-stack producer names physical tensors as
    ``...gate_up_proj.format_group_fp8_cb_k28``.  Gridbook's logical role
    target preserves that final discriminator while replacing the packed
    parent immediately before it.
    """

    packed = str(packed_qname)
    if "." not in packed:
        return packed, None
    parent, leaf = packed.rsplit(".", 1)
    if leaf.startswith(_FORMAT_GROUP_PREFIX):
        return parent, leaf
    return packed, None


def logical_role_qname(packed_qname: str, projection: str) -> str:
    """Replace a packed expert leaf with one ordinary logical projection.

    An optional ``format_group_*`` suffix is retained.  This is the exact
    target namespace consumed by Gridbook's split routed-MoE declaration.
    """

    packed = str(packed_qname)
    projection = str(projection)
    packed_parent, discriminator = _packed_parent_and_suffix(packed)
    if (
        ".experts." not in packed_parent
        or not packed_parent.rsplit(".", 1)[-1]
    ):
        raise ValueError(f"{packed_qname!r} is not a routed expert target")
    if projection not in {"gate_proj", "up_proj", "down_proj"}:
        raise ValueError(f"unsupported routed expert projection {projection!r}")
    logical = packed_parent.rsplit(".", 1)[0] + "." + projection
    return (
        logical
        if discriminator is None
        else logical + "." + discriminator
    )


def bundle_role_qname(packed_qname: str, projection: str) -> str:
    """Role-keyed cell name in the immutable bundle, without a subgroup.

    Under role keying one learned book is trained per
    ``(layer, projection, rung)`` over the complete expert population.  A split
    format subgroup therefore reuses the same bundle cell and validates the
    participating experts through its per-member aliases; the format-group
    discriminator belongs only to the artifact/runtime target namespace.
    """

    packed_parent, _discriminator = _packed_parent_and_suffix(packed_qname)
    return logical_role_qname(packed_parent, projection)


def bundle_stack_qname(packed_qname: str) -> str:
    """Stack-keyed cell name in the immutable bundle, without a subgroup.

    Under stack keying one learned book is trained per ``(layer, stack, rung)``
    over both projections of the fused stack, so the cell is named by the
    packed parent itself — the same name the artifact's config target carries.
    """

    packed_parent, _discriminator = _packed_parent_and_suffix(packed_qname)
    if ".experts." not in packed_parent:
        raise ValueError(f"{packed_qname!r} is not a routed expert target")
    leaf = packed_parent.rsplit(".", 1)[-1]
    if leaf not in ROUTED_STACK_KEYS:
        raise ValueError(
            f"{packed_qname!r} does not name a routed packed-expert stack "
            f"{list(ROUTED_STACK_KEYS)}"
        )
    return packed_parent


def bundle_book_qname(
    packed_qname: str,
    projection: str,
    *,
    keying: str,
) -> str:
    """Bundle cell name for one routed book under *keying*."""

    keying = normalize_routed_book_keying(keying)
    if keying == ROUTED_BOOK_KEYING_STACK:
        return bundle_stack_qname(packed_qname)
    return bundle_role_qname(packed_qname, projection)


def fused_targets_with_split_books(
    books_by_fused_target: Mapping[str, Mapping[str, Sequence[str] | str]],
) -> dict[str, tuple[str, ...]]:
    """Return the fused targets whose scheme would name more than one book.

    The input maps one physical fused weight to the codebook reference each of
    its roles would name.  The predicate is purely structural — it counts
    distinct references a producer is about to write — so it is a fact about
    the artifact, never a claim about what a serving runtime does with it.
    """

    split: dict[str, tuple[str, ...]] = {}
    for raw_target, refs_by_role in books_by_fused_target.items():
        distinct = {
            (
                (str(ref),)
                if isinstance(ref, str)
                else tuple(str(name) for name in ref)
            )
            for ref in refs_by_role.values()
        }
        if len(distinct) > 1:
            split[str(raw_target)] = tuple(
                sorted(",".join(names) for names in distinct)
            )
    return dict(sorted(split.items()))


def describe_split_book_refusal(
    split: Mapping[str, Sequence[str]],
    *,
    override_flag: str = "--allow-per-role-books",
) -> str:
    """Human-facing message for a fused weight that names several books."""

    lines = [
        f"  {target}: {len(books)} books -> " + "; ".join(books)
        for target, books in sorted(split.items())
    ]
    return (
        f"refusing to export {len(split)} fused routed weight(s) whose scheme "
        "names more than one codebook:\n"
        + "\n".join(lines)
        + "\n"
        "Campaign rule R1: routed learned books are burned per "
        "(layer, stack, rung), so a fused gate/up stack names ONE book. Books "
        "split per role make gate and up resolve to different tables, which "
        "cannot attest the pinned runtime's persistent-B FP8 fast lane for "
        "these layers. Re-burn the stack pooled (build_cb_learned_bundle "
        f"--routed-book-keying stack), or pass {override_flag} to ship the "
        "split books knowingly; the acknowledgement is stamped on the "
        "shipcard."
    )


def learned_role_qnames_for_packed(packed_qname: str) -> tuple[str, ...]:
    """Canonical Gridbook logical roles for a physical expert stack."""

    packed = str(packed_qname)
    packed_parent, _discriminator = _packed_parent_and_suffix(packed)
    if ".experts." not in packed_parent:
        return ()
    leaf = packed_parent.rsplit(".", 1)[-1]
    if leaf == "gate_up_proj":
        return (
            logical_role_qname(packed, "gate_proj"),
            logical_role_qname(packed, "up_proj"),
        )
    if leaf == "down_proj":
        return (logical_role_qname(packed, "down_proj"),)
    return ()


def stacked_role_col_weights(
    *,
    packed_qname: str,
    projection: str,
    member_qnames: Mapping[tuple[str, int], str],
    col_weights: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, tuple[str, ...]]:
    """Return exact per-expert imatrix rows for one logical pooled book."""

    selected = sorted(
        (
            int(expert_id),
            str(member),
        )
        for (member_projection, expert_id), member in member_qnames.items()
        if str(member_projection) == str(projection)
    )
    if not selected:
        raise ValueError(
            f"{packed_qname}: no {projection} members for learned role book"
        )
    # A full bundle build is separately required to cover 0..E-1.  Export may
    # pass a format subgroup such as expert ids [1, 7, 19]; sorting here defines
    # the local packed order that the per-expert declaration records.
    rows: list[torch.Tensor] = []
    names: list[str] = []
    for _expert_id, member in selected:
        if member not in col_weights:
            raise ValueError(
                f"{packed_qname}: learned role member {member!r} has no "
                "col_weights entry"
            )
        rows.append(torch.as_tensor(col_weights[member]).reshape(1, -1))
        names.append(member)
    widths = {int(row.shape[-1]) for row in rows}
    if len(widths) != 1:
        raise ValueError(
            f"{packed_qname}: {projection} col-weight widths disagree: {widths}"
        )
    return torch.stack(rows).contiguous(), tuple(names)


def split_role_rows(
    weight: torch.Tensor,
    roles: tuple[RoutedMoECodebookRole, ...],
) -> tuple[tuple[RoutedMoECodebookRole, torch.Tensor], ...]:
    """Split a physical rank-3 expert stack by declared contiguous role rows."""

    value = torch.as_tensor(weight)
    if value.ndim != 3:
        raise ValueError(
            f"routed learned role packing needs rank-3 weight, got {value.shape}"
        )
    if not roles:
        raise ValueError("routed learned role packing has no logical roles")
    offset = 0
    result = []
    for role in roles:
        rows = int(role.output_rows)
        if rows <= 0:
            raise ValueError(f"{role.qname}: output_rows must be positive")
        result.append((role, value[:, offset:offset + rows, :]))
        offset += rows
    if offset != int(value.shape[1]):
        raise ValueError(
            "routed learned role rows do not cover the fused stack: "
            f"covered={offset}, physical={int(value.shape[1])}"
        )
    return tuple(result)


__all__ = [
    "DEFAULT_ROUTED_BOOK_KEYING",
    "ROUTED_BOOK_KEYINGS",
    "ROUTED_BOOK_KEYING_ROLE",
    "ROUTED_BOOK_KEYING_STACK",
    "ROUTED_BOOK_KEY_NAMES",
    "ROUTED_MOE_CBL_BANK_RUNGS",
    "ROUTED_ROLE_PROJECTIONS",
    "ROUTED_STACK_KEYS",
    "RoutedMoECodebookRole",
    "bundle_book_qname",
    "bundle_role_qname",
    "bundle_stack_qname",
    "describe_split_book_refusal",
    "fused_targets_with_split_books",
    "learned_role_qnames_for_packed",
    "logical_role_qname",
    "normalize_routed_book_keying",
    "routed_book_key",
    "split_role_rows",
    "stacked_role_col_weights",
]
