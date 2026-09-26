"""Activation-cache adapter for production CB LDLQ assignment.

This is a reader over the established probe cache, not a second cache. Dense
and per-expert Linears reuse their captured input rows directly; packed-MoE
stages use the existing checkpoint replay in :mod:`prismaquant.moe_imatrix`.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import torch


_ACT_FNAME_SUB = re.compile(r"[^A-Za-z0-9_-]")


def fill_empty_expert_activation_rows(
    rows: tuple[torch.Tensor, ...],
    *,
    qname: str,
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
    """Apply the declared cold-expert prior to sparse routed LDLQ rows.

    A bounded module-level activation reservoir can legitimately miss experts
    that the full calibration forward routed. Observed experts retain their
    exact rows; empty slices receive the pooled routed rows from the same
    layer/projection. This is the activation analogue of the imatrix
    layer-routed-mean neutral prior and never fabricates a cross-layer sample.
    """
    missing = tuple(i for i, value in enumerate(rows) if not value.shape[0])
    if not missing:
        return rows, ()
    observed = [value for value in rows if value.shape[0]]
    if not observed:
        raise ValueError(
            f"{qname}: LDLQ routed replay has no observed expert rows"
        )
    widths = {int(value.shape[1]) for value in observed}
    if len(widths) != 1:
        raise ValueError(
            f"{qname}: LDLQ routed expert rows disagree on input width"
        )
    pooled = torch.cat(observed, dim=0).contiguous()
    filled = tuple(pooled if not value.shape[0] else value for value in rows)
    return filled, missing


class CBLDLQActivationLoader:
    """Load one target's Hessian rows lazily from the production act cache."""

    def __init__(
        self,
        activation_cache_dir: str | Path,
        *,
        model_dir: str | Path,
        profile,
        expert_stack_members: Mapping[
            str, Mapping[tuple[str, int], str]
        ] | None = None,
        replay_device: str | None = None,
    ) -> None:
        self.activation_cache_dir = Path(activation_cache_dir)
        self.model_dir = Path(model_dir)
        self.profile = profile
        self.expert_stack_members = dict(expert_stack_members or {})
        self.replay_device = replay_device

    def _direct(self, qname: str) -> torch.Tensor | None:
        path = self.activation_cache_dir / (
            _ACT_FNAME_SUB.sub("__", str(qname)) + ".pt"
        )
        if not path.is_file():
            return None
        blob = torch.load(path, map_location="cpu", weights_only=False)
        value = blob.get("inputs") if isinstance(blob, dict) else None
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(
                f"{qname}: LDLQ activation cache entry has no rank-2 inputs"
            )
        return value.detach().to(torch.float32).contiguous()

    def _direct_with_indices(
        self, qname: str
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        """Rows plus the global calibration row index each was sampled from."""
        path = self.activation_cache_dir / (
            _ACT_FNAME_SUB.sub("__", str(qname)) + ".pt"
        )
        if not path.is_file():
            return None
        blob = torch.load(path, map_location="cpu", weights_only=False)
        value = blob.get("inputs") if isinstance(blob, dict) else None
        if not isinstance(value, torch.Tensor) or value.ndim != 2:
            raise ValueError(
                f"{qname}: LDLQ activation cache entry has no rank-2 inputs"
            )
        indices = blob.get("row_indices") if isinstance(blob, dict) else None
        if not isinstance(indices, torch.Tensor):
            indices = None
        return value.detach().to(torch.float32).contiguous(), indices

    def _per_expert(self, qname: str) -> tuple[torch.Tensor, ...] | None:
        """One activation matrix per expert, unioned across fused members.

        Fused siblings (gate_proj/up_proj) consume the SAME input, so the fused
        serving unit needs one Hessian. But the activation reservoir samples
        each projection independently, so their captured matrices are different
        subsets of the same stream -- measured on DSv4 L0/expert 0: 64 rows
        each, intersection 15, union 113, and rows at shared indices are
        bit-identical. Requiring the matrices to be equal (as this did before)
        therefore fails on correct data.

        Union by global ``row_indices`` instead. That satisfies the fused-unit
        contract, and because the members sampled different tokens it also
        RAISES Hessian support for fused units (64 -> ~113 rows here), which
        matters: support is the binding constraint on whether LDLQ generalises
        at all (see dq-runs/dsv4-flash-0731/ldlq-delta/LDLQ_DIAGNOSIS.md).

        The shared-input assumption is still enforced, and more sharply than
        before: any global row index contributed by two members must carry the
        SAME vector. Absent ``row_indices`` the loader cannot align samples and
        falls back to the strict equality check, fail-closed.
        """
        members = self.expert_stack_members.get(qname)
        if not members:
            return None
        by_expert: dict[int, list[str]] = {}
        for (_projection, expert), member in sorted(members.items()):
            by_expert.setdefault(int(expert), []).append(str(member))
        rows: list[torch.Tensor] = []
        for expert in sorted(by_expert):
            loaded = [self._direct_with_indices(name)
                      for name in by_expert[expert]]
            loaded = [value for value in loaded if value is not None]
            if not loaded:
                # Never-routed expert: the calibration forward sent it no
                # tokens, so there is no capture to read. That is legitimate
                # and common here -- this export reports 3984 never-routed
                # expert projections across 60 stacks. Emit an empty slice and
                # let the declared cold-expert prior fill it below, exactly as
                # the routed-replay path already does. Raising instead would
                # make LDLQ unusable on any MoE with cold experts.
                rows.append(None)
                continue
            if len(loaded) == 1:
                rows.append(loaded[0][0])
                continue
            widths = {int(values.shape[1]) for values, _ in loaded}
            if len(widths) != 1:
                raise ValueError(
                    f"{qname}: fused expert {expert} members disagree on "
                    "their captured LDLQ input width"
                )
            if any(indices is None for _values, indices in loaded):
                reference = loaded[0][0]
                if any(
                    tuple(values.shape) != tuple(reference.shape)
                    or not torch.equal(values, reference)
                    for values, _ in loaded[1:]
                ):
                    raise ValueError(
                        f"{qname}: fused expert {expert} members disagree on "
                        "their captured LDLQ input rows, and carry no "
                        "row_indices to align them"
                    )
                rows.append(reference)
                continue
            merged: dict[int, torch.Tensor] = {}
            for values, indices in loaded:
                if int(indices.shape[0]) != int(values.shape[0]):
                    raise ValueError(
                        f"{qname}: expert {expert} row_indices length "
                        f"{int(indices.shape[0])} != rows "
                        f"{int(values.shape[0])}"
                    )
                for position, raw_index in enumerate(indices.tolist()):
                    index = int(raw_index)
                    row = values[position]
                    seen = merged.get(index)
                    if seen is not None and not torch.equal(seen, row):
                        raise ValueError(
                            f"{qname}: fused expert {expert} members captured "
                            f"DIFFERENT vectors for global calibration row "
                            f"{index}; fused siblings must share one input"
                        )
                    merged[index] = row
            rows.append(
                torch.stack([merged[key] for key in sorted(merged)]).contiguous()
            )
        observed = [value for value in rows if value is not None]
        if not observed:
            raise ValueError(
                f"{qname}: no expert in the stack has LDLQ activation rows"
            )
        width = int(observed[0].shape[1])
        filled, missing = fill_empty_expert_activation_rows(
            tuple(
                value if value is not None else torch.empty(
                    (0, width), dtype=observed[0].dtype)
                for value in rows
            ),
            qname=str(qname),
        )
        if missing:
            print(
                f"[cb-ldlq] {qname}: {len(missing)} never-routed expert(s) "
                f"took the pooled cold-expert prior (layer/projection routed "
                f"rows); indices {list(missing)[:8]}"
                + ("..." if len(missing) > 8 else ""),
                flush=True,
            )
        return filled

    def load(
        self,
        qname: str,
        *,
        stack_size: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        direct = self._direct(qname)
        if direct is not None:
            return direct
        per_expert = self._per_expert(qname)
        if per_expert is not None:
            if stack_size is not None and len(per_expert) != int(stack_size):
                raise ValueError(
                    f"{qname}: LDLQ activation stacks={len(per_expert)} != "
                    f"weight stacks={stack_size}"
                )
            return per_expert

        from .moe_imatrix import (
            RoutedActivationSamples,
            synthesize_packed_expert_activation_samples,
        )

        replayed = synthesize_packed_expert_activation_samples(
            self.model_dir,
            self.activation_cache_dir,
            {str(qname)},
            self.profile,
            device=self.replay_device,
        ).get(str(qname))
        if isinstance(replayed, RoutedActivationSamples):
            if stack_size is None:
                return replayed.values
            rows = tuple(
                replayed.values[replayed.expert_indices == expert].contiguous()
                for expert in range(int(stack_size))
            )
            rows, _missing = fill_empty_expert_activation_rows(
                rows, qname=str(qname))
            return rows
        if isinstance(replayed, torch.Tensor) and replayed.ndim == 2:
            return replayed.detach().to(torch.float32).contiguous()
        raise ValueError(
            f"{qname}: no value-bearing activation rows for LDLQ export; "
            "rebuild the production activation cache"
        )


__all__ = ["CBLDLQActivationLoader", "fill_empty_expert_activation_rows"]
