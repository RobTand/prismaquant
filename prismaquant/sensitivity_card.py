"""The PrismaQuant Sensitivity Card: a shareable, format-independent probe artifact.

WHY THIS EXISTS
---------------
Probing a model is the expensive, model-specific half of PrismaQuant. Choosing
formats is the cheap, *platform*-specific half. Today those two halves are fused:
``probe.pkl`` carries scalars that are only meaningful next to a rendered cost
cache built for one particular format menu, so a new menu means a new probe.

A Sensitivity Card breaks that coupling. It is measured **once** per
(model, calibration) and carries enough structure that an *arbitrary* format
menu can be priced locally, without touching the model again. The author of a
downstream artifact supplies their own format list and platform profile; the
card supplies the sensitivity.

WHAT MAKES AN ARBITRARY MENU PRICEABLE
--------------------------------------
The probe already computes, per Linear, the per-element diagonal empirical
Fisher of the weight matrix::

    H[o, i] = sum_t  g[t, o]^2 * x[t, i]^2

(`incremental_probe.py`: ``chunk_h = gy2_sq.t() @ x2_sq``). Storing ``H`` is
what makes a full-detail probe unshippable -- the unified-sweep path documents
it as "47k x 17 MB = 800 GB CPU, doesn't fit" and therefore keeps only the two
scalars ``h_trace`` and ``h_w2_sum``.

But every quantity a format-agnostic cost needs is a **marginal** of ``H``, and
each marginal is a pair of reductions that never forms the [out, in] matrix:

    fisher_row[o] = sum_i H[o, i] = sum_t g[t,o]^2 * (sum_i x[t,i]^2)
    fisher_col[i] = sum_o H[o, i] = sum_t (sum_o g[t,o]^2) * x[t,i]^2

Cost: ``out + in`` floats per Linear instead of ``out * in``. For a 27B dense
model that is tens of MB rather than hundreds of GB, and it is available even on
the memory-bounded unified sweep that cannot accumulate ``h_full`` at all.

Two further vectors are stored because they are *not* recoverable from the
weight-Fisher marginals and are what activation-quantization awareness
(AQUA-AURA) is built on:

    act_sq_sum[i] = sum_t x[t, i]^2      # the imatrix / diag(X^T X)
    g_sq_sum[o]   = sum_t g[t, o]^2      # the OUTPUT-space Fisher diagonal

``act_sq_sum`` lets any downstream format weight its weight error by the
activation distribution the layer actually sees. ``g_sq_sum`` is the term that
converts an *output* perturbation into a loss delta, which is precisely what an
input-side (activation-quantization) error needs and what ``h_trace`` -- a
weight-space quantity -- cannot supply.

Consistency is checkable for free: ``sum(fisher_row) == sum(fisher_col) ==
h_trace_raw`` up to float error. :func:`SensitivityUnit.validate` enforces it.

CURRENCY IS EXPLICIT AND FAIL-CLOSED
------------------------------------
``activation_fair_pricing.py`` is a 120-line autopsy of what happens when one
family is priced on two bases. Every cost component in this module therefore
declares its :class:`Currency`, and mixing them raises rather than silently
producing a number. See also the CB-lane finding that a factorization law which
holds in activation currency *fails* in weight_mse currency across a basis
change.

CALIBRATION IS IDENTITY
-----------------------
A card measured on one calibration is not interchangeable with another, exactly
as a CB codebook's imatrix is hashed into its book key. ``calib_hash`` is part
of the card's identity; :func:`SensitivityCard.assert_compatible` refuses
cross-calibration merges and comparisons.

STRUCTURE vs POLICY
-------------------
The card carries model **structure** (which Linears are siblings, their shapes,
their source dtype) because that is a property of the checkpoint and is true for
every consumer. It deliberately does **not** carry serving **policy** (that
fused siblings must share one format, that packed experts must use vLLM
canonical scheme names) because that is a property of the downstream runtime and
belongs to the profile the author names. Baking vLLM's packing rules into a
shareable file would make the file wrong for llama.cpp, and wrong the day vLLM
changes.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .digests import text_sha256hex

# 1.1 adds the packed-expert A-side vectors (``expert_*``). Additive: a 1.0
# reader sees array keys it never asks for, and a 1.1 card whose probe predates
# the F.linear interception simply omits them.
SCHEMA_VERSION = "1.1"

# Vectors are stored in float32 on disk. They are sums of squares -- strictly
# non-negative and spanning a wide dynamic range across channels -- so float16
# would silently flush the small tail to zero and underflow exactly the
# low-sensitivity channels a low-bit format most wants to exploit.
VECTOR_DTYPE = np.float32


class Currency(enum.Enum):
    """The space a cost component is measured in.

    Mixing these is the failure mode `activation_fair_pricing.py` documents
    (84% rung-order violations when one family was priced on two bases), so the
    unit is carried with the number rather than assumed by the caller.
    """

    #: Mean squared error on the weight tensor itself. Format-local, cheap,
    #: and the basis `weight_mse` scores are reported in.
    WEIGHT_MSE = "weight_mse"

    #: Mean squared error on the layer OUTPUT (``y = Wx``). What a render-score
    #: measures, and the space an activation-side perturbation naturally lives in.
    OUTPUT_MSE = "output_mse"

    #: A predicted delta in the model's loss (nats). The only currency the
    #: allocator's knapsack may sum across units, because only loss is additive.
    DELTA_LOSS = "delta_loss"


class RenderBasis(enum.Enum):
    """How the weight error a consumer computes will have been produced.

    This is not a footnote. RTN-vs-GPTQ dW is immaterial at fp4 but worth ~+36%
    at fp8, so a card priced on one basis mis-ranks 8-bit rungs on the other.
    The shareable tier is necessarily RTN: GPTQ needs the full Hessian
    (~100 MB/Linear at 27B scale, ~40 GB total), which is not shippable.
    """

    #: Round-to-nearest. Computable from the weight alone. The only basis a
    #: shareable card can promise, because it needs no Hessian.
    RTN = "rtn"

    #: Error-compensated render (GPTQ and friends). Requires local Hessians;
    #: reproducible only by a consumer who re-derives them.
    COMPENSATED = "compensated"


@dataclasses.dataclass(frozen=True)
class UnitTopology:
    """Model-structural facts about a decision unit.

    Structure, not policy: this says q/k/v *are* siblings in one attention
    block, not that a given runtime requires them to share a format.
    """

    #: Dotted module path, e.g. ``model.layers.3.self_attn.q_proj``.
    name: str
    #: Transformer block index, or None for non-block modules (embeddings, head).
    layer_index: int | None = None
    #: Coarse functional role: "q", "k", "v", "o", "gate", "up", "down", ...
    #: Used by consumers that price per-role; never used to *ban* a format.
    role: str | None = None
    #: Stable id shared by Linears that are siblings in one fused op
    #: (q/k/v of a block, gate/up of an MLP). None if the unit stands alone.
    fused_group: str | None = None
    #: Stable id shared by experts packed into one served tensor. None if dense.
    packed_group: str | None = None
    #: Expert ordinal within ``packed_group``, if applicable.
    expert_id: int | None = None
    #: dtype of the tensor in the *source* checkpoint, e.g. "bfloat16",
    #: "float8_e4m3fn". Passthrough formats are only legal when this matches,
    #: so a consumer needs it to evaluate legality without opening the model.
    source_dtype: str | None = None


@dataclasses.dataclass(frozen=True)
class SensitivityUnit:
    """Everything needed to price one decision unit under an arbitrary format.

    All raw accumulators are stored unnormalized (as summed over calibration
    tokens) together with ``n_tokens``; normalization is the consumer's choice
    and a normalized value cannot be un-normalized.
    """

    topology: UnitTopology

    out_features: int
    in_features: int
    n_params: int

    #: Global calibration token count used to normalize every row. MoE experts
    #: are normalized by the SAME global count as dense rows: tokens never
    #: routed to an expert contribute zero gradient to a mean-delta-loss
    #: objective, so dividing by a per-expert routed count inverts importance
    #: weighting (the bug removed in PR #14's `finalize_fisher_stats`).
    n_tokens: int

    #: sum over all (o, i) of H[o, i]. Total weight-Fisher curvature mass.
    h_trace_raw: float
    #: sum over all (o, i) of H[o, i] * W[o, i]^2.
    h_w2_sum_raw: float

    w_norm_sq: float
    w_max_abs: float

    #: sum_i H[o, i] -- per-OUTPUT-channel marginal of the weight Fisher.
    fisher_row: np.ndarray | None = None
    #: sum_o H[o, i] -- per-INPUT-channel marginal of the weight Fisher.
    fisher_col: np.ndarray | None = None
    #: sum_t x[t, i]^2 -- the imatrix. Activation second moment, gradient-free.
    act_sq_sum: np.ndarray | None = None
    #: sum_t g[t, o]^2 -- OUTPUT-space Fisher diagonal. The AQUA-AURA term.
    g_sq_sum: np.ndarray | None = None
    #: Per-input-channel max |x|, when captured. Activation-quantization error
    #: is driven by outliers, which a second moment averages away.
    act_absmax: np.ndarray | None = None

    #: Fraction of calibration tokens routed to this unit (MoE experts only).
    #: Reported for diagnostics; it must NOT be used to rescale the Fisher rows
    #: (see ``n_tokens``).
    route_prob: float | None = None

    # ------------------------------------------------- packed-expert A-side
    # A packed [E, M, N] MoE parameter is ONE decision unit (the serving
    # runtime cannot give two experts in a tensor different formats), but its
    # A-side is not one number's worth of structure: routing means expert e
    # sees a different token subset, so g and the activation distribution
    # both vary per expert. Collapsing them into the 1-D vectors above would
    # destroy the W-vs-routing correlation --
    #   sum_e W_e^2 g_e var_e  !=  (sum_e W_e^2)(sum_e g_e)(sum_e var_e)
    # -- and routing concentration makes that gap large, not marginal.
    #
    # Present only for packed units, and only from a probe that ran the
    # F.linear interception (schema 1.1+). ``out_features``/``in_features``
    # describe ONE expert's slice; ``n_params`` counts all E.

    #: [E, out_features] -- sum over tokens ROUTED TO e of g[t, o]^2.
    expert_g_sq_sum: np.ndarray | None = None
    #: [E, in_features] -- sum over tokens ROUTED TO e of x[t, i]^2.
    expert_act_sq_sum: np.ndarray | None = None
    #: [E, in_features] -- max |x[t, i]| over tokens routed to e.
    expert_act_absmax: np.ndarray | None = None
    #: [E] -- how many calibration tokens were routed to each expert. The
    #: denominator for ``expert_act_sq_sum`` ONLY: it fits a per-token noise
    #: magnitude, so it must divide by the tokens that actually flowed.
    #: ``expert_g_sq_sum`` is normalized by the global ``n_tokens`` like every
    #: other row, because that is what carries the expert's share of the
    #: objective. Using one denominator for both discounts a rare expert
    #: twice -- PR #14's inverted importance weighting, in mirror image.
    expert_tokens: np.ndarray | None = None

    # ---------------------------------------------------------------- helpers

    @property
    def h_trace(self) -> float:
        """Token-normalized weight-Fisher trace -- today's scalar sensitivity."""
        return self.h_trace_raw / max(1, self.n_tokens)

    @property
    def n_experts(self) -> int | None:
        """Experts packed into this unit, or None when it is dense."""
        if self.expert_g_sq_sum is None:
            return None
        return int(np.asarray(self.expert_g_sq_sum).shape[0])

    @property
    def has_expert_activation_stats(self) -> bool:
        """True when the A-side can be priced PER EXPERT for this unit.

        All three arrays or none: ``expert_tokens`` is what makes the two sums
        interpretable, so a unit carrying sums without their routed denominator
        is not usable and must not be quietly half-priced.
        """
        return (self.expert_g_sq_sum is not None
                and self.expert_act_sq_sum is not None
                and self.expert_tokens is not None)

    @property
    def has_vectors(self) -> bool:
        """True when this unit can price a format beyond the scalar model."""
        return self.fisher_col is not None and self.fisher_row is not None

    def rank1_fisher(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Return (row, col) scaled so ``outer(row, col)`` approximates H.

        H is a sum over tokens of outer(g_t^2, x_t^2), so it is exactly rank-1
        when one token dominates and increasingly not so otherwise. The rank-1
        reconstruction ``outer(row, col) / h_trace`` is therefore an
        approximation -- but a strictly more descriptive one than the scalar
        ``h_trace``, which is the rank-0 collapse of the same object.
        """
        if not self.has_vectors or self.h_trace_raw <= 0.0:
            return None
        scale = 1.0 / math.sqrt(self.h_trace_raw)
        return (
            np.asarray(self.fisher_row, dtype=np.float64) * scale,
            np.asarray(self.fisher_col, dtype=np.float64) * scale,
        )

    def validate(self, *, rtol: float = 1e-3) -> None:
        """Fail closed on internally inconsistent rows.

        The marginal identity ``sum(fisher_row) == sum(fisher_col) ==
        h_trace_raw`` is free to check and catches the whole class of bugs where
        one accumulator is normalized and another is not.
        """
        if self.out_features <= 0 or self.in_features <= 0:
            raise ValueError(f"{self.topology.name}: non-positive shape")
        if self.n_tokens <= 0:
            raise ValueError(f"{self.topology.name}: n_tokens must be positive")
        if self.h_trace_raw < 0.0:
            raise ValueError(f"{self.topology.name}: negative h_trace_raw")

        for label, vec, expect in (
            ("fisher_row", self.fisher_row, self.out_features),
            ("g_sq_sum", self.g_sq_sum, self.out_features),
            ("fisher_col", self.fisher_col, self.in_features),
            ("act_sq_sum", self.act_sq_sum, self.in_features),
            ("act_absmax", self.act_absmax, self.in_features),
        ):
            if vec is None:
                continue
            arr = np.asarray(vec)
            if arr.shape != (expect,):
                raise ValueError(
                    f"{self.topology.name}: {label} has shape {arr.shape}, "
                    f"expected ({expect},)")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{self.topology.name}: {label} has non-finite entries")
            if label != "act_absmax" and np.any(arr < 0.0):
                raise ValueError(f"{self.topology.name}: {label} has negative entries")

        if self.h_trace_raw > 0.0:
            for label, vec in (("fisher_row", self.fisher_row),
                               ("fisher_col", self.fisher_col)):
                if vec is None:
                    continue
                total = float(np.asarray(vec, dtype=np.float64).sum())
                if not math.isclose(total, self.h_trace_raw, rel_tol=rtol):
                    raise ValueError(
                        f"{self.topology.name}: sum({label})={total:.6g} does not "
                        f"match h_trace_raw={self.h_trace_raw:.6g}. The marginals "
                        f"and the trace must come from the same accumulator.")


@dataclasses.dataclass(frozen=True)
class CardProvenance:
    """Identity of a card. Two cards are comparable only if these agree."""

    model_id: str
    #: Hash of the calibration text + tokenization + sample count. Calibration
    #: is identity: a card measured on other text is a different card.
    calib_hash: str
    n_calib_samples: int
    seq_len: int
    #: git commit of the probe that produced this card.
    probe_commit: str
    #: The render basis a consumer should assume when computing weight error.
    render_basis: RenderBasis = RenderBasis.RTN
    #: Free-form notes (e.g. "unified-sweep, h_full unavailable").
    notes: str = ""

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "model_id": self.model_id,
                "calib_hash": self.calib_hash,
                "n_calib_samples": self.n_calib_samples,
                "seq_len": self.seq_len,
                "render_basis": self.render_basis.value,
            },
            sort_keys=True,
        )
        return text_sha256hex(payload)


class SensitivityCard:
    """A model's measured sensitivity, independent of any format menu."""

    def __init__(
        self,
        provenance: CardProvenance,
        units: Iterable[SensitivityUnit],
        *,
        schema_version: str = SCHEMA_VERSION,
    ) -> None:
        self.provenance = provenance
        self.schema_version = schema_version
        self._units: dict[str, SensitivityUnit] = {}
        for unit in units:
            if unit.topology.name in self._units:
                raise ValueError(f"duplicate unit {unit.topology.name}")
            self._units[unit.topology.name] = unit

    # ------------------------------------------------------------- accessors

    def __len__(self) -> int:
        return len(self._units)

    def __contains__(self, name: object) -> bool:
        return name in self._units

    def __getitem__(self, name: str) -> SensitivityUnit:
        return self._units[name]

    def units(self) -> Sequence[SensitivityUnit]:
        return tuple(self._units.values())

    def names(self) -> Sequence[str]:
        return tuple(self._units)

    def fused_groups(self) -> Mapping[str, tuple[str, ...]]:
        """Sibling identity as measured. Whether siblings MUST share a format is
        the consumer's platform policy, not the card's claim."""
        groups: dict[str, list[str]] = {}
        for unit in self._units.values():
            gid = unit.topology.fused_group
            if gid:
                groups.setdefault(gid, []).append(unit.topology.name)
        return {k: tuple(v) for k, v in groups.items()}

    def packed_groups(self) -> Mapping[str, tuple[str, ...]]:
        groups: dict[str, list[str]] = {}
        for unit in self._units.values():
            gid = unit.topology.packed_group
            if gid:
                groups.setdefault(gid, []).append(unit.topology.name)
        return {k: tuple(v) for k, v in groups.items()}

    @property
    def has_vectors(self) -> bool:
        return all(u.has_vectors for u in self._units.values())

    # ------------------------------------------------------------ invariants

    def validate(self) -> None:
        for unit in self._units.values():
            unit.validate()

    def assert_compatible(self, other: "SensitivityCard") -> None:
        """Refuse cross-calibration comparison or merge.

        A CB book hashes its imatrix into its key for exactly this reason; a
        sensitivity card is the same kind of object and gets the same rule.
        """
        if self.provenance.calib_hash != other.provenance.calib_hash:
            raise ValueError(
                "refusing to combine sensitivity cards from different "
                f"calibrations ({self.provenance.calib_hash[:12]} vs "
                f"{other.provenance.calib_hash[:12]}). Calibration is identity: "
                "rebase one card onto the other's calibration instead.")
        if self.provenance.model_id != other.provenance.model_id:
            raise ValueError(
                f"model mismatch: {self.provenance.model_id} vs "
                f"{other.provenance.model_id}")
        if self.provenance.render_basis != other.provenance.render_basis:
            raise ValueError(
                "render basis mismatch: "
                f"{self.provenance.render_basis.value} vs "
                f"{other.provenance.render_basis.value}. RTN-vs-compensated dW "
                "is immaterial at fp4 but ~+36% at fp8, so these do not compare.")

    # ------------------------------------------------------------------- I/O

    def to_npz(self, path: str) -> None:
        """Write the card as a single compressed .npz.

        One self-describing file, no pickle: a shareable artifact must be
        loadable by someone who does not run our code, and must not execute
        arbitrary objects on load.
        """
        arrays: dict[str, np.ndarray] = {}
        index: list[dict[str, Any]] = []

        for i, unit in enumerate(self._units.values()):
            entry: dict[str, Any] = {
                "name": unit.topology.name,
                "layer_index": unit.topology.layer_index,
                "role": unit.topology.role,
                "fused_group": unit.topology.fused_group,
                "packed_group": unit.topology.packed_group,
                "expert_id": unit.topology.expert_id,
                "source_dtype": unit.topology.source_dtype,
                "out_features": unit.out_features,
                "in_features": unit.in_features,
                "n_params": unit.n_params,
                "n_tokens": unit.n_tokens,
                "h_trace_raw": unit.h_trace_raw,
                "h_w2_sum_raw": unit.h_w2_sum_raw,
                "w_norm_sq": unit.w_norm_sq,
                "w_max_abs": unit.w_max_abs,
                "route_prob": unit.route_prob,
                "vectors": [],
            }
            for field in ("fisher_row", "fisher_col", "act_sq_sum",
                          "g_sq_sum", "act_absmax",
                          "expert_g_sq_sum", "expert_act_sq_sum",
                          "expert_act_absmax", "expert_tokens"):
                vec = getattr(unit, field)
                if vec is None:
                    continue
                key = f"v{i}_{field}"
                arrays[key] = np.asarray(vec, dtype=VECTOR_DTYPE)
                entry["vectors"].append(field)
            index.append(entry)

        header = {
            "schema_version": self.schema_version,
            "provenance": {
                "model_id": self.provenance.model_id,
                "calib_hash": self.provenance.calib_hash,
                "n_calib_samples": self.provenance.n_calib_samples,
                "seq_len": self.provenance.seq_len,
                "probe_commit": self.provenance.probe_commit,
                "render_basis": self.provenance.render_basis.value,
                "notes": self.provenance.notes,
            },
            "units": index,
        }
        np.savez_compressed(
            path, __header__=np.frombuffer(
                json.dumps(header).encode(), dtype=np.uint8), **arrays)

    @classmethod
    def from_npz(cls, path: str) -> "SensitivityCard":
        with np.load(path, allow_pickle=False) as data:
            header = json.loads(bytes(data["__header__"]).decode())
            prov_raw = header["provenance"]
            provenance = CardProvenance(
                model_id=prov_raw["model_id"],
                calib_hash=prov_raw["calib_hash"],
                n_calib_samples=int(prov_raw["n_calib_samples"]),
                seq_len=int(prov_raw["seq_len"]),
                probe_commit=prov_raw["probe_commit"],
                render_basis=RenderBasis(prov_raw["render_basis"]),
                notes=prov_raw.get("notes", ""),
            )
            units = []
            for i, entry in enumerate(header["units"]):
                vecs = {
                    field: np.array(data[f"v{i}_{field}"])
                    for field in entry["vectors"]
                }
                units.append(SensitivityUnit(
                    topology=UnitTopology(
                        name=entry["name"],
                        layer_index=entry["layer_index"],
                        role=entry["role"],
                        fused_group=entry["fused_group"],
                        packed_group=entry["packed_group"],
                        expert_id=entry["expert_id"],
                        source_dtype=entry["source_dtype"],
                    ),
                    out_features=int(entry["out_features"]),
                    in_features=int(entry["in_features"]),
                    n_params=int(entry["n_params"]),
                    n_tokens=int(entry["n_tokens"]),
                    h_trace_raw=float(entry["h_trace_raw"]),
                    h_w2_sum_raw=float(entry["h_w2_sum_raw"]),
                    w_norm_sq=float(entry["w_norm_sq"]),
                    w_max_abs=float(entry["w_max_abs"]),
                    route_prob=entry["route_prob"],
                    **vecs,
                ))
        return cls(provenance, units, schema_version=header["schema_version"])
