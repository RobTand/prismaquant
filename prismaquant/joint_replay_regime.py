"""The Stage B replay regime: how a layer quantum accumulates its statistics.

A quantum that replays from the one-pass spill (PQ #994) accumulates each
Linear's FP32 statistics in one of two modes, at one capture batch size:

- ``capture_batch``: stored batches per forward/backward when the spill is
  captured. 1 runs one backward per stored batch, as the windowed replay does.
- ``accumulation``: ``per_invocation`` runs one FP32 GEMM per backward
  invocation and sums the matrices in firing order, which is what the live
  hook does. ``operator_gemm`` runs one FP32 GEMM per operator over the
  operator's spilled rows, ``chunk_rows`` rows at a time.

The default regime (batch 1, per invocation) is the bitwise one, and it
stamps nothing. Any other regime changes the arithmetic, so
:func:`stamp_replay_regime` records it in ``statistics_arithmetic_identity``.
From there it reaches every cost row's probe identity, and the campaign join
refuses quanta whose regimes differ.

The regime is a launch setting, not a plan field. One environment variable
carries it, sealed in the campaign container spec that every quantum of a
dispatch shares::

    PRISMAQUANT_STAGE_B_REPLAY_REGIME=capture_batch=4,accumulation=operator_gemm,chunk_rows=65536

Unset means the default. The default spelled out is refused, so the default
has one spelling and its quanta keep the identity they had before regimes
existed.

This module imports nothing heavy; the dispatcher validates with it.
"""
from __future__ import annotations

from collections.abc import Mapping

REPLAY_REGIME_ENV = "PRISMAQUANT_STAGE_B_REPLAY_REGIME"
REPLAY_REGIME_SCHEMA = "prismaquant.stage_b.replay_regime.v1"
#: The key the stamp occupies in ``statistics_arithmetic_identity``.
REPLAY_REGIME_FIELD = "stage_b_replay_regime"

PER_INVOCATION = "per_invocation"
OPERATOR_GEMM = "operator_gemm"
ACCUMULATIONS = (PER_INVOCATION, OPERATOR_GEMM)

DEFAULT_REPLAY_REGIME = {"capture_batch": 1, "accumulation": PER_INVOCATION,
                         "chunk_rows": None}
_FIELDS = frozenset(DEFAULT_REPLAY_REGIME)

#: ``operator_accumulation`` under ``operator_gemm``: one FP32 GEMM per
#: operator per row chunk, over the rows in the order the spill recorded them,
#: the chunk products summed in that order.
OPERATOR_GEMM_ACCUMULATION = "fp32_gemm_over_spilled_rows_in_capture_order_by_row_chunk"


class ReplayRegimeRefused(ValueError):
    """A malformed replay regime, or the default one spelled out."""


def _parse(text: str) -> dict:
    if not text or text != text.strip():
        raise ReplayRegimeRefused(f"replay regime {text!r} is empty or padded")
    fields: dict = {}
    for part in text.split(","):
        key, sep, value = part.partition("=")
        if not sep or key not in _FIELDS or key in fields or not value:
            raise ReplayRegimeRefused(
                f"replay regime field {part!r}: expected each of "
                f"{sorted(_FIELDS)} at most once, spelled key=value")
        fields[key] = value
    for key in ("capture_batch", "chunk_rows"):
        value = fields.get(key)
        if value is None:
            continue
        if not value.isdecimal() or value != str(int(value)):
            raise ReplayRegimeRefused(f"replay regime {key}={value!r} is not a canonical integer")
        fields[key] = int(value)
    return fields


def normalize_replay_regime(regime) -> dict:
    """The complete regime for ``None``, a compact string or a mapping."""
    if regime is None:
        return dict(DEFAULT_REPLAY_REGIME)
    if isinstance(regime, str):
        fields = _parse(regime)
    elif isinstance(regime, Mapping):
        fields = dict(regime)
        if not set(fields) <= _FIELDS:
            raise ReplayRegimeRefused(
                f"replay regime fields {sorted(set(fields) - _FIELDS)} are unknown")
    else:
        raise ReplayRegimeRefused(f"replay regime {regime!r} is neither a string nor a mapping")
    result = {**DEFAULT_REPLAY_REGIME, **fields}
    batch, accumulation, chunk = (result["capture_batch"], result["accumulation"],
                                  result["chunk_rows"])
    if type(batch) is not int or batch < 1:
        raise ReplayRegimeRefused(f"replay regime capture_batch={batch!r} is not a positive integer")
    if accumulation not in ACCUMULATIONS:
        raise ReplayRegimeRefused(
            f"replay regime accumulation={accumulation!r} is not one of {list(ACCUMULATIONS)}")
    if accumulation == OPERATOR_GEMM:
        if type(chunk) is not int or chunk < 1:
            raise ReplayRegimeRefused(
                "operator_gemm accumulation needs an explicit positive chunk_rows: "
                "the chunk sets the FP32 summation order")
    elif chunk is not None:
        raise ReplayRegimeRefused("chunk_rows applies only to operator_gemm accumulation")
    return result


def replay_regime_identity(regime) -> dict | None:
    """``None`` for the default regime, else the block the arithmetic stamps."""
    regime = normalize_replay_regime(regime)
    if regime == DEFAULT_REPLAY_REGIME:
        return None
    return {"schema": REPLAY_REGIME_SCHEMA, **regime}


def stamp_replay_regime(arithmetic: Mapping, regime) -> dict:
    """A copy of ``arithmetic`` with the regime stamped.

    The default regime returns an unchanged copy. ``operator_gemm`` also
    replaces ``operator_accumulation``, because it replaces that arithmetic.
    """
    result = dict(arithmetic)
    identity = replay_regime_identity(regime)
    if identity is None:
        return result
    if REPLAY_REGIME_FIELD in result:
        raise ReplayRegimeRefused("the arithmetic identity already carries a replay regime")
    result[REPLAY_REGIME_FIELD] = identity
    if identity["accumulation"] == OPERATOR_GEMM:
        result["operator_accumulation"] = OPERATOR_GEMM_ACCUMULATION
    return result


def replay_regime_of(arithmetic: Mapping) -> dict:
    """The regime an arithmetic identity records.

    Refuses a stamped default, a malformed block, and an
    ``operator_accumulation`` that disagrees with the stamped mode.
    """
    block = arithmetic.get(REPLAY_REGIME_FIELD)
    if block is None:
        regime = dict(DEFAULT_REPLAY_REGIME)
    else:
        if (not isinstance(block, Mapping) or block.get("schema") != REPLAY_REGIME_SCHEMA
                or set(block) != _FIELDS | {"schema"}):
            raise ReplayRegimeRefused("malformed Stage B replay regime stamp")
        regime = normalize_replay_regime({key: block[key] for key in _FIELDS})
        if regime == DEFAULT_REPLAY_REGIME:
            raise ReplayRegimeRefused(
                "a stamped default replay regime; the default stamps nothing")
    gemm = regime["accumulation"] == OPERATOR_GEMM
    if gemm != (arithmetic.get("operator_accumulation") == OPERATOR_GEMM_ACCUMULATION):
        raise ReplayRegimeRefused(
            "operator_accumulation disagrees with the stamped replay regime")
    return regime


def handoff_regime_refusal(regime, *, chain_batch_size: int) -> str | None:
    """Why a band-serial producer (PQ #996) cannot run ``regime``, or ``None``.

    A producer hands off the boundary plane its capture pass wrote, and the
    handoff must equal the plane the consumer's chain rebuild ends on. The
    chain rebuild runs the Stage A slice's chain regime (PQ #997), whose
    ``batch_size`` sets the GEMM shapes and so the rounding; the capture pass
    writes the plane through one backward per ``capture_batch`` stored
    batches, grouped as the chain groups them (``joint_cost_quantum.
    capture_group``, ``joint_adjoint_checkpoints._roll_group``). The two
    planes are the same arithmetic exactly when the two batch sizes are
    equal, so a producer is admitted at the chain's batch size and refused
    at any other. The accumulation mode is admitted: it changes only the
    statistics, never the plane.
    """
    batch = normalize_replay_regime(regime)["capture_batch"]
    if type(chain_batch_size) is not int or chain_batch_size < 1:
        raise ReplayRegimeRefused(
            f"chain batch size must be a positive integer, got {chain_batch_size!r}")
    if batch == chain_batch_size:
        return None
    return (f"capture_batch={batch} writes the boundary plane through one backward "
            f"per {batch} stored batches, and the consumer's chain rebuild runs the "
            f"Stage A chain regime at batch size {chain_batch_size}; a band-serial "
            "handoff must equal the plane that rebuild ends on, so emit no handoff, "
            f"or capture at batch {chain_batch_size}")


def replay_regime_from_environment(environ: Mapping) -> dict | None:
    """The regime a launch declares, or ``None`` (the default) when unset."""
    text = environ.get(REPLAY_REGIME_ENV)
    if text is None:
        return None
    regime = normalize_replay_regime(text)
    if regime == DEFAULT_REPLAY_REGIME:
        raise ReplayRegimeRefused(
            f"{REPLAY_REGIME_ENV}={text!r} spells the default regime; leave it unset")
    return regime
