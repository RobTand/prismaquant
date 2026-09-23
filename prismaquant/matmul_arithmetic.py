"""The matmul settings every Stage A, Stage B and joint AURA entry point pins.

Each entry point pins the float32 matmul precision to ``"highest"`` and turns
TF32 off, and :func:`arithmetic_identity` records both. A third setting also
decides the rounding of the bf16 GEMMs these runs are made of:
``torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction``. When it
is True, which is PyTorch's default, cuBLAS may reduce a split-K bf16 GEMM's
partial sums in bf16. No entry point set it and no identity recorded it
(PQ #1028).

On the GLM-shaped proxy of PQ #994 at the campaign accumulation
(``operator_gemm``) the flag made no difference at capture batch 1: the flag
on and off gave byte-identical records. At capture batch 8 with the flag on,
the per-operator statistics moved 25 to 50 times further from an FP32
accumulation reference than batch 1 does, and 129 of 16384 tokens changed
experts. With the flag off, batch 8 was as close to the reference as batch 1,
with no expert changes (PB ``c0f57b68de40``). The proxy resolved no time cost
for the flag (PB ``028b76eef2fa``); it ran at about one sixth of the GB10
power envelope, so the real-scale cost is Stage A's batch sweep to measure.

One environment variable sets it, sealed in the campaign container spec like
the Stage B replay regime::

    PRISMAQUANT_BF16_REDUCED_PRECISION_REDUCTION=off

Unset keeps PyTorch's default and stamps nothing, so every identity, record
and receipt written before this module keeps its bytes. ``off`` pins the flag
to False and stamps ``allow_bf16_reduced_precision_reduction: false`` into
every arithmetic identity the run writes. ``on`` spells the default and is
refused, so the default has one spelling.

The stamp is read from the live flag, never from the environment, so an
identity always says what the arithmetic ran under.

This module imports nothing heavy at module level; the dispatcher validates
the environment with it.
"""
from __future__ import annotations

from collections.abc import Mapping
import os

#: The launch setting. Unset is the default; ``off`` is the only other value.
BF16_REDUCTION_ENV = "PRISMAQUANT_BF16_REDUCED_PRECISION_REDUCTION"
#: The key a non-default setting occupies in an arithmetic identity or a
#: Stage A run identity. PyTorch's own name for the flag.
BF16_REDUCTION_FIELD = "allow_bf16_reduced_precision_reduction"
_OFF = "off"
_DEFAULT_SPELLING = "on"


class MatmulArithmeticRefused(ValueError):
    """A malformed setting or stamp, or the default spelled out."""


def bf16_reduction_from_environment(environ: Mapping) -> bool:
    """The flag value a launch declares: True when unset, False for ``off``."""
    text = environ.get(BF16_REDUCTION_ENV)
    if text is None:
        return True
    if text == _OFF:
        return False
    if text == _DEFAULT_SPELLING:
        raise MatmulArithmeticRefused(
            f"{BF16_REDUCTION_ENV}={text!r} spells the default; leave it unset")
    raise MatmulArithmeticRefused(
        f"{BF16_REDUCTION_ENV}={text!r}: the only value is {_OFF!r}; unset is the default")


def pin_matmul_arithmetic(environ: Mapping | None = None) -> bool:
    """Pin the matmul settings an entry point runs under; return the bf16 flag.

    Float32 matmul precision ``"highest"`` and TF32 off, as every entry point
    already pinned, plus the bf16 reduced-precision reduction flag from
    ``environ`` (the process environment when ``None``). Unset pins PyTorch's
    default, True, which is what these runs used before the flag was pinned.
    A malformed setting refuses before anything is pinned.
    """
    allow = bf16_reduction_from_environment(os.environ if environ is None else environ)
    import torch

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = allow
    return allow


def bf16_reduction_stamp() -> dict:
    """The identity fragment for the live flag: empty when it is on (default)."""
    import torch

    if bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction):
        return {}
    return {BF16_REDUCTION_FIELD: False}


def bf16_reduction_of(identity: Mapping) -> bool:
    """The flag an identity records: True when absent, False when stamped.

    A stamped True, or any value that is not the boolean False, is refused: a
    stamped default would give one arithmetic two spellings and two digests.
    """
    if BF16_REDUCTION_FIELD not in identity:
        return True
    if identity[BF16_REDUCTION_FIELD] is False:
        return False
    raise MatmulArithmeticRefused(
        f"{BF16_REDUCTION_FIELD}={identity[BF16_REDUCTION_FIELD]!r} is stamped; only "
        "False is ever stamped, and the default is spelled by the key's absence")
