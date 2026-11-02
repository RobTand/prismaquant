"""Shared refusal vocabulary for the retired Gridbook codebook lane."""
from __future__ import annotations

import re


# The retired Gridbook lane's codebook rungs (NVFP4_CB_K<k>, FP8_CB_K<k>).
# They are no longer registered. A stale artifact that names one gets this
# refusal, never a silent skip: RetiredFormatError is deliberately NOT a
# KeyError, so a reader that tolerates unknown names with ``except KeyError``
# still refuses a retired one.
RETIRED_CODEBOOK_FORMAT_RE = re.compile(r"(?:NVFP4_CB_K|FP8_CB_K)\d+")
RETIRED_CODEBOOK_ARCHIVE = "archive/gridbook_lane_2026-09-02"


#: Shape of a retired codebook rung name (the Gridbook lane, archived
#: 2026-09-02; remnant cleanup 2026-09-25, #1304). A shape test only,
#: kept here so the torch-free readers (``schemas``, ``layer_config``)
#: can spot one; ``retired_formats`` owns the shared refusal.
RETIRED_CODEBOOK_NAME_RE = re.compile(r"^(?:NVFP4_CB_K|FP8_CB_K)\d+$")


class RetiredFormatError(ValueError):
    """A format name that belongs to a retired lane and is no longer served."""


def retired_codebook_error(name: str) -> RetiredFormatError:
    """The same refusal for registry lookups and torch-free artifact readers."""
    return RetiredFormatError(
        f"format {name!r} belongs to the retired Gridbook codebook "
        f"lane (NVFP4-CB / FP8-CB), removed on 2026-09-02; its "
        f"format, cost and render code was archived on 2026-09-25 "
        f"(#1304). A cost table, layer config or selection that "
        f"names it can no longer be priced, rendered or exported "
        f"from this repository. See {RETIRED_CODEBOOK_ARCHIVE}/"
        f"README.md."
    )
