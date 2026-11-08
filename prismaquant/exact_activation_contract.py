"""Metadata shared by exact activation writers and finite recovery planning."""
from __future__ import annotations

from dataclasses import dataclass, field

EXACT_ACTIVATION_SCHEMA = 'prismaquant.exact_activation_entry.v1'


@dataclass(frozen=True)
class ExactActivationReference:
    """An immutable exact tensor receipt, never an activation sample/cache."""

    path: str
    name: str
    metadata_json: str
    shape: tuple[int, ...]
    dtype: str
    tensor_bytes: int
    file_bytes: int
    sha256: str
    writer_file_identity: tuple[int, ...] | None = field(default=None, compare=False, repr=False)
