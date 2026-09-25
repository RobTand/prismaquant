"""Digest encodings: one exact byte recipe per named profile (PQ #1301).

A digest in this tree is an identity. It is stored in a manifest, a receipt or
a shard name, and another run, host or tool must reproduce it byte for byte, so
two recipes that differ in any byte are two profiles. Each profile has one
spelling here, and every site that hashes, stores or compares with that
recipe uses it. A site whose bytes differ from every profile keeps its own
code; it is never moved onto a profile whose bytes are "close enough".

JSON profiles, all with sorted keys and compact ``(",", ":")`` separators:

- Round trip, UTF-8, strict (``canonical_json``, ``canonical_json_bytes``,
  ``canonical_json_sha256``, ``canonical_json_sha256_normalized``). The value
  is encoded, decoded and encoded again, so a non-string key is stringified
  before sorting, and a refusal is ``ValueError(f"{where} is not canonical
  JSON data")``.
- ``DIRECT_UTF8_STRICT``: one ``json.dumps`` with ``ensure_ascii=False`` and
  ``allow_nan=False``.
- ``DIRECT_ASCII_STRICT``: one ``json.dumps`` with ``ensure_ascii=True`` and
  ``allow_nan=False``.
- ``DIRECT_ASCII_LAX``: one ``json.dumps`` with ``ensure_ascii=True`` and
  ``allow_nan=True``, so ``NaN`` and ``Infinity`` are written, not refused.
- ``DIRECT_ASCII_LAX_DEFAULT_STR``: ``DIRECT_ASCII_LAX`` with ``default=str``.

The families are not interchangeable. The round trip and a direct encoding
differ on a mapping with integer keys (``{10: .., 9: ..}``); ASCII and UTF-8
differ on any non-ASCII character; strict and lax differ on ``NaN``. A direct
profile raises ``json``'s own ``TypeError`` or ``ValueError`` unchanged.

This module imports only the standard library and nothing from the package,
so a light tool can import it without pulling in torch.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import math


def canonical_json(value: object, *, where: str) -> object:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} is not canonical JSON data") from exc
    return json.loads(encoded)


def canonical_json_bytes(value: object, *, where: str) -> bytes:
    """The exact bytes ``canonical_json_sha256`` digests.

    Consumers that must *publish* canonical bytes -- not only hash them --
    read them here, so there is one canonical JSON encoding in the tree
    rather than a second spelling of the same ``json.dumps`` keywords.
    """
    canonical = canonical_json(value, where=where)
    try:
        return json.dumps(
            canonical,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} is not canonical JSON data") from exc


def canonical_json_sha256(value: object, *, where: str) -> str:
    return hashlib.sha256(canonical_json_bytes(value, where=where)).hexdigest()


#: The exact types ``json.loads`` produces. Exact, not ``isinstance``: a subclass
#: may override ``__str__``/``__repr__``, which would encode to bytes the stdlib
#: encoder would never write for parsed JSON.
_ALLOWED_SCALARS = (str, int, float, bool, type(None))

#: One encoder with the one canonical set of options, so the normalized path and
#: the generic path cannot drift apart in a keyword.
_CANONICAL_ENCODER = json.JSONEncoder(
    sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _require_normalized_json(value: object, *, where: str) -> None:
    """Refuse anything ``json.loads`` cannot produce, before encoding it.

    The generic path normalizes first: ``json.dumps`` stringifies an ``int``,
    ``float``, ``bool`` or ``None`` key, so a graph whose keys are not strings
    encodes to bytes that a reload would encode again differently. That
    normalization is why the generic path cannot simply be streamed, and it is
    why this path refuses such a graph instead of digesting different bytes. A
    ``tuple`` key is refused by ``json`` itself, in both paths.
    """
    active: set = set()
    stack = [(value, False)]
    while stack:
        item, leaving = stack.pop()
        if leaving:
            active.discard(id(item))
            continue
        if item is None or item is True or item is False:
            continue
        kind = type(item)
        if kind is str or kind is int:
            continue
        if kind is float:
            if not math.isfinite(item):
                raise ValueError(f"{where} is not canonical JSON data")
            continue
        if kind is list:
            marker = id(item)
            if marker in active:
                raise ValueError(f"{where} is not canonical JSON data: a cycle")
            active.add(marker)
            stack.append((item, True))
            stack.extend((child, False) for child in item)
            continue
        if kind is dict:
            marker = id(item)
            if marker in active:
                raise ValueError(f"{where} is not canonical JSON data: a cycle")
            active.add(marker)
            stack.append((item, True))
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError(
                        f"{where} is not normalized JSON: a mapping key is not a "
                        f"string but {type(key).__name__}")
                stack.append((child, False))
            continue
        raise ValueError(
            f"{where} is not normalized JSON: {kind.__name__} has no canonical "
            "JSON encoding")


def canonical_json_sha256_normalized(value: object, *, where: str) -> str:
    """``canonical_json_sha256``, for input that is already normalized JSON.

    ``json.loads`` output is normalized by construction: object keys are
    ``str``, containers are ``dict`` and ``list``, scalars are the JSON scalars.
    For that shape the ``dumps``->``loads`` in ``canonical_json`` is the
    identity on the encoding, so this validates the shape and then streams
    ``json.JSONEncoder.iterencode`` -- the stdlib encoder, with the same options
    -- into the digest, instead of building the encoded string, the second
    graph, and the second encoded string.

    The digest is the value ``canonical_json_sha256`` returns for the same
    value. ``tests/test_canonical_json_normalized.py`` holds that equality over
    Unicode, escapes, floats, negative zero, null, booleans and key order, and
    holds the refusals for input that is not normalized. A caller whose value is
    not already normalized JSON -- a non-string key, a tuple, a cycle -- must
    use ``canonical_json_sha256``; a tuple key is refused by ``json`` in both.
    """
    _require_normalized_json(value, where=where)
    digest = hashlib.sha256()
    for chunk in _CANONICAL_ENCODER.iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True)
class JsonProfile:
    """One direct JSON encoding: sorted keys, compact separators, these options."""

    name: str
    ensure_ascii: bool
    allow_nan: bool
    default: Callable[[object], object] | None = None

    def text(self, value: object) -> str:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=self.ensure_ascii,
            allow_nan=self.allow_nan,
            default=self.default,
        )

    def encoded(self, value: object) -> bytes:
        return self.text(value).encode("utf-8")

    def sha256(self, value: object) -> str:
        return hashlib.sha256(self.encoded(value)).hexdigest()


DIRECT_UTF8_STRICT = JsonProfile("direct-utf8-strict", ensure_ascii=False, allow_nan=False)
DIRECT_ASCII_STRICT = JsonProfile("direct-ascii-strict", ensure_ascii=True, allow_nan=False)
DIRECT_ASCII_LAX = JsonProfile("direct-ascii-lax", ensure_ascii=True, allow_nan=True)
DIRECT_ASCII_LAX_DEFAULT_STR = JsonProfile(
    "direct-ascii-lax-default-str", ensure_ascii=True, allow_nan=True, default=str)
