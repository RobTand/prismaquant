"""Digest encodings: one exact byte recipe per named profile (PQ #1301).

A digest in this tree is an identity. It is stored in a manifest, a receipt or
a shard name, and another run, host or tool must reproduce it byte for byte, so
two recipes that differ in any byte are two profiles. Each profile has one
spelling here, and every site that hashes, stores or compares with that
recipe uses it. A site whose bytes differ from every profile keeps its own
code; it is never moved onto a profile whose bytes are "close enough".

JSON profiles, all with sorted keys. Separators are compact ``(",", ":")``
unless the profile states otherwise:

- ``DIRECT_ASCII_SPACED_LAX``: default-spaced ``(", ", ": ")`` separators,
  ``ensure_ascii=True``, ``allow_nan=True`` and no fallback serializer. This
  is direct JSON, not a round trip; its inherited spaces are identity bytes.
- ``DIRECT_ASCII_SPACED_STRICT``: the same direct, ASCII-escaped, default-spaced
  recipe with ``allow_nan=False`` and no fallback serializer. Snapshot readers
  retain their own ``json.loads`` round trip and publication writers their LF.
- ``DIRECT_UTF8_INDENT2_STRICT``: two-space indentation, UTF-8 characters,
  strict nonfinite refusal and no fallback serializer. No final LF is added.
- ``DIRECT_ASCII_INDENT2_LAX``: two-space indentation, ``(",", ": ")``
  separators, ASCII escaping, lax nonfinite values and no fallback serializer.
  File writers retain their own final LF and publication policy.

- Round trip, UTF-8, strict (``canonical_json``, ``canonical_json_bytes``,
  ``canonical_json_sha256``, ``canonical_json_sha256_normalized``). The value
  is encoded, decoded and encoded again, so a non-string key is stringified
  before sorting, and a refusal is ``ValueError(f"{where} is not canonical
  JSON data")``.
- ``DIRECT_UTF8_STRICT``: one ``json.dumps`` with ``ensure_ascii=False`` and
  ``allow_nan=False``.
- ``DIRECT_UTF8_LAX``: one direct encoding with ``ensure_ascii=False`` and
  ``allow_nan=True``, compact separators and no fallback serializer. Nonfinite
  values remain JSON tokens; encoding to bytes uses strict UTF-8.
- ``DIRECT_ASCII_STRICT``: one ``json.dumps`` with ``ensure_ascii=True`` and
  ``allow_nan=False``.
- ``DIRECT_ASCII_LAX``: one ``json.dumps`` with ``ensure_ascii=True`` and
  ``allow_nan=True``, so ``NaN`` and ``Infinity`` are written, not refused.
- ``DIRECT_ASCII_LAX_DEFAULT_STR``: ``DIRECT_ASCII_LAX`` with ``default=str``.

Byte profiles, lowercase-hex SHA-256 except the native Git SHA-1 profile:

- ``git_blob_sha1hex``: native Git blob SHA1 over its ASCII type/length/NUL
  header and exact owned payload, for independently published auxiliary IDs.
- ``bytes_sha256hex``: the bytes as given (``bytes``, ``bytearray`` or
  ``memoryview``).
- ``text_sha256hex``: the text encoded as strict UTF-8, so a lone surrogate
  raises ``UnicodeEncodeError``.
- ``newline_utf8_bytes`` / ``newline_utf8_sha256``: strings in caller order,
  joined with LF, strict UTF-8, with no added final LF. Embedded newlines are
  not escaped. ``sorted_newline_utf8_sha256`` sorts the input first. Neither
  profile validates names or removes duplicates.
- ``file_sha256hex``: a file's bytes, read in ``block_size`` pieces
  (``FILE_BLOCK_BYTES`` unless the site keeps its own). The block size changes
  only how the file is read, never the digest. The path may be a ``str`` or a
  ``PathLike``; a missing path or a directory raises what ``open`` raises.

- ``length_framed_bytes_sha256``: caller-owned prefix followed by caller-ordered
  raw byte frames, each preceded by its eight-byte big-endian byte length.
  No normalization, sorting, reconstruction or final trailer. Domain tags stay
  with the caller and are not interchangeable.
- ``LengthFramedSourceSha256``: incremental records in caller order, each
  strict UTF-8 name preceded by its four-byte big-endian byte length, then a
  payload preceded by its eight-byte big-endian byte length. No sorting,
  reconstruction, separators or final trailer; those stay with the caller.

Checkpoint JSON stream:

- ``checkpoint_json_sha256``: ``DIRECT_UTF8_STRICT`` bytes partitioned at
  depth two; shallow dict keys require strings, deeper leaves preserve the
  direct encoder's accepted keys. The caller retains its key-error class.
  No normalization or whole-graph precheck changes first-error order.

Pickle profile:

- ``canonical_pickle_bytes``: ``pickle.dumps`` at explicit protocol 4 of a
  rebuilt copy, preserving the historical Python 3.12 encoding (PQ #1450).
  In the copy, every plain ``dict``, ``list`` and ``tuple`` is a fresh object,
  and every equal ``str`` or ``bytes`` is one shared object. A pickle writes
  a shared object once and refers back to it, so its bytes
  depend on which objects the caller happened to share. This profile makes
  that a function of the value: the same plain-typed value, in the same key
  order, gives the same bytes whichever path built it (PQ #1403). Other types
  are pickled as given, and a container that contains itself is refused.

The JSON families are not interchangeable. The round trip and a direct encoding
differ on a mapping with integer keys (``{10: .., 9: ..}``); ASCII and UTF-8
differ on any non-ASCII character; strict and lax differ on ``NaN``. A direct
profile raises ``json``'s own ``TypeError`` or ``ValueError`` unchanged.

This module imports only the standard library and nothing from the package,
so a light tool can import it without pulling in torch.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
import re
from typing import BinaryIO, Protocol


# A regex checks the underlying text, not a str subclass's Python length or
# iteration hooks. Callers retain their own coercion, exact-type and errors.
SHA256_HEX = re.compile(r"[0-9a-f]{64}")


def is_sha256hex(value: object) -> bool:
    """Whether value is str (including subclasses) containing 64 lowercase hex digits."""
    return isinstance(value, str) and SHA256_HEX.fullmatch(value) is not None


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


class _JsonChunkEncoder(Protocol):
    def iterencode(self, value: object) -> Iterable[str]: ...


def _stream_sha256(encoder: _JsonChunkEncoder, value: object) -> str:
    """Stream ``encoder.iterencode`` chunks, as UTF-8, into one SHA-256.

    The one streaming recipe for every digest owner: chunks arrive as
    ``str`` in the encoder's own spelling and are UTF-8-encoded exactly as a
    caller feeding ``iterencode`` into a hasher does.
    """
    digest = hashlib.sha256()
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


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
    return _stream_sha256(_CANONICAL_ENCODER, value)


@dataclass(frozen=True)
class JsonProfile:
    """One direct JSON encoding: sorted keys and exactly these options.

    Separators default to compact; a spaced profile must name its own pair.
    """

    name: str
    ensure_ascii: bool
    allow_nan: bool
    default: Callable[[object], object] | None = None
    separators: tuple[str, str] = (",", ":")
    indent: int | None = None

    def _encoder(self) -> json.JSONEncoder:
        """The stdlib encoder with exactly this profile's options."""
        return json.JSONEncoder(
            sort_keys=True,
            separators=self.separators,
            indent=self.indent,
            ensure_ascii=self.ensure_ascii,
            allow_nan=self.allow_nan,
            default=self.default,
        )

    def text(self, value: object) -> str:
        return self._encoder().encode(value)

    def encoded(self, value: object) -> bytes:
        return self.text(value).encode("utf-8")

    def sha256(self, value: object) -> str:
        return hashlib.sha256(self.encoded(value)).hexdigest()

    def sha256_streamed(self, value: object) -> str:
        """Hash this profile's encoding without materializing the full text.

        Byte-identical to ``sha256`` for values this profile accepts: the
        encoder options are the profile's own, streamed chunk by chunk in
        UTF-8 exactly as a caller feeding ``iterencode`` into a hasher does.
        """
        return _stream_sha256(self._encoder(), value)


DIRECT_UTF8_STRICT = JsonProfile("direct-utf8-strict", ensure_ascii=False, allow_nan=False)
DIRECT_UTF8_LAX = JsonProfile("direct-utf8-lax", ensure_ascii=False, allow_nan=True)
DIRECT_ASCII_STRICT = JsonProfile("direct-ascii-strict", ensure_ascii=True, allow_nan=False)
DIRECT_ASCII_LAX = JsonProfile("direct-ascii-lax", ensure_ascii=True, allow_nan=True)
DIRECT_ASCII_LAX_DEFAULT_STR = JsonProfile(
    "direct-ascii-lax-default-str", ensure_ascii=True, allow_nan=True, default=str)
DIRECT_ASCII_SPACED_LAX = JsonProfile(
    "direct-ascii-spaced-lax", ensure_ascii=True, allow_nan=True,
    separators=(", ", ": "))
DIRECT_ASCII_SPACED_STRICT = JsonProfile(
    "direct-ascii-spaced-strict", ensure_ascii=True, allow_nan=False,
    separators=(", ", ": "))
DIRECT_UTF8_INDENT2_STRICT = JsonProfile(
    "direct-utf8-indent2-strict", ensure_ascii=False, allow_nan=False,
    separators=(",", ": "), indent=2)
DIRECT_ASCII_INDENT2_LAX = JsonProfile(
    "direct-ascii-indent2-lax", ensure_ascii=True, allow_nan=True,
    separators=(",", ": "), indent=2)


class _CheckpointJsonEncoder:
    """The inherited two-level partition, with the reader's own key error."""

    def __init__(self, error: type[Exception]):
        self.error = error

    def iterencode(self, value: object) -> Iterable[str]:
        return self._chunks(value, 0)

    def _chunks(self, value: object, depth: int):
        encode = DIRECT_UTF8_STRICT._encoder().encode
        if depth >= 2 or not isinstance(value, (dict, list)):
            yield encode(value)
            return
        if isinstance(value, dict):
            yield "{"
            for index, key in enumerate(sorted(value)):
                if not isinstance(key, str):
                    raise self.error("checkpoint identity has a non-string key")
                yield ("," if index else "") + encode(key) + ":"
                yield from self._chunks(value[key], depth + 1)
            yield "}"
        else:
            yield "["
            for index, item in enumerate(value):
                if index:
                    yield ","
                yield from self._chunks(item, depth + 1)
            yield "]"


def checkpoint_json_sha256(value: object, *, error: type[Exception] = ValueError) -> str:
    """Strict direct UTF-8 JSON, partitioned at depth two without normalization.

    Shallow dict keys must be strings; deeper leaves retain the stdlib direct
    encoder's acceptance and errors. The caller supplies its existing key-error
    class. Chunk boundaries, first-error order and UTF-8 error positions stay
    with this inherited checkpoint profile rather than a whole-graph precheck.
    """
    return _stream_sha256(_CheckpointJsonEncoder(error), value)


def canonical_pickle_bytes(value: object) -> bytes:
    import pickle

    shared: dict = {}
    open_containers: set[int] = set()

    def rebuild(item):
        kind = type(item)
        if kind is str or kind is bytes:
            return shared.setdefault((kind, item), item)
        if kind is dict or kind is list or kind is tuple:
            if id(item) in open_containers:
                raise ValueError("canonical_pickle_bytes refuses a container that contains itself")
            open_containers.add(id(item))
            try:
                if kind is dict:
                    return {rebuild(key): rebuild(entry) for key, entry in item.items()}
                rebuilt = [rebuild(entry) for entry in item]
                return rebuilt if kind is list else tuple(rebuilt)
            finally:
                open_containers.discard(id(item))
        return item

    return pickle.dumps(rebuild(value), protocol=4)


#: The read size ``file_sha256hex`` uses when a site does not keep its own.
FILE_BLOCK_BYTES = 8 << 20

#: The guarded source hash's read block (``tessera_calibration_cache.sha256``).
#: With ``release_read_pages`` it is also the page window: each block's pages
#: are advised away right after the digest consumes it, so one hash holds at
#: most this block and its pages. Admission charges exactly that
#: (``autoscale.selected_anchor_resources``, RobTand/prismaquant#1491), so the
#: two read the same number from here.
SOURCE_HASH_BLOCK_BYTES = 16 * 1024**2


def length_framed_bytes_sha256(frames: Iterable[bytes], *, prefix: bytes) -> str:
    """Prefix plus ordered u64-BE-length/raw-byte frames, consumed once."""
    digest = hashlib.sha256(prefix)
    for raw in frames:
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


class LengthFramedSourceSha256:
    """Incremental be32/name/be64/payload source profile; caller owns order."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()

    def update(self, name: str, payload: bytes) -> None:
        encoded = name.encode("utf-8")
        self._digest.update(len(encoded).to_bytes(4, "big"))
        self._digest.update(encoded)
        self._digest.update(len(payload).to_bytes(8, "big"))
        self._digest.update(payload)

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


class LegacyNulSourceSha256:
    """Inherited name/NUL/payload/NUL profile, not an unambiguous file map.

    Caller ordering and validation stay outside the owner. NUL payloads can
    imitate entry boundaries (#1762); changing that protocol is not dedup.
    """

    def __init__(self) -> None:
        self._digest = hashlib.sha256()

    def update(self, name: str, payload: bytes) -> None:
        encoded = name.encode("utf-8")
        self._digest.update(encoded)
        self._digest.update(b"\0")
        self._digest.update(payload)
        self._digest.update(b"\0")

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


SOURCE_TREE_V1 = "prismaquant.source_tree.v1"
SOURCE_TREE_V2 = "prismaquant.source_tree.v2"


def source_tree_profiles(records: Iterable[tuple[str, bytes]]) -> dict[str, str]:
    """Preserve caller-ordered NUL v1; add tagged u64/u64 byte-sorted v2.

    The caller still owns its suffix roster and relative-name root. NUL is
    legitimate content: only v1 treats it as a separator. v2 starts with its
    ASCII profile name and NUL, then unsigned eight-byte big-endian lengths
    for both the UTF-8 name and the raw content, with no trailing separator.
    """
    entries = list(records)
    if len({name for name, _ in entries}) != len(entries):
        raise ValueError("source profiles require unique file names")
    legacy = LegacyNulSourceSha256()
    encoded = []
    for name, raw in entries:
        if not isinstance(name, str) or not isinstance(raw, bytes):
            raise TypeError("source profiles require UTF-8 names and byte contents")
        legacy.update(name, raw)
        encoded.append((name.encode("utf-8"), raw))
    def frames():
        for name, raw in sorted(encoded, key=lambda entry: entry[0]):
            yield name
            yield raw

    framed = length_framed_bytes_sha256(
        frames(), prefix=SOURCE_TREE_V2.encode("ascii") + b"\0")
    return {SOURCE_TREE_V1: legacy.hexdigest(), SOURCE_TREE_V2: framed}


def compare_source_profiles(left: Mapping[str, str], right: Mapping[str, str]) -> dict[str, str]:
    """Compare v2 when both declare it, otherwise explicitly report v1.

    An advertised malformed/unknown profile is never downgraded to legacy.
    Matching v1 cannot mask a mismatch in v2; matching v2 need not compare
    the caller-ordered legacy transcript across the two sides.
    """
    allowed = {SOURCE_TREE_V1, SOURCE_TREE_V2}
    for profiles in (left, right):
        if (not isinstance(profiles, Mapping) or SOURCE_TREE_V1 not in profiles
                or not set(profiles) <= allowed
                or any(not is_sha256hex(value) for value in profiles.values())):
            raise ValueError("source profiles require labelled, exact known SHA256 values")
    profile = SOURCE_TREE_V2 if SOURCE_TREE_V2 in left and SOURCE_TREE_V2 in right else SOURCE_TREE_V1
    if left[profile] != right[profile]:
        raise ValueError(f"{profile}: source profile mismatch")
    return {"status": "framed_v2" if profile == SOURCE_TREE_V2 else "legacy_framing",
            "profile": profile, "sha256": left[profile]}


def git_blob_sha1hex(data: bytes | bytearray | memoryview) -> str:
    """Native Git object ID: SHA1 of ``b'blob ' + size + b'\\0' + data``.

    Publisher Git auxiliaries name this recipe, not plain payload SHA1. The
    caller supplies its already-owned contiguous bytes; no file is opened.
    """
    digest = hashlib.sha1(b'blob ' + str(len(data)).encode('ascii') + b'\0')
    digest.update(data)
    return digest.hexdigest()


def bytes_sha256hex(data: bytes | bytearray | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def text_sha256hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def newline_utf8_bytes(lines: Iterable[str]) -> bytes:
    """Caller order, LF separators, strict UTF-8, no added trailing LF."""
    return "\n".join(lines).encode("utf-8")


def newline_utf8_sha256(lines: Iterable[str]) -> str:
    return bytes_sha256hex(newline_utf8_bytes(lines))


def sorted_newline_utf8_sha256(names: Iterable[str]) -> str:
    """The sorted-name roster recipe; validation stays with the caller."""
    return newline_utf8_sha256(sorted(names))


def file_sha256hex(path: str | os.PathLike, *, block_size: int = FILE_BLOCK_BYTES) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def file_digest_sha256hex(handle: BinaryIO) -> str:
    """Digest an already-open, stat-fenced file handle in one stream.

    ``file_sha256hex`` opens the path itself; a site that fenced its read with
    ``os.open`` + stat comparisons around the hash already holds the pinned
    descriptor and must not reopen the path. This is the one spelling for that
    shape: the stdlib streaming ``file_digest`` over the caller's handle,
    reading it from its current position to EOF.
    """
    return hashlib.file_digest(handle, "sha256").hexdigest()


def indent2_json_file_bytes(value: object) -> bytes:
    """The pretty capture-file JSON spelling plus one trailing LF.

    Sorted keys, two-space indent, strict about non-finite numbers, default
    separators, UTF-8, and exactly one LF after the text.
    """
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8")


def hex_chain_sha256hex(left: str, right: str) -> str:
    """The ordered hex-identity chain: concatenate two hex strings, hash once.

    ``merge_load_execution``/``fold_load_receipt`` fold receipts in load order;
    the concatenated ASCII hex string hashed as UTF-8 is the whole recipe.
    """
    return text_sha256hex(left + right)
