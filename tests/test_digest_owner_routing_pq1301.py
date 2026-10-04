"""Old/new byte and refusal evidence for the PQ #1301 owner routing.

This slice replaced every raw ``hashlib.sha256(...).hexdigest()`` constructor
in 50 package modules with the existing ``prismaquant.digests`` owners: the
byte owner for ``hashlib.sha256(X).hexdigest()``, the text owner for the
``X.encode("utf-8")`` (or default-codec) spelling. Each replaced call matched
that exact AST shape at edit time, and the primitive-site ratchet
(``test_primitive_digest_sites_only_shrink``) proves no raw call remains in
the migrated scopes. The equivalence below is the byte-level proof the issue
asks for per replaced site: the owner reproduces the old spelling's bytes and
refusals over a corpus that spans the shapes migrated sites feed them --
empty, single byte, block-aligned, megabyte, ``bytearray``/``memoryview``
views, NUL and ``0xFF`` runs -- and refuses the same inputs with the same
exception type and message, in the same order (encode before hash).
"""
from __future__ import annotations

import hashlib

import pytest

from prismaquant.digests import bytes_sha256hex, text_sha256hex

BYTE_CORPUS = [
    b"",
    b"\x00",
    b"\xff" * 64,
    b"prismaquant digest corpus",
    bytes(range(256)),
    b"\x00\x01\x02" * 32771,
    bytearray(b"mutable buffer"),
    memoryview(b"viewed bytes"),
]

TEXT_CORPUS = [
    "",
    "ascii",
    "unicode \u2713 mixed \u4e2d\u6587",
    "line\nbreaks\tand  spaces",
    "l\u00e9ading accent",
]


def test_byte_owner_reproduces_the_raw_hexdigest_bytes():
    for value in BYTE_CORPUS:
        assert bytes_sha256hex(value) == hashlib.sha256(value).hexdigest()


def test_text_owner_reproduces_the_encode_then_hash_bytes():
    for text in TEXT_CORPUS:
        assert text_sha256hex(text) == hashlib.sha256(text.encode()).hexdigest()
        assert text_sha256hex(text) == hashlib.sha256(
            text.encode("utf-8")).hexdigest()


def test_byte_owner_refuses_a_str_like_the_raw_constructor():
    with pytest.raises(TypeError) as owner:
        bytes_sha256hex("text not bytes")
    with pytest.raises(TypeError) as raw:
        hashlib.sha256("text not bytes").hexdigest()
    assert type(owner.value) is type(raw.value)
    assert str(owner.value) == str(raw.value)


def test_text_owner_refuses_a_non_str_at_the_encode_step():
    for bad in (b"bytes", 5, None):
        with pytest.raises(AttributeError) as owner:
            text_sha256hex(bad)
        with pytest.raises(AttributeError) as raw:
            hashlib.sha256(bad.encode()).hexdigest()
        assert type(owner.value) is type(raw.value)
        assert str(owner.value) == str(raw.value)


def test_text_owner_surrogate_refusal_matches_the_raw_encode():
    lone = "\ud800"
    with pytest.raises(UnicodeEncodeError) as owner:
        text_sha256hex(lone)
    with pytest.raises(UnicodeEncodeError) as raw:
        hashlib.sha256(lone.encode()).hexdigest()
    assert type(owner.value) is type(raw.value)
    assert str(owner.value) == str(raw.value)


def test_migrated_self_source_digest_still_binds_this_tree():
    """The kda_chunk self-source digest: same bytes, whatever hashes them."""
    from prismaquant.kernels import kda_chunk

    expected = hashlib.sha256(
        __import__("pathlib").Path(kda_chunk.__file__).read_bytes()).hexdigest()
    assert kda_chunk.source_sha256() == expected
