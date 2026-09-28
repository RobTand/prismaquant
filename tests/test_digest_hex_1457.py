"""Frozen pre-consolidation SHA-256 validation outcomes (#1457)."""
from __future__ import annotations

import importlib
from pathlib import PurePosixPath

import pytest

from tests.golden_table import GoldenTable

GOLDEN = GoldenTable("digest_hex_1457")


class _String(str):
    pass


class _ShortLength(str):
    def __len__(self):
        return 0


class _ClaimedLength(str):
    def __len__(self):
        return 64


class _NonHexIteration(str):
    def __iter__(self):
        return iter("z" * 64)


class _HexIteration(str):
    def __iter__(self):
        return iter("a" * 64)


class _DifferentText(str):
    def __str__(self):
        return "not a digest"


class _FalsyString(str):
    def __bool__(self):
        return False


INPUTS = {
    "lower": "a" * 64,
    "digits": "0123456789abcdef" * 4,
    "upper": "A" * 64,
    "mixed": "a" * 32 + "A" * 32,
    "empty": "",
    "short": "a" * 63,
    "long": "a" * 65,
    "newline": "a" * 64 + "\n",
    "embedded_newline": "a" * 31 + "\n" + "a" * 32,
    "nul": "a" * 63 + "\x00",
    "space": " " + "a" * 64,
    "unicode": "é" * 64,
    "none": None,
    "integer": 1,
    "boolean": False,
    "float": 1.5,
    "bytes": b"a" * 64,
    "path": PurePosixPath("a" * 64),
    "nested": {"digest": [1, 2.5]},
    "subclass": _String("a" * 64),
    "short_length_subclass": _ShortLength("a" * 64),
    "claimed_length_subclass": _ClaimedLength("a"),
    "nonhex_iteration_subclass": _NonHexIteration("a" * 64),
    "hex_iteration_subclass": _HexIteration("z" * 64),
    "different_text_subclass": _DifferentText("a" * 64),
    "falsy_subclass": _FalsyString("a" * 64),
}

SITES = [
    "prismaquant.artifact_collection._sha256",
    "prismaquant.prepriced_cost._require_sha256",
    "prismaquant.prismasnap_checkpoint._require_sha256",
    "prismaquant.prismasnap_validation._require_sha256",
]
PATTERN_SITES = [
    "prismaquant.artifact_collection._SHA256",
    "prismaquant.prismasnap_checkpoint._SHA256",
    "prismaquant.prismasnap_validation._SHA256",
]


def _lookup(ref):
    module, _, name = ref.rpartition(".")
    return getattr(importlib.import_module(module), name)


@pytest.mark.parametrize("site", SITES)
@pytest.mark.parametrize("case", INPUTS)
def test_hex_wrapper_preserves_outcome_and_identity(site, case):
    value = INPUTS[case]
    function = _lookup(site)

    def call():
        result = function(value, where="fixture.field")
        return type(result).__name__, result, result is value

    GOLDEN.call(call)


@pytest.mark.parametrize("site", PATTERN_SITES)
@pytest.mark.parametrize("case", INPUTS)
def test_inline_pattern_binding_preserves_match_and_errors(site, case):
    pattern = _lookup(site)
    GOLDEN.call(lambda: bool(pattern.fullmatch(INPUTS[case])))


def test_regex_profiles_share_the_owner_and_ignore_python_string_hooks():
    from prismaquant.digests import SHA256_HEX, is_sha256hex

    assert all(_lookup(site) is SHA256_HEX for site in PATTERN_SITES)
    for case in ("lower", "subclass", "short_length_subclass",
                 "nonhex_iteration_subclass", "different_text_subclass",
                 "falsy_subclass"):
        assert is_sha256hex(INPUTS[case]) is True
    for case in ("upper", "newline", "bytes", "claimed_length_subclass",
                 "hex_iteration_subclass", "none"):
        assert is_sha256hex(INPUTS[case]) is False
