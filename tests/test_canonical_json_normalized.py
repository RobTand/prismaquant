"""The streaming digest is the same digest, for input that is already JSON.

``canonical_json_sha256`` normalizes before it hashes: one ``json.dumps``, a
``json.loads`` of that string, then a second ``json.dumps`` and the UTF-8
encode. On the joint loader's identity block those copies are what the new path
removes, so ``canonical_json_sha256_normalized`` validates the parsed shape and
streams the encoder's own chunks into the hash. The digest must not move, and
input the generic path would normalize must be refused rather than silently
digested as different bytes.
"""
from __future__ import annotations

import json

import pytest

from prismaquant.cost_stage_checkpoint import (
    canonical_json_sha256,
    canonical_json_sha256_normalized,
)

#: Values ``json.loads`` can produce, chosen for the encoder rules that are
#: easiest to get subtly wrong: escapes, non-BMP text, float repr, negative
#: zero, the extremes of the float range, integers past 2**63, key order.
ROUND_TRIP_CASES = [
    {},
    [],
    None,
    True,
    False,
    0,
    -1,
    2 ** 63,
    -2 ** 63 - 1,
    10 ** 40,
    0.0,
    -0.0,
    0.1,
    1e16,
    1e-300,
    5e-324,
    1.7976931348623157e308,
    "",
    "ascii",
    "quote\"backslash\\slash/",
    "control \u0000 \u001f \u007f",
    "caf\u00e9 na\u00efve",
    "emoji \U0001f600\U0001f1fa\U0001f1f8",
    "\u2028\u2029 separators",
    {"b": 1, "a": 2, "C": 3, "\u00e9": 4, "\U0001f600": 5},
    {"nested": {"deeper": [{"k": [1, 2.5, None, False]}]}},
    {"mixed": [1, "two", 3.0, None, True, {"z": [], "a": {}}]},
    [{"a": 1}, {"a": 1}],
]


@pytest.mark.parametrize("value", ROUND_TRIP_CASES,
                         ids=[repr(value)[:40] for value in ROUND_TRIP_CASES])
def test_the_streamed_digest_equals_the_generic_digest(value):
    assert canonical_json_sha256_normalized(value, where="case") == \
        canonical_json_sha256(value, where="case")


def test_a_parsed_document_round_trips_through_both_paths():
    """The real input shape is ``json.loads`` output, not a hand-built dict."""
    text = json.dumps({"units": {"a": {"menu": ["BF16", "R1000", "R1001"],
                                       "weight": {"shape": [16, 16]}},
                                 "b": {"menu": ["BF16"], "note": "\u00e9\u0007"}}},
                      ensure_ascii=False, sort_keys=False)
    parsed = json.loads(text)
    assert canonical_json_sha256_normalized(parsed, where="parsed") == \
        canonical_json_sha256(parsed, where="parsed")


def test_the_digest_changes_when_a_leaf_is_tampered_with():
    identity = {"units": {"a": {"menu": ["BF16", "R1000"]}}, "seal_source": "x"}
    before = canonical_json_sha256_normalized(identity, where="identity")
    tampered = json.loads(json.dumps(identity))
    tampered["units"]["a"]["menu"][1] = "R1001"
    after = canonical_json_sha256_normalized(tampered, where="identity")
    assert before != after
    assert after == canonical_json_sha256(tampered, where="identity")


def test_a_key_order_change_does_not_change_the_digest():
    left = {"b": 1, "a": {"y": 2, "x": 3}}
    right = {"a": {"x": 3, "y": 2}, "b": 1}
    assert canonical_json_sha256_normalized(left, where="order") == \
        canonical_json_sha256_normalized(right, where="order")


@pytest.mark.parametrize("value, match", [
    ({1: "a"}, "mapping key is not a string"),
    ({True: "a"}, "mapping key is not a string"),
    ({(1, 2): "a"}, "mapping key is not a string"),
    ({None: "a"}, "mapping key is not a string"),
    (("tuple",), "tuple has no canonical JSON encoding"),
    ({"a": (1, 2)}, "tuple has no canonical JSON encoding"),
    ({"a": {1, 2}}, "set has no canonical JSON encoding"),
    ({"a": object()}, "object has no canonical JSON encoding"),
    ({"a": float("nan")}, "not canonical JSON data"),
    ({"a": float("inf")}, "not canonical JSON data"),
    ({"a": float("-inf")}, "not canonical JSON data"),
    ({"a": b"bytes"}, "bytes has no canonical JSON encoding"),
])
def test_input_the_generic_path_would_normalize_is_refused(value, match):
    with pytest.raises(ValueError, match=match):
        canonical_json_sha256_normalized(value, where="bad")


def test_a_cycle_is_refused_rather_than_recursing():
    value = {"a": []}
    value["a"].append(value)
    with pytest.raises(ValueError, match="a cycle"):
        canonical_json_sha256_normalized(value, where="cycle")


def test_a_repeated_but_not_cyclic_container_is_accepted():
    """Sharing is not a cycle: the same sub-dict twice has a canonical digest."""
    shared = {"leaf": [1, 2]}
    value = {"a": shared, "b": shared}
    assert canonical_json_sha256_normalized(value, where="shared") == \
        canonical_json_sha256(value, where="shared")


def test_the_generic_helper_still_normalizes_a_non_string_key():
    """The generic path keeps stringifying an integer key; only the new one refuses."""
    assert canonical_json_sha256({1: "a"}, where="int key") == \
        canonical_json_sha256({"1": "a"}, where="string key")
    assert json.loads(json.dumps({1: "a"}, sort_keys=True)) == {"1": "a"}


class _Str(str):
    """A string subclass: ``json.loads`` never produces one for a parsed key."""


class _Int(int):
    """An int subclass whose ``__str__`` may diverge from the encoder's."""

    def __str__(self):
        return "999"


@pytest.mark.parametrize("value, match", [
    ({_Str("a"): 1}, "mapping key is not a string"),
    ({"a": _Int(1)}, "_Int has no canonical JSON encoding"),
    ({"a": _Str("x")}, "_Str has no canonical JSON encoding"),
    ([_Int(1)], "_Int has no canonical JSON encoding"),
])
def test_a_subclass_is_refused_because_loads_produces_exact_types(value, match):
    """Exact built-ins only: a subclass can encode to bytes the parse never wrote."""
    with pytest.raises(ValueError, match=match):
        canonical_json_sha256_normalized(value, where="subclass")
