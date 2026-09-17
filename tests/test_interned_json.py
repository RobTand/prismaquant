"""The joint checkpoint's reader is ``json.load`` plus string sharing.

``json.loads`` makes a new ``str`` for every occurrence, so the merged
checkpoint's 205,243,544 menu names are 205,243,544 objects holding 5,635
distinct values. ``prismaquant.interned_json`` reads the file with the standard
library's own reader and one ``object_pairs_hook`` that shares equal strings
and equal lists of strings -- never the lists themselves. These cases hold the
result equal to ``json.loads`` over escapes, Unicode, number formats, duplicate
keys, whitespace and refusal, hold exact value types and the canonical digest
unmoved, and hold the sharing to things that cannot be observed through a
mutation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from prismaquant.cost_stage_checkpoint import (
    canonical_json_sha256_normalized)
from prismaquant.interned_json import interning_pairs_hook, load_json_file

DOCUMENTS = [
    "{}",
    "[]",
    "null",
    "true",
    "false",
    "0",
    "-0",
    "0.0",
    "1",
    "-17",
    "1.5",
    "-2.25e-3",
    "123456789012345678901234567890",
    "1.7976931348623157e308",
    "5e-324",
    '"a string"',
    '""',
    '"\\"quoted\\" and \\\\ escaped"',
    '"\\u00e9\\u00e8 \\u4e2d\\u6587 \\ud83d\\ude00"',
    "\"raw \u00e9\u4e2d\U0001f600 utf-8\"",
    '"\\t\\n\\r\\b\\f tab and newline"',
    '{"a": 1, "b": [1, 2, 3], "c": {"d": null, "e": true}}',
    '{"dup": 1, "dup": 2}',
    '{"nested": [{"a": [{"b": [1, [2, [3, [4]]]]}]}]}',
    '  {  "a"  :  [ 1 ,  2 ]  }  \n',
    '{"empty": {}, "empties": [[], {}, [[]]]}',
    '{"keys": {"with space": 1, "\\u00e9": 2, "": 3}}',
    '[true, false, null, 0, "", {}, []]',
    '{"deep": ' + "[" * 40 + "1" + "]" * 40 + "}",
]


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "document.json"
    path.write_bytes(text.encode("utf-8"))
    return path


@pytest.mark.parametrize("text", DOCUMENTS)
def test_every_document_parses_exactly_what_json_loads_parses(tmp_path, text):
    path = _write(tmp_path, text)
    expected = json.loads(text)
    found = load_json_file(path)
    assert found == expected
    assert type(found) is type(expected)
    # The canonical digest is also the exact-type contract: a value the reader
    # moved, or a subclass it invented, refuses here.
    assert (canonical_json_sha256_normalized(found, where="case")
            == canonical_json_sha256_normalized(expected, where="case"))


def test_non_finite_constants_are_the_stdlib_ones(tmp_path):
    text = '{"nan": NaN, "pos": Infinity, "neg": -Infinity}'
    found = load_json_file(_write(tmp_path, text))
    expected = json.loads(text)
    assert math.isnan(found["nan"]) and math.isnan(expected["nan"])
    assert found["pos"] == expected["pos"] == math.inf
    assert found["neg"] == expected["neg"] == -math.inf
    over = '"1e999"'
    assert load_json_file(_write(tmp_path, over)) == json.loads(over)


@pytest.mark.parametrize("text", [
    "",
    "   \n ",
    "{",
    "}",
    "[1, 2",
    "[1, 2,]",
    '{"a": 1,}',
    "{a: 1}",
    "{1: 2}",
    '{"a" 1}',
    '{"a": }',
    "01",
    "tru",
    '"unterminated',
    '{"a": 1} {"b": 2}',
    '{"a": 1} trailing',
    "[1] extra",
    '"a" "b"',
])
def test_every_refusal_json_loads_makes_is_a_refusal_here(tmp_path, text):
    path = _write(tmp_path, text)
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)
    with pytest.raises(json.JSONDecodeError):
        load_json_file(path)


def test_a_bom_is_refused_like_the_text_json_loads_saw(tmp_path):
    # The loader this reader replaces did `json.loads(path.read_text())`, which
    # refuses a BOM as the first character. Accepting it here would be a
    # relaxation of an input gate, so it is refused.
    text = '\ufeff{"a": [1]}'
    path = tmp_path / "bom.json"
    path.write_bytes(text.encode("utf-8"))
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)
    with pytest.raises(json.JSONDecodeError):
        load_json_file(path)


def test_a_handle_and_a_path_read_the_same_document(tmp_path):
    text = json.dumps({"raw": "\u00e9\U0001f600", "n": [1, 2.5]})
    path = _write(tmp_path, text)
    expected = json.loads(text)
    assert load_json_file(path) == expected
    with path.open("r", encoding="utf-8") as handle:
        assert load_json_file(handle) == expected


def test_object_pairs_hook_is_applied_where_the_decoder_applies_it(tmp_path):
    def reject_duplicates(pairs):
        seen = set()
        for key, _ in pairs:
            if key in seen:
                raise ValueError(f"duplicate key {key!r}")
            seen.add(key)
        return dict(pairs)

    class Marked(dict):
        """The hook's return value is what an object becomes."""

    with pytest.raises(ValueError, match="duplicate key"):
        load_json_file(_write(tmp_path, '{"a": 1, "a": 2}'),
                       object_pairs_hook=reject_duplicates)
    found = load_json_file(_write(tmp_path, '{"a": 1, "b": [2]}'),
                           object_pairs_hook=lambda pairs: Marked(pairs))
    assert type(found) is Marked and found == {"a": 1, "b": [2]}
    assert type(found["b"]) is list


def _typed(value):
    """Every value with its exact type, so ``1`` and ``True`` cannot compare equal."""
    if type(value) is dict:
        return {key: _typed(item) for key, item in value.items()}
    if type(value) is list:
        return [_typed(item) for item in value]
    if type(value) is bool:
        return ("bool", value)
    if type(value) is int:
        return ("int", value)
    if type(value) is float:
        return ("float", value)
    if value is None:
        return ("null", None)
    return (type(value).__name__, value)


#: Lists that hold something besides ``str``. As *values* several of these are
#: equal to each other -- ``['x', 1]``, ``['x', True]`` and ``['x', 1.0]`` --
#: and ``['x', {}]`` cannot even be hashed, so a value-keyed memo must not see
#: them at all. Each document must round-trip with its own members and types.
MIXED_MEMBER_DOCUMENTS = [
    '{"menu": ["x", {}]}',
    '{"menu": ["x", {"nested": [1, "y"]}]}',
    '{"menu": ["x", [1, 2]]}',
    '{"menu": ["x", [["deep"]]]}',
    '{"menu": ["x", 1]}',
    '{"menu": ["x", true]}',
    '{"menu": ["x", 1.0]}',
    '{"menu": ["x", null]}',
    '{"menu": ["x", "y", {"a": [1, true]}]}',
    '[["x", 1], ["x", true], ["x", 1.0], ["x", {}]]',
    '{"units": {"a": ["x", 1], "b": ["x", true], "c": ["x", 1.0]}}',
]


@pytest.mark.parametrize("text", MIXED_MEMBER_DOCUMENTS)
def test_a_list_holding_anything_but_strings_keeps_its_own_members(tmp_path, text):
    found = load_json_file(_write(tmp_path, text))
    expected = json.loads(text)
    assert found == expected
    assert _typed(found) == _typed(expected)
    assert (canonical_json_sha256_normalized(found, where="mixed")
            == canonical_json_sha256_normalized(expected, where="mixed"))


def test_equal_but_differently_typed_lists_are_not_substituted_for_each_other(tmp_path):
    # The defect this holds shut: ``['x', 1]``, ``['x', True]`` and
    # ``['x', 1.0]`` are the same *value* as memo keys, and ``['x', {}]``
    # cannot be a key at all. Sharing is restricted to lists of exact strings.
    text = json.dumps({"units": {"int": ["x", 1], "bool": ["x", True],
                                 "float": ["x", 1.0], "dict": ["x", {"n": 1}],
                                 "shipped": ["x", "x", "y"]}})
    expected = json.loads(text)
    found = load_json_file(_write(tmp_path, text))
    units = found["units"]
    assert type(units["int"][1]) is int and units["int"][1] == 1
    assert type(units["bool"][1]) is bool and units["bool"][1] is True
    assert type(units["float"][1]) is float and units["float"][1] == 1.0
    assert type(units["dict"][1]) is dict and units["dict"][1] == {"n": 1}
    assert _typed(found) == _typed(expected)
    # ``["x", 1] == ["x", True]`` is true in Python, so the digest is the
    # witness that the document still says what it said.
    assert (canonical_json_sha256_normalized(found, where="found")
            == canonical_json_sha256_normalized(expected, where="expected"))
    # The all-string list next to them is still shared, member by member.
    assert units["shipped"][1] is units["shipped"][0]


def _menu_document(units, menu, *, width=3):
    """``units`` records that each hold the same ``menu``, except the last one."""
    records = {}
    for index in range(units):
        name = f"model.language_model.layers.{index}.mlp.down_proj"
        items = [f"{menu}_{position}" for position in range(width)]
        if index == units - 1:
            items = list(reversed(items))
        records[name] = {"hessian": {"sha256": "a" * 64},
                         "input_global_scale": 1.5 + index,
                         "menu": items}
    return {"identity": {"units": records}, "units": [],
            "schema": "fixture", "stage": "fixture"}


def _distinct_menu_strings(value):
    return {id(item) for unit in value["identity"]["units"].values()
            for item in unit["menu"]}


def test_sharing_removes_the_duplicate_menu_objects(tmp_path):
    # The defect in one case: the plain parse holds one object per occurrence,
    # the sharing parse holds one per distinct string. Nothing else moves.
    units, width = 40, 8
    text = json.dumps(_menu_document(units, "TESSERA_E4M3_K2", width=width))
    plain = json.loads(text)
    shared = load_json_file(_write(tmp_path, text))
    assert shared == plain
    assert (canonical_json_sha256_normalized(shared, where="shared")
            == canonical_json_sha256_normalized(plain, where="plain"))
    assert len(_distinct_menu_strings(plain)) == units * width
    assert len(_distinct_menu_strings(shared)) == width


def test_sharing_never_hands_two_units_the_same_list(tmp_path):
    units, width = 6, 5
    text = json.dumps(_menu_document(units, "TESSERA_E4M3_K2", width=width))
    shared = load_json_file(_write(tmp_path, text))
    first, second = list(shared["identity"]["units"].values())[:2]
    assert first["menu"] is not second["menu"]
    assert first["menu"][0] is second["menu"][0]
    first["menu"][0] = "mutated"
    assert second["menu"][0] != "mutated"
    assert first["menu"][1:] == second["menu"][1:]


def test_a_menu_that_differs_stays_a_different_menu(tmp_path):
    units, width = 5, 4
    text = json.dumps(_menu_document(units, "TESSERA_E4M3_K2", width=width))
    shared = load_json_file(_write(tmp_path, text))
    *same, different = list(shared["identity"]["units"].values())
    assert different["menu"] == list(reversed(same[0]["menu"]))
    assert different["menu"] != same[0]["menu"]
    assert different["menu"] is not same[0]["menu"]
    assert sorted(different["menu"]) == sorted(same[0]["menu"])
    assert all(left == right for left, right in zip(same[0]["menu"], same[1]["menu"]))


def test_interning_can_be_turned_off_and_changes_only_identity(tmp_path):
    units, width = 5, 4
    text = json.dumps(_menu_document(units, "TESSERA_E4M3_K2", width=width))
    path = _write(tmp_path, text)
    plain = load_json_file(path, intern_strings=False)
    shared = load_json_file(path)
    assert plain == shared
    assert len(_distinct_menu_strings(plain)) == units * width
    assert (canonical_json_sha256_normalized(plain, where="plain")
            == canonical_json_sha256_normalized(shared, where="shared"))


def test_the_memos_are_bounded_so_all_distinct_strings_do_not_grow_a_copy(tmp_path):
    text = json.dumps({"values": [f"unique-{index}" for index in range(500)]})
    path = _write(tmp_path, text)
    stats = {}
    found = load_json_file(path, string_memo_limit=4, list_memo_limit=2, stats=stats)
    assert found == json.loads(text)
    assert stats["strings_seen"] == 4


def test_the_reader_reports_what_it_shared(tmp_path):
    text = json.dumps(_menu_document(20, "TESSERA_E4M3_K2", width=6))
    stats = {}
    found = load_json_file(_write(tmp_path, text), stats=stats)
    assert found == json.loads(text)
    assert stats["lists_memoized"] == 2      # the shared menu and its reverse
    assert stats["lists_shared"] == 18       # every other unit's menu
    # The counters are evidence of what the hook did, not a tally of the
    # document: the list memo answers a repeat menu without re-interning its
    # members one at a time, which is the whole point of having it.
    assert stats["strings_shared"] > 0


def test_the_published_hook_is_the_whole_reader(tmp_path):
    # What the architecture decision turns on: plain ``json.load`` with
    # ``object_pairs_hook=interning_pairs_hook()`` already removes the
    # duplicate menu objects, so the reader is stdlib plus this hook.
    units, width = 6, 5
    path = _write(tmp_path, json.dumps(_menu_document(units, "TESSERA_E4M3_K2", width=width)))
    with path.open("r", encoding="utf-8") as handle:
        found = json.load(handle, object_pairs_hook=interning_pairs_hook())
    assert len(_distinct_menu_strings(found)) == width
    first, second = list(found["identity"]["units"].values())[:2]
    assert first["menu"] is not second["menu"]
    assert first["menu"][0] is second["menu"][0]
    # And it is the same value the reader returns.
    assert found == load_json_file(path)
