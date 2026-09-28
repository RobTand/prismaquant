"""Freeze the bytes and outcomes of the pre-consolidation roster sites (#1446).

The golden table is recorded through PB on the old implementations before
moving them. Capturing SHA-256's input proves byte identity, not merely digest
equality. The replay-frontier and Stage B candidate rosters are different
recipes and are deliberately not part of this table.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from prismaquant import digests
from prismaquant import joint_head_walk_quanta as head
from prismaquant import joint_layer_quanta as layer
from tests.golden_table import GoldenTable, outcome

GOLDEN = GoldenTable("digest_rosters_1446")

INPUTS = {
    "empty": [],
    "one": ["model.layers.0.q_proj"],
    "reverse": ["z", "a", "middle"],
    "unicode": ["é", "漢字", "😀", "a"],
    "newlines": ["b\n", "a\nb"],
    "empty_name": [""],
    "duplicates": ["x", "x"],
    "tuple": ("z", "a"),
    "text": "za",
    "float": [1.5],
    "mixed": ["a", 1.5],
    "path": [PurePosixPath("weights/x")],
    "nested": [{"nested": [1, 2.5]}],
    "none": None,
    "surrogate": ["a\ud800"],
}


def _capture(monkeypatch, call):
    encoded = []
    original = hashlib.sha256

    def record(data=b"", *args, **kwargs):
        encoded.append(bytes(data).hex())
        return original(data, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(hashlib, "sha256", record)
        result = outcome(call)
    return {"outcome": result, "encoded_hex": encoded}


@pytest.mark.parametrize("site", ["head", "layer"])
@pytest.mark.parametrize("case", INPUTS)
def test_roster_site(monkeypatch, site, case):
    function = head.roster_digest if site == "head" else layer.roster_digest
    GOLDEN.check(_capture(monkeypatch, lambda: function(copy.deepcopy(INPUTS[case]))))


@pytest.mark.parametrize("site", ["head", "layer"])
def test_roster_iterator(monkeypatch, site):
    # Preserve the runtime behavior even for inputs outside the annotation.
    function: Any = head.roster_digest if site == "head" else layer.roster_digest
    GOLDEN.check(_capture(monkeypatch, lambda: function(iter(["z", "é", "a"]))))


@pytest.mark.parametrize("width", [1, 2, 4])
def test_head_walk_slice_bytes(monkeypatch, width):
    names = ["z", "é", "a", "model.layers.0.q_proj"]
    descriptors = head.head_walk_quanta(names, max_units_per_quantum=width)
    GOLDEN.check(_capture(monkeypatch, lambda: head.head_walk_quanta(
        names, max_units_per_quantum=width)))
    for descriptor in descriptors:
        GOLDEN.check(_capture(monkeypatch, lambda descriptor=descriptor: head.check_quantum_for_roster(
            descriptor, names)))
    corrupt = dict(descriptors[0], slice_sha256="0" * 64)
    GOLDEN.check(_capture(monkeypatch, lambda: head.check_quantum_for_roster(corrupt, names)))


def test_newline_profiles_keep_order_and_delimiters():
    assert head.roster_digest is digests.sorted_newline_utf8_sha256
    assert head.roster_digest(names=["z", "a"]) == layer.roster_digest(qnames=["z", "a"])
    assert digests.newline_utf8_bytes(["z", "é"]) == b"z\n\xc3\xa9"
    assert digests.newline_utf8_bytes(["é", "z"]) == b"\xc3\xa9\nz"
    assert digests.newline_utf8_bytes([]) == b""
    assert digests.newline_utf8_bytes([""]) == b""
    assert digests.newline_utf8_bytes(["a\n", "b"]) == b"a\n\nb"
    assert digests.sorted_newline_utf8_sha256(["z", "a"]) == (
        "64c921e065cd064a44d3489523757c632d4ce09132e80a1ef6d58cee9456e913")


_STORED_RECORD = Path(
    "/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/"
    "allocation/joint-panel/layer-quanta/layer-000.json")


def test_real_stored_takeover_roster():
    if not _STORED_RECORD.is_file():
        pytest.skip(f"fleet-only stored record absent: {_STORED_RECORD}")
    record = json.loads(_STORED_RECORD.read_text(encoding="utf-8"))
    binding = record["campaign"]
    prepared = json.loads(Path(binding["prepared_path"]).read_text(encoding="utf-8"))
    names = list(prepared["formats_by_qname"])
    assert len(names) == 36423
    expected = "4da42b8df5cbf1720406ef7f4c19ac5b7f11bdc5e0f51d0fef6378970a1ee6ea"
    assert binding["unit_roster_sha256"] == expected
    assert layer.roster_digest(names) == expected
    assert head.roster_digest(names) == expected
