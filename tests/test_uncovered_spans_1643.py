"""Uncovered-spans twins share the source_read_plan owner (issue #1643).

``joint_layer_quanta.uncovered_source_spans`` was the same algorithm as
``source_read_plan.uncovered_spans`` over plain tuples, minus the ``int()``
coercion (the identity on every value the joint module passes). The wrapper
keeps its name and signature for the four internal call sites and the test
importers. This test pins the wrapper against a verbatim copy of the
pre-change body: identical uncovered sets on fixed inputs, including partial
overlap, straddling spans, and path normalization.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from prismaquant.joint_layer_quanta import (  # noqa: E402
    uncovered_source_spans as wrapper,
)


def _reference(entries, spans):
    by_path = {}
    for entry in entries:
        by_path.setdefault(os.path.normpath(entry["path"]), []).append(
            (entry["offset"], entry["offset"] + entry["bytes"]))
    return [(path, start, end) for path, start, end in spans
            if not any(low <= start and end <= high
                       for low, high in by_path.get(os.path.normpath(path), ()))]


ENTRIES = [
    {"path": "/m/a", "offset": 0, "bytes": 100},
    {"path": "/m/a", "offset": 200, "bytes": 50},
    {"path": "./m/b", "offset": 10, "bytes": 20},
]

SPANS = [
    ("/m/a", 0, 100),      # covered outright
    ("/m/a", 50, 150),     # straddles the gap -> uncovered
    ("/m/a", 200, 250),    # covered outright
    ("/m/a", 0, 250),      # no single entry covers -> uncovered
    ("m/b", 10, 30),       # covered after normpath
    ("/m/c", 0, 10),       # unknown path -> uncovered
]


def test_wrapper_matches_reference():
    assert wrapper(ENTRIES, SPANS) == _reference(ENTRIES, SPANS) == [
        ("/m/a", 50, 150),
        ("/m/a", 0, 250),
        ("/m/c", 0, 10),
    ]


def test_wrapper_matches_reference_empty():
    assert wrapper([], []) == _reference([], []) == []
    assert wrapper([], SPANS) == _reference([], SPANS) == SPANS
