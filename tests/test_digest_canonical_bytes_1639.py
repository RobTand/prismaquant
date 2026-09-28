"""Canonical-JSON bytes/sha twins delegate to the digests owner (issue #1639).

``tools/dsv4_wikitext_inputs.py`` and ``tools/full_kl_teacher_payload.py``
spelled the same canonical dumps keywords as ``prismaquant/digests.py``.
Both wrappers now delegate (bytes and sha) with their own error types and
messages preserved. The fixture holds the pre-change bytes: every golden
input must still encode and hash identically, and refusals must keep their
types. (The tuple input serializes like its list form; both spellings agree,
pinned by case1.)
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "tools"))
sys.path.insert(0, str(REPOSITORY))


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WIKITEXT = _load("golden_wikitext_1639", "tools/dsv4_wikitext_inputs.py")
TEACHER = _load("golden_teacher_1639", "tools/full_kl_teacher_payload.py")

GOLDENS = json.loads(
    (REPOSITORY / "tests" / "fixtures"
     / "digest_canonical_bytes_1639.json").read_text())

CASE_INPUTS = [
    {"files": {"b": [1, 2, {"a": None}]}, "n": 1.5, "u": "héllo✓"},
    {"t": (1, 2), "z": [{"k": "v"}], "e": {}, "l": []},
    {},
    [],
    "s",
    5,
    3.14,
    True,
    None,
]


@pytest.mark.parametrize("index", range(len(CASE_INPUTS)))
def test_bytes_match_golden(index):
    want = GOLDENS[f"case{index}"]
    value = CASE_INPUTS[index]
    assert WIKITEXT.canonical_json_bytes(value).hex() == want["hex"]
    assert TEACHER.canonical_json_bytes(value).hex() == want["hex"]


@pytest.mark.parametrize("index", range(len(CASE_INPUTS)))
def test_sha_match_golden(index):
    want = GOLDENS[f"case{index}"]
    value = CASE_INPUTS[index]
    assert WIKITEXT.canonical_sha256(value) == want["sha"]
    assert TEACHER.canonical_sha256(value) == want["sha"]


def test_refusals_keep_types():
    for bad in (float("nan"), float("inf"), {1, 2}, object()):
        with pytest.raises(WIKITEXT.DSv4WikiTextInputsError):
            WIKITEXT.canonical_json_bytes(bad)
        with pytest.raises(WIKITEXT.DSv4WikiTextInputsError):
            WIKITEXT.canonical_sha256(bad)
        with pytest.raises(TEACHER.TeacherPayloadError):
            TEACHER.canonical_json_bytes(bad)
        with pytest.raises(TEACHER.TeacherPayloadError):
            TEACHER.canonical_sha256(bad)
