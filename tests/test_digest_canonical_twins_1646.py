"""Remaining canonical twins delegate to matching DIRECT profiles (issue #1646).

``tools/measure_vllm_wikitext_ppl._canonical_sha256`` (compact/utf8/strict)
delegates to ``DIRECT_UTF8_STRICT.sha256``;
``prismaquant/shipcard._canonical_json`` (compact/ascii/lax + ``default=str``,
callers ``.encode`` it) delegates to ``DIRECT_ASCII_LAX_DEFAULT_STR.text``.
Both were single dumps with exactly the profile's options, so the delegation
is byte-identical by construction, including int keys (numeric sort, #1640
lesson). The fixture holds base-computed sha/text on fixed inputs (unicode,
int keys, NaN literal for the lax profile).

Kept, not widened: ``reseal_campaign_identity`` (stdlib-only by documented
decision), ``measure_served_gold`` (torch-free), ``serve_fingerprint``
(serving container, no package).
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


PPL = _load("twin_ppl_1646", "tools/measure_vllm_wikitext_ppl.py")

from prismaquant.shipcard import _canonical_json  # noqa: E402
import hashlib  # noqa: E402

GOLDENS = json.loads(
    (REPOSITORY / "tests" / "fixtures"
     / "digest_canonical_twins_1646.json").read_text())

PPL_INPUTS = [
    {"model": "m", "ppl": 12.5, "u": "héllo✓"},
    {10: "a", 9: "b"},
    {"x": {10: 1, 9: 2, 100: 3}},
]

SHIP_INPUTS = [
    {"quant": {"bits": 4}, "tags": ["a", "b"]},
    {10: "a", 9: "b"},
    {"u": "héllo✓", "nan": float("nan")},
]


@pytest.mark.parametrize("index", range(len(PPL_INPUTS)))
def test_ppl_sha_matches_golden(index):
    assert PPL._canonical_sha256(PPL_INPUTS[index]) == GOLDENS[f"ppl{index}"]["sha"]


@pytest.mark.parametrize("index", range(len(SHIP_INPUTS)))
def test_ship_text_and_sha_match_golden(index):
    text = _canonical_json(SHIP_INPUTS[index])
    assert text == GOLDENS[f"ship{index}"]["text"]
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == GOLDENS[f"ship{index}"]["sha"]
