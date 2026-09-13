"""Lane-eligibility schema v10: the PLATFORM entry, and the refusal before it.

Tessera's contract v23 (its #456, merged as #464) moves the lane table to
``tessera.lane-eligibility.v10``.  The change is one shape, inside
``lane_eligibility.platforms``::

    "sm_121": {...}                       # v9: the VALUE was never read
    "sm_121": {"backend": "cuda", "compute_capability": [12, 1],
               "serve_image": "vllm/vllm-openai@sha256:...",
               "executes": {"TESSERA_E2M1_K2": "e2m1_group16_ue4m3_static",
                            "TESSERA_E4M3_K1": "fp8_per_token_dynamic",
                            "TESSERA_BF16_K1": "bf16_unquantized"}}

and it exists because a cell is a RECEIPT.  Under v9 the only way to say
anything about a platform was to have served on it, so there was no way to
publish the fact that a family has no native route on a device -- which is
exactly what an RDNA3.5 or RDNA4 box needs said about E2M1 and E4M3 before a
model load reaches a kernel.  Silence had to stand in for it, and silence is
not an attestation (principle 9).  ``null`` now says it, and it is a claim
somebody looked.

**What this module pins is the admission of the grammar, and the refusal that
stood before it.**  A schema bump here is deliberately NOT additive: a v9
reader must not read a v10 document, because a platform entry is now an
object with meaning rather than a key, and a reader that went on treating
``platforms`` as a key set would read an unbacked platform as an ordinary one.
Both of this repository's readers are closed over schema NAMES for that
reason, so before this change they refused the real v23 bytes outright -- by
name, not by a missing field.  That refusal is the designed fail-closed and it
is asserted here on the real packaged file, because a test that only shows the
new grammar working never shows what the version number is for.

No gate here reads a platform's ``executes`` yet; PrismaQuant #528 is the
first, and it re-reviews the pin answer's projection when it does.  What v10
buys at THIS pin is that the document parses at all.
"""
from __future__ import annotations

import hashlib
import json

import pytest
from importlib.resources import as_file

from prismaquant import lane_eligibility as lane
from prismaquant import tessera_render as tr
from prismaquant import tessera_runtime_contract as contract
from prismaquant.tessera_serving_runtime_pin import (
    TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256,
)


def _packaged_bytes() -> bytes:
    with as_file(tr.tessera_serving_contract_path()) as path:
        return path.read_bytes()


def _packaged_contract() -> dict:
    return json.loads(_packaged_bytes())


# ---------------------------------------------------------------------------
# v10 is the current grammar, and it inherits every property v9 had
# ---------------------------------------------------------------------------
def test_v10_is_the_current_grammar():
    assert (lane.LANE_ELIGIBILITY_SCHEMA_TESSERA
            == lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V10
            == "tessera.lane-eligibility.v10")
    # The two readers cannot disagree about which schema is newest.
    assert contract.TESSERA_LANE_SCHEMA == lane.LANE_ELIGIBILITY_SCHEMA_TESSERA


def test_v10_joins_every_set_v9_is_in():
    """v10 cells are v9 cells byte for byte, so every property set must carry it.

    This is the failure mode the sets exist to prevent, read forwards: a bump
    that lands in ``LANE_ELIGIBILITY_SCHEMAS`` alone parses the document and
    then reads its attested cells as ones that publish no evidence, no derived
    attribution and no smoke record -- a silent demotion, with nothing red.
    Asserted as "every set", derived from membership of v9, rather than as a
    list of five names a sixth set could be added beside.
    """
    sets = {
        name: value for name, value in vars(lane).items()
        if name.endswith("_LANE_SCHEMAS") and isinstance(value, frozenset)
    }
    assert sets, "no lane-schema property sets found"
    carrying_v9 = {
        name for name, value in sets.items()
        if lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V9 in value
    }
    # v9 is in more than just the accepted-schemas set, or this test is vacuous.
    assert len(carrying_v9) >= 5, sorted(carrying_v9)
    missing = sorted(
        name for name in carrying_v9
        if lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V10 not in sets[name]
    )
    assert not missing, (
        f"lane schema v10 is missing from {missing}; v10 republishes the v9 "
        "cells byte for byte, so a set it is absent from reads an attested "
        "cell as one that publishes no evidence")


# ---------------------------------------------------------------------------
# The packaged contract, at the pinned digest
# ---------------------------------------------------------------------------
def test_the_packaged_contract_is_v23_at_the_pinned_digest():
    raw = _packaged_bytes()
    assert (hashlib.sha256(raw).hexdigest()
            == TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256), (
        "the installed Tessera is not the pinned one; install the pinned "
        "commit rather than relaxing this check")
    payload = json.loads(raw)
    assert payload["contract_version"] == 23
    assert (payload["lane_eligibility"]["schema"]
            == lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V10)


def _packaged_table():
    with as_file(tr.tessera_serving_contract_path()) as path:
        return lane.load_eligibility_table(contract_path=path)


def test_the_v10_table_parses_and_publishes_the_two_amd_platforms():
    table = _packaged_table()
    assert table.present
    assert table.schema == lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V10
    assert {"sm_121", "gfx1151", "gfx1201"} <= set(table.platforms)
    # No AMD cell ships at v23: a cell is a device receipt and none was taken.
    assert {cell.platform for cell in table.cells} == {"sm_121"}
    assert len(table.cells) == 10


def test_a_declared_platform_with_no_cell_is_still_a_refusal_to_claim():
    """Backing is not a receipt, and v10 does not blur the two.

    ``gfx1151``'s entry says the build executes ``bf16_unquantized`` for
    ``TESSERA_BF16_K1``.  It ships no cell, so nothing on this side is
    attested for it: the cell-based resolution that decides export keeps
    answering ``unattested`` for every family on that platform, and export
    fails closed.  Asserted at the table, which is the object that seam reads.
    """
    table = _packaged_table()
    amd = [c for c in table.cells if c.platform in ("gfx1151", "gfx1201")]
    assert not amd, [c.id for c in amd]


# ---------------------------------------------------------------------------
# The refusal that stood before this change -- the designed fail-closed
# ---------------------------------------------------------------------------
def test_a_v9_closed_eligibility_reader_refuses_the_real_v23_bytes(monkeypatch):
    """By NAME, not by a missing field.  The whole point of a versioned schema.

    The sets are restored to their pre-#527 value and the REAL packaged v23
    block is handed to the parser.  It must refuse, and the message must name
    the schema -- "missing field(s) ['executes']" would send its reader off to
    edit a table rather than to install a release the reader was written for.
    """
    pre_527 = frozenset(
        lane.LANE_ELIGIBILITY_SCHEMAS - {lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V10})
    monkeypatch.setattr(lane, "LANE_ELIGIBILITY_SCHEMAS", pre_527)
    payload = _packaged_contract()
    with pytest.raises(lane.LaneEligibilityError) as excinfo:
        lane._parse_table(
            payload["lane_eligibility"], payload["formats"], "", "commit", "sha",
            native_extensions=payload["native_extensions"])
    message = str(excinfo.value)
    assert "tessera.lane-eligibility.v10" in message
    assert "schema must be one of" in message


def test_a_v9_closed_contract_reader_refuses_the_real_v23_bytes(monkeypatch, tmp_path):
    """The second reader, refusing the same bytes for the same reason."""
    pre_527 = frozenset(
        contract.TESSERA_LANE_SCHEMAS - {lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V10})
    monkeypatch.setattr(contract, "TESSERA_LANE_SCHEMAS", pre_527)
    raw = _packaged_bytes()
    path = tmp_path / "runtime_contract.json"
    path.write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    with pytest.raises(contract.TesseraContractError) as excinfo:
        contract._load_at(str(path), sha, contract.TESSERA_DEV_PIN_COMMIT)
    message = str(excinfo.value)
    assert "tessera.lane-eligibility.v10" in message
    assert "lane_eligibility.schema must be one of" in message


# ---------------------------------------------------------------------------
# The answer moved by one entry, and the platform axis is not in it yet
# ---------------------------------------------------------------------------
def test_the_pin_answer_names_v10_and_does_not_project_the_platform_axis():
    """Why the reviewed diff is one field.

    ``TESSERA_DEV_PIN_ANSWER`` is the projection an ADMISSION gate reads, and
    it re-stales when that projection WIDENS as much as when a value moves.
    At this pin nothing here decides on a platform's ``executes``, so the
    answer must not carry it: an answer that transcribed a field no gate reads
    would be a review of something nobody acts on, and the next real reader
    would then land with no diff to review.
    """
    answer = contract.TESSERA_DEV_PIN_ANSWER
    assert answer["lane_schema"] == lane.LANE_ELIGIBILITY_SCHEMA_TESSERA_V10
    assert "platforms" not in answer
    assert "contract_version" not in answer
