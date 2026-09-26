"""The routed FP8 evidence Tessera contract v38 withdrew, for the tests built on it.

Contract v38 (Tessera #604) re-minted the two routed ``TESSERA_E4M3_K1`` cells
under the same ids, for q896 on a different image and a compact launch, and
published them ``route_only``: no KL entry, ``smoke.status: not_recorded``,
``record: null``. The v37 cells carried the one real smoke RECORD in the table
(status ``recorded``, receipt ``docs/measurements/moe-smoke-recorded-2026-09-05.md``)
and the LFM batch KL lower bound, and the evidence-grammar and
evidence-admission tests in this suite need real evidence to run on: replaying
the grammar against invented evidence would weaken exactly the differential
checks those tests exist for.

So the v37 cells are quoted in ``fixtures/tessera_cells_withdrawn_v38.json``,
byte-for-byte from Tessera's own ``tests/fixtures/
lane_eligibility_cells_withdrawn_v38.json`` at af7a86d43 (its ``source`` field
names the v37 document it quotes). Nothing here resolves that file as a
contract. :func:`with_quoted_evidence` splices ONLY the quoted ``evidence``
block onto the shipped cells of the same id -- the shipped rung, image and
launch stay what v38 publishes -- so a test that re-validates through
Tessera's own ``validate_serving_contract`` still checks a document Tessera's
current grammar accepts.

Each module that uses this also asserts what the shipped table now says
(see :func:`shipped_routed_smoke_statuses`), so the downgrade is stated, not
hidden.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "tessera_cells_withdrawn_v38.json"

MOE_DECODE = "tessera_e4m3_k1_routed_moe_sm121_decode_resident"
MOE_BATCH = "tessera_e4m3_k1_routed_moe_sm121_batch_resident"
ROUTED_IDS = (MOE_DECODE, MOE_BATCH)
#: The rung the shipped routed cells attest (v38). The quoted cells attested
#: 1024, which is now unattested for ``routed_moe`` under every image.
ROUTED_RUNG = 896


def withdrawn_v38_cells() -> list[dict]:
    """The two v37 routed cells, as v37 published them."""
    cells = json.loads(FIXTURE.read_text(encoding="utf-8"))["cells"]
    assert sorted(c["id"] for c in cells) == sorted(ROUTED_IDS), cells
    return copy.deepcopy(cells)


def shipped_routed_smoke_statuses(payload: dict) -> dict[str, str]:
    """What the given (unspliced) table publishes for the routed smoke."""
    return {cell["id"]: cell["evidence"]["smoke"]["status"]
            for cell in payload["lane_eligibility"]["cells"]
            if cell["id"] in ROUTED_IDS}


def with_quoted_evidence(payload: dict) -> dict:
    """A copy of ``payload`` whose routed cells carry the quoted v37 evidence."""
    quoted = {cell["id"]: cell["evidence"] for cell in withdrawn_v38_cells()}
    moved = copy.deepcopy(payload)
    spliced = 0
    for cell in moved["lane_eligibility"]["cells"]:
        if cell["id"] in quoted:
            cell["evidence"] = copy.deepcopy(quoted[cell["id"]])
            spliced += 1
    assert spliced == len(ROUTED_IDS), spliced
    return moved
