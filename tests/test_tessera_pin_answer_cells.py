"""The reviewed answer's cell roster IS the pinned document's, row for row.

``TESSERA_DEV_PIN_ANSWER`` is a literal, and a literal is a transcription that
can be edited without the thing it transcribes moving.  Every other test in
this repository reads one side or the other: the drift check inside
``load_tessera_contract`` compares them only when the development pin is
REQUESTED, and the tests that assert the packaged file never look at the
literal.  So a row could be dropped from the answer, or invented in it, and the
suite would stay green while the pin admitted a roster nobody published.

This file closes that.  It reads the packaged contract at the serving pin's
digest and asserts the answer's ``(id, platform)`` set equals the document's,
which is the pair the v24 bump made load-bearing: the document now has two
platforms, and a row that could not show its platform would publish two rows a
reader cannot tell apart.

It also asserts the shape the answer is a projection OF, so that a contract
that quietly stopped publishing a field is caught here rather than by a
``KeyError`` inside the parser:

* every cell id follows Tessera's derived grammar,
  ``family_structure_platform_regime[_residency]`` (its #111), spelled off the
  row's own fields rather than typed;
* every cell carries a ``platform`` that the table declares;
* every cell carries a ``runtime`` scope with its own ``vllm`` and ``torch``
  versions -- the answer's columns 14 and 15, which scope the claim the same
  way the image does;
* every cell's ``evidence`` is CLOSED: exactly ``artifact``, ``grade``, ``kl``
  and ``smoke``, no more and no fewer.  ``cell_evidence_admits`` decides on
  that block, so a fifth key is a fact a gate would not read and a missing one
  is a fact it would read as absent.
"""
from __future__ import annotations

import hashlib
import json

from importlib.resources import as_file

from prismaquant import lane_eligibility as lane
from prismaquant import tessera_render as tr
from prismaquant import tessera_runtime_contract as contract
from prismaquant.tessera_serving_runtime_pin import (
    TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256,
)

#: The smallest roster this pin is allowed to carry. A floor rather than an
#: equality so that a contract which ADDS a cell fails the identity assertion
#: below -- which names what moved -- instead of failing a count that does not.
MINIMUM_CELLS = 14

EVIDENCE_KEYS = {"artifact", "grade", "kl", "smoke"}


def _packaged() -> dict:
    with as_file(tr.tessera_serving_contract_path()) as path:
        raw = path.read_bytes()
    assert (hashlib.sha256(raw).hexdigest()
            == TESSERA_SERVING_RUNTIME_PINNED_CONTRACT_SHA256), (
        "the installed Tessera is not the pinned one; install the pinned "
        "commit rather than relaxing this check")
    return json.loads(raw)


def _document_cells() -> list[dict]:
    return list(_packaged()["lane_eligibility"]["cells"])


def test_the_answers_cells_are_the_documents_cells():
    """The identity this file exists for, and the only assertion that names it."""
    document = {(str(row["id"]), str(row["platform"]))
                for row in _document_cells()}
    answer = {(str(row[0]), str(row[1]))
              for row in contract.TESSERA_DEV_PIN_ANSWER["cells"]}
    assert answer == document, {
        "in the answer, not the document": sorted(answer - document),
        "in the document, not the answer": sorted(document - answer),
    }
    assert len(answer) >= MINIMUM_CELLS, sorted(answer)


def test_the_answer_carries_more_than_one_platform():
    """Why the platform column is load-bearing rather than decorative.

    Before contract v24 every cell was ``sm_121`` and the column could have
    been dropped without any row becoming ambiguous. It cannot now: the
    document publishes cells on two platforms, so a roster keyed on the id
    alone would be a roster whose reader cannot say where a row was measured.
    """
    platforms = {str(row[1]) for row in contract.TESSERA_DEV_PIN_ANSWER["cells"]}
    assert len(platforms) >= 2, sorted(platforms)
    declared = set(_packaged()["lane_eligibility"]["platforms"])
    assert platforms <= declared, sorted(platforms - declared)


def test_every_cell_id_follows_the_derived_grammar():
    """Tessera #111: ids are DERIVED, so they are re-derivable here.

    Spelled off each row's own ``family``/``structure``/``platform``/``regime``
    rather than matched against a pattern, because a pattern would also accept
    a row whose id names a different cell than its fields do -- which is the
    failure an id grammar exists to prevent.
    """
    for row in _document_cells():
        stem = "_".join((
            str(row["family"]).lower(),
            str(row["structure"]),
            str(row["platform"]).replace("_", ""),
            str(row["regime"]),
        ))
        allowed = {stem, f"{stem}_resident", f"{stem}_streamed"}
        assert str(row["id"]) in allowed, (row["id"], sorted(allowed))


def test_every_cell_publishes_its_runtime_scope():
    for row in _document_cells():
        runtime = row["runtime"]
        assert isinstance(runtime.get("image"), str) and runtime["image"]
        assert isinstance(runtime.get("vllm"), str) and runtime["vllm"]
        assert isinstance(runtime.get("torch"), str) and runtime["torch"]
        assert runtime.get("execution_modes"), row["id"]


def test_every_cells_evidence_block_is_closed():
    for row in _document_cells():
        evidence = row["evidence"]
        assert set(evidence) == EVIDENCE_KEYS, (row["id"], sorted(evidence))
        assert evidence["grade"] in lane.EVIDENCE_GRADES, evidence["grade"]
        assert set(evidence["smoke"]) == {
            "status", "receipt", "attribution", "control", "record"}, row["id"]


def test_the_projection_is_positional_and_its_width_is_uniform():
    """The columns documented beside the literal, asserted as a shape.

    A row that grew or lost a column is a WIDENED or narrowed projection, and
    the pin's own rule says that is a re-review even when no published value
    moved. The width is read off the answer rather than typed so that the
    re-review lands on the comment block, not here.
    """
    rows = contract.TESSERA_DEV_PIN_ANSWER["cells"]
    widths = {len(row) for row in rows}
    assert len(widths) == 1, sorted(widths)
    # 17 columns: 0-12 unconditional, 13-15 the runtime scope, 16 the evidence.
    assert widths == {17}, sorted(widths)
    for row in rows:
        assert isinstance(row[13], dict) and set(row[13]) == {
            "image", "execution_modes"}
        assert len(row[16]) == 7, row[0]
