"""CPU-only structural acceptance for #2810's pending qualification index.

These checks do not attest GPU execution, numerical acceptance, authority, or
truth of prose. Review the proof rules and every row's content separately.
The immutable research harness is not part of the requested two-document
implementation/publication slice. Git provenance belongs to TrustLadder's
publication audit, not a test that fabricates refs or assumes sandbox history.
"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs/design/pending_gpu_1921.md"
ROW_TITLES = (
    "current selected MTP whole-artifact/cache/wire/byte qualification",
    "MTP served-load/speculative correctness",
    "matched MTP BF16/EXL3 acceptance and speed/quality A/B",
    "complete real multimodal visual/merger Fisher/Hessian/activation capture",
    "exact-pin admitted per-shape Tessera rung quality/cost/byte/runtime panel",
    "selected original-wire cache/export/full-artifact witnesses",
    "plugin recognition/TP/mixed-rung output and BF16/EXL3 serving A/B",
)
FIELDS = ("Inputs", "Identities", "Pass bar")


def _record():
    assert RECORD.is_file(), (
        "#2810 requires docs/design/pending_gpu_1921.md; the campaign tmp "
        "record is not a durable repository qualification index"
    )
    return RECORD.read_text(encoding="utf-8")


def test_seven_handoff_rows_each_have_three_substantive_fields():
    text = _record()
    # Numbered row headings use level two; their required fields use level
    # three. Other unnumbered shared-rule headings may surround the rows.
    rows = list(re.finditer(r"^## ([0-9]+)[.)] (.+?)\s*$", text, re.MULTILINE))
    assert [(int(row[1]), row[2]) for row in rows] == list(
        enumerate(ROW_TITLES, start=1)
    ), "Preserve exactly seven numbered handoff row titles, in order"
    for field in FIELDS:
        assert len(re.findall(
            rf"^### {re.escape(field)}\s*$", text, re.MULTILINE
        )) == 7, f"Expected exactly seven {field!r} field headings"
    for row in rows:
        # Stop at the next level-two heading, including unnumbered headings.
        body = re.split(r"^## ", text[row.end():], maxsplit=1, flags=re.MULTILINE)[0]
        fields = list(re.finditer(r"^### (.+?)\s*$", body, re.MULTILINE))
        assert [field[1] for field in fields] == list(FIELDS), (
            f"Row {row[1]} needs Inputs, Identities and Pass bar, in order"
        )
        for index, field in enumerate(fields):
            end = fields[index + 1].start() if index + 1 < len(fields) else len(body)
            assert body[field.end():end].strip(), (
                f"Row {row[1]} has an empty {field[1]} section"
            )


def test_docs_index_exposes_the_pending_qualification_record():
    index = (ROOT / "docs/README.md").read_text(encoding="utf-8")
    entries = [line for line in index.splitlines() if "design/pending_gpu_1921.md" in line]
    assert len(entries) == 1, "Index the durable #2810 record once in docs/README.md"
    entry = entries[0]
    assert re.search(r"`design/pending_gpu_1921\.md`|\]\((?:\./)?design/pending_gpu_1921\.md\)", entry), (
        "Use the existing backticked repository-relative path or a relative Markdown link"
    )
    assert re.search(r"\bpending\b", entry, re.IGNORECASE), (
        "The index must describe pending evidence, not completed qualification"
    )
    assert re.search(r"qualif|evidence", entry, re.IGNORECASE), (
        "Explain that this is a qualification/evidence index"
    )
