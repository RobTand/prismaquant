"""CPU documentary checks for #2768, not numerical or GPU qualification.

Like test_architecture_doc and test_mtp_option_b_protocol_2771, these check
that a reviewable record exposes the required decisions. A reviewer must
validate the source/receipt claims and semantic dispositions; keywords and
shrinking inventory counts cannot establish consolidation or pilot acceptance.
No external fleet paths, historical Git objects, or network access are needed.
"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs/audits/umbrella_acceptance_1295.md"


def _text():
    assert RECORD.is_file(), (
        "#2768 lacks the source-bound umbrella acceptance record: "
        "docs/audits/umbrella_acceptance_1295.md"
    )
    return RECORD.read_text(encoding="utf-8")


def _section(heading):
    match = re.search(
        rf"^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)",
        _text(), re.MULTILINE | re.DOTALL,
    )
    assert match and match.group(1).strip(), f"Missing acceptance section: {heading}"
    return match.group(1)


def _mentions(text, *concepts):
    normalized = re.sub(r"\s+", " ", text).lower()
    for concept in concepts:
        assert concept.lower() in normalized, f"Disposition {concept!r} explicitly"


def _local_paths(text, suffix):
    # Backticked repo-relative evidence references may include symbols/lines.
    paths = re.findall(rf"`((?:prismaquant|tools|tests)/[^`\s:]+\.{suffix})(?::[^`]*)?`", text)
    assert paths, f"Expected local .{suffix} evidence reference in {text!r}"
    for path in paths:
        assert (ROOT / path).is_file(), f"Evidence path does not exist: {path}"


def test_acceptance_matrix_accounts_for_all_submitted_children():
    matrix = _section("Acceptance matrix")
    assert re.search(r"\b[0-9a-f]{40}\b", _text()), "Bind inspection to a source revision"
    lines = [line for line in matrix.splitlines() if line.startswith("|")]
    assert lines, "One acceptance matrix is required"
    headers = [cell.strip().lower() for cell in lines[0].strip("|").split("|")]
    assert headers == [
        "child", "outcome", "source", "verification", "status", "limitations",
    ]
    rows = {}
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        assert len(cells) == len(headers), f"Malformed matrix row: {line}"
        assert all(cells), f"Empty acceptance field: {line}"
        match = re.fullmatch(r"#(\d+)", cells[0])
        assert match, f"Child column must name one issue: {cells[0]}"
        assert match[1] not in rows, f"Duplicate child row: {cells[0]}"
        rows[match[1]] = cells
    required = {"1292", "1293", "1294", "1299", "1300", "1301", "1302", "1303", "1304"}
    assert required <= rows.keys(), f"Missing child outcomes: {required - rows.keys()}"
    for child in required:
        cells = rows[child]
        _local_paths(cells[2], "py")
        _local_paths(cells[3], "py")
        assert re.search(
            r"receipt|test evidence|historical|unavailable|not run", cells[3], re.I,
        ), f"#{child}: distinguish observed results from test-source coverage"


def test_digest_reconciliation_preserves_refresh_and_existing_owners():
    digest = _section("Digest consolidation (#1301)")
    _mentions(
        digest,
        "docs/audits/digest_site_census_pq1301_2026-10-04.json",
        "refresh_2540", "supersed", "tests/fixtures/duplication_baseline.json",
        "current", "scope", "call", "profile", "identity", "receipt",
        "#2649", "#2660", "#2662", "#2663", "#2648", "#2646",
        "#2638", "#2637", "#2636", "#2635", "#2630", "#2629",
        "#2628", "#2627", "#2626", "#2650", "#2651", "#2751",
        "#2841", "#2842", "#2843", "#2544",
    )
    # These are historical refresh measurements, not required current counts.
    for historical in (304, 420):
        assert str(historical) in digest, f"Distinguish historical {historical} from current totals"
    _mentions(digest, "seven", "live", "retir", "unmet")
    _local_paths(digest, "py")


def test_numerics_reconciliation_dispositions_each_residual_family():
    numerics = _section("Domain numerics (#1303)")
    _mentions(
        numerics,
        "docs/design/domain_numerics_census_1303_20261002.md",
        "docs/ARCHITECTURE.md", "non-exhaustive", "header", "acquisition",
        "refusal", "index", "shard", "rtn", "tie", "mx", "448",
        "layer", "grammar", "fp8", "activation", "device", "vendored",
    )
    # A per-family table exposes semantic dispositions, not an occurrence count.
    rows = [line for line in numerics.splitlines() if line.startswith("|")]
    for family in ("header", "index", "rtn", "mx", "layer"):
        candidates = [line for line in rows if family in line.lower()]
        assert candidates, f"Missing residual family disposition: {family}"
        assert any(re.search(
            r"delegated|intentionally different|still owed", line, re.I,
        ) for line in candidates), f"No semantic disposition for {family}"
        _local_paths("\n".join(candidates), "py")
    _mentions(numerics, "numerical", "#1303", "receipt", "unmet")


def test_w1_verdict_and_parent_gate_are_evidence_bounded():
    pilot = _section("W1 pilot verdict (#1293)")
    _mentions(
        pilot, "#2767", "#2836", "#2837", "gpu", "source", "lifetime",
        "full-shape", "before", "after", "receipt", "cpu",
    )
    assert re.search(r"blocked|pending|not accepted", pilot, re.I), (
        "Record W1's actual blocked/not-accepted disposition; CPU prerequisites "
        "do not replace its outstanding genuine pilot and matched executions"
    )
    _mentions(pilot, "docs/results/2026-10-02_pilot_sdk4_consumer.md")
    parent = _section("Parent verdict and authority")
    _mentions(parent, "#1295", "#1293", "#1301", "#1303", "pending", "trustladder")
    _mentions(parent, "unmet", "publication", "issue transitions")
    assert re.search(r"not.*(?:gpu|qualification)|(?:no|neither).*gpu", parent, re.I | re.S), (
        "State the limits of the CPU guards, rather than treating them as GPU evidence"
    )
