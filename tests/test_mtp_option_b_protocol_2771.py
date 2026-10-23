"""CPU-only documentary regression checks for the bounded #2771 record.

These check that the single handoff record exposes the required decisions and
preserves historical accounting, not that an approval is authentic or that a
candidate has qualified. A reviewer must verify referenced immutable inputs,
authority, numerical rules and blocked dispositions. No fleet data is opened.
The record may adopt an existing protocol by immutable reference rather than
create a competing campaign; its local sections must explain that disposition.
"""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "docs/design/glm_mtp_option_b_protocol_2771.md"
HISTORY = ROOT / "docs/measurements/glm-mtp-bf16-fixed-2026-09-26.md"


def _section(heading):
    assert RECORD.is_file(), (
        "#2771 lacks its bounded protocol/adoption and approval-status record: "
        "docs/design/glm_mtp_option_b_protocol_2771.md"
    )
    text = RECORD.read_text(encoding="utf-8")
    match = re.search(
        rf"^## {re.escape(heading)}\s*\n(.*?)(?=^## |\Z)",
        text, re.MULTILINE | re.DOTALL,
    )
    assert match and match.group(1).strip(), f"Missing protocol section: {heading}"
    return re.sub(r"\s+", " ", match.group(1)).lower()


def _mentions(text, *concepts):
    for concept in concepts:
        assert concept.lower() in text, f"Protocol must disposition {concept!r}"


def test_immutable_panel_and_full_mtp_coverage_are_exposed():
    panel = _section("Immutable inputs and panel")
    _mentions(
        panel, "source", "calibration", "teacher", "tokenizer", "workload",
        "tessera", "runtime", "tensor parallel", "candidate", "quantized",
        "bf16", "shared", "routed", "unsupported",
        "brandonmusic/GLM-5.3-Flash-tr3-4bpw",
    )
    assert re.search(r"\b864\b", panel), "All 864 routed units must be accounted for"
    assert re.search(r"\b(3|three)\b", panel), "All three shared units need evidence"
    assert re.search(r"\b(2|two)\b", panel), (
        "Disposition the two-quantized-choice panel, or cite its approved exception"
    )


def test_byte_decision_retains_the_actual_historical_discrepancy():
    accounting = _section("Byte accounting")
    _mentions(
        accounting, "global cap", "all-file", "shard", "fixed", "reserve",
        "tolerance", "bf16", "body", "authority", "waiv",
    )
    # Historical facts are not a newly authorized cap. In particular, a waiver
    # and the larger all-file rival inventory must not erase the strict miss.
    historical = HISTORY.read_text(encoding="utf-8")
    for value in (175_642_157_752, 175_790_111_275, 175_643_087_583, 929_831):
        assert f"{value:,}" in historical
        assert str(value) in re.sub(r"[,_ ]", "", accounting), (
            f"Byte decision must preserve historical quantity {value:,} B"
        )
    assert "a5fee929cf4888b1824323e33e8a19b60129e025" in RECORD.read_text(
        encoding="utf-8"
    ), "Retain the exact rival revision when reconciling its inventories"


def test_evaluation_rules_are_recorded_before_new_results():
    evaluation = _section("Prospective evaluation")
    _mentions(
        evaluation, "speculative-on", "speculative-off", "own", "greedy",
        "zero token mismatches", "body", "without speculation", "kl",
        "sample", "stochastic", "numerical", "quality", "warmup",
        "repetition", "counter", "reset", "attribution", "uncertainty",
        "pareto", "acceptance", "decode", "nondominated", "approv",
    )


def test_existing_children_and_approval_boundary_are_explicit():
    handoff = _section("Child handoff and historical reuse")
    _mentions(
        handoff, "#1271", "#2530", "#2532", "#2534", "#2536", "#2539",
        "receipt", "reus", "historical", "prospective", "#2772",
        "dflash", "#2674", "#2675", "#2676",
    )
    approval = _section("Ownership and approval")
    _mentions(approval, "authority", "content", "approval", "unresolved", "qualification")
    assert re.search(r"\b(block|prevent|withhold)\w*\b", approval), (
        "An unresolved decision must block affected qualification, not trigger "
        "a credential/physical-action request"
    )
