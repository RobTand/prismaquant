"""Decision record completeness for prismaquant#2577.

The record carries the CEO decision (fleetgraph#16) for the parent thread
of prismaquant#1921. It must name the capture rule, the producer plan,
the admission plan, the artifact plan, the #1271 budget rule with its
ban on quiet relabel, and the no-code-change statement. No production
code is touched by this issue.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTE = ROOT / "docs" / "results" / "2026-10-09_visual-capture-rule-2577.md"


def _text() -> str:
    assert NOTE.exists(), f"decision record is absent: {NOTE}"
    return NOTE.read_text(encoding="utf-8")


def test_note_names_strict_full_scope_rule():
    text = _text()
    assert "strict full-scope capture" in text
    assert "all vision Linears and all merger Linears" in text
    assert "fails closed" in text


def test_note_names_producer_admission_and_artifact_plan():
    text = _text()
    assert "external Tessera producer" in text
    assert "exact pinned Tessera runtime" in text
    assert "requires_plugin tessera" in text
    assert "byte witnesses" in text


def test_note_cites_1271_and_bans_quiet_relabel():
    text = _text()
    assert "prismaquant/issues/1271" in text
    assert "Ban quiet relabel" in text


def test_note_states_no_code_change():
    text = _text()
    assert "None. This note changes no code" in text
