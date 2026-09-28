"""Seal-phase row construction shares one helper (issue #1635).

The three ``_seal_phase`` closures in ``joint_layer_quanta.py`` and the
``seal`` closure in ``band_serial_manifest`` build phase rows, not digests:
``{"name", "entry_indices", "bytes", "cumulative_bytes"}`` with a running
total, refusing an empty phase. All three checked sites now delegate to
``joint_layer_quanta.append_read_phase`` with their own refusal; the
unchecked ``build_adjoint_manifest`` closure stays as it is (adding a refusal
there would be a new refusal).

These tests pin the helper against verbatim copies of the pre-change closure
bodies: identical rows, identical cumulative counts, identical refusal texts
and types. No allowlist entry moves -- ``size <= 0`` is not the lint's
``==``/``!=`` on a digest name, so these closures hold no counted seal site.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from prismaquant.joint_layer_quanta import append_read_phase  # noqa: E402


ENTRIES = [{"bytes": 10}, {"bytes": 20}, {"bytes": 30}]


def _reference_quanta_phase(phases, manifest_entries, name, indices,
                            cumulative):
    # Verbatim pre-change body of the build_quantum_boundary_readset /
    # build_quantum_executable_manifest closures (ValueError refusal).
    size = sum(manifest_entries[index]["bytes"] for index in indices)
    if size <= 0:
        raise ValueError(f"read phase {name} is empty: refusing")
    cumulative += size
    phases.append({"name": name, "entry_indices": list(indices),
                   "bytes": size, "cumulative_bytes": cumulative})
    return cumulative


def _reference_handoff_seal(phases, entries, name, indices, cumulative):
    # Verbatim pre-change body of the band_serial_manifest closure
    # (QuantumHandoffRefused refusal, no ": refusing" suffix).
    from prismaquant.joint_quantum_handoff import QuantumHandoffRefused

    size = sum(entries[index]["bytes"] for index in indices)
    if size <= 0:
        raise QuantumHandoffRefused(f"read phase {name} is empty")
    cumulative += size
    phases.append({"name": name, "entry_indices": list(indices),
                   "bytes": size, "cumulative_bytes": cumulative})
    return cumulative


def _quanta_refuse(phase):
    return ValueError(f"read phase {phase} is empty: refusing")


def _handoff_refuse(phase):
    from prismaquant.joint_quantum_handoff import QuantumHandoffRefused

    return QuantumHandoffRefused(f"read phase {phase} is empty")


def test_helper_matches_quanta_closure_row_for_row():
    want_phases, got_phases = [], []
    want_total = _reference_quanta_phase(
        want_phases, ENTRIES, "head", [0, 2], 0)
    got_total = append_read_phase(
        got_phases, ENTRIES, "head", [0, 2], 0, refuse=_quanta_refuse)
    want_total = _reference_quanta_phase(
        want_phases, ENTRIES, "tail", [1], want_total)
    got_total = append_read_phase(
        got_phases, ENTRIES, "tail", [1], got_total, refuse=_quanta_refuse)
    assert got_phases == want_phases == [
        {"name": "head", "entry_indices": [0, 2],
         "bytes": 40, "cumulative_bytes": 40},
        {"name": "tail", "entry_indices": [1],
         "bytes": 20, "cumulative_bytes": 60},
    ]
    assert got_total == want_total == 60


def test_helper_matches_handoff_closure_row_for_row():
    from prismaquant.joint_quantum_handoff import QuantumHandoffRefused

    want_phases, got_phases = [], []
    want_total = _reference_handoff_seal(
        want_phases, ENTRIES, "handoff-load", [0, 1, 2], 0)
    got_total = append_read_phase(
        got_phases, ENTRIES, "handoff-load", [0, 1, 2], 0,
        refuse=_handoff_refuse)
    assert got_phases == want_phases == [
        {"name": "handoff-load", "entry_indices": [0, 1, 2],
         "bytes": 60, "cumulative_bytes": 60},
    ]
    assert got_total == want_total == 60
    with pytest.raises(QuantumHandoffRefused,
                       match=r"^read phase x is empty$"):
        append_read_phase([], ENTRIES, "x", [], 0, refuse=_handoff_refuse)


def test_quanta_refusal_text_and_type():
    with pytest.raises(ValueError,
                       match=r"^read phase x is empty: refusing$"):
        append_read_phase([], ENTRIES, "x", [], 0, refuse=_quanta_refuse)
    with pytest.raises(ValueError,
                       match=r"^read phase x is empty: refusing$"):
        append_read_phase([], [{"bytes": 0}], "x", [0], 0,
                          refuse=_quanta_refuse)
