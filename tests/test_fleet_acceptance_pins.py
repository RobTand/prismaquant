"""Pin resolution proves the commit and verifies the tree (or skips named)."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fleet_acceptance_pins as pins


def test_recorded_snapshots_name_exact_trees():
    """ID-08: every result can say which PB rev + tree and PQ HEAD ran."""
    checkout = Path(__file__).resolve().parents[1]
    try:
        snapshots = pins.record_snapshots(checkout=checkout)
    except pins.NonQualified as exc:
        pytest.skip(f"nonqualified: {exc.reason}")
    assert snapshots["pb_rev"] == pins.PB_CANDIDATE_REV
    assert len(snapshots["pq_head"]) == 40


def test_resolution_verifies_the_whole_tree(tmp_path):
    """Resolve proves the commit; extraction matches the revision exactly.

    Needs the pinned checkout's objects; otherwise this skips named --
    the scenarios depending on it skip the same way.
    """
    try:
        resolved = pins.resolve_pb_candidate(tmp_path / "candidate")
    except pins.NonQualified as exc:
        pytest.skip(f"nonqualified: {exc.reason}")
    assert resolved["rev"] == pins.PB_CANDIDATE_REV
    assert len(resolved["tree_sha256"]) == 64
    tree = Path(resolved["tree"])
    for name in pins.REUSED_PREFIXES:
        assert (tree / name).is_dir(), name
    assert (tree / "src" / "prismabuild" / "reader_lease.py").is_file()
    assert (tree / "tools" / "fleet" / "resource_broker.py").is_file()
