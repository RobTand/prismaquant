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


@pytest.mark.parametrize("placement", ["inside", "sibling"])
def test_generated_fixture_placement_preserves_the_source_cleanliness_guard(
        tmp_path, placement):
    import subprocess

    checkout = tmp_path / "authenticated-checkout"
    checkout.mkdir()
    (checkout / "committed.py").write_text("# authenticated source\n")
    for args in (("init", "-q"), ("add", "committed.py"),
                 ("-c", "user.name=CPU fixture", "-c",
                  "user.email=fixture@example.invalid", "commit", "-q",
                  "-m", "authenticated source")):
        subprocess.run(["git", "-C", str(checkout), *args],
                       check=True, capture_output=True, text=True)
    head = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                          check=True, capture_output=True, text=True).stdout.strip()
    workspace = (checkout / "generated" if placement == "inside"
                 else checkout.parent / "fixture-work")
    workspace.mkdir()
    (workspace / "fixture.py").write_text("# independently generated fixture\n")
    if placement == "inside":
        with pytest.raises(pins.NonQualified, match="uncommitted executable files") as refused:
            pins.record_snapshots(checkout=checkout)
        assert refused.value.detail["dirty"] == ["generated/"]
    else:
        snapshots = pins.record_snapshots(checkout=checkout)
        assert snapshots["pq_head"] == head
        assert snapshots["pb_rev"] == pins.PB_CANDIDATE_REV


def test_explicit_acceptance_objects_do_not_probe_fleet_or_network(tmp_path, monkeypatch):
    import fleet_acceptance_pins as pins

    root = tmp_path / "source"
    root.mkdir()
    for name in ("src/prismabuild/reader_lease.py", "tools/fleet/resource_broker.py"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# pinned acceptance fixture\n")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "acceptance fixture")
    revision = git("rev-parse", "HEAD")
    monkeypatch.setattr(pins, "PB_CANDIDATE_REV", revision)
    monkeypatch.setenv("PRISMAQUANT_PB_ACCEPTANCE_GIT_DIR", str(root / ".git"))
    class FleetProbeForbidden:
        def __truediv__(self, name):
            raise AssertionError("explicit objects must suppress fleet lookup")
    monkeypatch.setattr(pins, "PB_PINNED_CHECKOUT", FleetProbeForbidden())
    monkeypatch.setattr(pins, "PB_PINNED_CHECKOUT_LEGACY", FleetProbeForbidden())
    def network_forbidden(*args, **kwargs):
        raise AssertionError("explicit objects must suppress network fallback")
    monkeypatch.setattr(pins, "_resolve_via_network", network_forbidden)
    result = pins.resolve_pb_candidate(tmp_path / "resolved")
    assert result["rev"] == revision
    assert (Path(result["tree"]) / "src/prismabuild/reader_lease.py").read_bytes() == b"# pinned acceptance fixture\n"
