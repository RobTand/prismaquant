"""Generated fixtures cannot silently enter native authenticated source."""
from pathlib import Path
import pytest
from prismaquant.flux_jobs import _source_identity

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
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", "-q", head],
                   check=True, capture_output=True, text=True)
    workspace = (checkout / "generated" if placement == "inside"
                 else checkout.parent / "fixture-work")
    workspace.mkdir()
    (workspace / "fixture.py").write_text("# independently generated fixture\n")
    if placement == "inside":
        with pytest.raises(ValueError, match="source checkout must be clean"):
            _source_identity(checkout)
    else:
        snapshots = _source_identity(checkout)
        assert snapshots["commit"] == head
        tree = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD^{tree}"],
                              check=True, capture_output=True, text=True).stdout.strip()
        assert snapshots["tree"] == tree and snapshots["checkout"] == str(checkout)
