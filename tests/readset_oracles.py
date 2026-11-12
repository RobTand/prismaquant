"""Independent read-plan oracles for generated readset manifests (no executor SDK)."""


def phase_ranges(manifest):
    """The byte range of each phase in plan order, from the manifest's own cumulative table."""
    phases = (manifest["read_plan"]["phases"] if "read_plan" in manifest
              else manifest["annotations"]["phases"])
    ranges, at = [], 0
    for phase in phases:
        assert phase["cumulative_bytes"] == at + phase["bytes"], phase["name"]
        ranges.append({"name": phase["name"], "start_bytes": at, "end_bytes": phase["cumulative_bytes"]})
        at = phase["cumulative_bytes"]
    return ranges


def validate_plan(manifest):
    """Shape invariants the staged read contract also enforces: finite entries, in-range indices."""
    entries = manifest["entries"]
    assert manifest["entry_count"] == len(entries)
    assert manifest["total_bytes"] == sum(entry["bytes"] for entry in entries)
    for entry in entries:
        assert entry["path"].startswith("/") and entry["offset"] >= 0 and entry["bytes"] > 0
    for phase in manifest.get("read_plan", {}).get("phases", []):
        assert all(0 <= i < len(entries) for i in phase["entry_indices"]), phase["name"]
    return manifest
