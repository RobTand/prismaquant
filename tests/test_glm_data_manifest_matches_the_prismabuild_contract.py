"""What the campaign writes is what PrismaBuild will accept, key for key.

``prismabuild.core.validate_data_manifest`` builds the manifest through
``_exact_mapping``: an unknown key is a refusal, not an ignored extra, and both
totals are recomputed and compared.  The producer lives in this repo and the
validator lives in PrismaBuild, so nothing in either tree fails when they
drift -- the campaign would simply be refused at submission, 1023 GB of warm
plan after the fact.  The contract is restated here from
``src/prismabuild/core.py`` (``_DATA_MANIFEST_KEYS``,
``_DATA_MANIFEST_ENTRY_KEYS``, ``validate_data_manifest``) because PrismaBuild
is not importable from this environment; it is seven key names and four rules,
and a mismatch is the only thing this test exists to catch.
"""

import json
import posixpath

from experiments import glm_data_manifests
from experiments.glm_data_manifests import SCHEMA, build_manifest

from test_glm_arc_prewarm_weight_extents import _campaign

#: prismabuild.core._DATA_MANIFEST_KEYS
PB_MANIFEST_KEYS = {"schema", "produced_by", "mount_prefix", "entries",
                    "entry_count", "total_bytes", "annotations"}
#: prismabuild.core._DATA_MANIFEST_ENTRY_KEYS
PB_ENTRY_KEYS = {"path", "offset", "bytes", "sha256"}


def test_a_built_manifest_carries_exactly_the_keys_prismabuild_accepts(
    tmp_path, monkeypatch,
):
    campaign, shard = _campaign(tmp_path)
    # The fixture's files are not under /mnt/shared, and the mount prefix is
    # the producer's own constant, so point the producer at the fixture's root
    # rather than moving the fixture.
    monkeypatch.setattr(glm_data_manifests, "SHARED_MOUNT", str(tmp_path))

    manifest = build_manifest(
        campaign, "row-0000", {"tool": "test", "unix": 0})

    assert set(manifest) == PB_MANIFEST_KEYS
    assert manifest["schema"] == SCHEMA == "prismaquant.prismabuild.data_manifest.v1"
    assert isinstance(manifest["produced_by"], dict)
    assert isinstance(manifest["annotations"], dict)

    prefix = manifest["mount_prefix"]
    assert prefix.startswith("/") and prefix == posixpath.normpath(prefix)
    assert prefix != "/"

    entries = manifest["entries"]
    assert entries, "an empty entry list is refused"
    seen = set()
    total = 0
    for entry in entries:
        assert set(entry) == PB_ENTRY_KEYS
        assert entry["path"].startswith(prefix + "/")
        assert entry["bytes"] > 0
        assert entry["offset"] >= 0
        assert (entry["path"], entry["offset"]) not in seen
        seen.add((entry["path"], entry["offset"]))
        total += entry["bytes"]

    # Both totals are recomputed by the validator and must agree: the prewarm
    # loop budgets ARC against total_bytes before it opens anything.
    assert manifest["entry_count"] == len(entries)
    assert manifest["total_bytes"] == total

    # And the whole thing is JSON, since it is written to a file and read back
    # by a different program.
    assert json.loads(json.dumps(manifest)) == manifest
