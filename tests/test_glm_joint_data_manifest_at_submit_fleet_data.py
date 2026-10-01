"""The real joint pass's read set, measured on the frozen census (PQ #1014).

This test walks about 36k checkpoint shards and a campaign tree that no
PrismaBuild action declares, so it carries the ``fleet_data`` mark. It lives
in its own file because ``pbtest`` decides fleet data per file: a file that
marks any test needs a ``--data-manifest``, and the merge queue leaves such
files out. Keeping the mark here lets the hermetic submit tests in
``test_glm_joint_data_manifest_at_submit.py`` run in the queue (PQ #1954).
"""

import gzip
import json
from pathlib import Path

import pytest

from test_glm_joint_data_manifest_at_submit import (  # noqa: E402,F401
    glm_data_manifests,
    scratch,
)


#: The frozen census the measured numbers in the issue and in
#: ``docs/ARCHITECTURE.md`` came from. Absent on a fleet CPU box that has no
#: campaign mount, and absent before the campaign merge publishes the
#: checkpoint, so the smoke names its reason rather than failing.
BASE = Path("/mnt/shared/tessera-measurements/glm-canonical-census-20260908")
REAL_PLAN = BASE / "first-proof-joint-preparation-03" / "plan.inputs-resolved.json"
REAL_PLAN_TEMPLATE = BASE / "first-proof-joint-preparation-03" / "dryrun" / "plan.template.json"
#: The joint pass's whole read set, measured from the dl380g10 local pool at
#: 07:57Z on 2026-09-13: 5.20 TB over 371,734 entries. The band is wide on
#: purpose, because the tree is being written while it is read. A joint
#: ``prepare`` has been running against this plan since 04:59Z, and it writes
#: a decoded shard for every rung the campaign adopted rather than encoded --
#: each one marked by a ``.render_origin.json`` record beside it in the row
#: cache. Those records are written from one place only -- the head's
#: ``_resolve_render_origin`` -- and the running pass was still writing them
#: at 08:12Z, three hours after it started, so the head placement below is
#: measured rather than inferred. The read set gained 3,216 renders in
#: eighteen minutes (5.14 TB at
#: 07:39Z, 5.20 TB at 07:57Z), and the brief's 4.75 TB estimate was taken
#: earlier the same morning against fewer shards. Projected settled size once
#: the remaining 97,302 shards exist: 5.20 TB + 97,302 x the 16.8 MB mean
#: render = about 6.8 TB, which is where the upper bound below comes from.
#: What the test pins is the shape -- one head phase and bounded unit phases,
#: a read set in the terabytes, dominated by captures, renders and wires --
#: not a byte count of a tree that is still being written.
TOTAL_BYTES_BAND = (4.5e12, 7.5e12)
MIN_EXPECTED_PHASES = 46


def _real_plan() -> Path | None:
    for candidate in (REAL_PLAN, REAL_PLAN_TEMPLATE):
        try:
            if candidate.is_file():
                return candidate
        except OSError:
            continue
    return None


#: Walks about 36k checkpoint shards and the campaign tree that no
#: PrismaBuild action declares: skipped unless asked for (PQ #1014).
@pytest.mark.fleet_data
def test_the_real_joint_pass_read_set_is_terabytes_in_bounded_phases(scratch):
    plan = _real_plan()
    if plan is None:
        pytest.skip(f"the frozen joint plan is absent: {REAL_PLAN}")
    payload = json.loads(plan.read_text())
    checkpoint = Path(payload["inputs"]["merged_checkpoint"]["path"])
    parts = checkpoint.with_name(checkpoint.name + ".parts") / "units"
    # The merge publishes the shards first and the manifest that names them
    # last, so both halves are checked: a joint pass is not submittable until
    # the manifest is there.
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        pytest.skip(
            "the campaign merge has not published the merged checkpoint yet: "
            f"{checkpoint}")
    if not parts.is_dir():
        pytest.skip(
            "the campaign merge has not published the checkpoint unit shards "
            f"yet: {parts}")
    source_cache = Path(payload["output_root"]) / "prepare/source-identity.json"
    if source_cache.is_file():
        cached = json.loads(source_cache.read_text())
        first = cached["fingerprints"][0]
        if first["device"] != Path(first["path"]).stat().st_dev:
            pytest.skip("the full-source SHA cache was written through another "
                        "mount instance; this host cannot submit its reuse "
                        "request")

    manifest = glm_data_manifests.build_joint_pass_manifest(
        str(plan), command="prepare",
        produced_by={"tool": "test", "commit": "0" * 40})

    phases = manifest["annotations"]["phases"]
    assert MIN_EXPECTED_PHASES <= len(phases) <= 2048, len(phases)
    assert phases[0]["name"] == "head"
    assert {int(phase["name"].split("-")[1]) for phase in phases[1:]} == set(range(45))
    assert all(phase["name"].startswith("layer-") and "-part-" in phase["name"]
               for phase in phases[1:])
    assert len(manifest["annotations"]["phase_start_units"]) == 36423
    low, high = TOTAL_BYTES_BAND
    assert low <= manifest["total_bytes"] <= high, (
        f"{manifest['total_bytes']} bytes is outside the measured band "
        f"{low:.3g}-{high:.3g}")
    annotations = manifest["annotations"]
    kinds = annotations["bytes"]
    assert kinds["captures"] > 2e12 and kinds["renders"] > 1e12
    assert kinds["wires"] > 7e11 and kinds["source_extents"] > 5e11

    # Rungs this campaign adopted have a wire and no decoded shard. The head
    # decodes one per cell, so every such wire is declared in the head phase
    # and its shard is declared nowhere.
    assert annotations["renders_absent"] == (
        annotations["measured_cells"] - annotations["counts"]["renders"])
    assert phases[0]["bytes"] >= annotations["synthesized_render_wire_bytes"]

    # The full read set exceeds the 64 MiB plain-file ceiling, but current PB
    # accepts a gzip member up to 64 MiB stored / 512 MiB expanded. The submit
    # path seals that member rather than truncating the read set.
    encoded = json.dumps(manifest, separators=(",", ":")).encode() + b"\n"
    assert len(encoded) > glm_data_manifests.MAX_MANIFEST_BYTES, (
        "the joint pass's read set now fits the plain manifest ceiling")
    assert len(encoded) <= 512 * 1024 * 1024
    assert len(gzip.compress(encoded, mtime=0)) <= glm_data_manifests.MAX_MANIFEST_BYTES

    # Written under the test's own scratch directory, never into the frozen
    # campaign tree.
    (scratch / "joint.prepare.json").write_bytes(encoded)
