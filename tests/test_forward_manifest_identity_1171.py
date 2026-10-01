"""Forward group samples stay in execution metadata, not shared read-set identity."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from test_dispatch_stage_a_forward_split import FORWARD, N_BATCHES, RANGES, _package


@pytest.mark.parametrize("ranges", [RANGES, list(reversed(RANGES))])
def test_forward_groups_have_one_complete_manifest_identity(tmp_path, ranges):
    _plan, package = _package(tmp_path, tmp_path / "run", ranges=ranges)
    records = package["quanta"]
    wires = [Path(row["data_manifest"]["path"]).read_bytes() for row in records]
    assert len(set(wires)) == 1, "group samples must not change shared read-set bytes"
    assert len({row["data_manifest"]["sha256"] for row in records}) == 1
    assert [row["samples"] for row in records] == RANGES
    for wire in wires:
        manifest = json.loads(gzip.decompress(wire))
        assert [phase["name"] for phase in manifest["read_plan"]["phases"]] == ["head", *FORWARD]
        assert manifest["annotations"]["forward_split"] == {
            "role": "quantum", "ranges": RANGES,
            "n_batches": N_BATCHES, "group_size": 2,
        }
    prep = json.loads(gzip.decompress(Path(package["prep"]["data_manifest"]["path"]).read_bytes()))
    assert [phase["name"] for phase in prep["read_plan"]["phases"]] == ["head"]
    assert prep["annotations"]["forward_split"]["ranges"] == RANGES


def test_group_manifest_metadata_is_independently_owned(tmp_path):
    from tools.build_stagea_split_package import forward_manifests

    _plan, package = _package(tmp_path, tmp_path / "run")
    wire = Path(package["quanta"][0]["data_manifest"]["path"]).read_bytes()
    original = json.loads(gzip.decompress(wire))
    _prep, groups = forward_manifests(original, ranges=RANGES,
                                      n_batches=N_BATCHES, group_size=2)
    first, second = groups.values()
    first["annotations"]["forward_split"]["ranges"][0][1] = 999
    assert second["annotations"]["forward_split"]["ranges"] == RANGES


def test_forward_manifest_identity_does_not_expand_execution_partitions(tmp_path):
    _plan, package = _package(tmp_path, tmp_path / "run")
    partitions = [row["samples"] for row in package["quanta"]]
    assert partitions == RANGES
    assert [index for start, stop in partitions for index in range(start, stop)] == list(range(N_BATCHES))
    assert package["round_source_bytes"] == sum(row["source_bytes"] for row in package["quanta"])
