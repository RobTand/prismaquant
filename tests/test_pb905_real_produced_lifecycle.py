"""Tiny private PB lifecycle, not a 12 GiB pacing measurement or live rollout."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments import pb905_checkpoint_export as exp
from prismaquant.cost_streaming import StreamedBoundaryArtifacts
from test_band_serial_handoff_produced import band_campaign, pb_source, producer_owner
import test_stage_a_produced_boundary_chain as chain
from test_stage_a_produced_boundary_chain import _isolated_launch_context  # noqa: F401

pytestmark = pytest.mark.own_process


def test_frozen_arms_keep_actual_export_commit_release_lifecycle(tmp_path, monkeypatch):
    _src, pb_repo = pb_source(monkeypatch)
    from prismaquant.produced_output_spool import ROOT_ENV, MAX_ENV
    from prismaquant.staged_lease import sdk_submodule
    from prismabuild import pool, storage_tiers
    producer, _consumer, storage = band_campaign(tmp_path)
    capacity = {"cpu": 4, "mem_gb": 4}
    spool_root = tmp_path / "spool"
    publication, queue, _env = producer_owner(
        tmp_path, pb_repo, producer, storage,
        producer_environment={ROOT_ENV: str(spool_root), MAX_ENV: str(1 << 20)},
        claim_capacity=capacity)
    storage = {**storage, "directory": str(Path(publication.output_prefix) / "exact")}
    # The existing tier's ordinary fill rule, not a patched admission decision.
    queue.mint_tier_capacity(chain.TIER, {chain.KIND: 4, storage_tiers.FILL_KIND: 32})
    path = queue.root / "tiers" / (chain.TIER + ".json")
    offered = json.loads(path.read_text())
    offered["tokens"] = {chain.KIND: 4, storage_tiers.FILL_KIND: 32}
    pool._write_json_atomic(path, offered)
    owner = StreamedBoundaryArtifacts(storage, driven_by="writer")
    owner.bind({"component": "PB905 real lifecycle"}, n_probes=1, published=True)
    owner.bind_produced_output(publication, group_size=2, n_batches=4,
                              max_entry_tensor_bytes=64, origin_lifetime="retain")
    arms = {publication.batch_id_for(kind=exp.KIND, boundary_index=index, group_index=0): index == 0
            for index in range(2)}
    wrapped = exp.FrozenArmBackend(owner._local_output_spool.backend, arms)
    owner._local_output_spool.backend = wrapped
    with chain._fleet(queue, tmp_path, capacity=capacity) as fleet:
        with owner:
            first = owner.write_produced_files([("first.bin", b"genuine-component")],
                                               kind=exp.KIND, boundary_index=0)
            second = owner.write_produced_files([("second.bin", b"genuine-component")],
                                                kind=exp.KIND, boundary_index=1)
    assert Path(first[0].path).read_bytes() == Path(second[0].path).read_bytes()
    outcomes = [json.loads(line) for line in fleet.stdout.splitlines() if line.startswith("{")]
    assert len(outcomes) == 2 and all(row["rc"] == 0 for row in outcomes), fleet.stderr[-3000:]
    observations = wrapped.observations
    assert [row["paced"] for row in observations] == [True, False]
    po = sdk_submodule("produced_output")
    commits = json.loads((Path(po.instance_dir(queue.root, publication.instance)) / "commitments.json").read_text())["batches"]
    for batch_id in arms:
        assert commits[batch_id]["origin_only"] is True
        assert commits[batch_id].get("lifetime", "retain") == "retain"
    assert not list(spool_root.rglob("*.bin"))
    report = owner.produced_output_report()["local_spool"]["exports"]
    assert len(report) == 2 and all(row["landed_unix"] and row["released_unix"] for row in report)
