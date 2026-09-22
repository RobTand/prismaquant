"""Real PB source-host export -> canonical descriptor -> strict staged read.

Executed inside an admitted CPU test. The private fleet uses the actual PB
claim/execute/finish path for both export and stage movement; no payload mover,
receipt, progress implementation or leased read is replaced.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import test_stage_a_produced_boundary_chain as chain
from test_stage_a_produced_boundary_chain import _isolated_launch_context  # noqa: F401


def test_actual_export_ack_precedes_pq_publication_progress_and_strict_read(tmp_path, monkeypatch):
    from fleet_sdk import require_prismabuild_sdk
    require_prismabuild_sdk()
    pin_path = Path(__file__).with_name("stagea_local_spool_pb_pin.json")
    pin = json.loads(pin_path.read_text())
    root = Path(pin["bundle_root"])
    assert root.is_dir(), "the qualified PB source bundle must be provisioned"
    for name, digest in pin["files"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
    monkeypatch.setattr(chain, "PIN_PATH", pin_path)
    from prismaquant.produced_output_spool import ROOT_ENV, MAX_ENV
    storage, publication, queue, env, pb_repo = chain._bound_owner(
        tmp_path, staging_timeout_s=90, published=True,
        claim_capacity={"cpu": 4, "mem_gb": 4},
        producer_environment={ROOT_ENV: str(tmp_path / "local-output"),
                              MAX_ENV: str(1 << 20)})
    assert storage._local_output_spool is not None
    from prismaquant.joint_run_progress import JointRunProgress
    progress = JointRunProgress(layers=4, partitions=chain.GROUP_SIZE,
                               phases=["head", "layer-0"], log=lambda line: None)
    storage.watch_progress(progress)
    references = chain._write_group(storage)
    group = next(iter(storage._produced_groups.values()))
    batch_id = group["batch_id"]
    source_paths = list(storage._local_output_spool.directory(batch_id).glob("*.pt"))
    assert len(source_paths) == chain.GROUP_SIZE
    assert progress.units == 0
    assert group["published"] is None
    assert all(not Path(ref.path).exists() for ref in references)
    # The existing exact-activation reader records pin/tier identity but does
    # not increment the resolver byte counters. Guard canonical payload opens
    # directly, so zero origin reads is proved rather than inferred from them.
    import builtins
    canonical_paths = {ref.path for ref in references}
    original_open = builtins.open
    original_path_open = Path.open
    def guarded_open(path, mode="r", *args, **kwargs):
        if isinstance(path, (str, Path)) and str(path) in canonical_paths and "r" in mode:
            raise AssertionError("consumer opened canonical pool payload")
        return original_open(path, mode, *args, **kwargs)
    def guarded_path_open(path, mode="r", *args, **kwargs):
        if str(path) in canonical_paths and "r" in mode:
            raise AssertionError("consumer opened canonical pool payload")
        return original_path_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    # The real fleet begins only after the no-ack assertions; PQ's read must
    # wait for its export, then publish and wait for the separately admitted
    # stage mover before opening any tensor.
    with chain._fleet(queue, tmp_path, capacity={"cpu": 4, "mem_gb": 4}) as fleet:
        chain._strict(monkeypatch, env, pb_repo, queue)
        with storage.prefetch(references) as window:
            for index, reference in enumerate(references):
                assert torch.equal(storage.get(window, reference),
                                   torch.arange(8, dtype=torch.float32) + index)
            resolver = group["context"][0]
            report = resolver.report()
            assert report["bytes_from_pool"] == 0
            assert report["serving_tiers"] and all(
                row["pin_id"] and row["serving_tier"] == "stage" for row in report["serving_tiers"])
            assert storage.telemetry["read_tensor_bytes"] == sum(ref.tensor_bytes for ref in references)
        storage.settle_local_output()
        assert progress.units == chain.GROUP_SIZE
        assert not any(path.exists() for path in source_paths)
        assert all(Path(ref.path).stat().st_size == ref.file_bytes for ref in references)
    outcomes = [json.loads(line) for line in fleet.stdout.splitlines() if line.startswith("{")]
    assert len(outcomes) >= 2 and all(row["rc"] == 0 for row in outcomes), outcomes
    assert storage.telemetry["produced_groups_published"] == 1
    assert storage.telemetry["produced_groups_materialized"] == 1
    assert storage.produced_output_report()["local_spool"]["pending_groups"] == 0
    assert not list(queue.root.rglob("*.lease.json"))
