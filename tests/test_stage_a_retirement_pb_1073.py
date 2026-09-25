"""A retirement's PrismaBuild accounting: the durable charge drops by the pins (PQ #1073).

A fixture Stage A run seals its referenced checkpoints; its entries are then
committed to a real PrismaBuild owner on a private queue as two batches, one
holding exactly the entries a checkpoint references and one holding a forward
boundary entry, and the run's chain state names that owner as its producer.
The retirement finds both batches from PrismaBuild's own records, retires the
first, leaves the second, and ``reclaim_origin`` drops the durable charge by
exactly the first batch's bytes.

The batches are committed with ``commit_origin_batch`` under a write-only
template: that is the one way a test can commit files it did not write
through a mover. ``reclaim_origin`` treats a staged batch, which Stage A
commits, the same way: it reads the batch record and stats each path.

Containment is PrismaBuild's own check. The owner is still claimed here, so
the real check refuses, and the retirement with it; the successful path
replaces the check with a recorder, as ``test_stage_a_chain_resume`` does.

PrismaBuild comes from the published generation pinned in
``pb_runtime_generation_pin.json`` (shared, PQ #1084). ``own_process`` (PQ #1008).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.own_process

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import test_stage_a_produced_boundary_chain as chain  # noqa: E402
from test_stage_a_produced_boundary_chain import (  # noqa: E402,F401
    _isolated_launch_context)
from test_stage_a_chain_resume import _offline_tier_policy  # noqa: E402,F401
from test_stage_a_retirement_1073 import (  # noqa: E402
    argv, completed, last_json, plane_paths, retire_main, successor)

PIN_PATH = chain.PB_GENERATION_PIN
SLOT = "boundary_entries"


def _pb(monkeypatch):
    monkeypatch.setattr(chain, "PIN_PATH", PIN_PATH)
    _src, pb_repo = chain._pb_source()
    from prismaquant.staged_lease import set_lease_helper_root
    set_lease_helper_root(str(pb_repo))
    return pb_repo


def _template(prefix: Path) -> dict:
    body = {"schema": "prismaquant.prismabuild.produced_output_template.v1",
            "version": 1, "output_prefix": str(prefix),
            "slots": {SLOT: {"class": "payload"}},
            "durable_maxima": {"payload_max_bytes": 1 << 30,
                               "checkpoint_max_bytes": 0, "temp_max_bytes": 1 << 30},
            "working_demands": {chain.TIER: {"minimum_gib": 0, "window_gib": 0}},
            "permitted_tiers": [chain.TIER], "write_only": True}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    return {**body, "template_id": f"pq-test-retirement-{digest}"}


def _owner(tmp_path, pb_repo, template):
    """A claimed owner that declared ``template``, and its bound publication."""
    from prismabuild import produced_output as po
    from prismaquant.stage_a_produced_output import BoundaryProducedPublication

    cas_root = tmp_path / "cas"
    q = chain._queue(tmp_path, gib=4)
    owner = chain._sealed_producer_request(tmp_path, cas_root, pb_repo, template)
    q.publish(action_key=owner, cas_root=str(cas_root),
              worker_script=str(pb_repo / "tools" / "prismabuild_worker.py"),
              checkout_root=str(tmp_path / "mover-checkout"),
              resources={"cpu": 1, "mem_gb": 1, **po.owner_demand_terms(template)},
              produced_output_template=template)
    claimed = q.claim(owner="w-owner")
    assert claimed is not None and claimed["action_key"] == owner
    control = chain._broker_control(q, owner)
    po.declare_template(q.root, template)
    env = {"PRISMABUILD_ACTION_KEY": owner,
           "PRISMABUILD_ACTION_NONCE": control["nonce"],
           "PRISMABUILD_ACTION_SCOPE": control["scope_id"]}
    publication = BoundaryProducedPublication.bind_from_admitted_owner(
        queue_root=q.root, tier=chain.TIER, env=env, slot=SLOT)
    return q, publication


def _commit(publication, batch_id, paths):
    from prismabuild import produced_output as po

    sizes = {path: os.path.getsize(path) for path in paths}
    publication.require_prewrite(batch_id=batch_id,
                                 payload_ceiling_bytes=sum(sizes.values()),
                                 paths=sorted(sizes))
    descriptors = [po.validate_descriptor({
        "schema": po.DESCRIPTOR_SCHEMA_V2, "slot": SLOT, "artifact_class": "payload",
        "path": path, "bytes": size,
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "producer_generation": publication.generation,
        "owner_action_key": publication.instance["owner_action_key"],
        "owner_attempt": dict(publication.instance["owner_attempt"]),
    }, publication.template, publication.instance) for path, size in sorted(sizes.items())]
    out = po.commit_origin_batch(publication.queue, publication.instance,
                                 publication.template, descriptors, batch_id=batch_id)
    assert out.get("ok"), out
    return sum(sizes.values())


def _name_producer(space, producer):
    """Seal ``producer`` into the run's chain state, as a PB-owned run records it."""
    from prismaquant.stage_a_chain_resume import _seal, chain_state_path

    path = chain_state_path(space)
    state = json.loads(path.read_text())
    state["producer"] = producer
    path.write_text(json.dumps(_seal(
        {k: v for k, v in state.items() if k != "chain_state_sha256"})))


def test_retirement_reclaims_exactly_the_pinned_batches(tmp_path, monkeypatch, capsys):
    pb_repo = _pb(monkeypatch)
    from prismaquant import stage_a_chain_resume as resume_mod
    from prismaquant import stage_a_retirement as retire

    root = tmp_path / "run"
    space = completed(root, monkeypatch)
    succ = successor(tmp_path, monkeypatch)
    pinned, _digests = plane_paths(space)
    forward = sorted(str(p) for p in (space / "exact-boundaries").rglob("*.pt")
                     if str(p) not in set(pinned))
    assert pinned and forward

    q, publication = _owner(tmp_path, pb_repo, _template(space / "exact-boundaries"))
    pinned_bytes = _commit(publication, "cotangent-pins", pinned)
    forward_bytes = _commit(publication, "forward-0", forward[:1])
    assert publication.durable_charge()["payload"] == pinned_bytes + forward_bytes
    producer = resume_mod.producer_binding(publication)
    _name_producer(space, producer)
    bindings = tmp_path / "bindings"
    bindings.mkdir()

    # The owner is still claimed: PrismaBuild's own containment check refuses.
    assert retire_main(argv(root, succ, [bindings])) == 3
    assert "not contained" in capsys.readouterr().err
    assert all(os.path.exists(path) for path in pinned)

    asked = []
    monkeypatch.setattr(resume_mod, "require_producer_contained", asked.append)
    assert retire_main(argv(root, succ, [bindings])) == 0
    report = last_json(capsys)
    assert asked == [producer]
    assert report["batches"] == {"cotangent-pins": "reclaimed"}
    assert [row["batch_id"] for row in report["untouched_batches"]] == ["forward-0"]
    assert report["durable_charge_before"] - report["durable_charge_after"] == pinned_bytes
    assert publication.durable_charge()["payload"] == forward_bytes
    assert not any(os.path.exists(path) for path in pinned)
    assert os.path.exists(forward[0])
    assert retire.retirement_record_path(space).is_file()


def test_a_batch_mixing_pins_and_live_entries_refuses(tmp_path, monkeypatch, capsys):
    pb_repo = _pb(monkeypatch)
    from prismaquant import stage_a_chain_resume as resume_mod

    root = tmp_path / "run"
    space = completed(root, monkeypatch)
    succ = successor(tmp_path, monkeypatch)
    pinned, _digests = plane_paths(space)
    forward = sorted(str(p) for p in (space / "exact-boundaries").rglob("*.pt")
                     if str(p) not in set(pinned))
    _q, publication = _owner(tmp_path, pb_repo, _template(space / "exact-boundaries"))
    _commit(publication, "mixed", [pinned[0], forward[0]])
    _name_producer(space, resume_mod.producer_binding(publication))
    monkeypatch.setattr(resume_mod, "require_producer_contained", lambda producer: None)
    bindings = tmp_path / "bindings"
    bindings.mkdir()

    assert retire_main(argv(root, succ, [bindings])) == 3
    assert "batch mixed mixes checkpoint entries" in capsys.readouterr().err
    assert os.path.exists(pinned[0]) and (space / "checkpoints").is_dir()


def test_a_changed_pinned_file_refuses(tmp_path, monkeypatch, capsys):
    pb_repo = _pb(monkeypatch)
    from prismaquant import stage_a_chain_resume as resume_mod

    root = tmp_path / "run"
    space = completed(root, monkeypatch)
    succ = successor(tmp_path, monkeypatch)
    pinned, _digests = plane_paths(space)
    _q, publication = _owner(tmp_path, pb_repo, _template(space / "exact-boundaries"))
    _commit(publication, "cotangent-pins", pinned)
    _name_producer(space, resume_mod.producer_binding(publication))
    monkeypatch.setattr(resume_mod, "require_producer_contained", lambda producer: None)
    # Replaced after the commit: another file under the committed name.
    payload = Path(pinned[0]).read_bytes()
    os.unlink(pinned[0])
    Path(pinned[0]).write_bytes(payload)
    bindings = tmp_path / "bindings"
    bindings.mkdir()

    assert retire_main(argv(root, succ, [bindings])) == 3
    assert "is not the file its commit recorded" in capsys.readouterr().err
    assert all(os.path.exists(path) for path in pinned)
