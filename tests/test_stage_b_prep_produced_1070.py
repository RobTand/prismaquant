"""The Stage B preparation's outputs as real PrismaBuild produced output (PQ #1070).

The generator runs with the preparation's options and ``--produced-output``
under a real admitted owner on a private queue: the owner's sealed request
declares the write-only template the submission tool wrote, the queue claims
it, and the generator binds from the launch environment exactly as it does in
production (the queue root from ``PRISMABUILD_RESIDENCY_MAP``). What
PrismaBuild then holds is checked against the metadata root on disk:

* every file the run wrote is in a committed origin batch, with its bytes and
  digest, and nothing else is;
* a replay over the same root writes nothing and commits nothing;
* a path outside the template's prefix, a template over another root and a
  second group of one kind refuse before a byte is written.

PrismaBuild comes from the published runtime generation pinned in
``pb_runtime_generation_pin.json`` (shared, PQ #1084): the Stage A pin predates the write-only
template (PB #912). Marked ``own_process`` because resolving a pinned
candidate leaves ``prismabuild`` in ``sys.modules`` (PQ #1008).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

# PQ #1008: one pinned prismabuild per process (tests/conftest.py).
pytestmark = pytest.mark.own_process

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import test_stage_a_produced_boundary_chain as chain  # noqa: E402
from test_stage_a_produced_boundary_chain import (  # noqa: E402,F401
    _isolated_launch_context)
from test_stage_b_prep_io_1070 import (  # noqa: E402
    generator_args, prep_fixture, write_submission)

PIN_PATH = chain.PB_GENERATION_PIN


def _pb(monkeypatch):
    monkeypatch.setattr(chain, "PIN_PATH", PIN_PATH)
    _src, pb_repo = chain._pb_source()
    from prismaquant.staged_lease import set_lease_helper_root
    set_lease_helper_root(str(pb_repo))
    from prismabuild import produced_output as po
    if not callable(getattr(po, "commit_origin_batch", None)):
        pytest.fail("the pinned PrismaBuild has no commit_origin_batch")
    return pb_repo


def admitted_owner(tmp_path, pb_repo, template, monkeypatch):
    """A real owner that declared ``template``, claimed, in this process's env."""
    from prismabuild import produced_output as po

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
    chain._announce_tier(q, tmp_path / "stage", pb_repo)
    # The launcher's own variables: the queue root is derived from the map's
    # path, which need not exist for a write-only owner.
    monkeypatch.setenv("PRISMABUILD_ACTION_KEY", owner)
    monkeypatch.setenv("PRISMABUILD_ACTION_NONCE", control["nonce"])
    monkeypatch.setenv("PRISMABUILD_ACTION_SCOPE", control["scope_id"])
    monkeypatch.setenv("PRISMABUILD_RESIDENCY_MAP",
                       str(q.root / "residency" / f"{owner}.json"))
    return q, owner


def _files_under(root: Path) -> dict[str, bytes]:
    return {str(path): path.read_bytes() for path in sorted(root.rglob("*"))
            if path.is_file()}


def test_the_generator_commits_every_file_it_writes(tmp_path, monkeypatch):
    pb_repo = _pb(monkeypatch)
    from prismabuild import produced_output as po
    import regenerate_joint_quanta as regen
    from prismaquant import stage_b_prep_io as io

    campaign, receipt_path, _spans = prep_fixture(tmp_path)
    metadata = tmp_path / "metadata-generation"
    _submission, _manifest, template = write_submission(
        tmp_path, campaign, receipt_path, metadata, tier=chain.TIER)
    q, owner = admitted_owner(tmp_path, pb_repo, template, monkeypatch)
    io.reset_preparation_publications_for_tests()

    argv = generator_args(tmp_path, campaign, receipt_path, metadata) + ["--produced-output"]
    assert regen.main(argv) == 0
    publication = io._BOUND[owner]
    assert [batch["kind"] for batch in publication.batches] == ["manifests", "records"]
    refs = [batch["ref"] for batch in publication.batches]
    committed = po.origin_batch_manifest(q.root, refs)

    written = _files_under(metadata)
    assert written, "the generator wrote nothing"
    assert {entry["path"] for entry in committed["entries"]} == set(written)
    for entry in committed["entries"]:
        payload = written[entry["path"]]
        assert entry["bytes"] == len(payload)
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert sum(batch["files"] for batch in publication.batches) == len(written)

    # A replay over the same root: every file is present with its bytes, so
    # nothing is written, prewritten or committed.
    assert regen.main(argv) == 0
    assert [batch["kind"] for batch in publication.batches] == ["manifests", "records"]
    assert _files_under(metadata) == written


def test_a_bound_publication_refuses_before_writing(tmp_path, monkeypatch):
    pb_repo = _pb(monkeypatch)
    from prismaquant import stage_b_prep_io as io

    metadata = tmp_path / "metadata"
    template = io.build_preparation_template(
        metadata_root=metadata, tier=chain.TIER,
        payload_max_bytes=io.preparation_payload_ceiling(1))
    _q, owner = admitted_owner(tmp_path, pb_repo, template, monkeypatch)
    io.reset_preparation_publications_for_tests()

    with pytest.raises(io.PreparationPublicationRefused, match="declares"):
        io.bind_preparation_publication(tmp_path / "elsewhere", required=True)
    publication = io.bind_preparation_publication(metadata, required=True)
    assert io.bind_preparation_publication(metadata, required=True) is publication

    outside = tmp_path / "outside.json"
    with pytest.raises(io.PreparationPublicationRefused, match="outside the template"):
        io.publish_files(publication, "stray", [(outside, b"x", "stray")])
    assert not outside.exists()

    inside = metadata / "a.json"
    metadata.mkdir()
    assert io.publish_files(publication, "one", [(inside, b"A", "a")]) == [inside]
    with pytest.raises(io.PreparationPublicationRefused, match="once per action"):
        io.publish_files(publication, "one", [(metadata / "b.json", b"B", "b")])
    assert not (metadata / "b.json").exists()
    assert publication.batches[-1]["kind"] == "one"
    assert owner
