"""A producer quantum's handoff through a real PrismaBuild produced output (PQ #996).

Inside an admitted action, quantum ``L`` writes its handoff entries into the
output prefix its row declared (``--produced-output-template``). This drives
that path end to end on tiny tensors: the template is the one the dispatcher
writes for the row, the owner is a real sealed request published, claimed
and bound on a private queue, and the emitter writes through the bound
publication. The consumer quantum then binds the handoff with its own
checks.

The template is write-only (PrismaBuild #912, PQ #1075): the producer never
reads its handoff back, so every group -- each entry group, then the record
group -- is committed at its origin as a batch with the ``consumed``
lifetime (PrismaBuild #914). The consumer stages the entries as ordinary
inputs of its derived readset, and PrismaBuild retires the batches once
every consumer that declared them has succeeded.
``test_band_serial_handoff_spool_real_pb`` drives the same emitter through
PrismaBuild's local output spool.

The PrismaBuild these tests run against is the published runtime generation
``pb_runtime_generation_pin.json`` names, the one pin the write-only
produced-output suites share (PQ #1084): the older bundles the Stage A tests
pin reject a write-only template. The module is marked
``own_process``: resolving the pinned bundle leaves ``prismabuild`` in
``sys.modules``, so in a shared session its tests run in a child pytest of
their own (``tests/conftest.py``, PQ #1008).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

# PQ #1008: one pinned prismabuild per process (tests/conftest.py).
pytestmark = pytest.mark.own_process

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import test_stage_a_produced_boundary_chain as chain  # noqa: E402
from test_quantum_executable_readset import (  # noqa: E402
    N_BATCHES, N_PROBES, _bind_slice, _tiny_receipt, _tiny_records)
from test_stage_a_produced_boundary_chain import (  # noqa: E402,F401
    _isolated_launch_context)

PREFETCH = 2
ARTIFACT_MAX = 1 << 20
#: The PrismaBuild with write-only templates, origin batches and their
#: retirement (#912, #914): the shared published-generation pin (PQ #1084).
ORIGIN_PIN = chain.PB_GENERATION_PIN
#: Entry groups per handoff: each probe's batches in groups of ``PREFETCH``.
ENTRY_GROUPS = N_PROBES * -(-N_BATCHES // PREFETCH)


def pb_source(monkeypatch):
    """Resolve the pinned PrismaBuild with origin batches: ``(src, bundle)``.

    Also names it as the lease helper root, so ``sdk_submodule`` loads the
    same generation.
    """
    monkeypatch.setattr(chain, "PIN_PATH", ORIGIN_PIN)
    src, pb_repo = chain._pb_source()
    from prismaquant.staged_lease import set_lease_helper_root
    set_lease_helper_root(str(pb_repo))
    return src, pb_repo


class _Owner:
    def __init__(self, state):
        self.state = state

    def state_dict(self):
        return dict(self.state)


def _adjoint_slice(record):
    return json.loads(Path(record["adjoint"]["slice_path"]).read_text())


def band_campaign(tmp_path):
    """Layers 3 (producer) and 2 (consumer) of band 4, bound to Stage A."""
    from prismaquant.cost_streaming import BOUNDARY_STORAGE_SCHEMA

    records, _parent = _tiny_records(tmp_path)
    receipt = _tiny_receipt(tmp_path, records[0]["campaign"])
    bound = {record["layer"]: _bind_slice(
        record, receipt, tmp_path / "adjoint-slices")[0]
        for record in records if record["layer"] in (2, 3)}
    storage = {"schema": BOUNDARY_STORAGE_SCHEMA,
               "directory": str(tmp_path / "exact"),
               "max_resident_bytes": 1 << 24, "max_auxiliary_bytes": 1 << 24,
               "max_artifact_bytes": ARTIFACT_MAX, "prefetch_batches": PREFETCH}
    return bound[3], bound[2], storage


def producer_owner(tmp_path, pb_repo, producer, storage, *,
                   producer_environment=None, claim_capacity=None,
                   max_attempts=None):
    """The dispatcher's template for ``producer``, and a real admitted owner
    that declared it on a private queue: ``(publication, queue, env)``.

    ``max_attempts=1`` makes a failed finish terminal instead of a requeue.
    """
    from prismabuild import produced_output as po

    import dispatch_joint_quanta as dispatch
    from prismaquant.joint_quantum_handoff import handoff_root
    from prismaquant.stage_a_produced_output import BoundaryProducedPublication

    # The campaign root the records were cut under (_tiny_records): the
    # template names the handoff directory under the dispatch's output root
    # (PQ #1200), and the emitter writes inside the record's output space.
    template_path = dispatch.handoff_template_path(
        producer, plan={"execution": {"boundary_storage": storage}},
        adjoint_slice=_adjoint_slice(producer), tier=chain.TIER,
        output_root=tmp_path / "run")
    template = json.loads(template_path.read_text())
    assert template["output_prefix"] == str(
        handoff_root(producer["output_space"]["root"]).resolve())
    assert template["durable_maxima"]["payload_max_bytes"] == ARTIFACT_MAX
    # Write-only (PQ #1075): no stage window, so admission charges nothing.
    assert template["write_only"] is True
    assert template["working_demands"] == {
        chain.TIER: {"minimum_gib": 0, "window_gib": 0}}
    assert po.owner_demand_terms(template) == {}

    cas_root = tmp_path / "cas"
    q = chain._queue(tmp_path, gib=4)
    owner = chain._sealed_producer_request(
        tmp_path, cas_root, pb_repo, template,
        producer_environment=producer_environment)
    q.publish(action_key=owner, cas_root=str(cas_root),
              worker_script=str(pb_repo / "tools" / "prismabuild_worker.py"),
              checkout_root=str(tmp_path / "mover-checkout"),
              resources={"cpu": 1, "mem_gb": 1, **po.owner_demand_terms(template)},
              produced_output_template=template,
              **({} if max_attempts is None else {"max_attempts": max_attempts}))
    claimed = q.claim(owner="w-owner", capacity=claim_capacity)
    assert claimed is not None and claimed["action_key"] == owner
    control = chain._broker_control(q, owner)
    env = {"PRISMABUILD_ACTION_KEY": owner,
           "PRISMABUILD_ACTION_NONCE": control["nonce"],
           "PRISMABUILD_ACTION_SCOPE": control["scope_id"],
           **(producer_environment or {})}
    po.declare_template(q.root, template)
    publication = BoundaryProducedPublication.bind_from_admitted_owner(
        queue_root=q.root, tier=chain.TIER, env=env, command_extra=("--unpaced",))
    chain._announce_tier(q, tmp_path / "stage", pb_repo)
    return publication, q, env


def emit_handoff(producer, storage, publication, *, emitters=None):
    from prismaquant.joint_quantum_handoff import HandoffEmitter

    plane = {(p, b): torch.full((2, 4), 10.0 * p + b)
             for p in range(N_PROBES) for b in range(N_BATCHES)}
    owners = [[_Owner({"scale": float(p + b)}) for b in range(N_BATCHES)]
              for p in range(N_PROBES)]
    emitter = HandoffEmitter(record=producer, adjoint_slice=_adjoint_slice(producer),
                             boundary_storage=storage, capture_batch=1,
                             publication=publication)
    if emitters is not None:
        emitters.append(emitter)
    published = emitter.emit(grad_plane=plane, cotangent_owners=owners,
                             n_probes=N_PROBES, n_batches=N_BATCHES)
    return published, plane


def check_consumer_binds(published, plane, consumer, publication):
    """The consumer's own checks accept the handoff, and it is the plane."""
    from prismaquant.joint_adjoint_checkpoints import read_exact_entry_tensors
    from prismaquant.joint_quantum_handoff import load_quantum_handoff

    handoff = load_quantum_handoff(published["path"], published["sha256"],
                                   record=consumer,
                                   adjoint_slice=_adjoint_slice(consumer))
    for entry in handoff["activation_entries"]:
        assert publication.contains(Path(entry["path"]))
        coordinates = entry["metadata"]["identity"]["coordinates"]
        tensor = read_exact_entry_tensors(
            [entry], expected_session=handoff["session"])[entry["name"]]
        assert torch.equal(tensor, plane[(coordinates["probe"],
                                          coordinates["batch"])])
    assert publication.contains(Path(handoff["owner_states"]["path"]))
    assert publication.contains(Path(published["path"]))
    return handoff


def check_origin_batches(publication, producer, published, handoff):
    """Every handoff group is committed at its origin, consumed (PQ #1075).

    One batch per entry group, then the record group (``owner-states.pkl``
    and ``handoff.json``, PQ #1015), in that order: each origin-only, with
    the ``consumed`` lifetime, its prewrite consumed by the commit and no
    temporary left. The batch-only manifest PrismaBuild derives from the
    refs names every handoff file with the bytes and sha256 the handoff
    records, which is what a consumer checks each row against before it
    reads it. Returns the refs.
    """
    from prismabuild import produced_output as po
    from prismaquant.joint_quantum_handoff import (
        HANDOFF_ORIGIN_LIFETIME, HANDOFF_OWNER_STATES_NAME,
        HANDOFF_RECORD_BATCH_KIND, HANDOFF_RECORD_NAME)

    queue_root, instance = publication.queue.root, publication.instance
    refs = published["origin_batches"]
    assert len(refs) == ENTRY_GROUPS + 1, refs
    record_batch = publication.batch_id_for(
        kind=HANDOFF_RECORD_BATCH_KIND, boundary_index=int(producer["layer"]),
        group_index=0)
    assert refs[-1]["batch_id"] == record_batch, "the record group commits last"
    assert len({ref["batch_id"] for ref in refs}) == len(refs)
    batches = po._read_commitments(
        po._commitments_path(queue_root, instance))["batches"]
    directory = Path(published["path"]).parent
    finals = [directory / HANDOFF_OWNER_STATES_NAME, directory / HANDOFF_RECORD_NAME]
    for ref in refs:
        assert ref["owner_action_key"] == instance["owner_action_key"]
        assert ref["template_id"] == instance["template_id"]
        entry = batches[ref["batch_id"]]
        assert entry["origin_only"] is True and entry["mover_key"] is None
        assert entry["lifetime"] == HANDOFF_ORIGIN_LIFETIME == "consumed"
        assert entry["manifest_digest"] == ref["manifest_digest"]
        assert not entry["origin_reclaimed"]
        assert po._read_prewrite(
            po._prewrites_dir(queue_root, instance)
            / f"{ref['batch_id']}.prewrite.json") is None, (
            "the commit consumed the group's prewrite")
    assert batches[record_batch]["paths"] == sorted(str(f) for f in finals)
    assert not any(Path(str(f) + ".tmp").exists() for f in finals)

    manifest = po.origin_batch_manifest(queue_root, refs)
    rows = {row["path"]: row for row in manifest["entries"]}
    expected = {entry["path"]: (entry["file_bytes"], entry["sha256"])
                for entry in handoff["activation_entries"]}
    expected[str(finals[0])] = (handoff["owner_states"]["file_bytes"],
                                handoff["owner_states"]["sha256"])
    expected[str(finals[1])] = (Path(published["path"]).stat().st_size,
                                published["sha256"])
    assert {path: (row["bytes"], row["sha256"]) for path, row in rows.items()} == (
        expected)
    for path, (size, digest) in expected.items():
        raw = Path(path).read_bytes()
        assert len(raw) == size and hashlib.sha256(raw).hexdigest() == digest
    return refs


def test_a_producer_commits_its_handoff_as_origin_batches(
        tmp_path, monkeypatch):
    """Acceptance 1 (PQ #1075): write-only template, one origin batch per group."""
    _src, pb_repo = pb_source(monkeypatch)
    from prismaquant.joint_quantum_handoff import (
        QuantumHandoffRefused, bind_handoff_publication)
    from prismaquant.stage_a_produced_output import BoundaryProducedPublication

    producer, consumer, storage = band_campaign(tmp_path)
    publication, _q, env = producer_owner(tmp_path, pb_repo, producer, storage)
    assert publication.write_only is True

    # The quantum binds exactly as Stage A does in production (queue and
    # tier from the launch context); here that answer is the private queue's.
    calls = []

    def bind(cls, **kwargs):
        calls.append(kwargs)
        return publication

    monkeypatch.setattr(BoundaryProducedPublication, "bind_from_admitted_owner",
                        classmethod(bind))
    assert bind_handoff_publication(boundary_storage=storage, env={}) is None
    assert bind_handoff_publication(boundary_storage=storage, env=env) is publication
    assert calls == [{"queue_root": None, "tier": None, "env": env,
                      "command_extra": ()}]
    with pytest.raises(QuantumHandoffRefused, match="durable payload maximum"):
        bind_handoff_publication(
            boundary_storage={**storage, "max_artifact_bytes": ARTIFACT_MAX * 2},
            env=env)

    published, plane = emit_handoff(producer, storage, publication)
    handoff = check_consumer_binds(published, plane, consumer, publication)
    check_origin_batches(publication, producer, published, handoff)


def test_a_read_back_handoff_template_is_refused(tmp_path, monkeypatch):
    """A handoff template that reads back could never commit its groups.

    The producer refuses it before any write, at binding and again at the
    emitter, instead of leaving retained prewrites behind (PQ #1075).
    """
    pb_source(monkeypatch)
    from prismaquant.joint_quantum_handoff import (
        HandoffEmitter, QuantumHandoffRefused, bind_handoff_publication)
    from prismaquant.stage_a_produced_output import BoundaryProducedPublication

    class ReadBack:
        write_only = False

    monkeypatch.setattr(BoundaryProducedPublication, "bind_from_admitted_owner",
                        classmethod(lambda cls, **kwargs: ReadBack()))
    producer, _consumer, storage = band_campaign(tmp_path)
    with pytest.raises(QuantumHandoffRefused, match="not write-only"):
        bind_handoff_publication(boundary_storage=storage,
                                 env={"PRISMABUILD_ACTION_KEY": "0" * 64})
    with pytest.raises(QuantumHandoffRefused, match="not write-only"):
        HandoffEmitter(record=producer, adjoint_slice=_adjoint_slice(producer),
                       boundary_storage=storage, capture_batch=1,
                       publication=ReadBack())


def test_a_write_only_owner_names_its_lifetime_and_never_reads_back(
        tmp_path, monkeypatch):
    """The owner rules a write-only template brings (PQ #1075).

    Binding names the commit's lifetime, and only a write-only template
    takes one. A write-only owner refuses a write that asks to be read back,
    and one bound to a read-back template refuses produced files, which it
    could never commit. Each refusal comes before any byte is written.
    """
    _src, pb_repo = pb_source(monkeypatch)
    from prismaquant.cost_streaming import (
        StreamedBoundaryArtifacts, normalize_boundary_storage)
    from prismaquant.joint_quantum_handoff import handoff_root

    producer, _consumer, storage = band_campaign(tmp_path)
    publication, _q, _env = producer_owner(tmp_path, pb_repo, producer, storage)
    policy = normalize_boundary_storage({
        **storage, "directory": str(handoff_root(producer["output_space"]["root"]))})
    geometry = {"group_size": PREFETCH, "n_batches": N_BATCHES,
                "max_entry_tensor_bytes": 1 << 12}

    for lifetime in (None, "forever"):
        with StreamedBoundaryArtifacts(policy) as owner:
            owner.bind({"fixture": "lifetime"}, n_probes=N_PROBES, published=True)
            with pytest.raises(ValueError, match="name the commit's lifetime"):
                owner.bind_produced_output(publication, origin_lifetime=lifetime,
                                           **geometry)
    with StreamedBoundaryArtifacts(policy) as owner:
        owner.bind({"fixture": "window"}, n_probes=N_PROBES, published=True)
        with pytest.raises(ValueError, match="funds no window"):
            owner.bind_produced_output(publication, origin_lifetime="consumed",
                                       window_groups=2, **geometry)

    with StreamedBoundaryArtifacts(policy) as owner:
        owner.bind({"fixture": "read-back"}, n_probes=N_PROBES, published=True)
        owner.bind_produced_output(publication, origin_lifetime="consumed",
                                   **geometry)
        with pytest.raises(RuntimeError, match="never read back"):
            owner.write(torch.zeros(4), batch_index=0, boundary_index=3,
                        probe_index=0)
        assert not list(owner.directory.rglob("*.pt*"))
        assert owner.produced_origin_batches() == []

    class ReadBack:
        write_only = False
        template = {}

        @staticmethod
        def contains(path):
            return True

    with StreamedBoundaryArtifacts(policy) as owner:
        owner.bind({"fixture": "files"}, n_probes=N_PROBES, published=True)
        with pytest.raises(ValueError, match="only a write-only"):
            owner.bind_produced_output(ReadBack(), origin_lifetime="consumed",
                                       **geometry)
    with StreamedBoundaryArtifacts(policy) as owner:
        owner.bind({"fixture": "files"}, n_probes=N_PROBES, published=True)
        owner.bind_produced_output(ReadBack(), window_groups=2, **geometry)
        with pytest.raises(RuntimeError, match="needs a write-only"):
            owner.write_produced_files([("a.json", b"{}")], kind="handoff-record",
                                       boundary_index=3)
        assert not (owner.directory / "a.json").exists()


def _publish_consumer(q, pb_repo, tmp_path, key):
    """File the consumer's row, as ``pbrun`` does right after declaring it."""
    q.publish(action_key=key, cas_root=str(tmp_path / "cas"),
              worker_script=str(pb_repo / "tools" / "prismabuild_worker.py"),
              checkout_root=str(tmp_path / "consumer-checkout"),
              resources={"cpu": 1, "mem_gb": 1}, max_attempts=1)


def _consumer_executes(q, key):
    """Claim and finish the consumer's row through the real queue."""
    claimed = q.claim(owner="w-consumer")
    assert claimed is not None and claimed["action_key"] == key
    q.finish(key, status="executed")


def _charged(publication):
    from prismabuild import produced_output as po
    return po._class_sums(po._read_commitments(po._commitments_path(
        publication.queue.root, publication.instance))["batches"])


def test_prismabuild_retires_the_handoff_once_its_consumer_succeeds(
        tmp_path, monkeypatch):
    """Acceptance 2 (PQ #1075): PB #914's tick retires the committed handoff.

    Until a consumer declares the batches (the ``--after`` edge, PB #946),
    a succeeded producer's handoff waits, quietly. Once the consumer that
    declared every batch has executed, one tick deletes every handoff file
    and frees the durable charge. The queue root and the output prefix are
    both under ``tmp_path``: the tick deletes files.
    """
    _src, pb_repo = pb_source(monkeypatch)
    from prismabuild import pool
    from prismabuild import produced_output as po

    producer, consumer, storage = band_campaign(tmp_path)
    publication, q, _env = producer_owner(tmp_path, pb_repo, producer, storage)
    published, plane = emit_handoff(producer, storage, publication)
    handoff = check_consumer_binds(published, plane, consumer, publication)
    refs = check_origin_batches(publication, producer, published, handoff)
    manifest = po.origin_batch_manifest(q.root, refs)
    paths = [Path(row["path"]) for row in manifest["entries"]]
    owner = publication.instance["owner_action_key"]
    q.finish(owner, status="executed")
    assert q.item_path(pool.DONE, owner).exists()
    assert _charged(publication)["payload"] == sum(p.stat().st_size for p in paths)

    # No consumer has declared the handoff: a succeeded producer's batches
    # wait for one, and waiting is not news.
    assert po.origin_retirement_tick(q) == []
    assert all(p.is_file() for p in paths)

    # What ``pbrun`` does for a consumer that declares the batches: one
    # declaration per batch, then its row.
    key = hashlib.sha256(b"band-serial consumer L-1").hexdigest()
    for ref in refs:
        assert po.declare_origin_consumer(q, ref, consumer_action_key=key) == {
            "ok": True, "declared": True}
    _publish_consumer(q, pb_repo, tmp_path, key)
    assert po.origin_retirement_tick(q) == [], (
        "a queued consumer holds every batch, quietly")
    assert all(p.is_file() for p in paths)

    _consumer_executes(q, key)
    events = po.origin_retirement_tick(q)

    assert sorted(event["ref"]["batch_id"] for event in events) == sorted(
        ref["batch_id"] for ref in refs)
    for event in events:
        assert event["event"] == po.ORIGIN_RETIRED_EVENT
        assert event["reason"] == "consumed"
        assert event["consumers"] == [{"action_key": key, "state": "succeeded"}]
        assert event["superseded"] == [] and event["absent"] == []
    assert sorted(path for event in events for path in event["unlinked"]) == (
        sorted(str(p) for p in paths))
    assert not any(p.exists() for p in paths), "every handoff file is gone"
    assert _charged(publication) == {"payload": 0, "checkpoint": 0, "temp": 0}
    assert po.origin_retirement_tick(q) == []


def test_a_failed_producers_handoff_is_swept(tmp_path, monkeypatch):
    """No consumer declared it and its producer attempt failed: an orphan.

    A failed producer is never handed to a consumer, so PrismaBuild's sweep
    deletes its handoff instead of leaving it behind (PB #914).
    """
    _src, pb_repo = pb_source(monkeypatch)
    from prismabuild import produced_output as po

    producer, consumer, storage = band_campaign(tmp_path)
    publication, q, _env = producer_owner(tmp_path, pb_repo, producer, storage,
                                          max_attempts=1)
    published, plane = emit_handoff(producer, storage, publication)
    handoff = check_consumer_binds(published, plane, consumer, publication)
    refs = check_origin_batches(publication, producer, published, handoff)
    paths = [Path(row["path"])
             for row in po.origin_batch_manifest(q.root, refs)["entries"]]
    assert po.origin_retirement_tick(q) == [], "the live attempt keeps it"

    q.finish(publication.instance["owner_action_key"], status="failed")
    events = po.origin_retirement_tick(q)

    assert sorted(event["ref"]["batch_id"] for event in events) == sorted(
        ref["batch_id"] for ref in refs)
    assert {(event["event"], event["reason"]) for event in events} == {
        (po.ORIGIN_RETIRED_EVENT, "orphan")}
    assert not any(p.exists() for p in paths)
    assert _charged(publication) == {"payload": 0, "checkpoint": 0, "temp": 0}


# -- the handoff template's id (PQ #1054) --------------------------------------

def _dispatch_template(producer, storage, root, **kwargs):
    """The dispatcher's handoff template for ``producer``: ``(path, template)``."""
    import dispatch_joint_quanta as dispatch

    path = dispatch.handoff_template_path(
        producer, plan={"execution": {"boundary_storage": storage}},
        adjoint_slice=_adjoint_slice(producer), tier=chain.TIER,
        output_root=root / "out", **kwargs)
    return path, json.loads(path.read_text())


def test_two_scratch_roots_each_file_their_own_handoff_template(
        tmp_path, monkeypatch):
    """The same quantum id under two roots files two templates (PQ #1054).

    PrismaBuild files a template under its id and refuses a different body
    under the same id. The two roots' templates differ in their output
    prefix, so each must carry an id of its own, or every fresh run, arm or
    relaunch that reuses a quantum id is refused at its first producer.
    """
    pb_source(monkeypatch)
    from prismabuild import produced_output as po

    q = chain._queue(tmp_path)
    filed = {}
    for name in ("root-a", "root-b"):
        root = tmp_path / name
        root.mkdir()
        producer, _consumer, storage = band_campaign(root)
        _path, template = _dispatch_template(producer, storage, root)
        filed[name] = (producer["quantum_id"], template,
                       po.declare_template(q.root, template))

    (id_a, a, path_a), (id_b, b, path_b) = filed["root-a"], filed["root-b"]
    assert id_a == id_b
    assert a["output_prefix"] != b["output_prefix"]
    assert a["template_id"] != b["template_id"]
    for template in (a, b):
        assert template["template_id"].startswith(f"pq-stageb-handoff-{id_a}-")
    assert path_a != path_b
    assert json.loads(path_a.read_text()) == a
    assert json.loads(path_b.read_text()) == b


def test_a_redispatch_in_one_root_files_the_same_handoff_template(
        tmp_path, monkeypatch):
    """Dispatching one root's producer again is idempotent (PQ #1054).

    The same id and the same template, so PrismaBuild's second filing is a
    no-op. An explicit ``template_id`` still overrides the derived one.
    """
    pb_source(monkeypatch)
    from prismabuild import produced_output as po

    producer, _consumer, storage = band_campaign(tmp_path)
    first_path, first = _dispatch_template(producer, storage, tmp_path)
    second_path, second = _dispatch_template(producer, storage, tmp_path)
    assert second_path == first_path
    assert second == first

    q = chain._queue(tmp_path)
    filed = po.declare_template(q.root, first)
    raw = filed.read_bytes()
    assert po.declare_template(q.root, second) == filed
    assert filed.read_bytes() == raw

    _path, explicit = _dispatch_template(
        producer, storage, tmp_path, template_id="pq-stageb-handoff-explicit")
    assert explicit["template_id"] == "pq-stageb-handoff-explicit"
    assert {k: v for k, v in explicit.items() if k != "template_id"} == {
        k: v for k, v in first.items() if k != "template_id"}
