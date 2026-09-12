"""The adapter against the real decomposer, with the children actually running.

Every claim this harness makes is end to end.  A phase plan is emitted, handed
to PrismaBuild's own decomposer, cut into children, and each child is executed
in process by ``PoolQueue.serve_once`` -- a real subprocess, a real sealed
command, a real CAS receipt.  Then the group is closed, and what is asserted is
the cover: that between them the children answered the roster exactly once.

WHERE THE CODE UNDER TEST COMES FROM
====================================
PrismaBuild's decomposition is on the unmerged draft pull request PB #518 and is
**not deployed to the fleet** -- see the module comment on
``prismaquant.quality_prefill_pb_adapter``.  So this harness runs against a
*checkout* of that branch, never against the live fleet: nothing here submits
anything to ``/mnt/shared/prismabuild-fleet``, and the queue, the CAS and the
checkouts all live under ``tmp_path``.  Point ``QP_PB517_ROOT`` at the worktree;
without it the module skips, because the alternative -- a green suite that
silently tested nothing -- is worse than a visible skip.

The producers below are test producers.  A real one measures something; these
answer instantly, which is what makes running every child of a real cut cheap
enough to do in one test.  One of them deliberately does **not** use
``write_child_result_manifest``: the point of that one is that a producer which
bypasses PrismaQuant's own fail-closed writer is still caught, by the exact
cover, at the merge.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import pytest

from prismaquant import quality_prefill_pb_adapter as adapter

from test_quality_prefill_pb_adapter import (  # noqa: E402
    EXPECTED_CHILDREN, ROSTER_SIZE, phase_plan,
)


PB517_ROOT = Path(os.environ.get(
    "QP_PB517_ROOT", "/home/rob/tmp/pb517-preexecution-decomposer"))
PQ_ROOT = Path(__file__).resolve().parents[1]

_missing = not (PB517_ROOT / "src" / "prismabuild" / "decomposition.py").is_file()
pytestmark = pytest.mark.skipif(
    _missing,
    reason=(
        f"PrismaBuild pre-execution decomposition (PB #517 / draft PR #518) is "
        f"not checked out at {PB517_ROOT}; it is not deployed to the fleet "
        f"either, so this harness has nothing to run against. Set QP_PB517_ROOT "
        f"to a checkout of the #518 branch."
    ),
)

#: Wall clock just before the first import out of the #518 checkout, so a test
#: can tell a bytecode file this process wrote from one that was already there.
_IMPORT_EPOCH = time.time()

if not _missing:
    # The #518 checkout is another agent's worktree and this harness is a
    # reader of it.  Importing a module normally writes a ``__pycache__``
    # alongside its source, which would be this process modifying that
    # worktree; refuse to, before the first import from it.
    sys.dont_write_bytecode = True

    for entry in (PB517_ROOT / "src", PB517_ROOT / "tools" / "fleet",
                  PB517_ROOT / "tests"):
        if str(entry) in sys.path:
            sys.path.remove(str(entry))
        sys.path.insert(0, str(entry))
    from prismabuild import core as pb            # noqa: E402
    from prismabuild import decomposition as dc   # noqa: E402
    from prismabuild import pool                  # noqa: E402
    import pbcampaign                             # noqa: E402
    import pbrun                                  # noqa: E402

    # The deployed runtime also ships ``prismabuild`` and ``pbcampaign``, and an
    # earlier import in the same process would shadow these with a generation
    # that has no decomposer at all.  Saying so here beats a mystifying
    # AttributeError three tests later.
    for module in (pb, dc, pool, pbcampaign, pbrun):
        assert str(PB517_ROOT) in str(Path(module.__file__).resolve()), (
            f"{module.__name__} resolved to {module.__file__}, not to the "
            f"PB #518 checkout at {PB517_ROOT}"
        )


# --------------------------------------------------------------------------
# The producers a child runs
# --------------------------------------------------------------------------

#: Answers exactly its own batch, through PrismaQuant's fail-closed writer.  The
#: contract in one screen: read the envelope the reserved placeholder pointed
#: at, produce one result per task, publish the manifest the sealed action
#: already declared as this child's result.
HONEST_PRODUCER = '''
import json, sys
from prismaquant import quality_prefill_pb_adapter as adapter

envelope = adapter.read_batch_envelope(sys.argv[1])
# One argument, whatever the resolved path contains.  A shell-expanded
# placeholder would have split a path with a space in it into several.
assert len(sys.argv) == 2, sys.argv
results = {
    task["id"]: {
        "context_id": task["payload"]["context_id"],
        "currency": task["payload"]["currency"],
        "rate": task["payload"]["rate"],
        "measurement_status": "measured",
        "value": float(task["payload"]["rate"]) / 1000.0,
    }
    for task in envelope["tasks"]
}
adapter.write_child_result_manifest(envelope, results)
'''

#: Answers another child's batch.  It writes the manifest by hand on purpose:
#: ``write_child_result_manifest`` refuses this, and what is under test here is
#: what happens when a producer does not use it.  The shift is one whole batch,
#: so every child reports tasks that belong to a sibling.
WRONG_BATCH_PRODUCER = '''
import hashlib, json, sys
from prismaquant import quality_prefill_pb_adapter as adapter

envelope = adapter.read_batch_envelope(sys.argv[1])
BATCH = %(batch)d
TOTAL = %(total)d

def shifted(task_id):
    prefix, _, index = task_id.rpartition("/t")
    return "%%s/t%%04d" %% (prefix, (int(index) + BATCH) %% TOTAL)

manifest = {
    "schema": adapter.CHILD_RESULT_MANIFEST_SCHEMA,
    "parent_key": envelope["parent_key"],
    "plan_key": envelope["plan_key"],
    "child_ordinal": envelope["child_ordinal"],
    "results": [
        {
            "task_id": shifted(task["id"]),
            "output_id": task["output_id"],
            "value_sha256": hashlib.sha256(task["id"].encode()).hexdigest(),
        }
        for task in envelope["tasks"]
    ],
}
open(envelope["result_manifest_path"], "w").write(json.dumps(manifest))
'''

#: Answers its batch with a record of its own ``sys.argv``.  A child's private
#: checkout does not outlive the action, so the only evidence that survives is
#: what goes into the declared result -- and since a result reaches the manifest
#: as a digest, the test recomputes the digest of the record a correctly
#: substituted child must have produced.  The assertion here fails the action
#: outright; the digest comparison is what proves the path was the right one.
ARGV_PRODUCER = '''
import sys
from prismaquant import quality_prefill_pb_adapter as adapter

assert len(sys.argv) == 2, sys.argv
envelope = adapter.read_batch_envelope(sys.argv[1])
record = {"argv_len": len(sys.argv), "batch_path": sys.argv[1]}
adapter.write_child_result_manifest(
    envelope, {task["id"]: record for task in envelope["tasks"]})
'''


# --------------------------------------------------------------------------
# The fleet, entirely inside tmp_path
# --------------------------------------------------------------------------

def _checkout(root: Path, producer: str) -> Path:
    """A sealed source tree carrying one producer module.

    ``pbrun`` snapshots the action's working tree, so the producer has to be a
    committed file in it.  It is a module rather than a script path because the
    child runs ``interpreter -m name`` in its own private checkout, which is
    what the adapter's command shape declares.
    """

    work = root / "work"
    work.mkdir(parents=True)
    (work / "qp_child_producer.py").write_text(producer, encoding="utf-8")
    for args in (
        ("init", "-q"),
        ("config", "user.email", "test@example.invalid"),
        ("config", "user.name", "PrismaQuant test"),
        ("add", "qp_child_producer.py"),
        ("commit", "-qm", "sealed producer"),
    ):
        done = subprocess.run(["git", "-C", str(work), *args],
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stderr
    return work


def _queue(root: Path) -> "pool.PoolQueue":
    queue = pool.PoolQueue(root / "pb-queue")
    queue.announce(host="sparky", tags=["sparky", "gb10"], has_gpu=True,
                   capacity={"cpu": 4, "mem_gb": 16, "gpu": 1})
    return queue


@pytest.fixture()
def fleet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A whole PrismaBuild under ``tmp_path``: shared root, queue and CAS.

    The shared root carries a space, a single quote and a ``$`` deliberately.
    Every CAS blob path -- including the batch envelope PrismaBuild substitutes
    into ``{pb.task_batch}`` -- is derived from it, so every test in this module
    is also a test that the reserved placeholder is substituted rather than
    expanded (acceptance criterion 5).
    """

    shared = tmp_path / "p b's $hared"
    shared.mkdir()
    monkeypatch.setattr(pbrun, "SH", shared)
    monkeypatch.setattr(socket, "gethostname", lambda: "sparky")
    return shared, _queue(shared)


def _plan(work: Path, **overrides) -> dict:
    plan = phase_plan(
        cwd=str(work),
        interpreter=sys.executable,
        module="qp_child_producer",
        # ``prismaquant`` is imported by the producer and lives outside the
        # sealed fixture tree, so the sealed environment has to carry it.  A
        # real phase's cwd is the PrismaQuant checkout and needs none of this.
        env={"PYTHONPATH": str(PQ_ROOT)},
        **overrides,
    )
    return plan


def _decompose(plan: dict):
    request = dc.validate_logical_request(adapter.emit_logical_request(plan))
    return pbcampaign.decompose(request, transport="pool", priority=0)


def _serve(queue, *, limit: int | None = None) -> int:
    served = 0
    while limit is None or served < limit:
        one = queue.serve_once(
            tags=["sparky"], python=sys.executable, timeout_s=300.0,
            capacity={"cpu": 4, "mem_gb": 16, "gpu": 1},
        )
        if one is None:
            break
        assert one["status"] == "executed", one
        served += 1
    return served


def _group_path(shared: Path, group: dict) -> Path:
    parent = group["plan"]["parent_key"]
    return (shared / "cas" / pbcampaign.DECOMPOSITIONS / parent[:2] / parent
            / "group.json")


# --------------------------------------------------------------------------
# The cut
# --------------------------------------------------------------------------

def test_the_harness_only_reads_the_other_agents_worktree() -> None:
    """Importing must not leave bytecode behind in the #518 checkout.

    The checkout is a shared branch belonging to another agent; this harness is
    allowed to read it and nothing else.  An import writes a ``__pycache__``
    next to the source unless bytecode writing is off, so assert it is off and
    that every module borrowed from that tree came in without one.
    """
    assert sys.dont_write_bytecode is True

    for module in (pb, dc, pool, pbcampaign, pbrun):
        cached = getattr(module, "__cached__", None)
        if cached is None or str(PB517_ROOT) not in cached:
            continue
        if not Path(cached).exists():
            continue
        assert Path(cached).stat().st_mtime < _IMPORT_EPOCH, (
            f"importing {module.__name__} wrote {cached} into another agent's "
            f"worktree"
        )


def test_the_emitted_request_is_what_the_decomposer_reads(fleet) -> None:
    """Acceptance criterion 2, without executing anything.

    PrismaBuild's validator is the authority on the wire format, so the useful
    assertion is not that the emitted bytes look right but that PB accepts them
    unchanged -- and then cuts the known roster into the number of children the
    declared limits admit.
    """

    shared, _ = fleet
    plan = _plan(_checkout(shared, HONEST_PRODUCER))
    emitted = adapter.emit_logical_request(plan)

    validated = dc.validate_logical_request(emitted)
    assert adapter.document_bytes(validated) == adapter.document_bytes(emitted), (
        "PrismaBuild canonicalized the request, so the emitter and the "
        "validator disagree about some field's normal form"
    )

    partitions = dc.partition_roster(validated["roster"], validated["batch_policy"])
    assert len(partitions) == EXPECTED_CHILDREN
    assert [len(batch) for batch in partitions] == [22, 22, 22]
    assert sum(len(batch) for batch in partitions) == ROSTER_SIZE


def test_the_same_plan_cuts_to_the_same_parent_and_the_same_children(
    fleet,
) -> None:
    """A frozen plan is one campaign, not one campaign per submission.

    Determinism inside PrismaQuant is only half of it: the parent key is a hash
    of the emitted bytes plus what ``pbrun`` seals, so the test that matters is
    that a second decomposition of the same plan lands on the same parent and
    the same child action keys.
    """

    shared, _ = fleet
    plan = _plan(_checkout(shared, HONEST_PRODUCER))
    _, first = _decompose(plan)
    _, second = _decompose(plan)
    assert second["plan"]["parent_key"] == first["plan"]["parent_key"]
    assert second["plan"]["plan_key"] == first["plan"]["plan_key"]
    assert [child["action_key"] for child in second["children"]] == [
        child["action_key"] for child in first["children"]
    ]


# --------------------------------------------------------------------------
# The cover closes
# --------------------------------------------------------------------------

def test_every_child_runs_and_the_group_closes_on_an_exact_cover(
    fleet, capsys,
) -> None:
    """Acceptance criterion 3.

    Each child reads the envelope it was handed, writes the manifest its action
    declared, and the group receipt appears only because the manifests between
    them answer all 66 roster tasks once each.
    """

    shared, queue = fleet
    plan = _plan(_checkout(shared, HONEST_PRODUCER))
    records, group = _decompose(plan)
    assert len(group["children"]) == EXPECTED_CHILDREN
    assert {record["status"] for record in records} == {"submitted"}

    assert _serve(queue) == EXPECTED_CHILDREN

    cas = pb.PrismaBuildCAS(shared / "cas")
    assert pbcampaign.close_group(group, cas=cas) == 0

    receipt = json.loads(_group_path(shared, group).read_text())
    assert receipt["schema"] == dc.GROUP_RECEIPT_SCHEMA_V1
    assert receipt["task_count"] == ROSTER_SIZE
    assert receipt["child_count"] == EXPECTED_CHILDREN
    assert receipt["parent_key"] == group["plan"]["parent_key"]
    assert "group receipt" in capsys.readouterr().err


def test_each_childs_manifest_is_the_one_prismaquant_wrote(fleet) -> None:
    """The declared result really is the adapter's document, not a log.

    Read back the way the merge reads it -- through ``cas.result_path``, which
    re-verifies the bytes against the digest the receipt fixed -- and then
    through PrismaQuant's own reader, so both halves of the producer contract
    are shown to accept the same bytes.
    """

    shared, queue = fleet
    plan = _plan(_checkout(shared, HONEST_PRODUCER))
    _, group = _decompose(plan)
    _serve(queue)

    cas = pb.PrismaBuildCAS(shared / "cas")
    answered: set[str] = set()
    for ordinal, child in enumerate(group["children"]):
        raw = pbcampaign.child_result_manifest(child, cas=cas)
        assert raw is not None
        ours = adapter.validate_child_result_manifest(raw)
        theirs = dc.validate_child_result_manifest(raw)
        assert ours == theirs, "the two readers disagree about one document"
        assert ours["child_ordinal"] == ordinal
        assert ours["parent_key"] == group["plan"]["parent_key"]
        answered.update(entry["task_id"] for entry in ours["results"])
    assert answered == {item["task_id"] for item in plan["work_items"]}


def test_the_batch_path_reaches_the_child_as_one_argument(fleet) -> None:
    """Acceptance criterion 5.

    The shared root this fixture builds carries a space, a single quote and a
    ``$``, so the CAS path PrismaBuild substitutes into ``{pb.task_batch}``
    carries all three.  What each child received is checked twice over: the
    child asserts its own ``sys.argv`` has exactly two elements and fails the
    action if not, and the test recomputes the digest of the record a child that
    got exactly the right single argument would have published.  Shell expansion
    would have split the path into several arguments; interpolation into a
    larger argument would have changed it.  Either way the digest moves.
    """

    shared, queue = fleet
    plan = _plan(_checkout(shared, ARGV_PRODUCER))
    assert " " in str(shared) and "'" in str(shared) and "$" in str(shared)

    _, group = _decompose(plan)
    assert _serve(queue) == EXPECTED_CHILDREN

    cas = pb.PrismaBuildCAS(shared / "cas")
    assert pbcampaign.close_group(group, cas=cas) == 0

    request, frozen = group["request"], group["plan"]
    for ordinal, child in enumerate(group["children"]):
        envelope = dc.batch_envelope(request, frozen, child_ordinal=ordinal)
        # The blob is addressed by the digest of the file's bytes, so this is
        # the path PrismaBuild resolved into the placeholder -- derived here
        # rather than read back, which is what makes the comparison a proof.
        batch_path = str(cas.blob_path(adapter.document_file_sha256(envelope)))
        assert " " in batch_path and "'" in batch_path and "$" in batch_path
        expected = adapter.canonical_sha256(
            {"argv_len": 2, "batch_path": batch_path})

        manifest = adapter.validate_child_result_manifest(
            pbcampaign.child_result_manifest(child, cas=cas))
        assert {entry["value_sha256"] for entry in manifest["results"]} == {expected}


# --------------------------------------------------------------------------
# The cover does not close
# --------------------------------------------------------------------------

def test_a_child_that_never_ran_is_never_a_group_success(fleet, capsys) -> None:
    """Acceptance criterion 4, the missing-manifest half.

    Every child that ran passed.  The campaign still fails, because a roster
    with no answer for a third of it is not an answered roster.
    """

    shared, queue = fleet
    plan = _plan(_checkout(shared, HONEST_PRODUCER))
    _, group = _decompose(plan)
    assert _serve(queue, limit=EXPECTED_CHILDREN - 1) == EXPECTED_CHILDREN - 1

    cas = pb.PrismaBuildCAS(shared / "cas")
    assert pbcampaign.close_group(group, cas=cas) == 1
    assert "no group receipt" in capsys.readouterr().err
    assert not _group_path(shared, group).exists()


def test_a_child_answering_another_childs_batch_fails_the_cover(
    fleet, capsys,
) -> None:
    """Acceptance criterion 4, the wrong-batch half.

    Every child exits zero and every manifest is well formed; each one just
    answers for the batch next door.  Nothing below the cover can see that --
    each child's own receipt is perfectly good -- which is why the group
    receipt is not "did they all pass".
    """

    shared, queue = fleet
    producer = WRONG_BATCH_PRODUCER % {"batch": 22, "total": ROSTER_SIZE}
    plan = _plan(_checkout(shared, producer))
    _, group = _decompose(plan)
    assert _serve(queue) == EXPECTED_CHILDREN

    cas = pb.PrismaBuildCAS(shared / "cas")
    assert pbcampaign.close_group(group, cas=cas) == 1
    assert "not in its own batch" in capsys.readouterr().err
    assert not _group_path(shared, group).exists()


# --------------------------------------------------------------------------
# The exit status an operator actually sees
# --------------------------------------------------------------------------

def _request_file(shared: Path, plan: dict) -> str:
    # Beside the shared root, never inside it: PrismaBuild sweeps and counts
    # what lives under its own root, and a request file is the operator's.
    path = shared.parent / "request.json"
    adapter.write_logical_request(plan, path)
    return str(path)


def test_the_campaign_exit_status_follows_the_cover(fleet, capsys) -> None:
    """Acceptance criteria 3 and 4 at ``pbcampaign``'s own boundary.

    The file submitted is the one :func:`write_logical_request` produced, so
    this is the whole path an operator walks: emit, detach, serve, re-run for
    the verdict.  Zero when the roster is answered; non-zero when it is not,
    even though the table above the verdict is all cache hits.
    """

    shared, queue = fleet
    manifest = _request_file(shared, _plan(_checkout(shared, HONEST_PRODUCER)))

    assert pbcampaign.main(["--transport", "pool", "--detach", manifest]) == 0
    detached = [line for line in capsys.readouterr().out.splitlines() if line]
    assert len(detached) == EXPECTED_CHILDREN
    assert _serve(queue) == EXPECTED_CHILDREN
    assert pbcampaign.main(
        ["--transport", "pool", "--wait-s", "60", manifest]) == 0
    assert "group receipt" in capsys.readouterr().err


def test_a_campaign_whose_rows_all_passed_still_fails_on_an_open_cover(
    fleet, capsys,
) -> None:
    shared, queue = fleet
    producer = WRONG_BATCH_PRODUCER % {"batch": 22, "total": ROSTER_SIZE}
    manifest = _request_file(shared, _plan(_checkout(shared, producer)))

    assert pbcampaign.main(["--transport", "pool", "--detach", manifest]) == 0
    capsys.readouterr()
    assert _serve(queue) == EXPECTED_CHILDREN
    assert pbcampaign.main(
        ["--transport", "pool", "--wait-s", "60", manifest]) == 1
    printed = capsys.readouterr()
    assert all("cache_hit" in line for line in printed.out.splitlines()[1:])
    assert "not in its own batch" in printed.err
