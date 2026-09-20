"""Placement-tag regression for the joint-AURA quantum dispatcher (PQ #831).

PrismaBuild matches a row's tags as a **conjunction**: an offer is eligible
only when every wanted tag is among the tags the box offers
(``wanted.issubset(offer.tags)``, ``PoolQueue._matching_offers`` in
``src/prismabuild/pool.py``).  The dispatcher shipped
``--tag sparky --tag sparklina`` while each live Spark offers ``gb10`` plus
its own host name, so the dual-host policy admitted neither box and every
quantum row was unplaceable.  These tests pin the correction against the
*published* PrismaBuild matcher (the sealed mirror the workers run at
``/mnt/shared/prismabuild-fleet/repo``), not a local reimplementation of the
subset rule:

* the hermetic case replays the live offer shapes through the published
  matcher -- the host pair matches nothing, ``gb10`` matches both Sparks and
  excludes a box of another class;
* the live case asks the same question of the real worker offers through the
  public ``PoolQueue.placeable_hosts``, and skips when the fleet mount is not
  visible (the mount-dependent convention of the other campaign tests).

Both halves skip, never pass, when the published client or the live queue
cannot be read: a green run must mean the matcher actually answered.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from dispatch_joint_quanta import CONSUMER_TAGS  # noqa: E402

PUBLISHED_ROOT = Path("/mnt/shared/prismabuild-fleet/repo")
LIVE_QUEUE_ROOT = Path("/mnt/shared/prismabuild-fleet/pb-queue")

#: The shape of the demand the dispatcher seals for a quantum
#: (``--demand gpu=1,mem_gb=104 --cpus 10``).  Tags decide eligibility here;
#: capacity only has to fit so it never masks a tag refusal.
DEMAND = {"gpu": 1, "mem_gb": 104, "cpu": 10}

#: The live fleet's offer shapes read on 2026-09-20 (``workers/*.json``),
#: reduced to what the matcher reads.  The last entry is synthetic: matching
#: capacity and a GPU, an unrelated class tag -- it isolates the tag rule
#: from the capacity rule.
OFFERS = [
    {"host": "sparky", "tags": ["gb10", "progress-v1", "sparky"],
     "has_gpu": True, "capacity": {"cpu": 20, "gpu": 1, "mem_gb": 104}},
    {"host": "sparklina",
     "tags": ["gb10", "gx10-6b77", "progress-v1", "sparklina"],
     "has_gpu": True, "capacity": {"cpu": 20, "gpu": 1, "mem_gb": 104}},
    {"host": "dl380g10", "tags": ["cpu", "progress-v1", "x86"],
     "has_gpu": False, "capacity": {"cpu": 80, "gpu": 0, "mem_gb": 96}},
    {"host": "other-gpu-class", "tags": ["gfx1201", "rocm", "x86"],
     "has_gpu": True, "capacity": {"cpu": 16, "gpu": 1, "mem_gb": 104}},
]


def _published_pool():
    """The published PrismaBuild pool module, or a skip that names why not.

    The fleet's own worker loops import this tree (it is the ``repo`` symlink
    to the active generation); a test that re-implemented the matcher could
    stay green while placement still starved.  A half-visible mount is a
    skip, not a pass: the answer must come from the published bytes."""
    source = PUBLISHED_ROOT / "src" / "prismabuild" / "pool.py"
    if not source.is_file():
        pytest.skip(f"published PrismaBuild not visible at {PUBLISHED_ROOT}")
    published_src = str(PUBLISHED_ROOT / "src")
    if published_src not in sys.path:
        sys.path.insert(0, published_src)
    import prismabuild.pool as pool
    if not Path(pool.__file__).resolve().is_relative_to(
            (PUBLISHED_ROOT / "src").resolve()):
        pytest.skip(f"a different prismabuild is already imported: {pool.__file__}")
    return pool


def _matcher(pool):
    """The published matcher over an explicit offer list, no queue I/O.

    ``_matching_offers`` is the one rule ``placeable``, ``placeable_hosts``
    and the claim path all read; ``demand_of`` is a staticmethod, so a bare
    instance is enough to exercise it."""
    queue = pool.PoolQueue.__new__(pool.PoolQueue)
    return lambda tags: {
        offer["host"] for offer in queue._matching_offers(
            {"tags": list(tags), "resources": dict(DEMAND)}, live=OFFERS)}


def _item(tags):
    return {"tags": list(tags), "resources": dict(DEMAND)}


def test_published_matcher_reads_tags_as_a_conjunction():
    """Ground truth, from PB's own matcher: the shipped host pair matches
    nothing; the shared class tag matches both Sparks and no other box; and
    a class tag plus a host tag narrows to that host."""
    hosts = _matcher(_published_pool())
    assert hosts(("sparky", "sparklina")) == set()
    assert hosts(("gb10",)) == {"sparky", "sparklina"}
    assert hosts(("gb10", "sparky")) == {"sparky"}


def test_dispatcher_tags_are_the_shared_class_tag():
    """The dispatcher's own tags are placeable on both Sparks: one shared
    class tag, never the host pair that admitted neither box."""
    assert CONSUMER_TAGS == ("gb10",)
    hosts = _matcher(_published_pool())
    assert hosts(CONSUMER_TAGS) == {"sparky", "sparklina"}
    assert not {"sparky", "sparklina"} <= set(CONSUMER_TAGS)


def test_dispatcher_tags_are_placeable_on_the_live_gb10_fleet():
    """The acceptance check on the live fleet: the dispatcher's tags reach
    both Sparks and exclude every host of another class, while the old pair
    reaches nothing.  Skips if the mount, the queue, or either Spark offer
    is absent, so it never reports a pass it did not observe."""
    pool = _published_pool()
    if not (LIVE_QUEUE_ROOT / "workers").is_dir():
        pytest.skip(f"live PrismaBuild queue not visible at {LIVE_QUEUE_ROOT}")
    queue = pool.PoolQueue(root=LIVE_QUEUE_ROOT)
    offers = queue.offers()
    if not offers:
        pytest.skip("no live worker offers are visible")
    live_tags = {str(offer.get("host") or "?"):
                 {str(tag) for tag in (offer.get("tags") or [])}
                 for offer in offers}
    missing = {"sparky", "sparklina"} - set(live_tags)
    if missing:
        pytest.skip(f"live fleet does not offer both Sparks: missing {sorted(missing)}")

    placed = queue.placeable_hosts(_item(CONSUMER_TAGS))
    assert placed is not None, "placeable_hosts answered None with offers live"
    assert {"sparky", "sparklina"} <= set(placed)
    for host in placed:
        assert "gb10" in live_tags[host], f"{host} was placed without gb10"
    for host, tags in live_tags.items():
        if "gb10" not in tags:
            assert host not in placed, f"{host} has no gb10 but was placed"

    # The shipped defect, measured on the live fleet: two host names are a
    # conjunction no single box satisfies.
    assert queue.placeable_hosts(_item(("sparky", "sparklina"))) == []
