"""The live-fleet half of the placement-tag regression (PQ #831, #1014).

This test reads the live PrismaBuild queue, which no PB action declares, so
it carries the ``fleet_data`` mark. It lives in its own file because
``pbtest`` decides fleet data per file: a file that marks any test needs a
``--data-manifest``, and the merge queue leaves such files out. Keeping the
mark here lets the hermetic matcher tests in
``test_dispatch_shared_tag_placement.py`` run in the queue (PQ #1954).
"""
from __future__ import annotations

import pytest

from test_dispatch_shared_tag_placement import (  # noqa: E402
    CONSUMER_TAGS,
    LIVE_QUEUE_ROOT,
    _item,
    _published_pool,
)

pytestmark = pytest.mark.own_process


#: Reads the live PrismaBuild queue, which no PB action declares:
#: skipped unless asked for (PQ #1014). The hermetic tests in
#: ``test_dispatch_shared_tag_placement.py`` read only the published PB
#: source tree and stay selected.
@pytest.mark.fleet_data
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
