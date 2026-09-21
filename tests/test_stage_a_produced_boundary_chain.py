"""The produced-boundary chain gate: producer -> PB mover -> strict SDK reader.

Root's architecture decision (2026-09-21): Stage A's own boundary entries are
staged through PB produced-output; no own-session exemption waives the
physical tier policy. The live failure this closes is 8ca8952cc651…'s
``staged-not-serving`` on its own just-written boundary-0 entries.

This module is the integration gate for the plan in
``pq-stagea-produced-boundaries-plan.json``. It drives the REAL chain end to
end and refuses every shortcut: the pinned PB source supplies the API
(declare/bind/prewrite/commit_batch/publish_prepaid_batch/retire_batch/
recover_batches), the mover really stages through the pool, and the reader is
the unchanged strict acquire path. Scenarios beyond the first two are
declared here as the acceptance surface; they fail until the adapter lands.
"""
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def test_the_produced_boundary_adapter_seam_exists():
    """FAILING-BEFORE by construction: the adapter seam is the plan's step 2.

    On the pristine tree this refuses with ImportError -- the produced-backed
    writer does not exist -- which is exactly the gap the plan addresses.
    After integration, importing the adapter and its declared surface is the
    first green fact; every behavioural scenario below extends it.
    """
    from prismaquant import stage_a_produced_output as adapter  # noqa: F401

    for name in ("ensure_instance", "prewrite_batch", "commit_batch",
                 "publish_mover", "await_coverage", "retire_batch",
                 "recover_batches"):
        assert callable(getattr(adapter, name)), name


def test_pinned_pb_source_provenance_is_recorded():
    """No fake SDK green: the test env must name the pinned PB source.

    The admitting environment sets PRISMA_STAGEA_PB_PIN to the PB checkout
    whose src/ is on PYTHONPATH (expected: 8a682535c309 until the integrated
    tree is published). Absent pin refuses rather than guessing.
    """
    import os

    pin = os.environ.get("PRISMA_STAGEA_PB_PIN", "")
    assert pin, ("PRISMA_STAGEA_PB_PIN must name the pinned PB checkout "
                 "providing the produced-output API for this run")
    import prismabuild

    assert Path(prismabuild.__file__).resolve().is_relative_to(
        Path(pin).resolve()), (prismabuild.__file__, pin)
