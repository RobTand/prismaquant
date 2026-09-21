"""One skip for the tests that need a PrismaBuild SDK (PQ #886).

These suites run on the fleet, where the pbtest pin guard proves the
interpreter carries a single non-editable install of the reviewed
``prismabuild`` distribution before pytest starts. The hosted GitHub runner
installs no such distribution, so 65 tests failed and 29 errored there on
every push -- a check that is always red cannot show a regression.

The predicate is deliberately exactly "no installed distribution owns
``prismabuild``", which is the condition the runner is in. It is NOT "the
import failed": an SDK that is installed but shadowed, editable, or at the
wrong commit is a real defect on a box that is supposed to have it, and
``staged_lease.inject_installed_sdk_for_tests`` must keep failing loudly for
it. Nothing here changes that contract; this only decides whether a test
that needs the SDK runs at all.

Whether the SDK is installed is asked at call time through the
``importlib.metadata`` module attribute, so a test can put the runner's
answer in front of it.
"""
from __future__ import annotations

import importlib.metadata as metadata

import pytest

#: The single reason every skip in this lane carries.
SKIP_REASON = "prismabuild SDK absent: fleet-only, runs under pbtest"


def prismabuild_sdk_installed() -> bool:
    """True when an installed distribution owns the ``prismabuild`` import."""

    return bool(metadata.packages_distributions().get("prismabuild"))


def require_prismabuild_sdk() -> None:
    """Skip, naming the fleet, where no installed distribution owns it."""

    if not prismabuild_sdk_installed():
        pytest.skip(SKIP_REASON)
