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

``require_prismabuild_sdk`` is the right guard for a test that only needs
*some* ``prismabuild`` import to succeed. It is the WRONG guard for a test
that runs through ``prismaquant.staged_lease``'s reader-lease SDK: that SDK
resolves from a sealed generation tree (the fleet-injected
``PRISMABUILD_READER_HELPER_ROOT``, or an explicit test override), never
from package presence alone (``staged_lease._sdk``). A box with
``prismabuild`` pip-installed but no such tree -- an editable dev checkout,
this container's own venv -- passes ``require_prismabuild_sdk`` and then
hits ``LeaseRefused: lease-helper-unavailable`` mid-test: a FAILURE where the
missing fleet prerequisite should have been a named SKIP (PQ #1097).
``require_lease_helper`` below is that guard: it probes the SDK the lane
actually calls, not a proxy for it.
"""
from __future__ import annotations

import importlib.metadata as metadata

import pytest

#: The single reason every skip in this lane carries.
SKIP_REASON = "prismabuild SDK absent: fleet-only, runs under pbtest"

#: The reason a lease-helper skip carries; the refusal's own reason (e.g.
#: "lease-helper-unavailable") and the env var name are appended at call time
#: so a developer knows exactly what to set to run the lane locally.
LEASE_HELPER_SKIP_REASON = "prismabuild lease helper absent: fleet-only, runs under pbtest"


def prismabuild_sdk_installed() -> bool:
    """True when an installed distribution owns the ``prismabuild`` import."""

    return bool(metadata.packages_distributions().get("prismabuild"))


def require_prismabuild_sdk() -> None:
    """Skip, naming the fleet, where no installed distribution owns it."""

    if not prismabuild_sdk_installed():
        pytest.skip(SKIP_REASON)


def require_lease_helper() -> None:
    """Skip, naming what's missing, when the PB reader-lease SDK cannot
    actually resolve for this process (PQ #1097).

    Probes ``staged_lease.sdk_submodule`` -- the same public entry point the
    retirement lane calls -- rather than checking package presence: a
    distribution can be installed with no sealed generation tree behind it,
    which ``require_prismabuild_sdk`` cannot see. Only an *availability*
    refusal (``LeaseRefused.kind == "availability"``: nothing is wired up)
    is a skip; an *integrity* one (a divergent or unsupported SDK -- the
    tree resolves to something, but not honestly) is a real defect on a box
    that is supposed to have it and is re-raised, same as
    ``prismabuild_sdk_installed``'s own contract for a shadowed/editable/
    wrong-commit install.
    """

    from prismaquant.staged_lease import HELPER_ROOT_ENV_VAR, LeaseRefused, sdk_submodule

    try:
        sdk_submodule("reader_lease")
    except LeaseRefused as exc:
        if exc.kind != "availability":
            raise
        pytest.skip(f"{LEASE_HELPER_SKIP_REASON} (set {HELPER_ROOT_ENV_VAR}: {exc.reason})")
