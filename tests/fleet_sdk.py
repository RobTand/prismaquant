"""One skip for the tests that need a PrismaBuild SDK (PQ #886).

The reviewed ``prismabuild`` distribution is a test dependency: the ``pbsdk``
dependency group in ``pyproject.toml`` installs it as a non-editable Git
install at the retained Bazel dependency source pin (PQ #3216). The hosted
gate, the post-merge suite and the fleet's pbtest shards all provide it; on
the fleet the pbtest pin guard also proves it before pytest starts. Before
PQ #3216 no hosted job installed it, and these tests skipped there.

The predicate is deliberately exactly "no installed distribution owns
``prismabuild``", which is the condition an environment without the group is
in. It is NOT "the import failed": an SDK that is installed but shadowed, editable, or at the
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

``prismabuild_imports_restored``, which puts back where ``import prismabuild``
resolves (PQ #1281), lives in ``tests/prismabuild_imports.py``: ``tests/conftest.py``
wraps every test in it, and it imports nothing from ``prismaquant``.

The ``installed_client_sdk`` and ``installed_helper_sdk`` fixtures are here
rather than in ``tests/conftest.py`` (#3215): a test that requests one imports
it from this module, so only those tests depend on ``staged_lease``.
"""
from __future__ import annotations

import contextlib
import importlib.metadata as metadata
from pathlib import Path

import pytest

#: The single reason every skip in this lane carries.
SKIP_REASON = "prismabuild SDK absent: install the pbsdk dependency group"

#: The reason a lease-helper skip carries; the refusal's own reason (e.g.
#: "lease-helper-unavailable") and the env var name are appended at call time
#: so a developer knows exactly what to set to run the lane locally.
LEASE_HELPER_SKIP_REASON = "prismabuild lease helper absent: fleet-only, runs under pbtest"


#: The reason a test that needs PB's sealed source bundle skips where the
#: bundle is not mounted. The bundle is a git archive of PrismaBuild at the
#: reviewed commit (``tests/pb_runtime_generation_pin.json``); unlike the
#: installed SDK it carries PB's ``tools/`` (``tools/fleet/*.py``,
#: ``tools/prismabuild_worker.py``), which no ``prismabuild`` distribution
#: installs. Only the fleet's shared mount provides it.
PB_SOURCE_BUNDLE_SKIP_REASON = (
    "PrismaBuild source bundle absent: fleet-only, runs under pbtest")

#: The fleet's shared mount, which the bundle and shared fixtures live on.
SHARED_MOUNT = Path("/mnt/shared")


def require_pb_source_bundle(root: Path) -> None:
    """Skip, naming the bundle, where its root is not mounted (PQ #3216).

    Absence is the only skip. A root that exists with the wrong bytes is a
    defect on a box that is supposed to have it; callers keep failing on
    their digest checks.
    """

    if not Path(root).is_dir():
        pytest.skip(f"{PB_SOURCE_BUNDLE_SKIP_REASON} ({root})")


def require_shared_mount() -> None:
    """Skip, naming the mount, where the fleet's ``/mnt/shared`` is absent."""

    if not SHARED_MOUNT.is_dir():
        pytest.skip(f"{SHARED_MOUNT} absent: fleet-only, runs under pbtest")


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

    from prismaquant.staged_lease import LeaseRefused, sdk_submodule

    try:
        sdk_submodule("reader_lease")
    except LeaseRefused as exc:
        if exc.kind != "availability":
            raise
        pytest.skip(f"{LEASE_HELPER_SKIP_REASON} (set PRISMABUILD_READER_HELPER_ROOT: {exc.reason})")


@contextlib.contextmanager
def installed_client_bound(monkeypatch):
    """Bind the reviewed installed ``prismabuild.client`` for one test.

    PQ modules reach PrismaBuild only through ``staged_lease.client_sdk``
    (PB #1254), which resolves the generation PB injected before anything
    else. Inside a pbtest shard that is the fleet's live generation, not the
    reviewed pin this tree is tested against, so the injected root is
    removed first and the installed distribution is bound explicitly. In an
    environment with no installed distribution the test skips, naming the
    dependency group.
    """

    require_prismabuild_sdk()
    from prismaquant import staged_lease

    monkeypatch.delenv('PRISMABUILD_READER_HELPER_ROOT', raising=False)
    staged_lease.set_lease_helper_root(None)
    module = staged_lease.inject_installed_sdk_for_tests()
    try:
        yield module
    finally:
        staged_lease.clear_injected_sdk_for_tests()


@pytest.fixture
def installed_client_sdk(monkeypatch):
    """The reviewed installed ``prismabuild.client``, bound for this test.

    Not autouse: a test about the unbound state must be able to see it.
    See ``fleet_sdk.installed_client_bound``.
    """

    with installed_client_bound(monkeypatch) as module:
        yield module


@pytest.fixture
def installed_helper_sdk(tmp_path, monkeypatch):
    """Production resolver over the actual, Git/RECORD-qualified SDK sources.

    The published pbtest guard owns install verification, once per shard and
    before pytest; this fixture re-reads only the install identity, with the
    standard library, because the guard's module exists in the shard's main
    process and not in its xdist workers. This private helper view links those
    installed bytes, not an SDK4 archive or a copied package; it is a CPU
    fixture, never a published runtime generation. No test-only SDK injection
    supplies the production Gateway.
    """
    from prismaquant import staged_lease

    require_prismabuild_sdk()
    dist = staged_lease.reviewed_installed_distribution()
    package = Path(dist.locate_file("prismabuild")).resolve()
    helper = tmp_path / "installed-sdk-helper"
    (helper / "src").mkdir(parents=True)
    (helper / "src" / "prismabuild").symlink_to(package, target_is_directory=True)
    assert staged_lease._INJECTED is None
    monkeypatch.setattr(staged_lease, "_HELPER_ROOT", str(helper))
    sdk = staged_lease.client_sdk()
    assert Path(sdk.__file__).resolve() == package / "client.py"
    yield sdk
