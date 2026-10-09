"""PQ #1888/#1293/#2152: one reviewed SDK5 boundary, without legacy acceptance."""
from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

from fleet_sdk import require_prismabuild_sdk
from prismaquant import staged_lease

COMMIT = "027103d9a8417e06c7f13356e58779a313cd7088"

#: The historical SDK4 connected-fixture archive. It stays on the shared mount
#: as preserved history; the SDK5 consumer refuses it by version.
SDK4_ARCHIVE_COMMIT = "dc4803daaf09b6426083d2d36bd2a2da3d6832fe"
SDK4_ARCHIVE_ROOT = ("/mnt/shared/prismabuild-fleet/qualification/"
                     "pq-pb-sdk4-20261002/" + SDK4_ARCHIVE_COMMIT)


def test_reviewed_reader_and_sdk_move_together():
    assert staged_lease.PB_READER_LEASE_PIN_COMMIT == COMMIT
    assert staged_lease.PINNED_SDK_COMMIT == COMMIT
    assert staged_lease.PB_CLIENT_SDK_VERSION == 5


@pytest.mark.parametrize("version", [None, 1, 2, 3, 4])
def test_other_sdk_versions_still_refuse(version):
    names = {name: object() for name in staged_lease._REQUIRED_NAMES}
    module = SimpleNamespace(SDK_VERSION=version, **names)
    with pytest.raises(staged_lease.LeaseRefused, match="lease-helper-unsupported"):
        staged_lease._require_client_surface(module)


def test_sdk_missing_reader_surface_still_refuses():
    module = SimpleNamespace(SDK_VERSION=staged_lease.PB_CLIENT_SDK_VERSION)
    with pytest.raises(staged_lease.LeaseRefused, match="lease-helper-unsupported"):
        staged_lease._require_client_surface(module)


def test_preserved_sdk4_archive_refuses_under_the_sdk5_consumer(monkeypatch):
    """The real historical SDK4 tree is refused by version, never accepted.

    One SDK moves with the pin: the consumer demands exactly SDK5, so the
    intact SDK4 archive — still present on the shared mount as preserved
    history — must refuse by name rather than serve as a fallback. Where the
    archive was never mounted the refusal cannot be exercised and the test
    says so instead of inventing a tree.
    """
    require_prismabuild_sdk()
    if not Path(SDK4_ARCHIVE_ROOT, "src", "prismabuild", "client.py").is_file():
        pytest.skip(f"preserved SDK4 archive not mounted: {SDK4_ARCHIVE_ROOT}")
    monkeypatch.setattr(staged_lease, "_HELPER_ROOT", SDK4_ARCHIVE_ROOT)
    monkeypatch.delenv(staged_lease.HELPER_ROOT_ENV_VAR, raising=False)
    with pytest.raises(staged_lease.LeaseRefused, match="lease-helper-unsupported"):
        staged_lease.client_sdk()


@pytest.mark.usefixtures("pinned_pb_source")
def test_real_sdk5_sealed_tree_is_the_production_resolver():
    """The shared connected-fixture bundle is served by the production resolver.

    The bundle lives on the fleet's shared mount, so this test is fleet-only
    like every other PB SDK test: the hosted runner, which has no prismabuild
    distribution and no /mnt/shared, skips it through the one predicate
    tests/fleet_sdk.py owns (PQ #886, #1941). On a box with the SDK installed
    it runs, and a missing tree fails it loudly.
    """
    require_prismabuild_sdk()
    from fullstack_pb_generation import require_paths

    root = Path(require_paths()["root"])
    staged_lease.set_lease_helper_root(root)
    try:
        sdk = staged_lease.client_sdk()
        assert sdk.SDK_VERSION == staged_lease.PB_CLIENT_SDK_VERSION == 5
        assert isinstance(sdk.__file__, str)
        assert Path(sdk.__file__).resolve().is_relative_to(root / "src")
        assert "require_native_producer_context" in inspect.signature(
            sdk.read_verified_action_result).parameters
        assert callable(sdk.bind_standard_capture_command)
        assert sdk.READER_LEASE_TAG == "reader-lease-v1"
        assert callable(sdk.acquire_for) and callable(sdk.open_pinned)
    finally:
        staged_lease.set_lease_helper_root(None)


@pytest.mark.usefixtures("pinned_pb_source")
def test_band_planner_and_reader_share_the_reviewed_sdk5_root(monkeypatch):
    """The planner's PB modules and the reader's SDK come from one bundle."""
    require_prismabuild_sdk()
    from fullstack_pb_generation import require_paths
    from test_band_serial_dispatch import _pin_published_helper_root
    from test_quantum_executable_readset import _pb

    monkeypatch.setenv(staged_lease.HELPER_ROOT_ENV_VAR, "/unreviewed/active-sdk1")
    _pin_published_helper_root(monkeypatch)
    root = Path(require_paths()["root"])
    core, tiers, plans = _pb()
    sdk = staged_lease.client_sdk()
    assert sdk.SDK_VERSION == staged_lease.PB_CLIENT_SDK_VERSION
    for module in (core, tiers, plans, sdk):
        assert isinstance(module.__file__, str)
        assert Path(module.__file__).resolve().is_relative_to(root / "src")
