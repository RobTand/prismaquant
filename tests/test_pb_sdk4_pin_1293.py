"""PQ #1888/#1293: one reviewed SDK4 boundary, without legacy acceptance."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from fleet_sdk import require_prismabuild_sdk
from prismaquant import staged_lease

COMMIT = "dc4803daaf09b6426083d2d36bd2a2da3d6832fe"


def test_reviewed_reader_and_sdk_move_together():
    assert staged_lease.PB_READER_LEASE_PIN_COMMIT == COMMIT
    assert staged_lease.PINNED_SDK_COMMIT == COMMIT
    assert staged_lease.PB_CLIENT_SDK_VERSION == 4


def test_real_sdk4_sealed_tree_is_the_production_resolver(monkeypatch):
    # The sealed tree lives on the fleet's shared mount, so this test is
    # fleet-only like every other PB SDK test: the hosted runner, which has
    # no prismabuild distribution and no /mnt/shared, skips it through the
    # one predicate tests/fleet_sdk.py owns (PQ #886, #1941). On a box with
    # the SDK installed it runs, and a missing tree fails it loudly.
    require_prismabuild_sdk()
    root = "/mnt/shared/prismabuild-fleet/qualification/pq-pb-sdk4-20261002/" + COMMIT
    staged_lease.set_lease_helper_root(root)
    try:
        sdk = staged_lease.client_sdk()
        assert sdk.SDK_VERSION == 4
        assert callable(sdk.read_verified_action_result)
        assert callable(sdk.bind_standard_capture_command)
        from pathlib import Path
        assert isinstance(sdk.__file__, str)
        assert Path(sdk.__file__).resolve().is_relative_to(Path(root) / "src")
        assert sdk.READER_LEASE_TAG == "reader-lease-v1"
        assert callable(sdk.acquire_for) and callable(sdk.open_pinned)
    finally:
        staged_lease.set_lease_helper_root(None)


@pytest.mark.parametrize("version", [None, 1, 2, 3])
def test_other_sdk_versions_still_refuse(version):
    names = {name: object() for name in staged_lease._REQUIRED_NAMES}
    module = SimpleNamespace(SDK_VERSION=version, **names)
    with pytest.raises(staged_lease.LeaseRefused, match="lease-helper-unsupported"):
        staged_lease._require_client_surface(module)


def test_sdk_missing_reader_surface_still_refuses():
    module = SimpleNamespace(SDK_VERSION=4)
    with pytest.raises(staged_lease.LeaseRefused, match="lease-helper-unsupported"):
        staged_lease._require_client_surface(module)


def test_band_planner_and_reader_share_the_reviewed_sdk4_root(monkeypatch):
    from pathlib import Path
    from fullstack_pb_generation import require_paths
    from test_band_serial_dispatch import _pin_published_helper_root
    from test_quantum_executable_readset import _pb

    monkeypatch.setenv(staged_lease.HELPER_ROOT_ENV_VAR, "/unreviewed/active-sdk1")
    _pin_published_helper_root(monkeypatch)
    root = Path(require_paths()["root"])
    core, tiers, plans = _pb()
    sdk = staged_lease.client_sdk()
    assert sdk.SDK_VERSION == 4
    for module in (core, tiers, plans, sdk):
        assert isinstance(module.__file__, str)
        assert Path(module.__file__).resolve().is_relative_to(root / "src")
