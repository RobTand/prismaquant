"""PQ #1888: one reviewed SDK3 boundary, without legacy acceptance."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from prismaquant import staged_lease

COMMIT = "95a59051d48cda82eea7927f31870c6c862d7174"


def test_reviewed_reader_and_sdk_move_together():
    assert staged_lease.PB_READER_LEASE_PIN_COMMIT == COMMIT
    assert staged_lease.PINNED_SDK_COMMIT == COMMIT
    assert staged_lease.PB_CLIENT_SDK_VERSION == 3


def test_real_sdk3_sealed_tree_is_the_production_resolver(monkeypatch):
    root = "/mnt/shared/prismabuild-fleet/qualification/pq-pb-sdk3-20261001/" + COMMIT
    staged_lease.set_lease_helper_root(root)
    try:
        sdk = staged_lease.client_sdk()
        assert sdk.SDK_VERSION == 3
        from pathlib import Path
        assert isinstance(sdk.__file__, str)
        assert Path(sdk.__file__).resolve().is_relative_to(Path(root) / "src")
        assert sdk.READER_LEASE_TAG == "reader-lease-v1"
        assert callable(sdk.acquire_for) and callable(sdk.open_pinned)
    finally:
        staged_lease.set_lease_helper_root(None)


@pytest.mark.parametrize("version", [None, 1, 2, 4])
def test_other_sdk_versions_still_refuse(version):
    names = {name: object() for name in staged_lease._REQUIRED_NAMES}
    module = SimpleNamespace(SDK_VERSION=version, **names)
    with pytest.raises(staged_lease.LeaseRefused, match="lease-helper-unsupported"):
        staged_lease._require_client_surface(module)


def test_sdk3_missing_reader_surface_still_refuses():
    module = SimpleNamespace(SDK_VERSION=3)
    with pytest.raises(staged_lease.LeaseRefused, match="lease-helper-unsupported"):
        staged_lease._require_client_surface(module)


def test_band_planner_and_reader_share_the_reviewed_sdk3_root(monkeypatch):
    from pathlib import Path
    from fullstack_pb_generation import require_paths
    from test_band_serial_dispatch import _pin_published_helper_root
    from test_quantum_executable_readset import _pb

    monkeypatch.setenv(staged_lease.HELPER_ROOT_ENV_VAR, "/unreviewed/active-sdk1")
    _pin_published_helper_root(monkeypatch)
    root = Path(require_paths()["root"])
    core, tiers, plans = _pb()
    sdk = staged_lease.client_sdk()
    assert sdk.SDK_VERSION == 3
    for module in (core, tiers, plans, sdk):
        assert isinstance(module.__file__, str)
        assert Path(module.__file__).resolve().is_relative_to(root / "src")
