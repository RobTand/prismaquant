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
