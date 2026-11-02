"""The SDK guard decides the same way on every box (PQ #886, PQ #3216).

FAILING-BEFORE: with the runner's answer in front of it, the strict
reader's ``_pb()`` raised ``RuntimeError`` and the GitHub job stayed red for
a dependency it was never given. These tests put that answer in front of the
guard explicitly, so they run and pass on the fleet as well as on a runner
-- neither is asserting on its own host.
"""
from __future__ import annotations

import importlib.metadata as metadata

import pytest

import fleet_sdk
from fleet_sdk import (
    LEASE_HELPER_SKIP_REASON, PB_SOURCE_BUNDLE_SKIP_REASON, SKIP_REASON,
    prismabuild_sdk_installed, require_lease_helper, require_pb_source_bundle,
    require_prismabuild_sdk, require_shared_mount)


def test_the_reason_names_the_dependency_group_that_provides_the_sdk():
    """The SDK is a test dependency now, not a fleet-only one (PQ #3216)."""
    assert SKIP_REASON == (
        "prismabuild SDK absent: install the pbsdk dependency group")


def test_no_installed_distribution_skips_with_that_reason(monkeypatch):
    monkeypatch.setattr(metadata, "packages_distributions", lambda: {})
    assert prismabuild_sdk_installed() is False
    with pytest.raises(pytest.skip.Exception) as caught:
        require_prismabuild_sdk()
    assert str(caught.value) == SKIP_REASON


def test_an_installed_distribution_runs_the_test(monkeypatch):
    monkeypatch.setattr(metadata, "packages_distributions",
                        lambda: {"prismabuild": ["prismabuild"]})
    assert prismabuild_sdk_installed() is True
    require_prismabuild_sdk()


def test_the_strict_reader_entry_point_skips_instead_of_erroring(monkeypatch):
    """The exact seam the runner died on, with the runner's answer.

    ``inject_installed_sdk_for_tests`` keeps its fail-loud contract: it is
    never reached here, because a box with no installed distribution has
    nothing for it to prove. A box that HAS one and fails the proof still
    raises, which is what the pbtest pin guard relies on.
    """

    from test_strict_reader_tier_enforcement import _pb

    monkeypatch.setattr(metadata, "packages_distributions", lambda: {})
    with pytest.raises(pytest.skip.Exception) as caught:
        _pb()
    assert str(caught.value) == SKIP_REASON


# -- require_lease_helper (PQ #1097) ----------------------------------------
#
# require_prismabuild_sdk proves only package presence, which an editable
# dev checkout satisfies with no fleet-injected helper root behind it. These
# put staged_lease's own answer -- whether its SDK actually resolves -- in
# front of require_lease_helper, the same way the tests above do for
# require_prismabuild_sdk against importlib.metadata.

def test_the_lease_helper_reason_names_the_fleet_and_the_tool_that_runs_these():
    assert LEASE_HELPER_SKIP_REASON == (
        "prismabuild lease helper absent: fleet-only, runs under pbtest")


def test_no_lease_helper_root_skips_naming_the_env_var(monkeypatch):
    from prismaquant import staged_lease

    def refuses(name):
        raise staged_lease.LeaseRefused("lease-helper-unavailable", kind="availability")

    monkeypatch.setattr(staged_lease, "sdk_submodule", refuses)
    with pytest.raises(pytest.skip.Exception) as caught:
        require_lease_helper()
    assert str(caught.value) == (
        f"{LEASE_HELPER_SKIP_REASON} (set {staged_lease.HELPER_ROOT_ENV_VAR}: "
        "lease-helper-unavailable)")


def test_a_resolvable_lease_helper_runs_the_test(monkeypatch):
    from prismaquant import staged_lease

    monkeypatch.setattr(staged_lease, "sdk_submodule", lambda name: object())
    require_lease_helper()  # does not skip, does not raise


def test_a_divergent_lease_helper_fails_loudly_instead_of_skipping(monkeypatch):
    """An SDK that resolves to something, but not honestly, is a real defect
    on a box that is supposed to have it (PQ #1097): it must not be silently
    swallowed as though the fleet prerequisite were merely absent -- the
    same fail-loud contract ``inject_installed_sdk_for_tests`` already keeps
    for a shadowed/editable/wrong-commit install."""
    from prismaquant import staged_lease

    def divergent(name):
        raise staged_lease.LeaseRefused("lease-helper-divergent: shadowed", kind="integrity")

    monkeypatch.setattr(staged_lease, "sdk_submodule", divergent)
    with pytest.raises(staged_lease.LeaseRefused, match="lease-helper-divergent"):
        require_lease_helper()


# -- the PB source bundle and the shared mount (PQ #3216) --------------------
#
# With the SDK installed on hosted runners, tests that also need PB's sealed
# source bundle (PB tools/, which no distribution installs) or the fleet's
# shared mount used to fail on an assertion. Absence is a skip naming what is
# missing; a present bundle with wrong bytes still fails at its digest check.

def test_an_absent_source_bundle_skips_naming_it(tmp_path):
    missing = tmp_path / "no-bundle"
    with pytest.raises(pytest.skip.Exception) as caught:
        require_pb_source_bundle(missing)
    assert str(caught.value) == f"{PB_SOURCE_BUNDLE_SKIP_REASON} ({missing})"


def test_a_present_source_bundle_runs_the_test(tmp_path):
    require_pb_source_bundle(tmp_path)  # does not skip


def test_an_absent_shared_mount_skips_naming_it(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet_sdk, "SHARED_MOUNT", tmp_path / "shared")
    with pytest.raises(pytest.skip.Exception) as caught:
        require_shared_mount()
    assert str(caught.value) == f"{tmp_path / 'shared'} absent: fleet-only, runs under pbtest"


def test_a_present_shared_mount_runs_the_test(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet_sdk, "SHARED_MOUNT", tmp_path)
    require_shared_mount()  # does not skip


def test_a_wrong_byte_bundle_still_fails_its_digest_check(tmp_path, monkeypatch):
    """Only absence skips: a mounted bundle whose bytes differ from the pin
    is a defect on a box that is supposed to have it."""
    import json
    import fullstack_pb_generation as published

    pin = json.loads(published.PIN_PATH.read_text())
    root = tmp_path / pin["runtime_generation"]
    root.mkdir()
    monkeypatch.setenv(fleet_sdk.PB_SOURCE_BUNDLE_ENV, str(root))
    pin["bundle_root"] = str(root)
    pin["files"] = {"src/prismabuild/core.py": "0" * 64}
    (root / "src" / "prismabuild").mkdir(parents=True)
    (root / "src" / "prismabuild" / "core.py").write_text("")
    path = tmp_path / "pin.json"
    path.write_text(json.dumps(pin))
    monkeypatch.setattr(published, "PIN_PATH", path)
    monkeypatch.setattr(published, "_ROOT", None)
    monkeypatch.setattr(metadata, "packages_distributions",
                        lambda: {"prismabuild": ["prismabuild"]})
    with pytest.raises(AssertionError, match="src/prismabuild/core.py"):
        published._resolve_once()
