"""The fleet-only SDK guard decides the same way on both boxes (PQ #886).

FAILING-BEFORE: with the runner's answer in front of it, the strict
reader's ``_pb()`` raised ``RuntimeError`` and the GitHub job stayed red for
a dependency it was never given. These tests put that answer in front of the
guard explicitly, so they run and pass on the fleet as well as on a runner
-- neither is asserting on its own host.
"""
from __future__ import annotations

import importlib.metadata as metadata

import pytest

from fleet_sdk import (
    SKIP_REASON, prismabuild_sdk_installed, require_prismabuild_sdk)


def test_the_reason_names_the_fleet_and_the_tool_that_runs_these():
    assert SKIP_REASON == (
        "prismabuild SDK absent: fleet-only, runs under pbtest")


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
