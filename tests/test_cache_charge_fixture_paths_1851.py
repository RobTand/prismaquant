"""Pure argv fixtures must not depend on pytest's physical temporary root."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

import test_container_cache_charge_1091 as charge


@pytest.mark.parametrize("overlay", ["/tmp", "/var/tmp"])
@pytest.mark.parametrize(
    "name, parameters",
    [
        ("test_cache_forwarding_and_defaults_do_not_reuse_spill_or_tmp", {}),
        ("test_contained_cache_pins_are_preserved", {}),
        ("test_tmpdir_requires_an_explicit_other_charged_root", {}),
        ("test_tmpdir_cannot_reuse_cache_or_an_uncharged_directory", {"bad_tmp": "compile/row-tmp"}),
        ("test_tmpdir_cannot_reuse_cache_or_an_uncharged_directory", {"bad_tmp": "outside/row-tmp"}),
    ],
)
def test_pure_argv_cases_ignore_host_overlay_tmp(overlay, name, parameters):
    # These cases only construct Docker argv; never create these paths or run Docker.
    # Old fixtures consumed this host path as a container path. Fixed fixtures
    # instead declare a non-overlay container namespace, without relaxing guards.
    case = getattr(charge, name)
    supplied = dict(parameters)
    if "tmp_path" in inspect.signature(case).parameters:
        supplied["tmp_path"] = Path(overlay) / "pq-cache-charge-1851-host-fixture"
    case(**supplied)
