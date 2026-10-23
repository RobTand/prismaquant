"""The #2707 gate must exercise the installed PrismaBuild V2 validator.

The stale caller is itself a test. Delegate to it so the implementation can
repair that caller without changing this independent no-skip regression or
duplicating the COST fixture and its producer contract.
"""
import pytest

import test_glm_cost_v2_data_manifest as gate
from test_glm_cost_v2_data_manifest import (  # noqa: F401
    cost_fixture, shared_mount,
)


@pytest.fixture
def scratch(tmp_path):
    """Keep this regression's COST artifacts in the runner's private TMPDIR."""
    return tmp_path


def test_cost_v2_gate_runs_the_installed_validator(cost_fixture, monkeypatch):
    from fleet_sdk import require_prismabuild_sdk

    require_prismabuild_sdk()
    # Do not pre-import PB here: that would pin sys.modules to the installed
    # core and hide the gate's source-tree shadowing bug before it can run.

    try:
        gate.test_cost_v2_manifest_passes_published_prismabuild_validator(
            cost_fixture, monkeypatch)
    except pytest.skip.Exception as exc:
        pytest.fail(
            "The COST V2 release gate must run when the reviewed PrismaBuild "
            f"SDK is installed, without a published source-tree prerequisite: {exc}")
