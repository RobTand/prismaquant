"""The #2705 gate must run against installed PB, not disappear with a mount.

The code being repaired is itself a test/probe. Delegate to that existing
end-to-end gate so its fixture and real demand/placement assertions remain
owned in one place, while keeping this no-skip regression independent of the
file the implementation needs to update.
"""
from pathlib import Path

import pytest

import test_dispatch_joint_quanta as gate
from test_dispatch_joint_quanta import _portable_spec, campaign  # noqa: F401


@pytest.mark.parametrize("legacy_tree", ["as_deployed", "absent"])
def test_stage_b_accounting_gate_executes_without_legacy_pb_api(
        tmp_path, campaign, monkeypatch, legacy_tree):
    # Isolate any current PB host-admission state; no fleet submission or
    # execution is needed for this local queue-placement assertion.
    monkeypatch.setenv("PRISMABUILD_BOX_STATE_ROOT", str(tmp_path / "box-state"))
    if legacy_tree == "absent":
        def without_legacy_tree(value, *parts):
            if str(value) == "/mnt/shared/prismabuild-fleet/repo":
                return tmp_path / "unpublished-pb-tree"
            return Path(value, *parts)

        monkeypatch.setattr(gate, "Path", without_legacy_tree)

    try:
        gate.test_a_stage_b_row_charges_spill_cotangent_and_spool_to_the_box(
            tmp_path, campaign)
    except pytest.skip.Exception as exc:
        pytest.fail(
            "The Stage B accounting gate must exercise the installed PrismaBuild "
            f"API even without the historical published source mount: {exc}")
