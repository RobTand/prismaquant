"""Stage B retains exact scratch intent and refuses unqualified native admission.

The shared positive regression owns the independently known ceilings and the
before-launch refusal. Its fixture never invokes an executor or reads PB state.
"""
import test_dispatch_joint_quanta as gate
from test_dispatch_joint_quanta import _portable_spec, campaign  # noqa: F401


def test_stage_b_accounting_gate_retains_requirements_and_refuses_native_launch(tmp_path, campaign):
    gate.test_a_stage_b_row_charges_spill_cotangent_and_spool_to_the_box(tmp_path, campaign)
