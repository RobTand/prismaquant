"""The measured assignment must hold the same off-step peak as its frontier."""
from types import SimpleNamespace

import pytest

from prismaquant.serve_constraints import ServeSLOs, evaluate_measured_assignment
from prismaquant.transient_charge_boundary import BOUNDARIES, BOUNDARY_V1


def resources(**changes):
    return SimpleNamespace(**(dict(prefill_ms=0.0, decode_ms=0.0, serialized_bytes=0,
                                   resident_bytes=0, activation_bytes=0,
                                   peak_scratch_bytes=0, kv_bytes=0) | changes))


@pytest.mark.parametrize('peak,expected,feasible', [
    (None, 1800, True), (1700, 1800, True), (4096, 4096, False),
])
def test_evaluator_includes_the_declared_off_step_placement_peak(peak, expected, feasible):
    verdict = evaluate_measured_assignment(
        {'layer': 'FP8'}, option_assignments={('layer', 'FP8'): {'layer': 'FP8'}},
        resources={('layer', 'FP8'): resources(resident_bytes=1000, peak_scratch_bytes=200)},
        fixed_assignment={}, fixed_resources=resources(resident_bytes=500, activation_bytes=100,
                                                       non_step_transient_peak_bytes=peak),
        slos=ServeSLOs(device_budget_bytes=2000), table_identity={'synthetic': True},
        boundary=BOUNDARIES[BOUNDARY_V1])
    assert verdict.predicted['device_memory_bytes'] == expected
    assert verdict.feasible is feasible


def test_evaluator_refuses_overlapping_measured_operator_rows():
    from prismaquant.serve_constraints import ServeConstraintError

    assignment = {'a': 'FP8', 'b': 'FP8'}
    left, right = ('left', 'FP8'), ('right', 'FP8')
    option_assignments = {left: {'a': 'FP8'}, right: {'b': 'FP8'}}
    inputs = dict(
        option_assignments=option_assignments,
        resources={left: resources(prefill_ms=2.0), right: resources(prefill_ms=3.0)},
        fixed_assignment={}, fixed_resources=resources(),
        slos=ServeSLOs(), table_identity={'synthetic': True},
    )
    verdict = evaluate_measured_assignment(assignment, **inputs)
    assert verdict.predicted['operator_sum_prefill_ms'] == 5.0

    option_assignments[right]['a'] = 'FP8'
    with pytest.raises(ServeConstraintError) as exc:
        evaluate_measured_assignment(assignment, **inputs)
    assert str(exc.value) == 'measured runtime operator rows overlap'
