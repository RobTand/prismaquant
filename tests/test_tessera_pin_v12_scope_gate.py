"""Pin-update gate on v12 per-launch scope proof (PQ #2600).

A pin update to ``prismaquant/tessera_runtime/tessera_serving_runtime_pin.json``
is refused unless the four named v12 scope fixture classes -- census, derived,
absent and malformed -- all pass against the reader. The checks reuse the
fixture in ``tests/test_tessera_lane_v12.py``, not a new copy.

A producer schema change is refused unless its PR links consumer
compatibility work and names fixture results. That rule lives in
``prismaquant/tessera_pin_scope_gate.py`` beside the pin gate.
"""
import pytest

from prismaquant import tessera_pin_scope_gate as gate


def test_pin_gate_passes_when_all_four_scope_classes_pass():
    results = gate.run_v12_scope_proof()
    assert set(results) == set(gate.REQUIRED_SCOPE_CLASSES)
    assert all(results.values())
    gate.require_v12_scope_proof(results)


@pytest.mark.parametrize("missing", list(gate.REQUIRED_SCOPE_CLASSES))
def test_pin_gate_refuses_a_missing_scope_class(missing):
    results = {k: True for k in gate.REQUIRED_SCOPE_CLASSES if k != missing}
    with pytest.raises(gate.TesseraPinScopeGateError, match=missing):
        gate.require_v12_scope_proof(results)


@pytest.mark.parametrize("failing", list(gate.REQUIRED_SCOPE_CLASSES))
def test_pin_gate_refuses_a_failing_scope_class(failing):
    results = {k: (k != failing) for k in gate.REQUIRED_SCOPE_CLASSES}
    with pytest.raises(gate.TesseraPinScopeGateError, match=failing):
        gate.require_v12_scope_proof(results)


def test_pin_gate_refuses_an_empty_proof():
    with pytest.raises(gate.TesseraPinScopeGateError):
        gate.require_v12_scope_proof({})


def test_producer_schema_pr_refuses_without_linked_consumer_work():
    with pytest.raises(gate.ProducerSchemaPRError, match="consumer"):
        gate.check_producer_schema_pr(
            body="fixtures: census pass, derived pass, absent pass, malformed pass",
            linked=[],
        )


def test_producer_schema_pr_refuses_without_fixture_results():
    with pytest.raises(gate.ProducerSchemaPRError, match="[Ff]ixture"):
        gate.check_producer_schema_pr(
            body="see the linked consumer issue",
            linked=["prismaquant#2600"],
        )


def test_producer_schema_pr_refuses_when_one_scope_class_is_unnamed():
    with pytest.raises(gate.ProducerSchemaPRError, match="derived"):
        gate.check_producer_schema_pr(
            body="fixtures: census pass, absent pass, malformed pass",
            linked=["prismaquant#2600"],
        )


def test_producer_schema_pr_admits_with_linked_work_and_all_fixture_results():
    gate.check_producer_schema_pr(
        body="fixtures: census pass, derived pass, absent pass, malformed pass",
        linked=["prismaquant#2600"],
    )
