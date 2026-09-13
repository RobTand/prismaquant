"""One flag per gate: a fixed-resource refusal keeps the native attestation.

`admit_native_rows` and `admit_fixed_resources` attest different objects. The
first attests the per-row prices `build_runtime_resources` hands the DP; the
second attests the whole-engine charge the allocator adds once, outside the DP.
Before this split they shared one `producer_admitted` flag inside one `try`, so
the fixed-resource refusal that fires for every table at the producer's current
schema version discarded a native attestation that had passed -- 49 rows,
refusal `None`, the first real one ever produced (PQ #557).

The fixtures here are synthetic gate stand-ins. They establish which flag
governs which consumer, and nothing about any measurement.
"""
from types import SimpleNamespace

import pytest

from prismaquant import measured_runtime_prices as mrp
from prismaquant.measured_runtime_prices import (
    RuntimePriceError, RuntimeResources, admitted_fixed_resources, build_runtime_resources,
)
from prismaquant.runtime_provenance import admit_runtime_provenance

FIXED_REFUSAL = "no qualified recomputable full-engine resource partition: synthetic"


def _resources(**overrides):
    fields = {"prefill_ms": 1.0, "decode_ms": 0.5, "serialized_bytes": 2048,
              "resident_bytes": 1600, "peak_scratch_bytes": 1500,
              "activation_bytes": 500, "kv_bytes": 0}
    fields.update(overrides)
    return RuntimeResources(**fields)


def _table(**overrides):
    binding = mrp.RuntimeBinding(operator_route="synthetic", member_formats={"layer": "FP8"},
                                 member_operator_identity_sha256={"layer": "9" * 64},
                                 member_shapes={"layer": (16, 16)})
    row = SimpleNamespace(unit="layer", fmt="FP8", key=("layer", "FP8"), binding=binding,
                          resources=_resources())
    fields = {"runtime_provenance": {"path": "p", "sha256": "0" * 64},
              "native_rows_admitted": False, "fixed_resources_admitted": False,
              "fixed_resources_refusal": None,
              "fixed_resources": _resources(kv_bytes=2048), "rows": (row,)}
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _patch(monkeypatch, *, native_raises=None, fixed_raises=None):
    import prismaquant.runtime_provenance as rp
    monkeypatch.setattr(rp, "load_runtime_relation", lambda *a, **k: {"relation": True})

    def native(table, relation):
        if native_raises is not None:
            raise native_raises

    def fixed(table, relation):
        if fixed_raises is not None:
            raise fixed_raises

    monkeypatch.setattr(rp, "admit_native_rows", native)
    monkeypatch.setattr(rp, "admit_fixed_resources", fixed)


def test_a_fixed_resource_refusal_returns_and_keeps_the_native_attestation(monkeypatch, tmp_path):
    """The regression: the fixed gate refuses, the native gate still passed."""
    _patch(monkeypatch, fixed_raises=RuntimePriceError(FIXED_REFUSAL))
    table = _table(source_path=str(tmp_path / "table.json"), context=None)
    refusal = admit_runtime_provenance(table)
    assert refusal == FIXED_REFUSAL


def test_a_native_row_refusal_still_fails_the_whole_table(monkeypatch, tmp_path):
    """Priced rows without evidence price nothing, so that one still raises."""
    _patch(monkeypatch, native_raises=RuntimePriceError("native row refused"))
    table = _table(source_path=str(tmp_path / "table.json"), context=None)
    with pytest.raises(RuntimePriceError, match="native row refused"):
        admit_runtime_provenance(table)


def test_both_gates_passing_returns_no_refusal(monkeypatch, tmp_path):
    _patch(monkeypatch)
    assert admit_runtime_provenance(_table(source_path=str(tmp_path / "t.json"), context=None)) is None


def test_row_prices_are_governed_by_the_native_flag_not_the_fixed_one():
    """`build_runtime_resources` reads `row.resources` and never the fixed charge."""
    table = _table(native_rows_admitted=True, fixed_resources_admitted=False,
                   fixed_resources_refusal=FIXED_REFUSAL)
    candidate = SimpleNamespace(fmt="FP8", member_formats={"layer": "FP8"}, memory_bytes=2048)
    priced = build_runtime_resources(table, {"layer": [candidate]},
                                     expected_bindings={("layer", "FP8"): table.rows[0].binding})
    assert priced[("layer", "FP8")] is table.rows[0].resources


def test_unadmitted_rows_still_refuse():
    table = _table(native_rows_admitted=False)
    candidate = SimpleNamespace(fmt="FP8", member_formats={"layer": "FP8"}, memory_bytes=2048)
    with pytest.raises(RuntimePriceError, match="native-row producer admission"):
        build_runtime_resources(table, {"layer": [candidate]},
                                expected_bindings={("layer", "FP8"): table.rows[0].binding})


def test_the_fixed_charge_refuses_by_name_where_it_is_consumed():
    """The refusal is spent at the fixed charge, and says what is owed."""
    table = _table(native_rows_admitted=True, fixed_resources_admitted=False,
                   fixed_resources_refusal=FIXED_REFUSAL)
    with pytest.raises(RuntimePriceError) as caught:
        admitted_fixed_resources(table)
    assert FIXED_REFUSAL in str(caught.value)


def test_an_admitted_fixed_charge_is_returned():
    table = _table(native_rows_admitted=True, fixed_resources_admitted=True)
    assert admitted_fixed_resources(table) is table.fixed_resources


def test_a_v1_table_carries_no_provenance_and_is_returned_unchanged():
    """Recorded, not endorsed: see this module's companion finding in PQ #560."""
    table = _table(runtime_provenance=None)
    assert admitted_fixed_resources(table) is table.fixed_resources
