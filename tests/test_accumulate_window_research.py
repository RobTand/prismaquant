"""CPU checks for the PQ #1406 research arms (no GPU, no model)."""
import pytest
import torch
import hashlib
import importlib.util
import inspect
import json
import math
import pickle
import sys
import types
from pathlib import Path

# Frozen verbatim from d08fd5958ee6a323df41035b0fda763e7bb7e669, #1406.
# This test-only baseline makes missing restoration produce behavioral failures,
# not an import/collection failure. Tests never read git history or fleet data.
# Once restored, ALWAYS exercise the actual instrument, including import errors.
_INSTRUMENT_SPEC = importlib.util.find_spec("experiments.accumulate_window_research")
if _INSTRUMENT_SPEC is not None:
    from experiments import accumulate_window_research as research
else:
    research = types.ModuleType("historical_accumulate_window_research")
    _fixture = Path(__file__).parent / "fixture-data/accumulate_window_research_1406.py.txt"
    exec(compile(_fixture.read_text(), str(_fixture), "exec"), research.__dict__)

from prismaquant import joint_aura
from test_joint_aura_projection import _linear, _spec


def _lease():
    weight = torch.randn(8, 6)
    layer = _linear(weight)
    specs = {"q": _spec("q", lambda x: torch.round(x * 2) / 2),
             "identity": _spec("identity", lambda x: x, act_bits=None)}
    lease = joint_aura.JointOperatorStatisticsLease(
        {"u": layer}, {"u": specs},
        max_statistics_bytes=1 << 20, max_candidate_bytes=1 << 20)
    return lease


def _draws(seed=1406, steps=5):
    generator = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        rows = 1 + int(torch.randint(1, 7, (1,), generator=generator).item())
        x = torch.randn(rows, 6, generator=generator)
        g = torch.randn(rows, 8, generator=generator) * 1e-3
        yield x, x.float(), g.float()


def _run_original_drift_free():
    ref = _lease()
    for x, x2, g2 in _draws():
        ref._observe_rows("u", x, x2, g2, calls=1)
    return ref


def test_fused_arm_matches_checkout_within_rounding():
    ref = _run_original_drift_free()
    fused = _lease()
    for x, x2, g2 in _draws():
        research.fused_observe_rows(fused, "u", x, x2, g2, calls=1)
    assert set(fused._operators) == set(ref._operators)
    for key in ref._operators:
        drift = research.matrix_drift(ref._operators[key], fused._operators[key])
        assert drift["max_rel_to_max"] < 1e-4
    assert fused.telemetry["operator_gemms"] == ref.telemetry["operator_gemms"]
    assert fused.telemetry["qdq_calls"] == ref.telemetry["qdq_calls"]
    assert fused._observed_tokens == ref._observed_tokens
    assert fused._observed_calls == ref._observed_calls


def test_buffered_arm_matches_checkout_within_rounding():
    ref = _run_original_drift_free()
    for size in (2, 4):
        buffered = _lease()
        buf = research.RowBuffer(size, 1 << 20)
        research.CURRENT_BUFFER = buf
        try:
            pending = 0
            for x, x2, g2 in _draws():
                research.buffered_observe_rows(buffered, "u", x, x2, g2, calls=1)
                pending += 1
                if pending % size == 0:
                    buf.flush(joint_aura.JointOperatorStatisticsLease._observe_rows)
            buf.flush(joint_aura.JointOperatorStatisticsLease._observe_rows)
        finally:
            research.CURRENT_BUFFER = None
        assert set(buffered._operators) == set(ref._operators)
        for key in ref._operators:
            drift = research.matrix_drift(ref._operators[key],
                                          buffered._operators[key])
            assert drift["max_rel_to_max"] < 1e-4
        flushes = -(-5 // size)
        assert buffered.telemetry["operator_gemms"] == 2 * flushes
        assert buffered.telemetry["qdq_calls"] == flushes
        assert buffered._observed_tokens == ref._observed_tokens
        assert buffered._observed_calls == ref._observed_calls
        assert buf.peak_bytes > 0


def test_row_buffer_cap_refuses():
    lease = _lease()
    buf = research.RowBuffer(128, 64)
    x = torch.randn(4, 6)
    with pytest.raises(RuntimeError, match="cap"):
        for _ in range(10):
            buf.append("u", lease, x, x.float(), torch.randn(4, 8), calls=1)


def test_row_buffer_flush_empty_is_noop():
    buf = research.RowBuffer(8, 1 << 20)
    buf.flush(lambda *a, **k: (_ for _ in ()).throw(AssertionError("no calls")))
    assert buf.peak_bytes == 0


def test_matrix_drift_reports_identity():
    value = torch.randn(6, 4)
    drift = research.matrix_drift(value, value.clone())
    assert drift == {"max_abs": 0.0, "max_rel_to_max": 0.0,
                     "mismatched_fraction": 0.0}


def test_kernel_family_buckets_gemm_kinds():
    assert research.kernel_family("void cutlass::Kernel2<cutlass_80_simt_sgemm_foo>") == "simt_gemm"
    assert research.kernel_family("void at::native::vectorized_elementwise_kernel<4, foo>") == "elementwise"
    assert research.kernel_family("void cutlass::Kernel2<cutlass_80_tensorop_bf16_bar>") == "tensor_gemm"
    assert research.kernel_family("ampere_sgemm_128x128_nn") == "tensor_gemm"


def test_priced_dloss_is_half_squared_total():
    assert research.priced_dloss(2.0) == 2.0
    assert research.priced_dloss(0.0) == 0.0


def test_field_drift_counts_identical_terms():
    terms = {"a": {"weight": 1.0, "activation": 2.0, "mixed": 3.0, "total": 6.0}}
    drift = research.field_drift(terms, {k: dict(v) for k, v in terms.items()})
    assert drift["total"] == {"max_abs": 0.0, "max_rel": 0.0, "median_rel": 0.0,
                              "identical": 1, "count": 1}


# #2896 acceptance. The published payload covers more than the selected window;
# the expected keys must come independently from that window's admitted specs.
def _reference_fixture(tmp_path, count=710):
    keys = {(f"mtp.unit.{i}", "R1024") for i in range(count)}
    costs = {name: {fmt: {"signed_components_per_probe": [{"total": 0.0}],
                         "signed_per_probe": [0.0] * 4}}
             for name, fmt in keys}
    # A genuine payload term outside the selected window. It must not be
    # substituted for a selected key, even if all 710 values would match.
    costs["other.window"] = {"R1024": {
        "signed_components_per_probe": [{"total": 0.0}],
        "signed_per_probe": [0.0] * 4}}
    path = tmp_path / "joint-cost.pkl"
    path.write_bytes(pickle.dumps({"costs": costs}))
    return path, keys, {key: {"total": 0.0} for key in keys}


def _compare(path, terms, expected_keys):
    # Allow the frozen legacy helper to demonstrate its actual false positive.
    # Restored code must accept independent expected_keys (see separate test).
    if "expected_keys" in inspect.signature(research._against_payload).parameters:
        return research._against_payload(path, terms, expected_keys=expected_keys)
    return research._against_payload(path, terms)


def test_instrument_is_restored_and_comparison_accepts_independent_expected_keys():
    assert _INSTRUMENT_SPEC is not None, "restore the #1406 research instrument"
    assert "expected_keys" in inspect.signature(research._against_payload).parameters


def test_exact_selected_710_probe_zero_totals_are_accepted(tmp_path):
    path, keys, terms = _reference_fixture(tmp_path)
    result = _compare(path, terms, keys)
    assert result["compared"] == result["bitwise_equal"] == 710
    assert result["max_rel"] == 0.0


@pytest.mark.parametrize("mutation", ["missing", "extra", "same_count_wrong_window"])
def test_reference_refuses_wrong_selected_window_key_set(tmp_path, mutation):
    path, keys, terms = _reference_fixture(tmp_path)
    if mutation != "extra":
        terms.pop(sorted(keys)[0])
    if mutation != "missing":
        terms[("other.window", "R1024")] = {"total": 0.0}
    with pytest.raises((ValueError, RuntimeError), match="(?i)(key|term|window|710|coverage)"):
        _compare(path, terms, keys)


@pytest.mark.parametrize("count", [0, 709, 711])
def test_reference_refuses_non_710_even_when_keys_and_values_match(tmp_path, count):
    path, keys, terms = _reference_fixture(tmp_path, count)
    with pytest.raises((ValueError, RuntimeError), match="(?i)(710|count|term|window)"):
        _compare(path, terms, keys)


@pytest.mark.parametrize("changed", [-0.0, math.nextafter(0.0, 1.0)])
def test_reference_counts_unequal_float_bits_not_numeric_equality(tmp_path, changed):
    path, keys, terms = _reference_fixture(tmp_path)
    terms[sorted(keys)[0]]["total"] = changed
    result = _compare(path, terms, keys)
    assert result["compared"] == 710
    assert result["bitwise_equal"] == 709


def test_launcher_consumes_exact_final_prepared_plan_without_relocation(tmp_path, monkeypatch):
    """Only dispatch is stubbed: no model, process, GPU, or historical cache."""
    from prismaquant import tessera_joint_aura as aura

    out = tmp_path / "fresh-research"
    out.mkdir()
    config = {"output_root": str(out), "execution": {
        "projection_backend": {"name": "torch"}}}
    plan = tmp_path / "final-plan.json"
    plan.write_text(json.dumps(config))
    prepared = tmp_path / "prepared.json"
    prepared.write_text(json.dumps({"plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest()}))
    reference = tmp_path / "reference.pkl"
    reference.write_bytes(pickle.dumps({"costs": {}}))
    before = plan.read_bytes()
    dispatched = []
    monkeypatch.setattr(sys, "argv", ["accumulate_window_research", "--plan", str(plan),
        "--prepared", str(prepared), "--prepared-sha256",
        hashlib.sha256(prepared.read_bytes()).hexdigest(),
        "--reference-cost", str(reference), "--out", str(out), "--arms", "A"])
    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    monkeypatch.setattr(research, "install", lambda args: None)
    monkeypatch.setattr(research.subprocess, "Popen",
                        lambda *args, **kwargs: types.SimpleNamespace(terminate=lambda: None))
    monkeypatch.setattr(aura, "main", lambda: dispatched.append(list(sys.argv)))
    research.main()
    assert len(dispatched) == 1
    argv = dispatched[0]
    assert argv[argv.index("--plan") + 1] == str(plan)
    assert argv[argv.index("--plan-sha256") + 1] == hashlib.sha256(before).hexdigest()
    assert argv[argv.index("--prepared") + 1] == str(prepared)
    assert argv[argv.index("--prepared-sha256") + 1] == hashlib.sha256(prepared.read_bytes()).hexdigest()
    assert plan.read_bytes() == before
    assert not (out / "plan-research.json").exists()
    assert not (out / "prepare/source-identity.json").exists()


@pytest.mark.parametrize("field", ["plan_sha256", "projection_backend", "source_model_identity"])
def test_certified_prepared_bindings_refuse_stale_values(monkeypatch, field):
    """Regression guard on the shared owner, not a new permissive local gate."""
    from prismaquant import tessera_joint_aura as aura

    monkeypatch.setenv("PRISMAQUANT_DEV_MODE", "0")
    expected = {"plan_sha256": "a" * 64,
                "projection_backend": {"name": "torch"},
                "source_model_identity": {"content_sha256": "b" * 64}}[field]
    stale = {"plan_sha256": "c" * 64,
             "projection_backend": {"name": "fused_fp32_v1"},
             "source_model_identity": {"content_sha256": "d" * 64}}[field]
    with pytest.raises(ValueError, match="(?i)(prepared|identity|mismatch)"):
        aura.require_prepared_binding(field, stale, expected)
