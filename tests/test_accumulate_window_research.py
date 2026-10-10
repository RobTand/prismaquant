"""CPU checks for the PQ #1406 research arms (no GPU, no model)."""
import pytest
import torch
from experiments import accumulate_window_research as research
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
                    buf.flush(buffered._observe_rows)
            buf.flush(buffered._observe_rows)
        finally:
            research.CURRENT_BUFFER = None
        assert set(buffered._operators) == set(ref._operators)
        for key in ref._operators:
            drift = research.matrix_drift(ref._operators[key],
                                          buffered._operators[key])
            assert drift["max_rel_to_max"] < 1e-4
        assert buffered.telemetry["operator_gemms"] == ref.telemetry["operator_gemms"]
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
