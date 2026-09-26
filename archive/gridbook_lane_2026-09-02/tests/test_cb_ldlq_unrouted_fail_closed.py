"""Never-routed experts under an LDLQ context: explicit fail-closed opt-in.

Declared never-routed experts have no calibration activations by
construction, and the export-time holdout gate fail-closes such cells to
the raw render (raw_uncertifiable_too_few_rows). The cost stage's
weight-only row emission must therefore be able to render them RAW under
an LDLQ context via the EXPLICIT `ldlq_missing_activation_ok` opt-in —
while the default path keeps raising, so a broken activation loader can
never silently produce an all-raw cost table stamped as LDLQ.

Regression for the 2026-08-09 burn crash:
  ValueError: NVFP4_CB_K12: LDLQ requires calibration activation rows
  (measure_quant_cost._emit_weight_only_rows -> cb_fields_for_context)
"""
import pytest

torch = pytest.importorskip("torch")

from prismaquant import format_registry as fr
from prismaquant.nvfp4_cb_footprint import (
    CBSerializationContext,
    cb_fields_for_context,
)


def _inputs(seed=7):
    g = torch.Generator().manual_seed(seed)
    weight = torch.randn(8, 256, generator=g) * 0.25
    col_weights = torch.rand(256, generator=g) + 0.05
    return weight, col_weights


@pytest.mark.parametrize("fmt", ["NVFP4_CB_K12", "FP8_CB_K28"])
def test_default_still_raises_without_activation_rows(fmt):
    weight, col_weights = _inputs()
    ctx = CBSerializationContext.production(encode_tier="balanced", ldlq=True)
    with pytest.raises(ValueError, match="calibration activation rows"):
        cb_fields_for_context(
            fr.get_format(fmt), weight, context=ctx,
            col_weights=col_weights, activation_rows=None,
        )


@pytest.mark.parametrize("fmt", ["NVFP4_CB_K12", "FP8_CB_K28"])
def test_opt_in_returns_the_exact_raw_render_and_populates_sidecar(fmt):
    weight, col_weights = _inputs()
    spec = fr.get_format(fmt)
    ldlq_ctx = CBSerializationContext.production(
        encode_tier="balanced", ldlq=True)
    raw_out: dict = {}
    fields = cb_fields_for_context(
        spec, weight, context=ldlq_ctx, col_weights=col_weights,
        activation_rows=None, raw_fields_out=raw_out,
        ldlq_missing_activation_ok=True,
    )
    # Identity: byte-equal to the plain no-LDLQ render (fail-closed = raw).
    raw_ctx = CBSerializationContext.production(
        encode_tier="balanced", ldlq=False)
    reference = cb_fields_for_context(
        spec, weight, context=raw_ctx, col_weights=col_weights,
    )
    assert torch.equal(fields["indices"], reference["indices"])
    assert torch.equal(fields["scales"], reference["scales"])
    # Sidecar populated so the raw-table extractor's completeness holds.
    assert raw_out.get("ldlq_applied") is True
    assert raw_out["fields"] is fields
