"""PrismaQuant: mixed-native quantization policy engine for LLMs.

Canonical pipeline:
    1. sensitivity_probe.py         measure per-Linear Fisher sensitivity
    2. measure_quant_cost.py        measure per-Linear × per-format functional cost
    3. allocator.py                 solve additive mixed-format frontier
    4. measure_interactions.py      probe sparse pairwise interactions near the knee
    5. quadratic_refine_allocator.py refine the knee neighborhood under interactions
    6. calibrate_allocator.py       validate predicted frontier points against real KL

The additive allocator stays as the fast global backbone. Interaction and
calibration stages refine the deployment point without turning the whole
optimization into an intractable dense quadratic problem.
"""
from .format_registry import FormatSpec, REGISTRY, register_format


# Transformers compatibility polyfill.
# Some remote modeling files (e.g. MiniMax-M2/M2.7's modeling_minimax_m2.py)
# import `OutputRecorder` from `transformers.utils.generic`. In
# transformers 5.x that symbol moved to `transformers.modeling_utils`.
# Re-expose the 4.x import path so remote code that matches the checkpoint
# tensor naming still loads. Idempotent — no-op if the symbol is already
# there (4.x or future 5.x that re-exports it).
def _polyfill_transformers() -> None:
    try:
        # OutputRecorder: transformers.utils.generic → transformers.modeling_utils
        import transformers.utils.generic as _gen
        if not hasattr(_gen, "OutputRecorder"):
            import transformers.modeling_utils as _mu
            if hasattr(_mu, "OutputRecorder"):
                _gen.OutputRecorder = _mu.OutputRecorder
    except Exception:
        pass
    try:
        # `_init_weights` is wasted work across every prismaquant
        # model-load path: we build a meta skeleton via `from_config`
        # then immediately overwrite every parameter from the
        # checkpoint via `_materialize` / `_fast_install`. Running
        # `_init_weights` in between costs bounded real time on small
        # models and is a compatibility landmine on remote modeling
        # files — transformers 5.x's `_init_weights` now expects
        # rotary modules to expose `compute_default_rope_parameters`,
        # which older remote modeling files (MiniMax M2/M2.7) don't
        # provide. No-op it globally at import time.
        import transformers.modeling_utils as _mu
        if hasattr(_mu, "PreTrainedModel") and \
                not getattr(_mu.PreTrainedModel, "_prismaquant_init_noop", False):
            _mu.PreTrainedModel._initialize_weights = (
                lambda self, *a, **kw: None)
            _mu.PreTrainedModel._prismaquant_init_noop = True
    except Exception:
        pass
    try:
        # Transformers 5.x's fine-grained-FP8 quantizer's
        # `validate_environment` has an operator-precedence bug —
        # the `"disk" in device_map.values()` check is not gated on
        # `pre_quantized`, so it rejects disk-device-map loads even
        # for pre-quantized checkpoints. Override with a bypass:
        # prismaquant's streaming loader (for probe/cost) REQUIRES
        # disk-device-map to offload body layers, and our caller
        # guarantees the checkpoint is pre-quantized (it's the
        # source on disk) so this rejection is a false positive.
        from transformers.quantizers.quantizer_finegrained_fp8 import (
            FineGrainedFP8HfQuantizer as _FP8Q,
        )
        if not getattr(_FP8Q, "_prismaquant_validator_bypass", False):
            def _bypass_validate(self, *a, **kw):
                return None
            _FP8Q.validate_environment = _bypass_validate
            _FP8Q._prismaquant_validator_bypass = True
    except Exception:
        pass
    try:
        # ROPE_INIT_FUNCTIONS['default'] was removed in transformers 5.x
        # (renamed to 'linear', which takes a 'factor' kwarg the old
        # default never needed). Remote modeling files from older
        # checkpoints still look up 'default'. Re-register the old
        # implementation verbatim — a ~6-line function computing the
        # standard rotary inv_freq schedule with no scaling.
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        if "default" not in ROPE_INIT_FUNCTIONS:
            import torch as _torch
            def _compute_default_rope_parameters(config=None, device=None, **_):
                base = config.rope_theta
                partial = getattr(config, "partial_rotary_factor", 1.0)
                head_dim = getattr(
                    config, "head_dim",
                    config.hidden_size // config.num_attention_heads)
                dim = int(head_dim * partial)
                inv_freq = 1.0 / (base ** (
                    _torch.arange(0, dim, 2, dtype=_torch.int64)
                        .to(dtype=_torch.float32, device=device) / dim))
                return inv_freq, 1.0
            ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
    except Exception:
        pass


_polyfill_transformers()
del _polyfill_transformers
