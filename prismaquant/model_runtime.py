"""Explicit numerical model compatibility, initialized by model loaders."""
from __future__ import annotations

import contextlib as _contextlib
from contextvars import ContextVar as _ContextVar
from threading import Lock
from .streaming_initialization import (
    clear_pretrained_initialization_contract,
    record_pretrained_initialization,
)

_checkpoint_initializing = _ContextVar("prismaquant_checkpoint_initializing", default=False)
_genuine_initializing = _ContextVar("prismaquant_genuine_initializing", default=False)
_runtime_lock = Lock()
_runtime_initialized = False

def _ensure_triton_cache_writable() -> None:
    """Keep Triton-backed model code from falling back to CPU.

    Some of the Qwen linear-attention dependencies import FLA, which asks
    Triton for the active CUDA target at import time.  If Triton cannot write
    its cache, FLA catches the exception, decides the platform is CPU-only,
    and later crashes in the CUDA forward path.  Respect an explicit
    TRITON_CACHE_DIR; otherwise redirect only when the default cache is not
    writable by this user.
    """
    import os
    import tempfile
    from pathlib import Path

    def _is_writable_dir(path: Path) -> bool:
        try:
            path.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                prefix=".prismaquant-write-",
                dir=path,
                delete=True,
            ):
                pass
            return True
        except Exception:
            return False

    configured = os.environ.get("TRITON_CACHE_DIR")
    if configured:
        try:
            Path(configured).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return

    default = Path.home() / ".triton" / "cache"
    if _is_writable_dir(default):
        return

    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    fallback = base / "prismaquant" / "triton"
    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["TRITON_CACHE_DIR"] = str(fallback)


def _mark_fully_loaded_modules(model) -> None:
    """Flag every module whose own tensors all came from the checkpoint.

    Transformers 5.6.0's ``_initialize_missing_keys`` still documents that it
    "propagates this flag to modules", but its body does so only under FSDP.
    Everywhere else it runs ``_init_weights`` on every module and relies on
    the initializer calling the guarded ``transformers.initialization``
    functions, which skip a tensor already marked ``_is_hf_initialized``.  An
    initializer that writes a tensor directly (``weight.fill_``,
    ``bias.zero_()`` -- the vendored DeepSeek-V4 router does this to a loaded
    buffer) then overwrites checkpoint values after the load.  Transformers
    applies exactly this module rule to remote code; it is applied here to
    every checkpoint load, before missing-state initialization runs.

    A module with no tensors of its own, or with any tensor the checkpoint did
    not supply (a nonpersistent buffer, a missing key), is left unmarked, so
    its initializer still runs.
    """
    for module in model.modules():
        if getattr(module, "_is_hf_initialized", False):
            continue
        tensors = [*module.parameters(recurse=False),
                   *(b for b in module.buffers(recurse=False) if b is not None)]
        if tensors and all(getattr(t, "_is_hf_initialized", False) for t in tensors):
            module._is_hf_initialized = True


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
        # From-config meta skeletons overwrite checkpoint parameters later,
        # so they retain the existing no-init policy. Ordinary checkpoint
        # loads must still initialize missing state, including nonpersistent
        # buffers that Transformers rematerializes with empty_like.
        import transformers.modeling_utils as _mu
        if hasattr(_mu, "PreTrainedModel") and \
                not getattr(_mu.PreTrainedModel, "_prismaquant_init_noop", False):
            model_class = _mu.PreTrainedModel
            real_initialize = model_class._initialize_weights
            real_missing = model_class._initialize_missing_keys

            def _initialize_for_checkpoint(self, *args, **kwargs):
                if _checkpoint_initializing.get() or _genuine_initializing.get():
                    return real_initialize(self, *args, **kwargs)
                return None

            def _initialize_missing_checkpoint_state(self, *args, **kwargs):
                import transformers
                # An unsuccessful repeated finalization must not retain an
                # earlier completion descriptor.
                clear_pretrained_initialization_contract(self)
                _mark_fully_loaded_modules(self)
                token = _checkpoint_initializing.set(True)
                try:
                    result = real_missing(self, *args, **kwargs)
                finally:
                    _checkpoint_initializing.reset(token)
                record_pretrained_initialization(
                    self, transformers_version=transformers.__version__)
                return result

            model_class._initialize_weights = _initialize_for_checkpoint
            model_class._initialize_missing_keys = _initialize_missing_checkpoint_state
            model_class._prismaquant_init_noop = True
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
        # Old remote model files still request the removed default entry.
        # Keep their inverse-frequency arithmetic unchanged. Current configs
        # store the same parameters in the upstream rope_parameters mapping.
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        if "default" not in ROPE_INIT_FUNCTIONS:
            import torch as _torch
            def _compute_default_rope_parameters(config=None, device=None, **_):
                if hasattr(config, "rope_theta"):
                    base = config.rope_theta
                    partial = getattr(config, "partial_rotary_factor", 1.0)
                    head_dim = getattr(
                        config, "head_dim",
                        config.hidden_size // config.num_attention_heads)
                else:
                    parameters = config.rope_parameters
                    if _.get("layer_type") is not None:
                        parameters = parameters[_["layer_type"]]
                    base = parameters["rope_theta"]
                    partial = parameters.get("partial_rotary_factor", 1.0)
                    head_dim = (getattr(config, "head_dim", None)
                                or config.hidden_size // config.num_attention_heads)
                dim = int(head_dim * partial)
                inv_freq = 1.0 / (base ** (
                    _torch.arange(0, dim, 2, dtype=_torch.int64)
                        .to(dtype=_torch.float32, device=device) / dim))
                return inv_freq, 1.0
            ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters
    except Exception:
        pass
    try:
        # Qwen3 upstream recomputes rotary cos/sin with a per-forward
        # FP32 BMM. Under deterministic cuBLAS plus PrismaQuant imports,
        # that BMM can produce NaNs after model.cuda(). Route Qwen3
        # AutoModel loads to the vendored cached-RoPE copy.
        from .vendored import register_qwen3 as _register_qwen3
        _register_qwen3()
    except Exception:
        pass


@_contextlib.contextmanager
def genuine_weight_initialization():
    """Build a model the way transformers would, inside a PrismaQuant process.

    `_polyfill_transformers` suppresses `PreTrainedModel._initialize_weights`
    outside checkpoint missing-state finalization, because streaming loads
    build a `from_config` skeleton and overwrite checkpoint parameters. That makes `from_config` construction
    silently return **uninitialized** parameters for any tensor the
    modeling file allocates as a bare `nn.Parameter(torch.empty(...))` --
    routed-expert weights and hyper-connection tensors are the common
    cases -- and heap contents are neither reproducible nor guaranteed
    finite, so a forward on such a model can be fine on one run and `nan`
    on the next.

    Wrap a from-config construction in this context manager when the model
    it returns is used as-is:

        with genuine_weight_initialization():
            model = SomeForCausalLM(config)

    A no-op when the polyfill never applied (older transformers, or an
    import that raised), so callers need not test for it.
    """
    initialize_model_runtime()
    token = _genuine_initializing.set(True)
    try:
        yield
    finally:
        _genuine_initializing.reset(token)


def initialize_model_runtime() -> None:
    """Prepare Triton and Transformers before resolving or constructing a model."""
    global _runtime_initialized
    if _runtime_initialized:
        return
    with _runtime_lock:
        if _runtime_initialized:
            return
        _ensure_triton_cache_writable()
        _polyfill_transformers()
        _runtime_initialized = True
