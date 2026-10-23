"""Vendored modeling code for architecture gaps and local compatibility fixes.

Currently:
- transformers_deepseek_v4/ — from transformers PR #45643 (Arthur Zucker's
  add-deepseek-v4 branch, vendored 2026-04-27). Required because DSv4 is
  the test target for the prismaquant DSv4-Flash-Base run.
- transformers_qwen3/ — local Qwen3 copy with RoPE cos/sin precomputed at
  init time to avoid deterministic cuBLAS NaNs in per-forward RoPE matmul.

Registering a vendored class is *verified*, never assumed. A registration that
silently fails to apply runs the probe on upstream modelling code — a numerical
change with no symptom — so every registration here re-resolves what the auto
classes will actually build and raises `VendoredOverrideError` if it is not the
vendored class. See `_force_auto_causal_lm_override`.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import NoReturn

_VENDOR_DIR = Path(__file__).parent


_QWEN3_REGISTERED = False

#: model_type -> failure message, for any override whose verification failed.
#: Both current callers of the register functions (`prismaquant/__init__.py` and
#: `model_profiles/registry.py`) wrap the call in `except Exception: pass`, so a
#: raised error alone can be swallowed; this survives it, alongside the stderr
#: banner printed by `_raise_override_failure`.
OVERRIDE_ERRORS: dict[str, str] = {}


class VendoredOverrideError(RuntimeError):
    """A vendored modelling override did not actually take effect."""


def _resolve_auto_causal_lm(model_type: str) -> type | None:
    """Return the class `AutoModelForCausalLM` will build for `model_type`.

    Config-only: resolves classes, never instantiates weights. This mirrors what
    `_BaseAutoModelClass.from_config`/`from_pretrained` do — the config class is
    resolved the way `AutoConfig` resolves it (`CONFIG_MAPPING[model_type]`),
    then looked up in `MODEL_FOR_CAUSAL_LM_MAPPING`, which *is*
    `AutoModelForCausalLM._model_mapping`. Returns None when nothing resolves.
    """
    from transformers import CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING

    try:
        config_cls = CONFIG_MAPPING[model_type]
        resolved = MODEL_FOR_CAUSAL_LM_MAPPING[config_cls]
    except KeyError:
        return None
    if isinstance(resolved, (list, tuple)):
        # `_get_model_class` disambiguates by config.architectures and otherwise
        # takes element 0; an override registration is always a single class.
        resolved = resolved[0] if resolved else None
    return resolved


def _make_config_shim(config_cls: type) -> type:
    """A PrismaQuant-owned subclass of `config_cls`, stable across calls.

    Same `__name__` as the parent (`AutoModelForCausalLM.register` requires the
    model's `config_class.__name__` to match, and `_LazyAutoMapping` resolves
    every *other* auto mapping by config-class name, so tokenizer/base-model
    lookups keep working) but a `prismaquant.vendored` `__module__`, which is
    what transformers >= 5.13.0 filters on. Bound into this module's globals so
    configs of this class stay picklable (probe.pkl carries model metadata).
    """
    shim_qualname = f"_{config_cls.__name__}Shim"
    cached = globals().get(shim_qualname)
    if isinstance(cached, type) and cached.__bases__ == (config_cls,):
        return cached
    shim = type(
        config_cls.__name__,
        (config_cls,),
        {
            "__module__": __name__,
            "__qualname__": shim_qualname,
            "__doc__": (
                f"PrismaQuant subclass of {config_cls.__module__}."
                f"{config_cls.__name__}. Behaviourally identical; exists only so "
                "the vendored model class can be registered against a config "
                "class that transformers will accept as an override key."
            ),
        },
    )
    globals()[shim_qualname] = shim
    return shim


def _register_config_shim_override(
    model_type: str, config_cls: type, model_cls: type
) -> None:
    """Re-key the override on a PrismaQuant-owned config subclass.

    transformers >= 5.13.0 drops `AutoModelForCausalLM.register(cfg, cls)` on
    the floor whenever `cfg.__module__` starts with "transformers." — see
    `_LazyAutoMapping.register` in `transformers/models/auto/auto_factory.py`
    (5.14.1: line 680). The guard is aimed at remote code silently remapping a
    native config for a whole session; it also makes a deliberate local override
    of a natively supported model_type impossible through that call.

    `AutoConfig`'s mapping (`_LazyConfigMapping.register`) applies no such
    filter and its `_extra_content` wins in `__getitem__`, so registering a
    subclass of the native config for the same model_type makes
    `AutoConfig.from_pretrained`/`for_model` produce that subclass, and the
    model registration keyed on it survives. Public API only — no transformers
    internals are patched.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    shim = _make_config_shim(config_cls)
    AutoConfig.register(model_type, shim, exist_ok=True)
    AutoModelForCausalLM.register(shim, model_cls, exist_ok=True)


def _fatal(model_type: str, msg: str) -> NoReturn:
    """Record, shout, then raise.

    The raise alone is not enough: `prismaquant/__init__.py` and
    `model_profiles/registry.py` both call the register functions inside
    `except Exception: pass`, so the banner and `OVERRIDE_ERRORS` are what
    survive a swallowed exception.

    The banner prints once per distinct failure (the register functions are
    re-entered on every profile detect, and a failed registration is never
    cached as done); the raise happens on every call.
    """
    already_reported = OVERRIDE_ERRORS.get(model_type) == msg
    OVERRIDE_ERRORS[model_type] = msg
    if not already_reported:
        banner = "=" * 72
        print(
            f"\n{banner}\nPRISMAQUANT FATAL — vendored modelling override failed\n"
            f"{banner}\n{msg}\n{banner}\n",
            file=sys.stderr,
            flush=True,
        )
    raise VendoredOverrideError(msg)


def _raise_override_failure(
    model_type: str, model_cls: type, resolved: type | None
) -> NoReturn:
    import transformers

    resolved_name = (
        "<unresolvable>"
        if resolved is None
        else f"{resolved.__module__}.{resolved.__name__}"
    )
    msg = (
        f"PrismaQuant's vendored {model_type} modelling code did not take "
        f"effect: AutoModelForCausalLM resolves {resolved_name} for "
        f"model_type={model_type!r}, not {model_cls.__module__}."
        f"{model_cls.__name__} (transformers {transformers.__version__}). "
        "Both the direct AutoModelForCausalLM.register() and the AutoConfig "
        "config-subclass fallback failed to change the resolved class. "
        "transformers 5.13.0 made register() a silent no-op for configs in the "
        "transformers namespace (_LazyAutoMapping.register in "
        "transformers/models/auto/auto_factory.py); a version that also filters "
        "AutoConfig.register would leave no supported override path. Loading "
        "this architecture on upstream modelling code is an invisible numerical "
        "change, so PrismaQuant refuses it. Remedy: pin transformers to a "
        "version where the override applies (last verified end-to-end: 5.12.1 "
        "direct, 5.14.1 via the config subclass) or load the vendored class "
        "explicitly instead of through the auto classes."
    )
    _fatal(model_type, msg)


def _force_auto_causal_lm_override(
    model_type: str, config_cls: type, model_cls: type
) -> str:
    """Make `AutoModelForCausalLM` resolve `model_cls` for `model_type`, or die.

    Returns the route that worked ("direct" or "config_shim"). Raises
    `VendoredOverrideError` if neither did.

    Scope of the guarantee: the resolution checked here is the one every
    PrismaQuant load path takes (`AutoConfig.from_pretrained`/`for_model`, then
    `AutoModelForCausalLM.from_pretrained`/`from_config`). Calling
    `from_config()` with an explicitly constructed *upstream* config class is
    not covered on transformers >= 5.13.0 — the model mapping is keyed on the
    exact config class and transformers refuses to remap a native one.
    """
    from transformers import AutoModelForCausalLM

    AutoModelForCausalLM.register(config_cls, model_cls, exist_ok=True)
    if _resolve_auto_causal_lm(model_type) is model_cls:
        OVERRIDE_ERRORS.pop(model_type, None)
        return "direct"

    _register_config_shim_override(model_type, config_cls, model_cls)
    resolved = _resolve_auto_causal_lm(model_type)
    if resolved is model_cls:
        OVERRIDE_ERRORS.pop(model_type, None)
        return "config_shim"

    _raise_override_failure(model_type, model_cls, resolved)


def register_qwen3() -> None:
    """Route Qwen3 causal-LM loads to PrismaQuant's vendored model.

    Idempotent — safe to call on any PrismaQuant import or profile detect.
    Uses the upstream Qwen3Config and only replaces the causal-LM model class.

    Verified, not assumed: `AutoModelForCausalLM.register()` is a silent no-op
    on transformers >= 5.13.0 for a natively supported model_type, and running
    the probe on upstream Qwen3 modelling code (per-forward RoPE BMM, the
    deterministic-cuBLAS NaN this vendoring exists to avoid) has no symptom
    other than the numbers. So this re-resolves the class the auto classes will
    build, falls back to a config-subclass registration, and raises
    `VendoredOverrideError` if the vendored class still is not what resolves.
    The registered flag is only set once that check passes.
    """
    global _QWEN3_REGISTERED
    if _QWEN3_REGISTERED:
        return

    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from .transformers_qwen3.modeling_qwen3 import Qwen3ForCausalLM

    _force_auto_causal_lm_override("qwen3", Qwen3Config, Qwen3ForCausalLM)
    _QWEN3_REGISTERED = True


def register_deepseek_v4() -> None:
    """Register vendored DeepSeek-V4 modeling code with transformers.

    Idempotent — safe to call multiple times. After this returns:
    - `transformers.models.deepseek_v4` is importable
    - `AutoConfig.from_pretrained` resolves model_type='deepseek_v4'
    - `AutoModelForCausalLM.from_pretrained` instantiates DeepseekV4ForCausalLM

    Verified, not assumed, on both counts. transformers gained a *native*
    `deepseek_v4` (absent in 5.7.0, present by 5.9.0), so the vendored package
    now shares a module path with an upstream implementation: if that upstream
    module is imported first, the idempotence check below would otherwise return
    having installed nothing. And from 5.13.0 the model-class registration is
    filtered out (see `_register_config_shim_override`). Both are silent, so both
    are checked.
    """
    pkg_name = "transformers.models.deepseek_v4"
    vendored_dir = _VENDOR_DIR / "transformers_deepseek_v4"
    existing = sys.modules.get(pkg_name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        existing_dir = Path(existing_file).resolve().parent if existing_file else None
        if existing_dir == vendored_dir.resolve():
            return  # already ours — genuinely idempotent
        _fatal(
            "deepseek_v4",
            f"{pkg_name} is already imported from {existing_file!r}, which is "
            f"not PrismaQuant's vendored copy at {vendored_dir}. transformers "
            "ships a native deepseek_v4 and it was imported before "
            "register_deepseek_v4() ran, so the vendored DSv4 modelling code "
            "(PR #45643 + the per-expert ModuleList the Fisher probe needs) "
            "cannot be installed in this process. Call "
            "prismaquant.vendored.register_deepseek_v4() before anything "
            "touches transformers' deepseek_v4 module.",
        )

    # Ensure parent package is importable.
    importlib.import_module("transformers.models")

    # Extend transformers' ALLOWED_LAYER_TYPES to accept DSv4's new attention
    # types. PR #45643 lands these in transformers.configuration_utils; on
    # 5.5.4 we monkey-patch the tuple. Idempotent: re-running just rebuilds
    # the same set.
    from transformers import configuration_utils as _cu
    _DSV4_TYPES = ("compressed_sparse_attention", "heavily_compressed_attention")
    _existing = tuple(_cu.ALLOWED_LAYER_TYPES)
    _to_add = tuple(t for t in _DSV4_TYPES if t not in _existing)
    if _to_add:
        _cu.ALLOWED_LAYER_TYPES = _existing + _to_add

    # Register sqrtsoftplus in ACT2FN (used by DSv4's TopKRouter scoring_func).
    # PR #45643 adds this to transformers.activations; we patch it here.
    import torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    from transformers.activations import ACT2FN

    class _SqrtSoftplus(_nn.Module):
        def forward(self, x):
            return torch.sqrt(_F.softplus(x))

    if "sqrtsoftplus" not in ACT2FN:
        ACT2FN["sqrtsoftplus"] = _SqrtSoftplus

    if not (vendored_dir / "__init__.py").exists():
        raise RuntimeError(f"vendored DSv4 missing at {vendored_dir}")

    spec = importlib.util.spec_from_file_location(
        pkg_name,
        str(vendored_dir / "__init__.py"),
        submodule_search_locations=[str(vendored_dir)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to build import spec for {pkg_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = module
    spec.loader.exec_module(module)

    # Register with the auto-classes so AutoConfig / AutoModelForCausalLM work.
    from transformers import AutoConfig
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

    # AutoConfig first: on a transformers without a native deepseek_v4 the model
    # mapping cannot resolve the config class at all until this lands.
    AutoConfig.register("deepseek_v4", DeepseekV4Config, exist_ok=True)
    _force_auto_causal_lm_override(
        "deepseek_v4", DeepseekV4Config, DeepseekV4ForCausalLM
    )

    # Select the per-expert ModuleList constructor on SparseMoeBlock so
    # prismaquant's per-Linear Fisher hooks observe each routed expert
    # individually. Must run AFTER the modeling module exec_module above
    # but BEFORE any SparseMoeBlock is instantiated. vLLM serving uses
    # the packed PR #40860 format directly; this only affects in-process
    # model construction in the probe path.
    from .dsv4_probe_experts import enable_per_expert_experts
    enable_per_expert_experts()
