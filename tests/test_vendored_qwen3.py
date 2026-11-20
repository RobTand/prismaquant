import pickle
import sys
import types

import pytest
import torch
import transformers


def _tiny_qwen3_kwargs():
    return dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rope_theta=10000.0,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _tiny_qwen3_config():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    return Qwen3Config(**_tiny_qwen3_kwargs())


def test_qwen3_auto_model_uses_vendored_causal_lm():
    """The contract: the auto classes build the vendored Qwen3, on every version.

    `AutoConfig.for_model` resolves the config class exactly as
    `AutoConfig.from_pretrained` does (`CONFIG_MAPPING[model_type]`), which is
    the entry point of every PrismaQuant load path, so this covers both the
    direct registration (transformers <= 5.12.1) and the config-subclass
    fallback (>= 5.13.0, where `AutoModelForCausalLM.register()` on a native
    model_type is dropped).
    """
    import prismaquant  # noqa: F401  (import-time register_qwen3)
    from prismaquant.vendored import register_qwen3
    from prismaquant.vendored.transformers_qwen3 import Qwen3ForCausalLM
    from transformers import AutoConfig, AutoModelForCausalLM

    register_qwen3()  # raises VendoredOverrideError if the override is a no-op

    cfg = AutoConfig.for_model("qwen3", **_tiny_qwen3_kwargs())
    model = AutoModelForCausalLM.from_config(cfg)

    assert isinstance(model, Qwen3ForCausalLM)


def test_register_qwen3_verification_is_config_only_and_agrees_with_from_config():
    """The cheap verification must agree with what `from_config` actually builds."""
    import prismaquant  # noqa: F401
    from prismaquant.vendored import _resolve_auto_causal_lm, register_qwen3
    from prismaquant.vendored.transformers_qwen3 import Qwen3ForCausalLM
    from transformers import AutoConfig, AutoModelForCausalLM

    register_qwen3()

    resolved = _resolve_auto_causal_lm("qwen3")
    assert resolved is Qwen3ForCausalLM

    cfg = AutoConfig.for_model("qwen3", **_tiny_qwen3_kwargs())
    assert type(AutoModelForCausalLM.from_config(cfg)) is resolved


def test_force_override_raises_loudly_when_resolution_disagrees(monkeypatch, capsys):
    """A registration that does not apply must fail loudly, not silently pass.

    This is the version-independent half of the fix: whatever transformers does
    with `register()`, PrismaQuant refuses to continue on a modelling path it did
    not install. The shim step is stubbed out so no global transformers mapping
    is mutated by this test.
    """
    import prismaquant  # noqa: F401
    import prismaquant.vendored as vendored
    from prismaquant.vendored.transformers_qwen3 import Qwen3ForCausalLM
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    class _NotTheVendoredClass:
        pass

    monkeypatch.setattr(
        vendored, "_resolve_auto_causal_lm", lambda model_type: _NotTheVendoredClass
    )
    monkeypatch.setattr(
        vendored,
        "_register_config_shim_override",
        lambda model_type, config_cls, model_cls: None,
    )
    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "qwen3", "")

    with pytest.raises(vendored.VendoredOverrideError) as excinfo:
        vendored._force_auto_causal_lm_override(
            "qwen3", Qwen3Config, Qwen3ForCausalLM
        )

    msg = str(excinfo.value)
    assert "qwen3" in msg
    assert transformers.__version__ in msg
    assert "_NotTheVendoredClass" in msg
    assert "pin transformers" in msg
    # Recorded, so the two callers that wrap this in `except Exception: pass`
    # cannot make the failure disappear entirely.
    assert vendored.OVERRIDE_ERRORS["qwen3"] == msg
    # And printed where an operator will see it even if the raise is swallowed.
    assert "PRISMAQUANT FATAL" in capsys.readouterr().err


def test_register_qwen3_does_not_cache_a_failed_registration(monkeypatch):
    """A failed verification must leave the module retryable, not marked done."""
    import prismaquant  # noqa: F401
    import prismaquant.vendored as vendored

    class _NotTheVendoredClass:
        pass

    monkeypatch.setattr(vendored, "_QWEN3_REGISTERED", False)
    monkeypatch.setattr(
        vendored, "_resolve_auto_causal_lm", lambda model_type: _NotTheVendoredClass
    )
    monkeypatch.setattr(
        vendored,
        "_register_config_shim_override",
        lambda model_type, config_cls, model_cls: None,
    )

    with pytest.raises(vendored.VendoredOverrideError):
        vendored.register_qwen3()

    assert vendored._QWEN3_REGISTERED is False


def test_config_shim_is_a_faithful_stand_in_for_the_upstream_config():
    """The >= 5.13.0 fallback re-keys on this subclass; it must change nothing.

    Serialized config bytes are a hard constraint (vLLM reads them), so the shim
    is only acceptable if `to_dict()` is identical and `model_type` survives.
    """
    from prismaquant.vendored import _make_config_shim
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    shim = _make_config_shim(Qwen3Config)

    assert shim is _make_config_shim(Qwen3Config)  # stable identity across calls
    assert issubclass(shim, Qwen3Config)
    # `AutoModelForCausalLM.register` requires config_class.__name__ to match,
    # and every other auto mapping resolves by config-class name.
    assert shim.__name__ == "Qwen3Config"
    # ... while the module is what transformers >= 5.13.0 filters registrations on.
    assert not shim.__module__.startswith("transformers.")
    assert shim.model_type == Qwen3Config.model_type

    kwargs = _tiny_qwen3_kwargs()
    upstream_cfg = Qwen3Config(**kwargs)
    shim_cfg = shim(**kwargs)
    assert shim_cfg.to_dict() == upstream_cfg.to_dict()
    assert shim_cfg.to_json_string() == upstream_cfg.to_json_string()
    assert isinstance(shim_cfg, Qwen3Config)
    # probe.pkl carries model metadata, so the config class must stay picklable.
    assert type(pickle.loads(pickle.dumps(shim_cfg))) is shim


def test_register_deepseek_v4_refuses_a_foreign_module(monkeypatch, capsys):
    """transformers ships a native deepseek_v4 (>= 5.9.0) at the same module path.

    If it is imported first, the old idempotence check returned having installed
    nothing and the probe silently ran upstream DSv4 modelling code with the
    packed expert layout the Fisher hooks cannot see.
    """
    import prismaquant.vendored as vendored

    foreign = types.ModuleType("transformers.models.deepseek_v4")
    foreign.__file__ = "/usr/lib/transformers/models/deepseek_v4/__init__.py"
    monkeypatch.setitem(sys.modules, "transformers.models.deepseek_v4", foreign)
    monkeypatch.setitem(vendored.OVERRIDE_ERRORS, "deepseek_v4", "")

    with pytest.raises(vendored.VendoredOverrideError) as excinfo:
        vendored.register_deepseek_v4()

    msg = str(excinfo.value)
    assert "deepseek_v4" in msg
    assert "vendored" in msg
    assert vendored.OVERRIDE_ERRORS["deepseek_v4"] == msg
    assert "PRISMAQUANT FATAL" in capsys.readouterr().err


def test_vendored_qwen3_rope_matches_upstream_on_cpu():
    from prismaquant.vendored.transformers_qwen3 import (
        Qwen3RotaryEmbedding as VendoredQwen3RotaryEmbedding,
    )
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3RotaryEmbedding as UpstreamQwen3RotaryEmbedding,
    )

    cfg = _tiny_qwen3_config()
    x = torch.zeros((2, 5, cfg.hidden_size), dtype=torch.float32)
    position_ids = torch.tensor(
        [
            [0, 1, 2, 7, 15],
            [3, 4, 8, 16, 31],
        ],
        dtype=torch.long,
    )

    upstream = UpstreamQwen3RotaryEmbedding(cfg)
    vendored = VendoredQwen3RotaryEmbedding(cfg)

    upstream_cos, upstream_sin = upstream(x, position_ids)
    vendored_cos, vendored_sin = vendored(x, position_ids)

    assert vendored.cos_cached.shape == (cfg.max_position_embeddings, cfg.head_dim)
    torch.testing.assert_close(vendored_cos, upstream_cos, rtol=0, atol=0)
    torch.testing.assert_close(vendored_sin, upstream_sin, rtol=0, atol=0)


def test_vendored_qwen3_requires_exactly_one_input():
    from prismaquant.vendored.transformers_qwen3 import Qwen3ForCausalLM

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = Qwen3ForCausalLM(_tiny_qwen3_config()).model
    both = {"input_ids": torch.tensor([[1]]), "inputs_embeds": torch.zeros(1, 1, model.config.hidden_size)}
    model.layers[0].register_forward_pre_hook(lambda *_: pytest.fail("The decoder layer must not run."))
    for kwargs in ({}, both):
        with pytest.raises(ValueError, match="exactly one of input_ids or inputs_embeds"):
            model(**kwargs)


def test_vendored_qwen3_model_refreshes_invalid_rope_cache():
    from prismaquant.vendored.transformers_qwen3 import Qwen3ForCausalLM

    cfg = _tiny_qwen3_config()
    model = Qwen3ForCausalLM(cfg)
    rope = model.model.rotary_emb

    rope.inv_freq.fill_(float("nan"))
    rope.cos_cached.fill_(float("nan"))
    rope.sin_cached.fill_(float("nan"))

    model._prismaquant_reset_rope_caches()

    assert torch.isfinite(rope.inv_freq).all()
    assert torch.isfinite(rope.cos_cached).all()
    assert torch.isfinite(rope.sin_cached).all()
