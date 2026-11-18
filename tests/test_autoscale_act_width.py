"""Size retained activations from the widest Linear input."""
from prismaquant import autoscale as A


def test_act_width_takes_max_of_hidden_and_ffn():
    cfg = {"hidden_size": 5376, "intermediate_size": 21504,
           "num_hidden_layers": 60}
    assert A._act_width(cfg) == 21504


def test_act_width_reads_text_config_and_moe_keys():
    cfg = {"text_config": {"hidden_size": 4096, "moe_intermediate_size": 1536,
                           "intermediate_size": 12288}}
    assert A._act_width(cfg) == 12288


def test_act_width_collapses_to_hidden_when_no_ffn_declared():
    cfg = {"hidden_size": 4096, "num_hidden_layers": 32}
    assert A._act_width(cfg) == 4096


def test_estimate_scales_active_bytes_with_act_width():
    """Wide-MLP estimate must exceed the hidden-only estimate in proportion to
    the width ratio (activation term only; the gradient term is fixed)."""
    base = dict(model_path="/nonexistent", num_layers=60, hidden_size=5376,
                nsamples=8, seqlen=1024)
    _, active_hidden = A.estimate_per_layer_bytes(**base)               # act_width None -> hidden
    _, active_wide = A.estimate_per_layer_bytes(**base, act_width=21504)
    grad = 1 * 1024 ** 3  # fallback per-layer weight when no safetensors on disk
    act_hidden = active_hidden - grad
    act_wide = active_wide - grad
    assert abs(act_wide / act_hidden - 21504 / 5376) < 0.01


def test_cache_headroom_keeps_the_explicit_default_without_model_dimensions(tmp_path):
    import json

    assert A.pick_cache_headroom_gb(str(tmp_path), default=12.5) == (
        12.5, {"reason": "no config.json", "headroom_gb": 12.5})
    assert list(tmp_path.iterdir()) == []
    path = tmp_path / "config.json"
    for config in ({}, {"hidden_size": 64}, {"num_hidden_layers": 2}):
        raw = json.dumps(config)
        path.write_text(raw)
        assert A.pick_cache_headroom_gb(str(tmp_path), default=12.5) == (
            12.5, {"reason": "missing layer/hidden", "headroom_gb": 12.5})
        assert path.read_text() == raw
        assert set(tmp_path.iterdir()) == {path}
