from argparse import Namespace

import pytest
import torch

from prismaquant import build_production_cache
from prismaquant.model_profiles import DefaultProfile, detect_profile_with_warning
from prismaquant.calibration_data import CalibrationRequest, load_validation_calibration


def test_detect_profile_with_warning_logs_default_profile(tmp_path, capsys):
    missing_model = tmp_path / "missing-config-model"
    missing_model.mkdir()

    profile = detect_profile_with_warning(
        str(missing_model),
        entrypoint="unit-test",
    )

    captured = capsys.readouterr()
    assert isinstance(profile, DefaultProfile)
    assert "[unit-test] WARNING: resolved DefaultProfile" in captured.out
    assert "Architecture-specific fused-sibling" in captured.out


def test_validate_assignment_dataset_loader_forwards_calib_seed(monkeypatch):
    seen = {}

    def fake_load_calibration(tokenizer, source, n_samples, seqlen, *, calib_seed=42):
        seen["args"] = (tokenizer, source, n_samples, seqlen, calib_seed)
        return torch.zeros((n_samples, seqlen), dtype=torch.long)

    monkeypatch.setattr(
        "prismaquant.calibration_data.load_calibration",
        fake_load_calibration,
    )
    tokenizer = object()
    args = Namespace(
        dataset="calib.jsonl",
        n_calib_samples=3,
        calib_seqlen=8,
        calib_seed=123,
        calib_split="train",
        calib_repeats=1,
        calib_repeat_seed_stride=1000,
    )

    out = load_validation_calibration(tokenizer, CalibrationRequest.from_namespace(args)).selection_repeats

    assert len(out) == 1
    assert seen["args"] == (tokenizer, "calib.jsonl", 3, 8, 123)


def test_build_cache_dataset_loader_forwards_calib_seed(monkeypatch):
    seen = {}

    def fake_load_calibration(tokenizer, source, n_samples, seqlen, *, calib_seed=42):
        seen["args"] = (source, n_samples, seqlen, calib_seed)
        return torch.zeros((n_samples, seqlen), dtype=torch.long)

    monkeypatch.setattr(
        build_production_cache,
        "load_calibration",
        fake_load_calibration,
    )
    args = Namespace(
        dataset="calib.jsonl",
        n_calib_samples=5,
        calib_seqlen=16,
        calib_seed=321,
        calib_split="train",
    )

    out = build_production_cache._load_cache_calibration(object(), args)

    assert tuple(out.shape) == (5, 16)
    assert seen["args"] == ("calib.jsonl", 5, 16, 321)


@pytest.mark.parametrize("value,streaming", [("1:2", False), ("2:1", True), ("bad", True)])
def test_layer_range_cli_refuses_invalid_input(value, streaming):
    from prismaquant.build_production_cache import main
    with pytest.raises(SystemExit) as exc:
        main(["--model", "unused", "--output", "unused", "--layer-range", value]
             + (["--streaming"] if streaming else []))
    assert exc.value.code == 2



@pytest.mark.parametrize("start,expected", [(0, ["model.layers.0.x", "lm_head"]), (1, ["model.layers.1.x"])])
def test_layer_range_cli_filters_render_names(monkeypatch, tmp_path, start, expected):
    from prismaquant import build_production_cache as b
    from prismaquant import streaming_production_cache as stream
    from types import SimpleNamespace
    import transformers
    monkeypatch.setattr(b, "resolve_source_checkpoint_path", lambda path: path)
    monkeypatch.setattr(b, "require_cuda_hot_path", lambda *a: "cuda")
    monkeypatch.setattr(b, "_load_assignment", lambda *a: dict.fromkeys(["model.layers.0.x", "model.layers.1.x", "lm_head"], "NVFP4"))
    monkeypatch.setattr(b, "_load_col_weights", lambda *a: None)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: None)
    monkeypatch.setattr(b, "_load_cache_calibration", lambda *a: None)
    monkeypatch.setattr(b, "calibration_data_hash", lambda *a: "hash")
    monkeypatch.setattr(b, "detect_profile_with_warning", lambda *a, **k: SimpleNamespace(body_layer_prefix=lambda: "model.layers", live_to_recipe_name=lambda x: x))
    def capture(*a, **kw):
        assert kw["include_qnames"] == expected
        raise LookupError("range checked")
    monkeypatch.setattr(stream, "fill_production_weight_cache_streaming", capture)
    with pytest.raises(LookupError, match="range checked"):
        b.main(["--model", "unused", "--output", str(tmp_path / "out"), "--streaming",
                "--layer-range", f"{start}:{start + 1}", "--render-scope", "assignment",
                "--render-layer-config", "unused", "--cache-dir", str(tmp_path),
                "--activation-cache-dir", str(tmp_path)])
