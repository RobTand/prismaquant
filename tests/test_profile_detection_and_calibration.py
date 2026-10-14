from argparse import Namespace

import torch

from prismaquant import build_production_cache
from prismaquant.model_profiles import DefaultProfile, detect_profile_with_warning
from prismaquant.calibration_data import load_calibration_repeats


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

    out = load_calibration_repeats(tokenizer, args)

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
