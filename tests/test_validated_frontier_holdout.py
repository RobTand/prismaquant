"""Selection excludes the checked fit prefix of the actual captured draw."""

from prismaquant.calibration_data import CalibrationRequest, load_validation_calibration
from prismaquant import calibration_data
from types import SimpleNamespace

import pytest
import torch

import json
from tests.test_calibration_draw_packing import PositionTokenizer


def _args(**kw):
    base = dict(
        dataset="fake.jsonl", n_calib_samples=4, calib_seqlen=8,
        calib_repeats=1, calib_seed=42, calib_skip_first=0,
        calib_split="train", calib_repeat_seed_stride=997,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _fake_loader(record):
    def fake(tokenizer, source, n_samples, seqlen, *, calib_seed=42):
        record.append({"n": n_samples, "seed": calib_seed})
        # deterministic, prefix-stable: window i = [i*seqlen, ...)
        return torch.arange(n_samples * seqlen).reshape(n_samples, seqlen)
    return fake


def test_skip_first_uses_the_actual_owned_fit_and_selection_windows(tmp_path):
    corpus = tmp_path / "holdout.jsonl"
    corpus.write_text("".join(json.dumps({"text": ",".join(
        str(row * 100 + col) for col in range(8))}) + "\n" for row in range(8)))
    tokenizer = PositionTokenizer()
    request = CalibrationRequest.from_namespace(_args(
        dataset=str(corpus), calib_skip_first=4))
    draw = load_validation_calibration(tokenizer, request)
    assert torch.equal(draw.fit_ids, calibration_data.load_calibration(
        tokenizer, str(corpus), 4, 8, calib_seed=42))
    assert len(draw.selection_repeats) == 1
    assert draw.selection_repeats[0].shape == (4, 8)
    # IDs in this fixture identify unique corpus positions, including across documents.
    assert set(draw.fit_ids.flatten().tolist()).isdisjoint(
        draw.selection_repeats[0].flatten().tolist())


def test_skip_first_threads_seed_and_repeats(monkeypatch):
    calls = []
    monkeypatch.setattr(calibration_data, "load_calibration", _fake_loader(calls))
    reps = load_validation_calibration(
        None, CalibrationRequest.from_namespace(_args(calib_repeats=3, calib_skip_first=2, calib_seed=7))).selection_repeats
    assert len(reps) == 3
    assert calls[0]["seed"] == 7
    assert calls == [{"n": 3 * 4 + 2, "seed": 7}, {"n": 2, "seed": 7}]


def test_skip_first_insufficient_corpus_raises(monkeypatch):
    def short(tokenizer, source, n_samples, seqlen, *, calib_seed=42):
        n = min(n_samples, 5)
        return torch.arange(n * seqlen).reshape(n, seqlen)
    monkeypatch.setattr(calibration_data, "load_calibration", short)
    with pytest.raises(RuntimeError, match="skip-first"):
        load_validation_calibration(
            None, CalibrationRequest.from_namespace(_args(calib_skip_first=4))).selection_repeats
