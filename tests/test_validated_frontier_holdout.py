"""Pin review criticals C3/C5: validation windows disjoint from render calib.

The frontier-selecting KL must be measured on calibration windows the
probe/cost/render stages never saw. --calib-skip-first K drops the first K
windows of the (prefix-stable) deterministic loader, making the validation
set [K, K+n) token-disjoint from render calib [0, K) by construction.
"""

from prismaquant.calibration_data import CalibrationRequest, load_validation_calibration
from prismaquant import calibration_data
from types import SimpleNamespace

import pytest
import torch

import prismaquant.validate_assignments_kl as vak


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


def test_skip_first_yields_disjoint_windows(monkeypatch):
    calls = []
    monkeypatch.setattr(calibration_data, "load_calibration", _fake_loader(calls))
    full = load_validation_calibration(None, CalibrationRequest.from_namespace(_args())).selection_repeats[0]
    held = load_validation_calibration(
        None, CalibrationRequest.from_namespace(_args(calib_skip_first=4))).selection_repeats[0]
    assert calls[0]["n"] == 4 and calls[1]["n"] == 8
    assert held.shape[0] == 4
    render_tokens = set(full.flatten().tolist())
    held_tokens = set(held.flatten().tolist())
    assert not render_tokens & held_tokens, "validation overlaps render calib"


def test_skip_first_threads_seed_and_repeats(monkeypatch):
    calls = []
    monkeypatch.setattr(calibration_data, "load_calibration", _fake_loader(calls))
    reps = load_validation_calibration(
        None, CalibrationRequest.from_namespace(_args(calib_repeats=3, calib_skip_first=2, calib_seed=7))).selection_repeats
    assert len(reps) == 3
    assert calls[0]["seed"] == 7
    assert calls[0]["n"] == 3 * 4 + 2  # repeats*n + skip


def test_skip_first_insufficient_corpus_raises(monkeypatch):
    def short(tokenizer, source, n_samples, seqlen, *, calib_seed=42):
        n = min(n_samples, 5)
        return torch.arange(n * seqlen).reshape(n, seqlen)
    monkeypatch.setattr(calibration_data, "load_calibration", short)
    with pytest.raises(RuntimeError, match="skip-first"):
        load_validation_calibration(
            None, CalibrationRequest.from_namespace(_args(calib_skip_first=4))).selection_repeats
