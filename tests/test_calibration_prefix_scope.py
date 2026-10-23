"""Requested corpus limits can change a shuffled calibration prefix."""
import sys
from types import SimpleNamespace

import torch
import pytest

from prismaquant import calibration_data
from tests.test_calibration_draw_packing import PositionTokenizer


def test_streaming_shuffle_does_not_preserve_a_smaller_requested_prefix(monkeypatch):
    rows = [{'text': ','.join(str(100 * row + col) for col in range(4))}
            for row in range(32)]
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=lambda *a, **k: rows))
    tokenizer = PositionTokenizer()
    fit = calibration_data.load_calibration(tokenizer, 'synthetic_corpus', 1, 4, calib_seed=7)
    complete = calibration_data.load_calibration(tokenizer, 'synthetic_corpus', 3, 4, calib_seed=7)
    assert not torch.equal(fit, complete[:1])
    request = calibration_data.CalibrationRequest(
        source='synthetic_corpus', n_samples=2, seqlen=4, repeats=1,
        seed=7, skip_first=1, split='train', repeat_seed_stride=997,
    )
    with pytest.raises(ValueError, match='does not preserve the requested fit prefix'):
        calibration_data.load_validation_calibration(tokenizer, request)
