import json
import random
from types import SimpleNamespace

import torch

from prismaquant import calibration_data
from tests.test_calibration_draw_packing import PositionTokenizer


def _corpus(tmp_path):
    source = tmp_path / 'calibration.jsonl'
    source.write_text(''.join(json.dumps({'text': ','.join(
        str(row * 100 + col) for col in range(8))}) + '\n' for row in range(8)))
    return str(source)


def test_validation_request_preserves_explicit_zero_seed(tmp_path):
    source = _corpus(tmp_path)
    args = SimpleNamespace(dataset=source, n_calib_samples=2, calib_seqlen=4,
        calib_repeats=1, calib_seed=0, calib_skip_first=0,
        calib_split='train', calib_repeat_seed_stride=997)
    tokenizer = PositionTokenizer()
    draw = calibration_data.load_validation_calibration(
        tokenizer, calibration_data.CalibrationRequest.from_namespace(args))
    expected = calibration_data.load_calibration(tokenizer, source, 2, 4, calib_seed=0)
    assert torch.equal(draw.selection_repeats[0], expected)


def test_corpus_sampling_preserves_the_callers_random_state(tmp_path):
    state = random.getstate()
    try:
        calibration_data.load_calibration(PositionTokenizer(), _corpus(tmp_path), 2, 4, calib_seed=7)
        assert random.getstate() == state
    finally:
        random.setstate(state)
