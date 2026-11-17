import json
from types import SimpleNamespace

import torch

from prismaquant.calibration_data import load_calibration


class PositionTokenizer:
    eos_token_id = 999

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=torch.tensor([
            [int(value) for value in text.split(",")]], dtype=torch.long))


def test_heldout_packing_does_not_reuse_the_probe_document(tmp_path):
    corpus = tmp_path / "draw.jsonl"
    corpus.write_text("".join(json.dumps({"text": text}) + "\n" for text in (
        "0,1,2,3,4,5,6,7", "100,101,102", "200,201,202")))
    tokenizer = PositionTokenizer()
    probe = load_calibration(tokenizer, str(corpus), 1, 4)
    complete = load_calibration(tokenizer, str(corpus), 3, 4)
    assert torch.equal(probe, complete[:1])
    # Each non-EOS ID names one unique source position in this corpus.
    assert set(probe.flatten().tolist()).isdisjoint(set(complete[1:].flatten().tolist()))


def test_windowed_text_sampling_keeps_eos_and_refuses_short_corpora():
    from prismaquant.calibration_data import _sample_token_windows_from_texts

    class Tokenizer:
        eos_token_id = 99

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=[[11, 12]])

    tokenizer = Tokenizer()
    windows = _sample_token_windows_from_texts(["", "text"], tokenizer, 1, 3, seed=42)
    assert windows.dtype == torch.long
    assert windows.tolist() == [[11, 12, 99]]
    try:
        _sample_token_windows_from_texts(["text"], tokenizer, 2, 3, seed=42)
    except RuntimeError as exc:
        assert str(exc) == "only built 1 calibration windows; needed 2"
    else:
        raise AssertionError("short calibration corpus was accepted")
