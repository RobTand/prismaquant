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
