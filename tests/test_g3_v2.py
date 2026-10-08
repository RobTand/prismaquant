"""CPU contract tests use real tensors and ordered array files."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from prismaquant.g3_v2 import measure_g3, preflight_g3
from prismaquant.g3_numerics import token_kl, fp8_per_token_dynamic


def bind(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def put(path, value):
    path.write_text(json.dumps(value))
    return bind(path)


def make_retained_configuration(tmp_path):
    tokenizer_object = Tokenizer(WordLevel({"token0": 0, "token1": 1, "[prefix]": 2, "token3": 3,
                                          "token4": 4, "token5": 5, "token6": 6}, unk_token="token0"))
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text(tokenizer_object.to_str())
    tokenizer = bind(tokenizer_path)
    protocol = put(tmp_path/"protocol.json", {"schema": "prismaquant.g3_protocol/2", "name": "cpu-contract",
        "context_length": 5, "window_count": 2, "vocab_size": 7, "prefix_tokens": ["[prefix]"],
        "prefix_ids": [2], "tensor_parallel_size": 1, "tile_rows": 2})
    windows, teachers, candidates = [], [], []
    for i in range(2):
        tokens = tmp_path/f"tokens-{i}.npy"
        np.save(tokens, np.arange(5, dtype=np.int32))
        windows.append({"window_id": f"w{i}", "tokens_path": str(tokens), "tokens_sha256": bind(tokens)["sha256"]})
        logits = np.arange(28, dtype=np.float32).reshape(4, 7)/10+i
        teacher = tmp_path/f"teacher-{i}.npy"
        candidate = tmp_path/f"candidate-{i}.npy"
        np.save(teacher, logits)
        np.save(candidate, logits+np.arange(7, dtype=np.float32)/100)
        teachers.append({"window_id": f"w{i}", "path": str(teacher), "file_sha256": bind(teacher)["sha256"],
                         "array_sha256": hashlib.sha256(logits.tobytes()).hexdigest()})
        candidates.append({"window_id": f"w{i}", "path": str(candidate), "file_sha256": bind(candidate)["sha256"]})
    panel = put(tmp_path/"panel.json", {"schema": "prismaquant.g3_panel/1", "windows": windows})
    teacher = put(tmp_path/"teacher.json", {"windows": ["w0", "w1"], "prefix": {"ids": [2]},
        "arrays": teachers, "panel": panel, "tokenizer": tokenizer, "source_model_identity": {"model": "cpu-contract"}})
    candidate = put(tmp_path/"candidate.json", {"windows": ["w0", "w1"], "arrays": candidates})
    return {"schema": "prismaquant.g3_v2/1", "protocol": protocol, "tokenizer": tokenizer,
        "panel": panel, "teacher": teacher, "candidate": {"backend": "retained_logits", "receipt": candidate},
        "device": "cpu", "criteria": []}


@pytest.fixture
def configuration(tmp_path):
    return make_retained_configuration(tmp_path)


def test_retained_arrays_preserve_order_and_full_precision(configuration, tmp_path):
    facts = measure_g3(configuration, tmp_path/"result.json")
    assert facts["population"]["window_ids"] == ["w0", "w1"]
    array = np.load(facts["artifacts"][0]["path"])
    assert array.dtype == np.float64 and array.shape == (2, 4)
    assert facts["metrics"]["mean_kl"] == float(array.reshape(-1).mean())


def test_preflight_has_no_quality_metrics(configuration):
    assert "metrics" not in preflight_g3(configuration)


def test_teacher_pair_order_is_a_correctness_refusal(configuration, tmp_path):
    path = Path(configuration["teacher"]["path"])
    receipt = json.loads(path.read_text())
    receipt["arrays"].reverse()
    configuration["teacher"] = put(path, receipt)
    with pytest.raises(ValueError, match="order"):
        measure_g3(configuration, tmp_path/"bad.json")


def test_prefix_ids_come_from_the_actual_tokenizer(configuration):
    path = Path(configuration["tokenizer"]["path"])
    tokenizer = Tokenizer(WordLevel({"token0": 0, "token1": 1, "token2": 2, "[prefix]": 3,
                                   "token4": 4, "token5": 5, "token6": 6}, unk_token="token0"))
    path.write_text(tokenizer.to_str())
    configuration["tokenizer"] = bind(path)
    with pytest.raises(ValueError, match="prefix"):
        preflight_g3(configuration)


def test_teacher_integrity_cannot_be_a_dev_stamp(configuration, tmp_path):
    path = Path(json.loads(Path(configuration["teacher"]["path"]).read_text())["arrays"][0]["path"])
    np.save(path, np.ones((4, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="checksum"):
        measure_g3(configuration, tmp_path/"bad.json")


def test_kl_matches_accepted_fp64_operation_order():
    teacher = torch.arange(77, dtype=torch.float32).reshape(11, 7)/10
    candidate = teacher+torch.arange(7)/100
    t, c = torch.log_softmax(teacher.double(), -1), torch.log_softmax(candidate.double(), -1)
    expected = (t.exp()*(t-c)).sum(-1, dtype=torch.float64)
    assert torch.equal(token_kl(teacher, candidate, tile_rows=3, require_cuda=False), expected)


def test_fp8_signed_zero_and_tp_slices():
    x = torch.tensor([[0., -0., 1., -1., 448., -448., 0.00001, -0.00001]])
    codes, scales = fp8_per_token_dynamic(x)
    assert torch.signbit(codes.float())[0, 1]
    assert scales.dtype == torch.float32 and codes.dtype == torch.float8_e4m3fn


def test_changed_input_tokens_do_not_pair_with_the_old_teacher(configuration):
    panel_path = Path(configuration["panel"]["path"])
    panel = json.loads(panel_path.read_text())
    window = panel["windows"][0]
    tokens_path = Path(window["tokens_path"])
    tokens = np.load(tokens_path, allow_pickle=False)
    tokens[0] = 1
    np.save(tokens_path, tokens, allow_pickle=False)
    window["tokens_sha256"] = bind(tokens_path)["sha256"]
    configuration["panel"] = put(panel_path, panel)
    with pytest.raises(ValueError, match="panel.*pairing"):
        preflight_g3(configuration)
