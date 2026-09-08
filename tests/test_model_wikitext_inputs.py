"""Offline inputs bind a model without relaxing the DSv4 release contract."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from tools import dsv4_wikitext_inputs as inputs
from tools import prepare_dsv4_wikitext_inputs as prepare


@pytest.fixture
def materializer(monkeypatch, tmp_path):
    model = tmp_path / "source-model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({
        "model_type": "glm5_next", "text_config": {
            "model_type": "glm5_next_text", "vocab_size": 32,
        },
    }))
    (model / "tokenizer.json").write_text('{"synthetic_tokenizer":true}')
    (model / "tokenizer_config.json").write_text(
        '{"tokenizer_class":"TokenizersBackend"}')
    calls = []

    class Tokenizer:
        def __len__(self):
            return 32

        def __call__(self, text, **kwargs):
            calls.append((text, kwargs))
            count = 20_000 if text == "train" else 10_000
            return SimpleNamespace(input_ids=[i % 31 for i in range(count)])

    constructor = SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=constructor, PreTrainedTokenizerFast=constructor))
    monkeypatch.setattr(prepare, "version", lambda _: inputs.DATASETS_VERSION)

    def corpus(*, split, cache_dir, dataset_repo="wikitext"):
        assert dataset_repo in {"wikitext", "Salesforce/wikitext"}
        evidence = inputs._expected_dataset(split=split)
        evidence.pop("total_tokens")
        return None, split, evidence

    monkeypatch.setattr(prepare, "_load_corpus", corpus)
    return model, calls


def test_model_v2_explicit_materialization_accepts_model_tokenizer(materializer):
    model, calls = materializer
    payload = prepare._build_payload(
        model=model, cache_dir="unused", input_schema="model-v2")
    assert payload["schema"] == "prismaquant.model_wikitext_inputs/2"
    assert payload["model"] == {
        "schema": "prismaquant.wikitext_input_model/1",
        "model_type": "glm5_next", "text_model_type": "glm5_next_text",
        "vocab_size": 32,
    }
    assert payload["full_kl"]["dataset"]["total_tokens"] == 20_000
    assert len(payload["full_kl"]["token_ids"]) == 8
    assert all(len(row) == 512 for row in payload["full_kl"]["token_ids"])
    assert len(payload["ppl"]["token_ids"]) == 8192
    assert all(kwargs == {"add_special_tokens": False} for _, kwargs in calls)


def test_v1_default_still_refuses_non_dsv4_tokenizer(materializer):
    model, _ = materializer
    with pytest.raises(inputs.DSv4WikiTextInputsError, match="tokenizer value identity differs"):
        prepare._build_payload(model=model, cache_dir="unused")


def _write(tmp_path, payload):
    path = tmp_path / "inputs.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _reseal(payload):
    payload["semantic_sha256"] = inputs.canonical_sha256({
        key: value for key, value in payload.items() if key != "semantic_sha256"
    })


def _v2(materializer):
    model, _ = materializer
    return prepare._build_payload(
        model=model, cache_dir="unused", input_schema="model-v2")


def test_v2_reader_requires_independent_file_sha(materializer, tmp_path):
    payload = _v2(materializer)
    path, digest = _write(tmp_path, payload)
    kwargs = dict(expected_tokenizer_identity=payload["tokenizer"],
                  expected_model_identity=payload["model"])
    with pytest.raises(inputs.DSv4WikiTextInputsError, match="independent.*SHA256"):
        inputs.load_wikitext_inputs(path, **kwargs)
    with pytest.raises(inputs.DSv4WikiTextInputsError, match="file SHA256"):
        inputs.load_wikitext_inputs(path, expected_sha256="0" * 64, **kwargs)
    assert inputs.load_wikitext_inputs(path, expected_sha256=digest, **kwargs) == payload


@pytest.mark.parametrize("field,value", [
    ("model_type", "other_model"), ("text_model_type", "other_text"),
    ("vocab_size", 33),
])
def test_v2_refuses_wrong_current_model(materializer, tmp_path, field, value):
    payload = _v2(materializer)
    path, digest = _write(tmp_path, payload)
    expected = {**payload["model"], field: value}
    with pytest.raises(inputs.DSv4WikiTextInputsError, match="model.*differs"):
        inputs.load_wikitext_inputs(path, expected_sha256=digest,
            expected_tokenizer_identity=payload["tokenizer"],
            expected_model_identity=expected)


def test_v2_refuses_wrong_current_tokenizer(materializer, tmp_path):
    payload = _v2(materializer)
    path, digest = _write(tmp_path, payload)
    expected = copy.deepcopy(payload["tokenizer"])
    expected["files"]["tokenizer.json"]["sha256"] = "0" * 64
    expected["content_sha256"] = inputs.canonical_sha256({"files": expected["files"]})
    with pytest.raises(inputs.DSv4WikiTextInputsError, match="tokenizer"):
        inputs.load_wikitext_inputs(path, expected_sha256=digest,
            expected_tokenizer_identity=expected,
            expected_model_identity=payload["model"])


@pytest.mark.parametrize("mutation,error", [
    ("token", "token values"), ("ppl_token", "token values"),
    ("token_range", "invalid token"), ("start", "window selection"),
    ("train_fingerprint", "dataset identity"),
    ("ppl_corpus", "dataset identity"), ("dataset_version", "producer version"),
    ("corpus_construction", "corpus construction"),
    ("extra_field", "fields are not closed"),
])
def test_v2_refuses_changed_values_and_metadata(materializer, tmp_path, mutation, error):
    payload = _v2(materializer)
    if mutation == "token":
        payload["full_kl"]["token_ids"][0][0] ^= 1
    elif mutation == "ppl_token":
        payload["ppl"]["token_ids"][0] ^= 1
    elif mutation == "token_range":
        payload["full_kl"]["token_ids"][0][0] = 32
    elif mutation == "start":
        payload["full_kl"]["selection"]["starts"][0] += 1
    elif mutation == "train_fingerprint":
        payload["full_kl"]["dataset"]["fingerprint"] = "changed"
    elif mutation == "ppl_corpus":
        payload["ppl"]["dataset"]["corpus_sha256"] = "0" * 64
    elif mutation == "dataset_version":
        payload["datasets_distribution"]["version"] = "4.8.3"
    elif mutation == "corpus_construction":
        payload["corpus_construction"]["join_separator"] = "\n"
    else:
        payload["unreviewed"] = True
    _reseal(payload)
    path, digest = _write(tmp_path, payload)
    with pytest.raises(inputs.DSv4WikiTextInputsError, match=error):
        inputs.load_wikitext_inputs(path, expected_sha256=digest,
            expected_tokenizer_identity=payload["tokenizer"],
            expected_model_identity=payload["model"])


def test_resealed_changed_tokens_cannot_keep_reviewed_file_identity(materializer, tmp_path):
    payload = _v2(materializer)
    path, original_digest = _write(tmp_path, payload)
    payload["full_kl"]["token_ids"][0][0] ^= 1
    payload["full_kl"]["token_ids_tensor_sha256"] = inputs._tensor_sha256(
        torch.tensor(payload["full_kl"]["token_ids"]))
    _reseal(payload)
    _write(tmp_path, payload)
    with pytest.raises(inputs.DSv4WikiTextInputsError, match="file SHA256"):
        inputs.load_wikitext_inputs(path, expected_sha256=original_digest,
            expected_tokenizer_identity=payload["tokenizer"],
            expected_model_identity=payload["model"])


def test_model_identity_ignores_quantization_config_but_binds_token_domain(materializer):
    model, _ = materializer
    before = inputs.wikitext_model_identity(model)
    path = model / "config.json"
    config = json.loads(path.read_text())
    config["quantization_config"] = {"quant_method": "tessera"}
    config["dtype"] = "bfloat16"
    path.write_text(json.dumps(config))
    assert inputs.wikitext_model_identity(model) == before
    config["text_config"]["vocab_size"] += 1
    path.write_text(json.dumps(config))
    assert inputs.wikitext_model_identity(model) != before


def test_v2_requires_current_model(materializer, tmp_path):
    payload = _v2(materializer)
    path, digest = _write(tmp_path, payload)
    with pytest.raises(inputs.DSv4WikiTextInputsError, match="model identity"):
        inputs.load_wikitext_inputs(path, expected_sha256=digest,
            expected_tokenizer_identity=payload["tokenizer"])


def test_v1_reader_continuity_and_fixed_values(materializer, tmp_path, monkeypatch):
    payload = _v2(materializer)
    payload.pop("model")
    payload.pop("source_config")
    payload["schema"] = inputs.DSV4_WIKITEXT_INPUTS_SCHEMA
    for key, value in {
        "TOKENIZER_IDENTITY_SHA256": payload["tokenizer"]["content_sha256"],
        "TOKENIZER_VOCAB_SIZE": 32,
        "FULL_KL_TOTAL_TOKENS": payload["full_kl"]["dataset"]["total_tokens"],
        "FULL_KL_STARTS": tuple(payload["full_kl"]["selection"]["starts"]),
        "FULL_KL_TOKEN_IDS_TENSOR_SHA256": payload["full_kl"]["token_ids_tensor_sha256"],
        "PPL_TOTAL_TOKENS": payload["ppl"]["dataset"]["total_tokens"],
        "PPL_TOKEN_IDS_SHA256": payload["ppl"]["token_ids_sha256"],
    }.items():
        monkeypatch.setattr(inputs, key, value)
    _reseal(payload)
    path, _ = _write(tmp_path, payload)
    assert inputs.load_wikitext_inputs(path,
        expected_tokenizer_identity=payload["tokenizer"]) == inputs.load_dsv4_wikitext_inputs(
            path, expected_tokenizer_identity=payload["tokenizer"])
    payload["ppl"]["token_ids"][0] ^= 1
    payload["ppl"]["token_ids_sha256"] = inputs.canonical_sha256(payload["ppl"]["token_ids"])
    _reseal(payload)
    _write(tmp_path, payload)
    with pytest.raises(inputs.DSv4WikiTextInputsError, match="token values"):
        inputs.load_wikitext_inputs(path, expected_tokenizer_identity=payload["tokenizer"])


def test_v2_ppl_routes_bound_inputs_without_tokenizer(materializer, tmp_path):
    from tools import measure_vllm_wikitext_ppl as ppl
    payload = _v2(materializer)
    path, digest = _write(tmp_path, payload)
    args = SimpleNamespace(model=str(materializer[0]), wikitext_inputs=str(path),
                           wikitext_inputs_sha256=digest)
    ids, evidence = ppl._load_measurement_ids(args,
        tokenizer_attestation=payload["tokenizer"])
    assert ids == payload["ppl"]["token_ids"]
    assert evidence["corpus_sha256"] == payload["ppl"]["dataset"]["corpus_sha256"]
    args.wikitext_inputs_sha256 = None
    with pytest.raises(inputs.DSv4WikiTextInputsError, match="independent.*SHA256"):
        ppl._load_measurement_ids(args, tokenizer_attestation=payload["tokenizer"])


def test_mapping_and_local_config_normalization_agree(materializer):
    model, _ = materializer
    assert inputs.normalize_wikitext_model_identity(json.loads(
        (model / "config.json").read_text())) == inputs.wikitext_model_identity(model)


def test_overlap_automaton_matches_brute_force_without_crossing_windows():
    import random
    import runpy
    namespace = runpy.run_path(str(Path(__file__).resolve().parents[1] /
        'docs/experiments/glm_model_wikitext_inputs_20260908/audit_overlap.py'))
    cls = namespace['Substrings']
    assert cls([[1, 2], [3, 4]]).match([1, 2, 3, 4])['longest_match']['length'] == 2
    rng = random.Random(7)
    for _ in range(80):
        rows = [[rng.randrange(7) for _ in range(15)] for _ in range(4)]
        query = [rng.randrange(7) for _ in range(24)]
        expected = 0
        for row in rows:
            for a in range(len(query)):
                for b in range(len(row)):
                    count = 0
                    while a+count < len(query) and b+count < len(row) and query[a+count] == row[b+count]:
                        count += 1
                    expected = max(expected, count)
        assert cls(rows).match(query)['longest_match']['length'] == expected
