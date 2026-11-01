"""Calibration input classification at the real shared loader boundary."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from prismaquant.calibration_data import load_calibration


class _Tokenizer:
    eos_token_id = 0

    def __call__(self, text, *, return_tensors="pt", truncation=False):
        return SimpleNamespace(input_ids=torch.tensor([[ord(c) % 31 + 1 for c in text]]))


@pytest.mark.parametrize("source", ["audit_calibration", "audit-owner/tiny-calibration"])
def test_hub_dataset_ids_reach_the_dataset_adapter(tmp_path, monkeypatch, source):
    monkeypatch.chdir(tmp_path)
    calls = []

    def load_dataset(name, **options):
        calls.append((name, options))
        return [{"text": "abcdefghijklmnopqrstuvwxyz"}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    ids = load_calibration(_Tokenizer(), source, n_samples=1, seqlen=8)
    assert ids.shape == (1, 8)
    assert calls == [(source, {"split": "train", "streaming": True})] * 2


@pytest.mark.parametrize("kind", ["absolute", "explicit_relative", "parent_relative", "jsonl", "txt"])
def test_missing_explicit_local_sources_refuse_before_dataset_access(tmp_path, monkeypatch, kind):
    monkeypatch.chdir(tmp_path)
    sources = {
        "absolute": str(tmp_path / "missing" / "corpus"),
        "explicit_relative": "./missing/corpus",
        "parent_relative": "../pq-audit-missing/corpus",
        "jsonl": "missing.jsonl",
        "txt": "missing.txt",
    }

    def load_dataset(*args, **kwargs):
        pytest.fail("an explicitly local missing corpus reached the dataset adapter")

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    with pytest.raises(FileNotFoundError, match="calibration file not found"):
        load_calibration(_Tokenizer(), sources[kind], n_samples=1, seqlen=8)


@pytest.mark.parametrize("name", ["calibration.jsonl", "dataset"])
def test_existing_local_corpus_keeps_local_parsing(tmp_path, monkeypatch, name):
    path = tmp_path / name
    path.write_text(json.dumps({"text": "abcdefghijklmnopqrstuvwxyz"}) + "\n")

    def load_dataset(*args, **kwargs):
        pytest.fail("an existing local corpus reached the dataset adapter")

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    ids = load_calibration(_Tokenizer(), str(path), n_samples=1, seqlen=8)
    assert ids.shape == (1, 8)


def test_existing_directory_keeps_the_dataset_adapter_boundary(tmp_path, monkeypatch):
    directory = tmp_path / "dataset"
    directory.mkdir()
    calls = []

    def load_dataset(name, **options):
        calls.append(name)
        return [{"text": "abcdefghijklmnopqrstuvwxyz"}]

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=load_dataset))
    ids = load_calibration(_Tokenizer(), str(directory), n_samples=1, seqlen=8)
    assert ids.shape == (1, 8)
    assert calls == [str(directory)] * 2
