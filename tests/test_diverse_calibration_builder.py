from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_builder_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "build_diverse_calibration.py"
    spec = importlib.util.spec_from_file_location("build_diverse_calibration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeTokenizer:
    eos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(range(1, len(str(text).split()) + 1))

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"tok{int(i)}" for i in ids)


def test_quota_counts_sum_to_requested_mix():
    mod = _load_builder_module()

    assert mod.quota_counts(256) == {
        "prose": 102,
        "code": 51,
        "math": 51,
        "multilingual": 52,
    }
    assert sum(mod.quota_counts(17).values()) == 17


def test_calibration_mix_refuses_fewer_than_five_rows():
    import pytest

    mod = _load_builder_module()

    for rows in (-1, 0, 4):
        with pytest.raises(ValueError) as excinfo:
            mod.quota_counts(rows)
        assert str(excinfo.value) == "total_rows must be at least 5"
    assert mod.quota_counts(5) == {"prose": 2, "code": 1, "math": 1, "multilingual": 1}


def test_pack_token_windows_emits_exact_count_and_metadata():
    mod = _load_builder_module()
    tokenizer = _FakeTokenizer()
    texts = ((("word " * 20), "unit-test-source") for _ in range(10))

    records = mod.pack_token_windows(
        texts,
        tokenizer,
        domain="prose",
        needed=3,
        target_tokens=8,
        seed=123,
        min_tokens=6,
    )

    assert len(records) == 3
    assert set(records[0]) == {"text"}
    assert all(record["text"] for record in records)


def test_write_jsonl_starts_with_manifest_and_text_rows(tmp_path):
    mod = _load_builder_module()
    records = [
        {
            "text": "hello",
            "domain": "prose",
            "source": "unit",
            "token_count": 1,
            "sha256": "a" * 64,
        }
    ]
    out = tmp_path / "diverse-v1.jsonl"

    mod.write_jsonl(
        out,
        records,
        tokenizer_path="/tmp/tokenizer",
        target_tokens=128,
        seed=7,
    )

    lines = out.read_text().splitlines()
    manifest = json.loads(lines[0])["__manifest__"]
    row = json.loads(lines[1])
    assert manifest["schema"] == "prismaquant.calibration.diverse_v1"
    assert manifest["row_count"] == 1
    assert manifest["target_tokens"] == 128
    assert manifest["seed"] == 7
    assert len(manifest["records_sha256"]) == 64
    assert "text" not in json.loads(lines[0])
    assert row["text"] == "hello"
