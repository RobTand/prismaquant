"""Tokenizer identity keeps teacher values and binds calibration templates."""
from __future__ import annotations

import hashlib
import json

import pytest

from prismaquant.tokenizer_identity import build_tokenizer_identity
from tools import full_kl_teacher_payload as teacher


def test_legacy_teacher_tokenizer_record_keeps_exact_values(tmp_path):
    legacy_files = {
        "added_tokens.json": b"{}",
        "merges.txt": b"a b\n",
        "sentencepiece.bpe.model": b"sentencepiece",
        "special_tokens_map.json": b'{"unk_token":"[UNK]"}',
        "spiece.model": b"spiece",
        "tokenizer.json": b'{"vocab":{}}',
        "tokenizer.model": b"model",
        "tokenizer_config.json": b'{"chat_template":"{{ messages }}"}',
        "vocab.json": b'{"a":0}',
    }
    files = {}
    for name, raw in legacy_files.items():
        (tmp_path / name).write_bytes(raw)
        files[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    raw = json.dumps({"files": files}, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=False, allow_nan=False).encode("utf-8")
    expected = {"schema": "prismaquant.tokenizer_identity/1", "files": files,
                "content_sha256": hashlib.sha256(raw).hexdigest()}
    assert build_tokenizer_identity(tmp_path) == expected
    assert teacher.tokenizer_identity(tmp_path) == expected


def test_empty_tokenizer_refuses_at_both_boundaries(tmp_path):
    with pytest.raises(ValueError, match="no tokenizer files found"):
        build_tokenizer_identity(tmp_path)
    with pytest.raises(teacher.TeacherPayloadError, match="no tokenizer files found"):
        teacher.tokenizer_identity(tmp_path)
