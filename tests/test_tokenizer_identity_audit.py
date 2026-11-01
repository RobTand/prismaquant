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


@pytest.mark.parametrize("relative_path", [
    "chat_template.jinja", "chat_template.json", "chat_templates/default.jinja",
    "chat_templates/tool_use.jinja", "chat_templates/nested/custom.jinja",
    "additional_chat_templates/default.jinja", "additional_chat_templates/tool_use.jinja",
])
def test_template_add_change_and_delete_update_identity(tmp_path, relative_path):
    (tmp_path / "tokenizer.json").write_bytes(b"{}")
    original = build_tokenizer_identity(tmp_path)
    template = tmp_path / relative_path
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text("{{ messages[0]['content'] }}")
    added = build_tokenizer_identity(tmp_path)
    assert relative_path in added["files"]
    assert added["content_sha256"] != original["content_sha256"]
    assert teacher.tokenizer_identity(tmp_path) == added
    template.write_text("{{ 'changed' }}")
    changed = build_tokenizer_identity(tmp_path)
    assert changed["content_sha256"] != added["content_sha256"]
    assert teacher.tokenizer_identity(tmp_path) == changed
    template.unlink()
    assert build_tokenizer_identity(tmp_path) == original
