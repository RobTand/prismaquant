"""Local tokenizer file identity shared by probes and teacher payloads."""
from __future__ import annotations

import os
from pathlib import Path

from .digests import DIRECT_UTF8_STRICT, file_sha256hex


TOKENIZER_IDENTITY_SCHEMA = "prismaquant.tokenizer_identity/1"

_TOKENIZER_FILENAMES = (
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)


def build_tokenizer_identity(model_dir: str | os.PathLike) -> dict[str, object]:
    """Bind local tokenizer files and separate chat-template bytes."""
    root = Path(model_dir).resolve(strict=True)
    files: dict[str, dict[str, object]] = {}
    paths = [root / name for name in _TOKENIZER_FILENAMES]
    for directory in ("chat_templates", "additional_chat_templates"):
        paths.extend(sorted((root / directory).rglob("*")))
    for path in paths:
        if path.is_file():
            files[path.relative_to(root).as_posix()] = {
                "bytes": int(path.stat().st_size),
                "sha256": file_sha256hex(path),
            }
    if not files:
        raise ValueError(f"no tokenizer files found under {root}")
    value_bearing = {"files": files}
    return {
        "schema": TOKENIZER_IDENTITY_SCHEMA,
        "content_sha256": DIRECT_UTF8_STRICT.sha256(value_bearing),
        **value_bearing,
    }

