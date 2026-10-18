"""Local tokenizer file identity shared by probes and teacher payloads."""
from __future__ import annotations

import os
from pathlib import Path

from .digests import DIRECT_UTF8_STRICT, file_sha256hex


TOKENIZER_IDENTITY_SCHEMA = "prismaquant.tokenizer_identity/1"

_TOKENIZER_FILENAMES = (
    "added_tokens.json",
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
    """Bind the exact local files that define tokenization for this lane."""
    root = Path(model_dir).resolve(strict=True)
    files: dict[str, dict[str, object]] = {}
    for name in _TOKENIZER_FILENAMES:
        path = root / name
        if path.is_file():
            files[name] = {
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

