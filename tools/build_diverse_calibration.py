#!/usr/bin/env python3
"""Build PrismaQuant diverse-v1 text calibration JSONL.

The first JSONL row is a manifest. All following rows are ``{"text": ...}``
records so the existing probe loader can consume the file unchanged.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

if __package__:
    from prismaquant.digests import bytes_sha256hex
else:
    # Stdlib-only script use binds the same tree's owner by file path, so the
    # host needs no installed package. digests.py is stdlib-only: register the
    # file module under a name carrying that path's exact bytes, so two
    # checkouts in one process never share a cache entry.
    _digest_owner_path = Path(__file__).resolve().parents[1] / "prismaquant" / "digests.py"
    _digest_owner_name = "_prismaquant_standalone_digest_owner_" + os.fsencode(_digest_owner_path).hex()
    if _digest_owner_name not in sys.modules:
        _digest_owner_spec = importlib.util.spec_from_file_location(_digest_owner_name, _digest_owner_path)
        _digest_owner_module = importlib.util.module_from_spec(_digest_owner_spec)
        sys.modules[_digest_owner_name] = _digest_owner_module
        try:
            _digest_owner_spec.loader.exec_module(_digest_owner_module)
        except BaseException:
            del sys.modules[_digest_owner_name]
            raise
    bytes_sha256hex = sys.modules[_digest_owner_name].bytes_sha256hex


# Repo-relative, and overridable by the same knob run-pipeline.sh reads, so the
# builder and the pipeline agree on where corpora live without either of them
# naming a developer's home directory.
DEFAULT_OUTPUT = str(
    Path(os.environ.get(
        "PRISMAQUANT_CALIBRATION_DIR",
        Path(__file__).resolve().parents[1] / "calibration",
    )) / "diverse-v1.jsonl"
)
# No default: the corpus is tokenized to a target length, so it is only
# reusable across models whose tokenizers agree. Guessing one silently ties the
# corpus to a model the caller never named.
DEFAULT_TOKENIZER = None
DEFAULT_ROWS = 256
DEFAULT_TARGET_TOKENS = 4096
DEFAULT_SEED = 20260509

MANIFEST: dict[str, Any] = {
    "schema": "prismaquant.calibration.diverse_v1",
    "version": "diverse-v1",
    "seed": DEFAULT_SEED,
    "row_count": DEFAULT_ROWS,
    "target_tokens": DEFAULT_TARGET_TOKENS,
    "mix": {"prose": 0.40, "code": 0.20, "math": 0.20, "multilingual": 0.20},
    "sources": {
        "prose": [
            {
                "dataset": "HuggingFaceFW/fineweb-edu",
                "config": "sample-10BT",
                "split": "train",
                "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
                "field": "text",
            },
        ],
        "code": [
            {
                "dataset": "bigcode/starcoderdata",
                "configs": ["python", "c", "cpp"],
                "split": "train",
                "revision": "9fc30b578cedaec69e47302df72cf00feed7c8c4",
                "note": "Primary plan source; currently gated on this host.",
            },
            {
                "dataset": "HSH-Intelligence/github-code-corpus-sample",
                "config": "default",
                "split": "train",
                "revision": "e77fb393858a90005371bdb28e9a5ff4fcd81eb8",
                "field": "code",
                "language_filter": ["python", "c", "cpp", "c++"],
            },
        ],
        "math": [
            {
                "dataset": "EleutherAI/proof-pile-2",
                "config": "algebraic-stack",
                "split": "train",
                "revision": "901a9273a770e9d4138c5ddd91802f9c5c6cdc4b",
                "note": "Primary plan source; installed datasets rejects script datasets.",
            },
            {
                "dataset": "open-web-math/open-web-math",
                "config": "default",
                "split": "train",
                "revision": "fde8ef8de2300f5e778f56261843dab89f230815",
                "field": "text",
            },
        ],
        "multilingual": [
            {
                "dataset": "Muennighoff/flores200",
                "config": "all",
                "split": "dev",
                "revision": "9660f10339cfc46a369a4d95fccb301a739c3fa8",
                "note": "FLORES200 fallback for unavailable Helsinki-NLP/flores200 id; installed datasets rejects script datasets.",
            },
            {
                "dataset": "hlillemark/flores200_8_val_test",
                "split": "val",
                "revision": "4958a03c342fa1bbe9a38fd4bba6e423254f62f5",
                "streaming": False,
                "languages": [
                    "eng_Latn",
                    "dan_Latn",
                    "deu_Latn",
                    "fra_Latn",
                    "spa_Latn",
                    "ita_Latn",
                    "nld_Latn",
                    "por_Latn",
                ],
            },
        ],
    },
}


def quota_counts(total_rows: int) -> dict[str, int]:
    """Return deterministic row counts for the 40/20/20/20 mix."""
    if total_rows < 5:
        raise ValueError("total_rows must be at least 5")
    prose = round(total_rows * 0.40)
    code = round(total_rows * 0.20)
    math = round(total_rows * 0.20)
    multilingual = total_rows - prose - code - math
    return {
        "prose": int(prose),
        "code": int(code),
        "math": int(math),
        "multilingual": int(multilingual),
    }


def _sha256_text(text: str) -> str:
    return bytes_sha256hex(text.encode("utf-8"))


def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return bytes_sha256hex(payload)


def _load_dataset_stream(source: dict[str, Any]):
    from datasets import load_dataset

    kwargs: dict[str, Any] = {
        "split": source.get("split", "train"),
        "streaming": bool(source.get("streaming", True)),
        "revision": source["revision"],
    }
    config = source.get("config")
    if config and config != "default":
        return load_dataset(source["dataset"], config, **kwargs)
    return load_dataset(source["dataset"], **kwargs)


def _field(row: dict[str, Any], name: str) -> str | None:
    value = row.get(name)
    return value if isinstance(value, str) and value.strip() else None


def _first_text(row: dict[str, Any]) -> str | None:
    for key in ("text", "content", "code", "document", "source", "target"):
        value = _field(row, key)
        if value:
            return value
    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value
    return None


def _iter_plain_texts(source: dict[str, Any]) -> Iterator[str]:
    field = source.get("field")
    for row in _load_dataset_stream(source):
        text = _field(row, field) if field else _first_text(row)
        if text:
            yield text


def _iter_code_texts(source: dict[str, Any]) -> Iterator[str]:
    allowed = {
        str(lang).lower().replace("++", "pp")
        for lang in source.get("language_filter", [])
    }
    for row in _load_dataset_stream(source):
        language = str(row.get("language", "")).lower().replace("++", "pp")
        if allowed and language and language not in allowed:
            continue
        text = _field(row, source.get("field", "code")) or _first_text(row)
        if text:
            yield text


def _iter_xnli_texts(source: dict[str, Any]) -> Iterator[str]:
    languages = [str(lang) for lang in source.get("languages", [])]
    for row in _load_dataset_stream(source):
        parts: list[str] = []
        premise = row.get("premise")
        if isinstance(premise, dict):
            for lang in languages:
                value = premise.get(lang)
                if isinstance(value, str) and value.strip():
                    parts.append(f"[{lang}] {value}")
        hypothesis = row.get("hypothesis")
        if isinstance(hypothesis, dict):
            lang_list = hypothesis.get("language")
            translations = hypothesis.get("translation")
            if isinstance(lang_list, list) and isinstance(translations, list):
                by_lang = {
                    str(lang): text
                    for lang, text in zip(lang_list, translations)
                    if isinstance(text, str) and text.strip()
                }
                for lang in languages:
                    value = by_lang.get(lang)
                    if value:
                        parts.append(f"[{lang}] {value}")
        if parts:
            yield "\n".join(parts)


def _iter_parallel_texts(source: dict[str, Any]) -> Iterator[str]:
    languages = {str(lang) for lang in source.get("languages", [])}
    for row in _load_dataset_stream(source):
        src_lang = row.get("source_lang")
        tgt_lang = row.get("target_lang")
        source_text = row.get("source")
        target_text = row.get("target")
        if isinstance(source_text, str) and (
            not languages or str(src_lang) in languages
        ):
            yield source_text
        if isinstance(target_text, str) and (
            not languages or str(tgt_lang) in languages
        ):
            yield target_text


def iter_texts_with_fallbacks(domain: str, sources: list[dict[str, Any]]) -> Iterator[tuple[str, str]]:
    """Yield ``(text, dataset_id)`` from the first usable source."""
    errors: list[str] = []
    for source in sources:
        dataset_id = str(source["dataset"])
        try:
            if domain == "code":
                iterator = _iter_code_texts(source)
            elif domain == "multilingual" and dataset_id == "facebook/xnli":
                iterator = _iter_xnli_texts(source)
            elif domain == "multilingual" and "flores" in dataset_id.lower():
                iterator = _iter_parallel_texts(source)
            else:
                iterator = _iter_plain_texts(source)
            yielded = False
            for text in iterator:
                yielded = True
                yield text, dataset_id
            if yielded:
                return
        except Exception as exc:
            errors.append(f"{dataset_id}: {type(exc).__name__}: {exc}")
    joined = "; ".join(errors) if errors else "no rows yielded"
    raise RuntimeError(f"no usable source for {domain}: {joined}")


def _encode(tokenizer, text: str) -> list[int]:
    return [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]


def pack_token_windows(
    texts: Iterable[tuple[str, str]],
    tokenizer,
    *,
    domain: str,
    needed: int,
    target_tokens: int,
    seed: int,
    min_tokens: int | None = None,
) -> list[dict[str, str]]:
    """Pack or window source texts into approximately target-token rows."""
    if needed <= 0:
        return []
    min_tokens = int(min_tokens or max(32, target_tokens // 8))
    rng = random.Random(seed)
    eos = getattr(tokenizer, "eos_token_id", None)
    eos = 0 if eos is None else int(eos)
    records: list[dict[str, str]] = []
    buffer: list[int] = []

    def emit(ids: list[int]) -> None:
        text = tokenizer.decode(ids[:target_tokens], skip_special_tokens=True).strip()
        if text:
            records.append({"text": text})

    for text, _source in texts:
        ids = _encode(tokenizer, text)
        if not ids:
            continue
        if len(ids) >= target_tokens:
            start = rng.randint(0, len(ids) - target_tokens)
            emit(ids[start:start + target_tokens])
        else:
            buffer.extend(ids)
            buffer.append(eos)
        while len(buffer) >= target_tokens:
            emit(buffer[:target_tokens])
            buffer = buffer[target_tokens:]
        if len(records) >= needed:
            return records[:needed]
    if len(records) < needed and len(buffer) >= min_tokens:
        emit(buffer)
    if len(records) < needed:
        raise RuntimeError(f"only built {len(records)}/{needed} {domain} rows")
    return records[:needed]


def build_records(tokenizer, *, rows: int, target_tokens: int, seed: int) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for domain, needed in quota_counts(rows).items():
        print(f"[diverse-calib] collecting {domain}: {needed} rows", flush=True)
        texts = iter_texts_with_fallbacks(domain, MANIFEST["sources"][domain])
        out.extend(pack_token_windows(
            texts,
            tokenizer,
            domain=domain,
            needed=needed,
            target_tokens=target_tokens,
            seed=seed + len(out),
        ))
    random.Random(seed).shuffle(out)
    return out


def write_jsonl(
    path: Path,
    records: list[dict[str, str]],
    *,
    tokenizer_path: str,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    seed: int = DEFAULT_SEED,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(MANIFEST)
    manifest["row_count"] = len(records)
    manifest["target_tokens"] = int(target_tokens)
    manifest["seed"] = int(seed)
    manifest["tokenizer"] = tokenizer_path
    manifest["records_sha256"] = _json_digest([
        _sha256_text(str(record.get("text", "")))
        for record in records
    ])
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"__manifest__": manifest}, ensure_ascii=False) + "\n")
        for record in records:
            fh.write(json.dumps({"text": record["text"]}, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                        required=DEFAULT_TOKENIZER is None,
                        help="model dir or HF id whose tokenizer defines the token budget")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    from transformers import AutoTokenizer

    local_only = Path(args.tokenizer).exists()
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=local_only,
    )
    records = build_records(
        tokenizer,
        rows=int(args.rows),
        target_tokens=int(args.target_tokens),
        seed=int(args.seed),
    )
    write_jsonl(
        Path(args.output),
        records,
        tokenizer_path=str(args.tokenizer),
        target_tokens=int(args.target_tokens),
        seed=int(args.seed),
    )
    print(f"[diverse-calib] wrote {len(records)} rows to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    # Some HF streaming datasets keep HTTP worker state alive past the final
    # yielded row. After the JSONL is closed and the status line is flushed,
    # skip interpreter teardown so those background retries cannot convert a
    # successful build into a fatal finalization error.
    import os

    os._exit(main())
