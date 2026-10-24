def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

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
