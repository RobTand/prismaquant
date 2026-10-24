def join_rows(rows) -> str:
    """The tools' corpus rule, in one place."""
    return "\n\n".join(
        row["text"]
        for row in rows
        if isinstance(row.get("text"), str) and row["text"].strip()
    )

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default="/home/rob/dq-runs/hf_datasets")
    ap.add_argument("--split", nargs="+", default=["train", "test"])
    ap.add_argument("--revision", default=WIKITEXT_REVISION)
    args = ap.parse_args()

    from datasets import load_dataset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, dict] = {}

    for split in args.split:
        ds = load_dataset(
            WIKITEXT_DATASET,
            WIKITEXT_CONFIG,
            split=split,
            cache_dir=args.cache_dir,
            revision=args.revision,
        )
        text = join_rows(ds)
        raw = text.encode("utf-8")
        path = out_dir / f"wikitext-2-raw-v1.{split}.txt"
        path.write_bytes(raw)
        entry = {
            "path": str(path),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "dataset": WIKITEXT_DATASET,
            "config": WIKITEXT_CONFIG,
            "revision": args.revision,
            "split": split,
            "datasets_fingerprint": getattr(ds, "_fingerprint", None),
            "n_rows_kept": sum(
                1 for row in ds
                if isinstance(row.get("text"), str) and row["text"].strip()
            ),
        }
        index[split] = entry
        print(f"[corpus] {split}: {entry['bytes']} bytes  "
              f"sha256 {entry['sha256'][:16]}  rows {entry['n_rows_kept']}")

    (out_dir / "corpus_index.json").write_text(json.dumps(index, indent=1))
    return 0
