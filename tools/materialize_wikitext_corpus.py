#!/usr/bin/env python3
"""Materialize the WikiText corpus a gold run scores, as plain UTF-8 text.

WHY THIS EXISTS
    The gold KL and PPL tools built their corpus with `datasets.load_dataset`,
    which drags in pyarrow + pandas + datasets.  The measurement containers do
    not carry those (`gridbook:0.8.6-clean-dde15e0`, retired 2026-09-02, had
    none of the three; see archive/gridbook_lane_2026-09-02/), and
    the fix must not be "pip install them at measurement time": that mutates the
    exact serving stack the number is fingerprinted against, and needs network
    on a path that should be offline and reproducible.

    So the corpus is resolved ONCE, here, on a host that has `datasets`, and
    written as the joined text the tools would have built themselves.  The gold
    tools then read bytes and hash them.

WHY IT IS ALSO BETTER PROVENANCE
    `--corpus-text-file` makes teacher, student and PPL prove they scored the
    same bytes -- a sha256 over the actual text -- instead of each independently
    calling load_dataset and trusting that two resolutions agree.  This is the
    same shape as the DSv4 offline path (tools/prepare_dsv4_wikitext_inputs.py),
    generalized to the ordinary lane.

    The join rule below is copied from the two tools verbatim: drop rows whose
    text is empty or whitespace, join the rest with a blank line.  If you change
    it here, the corpus_sha256 changes and every previously recorded gold number
    stops being comparable -- which is exactly what should happen.

Usage:
    python3 tools/materialize_wikitext_corpus.py --out-dir <dir> [--split train test]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

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

WIKITEXT_DATASET = "wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


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
            "sha256": bytes_sha256hex(raw),
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


if __name__ == "__main__":
    raise SystemExit(main())
