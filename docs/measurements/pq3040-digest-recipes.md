# Digest recipes for PQ #3040

The 73 calls come from the read-measurement ledger in `0559eba10`, run
`562fc73c`. The call coordinates and JSON options come from `6d8288adc`.
The JSON and binary owner recipes come from `42871a20a`. The ordered byte
streams reuse `chunks_sha256hex` on main at `207956b45`.
[Recipe ledger](../../tests/fixtures/digest_recipes_3040.json)

| Source recipe | Calls |
| --- | ---: |
| Sorted ASCII, indent 1, no fallback | 15 |
| Insertion order, ASCII, default spacing | 31 |
| Insertion order, ASCII, indent 1 | 13 |
| Insertion order, ASCII, indent 1, `default=str` | 2 |
| Insertion order, ASCII, indent 2 | 3 |
| Insertion order, ASCII, streaming indent 1 | 1 |
| Insertion order, UTF-8, streaming indent 2, reject nonfinite values | 2 |
| Native 32-byte SHA-256 | 1 |
| SHA-256 update after each write | 1 |
| SHA-256 of dtype and shape text followed by tensor or array bytes | 2 |
| SHA-256 with alternating JSON metadata and tensor bytes | 1 |
| SHA-256 updates across render callbacks | 1 |
| Total | 73 |

The test compares bytes, stream chunks, errors, and hashes with the
original call forms. The tensor feeds use views of contiguous storage.
The primitive-site scan exempts both digest owners. The census and
duplication baseline need no fixture changes.

```sh
python -m pytest -q tests/test_digest_recipes_3040.py
```

The Flux test run is pending.
