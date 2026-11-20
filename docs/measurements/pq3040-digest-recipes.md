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

Flux CPU run `f3cQKH4iXrP` at `dcd757813` (gate venv: Python 3.12.12,
torch 2.15.0+cpu): 557 passed, 8 skipped, job and payload exit 0. It ran
this test with the digest-profile, tensor-digest, guarded-read retention,
duplication, census, graph-impact and Bazel build-file tests.

## Follow-up: the parents' own calls (PQ #3031, #3033, #3036)

The #3040 calls do not overlap the three parents' ledgers on the retained
salvage branch: #3031 `joint-handoff/ledger.json` (15 calls), #3033
`research-reports/ledger.json` (26) and #3036 `t4-qualification/residuals.json`
(21). Main after #5188 covered 53 of the 62. Five JSON profiles complete them:
`DIRECT_ASCII_INDENT1_STRICT` (1 call), `DIRECT_ASCII_INDENT2_LAX_DEFAULT_STR`
(2), `INSERTION_ASCII_INDENT2_STRICT` (2), `INSERTION_ASCII_SPACED_LAX_DEFAULT_STR`
(3) and `INSERTION_ASCII_INDENT2_LAX_DEFAULT_STR` (1). The three `json.dump`
calls of #3033 use `INSERTION_ASCII_SPACED_LAX.dump`. None of the 62 is a hash
call. [Parent recipe ledger](../../tests/fixtures/digest_recipes_3031_3033_3036.json)
