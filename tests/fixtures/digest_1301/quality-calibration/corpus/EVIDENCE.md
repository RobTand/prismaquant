# Corpus assigned slice

Revision scope: corpus fixtures, corpus tests and this local ledger/evidence only. Acceptance/shared files and other consumers were not changed by this revision.

## Deployment and behavior

The existing repository-relative `profile_source_owner("digests")` loader remains unchanged. Fresh `python -S` subprocess coverage imports both actual consumers outside the repository working directory while refusing torch and prismaquant package imports. No new loader framework, JSON recipe or single-file deployment exception was added.

Three SHA operations use the owner: builder UTF8 text -> text_sha256hex; builder already-encoded JSON payload -> bytes_sha256hex; materializer already-written corpus bytes -> bytes_sha256hex. Four local serialization sites remain recipe gaps: sorted spaced UTF8/allow_nan JSON, two unsorted JSONL encoders, and the unsorted indented index encoder.

## Minimal independent originals

Fixtures now contain only verbatim original `_sha256_text`, `_json_digest`, `write_jsonl`, and materializer `join_rows`/`main` blocks: 3,016 bytes total instead of 18,345 bytes of complete modules. They are extracted from the verified recorded-base blobs, not from the modified consumers. Tests pin the block-file SHA256s and function roster, supply stdlib context explicitly, and deep-copy unchanged MANIFEST/default constants from the current consumer. AST comparison against the recorded base confirmed those constants unchanged. The blocks are not standalone modules or full Git blobs.

Differential tests compare original/current JSONL bytes, Unicode, NaN, ordering/spacing, digest/error behavior, corpus bytes and complete index bytes. The materializer uses a deterministic datasets double and checks file publication before hashing. No live dataset/network integration is claimed.

## Historical census discrepancy only (immutable registry is correct)

Inspected with `git rev-parse`, `git cat-file`, `git log --all --find-object` and blob-to-blob `git diff`:

- Recorded base `6d8288adc94b3c4b624c143ed372fb6513f32784` has builder blob `55be80fdb931621b0dcecb5bfd144d23691ff55a`; worktree HEAD `8bc0b5af8c69af352b559dccda99bd4e6fa8b418` has the same builder blob.
- Historical census builder blob `0ee2e204d5a7443ea6616f460d06893265b987dc` exists, and is verified at commit `f4436b7ca7e34bd7cb1cafa1f49e95d1365afb0e`. It must NOT be paired with the recorded base commit.
- Commit `a0f6d66368e54cca6823cd5f14277d3a64542f8d` has the newer builder blob. The blob diff adds positive target-token validation outside the fixture functions. Exact source-segment comparison confirmed `_sha256_text`, `_json_digest`, `write_jsonl` and MANIFEST identical across the historical census and base blobs. This historical census discrepancy does not change the tested recipes. The immutable research registry `tests/fixtures/digest_parent_completion_1301.json` correctly uses builder blob `55be80fdb931621b0dcecb5bfd144d23691ff55a` at base `6d8288adc94b3c4b624c143ed372fb6513f32784`, with JSONL original IDs at lines 357/359; it has no attribution defect.
- Materializer blob at recorded base and HEAD is `8ee8c0fe7bf515f552681c63d7f16e48f88d78ab`, matching its census blob.

Every ledger row now has normalized `source_commit`, `source_blob`, `owner_route`, `fixtures`, and `tests` fields. The source pair and original IDs match the immutable registry, including JSONL sites 357/359. Historical census blobs remain separate. Builder rows explicitly expose only the historical census discrepancy and its separately verified historical commit. `test_ledger_provenance` checks every original ID and source pair against the immutable registry rather than asserting the former registry/census confusion. No acceptance or immutable registry edits were made.

## Final revision check

`python -S -m unittest discover -s tests/test_digest_1301/quality-calibration -p test_corpus.py -v` passed all 6 tests (exit 0). This is the final revision check; no broader acceptance suite was run for this revision.

[Test evidence {"artifacts":{"stderr":{"digest":"e4a320df1aa98afe8822be97ce72100a396578a697cd86dd1632668cf11e8808","size":632},"stdout":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0}},"operation":"6456ba71b5b75499806e47ac2752bd1e767cf2ed9b006a1011130feb1b2ca08f","receipt":"2f8c20b2c8acc787029c3f39bd7f0d5f6de0f5bd4eb0827048a9e00754f50ef7"}]
