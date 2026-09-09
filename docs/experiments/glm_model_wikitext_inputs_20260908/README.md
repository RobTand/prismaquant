# GLM offline evaluation inputs — 2026-09-08

The original GLM tokenizer now materializes the existing fixed evaluation
workloads through explicit `--input-schema model-v2`. The default DSv4 v1
mode and its fixed tokenizer/value refusals remain unchanged. No model forward,
GPU work, capture rewrite, or pricing-checkout mutation occurred.

The real input is
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/model-bound-wikitext-inputs-01/glm-wikitext-inputs.v2.json`,
SHA256 `eb7c36a6211796bdaf523c20ca3583cbe4794e9583a11f3ec5f124641b374652`.
Consumers must supply this independent file hash, exact tokenizer-file identity
and the normalized current model family/vocabulary. The shared normalization
accepts embedded config mappings or a local model directory. Quantization config
bytes are excluded from candidate pairing; original config bytes remain pinned
as input provenance. PPL accepts the binding with `--wikitext-inputs-sha256`.

Materialization used the existing canonical Salesforce WikiText cache at
revision `b08601e04326c79dfdd32d625aee71d232d685c3` with scoped
`datasets==4.6.0`. The train/test corpus hashes match the historical exact
corpus. The canonical cache fingerprints are `5d4fb603254a7a5b` and
`a46124b21ac53738`; v2 pins these separately from the alias-era DSv4 values.
An initial real attempt refused the unavailable offline alias; a second refused
the fingerprint mismatch. Both negative results are retained beside the final
successful action and no output was accepted from either.

The source token domain is `glm5_next` / `glm5_next_text`, vocabulary 154,880,
tokenizer identity
`1e1880528a8923b057dcd8b97e6fe1dc29e2feaebe4a04c85a9aa66c482f07d0`.
Filtered full-corpus tokenization with special tokens disabled yields 2,440,670
train and 289,569 test tokens. The eight 512-token train starts at Python
Random seed 42 are 466956, 104902, 1153556, 1027150, 936213, 585264, 429895,
2287433. Their tensor SHA is
`d6549876d0bdef6ab5c5882003447ab4cd1df9bb05f1b4190ae4ed6294f595aa`.
The 8,192-token test prefix has token-list SHA
`0b82970699299b3060b9def584a25756f7a361c454dedd4bde630827a6b7fe86`.

## Overlap with original calibration

This input is **not certified held out**. The audit reused the accepted
512 × 512, seed-0 calibration artifact, SHA
`9cd1fa129f249abd80d22efaeb8bc7e8b2d3b4252f173a8c6f2b2e496a4f8329`,
and verified its original fit/text digests through the existing loader. It did
not reconstruct the original draw again.

There are zero identical 512-token windows. However, evaluation train window 5
tokens `[280:512]` exactly match calibration window 176 tokens `[0:232]`:
232 contiguous tokens. Longest exact matches for all eight evaluation windows
are `[7, 7, 9, 7, 8, 232, 7, 4]`. The test PPL prefix's longest contiguous
match is 12 tokens. These are token-value observations; repeated text can match
without proving identical corpus origin. Original calibration includes empty
rows and tokenizer defaults, while evaluation filters rows and disables special
tokens, so start indices cannot establish overlap or disjointness. The retained
audit reports `fit_overlap_status=unverified` and `heldout_claim=false`.

`audit_overlap.py` uses a CPU suffix automaton with separators that prohibit
matches across calibration-window boundaries. A brute-force randomized test
checks its longest-match result, and every reported longest-match witness is
checked against exact token slices. The audit SHA is
`30a0261a304c16a7e914b293bd06b5749290711ab147eefb59752a5d7b35048e`.

## Validation

All tests and materialization ran through PrismaBuild on DL380G10, GPU disabled,
with native threads bounded to one and assigned affinity preserved. CPU tests
used four xdist workers / eight GiB; tokenization used two CPUs / six GiB;
overlap used one CPU / four GiB. No throughput or GPU performance claim follows.

- Regression `605f436dd29e`: explicit v2 materialization failed on the absent
  API; default v1 correctly refused the non-DSv4 tokenizer (1 failed, 1 passed).
- Contract/PPL/overlap and architecture checks `2e102b37d3a8`: 42 passed,
  56 warnings, no skips. After canonical fingerprint binding,
  `9ab97a04432c`: 23 targeted tests passed, 56 warnings, no skips.
- Real materialization `2fe5cc25aeb1`: exit 0, exact output hash verified.
- Actual calibration overlap `0b2242a1fb5d`: exit 0, exact report hash verified.

`cpu-cas-source-audit.json` records terminal states, stdout hashes, CAS
claim/payload checks and sealed Git source snapshots. Final tested and
materialized implementation files match the committed files. CAS tooling's
claim checks do not independently verify worker attestations. The v2 input is
ready for the separate teacher/served-quality path; it is not a quality result.
