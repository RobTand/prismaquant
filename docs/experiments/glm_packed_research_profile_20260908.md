# GLM packed-research allocation profile — 2026-09-08

Explicit selection of `glm_packed_research_sm121` restricts GLM routed logical
expert units and aggregate stack names matching `(^|\.)mlp\.experts(\.|$)` to
legal `TESSERA_E4M3_K1` rungs and plain `BF16`. Tessera E2M1, Tessera BF16-body
and unrelated scalar formats are removed by the actual candidate filter before
allocation or stack grouping. Dense and shared-expert names retain the inherited
`tessera_research_sm121` policy.

The shared `ServingFormatRule.allow_tessera_families` field unions with exact
`allow_formats`; explicit denials still win. Family declarations must be
canonical and use the existing family grammar. Candidate names pass the full
existing rung parser, including rate bounds. No static rung list, registry row
or wildcard syntax is added. Profiles without the field retain their existing
behavior and acquire no new Tessera import through this field.

This is allocation policy, not an encoding, cache, prefetch or hot-path change.
Shape checks, exact source-payload ceilings and per-unit serving-context
admission remain on the existing candidate path. No GPU performance or quality
improvement is claimed. The profile is opt-in, `emulation_only`, has no export
lane, inherits TP world size 1 and grants no per-role expert format capability.
It changes no producer pin, runtime contract, menu default or ship gate. Full
model serving and TP2 qualification remain separate work.

PB CPU validation used Python 3.12 on `dl380g10`, with OMP/MKL/OpenBLAS/NumExpr
threads bounded to one and GPU visibility disabled:

| Check | Result | PB action |
|---|---|---|
| New profile, existing serving profiles and Tessera candidate-context tests, 4 workers / 8 GiB | 118 passed; no skips | `1f7de9ed8fd697f296640c82d77f3e99bf682727d3d7f66067e3bbe9fd14b128` |
| Architecture and docs-staleness tests, 2 workers / 4 GiB | 19 passed; no skips | `5ebbead650a8e7a5165e8e0bf73c95d2bcc47322f81e09dbfe3c75f9ec6e43b5` |
| Compile shared runtime and new tests, 1 worker / 1 GiB | exit 0 | `d7221b2976a1da33ddb736604617425d4e505c51b4a49f8aaad94ac261cbe978` |

The new tests exercise `build_candidates` with real format specs, profile
filters and footprint accounting for logical rank-2 experts, rank-3 stacks and
aggregate roots at q256 rungs 256, 1281 and 2048. They check E2M1, BF16-body,
other scalar and unknown-format refusals, preserve dense readable families,
reject malformed family declarations, retain explicit denials, refuse a
4,095-column shape and enforce the FP8 source-byte ceiling. The context test
controls only the attestation answer and lane reporting; it verifies that two
units cannot borrow each other's context approval through the real candidate
filter. It is not native runtime qualification.

The first run, `1f8ef275706d7e0a972032ffd208593de659a5b7b610b57cf1b941f4dc03b8d2`,
had 117 passes and one incorrect test expectation: a TP1 window with 4,095
columns reaches the exact footprint's hard superblock refusal, not a TP-shard
mask. The corrected test asserts that existing refusal through `build_candidates`;
no production shape gate changed.

Evidence root:
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/packed-research-profile-01/`.
The exact commands are in `cpu-campaign.json`; action-prefix `.pb-action.json`,
`.stdout.log` and `.cas-verification.json` retain terminal/log/CAS evidence.
`verified-actions.json` records independently checked stdout digests, CAS
payload verification and byte equality of the runtime, profile, new tests and
architecture document against each executed Git snapshot. Signer attestation
verification is not asserted by those CAS checks. No GPU action was submitted.
