# Tensor digest leaf #2638: scoped repair and revalidation

## Governing scope

Rob's 2026-10-23 clarification supersedes #2645's four-recipe interpretation:
keep both production-cache identities, add at most one tensor-owner recipe,
and refresh the consolidated site list against current main. This repair
keeps only `token_ids_int32_sha256`. The former four-site baseline reduction
and four-recipe validation claims are superseded, not current evidence.

The pre-repair source recheck used served main
`6d8288adc94b3c4b624c143ed372fb6513f32784`. The census now records the eight
in-scope calls, their `refresh_2540` references, current whole-file Git blob
identities and SHA-256s of AST function source segments under
`refresh_2540.verification.tensor_leaf_2638`. All five consumer function
segments matched the pre-PR implementations byte-for-byte, including where
surrounding files changed on main. This is a scoped refresh, not a claim that
the entire 2026-10-09 repository manifest was regenerated. Its original
`30b6231d` source identities remain historical; unrelated leaf records are
not rewritten. Integration onto a newer base must recheck these identities
and serialize the shared census/baseline edits. No Git rebase or publication
was performed here; TrustLadder owns those workflow operations.

## Consolidated site decisions

| Consumer | Decision and exact contract |
| --- | --- |
| `tessera_hessian.token_ids_sha256` | Delegate to the sole added owner recipe, `tensor_digests.token_ids_int32_sha256`. Cast each batch to int32 before its CPU transfer and fold batches in caller order, with no shape/dtype header or delimiter. Keep the owner's import lazy, as the former torch import was. |
| `stage_a_chain_seed.tensor_payload_sha256` | Retain the original implementation, including caller-sized element windows (64 MiB default), direct NumPy-buffer feeds, and one CPU chunk beside the device tensor. Neither a whole-host bytes recipe nor the shape/dtype-framed update owns this acquisition contract. |
| `production_weight_cache._cb_cache_tensor_identity` | Retain the original schema and 8 MiB flattened memoryview windows, including zero-byte handling. Do not materialize a second whole-payload bytes object. |
| `production_weight_cache._source_weight_value_identity` | Retain the original `(shape, hex)` result and recursive C-order traversal with at most 4,194,304 elements per contiguous CPU FP32 conversion, explicit little-endian f4 bytes, and chunk release. |
| `tessera_export_lane.hessian_capture_sha256` (four constructor/JSON calls) | Retain the sorted-JSON identity prefix, dtype/shape unit header, NUL framing, name order and ASCII unit-hexdigest fold. The eligible tensor-byte substep already calls `tensor_host_bytes`; no new helper or recipe is needed. Preserve capture-roster/provenance checks and the independent Tessera seal comparison. |

The three new retention decisions update their dispositions/cohorts and
summary counts in the census. Original source/contract evidence is preserved.
Export framing retains its existing independent-reference disposition under
#2543. The token call separately records its completed owner migration.

The consumer wrappers, scientific casts, seeds, sampling and schemas are
unchanged. Stage A, both weight-cache implementations and the capture function
are source-identical to their pre-PR versions. Tests bind retained function
source to the scoped refresh without requiring Git history or another
checkout. Independent goldens cover the multi-batch token fold and retained
payload, view and FP32 streams. No second tensor owner or JSON profile joins.
The primitive-digest ratchet removes only the token consumer: **304 → 303**
on this branch, rather than the rejected four-site reduction.

## Runtime integration required by the gates

The reported private-validator failure was reproduced with:

```sh
python -m pytest tests/test_tessera_lane_v12.py::test_v12_parses_and_keeps_the_launch_scope_on_the_record -q --basetemp="$TMPDIR/pytest"
```

Result before repair: one failure, naming missing
`tessera.serving.contract._evidence_smoke_record`.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"d65999972122e684e7a4880be108f3fd15d45c0f7ab9004911696ec63550f313","size":4484}},"operation":"542bf1d39a1ba31623f8c529a2096497ec1cb15cb17818c73bac9cad005ea298","receipt":"e59563d36f7c366edefc09157b15c8d841329d94e4a77916751da746eefe9a1d"}]

The repair takes main's public-metadata portion of `lane_eligibility`:
require `tessera.producer-metadata.v1` and named callable capabilities;
delegate grammar to `parse_smoke_record(record, control=..., where=...)`,
status/attribution derivation and lane requirements to their public owners.
Missing APIs, malformed records and disagreement still refuse. There is no
private fallback, copied grammar or serving-execution import. The v9 tests
patch the public API rather than stale private aliases; their assertions
and discovery are preserved. New tests cover capability/version refusal,
real producer grammar and metadata-only import boundaries.

The declared reader, producer, serving and frozen-domain pins are aligned
with main's API-only `aaa417003f0ee690e8ace138acd18df7f9531369` delivery,
not left pointing at an installation without that public API. Pin schema v2,
version 0.1.0 and contract SHA-256
`ee065629b081d913a0351e43160c5c6e1bd38fa628cafd51e756e9caf3bb334e`
are unchanged. Main's exporter/grammar fingerprints and audited rate counts
are unchanged. The existing serving-identity fixture changes only
`/pin/commit`; all gate answers must remain equal. No new runtime
qualification, compiled-cell waiver or admission policy is claimed.

## Verification

All pytest runs use external `$TMPDIR` scratch. CPU skips do not establish
CUDA behavior. No GPU or serving campaign was needed or run for this leaf.

### Named gates, baseline regeneration and documentation guards

```sh
cp tests/fixtures/duplication_baseline.json "$TMPDIR/baseline-before.json" && python tools/duplication_inventory.py --write-baseline && cmp "$TMPDIR/baseline-before.json" tests/fixtures/duplication_baseline.json && python -m pytest tests/test_tensor_digest_owner_recipes_2638.py tests/test_tensor_digests_1384.py tests/test_stage_a_chain_seed.py tests/test_production_weight_cache.py tests/test_render_identity_once_1192.py tests/test_pwc_file_load_receipts.py tests/test_tessera_export_lane.py tests/test_digest_export_lane_1599.py tests/test_tessera_campaign.py tests/test_chain_seed_routing_1705.py tests/test_tessera_priced_export_inputs.py tests/test_duplication_baseline.py tests/test_architecture_doc.py tests/test_docs_staleness.py tests/test_docs_entrypoints.py -q --basetemp="$TMPDIR/pytest"
```

**266 passed, 12 skipped**, exit 0. The full named modules ran without
weakening or deselection, including the calibration-draw identity test in
`test_tessera_campaign.py`. The generator reported 303 primitive sites;
`cmp` confirmed that regeneration is byte-identical to the repaired fixture.
The only removed pre-PR primitive site is `tessera_hessian.token_ids_sha256`.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"d0c37e8b5971ec84001f1fbc92d9508f36321a5b53ab3ca783e2370a3b314fe4","size":1222}},"operation":"67461085a3347df36b7addb5a88a89251d24aa675e560d7df1b69307b39a68f7","receipt":"664193dd1fd1d05a168aab09489232c7e13f32be481fdc5cd5a654e9247834ad"}]

### Public API, pin coherence and unchanged admission answers

```sh
python -m pytest tests/test_tessera_lane*.py tests/test_serving_lane_metadata.py tests/test_tessera_core_boundary.py tests/test_tolerant_tessera_readers.py tests/test_tessera_legal_domain.py tests/test_tessera_split_pin_drift.py tests/test_tessera_serving_pin.py tests/test_tessera_serving_code_identity.py tests/test_tessera_campaign_pins.py tests/test_tessera_campaign_head_roster.py -q --basetemp="$TMPDIR/pytest"
```

**459 passed**, exit 0. This includes real producer grammar, missing/wrong API
refusals, status/attribution delegation, lane admission, frozen pin/rate
coherence, metadata import boundaries, and the entire existing serving-code
identity snapshot check. The snapshot equality establishes that changing
only its pin-commit field did not change the captured admission answers.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"fccb0844bae751ee50c917f37d10d06b5b2897032636fc433d9bddec1f02e62b","size":1314}},"operation":"957f0eb3f284d6c9e41067156e424cb07aa5e7ee80eb8ee744a3d5f7ff7961aa","receipt":"74cb182d48a5cb3a7001fa532c93cb43180d3c1078420765f0f1312f670c9c06"}]

### Documentation close-out

After updating the runtime README's current-pin instructions and preserving
the old qualification as historical, reran:

```sh
python -m pytest tests/test_architecture_doc.py tests/test_docs_staleness.py tests/test_docs_entrypoints.py -q --basetemp="$TMPDIR/pytest"
```

**22 passed**, exit 0.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"7e1db7cc6ed8e29af8a115c867fd28add2c1af162f9efdb561ac1b761be367c7","size":749}},"operation":"9f831af805b161752936f7fb1d83f58d2964da1328e0fdf285dd01173fc00b49","receipt":"61f58c8a2aff114a6e8a55353c3295a5b4af81c3478ed93fc53e53f798517ba9"}]

### Additional diagnostic limits

An earlier broader pin/campaign diagnostic returned **157 passed, 7 failed**.
One failure was the snapshot's stale `/pin/commit`, subsequently corrected
and covered by the successful complete identity suite above. Six were in
`tests/test_tessera_campaign_pinned_roster.py`: unchanged callers/tests import
removed `fused_module` or `direct_consumer_weight` serving internals through
`tessera_lane`, `tessera_campaign`, or the test itself. These failures do not
reach the digest sites or the lane-reader repair. The responsible caller
functions are unchanged by this leaf; the installed runtime was already
`aaa41700` before the repair. This is an existing branch/runtime integration
limitation, not a successful optional-suite result. Those separate
public-owner migrations are outside this leaf; no full-suite pass is claimed.
The named campaign and export gates are the complete green suites above.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"9e99a75d17f316463a24fa07b37035f1169e9cc6c8cab35d8898deff8d936f66","size":13373}},"operation":"505ec6f1f9f16871b6236275a8d6dcc416dbad70582da3f3e2bec9dd5e1b5fe2","receipt":"ad114a6a1fe9029ff3159017d101f3f02a1ed38a847219151fac460e158b70aa"}]

Two setup attempts collected no tests (an existing fixture incorrectly named
as a retained output, then a mistyped import-boundary test filename). The
corrected final commands above ran successfully; neither setup attempt is
counted as a check pass. Their receipts are retained below. A local inventory
generation attempt timed out; the admitted generator and byte comparison
above completed. The final diff was inspected, including the single-field
snapshot edit and restored retention implementations; `git diff --check`
passed.

[Test evidence {"artifacts":{"stderr":{"digest":"116d4dd1603b6205e8ad03887bcad6efb7acf48cc90410f1fde6c4cf97c140c8","size":86},"stdout":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0}},"operation":"a9e0581633a3a49cd00583d991061f0082c956e7fc0f988f9414a606b7b965e0","receipt":"748629f55ee52ccede80d5168fafda567b71c90f282dc35b824c27a947cc107b"}]
[Test evidence {"artifacts":{"stderr":{"digest":"919f1a5412e1dd2ae324e7f51456d148ea8f50261f5ba24e013f908b801a78a5","size":74},"stdout":{"digest":"48382a6c797f47457ce8af7f2f9620c939e279236a421d0f4c33693c328fa083","size":193}},"operation":"19af31e7be3a363df56a38a15239650581bf504f7001c324f64b488fafb49b16","receipt":"a002ceb5d3ad4a61fa1229566e66e500ff1ebd0a08e49f809783473e47031d7d"}]
