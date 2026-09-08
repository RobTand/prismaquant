# Closed original GLM capture compatibility issuance

Issue: [#422](https://github.com/RobTand/prismaquant/issues/422).
Implementation tested: `2196dbcc2a47cffa7c1ddf375b9741f0d684c3f0`.
This is CPU/meta control-path validation; no GPU forward/backward, quality,
throughput, or new native qualification result is claimed.

The existing `create_capture_compatibility` API now has a deterministic CLI:
`python3 -m prismaquant.glm_capture_compatibility preflight|issue --plan PATH
--plan-sha256 SHA256`. The plan has closed keys, absolute paths and byte
bindings. Both modes use the actual corrected GLM runtime, the authenticated
model config, and the existing streaming meta constructor. The original
producer gates and compatibility issuer remain unchanged.

`pending-plan.json` is the reviewed provisional input. Its SHA256 is
`6f59204d81a178baa8e6ea1852b836aced9c59bbded1d9399336235d1b333c0a`.
It is also frozen at
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/joint-original-graph-implementation-01/compatibility-issuance-pending-plan.json`.
Exactly four fields are pending: `capture`, `producer.terminal`,
`producer.receipt`, and `producer.output`. Pending fields are permitted only
for preflight. A partial mixture is rejected. No compatibility receipt was
issued during this validation.

## Actual corrected-image preflight

PB action `0f582924c359b13f1cfc7d372acaf3fde097d3c6d11ce828d79297dfc40706c0`
ran CPU-only on sparklina with 2 CPUs, 8 GiB and native threads bounded to one.
The `gb10` eligibility tag reflects the existing ARM corrected-container
runtime dependency; no GPU resources were requested. The wrapper preserved
PB CPU affinity, omitted GPU exposure and set `CUDA_VISIBLE_DEVICES=''`.
`cpu-campaign.json` records the exact image content, archive, mounts, command,
environment and independent compile action. It was submitted with:

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbcampaign.py \
  /mnt/shared/tessera-measurements/glm-canonical-census-20260908/joint-original-graph-implementation-01/compatibility-issuance-cpu-campaign-01.json \
  --detach
```

The actual CLI result in `actual-preflight-output.txt` reports:

- Corrected image content SHA256 `d0256efb83294e879ca33dd2d3131e861221c415ac5b024c2415e51c5467f026`.
- All model parameters on meta, CUDA unavailable and uninitialized.
- 34 authenticated KDA modules.
- Derivative identity SHA256 `e989cbe2fa47144af509a373db896ae3c8c2ef3d1dbf0d6dbecc73f78d6f8dd6`.
- Execution identity SHA256 `a86c98992be523ab6d588130f6d59958347e322310d6e53e2ba6877d16eb7a04`, exactly matching the existing corrected native graph evidence.
- The existing closed causal-expression/layer-0/bounded-graph proof accepted.
- Status `preflight_pending_original_capture`, the four pending fields above,
  and `receipt: null`.

This constructs the real 40-layer GLM skeleton from the already byte-bound
configuration without loading model weight shards or starting weight-cache
prefetch. It does not substitute a fake module or execute model forward.

## CPU tests and evidence audit

All 170 tests passed, zero skips or failures, across 11 PB-managed shards on
dl380g10. Each shard reserved 1 CPU and 4 GiB, with one worker and one native
thread. The shared-constructor regression scope covers streaming attention
selection, wrapper configs, meta-init FLA priming, visual setup, buffer resume,
tied heads and source identity. The 21 CLI tests include malformed/partially
bound plans, pending issuance refusal, the actual incomplete-capture validator,
producer refusal before model construction, CUDA refusal, native execution
identity mismatch, and exclusive receipt publication under synthetic unit
fixtures. The synthetic publication fixture is not actual capture authority.

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbtest.py \
  --checkout /home/rob/tmp/pq-glm-capture-compatibility-issuance \
  --python /home/rob/venvs/pq-cpu312/bin/python \
  --workers-per-shard 1 --threads-per-shard 1 --mem-gb 4 \
  --priority -10 --timeout-s 300 \
  --json /mnt/shared/tessera-measurements/glm-canonical-census-20260908/joint-original-graph-implementation-01/compatibility-issuance-cpu-tests-01.json \
  tests/test_glm_capture_compatibility_cli.py \
  tests/test_glm_source_derivative.py tests/test_export_resume_source_identity.py \
  tests/test_prismaquant_visual_phase2.py tests/test_streaming_attention_backend.py \
  tests/test_streaming_text_only_wrapper_config.py tests/test_meta_init_fla_priming.py \
  tests/test_export_buffer_resume.py tests/test_tied_lm_head.py \
  tests/test_architecture_doc.py tests/test_docs_staleness.py
```

The independent PB compile action
`02d72f6dbfa0fc574f63fc14f81bea7c5dd876d05b3d6826d3b7779099228eab`
also exited zero; its empty stdout CAS payload is expected. Initial attempts to
supply redundant `-q`/`-rs` through `--pytest-args` were refused by the published
client before submission; the successful command uses its standard output.
No missing test tooling or collection was reported.

`cpu-evidence.json` records all 13 actions, terminal and output path/SHA256,
canonical receipt bodies, producer identity, source snapshots, result claims,
resource telemetry and sanitized cleanup evidence. Each terminal is done with
exit zero and completed cleanup. The audit verified canonical receipt and
producer digests, actual CAS payload bytes/size, and `pb_verify_claim` with
`checks_passed: true`. The claim API's `attestation_verified: null` is retained
because that API needs the full action manifest for that additional check.
Each snapshot bundle was hashed and fetched; its only diff against the tested
commit is PB's generated closure file. Selected runtime/test file hashes are
also recorded. Thus the all-file tree comparison, not the snapshot-parent
label alone, establishes the tested code. Subsequent commits add evidence only.

## Completion input and issuance

After the coordinator verifies original action
`8740a0b3456bb6cb334ae80b0e35fc3da31c62918abdda8c41ad226020c4e88a`
has completed, freeze a new copy of the plan replacing all four pending fields
with actual absolute path/SHA256 bindings. The capture binding is its completed
manifest, and the three producer bindings are the original action's terminal,
canonical CAS receipt JSON, and actual CAS output bytes. Preserve the reviewed
request, original image/source, corrected derivative policy and native proof
bindings. Hash the entire final plan after those substitutions.

Use the first action in `cpu-campaign.json` as the existing CPU-only corrected
container invocation, changing only the checkout to the reviewed integration
checkout and the CLI's plan path/digest. Run complete-plan `preflight` through
PB, inspect its terminal/CAS result, then issue through the same invocation with
`issue`. Do not fabricate completion artifacts or run issuance against the
pending plan. Issuance verifies the completed original capture and producer
before constructing the meta skeleton, revalidates all existing compatibility
gates immediately before publication, and writes the new sidecar with
exclusive-create semantics. Existing output causes refusal. The original
capture remains the bound input; it is never rewritten.
