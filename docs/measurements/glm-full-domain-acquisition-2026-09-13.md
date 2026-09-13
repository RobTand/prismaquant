# GLM full-domain adaptive acquisition: executed integration

2026-09-13; research software milestone for #581. This is an acquisition
request producer, not a completed allocation, quality measurement or export.

The complete legal domain stays available: E4M3 K1 R256–2048 (1,793 rates)
and BF16 K1 R256–4096 (3,841 rates). The adapter uses the existing legal-domain
resolver and adaptive RD-hull refiner. Missing endpoints and the two BF16
table-width transitions become bounded next-measurement requests. After those
witnesses exist, the allocation multiplier can prioritize interior refinement.
No extrapolated price is produced, and a full-grid measurement is not required
before proposing candidates. Exact selected-rate scoring and downstream
held-out validation remain necessary.

The existing GLM merged table contains 36,423 units, 197,990 measured scalar
output-MSE rows and 68,303 interpolated rows. Dense E4/BF prices occupy
R832–1088. Routed E4 has sparse anchors within that envelope; routed BF16 has
only R1024. These are activation-inclusive scalar prices, not joint-AURA rows.
A singleton supplies no slope. The successful PR #503 family-transfer study
also does not establish full-domain or routed-family transfer.

The command adapter reads only the selected unit journal parts, matches each
measured row to its anchor, checks the complete producer recipe and encoder
source digest, and preserves source/wire metadata and live pin identities.
Recorded wire metadata is checked; source tensors and wire bodies are not
re-read or requalified. It does not synthesize Fisher statistics. A generic
allocator invocation remains dependent on a valid serialized probe from the
joint run.

## Executed example

PB action `20eb73908520cf7f7726a1776dc65b194461b1fb0829a09b0bdccc80e3d09d47`
completed rc0 on sparklina with CPU-only admission, one CPU and 6 GiB. Its
result/CAS receipt and command are retained at:

`/home/rob/dq-runs/glm-campaign-takeover-20260913/allocation/pb-full-domain-requests-04.log`

Output:
`/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/allocation/full-domain-requests-04.json`

Output SHA256:
`43486d7e1239a524eee03f55eddf8b6333293a7aab57c61e9796dd8f504e9dd6`.

The example reads dense layer-0 down and routed layer-3 expert-0 down under
both families. It retains each complete domain and emits exactly eight next
measurements: E4 R256/R2048 and BF R256/R4096 for each member. The two units
are integration examples, not a frozen representative scientific pilot.
Atomic serving-group expansion is an explicit caller obligation. No GPU
measurement, interpolation acceptance or model-wide accuracy claim follows.

Source cost SHA256:
`cd21541019cb670876fcd3501cba8c58e3473044090e54f16726ad26b0eb27e1`.
The active producer is the frozen d403cc5a study state; its input-identity
encoder digest matches every consumed measured record:
`0833671bbddbc3fb7186bdbed0a905ef248c23ffdf1aeee0c083322680d20f6b`.

The example uses `/home/rob/venvs/pq-release/bin/python` on GB10 with the
frozen producer's `src` first after the checkout in `PYTHONPATH`. The x86
CPU environment `/home/rob/venvs/pq-cpu312/bin/python` also executes the tests.
The generic `pb-cpu` environment lacked `compressed_tensors`; those failed
attempts are retained as environment failures, not scientific negative data.

## Correct budget scope

All 120 source-header shards were read, without tensor payload reads. Every
cost unit matches a BF16 source weight. The quantizable population is
305,915,756,544 parameters. Actual immutable source tensor bytes are
30,815,140,728, including 291 FP32 tensors and 2,056 BF16 tensors.

The prior 155,668,854,528-byte target is quantizable-only. Its equivalent
whole-artifact cap, adding actual immutable bytes and the 268,435,456-byte
non-tensor reserve, is **186,752,430,712 bytes**:

```
--target-disk-gb 186.752430712 --artifact-overhead-reserve-bytes 268435456
```

Header digests, tensor inventory and arithmetic are retained in
`/home/rob/dq-runs/glm-campaign-takeover-20260913/allocation/source-byte-inventory.json`
and `source-tensors.json`. This is serialized-artifact accounting, not a TP2
resident-memory qualification.

## Validation and incidental repair

The final PB suite passed 31 tests, rc0, action
`66ade4ade2c777d0953d9f4b1e147df6754b771eb60354eb96918384b8e18b04`.
Tests exercise bounded endpoint requests, singleton support, illegal or
repeated rates, decision-focused refinement, actual adaptive-picker execution,
recipe/source rebinding refusal, and architecture staleness. Receipt logs are
under the same allocation directory.

The integration exposed an existing `TrellisAdaptiveRateProposal.as_dict()`
crash: it called a nonexistent `TesseraRateSurface.as_dict()`. A separate
commit serializes the surface through its existing canonical identity. The
pre-fix PB receipt `6574670092b2` reaches and records that AttributeError; the
regression test then passes. No allocator objective or serving gate changed.
