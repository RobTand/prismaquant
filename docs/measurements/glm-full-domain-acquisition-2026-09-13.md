# GLM full-domain adaptive acquisition: executed integration

2026-09-13; research software milestone for #581. This is an acquisition
request producer, not a completed allocation, quality measurement or export.

The complete legal domain stays available: E4M3 K1 R256–2048 (1,793 rates)
and BF16 K1 R256–4096 (3,841 rates). The adapter uses the existing legal-domain
resolver and adaptive RD-hull refiner. Missing endpoints and the two BF16
table-width transitions become bounded next-measurement requests. After those
witnesses exist, the allocation multiplier can prioritize interior refinement.
The current adapter reports `adaptive_converged: null`; full-grid coverage
is not an adaptive stopping criterion. The original request-04 artifact
predates that naming clarification and its `acquisition_complete` field meant
only full-grid measurement coverage. It is historical evidence, not a controller
completion gate. No extrapolated price is produced, and a full-grid measurement is not required
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

## Census completeness repair (2026-10-02, #2132)

The coverage ledger accepts `unit_shapes`, an explicit census mapping from
Linear name to `(rows, columns)`. Supply it when auditing a model-wide request:
the priced-row roster alone cannot reveal wholly unmeasured units or families.
Each requested family retains every legal rate at that unit's actual shape.
`producer_refused_q256` separately retains shape-illegal rates and their reasons;
for a real KDA `(8192, 128)` E4M3 Linear, 897 rates remain legal and 896
are explicitly refused rather than dropping the entire Linear or smoothing
the odd-q256 quota holes. Priced rows at those refused rates still fail closed.
Empty ledgers, absent selections and absent unit/family pairs now refuse
instead of vacuously certifying complete coverage. `missing_acquisition_work`
retains these missing cells in the existing v1 format. No missing cell receives
a price, runtime admission, export qualification or quality certification.
If the producer refuses every rate at a positive shape (for example arity-2
E2M1 at `(1, 32)`), the ledger retains all refusal reasons but never marks the
entry complete. Coverage and acquisition selections of such an entry refuse
with `no producer-legal rates`; no fictional missing legal rate is invented.

The near GLM milestone remains T8-only at the measured EXL3 serialized size.
The final objective is an actual per-Linear T4/T8/T16 accuracy/size/prefill/decode
frontier, not a uniform-format artifact or arbitrary weighted scalar default.
The 42 historical PACT gamut cells and native seven-rung subsets do not define
the legal q256 domain. Missing native, quality, original-source CUDA and
construction cells remain repair blockers; scalar screens, historical census
restamps and the retained TP1 timing pilot cannot authorize production picks.

## Explicit joint acquisition adapter (#2139)

The existing command accepts `--cost-currency joint-aura` without scalar
`--anchor-parts`. It calls the ordinary `require_run_currency` attestation and
joint raw-row validator, then checks the exact v2 run/probe, cached rendered
tensor, activation and per-unit source-weight bindings. Mixed/scalar-only
tables, missing signed W/A/mixed samples and plausible misbound metadata refuse.
Every selected joint measurement record is retained unchanged in the request.
No scalar MSE-to-Fisher conversion or activation multiplier is performed.

The current public writer provides byte estimates only for acquisition-bracket
ranking, not measured-wire prices. `prices` remains null; the request cannot
be an allocator payload, interpolation qualification, selected-assignment
confirmation or production promotion. A wholly unmeasured family retains its
legal domain and emits endpoint requests without invented joint rows.

The shared shape adapter now retains producer-refused holes and chooses legal
neighbors on both sides of table-width changes. Existing RD-hull brackets rank
interior requests; their midpoint is projected to an unmeasured legal point
inside the same bracket, with lower-q256 ties. Full-grid measurement is not a
proposal prerequisite; every unknown legal rate remains visible. Exact shipped
pick confirmation and held-out evaluation remain separate, and original-source
CUDA/provider plus fresh-global diagnostic prerequisites are not bypassed.

The actual adapter smoke exposed an existing BF16 footprint round-trip defect:
the writer priced CHANNEL reach sigma, but its recorded footprint/reconstructed
recipe discarded it, so revalidation differed by 23 bytes and refused the
candidate. The shared recipe/footprint owners now retain window seed, window
sigma and channel sigma rather than suppressing the mismatch or subtracting
bytes. All size claims remain the public writer's; no native admission changes.

## Executing bounded joint requests (#2171)

The existing anchor campaign accepts an explicit `--acquisition-request` and
`--acquisition-request-sha256` pair. The request is the existing joint acquisition
JSON, not an allocation, capture or export. Intake authenticates its actual cost
pickle and revalidates ordinary raw-v2 joint currency, run/probe/source evidence
and the complete legal domain. Duplicate, illegal, already measured, misbound or
qualification-claiming proposals refuse before campaign input work.

This explicit path requires research mode and one round, without a global rate
band, exhaustive-grid mode, audit extras or partial expert-partition semantics.
Both first-batch priming and round-one execution use the existing atomic groups
and one exact requested-rate selector: group-member requests are unioned only
within the actual shared legal grid. No rung is snapped, silently dropped or
replaced by an endpoint/uniform anchor. Every actual member needs a bound source
identity; the selected scope must be exactly the requested atomic expansion.
Actual source bytes are checked before encoding, outside per-anchor failures.

The output remains the normal source/H/recipe-bound scalar anchor journal and
wire/render cache. These establish candidate bytes, not joint Fisher prices.
Four-probe Stage B pricing, native prefill/decode context, selected assignment,
immutable export and held-out quality/serving remain separate actual transitions.
The original request keeps every deferred legal rate. This wiring does not adopt
a provider, promote row-zero diagnostics to full-calibration H, move a public or
private reader pin, or qualify any unsupported native cell.

## Bound per-row intake and control readset (#2195)

`load_joint_campaign_acquisition(binding, units=names)` authenticates the whole
original request and raw cost table before projecting onto explicit row units.
An altered unselected report or cost unit cannot escape validation. The pure
`project_joint_campaign_acquisition` helper accepts only an already authenticated
intake; it selects whole known units deterministically, retains deferred families
and empty atomic members, and does not infer groups from names. The runtime still
requires every actual member of each selected atomic group. An all-deferred row
contains no measurement work and is not an admitted zero-work execution.

Projection never rewrites the original request, creates another price currency,
changes its global request/cost/run/probe identity, or snaps requested rates. The
default `units=None` return remains unchanged.

The torch-free `tessera_acquisition_inputs` owner provides
`joint_campaign_acquisition_control_inputs(binding)`, reusing the same strict
bound JSON owner and declaring the actual whole request and raw cost, in that
order, with SHA256 and byte lengths. Standalone metadata builders stream-hash
the cost through the existing digest/stat-identity owners, without importing
PrismaQuant or Torch. Runtime intake supplies its existing fenced staged reader.
These are control inputs read before captures or weights, not a scientific-
admission shortcut: planning still performs complete intake validation. Drift
after cached reads refuses under the existing bound-byte/stat-fence owner. The
existing staged readset remains the
owner of mount, entry and consumption accounting. Candidate journals and decoded
wire caches still need the separate four-probe joint pricing and production
qualification transitions above.

The September 13 executed example and its quantizable population, immutable-byte
count and reserve are historical inputs, not authority for the current original
GLM5.3-Flash allocation or its EXL3 size comparison. The current campaign needs
its actual eligible Linear census and immutable serialized footprint; immutable
BF16/F32 tensors stay outside the quantizable bpp denominator. No reserve,
tolerance or serving-performance target is inferred from that historical example.
