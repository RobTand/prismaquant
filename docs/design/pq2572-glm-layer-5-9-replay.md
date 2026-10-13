# GLM layer 5 and 9 Stage B replay protocol

PRISMAQUANT-2572, contract revision 1. Base: `138069b799323435fb5f0b3cec3ee339f6264a62`.
This document defines an offline evidence protocol. **Historical replay results
are unmeasured.** All numbers and operands in the executable examples below are
**synthetic**; they do not identify or reconstruct a historical GLM target.
No production code, selector defaults, arithmetic, tolerances, cache behavior,
serving gate or publication path changes.

## Scope and design

Only the literal labels `e240` and `e131` can be requested. Each must resolve
through an authenticated, frozen registry to exactly one historical tuple
`(qualified_unit_name, layer, projection, format)`, with `type(layer) is int`
and layer in `{5, 9}`. The registry must be acquired from historical records,
not inferred from a label, expert number, suffix, model naming convention or
present-day roster. No alias, wildcard, case folding, whitespace trimming,
other label or other tuple is admitted. A caller supplies a literal label and
its entire tuple; both must match. A changed tuple needs new historical evidence,
not a fuzzy match. An incomplete registry or input binding cannot qualify.

The current checkout supplies no authenticated complete tuple for either label.
The README's L05 expert-240-down observation is a discovery lead, not an exact
qualified-name binding. This protocol does not guess `e131`'s name, projection,
format, calibration, or the association of either label with a layer. Synthetic
fixtures deliberately use `synthetic.*` names and a synthetic format, in a
separate mode that can never yield historical measurement acceptance.

The existing mechanism is joint operator statistics and candidate projection,
with the one-pass Stage B spill and existing weight/activation caches. The
protocol adds receipts and arithmetic checks in this document only.
Architect skipped: the brief fixes the evidence boundary and arithmetic; a
single document with explicit records needs no new application interface.
The throughput checkpoint for this documentation work is:

- Blocking first steps: read the retained evidence and the existing arithmetic.
- Independent workstreams: n/a; one owned document and delegation prohibited.
- Shared mutable state: n/a; checks use isolated in-memory synthetic records.
- Smallest safe decomposition: one writer owns schema, examples and checker
  together so they describe the same contract.

## Retained evidence and its limits

[`cpu_glm_provenance.py`](../../experiments/alloc_lead_asd/cpu_glm_provenance.py)
locates L05 records **outside this checkout**, under
`/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/`:

| Role | Relative location under that root |
| --- | --- |
| Stage B rows | `r13-stageb-20260923/a4/overlay/layer-quanta/layer-005/cost.pkl` |
| Boundary manifest | `r13-stageb-20260923/meta-045-fae344d/adjoint-slices/layer-005.json` |
| Cotangent handoff | `r13-stageb-20260923/a4/overlay/layer-quanta/layer-006/handoff/31f5259905fe4c35b91e678809f967f2/handoff.json` |
| Execution plan | `ws-sb4-1151/policy/overlay-plan.v8-chain.json` |

These are discovery locations, without authenticated byte counts or SHA-256 in
this protocol. They are not available tensor operands. The script audits
metadata and priced rows, not independent full-model cotangent correctness.
Its fixed format is `TESSERA_E4M3_K1_R1024`; that alone does not resolve labels.

The [README, Historical GLM comparison contract](../../experiments/alloc_lead_asd/README.md#historical-glm-comparison-contract)
records L05's 867 format rows sharing a K4 global-row probe identity, 512 by
512 calibration, all 512 output-logit positions, global token count 262144,
seeds 7000 through 7003, temperature 1, boundary 5 and cotangent 6 coverage.
It records additive A `0.0026182497805902175` and expert 240 down's A diagnostic
`0.0021725103132911663`. These are cited historical observations, **not measured
results of this replay** and not the historical joint-price reference. The
inspected historical replay directory contained only `L05.dry-inputs.json`,
not measured per-sequence/token pricing. Draw-summed components cannot recover
the missing row decomposition.

Neither reference supplies checkout-local source/rendered weights, boundary
activations, cotangents, calibration-token operands, or L09 bindings. Do not
copy these L05 settings into `e131` or L09. Authenticate each target's own
calibration identity, sequence length, token/logit scope, seeds, global
normalization, source model, activation policy and arithmetic from its records.
The README's tiny GLM and Qwen controls use different draws and cannot fill any
historical binding. Preserve original identity objects without pruning nullable
metadata: `activation_identity` can emit null static contracts, maxima and
scales, and execution arithmetic can preserve `operator_windows: null`.
A null optional field is not a missing tensor or a missing required identity.

## Input-manifest schema: `pq2572.inputs.v1`

JSON uses UTF-8, finite JSON numbers, exact integers for coordinates, and
lowercase 64-hex SHA-256. All required keys below must be present. Extra original
metadata is preserved. `null` is permitted only where specified. NaN and infinity
are invalid even if a permissive JSON parser accepts them.

An `Artifact` is `{location: nonempty string, bytes: nonnegative integer,
sha256: 64-hex string}`. Verify byte count and digest against bytes read; a path
or digest declaration alone is not authentication. An empty artifact can have
zero bytes when its role permits emptiness. Locations can be local paths or
immutable storage URIs, resolved by the admitted reader. Tensor descriptors
also bind logical tensor identity, shape and dtype; distinguish serialized
artifact bytes from logical tensor bytes when storage is serialized/compressed.
The executable raw synthetic fixtures use contiguous uncompressed operands,
so their artifact bytes equal shape product times dtype size.

A `Binding` is `{value: object|null, artifact: Artifact|null,
missing_reason: string|null}`. A complete binding has a structured value,
a receipt artifact whose decoded JSON equals that value, and a null missing
reason. A receipt is an authenticated extraction record, not a replacement for
original evidence: it carries the original record/operand artifacts and
locators listed below, which must also be verified. Operations must compare the
extraction with the original record at the locator. SHA-256 binds bytes, not
historical authority; the registry and extraction receipts require a trusted
historical capture/cost source and an independent authentication receipt.
An unavailable binding has `value: null`, `artifact: null`, and a specific
nonempty missing-evidence reason. It is recordable but cannot qualify.

Manifest fields are `schema`, `mode` (`historical` or `synthetic`),
`target_registry` (exactly `e240` and `e131` Binding entries), and `targets`
(exactly those same labels). Each target has `identity` (the exact tuple object,
or null while unresolved) and `bindings` (all roles in the following table).
Keep separate binding entries for each target even when bytes are shared.
Each registry value has `qualified_unit_name`, integer `layer`, `projection`
and `format`; its artifact authenticates that tuple, subject to the historical
source requirement above.

| Binding role | Required structured value and original evidence |
| --- | --- |
| `stage_b_cost` | `row_locator`, `original_artifact`, `original_row` preserved in full; `components` with one `{probe, weight, activation, mixed}` per authenticated probe; finite nonnegative `joint_price`. Extract from `signed_components_per_probe` and `predicted_dloss`, preserving native `probe_identity` and `joint_operator_identity` (including `qname`, `format`, `activation`). Bind the cost row's tuple through its locator/registry context, operator/probe digests and authoritative signed components and joint price. Do not use an activation-only diagnostic as joint price. |
| `boundary_activations` | Integer `boundary`, nonempty `records` of tensor descriptors, capture sessions and caller coordinates. Authenticate boundary and payload entries against the target's original capture manifest. |
| `cotangents` | Integer `boundary`, nonempty tensor `records`, probe/capture coordinates and session identities. Preserve which output boundary and backward selection generated each cotangent. |
| `source_weights` | `model_identity`, nonempty tensor `records` with `tensor_identity`, `shape`, `dtype`, operand `artifact`. Full model/checkpoint revision and the target's source tensor must be bound. |
| `rendered_weights` | `renderer_identity`, nonempty tensor `records`; bind the exact rendered candidate, layout, artifact and candidate format, not just a renderer name. |
| `calibration_tokens` | `calibration_identity`, positive integer `sequence_length`, `n_sequences`, `global_token_count`, literal `token_scope`, and nonempty tensor `records`. Bind token bytes, order, selected positions and logit/Fisher scope. |
| `probe_identity` | `layout`, explicit integer `seeds` in probe order, `normalization` with `global_token_count` and `rule`, `arithmetic` with `component_addition` and `square`, and complete `original_identity`. Original required fields: `schema`, `source_model`, `calibration_sha256`, `calibration_shape`, `calibration_dtype`, `n_probes`, `seed_base`, `token_scope`, `temperature`, `distribution`, `normalization`, `producer_source_sha256`, `source_execution`, `arithmetic`. Bind RNG algorithm/domain, row indexing, temperature and original arithmetic/kernel policy through that identity. |
| `activation_quantization` | Boolean `quantizes_input`, `quantizer`, complete `original_identity` from the historical activation receipt. Bind scale/clip/group/served-policy metadata and any required calibrated-scale operands. Inapplicable original scale/contract fields may remain null. |
| `rendered_assignment` | `assignment_identity`, `unit_formats` mapping qualified names to exact formats, `original_assignment` preserved. Bind the whole assignment used for the captured path, not a newly chosen fallback. |
| `execution_configuration` | `source_commit`, `projection_backend`, nonnegative integer `probe_microbatch`, positive integer `capture_batch`, `original_configuration` preserved. Bind batching, source kernels, model implementation, dtype, precision policy, fusion, windowing, spill settings and cache/residency configuration. |
| `container_identity` | Immutable `image_digest`, `image_reference`, `runtime_versions`, `modeling_digest`; bind the exact image and model/runtime files, not a mutable image tag alone. |
| `capture_records` | Nonempty `expected` list, in actual dispatch order, described below. Its receipt and each record's evidence artifact authenticate coverage independently of the produced row log. |

A tensor record requires a nonempty `tensor_identity`, `shape` (list of integer
dimensions, including zero only for authenticated emptiness), nonempty `dtype`,
and operand `artifact`. Retain original storage/selection/session metadata as
additional fields. Operations checks shape/dtype against original headers,
logical unit geometry and the existing replay reader, including source versus
rendered shape equality. Authenticated scale and routing operands use the same
Artifact and descriptor rules. The synthetic checker supports raw
`float32`, `float64`, `bfloat16`, `float16`, `int64`, `int32`, `uint8` fixtures;
it does not introduce a historical tensor reader. Its original-row artifact is
a JSON stand-in with native Stage B field names. Historical `cost.pkl` must
instead be read through the existing trusted cost reader, preserving the raw
cost artifact and the extracted original row in separate authenticated receipts;
the synthetic JSON decoder is not a pickle-reader replacement.

The executable **historical manifest template** `unmeasured_manifest()` below
expands every table role separately for each target, with null identity, null
value/artifact and a role-specific reason. It is structurally recordable but
intentionally cannot qualify. For example, its `e240` source-weight binding is:

```json
{
  "value": null,
  "artifact": null,
  "missing_reason": "e240: Exact source-model tensor identity and operands not authenticated"
}
```

Discovery paths and README calibration observations remain in the evidence
notes, rather than masquerading as complete authenticated bindings. Operations
fills nulls only after acquiring and verifying exact records, bytes and sources.

## Replay and row-log schema: `pq2572.rows.v1`

Use the existing `JointOperatorStatisticsLease` in
[`joint_aura.py`](../../prismaquant/joint_aura.py), its `project` method and
[`joint_projection_backend.py`](../../prismaquant/joint_projection_backend.py).
For the preserved normalized cotangent `g`, input `x`, activation residual
`dx = QDQ(x) - x`, and weight residual `dW = rendered_W - source_W`, the signed
contractions are W = `<g.T @ x, dW>`, A = `<g.T @ dx, source_W>` and
M = `<g.T @ dx, dW>`. Use the authenticated production upcasts, QDQ, backend,
chunking and accumulation rules; the formulas do not license a new tensor
implementation. Global probe normalization belongs to the whole selected
calibration draw. Do not renormalize a batch, sequence, expert's routed tokens
or attribution row; do not normalize already normalized cotangents twice.

[`joint_cost_quantum.py`](../../prismaquant/joint_cost_quantum.py)'s
`SpillSequenceAttribution` uses `JointBlockAttributionLease`,
`StageBReplaySpill.replay_records`, and resident candidate deltas. The spill
reader in [`joint_replay_spill.py`](../../prismaquant/joint_replay_spill.py)
redelivers captured invocations synchronously in capture order with capture
batch labels, selected gradients, shape/layout and checksum checks. Use the
existing `ProductionWeightCache`, activation cache/prefetch and spill admission
budgets. Required data that cannot be prefetched must fail, not silently
stream through a second cache. This document's CPU checks inspect scalar
records only; real replay runs in the known-good admitted GPU container.

Enable the existing opt-in execution selector only after exact tuple binding:

```json
{"sequence_attribution": {"candidates": [["<authenticated qualified name>", "<authenticated format>"]], "gate_relative": 0.001}}
```

Construct each layer's exact requested pair roster from the two bound labels.
Do not use `"all"`, wildcard patterns, aliases or an inferred tuple in this
protocol, even though the production selector supports `"all"`. Selection is
bound into run identity. The priced probe identity is preserved. This instrument
requires the one-pass spill reader and refuses unsupported windowed replay or
resume of committed rows without the requested attribution. Existing sidecar
block aggregation is descriptive; it does not supply per-token routing facts
or change authoritative Stage B prices. Retain invocation receipts separately
where a block contains multiple invocations. This protocol specifies their
required evidence; it does not add logging to the production implementation.

A row log is `{schema: "pq2572.rows.v1", rows: [Row, ...]}`. Each row contains:

| Field | Meaning |
| --- | --- |
| `target`, `identity` | Literal label and its exact four-field tuple |
| `probe` | Integer ordinal into the target's authenticated seeds |
| `capture_batch` | Original integer capture batch ID |
| `invocation_order` | Original integer dispatch ordinal, preserved, not regenerated by sorting |
| `dispatch_rows` | Authenticated selected row count; zero for an authenticated empty dispatch |
| `sequence`, `token`, `routing_slot` | Integer coordinates or null, only as retained caller/routing facts permit |
| `weight`, `activation`, `mixed` | Finite signed scalar contributions under original global normalization |
| `evidence` | Nonempty Artifact list locating original capture/dispatch facts and raw contraction output |

Every `capture_records.expected` entry has `probe`, `capture_batch`,
`invocation_order`, `dispatch_rows`, the three nullable coordinates, and
nonempty `evidence`. The key is
`(target, probe, capture_batch, invocation_order)`. The tuple, key, row count,
coordinates and order in the row log must match the authenticated expected list
exactly. Expected entries cannot be inferred from row-log length or from only
nonzero observations. Authenticate capture closure: all probes and capture
batches must be represented, including empty routing dispatches proven by
caller/routing records. Coverage is independently checked against that closure.
An expected empty dispatch requires an explicit row with W=A=M=0, not omission.
Absence of a record is not evidence of zero. Duplicate expected entries,
duplicate output rows, missing records, unexpected records, wrong targets,
wrong probes and changed dispatch order refuse qualification.

An invocation spanning several sequences/tokens/slots has those coordinates
null unless its retained facts allow a finer decomposition and each finer row
has its own authenticated expected key. Never infer `(sequence, token, slot)`
from a flattened row index, tensor shape, expert ID, or capture batch. The
synthetic fixture intentionally has null coordinates. A capture-batch result
must be called capture-batch attribution, not per-sequence pricing. A separately
retained coordinate artifact may refine the schema in an authenticated replay
packet; this protocol's checker refuses invented coordinates.

## Reconciliation and independent gates

For every target, component c in `{weight, activation, mixed}` and authenticated
probe p, with b enumerating expected invocation rows in dispatch order:

- `R[c,p] = math.fsum(x[c,p,b])`.
- `D[c,p] = math.fsum(abs(x[c,p,b]))`.
- Signed residual `r[c,p] = R[c,p] - H[c,p]`; absolute residual `abs(r[c,p])`.
- Component bound `B[c,p] = 0.001 * D[c,p]`. Pass iff absolute residual is at
  most B; if D is zero, require R exactly equal H. A normalized residual is
  `abs(r)/D` when D > 0, null when D = 0. Keep the exact-zero result explicit.

H comes from the authenticated historical Stage B cost row, never a sum of the
new rows. Compute each component's gate separately; matching W+A+M cannot hide
redistribution between W, A and M. Keep signs through all batches, invocations
and partitions before the single squaring.

For each probe compute `T[p] = (R[weight,p] + R[activation,p]) + R[mixed,p]` in
that exact left-to-right binary64 addition order, matching `project`'s
`weight + activation + mixed`. Do **not** replace this three-component expression
with `sum`, `fsum`, or a reassociation. Python versions can give `sum` different
floating-point behavior. `fsum` remains required for each component over b.
Use [`squared_signed`](../../prismaquant/joint_aura.py)'s multiplication
convention: `float(T) * float(T)`, not exponentiation. Reconstruct:

`P_joint = 0.5 * math.fsum(T[p] * T[p] for p in probes) / n_probes`

`P_A = 0.5 * math.fsum(R[activation,p] * R[activation,p] for p in probes) / n_probes`

P_A is an activation-only diagnostic, not an additive-unit aggregate, a KL
measurement, or the joint-price gate. The joint-price signed residual is
`P_joint - P_historical`; denominator `abs(P_historical)`; bound
`0.001 * abs(P_historical)`. Pass iff absolute residual is within that bound,
with exact equality at historical zero. Relative residual is null at zero.
Also retain historical signed totals and squared samples when present and
check their consistency with the original row arithmetic during authentication.

Recompute all gates from raw rows and authenticated references. Ignore stored
`pass` flags. Verify finiteness of all operands **and derived values**, including
fsums, D, residuals, bounds, total additions, products, sums of squares, means
and relative residuals. Overflow, nonfinite values, empty probe lists, missing
components, incomplete bindings or coverage failures cannot pass. Never
coerce booleans, numeric strings or null into numbers. Independent recomputation
must match the reported R, H, D, residuals, bounds, prices and gate results.

Per-target arithmetic acceptance requires complete authenticated bindings,
exact coverage, every component/probe gate, and the joint-price gate. Historical
measurement acceptance requires **both e240 and e131** to pass, plus historical
source authentication and execution/output receipts. One passing target is
insufficient. A synthetic arithmetic pass never qualifies a historical result.
The fixed protocol 0.001 bounds do not alter configurable production sidecar
tolerances, including the existing tests' 0.5 fixture.

## Verdict-report schema: `pq2572.report.v1`

A report must contain the following fields. Gates are claims to be independently
recomputed, not trusted inputs:

| Field | Required content |
| --- | --- |
| `schema`, `mode`, `historical_status` | Version; synthetic/historical; `unmeasured` until completed historical receipts exist |
| `input_manifest`, `row_log`, `raw_outputs` | Artifact references with byte counts/digests; include original stdout, contraction outputs and capture/execution receipts |
| `reproduction` | Exact commands, source commit, container/runtime/modeling identities, admission receipt, settings and output locations |
| `targets` | Both literal labels; exact tuple; missing bindings and reasons; expected/observed counts and coverage discrepancies; all per-probe R/H/D, signed/absolute/relative residuals, bounds and recomputed gates; reconstructed T, joint price, historical joint price and activation diagnostic |
| `arithmetic_accepted`, `measurement_accepted` | Independently recomputed aggregate decisions, with failure reasons; distinguish synthetic checking from historical acceptance |
| `verdict` | Literal `no verdict`, or exactly one named cause |
| `observations` | Cited measured block facts, each with Artifact and record/coordinate locator; distinguish cited historical observations from this replay's measurements |
| `hypotheses` | Unproved explanations, separate from observations |
| `discriminating_comparison` | For a cause: authenticated compared blocks/arms, what was held fixed, changed variable, measured result, and why it discriminates that cause; null for no verdict |
| `next_facts` | Exact missing facts, artifact/record/coordinate requests and comparison needed; not vague requests to investigate |
| `excluded_comparisons` | Comparison and reason for exclusion, including unauthenticated G3 bands |
| `profiling` | In-process traces and aligned Netdata evidence from **both boxes**, raw series/artifacts and time alignment; absent evidence stays null with reasons |

Reconstructing components using banked cotangents tests consistency **conditional
on those cotangents**. It does not independently establish their correctness.
A cotangent cause needs an independent full-model derivative or other matched
discriminating check with authenticated operand and source bindings.

A G3 comparison can support a cause only after authenticating historical band
scope, including **memo 8.1's binding**: memo artifact location/bytes/SHA-256,
section locator, original band/token/source-window mapping, prefix context,
loss object, teacher, assignment and calibration identities, and exact matched
block roster. No memo 8.1 binding is available here; exclude G3 from causal
evidence. The README describes 25 original-prefix rows `0,4,...,96` among 100
joined rows and differing 511/512-logit bands. Do not compare that subset's
finite WA-W with the whole 512-row A aggregate as if scope and loss agreed.
A metadata-only audit, a tiny control or a banked scalar is no substitute for
the missing matched block facts.

Current historical report: `historical_status: "unmeasured"`,
`measurement_accepted: false`, `verdict: "no verdict"`. Observations are the
cited metadata/discovery limitations above. Hypotheses include operand pairing,
cotangent accuracy, routing/capture effects and scope mismatch; none is a cause
finding. Next facts are the missing target registry tuples, original cost rows,
capture closure and tensor/assignment/calibration/probe/runtime bindings for
**each** target, then an independent matched comparison. `no verdict` cannot
waive any reconciliation gate. No speed, cost, residency or performance cause
claim is made. Operations must profile before/after any performance comparison,
retain both boxes' aligned Netdata CPU/IO/power series and rank GPU work by
work per joule; GB10 GPU utilization alone is not diagnostic.

## Worked synthetic examples

For both fixtures: two probes, three captured dispatches per probe, last dispatch
explicitly authenticated empty. Coordinates are null. Synthetic calibration is
two sequences of length two, scope `all`, global count four, seeds 11 and 12.
These settings apply only to these synthetic records.
The fixture's tensor bytes exercise artifact/schema validation. Its signed
scalar rows are stipulated arithmetic operands, not contractions computed
from those tensors; no tensor replay or cotangent-correctness claim follows.

| Target | Probe | Dispatch | W | A | M | Ordered W+A+M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| e240 | 0 | 0 | 4 | 1 | 1 | 6 |
| e240 | 0 | 1 | -3 | -2 | 1 | -4 |
| e240 | 0 | 2 (empty) | 0 | 0 | 0 | 0 |
| e240 | 1 | 0 | 1 | 2 | -1 | 2 |
| e240 | 1 | 1 | -1 | 1 | 1 | 1 |
| e240 | 1 | 2 (empty) | 0 | 0 | 0 | 0 |
| e131 | 0 | 0 | 2 | -1 | 0 | 1 |
| e131 | 0 | 1 | 0 | 0 | 0 | 0 |
| e131 | 0 | 2 (empty) | 0 | 0 | 0 | 0 |
| e131 | 1 | 0 | -2 | 1 | 0 | -1 |
| e131 | 1 | 1 | 0 | 0 | 0 | 0 |
| e131 | 1 | 2 (empty) | 0 | 0 | 0 | 0 |

Explicit expected outputs: e240 R by probe is `(1,-1,2)` and `(0,3,0)`;
D is `(7,3,2)` and `(2,3,2)`; totals are `[2,3]`, joint price **3.25**, A
price **2.5**. e131 R is `(2,-1,0)` and `(-2,1,0)`, totals `[1,-1]`, joint
price **0.5**, A price **0.5**. Synthetic historical-reference residuals are
zero and all arithmetic gates pass. Historical measurement acceptance is false.
For e240, squaring invocations prematurely gives
`0.5 * (36 + 16 + 0 + 4 + 1 + 0) / 2 = 14.25`, not 3.25. The check below
asserts both numbers from the fixture so this counterexample cannot silently drift.

Additional cancellation fixture: one probe's components `(1e16, 1, -1e16)`
produce ordered total zero and joint price zero, although compensated addition
would produce one and price 0.5. Component accumulation example
`fsum([1e16, 1, -1e16]) = 1` remains correct **within one component over rows**.
These distinguish component accumulation from the three-component addition.

Threshold examples used below have explicit expected results: R=1000,
H=1001, D=1000 passes at residual 1/bound 1; the next representable H above
1001 refuses. Joint R=1001, H=1000 passes at residual 1/bound 1; next
representable R above 1001 refuses. At D=0, R=H=0 passes and H=1 refuses.
Signed cancellation to R=0 with D=2 remains a nonzero-denominator case.

## Executable CPU-only checks

Run from the checkout root with standard-library Python (no GPU or imports of
production modules). The repository test interpreter works for these checks.
This launcher executes the single marked Python block without creating files:

```bash
for check in protocol_schema synthetic_reconciliation synthetic_refusals; do
  env XDG_CACHE_HOME=.cache OMP_NUM_THREADS=1 /mnt/shared/pg-fleet/test-venvs/pq-task-suite-layer-sdk5-20261009/bin/python - "$check" <<'PY'
import pathlib, sys
text = pathlib.Path('docs/design/pq2572-glm-layer-5-9-replay.md').read_text()
start = text.index('# BEGIN PQ2572 CHECKS\n')
end = text.index('# END PQ2572 CHECKS', start)
exec(compile(text[start:end], 'pq2572-protocol-checks', 'exec'))
PY
  status=$?
  if [ "$status" -ne 0 ]; then exit "$status"; fi
done
```

Expected stdout is exactly `protocol_schema: PASS`,
`synthetic_reconciliation: PASS`, `synthetic_refusals: PASS` in order, exit 0.
Schema checks exercise complete bindings, byte/digest authentication, exact
selection and preserved nullable metadata. Reconciliation checks use fixed
external numeric oracles above and recompute all gates, including after stored
flags are tampered with. Refusals cover partial bindings, thresholds, zero
references, missing/duplicate/empty coverage, redistribution and nonfinite
operands or derived values. These validate the protocol, **not historical GLM
reconstruction**. PrismaBuild separately runs the existing attribution, one-pass
spill and one-squaring test modules; this document changes none of their policies.

```python
# BEGIN PQ2572 CHECKS
import copy
import hashlib
import json
import math
import struct
import sys
from collections import Counter

LABELS = ('e240', 'e131')
COMPONENTS = ('weight', 'activation', 'mixed')
ROLES = ('stage_b_cost', 'boundary_activations', 'cotangents', 'source_weights',
         'rendered_weights', 'calibration_tokens', 'probe_identity',
         'activation_quantization', 'rendered_assignment',
         'execution_configuration', 'container_identity', 'capture_records')
TUPLE_KEYS = {'qualified_unit_name', 'layer', 'projection', 'format'}
COORDS = ('sequence', 'token', 'routing_slot')
SIZES = {'float32': 4, 'float64': 8, 'bfloat16': 2, 'float16': 2,
         'int64': 8, 'int32': 4, 'uint8': 1}


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def text_value(value):
    return isinstance(value, str) and bool(value)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def sha256(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def finite(value):
    need(type(value) in (int, float), 'missing or nonnumeric number')
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError('nonfinite number') from error
    need(math.isfinite(result), 'nonfinite number')
    return result


def finite_metadata(value):
    if type(value) in (int, float):
        finite(value)
    elif isinstance(value, dict):
        for item in value.values():
            finite_metadata(item)
    elif isinstance(value, list):
        for item in value:
            finite_metadata(item)


def fields(value, names):
    need(isinstance(value, dict) and set(names) <= set(value),
         'incomplete required binding fields')


def artifact_bytes(artifact, read):
    fields(artifact, ('location', 'bytes', 'sha256'))
    digest = artifact['sha256']
    need(text_value(artifact['location']) and integer(artifact['bytes'])
         and sha256(digest), 'invalid artifact')
    raw = read(artifact['location'])
    need(isinstance(raw, bytes) and len(raw) == artifact['bytes']
         and hashlib.sha256(raw).hexdigest() == digest, 'artifact authentication')
    return raw


def binding_value(binding, read):
    fields(binding, ('value', 'artifact', 'missing_reason'))
    need(isinstance(binding['value'], dict) and binding['artifact'] is not None
         and binding['missing_reason'] is None, 'unavailable binding')
    raw = artifact_bytes(binding['artifact'], read)
    parsed = json.loads(raw)
    finite_metadata(parsed)
    need(parsed == binding['value'], 'receipt value differs')
    return parsed


def identity(value):
    need(isinstance(value, dict) and set(value) == TUPLE_KEYS, 'incomplete tuple')
    need(type(value['layer']) is int and value['layer'] in (5, 9), 'layer scope')
    need(all(text_value(value[k]) and '*' not in value[k]
             for k in ('qualified_unit_name', 'projection', 'format')), 'tuple scope')
    return value


def select(manifest, label, requested, read):
    need(type(label) is str and label in LABELS, 'label scope')
    registered = identity(binding_value(manifest['target_registry'][label], read))
    need(identity(requested) == registered, 'exact target tuple differs')
    return registered


def tensor_records(value, read):
    fields(value, ('records',))
    need(isinstance(value['records'], list) and value['records'], 'missing operands')
    for record in value['records']:
        fields(record, ('tensor_identity', 'shape', 'dtype', 'artifact'))
        shape, dtype = record['shape'], record['dtype']
        need(text_value(record['tensor_identity']) and isinstance(shape, list)
             and shape and all(integer(n) for n in shape) and dtype in SIZES,
             'missing tensor identity, shape or dtype')
        raw = artifact_bytes(record['artifact'], read)
        need(len(raw) == math.prod(shape) * SIZES[dtype], 'operand geometry')


def record_key(record):
    fields(record, ('probe', 'capture_batch', 'invocation_order', 'dispatch_rows',
                    *COORDS, 'evidence'))
    need(all(integer(record[k]) for k in
             ('probe', 'capture_batch', 'invocation_order', 'dispatch_rows')),
         'invalid capture coordinates')
    need(all(record[k] is None or integer(record[k]) for k in COORDS),
         'invalid optional coordinates')
    return (record['probe'], record['capture_batch'], record['invocation_order'])


def evidence(records, read):
    need(isinstance(records, list) and records, 'missing evidence')
    for artifact in records:
        artifact_bytes(artifact, read)


def validate_target(manifest, label, read):
    target = manifest['targets'][label]
    fields(target, ('identity', 'bindings'))
    bound = select(manifest, label, target['identity'], read)
    need(set(target['bindings']) == set(ROLES), 'missing binding roles')
    values = {role: binding_value(target['bindings'][role], read) for role in ROLES}
    for role in ('boundary_activations', 'cotangents', 'source_weights',
                 'rendered_weights', 'calibration_tokens'):
        tensor_records(values[role], read)
    for role in ('boundary_activations', 'cotangents'):
        fields(values[role], ('boundary',))
        need(integer(values[role]['boundary']), 'missing boundary identity')
    for role, key in (('source_weights', 'model_identity'),
                      ('rendered_weights', 'renderer_identity')):
        fields(values[role], (key,))
        need(text_value(values[role][key]), 'missing weight identity')
    need([r['shape'] for r in values['source_weights']['records']] ==
         [r['shape'] for r in values['rendered_weights']['records']], 'weight shapes')
    calibration = values['calibration_tokens']
    fields(calibration, ('calibration_identity', 'sequence_length', 'n_sequences',
                         'token_scope', 'global_token_count'))
    need(text_value(calibration['calibration_identity']) and
         all(integer(calibration[k], 1) for k in
             ('sequence_length', 'n_sequences', 'global_token_count')) and
         calibration['token_scope'] in ('all', 'last', 'causal'), 'calibration fields')
    selected = {'all': calibration['sequence_length'], 'last': 1,
                'causal': calibration['sequence_length'] - 1}[calibration['token_scope']]
    need(selected > 0 and calibration['global_token_count'] ==
         selected * calibration['n_sequences'], 'global normalization geometry')
    need(len(calibration['records']) == 1 and calibration['records'][0]['shape'] ==
         [calibration['n_sequences'], calibration['sequence_length']], 'token tensor geometry')
    probe = values['probe_identity']
    fields(probe, ('layout', 'seeds', 'normalization', 'arithmetic', 'original_identity'))
    need(text_value(probe['layout']) and isinstance(probe['seeds'], list)
         and probe['seeds'] and all(integer(seed) for seed in probe['seeds'])
         and isinstance(probe['original_identity'], dict) and probe['original_identity'],
         'incomplete probe identity')
    fields(probe['normalization'], ('global_token_count', 'rule'))
    need(integer(probe['normalization']['global_token_count'], 1)
         and probe['normalization']['global_token_count'] == calibration['global_token_count']
         and text_value(probe['normalization']['rule']), 'normalization differs')
    fields(probe['arithmetic'], ('component_addition', 'square'))
    need(probe['arithmetic']['component_addition'] == '(W+A)+M'
         and probe['arithmetic']['square'] == 'multiply', 'arithmetic differs')
    original_probe = probe['original_identity']
    fields(original_probe, ('schema', 'source_model', 'calibration_sha256',
                           'calibration_shape', 'calibration_dtype', 'n_probes',
                           'seed_base', 'token_scope', 'temperature', 'distribution',
                           'normalization', 'producer_source_sha256',
                           'source_execution', 'arithmetic'))
    need(original_probe['schema'] == 'prismaquant.joint_aura.probes.v2'
         and integer(original_probe['n_probes'], 1)
         and original_probe['n_probes'] == len(probe['seeds'])
         and integer(original_probe['seed_base'])
         and probe['seeds'] == [original_probe['seed_base'] + p
                                for p in range(original_probe['n_probes'])], 'probe seeds differ')
    need(original_probe['calibration_sha256'] == calibration['records'][0]['artifact']['sha256']
         and original_probe['calibration_shape'] == calibration['records'][0]['shape']
         and text_value(original_probe['calibration_dtype'])
         and original_probe['token_scope'] == calibration['token_scope']
         and original_probe['distribution'] == 'rademacher'
         and finite(original_probe['temperature']) > 0
         and original_probe['normalization'] == probe['normalization']['rule']
         and sha256(original_probe['producer_source_sha256']), 'probe calibration differs')
    need(isinstance(original_probe['source_model'], dict) and original_probe['source_model']
         and isinstance(original_probe['source_execution'], dict)
         and original_probe['source_execution'], 'missing original source identity')
    fields(original_probe['arithmetic'], ('projection_dtype', 'delta_dtype', 'aggregation'))
    need(original_probe['arithmetic']['projection_dtype'] == 'torch.float32'
         and original_probe['arithmetic']['delta_dtype'] == 'torch.float32'
         and original_probe['arithmetic']['aggregation'] == 'sum_signed_invocations_then_square',
         'original arithmetic differs')
    activation = values['activation_quantization']
    fields(activation, ('quantizes_input', 'quantizer', 'original_identity'))
    need(type(activation['quantizes_input']) is bool and text_value(activation['quantizer']),
         'activation identity')
    original = activation['original_identity']
    fields(original, ('schema', 'quantizes_input', 'act_bits', 'act_dtype_name',
                      'act_group_size', 'quantizer', 'static_contract',
                      'activation_max_abs', 'input_global_scale', 'clip_enabled',
                      'served_scales_enabled'))
    need(original['schema'] == 'prismaquant.joint_aura.activation.v1'
         and type(original['quantizes_input']) is bool
         and type(original['clip_enabled']) is bool
         and type(original['served_scales_enabled']) is bool
         and original['quantizes_input'] == activation['quantizes_input']
         and original['quantizer'] == activation['quantizer'], 'activation receipt differs')
    assignment = values['rendered_assignment']
    fields(assignment, ('assignment_identity', 'unit_formats', 'original_assignment'))
    need(text_value(assignment['assignment_identity'])
         and isinstance(assignment['unit_formats'], dict)
         and assignment['unit_formats'].get(bound['qualified_unit_name']) == bound['format']
         and isinstance(assignment['original_assignment'], dict)
         and assignment['original_assignment'], 'rendered assignment differs')
    execution = values['execution_configuration']
    fields(execution, ('source_commit', 'projection_backend', 'probe_microbatch',
                       'capture_batch', 'original_configuration'))
    need(text_value(execution['source_commit']) and text_value(execution['projection_backend'])
         and integer(execution['probe_microbatch']) and integer(execution['capture_batch'], 1)
         and isinstance(execution['original_configuration'], dict)
         and execution['original_configuration'], 'execution fields')
    container = values['container_identity']
    fields(container, ('image_digest', 'image_reference', 'runtime_versions', 'modeling_digest'))
    need(text_value(container['image_reference']) and sha256(container['modeling_digest'])
         and isinstance(container['image_digest'], str)
         and container['image_digest'].startswith('sha256:')
         and sha256(container['image_digest'][7:])
         and isinstance(container['runtime_versions'], dict) and container['runtime_versions'],
         'container identity')
    cost = values['stage_b_cost']
    fields(cost, ('row_locator', 'original_artifact', 'original_row', 'components', 'joint_price'))
    need(text_value(cost['row_locator']) and isinstance(cost['original_row'], dict)
         and cost['original_row'], 'missing original cost row')
    need(json.loads(artifact_bytes(cost['original_artifact'], read)) == cost['original_row'],
         'original cost bytes differ')
    fields(cost['original_row'], ('joint_operator_identity', 'signed_components_per_probe',
                                  'predicted_dloss', 'probe_identity'))
    operator = cost['original_row']['joint_operator_identity']
    fields(operator, ('qname', 'format', 'activation'))
    need(operator['qname'] == bound['qualified_unit_name']
         and operator['format'] == bound['format']
         and operator['activation'] == activation['original_identity']
         and cost['original_row']['predicted_dloss'] == cost['joint_price']
         and cost['original_row']['probe_identity'] == probe['original_identity'],
         'cost extraction differs')
    n = len(probe['seeds'])
    need(isinstance(cost['components'], list) and len(cost['components']) == n,
         'missing historical components')
    need(isinstance(cost['original_row']['signed_components_per_probe'], list)
         and len(cost['original_row']['signed_components_per_probe']) == n,
         'missing original signed components')
    for p, component in enumerate(cost['components']):
        fields(component, ('probe', *COMPONENTS))
        need(type(component['probe']) is int and component['probe'] == p, 'historical probes')
        native = cost['original_row']['signed_components_per_probe'][p]
        fields(native, COMPONENTS)
        for c in COMPONENTS:
            need(finite(component[c]) == finite(native[c]), 'component extraction differs')
    need(finite(cost['joint_price']) >= 0, 'negative historical joint price')
    fields(values['capture_records'], ('expected',))
    expected = values['capture_records']['expected']
    need(isinstance(expected, list) and expected, 'missing capture closure')
    keys = [record_key(record) for record in expected]
    need(len(keys) == len(set(keys)) and {key[0] for key in keys} == set(range(n)),
         'duplicate or missing expected probes')
    for record in expected:
        evidence(record['evidence'], read)
    return values


def validate_manifest(manifest, read):
    fields(manifest, ('schema', 'mode', 'target_registry', 'targets'))
    need(manifest['schema'] == 'pq2572.inputs.v1'
         and manifest['mode'] in ('historical', 'synthetic'), 'manifest schema')
    need(isinstance(manifest['targets'], dict) and set(manifest['targets']) == set(LABELS)
         and isinstance(manifest['target_registry'], dict)
         and set(manifest['target_registry']) == set(LABELS), 'target roster')
    return {label: validate_target(manifest, label, read) for label in LABELS}


def metric(actual, historical, denominator):
    actual, historical, denominator = map(finite, (actual, historical, denominator))
    need(denominator >= 0, 'negative denominator')
    residual = finite(actual - historical)
    absolute = finite(abs(residual))
    bound = finite(0.001 * denominator)
    relative = finite(absolute / denominator) if denominator else None
    passed = actual == historical if denominator == 0 else absolute <= bound
    return dict(reconstructed=actual, historical=historical, denominator=denominator,
                signed_residual=residual, absolute_residual=absolute,
                bound=bound, relative_residual=relative, passed=passed)


def checked_fsum(values):
    try:
        return finite(math.fsum(finite(v) for v in values))
    except OverflowError as error:
        raise ValueError('nonfinite derived sum') from error


def ordered_total(parts):
    return finite(finite(parts['weight'] + parts['activation']) + parts['mixed'])


def price(totals):
    need(bool(totals), 'empty probes')
    squares = [finite(finite(t) * finite(t)) for t in totals]
    return finite(0.5 * checked_fsum(squares) / len(squares))


def reconcile(manifest, row_log, read):
    result = dict(schema='pq2572.report.v1', mode=manifest.get('mode'),
                  historical_status='unmeasured', arithmetic_accepted=False,
                  measurement_accepted=False, targets={}, verdict='no verdict', errors=[])
    try:
        values = validate_manifest(manifest, read)
        fields(row_log, ('schema', 'rows'))
        need(row_log['schema'] == 'pq2572.rows.v1' and isinstance(row_log['rows'], list),
             'row-log schema')
        need(all(isinstance(row, dict) and row.get('target') in LABELS
                 for row in row_log['rows']), 'unexpected row target')
        for label in LABELS:
            rows = [r for r in row_log['rows'] if r['target'] == label]
            expected = values[label]['capture_records']['expected']
            keys = [record_key(r) for r in rows]
            expected_keys = [record_key(e) for e in expected]
            counts, expected_counts = Counter(keys), Counter(expected_keys)
            coverage = dict(expected=len(expected), observed=len(rows),
                            missing=list((expected_counts - counts).elements()),
                            unexpected=list((counts - expected_counts).elements()),
                            duplicate=[k for k, v in counts.items() if v > 1],
                            order_matches=keys == expected_keys)
            result['targets'][label] = dict(coverage=coverage, arithmetic_accepted=False)
            need(counts == expected_counts and not coverage['duplicate']
                 and coverage['order_matches'], 'missing, duplicate or reordered coverage')
            for row, entry in zip(rows, expected):
                need(identity(row['identity']) == manifest['targets'][label]['identity'], 'row tuple differs')
                need(all(row[k] == entry[k] for k in ('dispatch_rows', *COORDS)),
                     'unsupported coordinates or dispatch count')
                evidence(row['evidence'], read)
                for c in COMPONENTS:
                    finite(row[c])
                if entry['dispatch_rows'] == 0:
                    need(all(row[c] == 0 for c in COMPONENTS), 'empty dispatch nonzero')
            probes = []
            historical = values[label]['stage_b_cost']
            for p, reference in enumerate(historical['components']):
                parts, gates = {}, {}
                for c in COMPONENTS:
                    xs = [finite(r[c]) for r in rows if r['probe'] == p]
                    parts[c] = checked_fsum(xs)
                    denominator = checked_fsum(abs(x) for x in xs)
                    gates[c] = metric(parts[c], reference[c], denominator)
                probes.append(dict(probe=p, components=gates, total=ordered_total(parts)))
            joint = price([p['total'] for p in probes])
            activation = price([p['components']['activation']['reconstructed'] for p in probes])
            joint_gate = metric(joint, historical['joint_price'], abs(historical['joint_price']))
            accepted = joint_gate['passed'] and all(g['passed'] for p in probes
                                                    for g in p['components'].values())
            result['targets'][label] = dict(identity=manifest['targets'][label]['identity'],
                                            coverage=coverage, probes=probes,
                                            joint_price=joint_gate, activation_price=activation,
                                            arithmetic_accepted=accepted)
        result['arithmetic_accepted'] = all(result['targets'][label]['arithmetic_accepted']
                                            for label in LABELS)
    except (ValueError, KeyError, TypeError, OverflowError, OSError) as error:
        result['errors'].append(str(error))
    return result


def unmeasured_manifest():
    reasons = {
        'stage_b_cost': 'Exact target row and its original cost bytes not acquired',
        'boundary_activations': 'Original capture entries and boundary operands unavailable locally',
        'cotangents': 'Original probe/capture cotangent operands unavailable locally',
        'source_weights': 'Exact source-model tensor identity and operands not authenticated',
        'rendered_weights': 'Exact historical rendered candidate operands not authenticated',
        'calibration_tokens': 'Target-specific token bytes and ordered calibration identity unavailable',
        'probe_identity': 'Target-specific complete original probe identity not authenticated',
        'activation_quantization': 'Target-specific original activation policy/scale receipt unavailable',
        'rendered_assignment': 'Exact historical rendered assignment not authenticated',
        'execution_configuration': 'Target-specific original execution configuration not authenticated',
        'container_identity': 'Exact historical image/runtime/modeling binding not authenticated',
        'capture_records': 'Original closed capture/dispatch roster including emptiness unavailable',
    }
    def absent(reason):
        return dict(value=None, artifact=None, missing_reason=reason)
    return dict(schema='pq2572.inputs.v1', mode='historical',
                target_registry={label: absent(label + ': exact historical tuple unavailable')
                                 for label in LABELS},
                targets={label: dict(identity=None,
                                     bindings={role: absent(label + ': ' + reasons[role])
                                               for role in ROLES}) for label in LABELS})


def fixture():
    blobs = {}
    def put(raw):
        location = 'synthetic://blob/' + str(len(blobs))
        blobs[location] = raw
        return dict(location=location, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    def bind(value):
        raw = json.dumps(value, sort_keys=True, allow_nan=False).encode()
        return dict(value=copy.deepcopy(value), artifact=put(raw), missing_reason=None)
    def tensor(name, shape, dtype, raw):
        return dict(tensor_identity=name, shape=shape, dtype=dtype, artifact=put(raw))
    manifest = dict(schema='pq2572.inputs.v1', mode='synthetic', target_registry={}, targets={})
    row_log = dict(schema='pq2572.rows.v1', rows=[])
    samples = {
        'e240': [[(4, 1, 1), (-3, -2, 1), (0, 0, 0)],
                 [(1, 2, -1), (-1, 1, 1), (0, 0, 0)]],
        'e131': [[(2, -1, 0), (0, 0, 0), (0, 0, 0)],
                 [(-2, 1, 0), (0, 0, 0), (0, 0, 0)]],
    }
    references = {'e240': [(1, -1, 2), (0, 3, 0)], 'e131': [(2, -1, 0), (-2, 1, 0)]}
    prices = {'e240': 3.25, 'e131': 0.5}
    for label, layer in (('e240', 5), ('e131', 9)):
        unit = dict(qualified_unit_name='synthetic.layer' + str(layer) + '.' + label,
                    layer=layer, projection='synthetic_down', format='SYNTHETIC_F32')
        manifest['target_registry'][label] = bind(unit)
        expected = []
        for p, blocks in enumerate(samples[label]):
            for order, contributions in enumerate(blocks):
                capture = dict(probe=p, capture_batch=order, invocation_order=order,
                               dispatch_rows=0 if order == 2 else 1,
                               sequence=None, token=None, routing_slot=None,
                               evidence=[put(json.dumps(dict(target=label, probe=p,
                                                             dispatch=order)).encode())])
                expected.append(capture)
                row_log['rows'].append(dict(target=label, identity=copy.deepcopy(unit),
                                            **copy.deepcopy(capture),
                                            **dict(zip(COMPONENTS, contributions))))
        tokens = tensor(label + '.tokens', [2, 2], 'int64', struct.pack('<4q', 1, 2, 3, 4))
        probe_original = dict(synthetic=True, schema='prismaquant.joint_aura.probes.v2',
                              source_model=dict(synthetic_model=label + '-synthetic-model'),
                              source_execution=dict(synthetic=True),
                              calibration_sha256=tokens['artifact']['sha256'],
                              calibration_identity=label + '-synthetic-calibration',
                              calibration_shape=[2, 2], global_token_count=4,
                              calibration_dtype='torch.int64', n_probes=2, seed_base=11,
                              token_scope='all', temperature=1, distribution='rademacher',
                              normalization='global_kl_fisher', producer_source_sha256='c' * 64,
                              arithmetic=dict(projection_dtype='torch.float32', delta_dtype='torch.float32',
                                              aggregation='sum_signed_invocations_then_square',
                                              operator_windows=None))
        activation_original = dict(schema='prismaquant.joint_aura.activation.v1',
                                   quantizes_input=False, act_bits=16, act_dtype_name='BF16',
                                   act_group_size=None, quantizer='identity', static_contract=None,
                                   activation_max_abs=None, input_global_scale=None,
                                   clip_enabled=False, served_scales_enabled=False)
        components = [dict(probe=p, **dict(zip(COMPONENTS, parts)))
                      for p, parts in enumerate(references[label])]
        original = dict(joint_operator_identity=dict(qname=unit['qualified_unit_name'],
                                                     format=unit['format'], activation=activation_original),
                        signed_components_per_probe=[dict(zip(COMPONENTS, parts)) for parts in references[label]],
                        predicted_dloss=prices[label], probe_identity=probe_original,
                        operator_windows=None)
        original_artifact = put(json.dumps(original, sort_keys=True, allow_nan=False).encode())
        matrix = struct.pack('<4f', 1, 2, 3, 4)
        weight = [tensor(label + '.weight', [2, 2], 'float32', matrix)]
        values = dict(
            stage_b_cost=dict(row_locator='synthetic-row', original_artifact=original_artifact,
                              original_row=original, components=components, joint_price=prices[label]),
            boundary_activations=dict(boundary=layer, records=[tensor(label + '.x', [2, 2], 'float32', matrix)]),
            cotangents=dict(boundary=layer + 1, records=[tensor(label + '.g', [2, 2], 'float32', matrix)]),
            source_weights=dict(model_identity=label + '-synthetic-model', records=weight),
            rendered_weights=dict(renderer_identity='synthetic-renderer', records=copy.deepcopy(weight)),
            calibration_tokens=dict(calibration_identity=label + '-synthetic-calibration',
                                    sequence_length=2, n_sequences=2, global_token_count=4,
                                    token_scope='all', records=[tokens]),
            probe_identity=dict(layout='global-row', seeds=[11, 12], original_identity=probe_original,
                                normalization=dict(global_token_count=4, rule='global_kl_fisher'),
                                arithmetic=dict(component_addition='(W+A)+M', square='multiply')),
            activation_quantization=dict(quantizes_input=False, quantizer='identity', original_identity=activation_original),
            rendered_assignment=dict(assignment_identity=label + '-synthetic-assignment',
                                     unit_formats={unit['qualified_unit_name']: unit['format']},
                                     original_assignment=dict(synthetic=True)),
            execution_configuration=dict(source_commit='synthetic-commit', projection_backend='synthetic-scalars',
                                         probe_microbatch=1, capture_batch=1,
                                         original_configuration=dict(sequence_attribution=None, operator_windows=None)),
            container_identity=dict(image_digest='sha256:' + 'a' * 64, image_reference='synthetic-image',
                                    runtime_versions=dict(python='synthetic'), modeling_digest='b' * 64),
            capture_records=dict(expected=expected))
        manifest['targets'][label] = dict(identity=unit, bindings={role: bind(values[role]) for role in ROLES})
    return manifest, row_log, blobs, bind


def refusal(action, fragment):
    try:
        action()
    except (ValueError, KeyError, TypeError) as error:
        need(fragment in str(error), 'wrong refusal: ' + str(error))
    else:
        raise AssertionError('expected refusal: ' + fragment)


def protocol_schema():
    manifest, rows, blobs, bind = fixture()
    values = validate_manifest(manifest, blobs.__getitem__)
    assert set(values) == {'e240', 'e131'}
    for label in LABELS:
        assert select(manifest, label, manifest['targets'][label]['identity'], blobs.__getitem__)['layer'] in (5, 9)
        assert values[label]['stage_b_cost']['original_row']['operator_windows'] is None
        assert values[label]['activation_quantization']['original_identity']['static_contract'] is None
        assert values[label]['probe_identity']['original_identity']['arithmetic']['operator_windows'] is None
    historical = unmeasured_manifest()
    assert reconcile(historical, rows, blobs.__getitem__)['arithmetic_accepted'] is False
    for label in LABELS:
        for role in ROLES:
            entry = historical['targets'][label]['bindings'][role]
            assert entry['value'] is None and entry['artifact'] is None and entry['missing_reason']
    for label in ('E240', ' e240', 'e240 ', 'expert240', '*', 'e999', ('e240',)):
        refusal(lambda: select(manifest, label, manifest['targets']['e240']['identity'], blobs.__getitem__), 'label scope')
    for key, invalid in (('qualified_unit_name', 'synthetic.alias'), ('qualified_unit_name', '*'), ('projection', 'up'),
                         ('format', 'OTHER'), ('layer', 9), ('layer', 5.0), ('layer', True)):
        requested = {**manifest['targets']['e240']['identity'], key: invalid}
        refusal(lambda: select(manifest, 'e240', requested, blobs.__getitem__),
                ('layer scope' if key == 'layer' and type(invalid) is not int
                 else 'tuple scope' if invalid == '*' else 'tuple differs'))
    original = manifest['targets']['e240']['bindings']['source_weights']
    blobs[original['artifact']['location']] += b' '
    refusal(lambda: validate_manifest(manifest, blobs.__getitem__), 'artifact authentication')


def synthetic_reconciliation():
    manifest, rows, blobs, bind = fixture()
    report = reconcile(manifest, rows, blobs.__getitem__)
    assert report['errors'] == [] and report['arithmetic_accepted'] is True
    assert report['measurement_accepted'] is False
    e240, e131 = (report['targets'][label] for label in LABELS)
    assert [p['total'] for p in e240['probes']] == [2.0, 3.0]
    assert e240['joint_price']['reconstructed'] == 3.25
    assert e240['activation_price'] == 2.5
    assert e131['joint_price']['reconstructed'] == 0.5 and e131['activation_price'] == 0.5
    assert [[p['components'][c]['denominator'] for c in COMPONENTS] for p in e240['probes']] == [[7, 3, 2], [2, 3, 2]]
    premature = 0.5 * math.fsum(ordered_total(r) * ordered_total(r)
                              for r in rows['rows'] if r['target'] == 'e240') / 2
    assert premature == 14.25 and premature != e240['joint_price']['reconstructed']
    rows['arithmetic_accepted'] = False
    for row in rows['rows']:
        row['passed'] = False
    assert reconcile(manifest, rows, blobs.__getitem__)['arithmetic_accepted'] is True
    for r in rows['rows']:
        if r['target'] == 'e240':
            r.update(weight=0, activation=0, mixed=0)
    first = rows['rows'][0]
    first.update(weight=1e16, activation=1.0, mixed=-1e16)
    cost = copy.deepcopy(manifest['targets']['e240']['bindings']['stage_b_cost']['value'])
    cost['components'] = [dict(probe=0, weight=1e16, activation=1.0, mixed=-1e16),
                          dict(probe=1, weight=0, activation=0, mixed=0)]
    cost['joint_price'] = 0
    cost['original_row'].update(signed_components_per_probe=[{c: item[c] for c in COMPONENTS}
                                                           for item in cost['components']], predicted_dloss=0)
    cost['original_artifact'] = bind(cost['original_row'])['artifact']
    manifest['targets']['e240']['bindings']['stage_b_cost'] = bind(cost)
    cancellation = reconcile(manifest, rows, blobs.__getitem__)
    assert cancellation['errors'] == [] and cancellation['arithmetic_accepted'] is True
    assert cancellation['targets']['e240']['joint_price']['reconstructed'] == 0
    assert cancellation['targets']['e240']['activation_price'] == 0.25
    assert checked_fsum([1e16, 1.0, -1e16]) == 1.0


def synthetic_refusals():
    assert metric(1000, 1001, 1000)['passed'] is True
    assert metric(1000, math.nextafter(1001.0, math.inf), 1000)['passed'] is False
    assert metric(1001, 1000, 1000)['passed'] is True
    assert metric(math.nextafter(1001.0, math.inf), 1000, 1000)['passed'] is False
    assert metric(0, 0, 0)['passed'] is True
    assert metric(0, 1, 0)['passed'] is False
    assert metric(1, 0, 0)['passed'] is False
    assert metric(0, 0, 2)['passed'] is True
    for role in ROLES:
        manifest, rows, blobs, bind = fixture()
        manifest['targets']['e240']['bindings'][role] = dict(
            value=None, artifact=None, missing_reason='synthetic missing evidence')
        assert reconcile(manifest, rows, blobs.__getitem__)['arithmetic_accepted'] is False, role
    for role in ROLES:
        manifest, rows, blobs, bind = fixture()
        partial = manifest['targets']['e240']['bindings'][role]['value']
        del partial[next(iter(partial))]
        manifest['targets']['e240']['bindings'][role] = bind(partial)
        assert reconcile(manifest, rows, blobs.__getitem__)['arithmetic_accepted'] is False, role
    manifest, rows, blobs, bind = fixture()
    target = manifest['targets']['e240']
    target['bindings']['source_weights'] = bind(dict(model_identity='e240-synthetic-model'))
    assert reconcile(manifest, rows, blobs.__getitem__)['arithmetic_accepted'] is False
    for field in ('tensor_identity', 'shape', 'dtype', 'artifact'):
        manifest, rows, blobs, bind = fixture()
        value = manifest['targets']['e240']['bindings']['source_weights']['value']
        del value['records'][0][field]
        manifest['targets']['e240']['bindings']['source_weights'] = bind(value)
        assert reconcile(manifest, rows, blobs.__getitem__)['arithmetic_accepted'] is False
    for field in ('n_probes', 'seed_base', 'calibration_shape', 'source_model', 'arithmetic'):
        manifest, rows, blobs, bind = fixture()
        value = manifest['targets']['e240']['bindings']['probe_identity']['value']
        del value['original_identity'][field]
        manifest['targets']['e240']['bindings']['probe_identity'] = bind(value)
        assert reconcile(manifest, rows, blobs.__getitem__)['arithmetic_accepted'] is False
    for mutation in ('missing', 'duplicate', 'order', 'empty', 'coordinate', 'expected_duplicate'):
        manifest, rows, blobs, bind = fixture()
        if mutation == 'missing':
            del rows['rows'][2]
        elif mutation == 'duplicate':
            rows['rows'].insert(1, copy.deepcopy(rows['rows'][0]))
        elif mutation == 'order':
            rows['rows'][0], rows['rows'][1] = rows['rows'][1], rows['rows'][0]
        elif mutation == 'empty':
            rows['rows'][2]['weight'] = 1
        elif mutation == 'coordinate':
            rows['rows'][0]['sequence'] = 0
        else:
            capture = manifest['targets']['e240']['bindings']['capture_records']['value']
            capture['expected'].append(copy.deepcopy(capture['expected'][0]))
            manifest['targets']['e240']['bindings']['capture_records'] = bind(capture)
        rows['arithmetic_accepted'] = True
        for row in rows['rows']:
            row['passed'] = True
        assert reconcile(manifest, rows, blobs.__getitem__)['arithmetic_accepted'] is False, mutation
    manifest, rows, blobs, bind = fixture()
    rows['rows'][0]['weight'] += 1
    rows['rows'][0]['activation'] -= 1
    report = reconcile(manifest, rows, blobs.__getitem__)
    assert report['targets']['e240']['joint_price']['passed'] is True
    assert report['targets']['e240']['probes'][0]['components']['weight']['passed'] is False
    assert report['arithmetic_accepted'] is False
    manifest, rows, blobs, bind = fixture()
    cost = manifest['targets']['e131']['bindings']['stage_b_cost']['value']
    cost['joint_price'] = 0.6
    cost['original_row']['predicted_dloss'] = 0.6
    cost['original_artifact'] = bind(cost['original_row'])['artifact']
    manifest['targets']['e131']['bindings']['stage_b_cost'] = bind(cost)
    report = reconcile(manifest, rows, blobs.__getitem__)
    assert report['targets']['e240']['arithmetic_accepted'] is True
    assert report['targets']['e131']['joint_price']['passed'] is False
    assert report['arithmetic_accepted'] is False
    for invalid in (math.nan, math.inf, -math.inf, None, True, '1'):
        manifest, rows, blobs, bind = fixture()
        rows['rows'][0]['weight'] = invalid
        assert reconcile(manifest, rows, blobs.__getitem__)['arithmetic_accepted'] is False
    for c in COMPONENTS:
        manifest, rows, blobs, bind = fixture()
        del rows['rows'][0][c]
        assert reconcile(manifest, rows, blobs.__getitem__)['arithmetic_accepted'] is False
    for invalid in (math.nan, math.inf, -math.inf):
        manifest, rows, blobs, bind = fixture()
        entry = manifest['targets']['e240']['bindings']['stage_b_cost']
        entry['value']['joint_price'] = invalid
        raw = json.dumps(entry['value'], sort_keys=True).encode()
        entry['artifact']['bytes'] = len(raw)
        entry['artifact']['sha256'] = hashlib.sha256(raw).hexdigest()
        blobs[entry['artifact']['location']] = raw
        assert reconcile(manifest, rows, blobs.__getitem__)['arithmetic_accepted'] is False
    manifest, rows, blobs, bind = fixture()
    rows['rows'][0]['weight'] = 1e308
    rows['rows'][1]['weight'] = 1e308
    assert reconcile(manifest, rows, blobs.__getitem__)['arithmetic_accepted'] is False
    rows['rows'][1]['weight'] = -1e308
    assert reconcile(manifest, rows, blobs.__getitem__)['arithmetic_accepted'] is False
    manifest, rows, blobs, bind = fixture()
    rows['rows'][0].update(weight=-1e200, activation=1e200, mixed=0)
    assert reconcile(manifest, rows, blobs.__getitem__)['arithmetic_accepted'] is False
    refusal(lambda: price([1e200]), 'nonfinite')
    refusal(lambda: price([1e154, 1e154]), 'nonfinite')
    refusal(lambda: ordered_total(dict(weight=1e308, activation=1e308, mixed=-1e308)), 'nonfinite')
    refusal(lambda: metric(1e308, -1e308, 1), 'nonfinite')
    refusal(lambda: price([]), 'empty probes')
    refusal(lambda: metric(1, 0, 5e-324), 'nonfinite')


CHECKS = dict(protocol_schema=protocol_schema,
              synthetic_reconciliation=synthetic_reconciliation,
              synthetic_refusals=synthetic_refusals)
if __name__ == '__main__':
    need(len(sys.argv) == 2 and sys.argv[1] in CHECKS, 'choose a documented check')
    CHECKS[sys.argv[1]]()
    print(sys.argv[1] + ': PASS')
# END PQ2572 CHECKS
```

The checker emits the arithmetic portion of the report, with independently
computed numbers and no historical measurement acceptance. Operations adds the
artifact/reproduction/profiling and verdict fields from the report schema;
those fields are mandatory for a completed historical packet. The CPU checker
is not a historical source-authentication authority, tensor-replay driver,
container executor or causal-verdict engine. Rerunning checks reads immutable
in-memory fixtures and creates no files; it is idempotent.

## Operations follow-up and open questions

Outside U4 time, obtain fleet admission and the known-good historical-compatible
container. The withdrawn replay request grants no current reservation. Acquire
and authenticate the missing bindings for both labels, including capture closure
and original receipt sources. Freeze the complete manifest before replay; retain
actual commands, raw outputs, execution receipts and their digests. Independently
recompute coverage, components, prices and both targets' gates. Publish either
one evidenced cause with a discriminating comparison or literal `no verdict`.
Attach in-process profiling and aligned Netdata evidence from both boxes.

Open questions are requests for exact evidence, not implementation choices:

- Which original registry/cost records authenticate each label's exact qualified
  name, layer, projection and format? Neither complete tuple is bound here.
- Where are each target's authenticated tensor, token, probe, scale, assignment,
  execution and immutable container records, with byte counts and SHA-256?
- Which retained caller/routing records authenticate ordered invocation coverage
  and empty dispatches, and which actually support sequence/token/slot coordinates?
- Where is memo 8.1's authenticated band binding and the matched-block G3
  decomposition? Until supplied, G3 is excluded from causal evidence.
- What independent matched derivative/comparison would discriminate a proposed
  cause after both targets reconcile? Banked-cotangent reconstruction alone
  cannot answer this.
