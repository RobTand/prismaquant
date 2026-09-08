# Full-engine fixed-resource admission: required producer contract

Status: implementation design for #420, 2026-09-08. **Admission remains closed.**
This document specifies the evidence and arithmetic needed before replacing
`runtime_provenance.admit_fixed_resources`' unconditional refusal. It does not
introduce an accepted producer schema or a measured runtime table. No new
GPU, served, latency, quality or capacity measurement was run for this design.

## Six-variant delivery does not require runtime-v2 admission

The fixed-resource gate applies only to the optional allocator input
`--measured-runtime-table`. It is **not a prerequisite for producing and
empirically comparing six independently allocated GLM variants from one common
probe**. The allocator's normal byte/quality path remains available without
that flag or `--measured-runtime-context` (`allocator.py:2096`). Do not invent
an operator-time table, fixed charge or predicted SLO merely to use this path.

The existing direct route is:

1. Freeze the complete source/calibration/common-probe and qualified cost
   artifacts once. Reuse them for six explicitly configured allocations through
   `python -m prismaquant.allocator --probe ... --costs ... --formats ...
   --target-bits ... --layer-config ...` (or the existing explicit disk-budget
   path). Each variant binds its allowed menu, constraints, assignment and
   serialization metadata; verify that six names actually identify six distinct
   assignments/artifacts. The budgets and menus come from the requested recipe
   definitions, not new values chosen by this design.
2. Materialize/export each assignment using the existing lane-specific export
   path, cache owners and exact-byte checks. Close actual load, generation,
   plugin/kernel, requested topology and runtime compatibility gates for each
   artifact. A TP2 research control or a runtime-v2 refusal neither substitutes
   for those gates nor waives them.
3. Compare actual quality against one fixed teacher on the same held-out
   tokens. Existing `tools/measure_served_gold.py dump` and `kl` bind served
   endpoint/manifest identity and replay matching corpus/tokenizer settings;
   their metric is declared top-K-plus-tail KL, not exact full-vocabulary KL.
   `tools/measure_vllm_full_kl.py` offers its separately scoped in-process metric.
   Use only a route that supports the actual target topology and teacher scope;
   hold vocabulary/support, sequence selection and scoring positions constant.
   The common quantization probe is not the held-out quality population.
4. Measure actual end-to-end prefill on each exported variant under the same
   explicit request count/length, batch/concurrency, cache state, graph mode,
   TP topology and host-load policy. Tessera's existing
   `experiments/window_gemv_load.py` shows the supported endpoint/engine-histogram
   mechanism: difference the server's TTFT counts/sums over the driven request
   window and keep profiler windows separate. Its current generated-prompt,
   concurrency-one workload is its own scope; it is not already a GLM
   calibration-workload adapter. Reuse that mechanism or the pinned engine's
   benchmark after binding the requested workload and actual server manifest.
   Record raw requests/results, failures, receipt hashes, both-host Netdata and
   in-process profiles. A histogram mean is not p95 or pure kernel prefill time.
5. Apply the user-selected rule to the six actual KL/prefill measurements,
   with exact bytes, measured fit and existing quality/uniform/downstream gates.
   `select_validated_frontier.py` currently supports measured KL and byte-budget
   selection; it has no direct served-prefill axis. Its metadata-preserving
   extension is separate ongoing work. Do not relabel
   `speed_quality_frontier`'s predicted loss/speed hints as served measurements.
   A direct comparison can select among the measured six without establishing
   an admitted runtime-v2 model or global runtime-optimal allocation.

Practical prerequisites are complete common-probe/cost coverage, six reviewable
assignments, valid exports, a qualified runtime that can load and serve each,
a common fixed-teacher/held-out protocol, a common served prefill workload,
and traceable selection metadata. They are the existing task gates. The
resource-contract work below remains bounded follow-up for predictive allocator
constraints; it must not become a new mandatory stage of this six-variant flow.

## Decision and current evidence

A producer/consumer JSON schema alone cannot close this gate. The current
producer deliberately leaves quantities unknown, and several unknowns affect
what the schema must represent. Implementing a positive path over today's
ledgers would treat unresolved ownership as evidence. The bounded next step
is to qualify the missing observations in the existing Tessera engine harness,
then implement a pure PrismaQuant artifact consumer over those versioned raw
observations. No serving-runtime import, second collector or dispatcher is
needed.

The inspected PrismaQuant base is `47aec941d382ba419666a653fa2436a9ae8d6d33`.
The inspected Tessera producer files are from
`98aa317a06e55731c36c53f509adc04260668cc1`. These sources establish:

| Source | Current contract and consequence |
| --- | --- |
| PQ `runtime_provenance.py:376` | `admit_fixed_resources` reads the receipt and refuses every current claim. Hashing a ledger or setting `complete` cannot satisfy it. |
| PQ `measured_runtime_prices.py:187` and `:446` | Resources have scalar resident, activation, scratch and KV fields. The v2 loader runs relation/native/fixed admission before setting `producer_admitted`. |
| PQ `serve_constraints.py:803` and `:845` | Candidate members must cover the expanded nonfixed assignment once. Resident bytes and times add; candidate activation and scratch use separate maxima above fixed terms. |
| Tessera `experiments/full_engine_worker.py:242` and `:322` | Model, cache, native boundary and runtime-root owners are observed. Runtime roots and native boundary tensors are still `shared`, with assignment/workload dependence unresolved. |
| Tessera `experiments/full_engine_resources.py:619` | Raw replay retains generation identities, aliases, escaping allocations and unknown domains, but emits null resources/timings and `admission: not_implemented`. |
| Tessera `experiments/native_operator_resources.py:1` and `:185` | Native scratch includes returned output. It excludes allocator reservation slack, context/library startup and model-global resources. Subtracting it from an engine peak does not identify a fixed charge. |
| Tessera `experiments/full_engine_timings.py:27` and `:166` | Same-event fixed gaps and native intervals exist, with launch/stream checks. Collection health and observer overhead are not qualified; the implementation accepts only cold 512-token prefill then one-token decode. |
| Tessera `experiments/capture_full_engine_resources.py:205` and `:244` | Both resource and timing runners require exactly one worker. Timing arms retain generated tokens and control/partition observations; admission remains disabled. |

The scalar device budget is not a physical-memory model for two TP ranks or
GB10 host/UMA backings. GPU-addressable bytes, CPU RSS and pinned pages can
refer to the same physical storage; summing them double counts, and dropping
host-only/context allocations undercounts. The first supported v2 adapter must
remain TP1, one device, resident, eager, with an explicit GPU-allocation scope.
It must not certify whole-host or UMA fit. If the requested budget means
physical host memory, or if TP2 is required, a versioned resource-vector and
solver/feasibility change is a prerequisite; neither rank sums nor rank maxima
can silently be written into v2 scalar fields.

## Minimal envelope to freeze after observer qualification

All names below describe a proposed contract, not a currently accepted schema.
The eventual receipt must use a new closed schema identifier. Unknown fields,
missing fields, duplicate JSON keys, nonfinite values, negative sizes, boolean
integers, duplicate IDs and unknown enum values refuse. Each reference uses
existing `{path, sha256}` artifact semantics and is rehashed by `ArtifactReader`.
The producer builds the receipt deterministically from raw artifacts; the
consumer independently recomputes its claims without invoking producer code.

| Required envelope member | Independently checked contents |
| --- | --- |
| `identity` | The exact v2 context, cost digest, relation digest, named full-engine run ID and that run's original runtime digest. Retain both identities. Bind original source model and the concrete exported checkpoint separately. |
| `reference` | Complete canonical census; fixed auxiliary assignment; one selected row per serving unit; whole-member `RuntimeBinding`; original wire and exporter artifact identities. This must partition the independently supplied expanded model roster. |
| `workload` | Raw calibration file identity and selected token row; actual prompt IDs, generated decode IDs and sampling; cold/warm cache protocol; batch, scheduled token counts and request boundaries. Recompute the workload digest from this record. A native decode proxy cannot impersonate actual engine decode. |
| `execution` | Selected and actual graph/residency/topology, worker/process/device roster, cache capacity, streams and concurrency. Reuse the original relation and configuration checks. A changed execution coordinate requires fresh evidence. |
| `observations` | Original startup-to-finish resource capture, allocation API/activity arguments, Torch snapshots/history, owner views, KV observations, timing captures/profiles and observer qualification artifacts. Every artifact names its observation run and interval. |
| `partition` | Exact allocation-generation, serialized-extent and timing-interval membership, derived from the observations. Stable semantic owner IDs map to concrete observed generations; no ownership is inferred from a pointer alone. |
| `derived` | Recomputed fixed `RuntimeResources`, candidate charges for the reference rows, domain totals and explicit scope. Declared numbers must equal recomputation; they are never inputs to it. |

Artifact identity is necessary but not sufficient. Qualification cannot be an
opaque proof hash or `qualified: true` field. The consumer must understand the
actual qualification outputs, bind them to exact collector/harness/library
bytes, and verify their assertions against raw inputs. Qualification failures
and unavailable domains remain visible and prohibit scalar projection.

Do not broaden `runtime_provenance_relation.v1` silently: it currently relates
exactly one full-engine run to native runs on the same GPU. If resource,
timing and observer-control runs have different manifests, retain each original
and define an explicit versioned multi-run relation with exhaustive dependencies
and intentional instrumentation differences. Replacing one run's digest with
another's would defeat the current gate.

## Recomputable memory and byte partition

1. Replay every supported allocation domain from before initialization to the
   declared terminal boundary. Use process/context/device, allocation address
   **and generation** to identify lifetimes. Match allocation APIs reciprocally
   to records, preserve delayed frees, and reject missing/dropped/truncated
   records and unhandled APIs. Torch suballocations and their parent CUDA
   segments are two views of the same backing, not additive memory.
2. At each ownership boundary, reconcile raw storage views with live backing
   generations. Every storage has one accounting owner; aliases may have many
   tensor names. Require complete nonoverlapping byte-extent coverage and an
   explicit remainder where padding exists. Reject a physical extent claimed
   by candidate and fixed owners or by two candidate units. Known sharing must
   be represented once at the existing common owner, with all aliases retained.
3. Classify bytes by both **ownership** (candidate unit, fixed, cache) and
   **lifetime/domain** (persistent device, carried activation, invocation-local
   scratch, reservation slack, external/static/context, host backing). The
   current `shared` label supplies neither classification nor invariance.
   Unknown, conflicting or cross-unit escaped ownership blocks admission.
4. Independently bind a candidate's retained weights/scales/metadata to its
   complete native row and source/render/wire identity. Fixed named model
   state binds to `fixed_assignment` or the immutable model roster. Any actual
   candidate owner absent from a row, or any fixed member inside a candidate
   row, refuses. Do not copy the reference assignment's BF16 weight size into
   a different prepared owner contract.
5. Deduplicate KV and recurrent views by physical backing generation and
   capacity. Recompute actual pool sizes and resolved limits from raw worker
   records; bind the explicit selected capacity policy. Prefix reset,
   scheduler limits, dtype, block/page geometry and reserved null blocks are
   part of the workload/configuration. A parameter-count formula or a free
   memory fraction is not the observed KV allocation.
6. Partition serialized bytes from exact exported file/tensor/resource
   extents. Reuse `artifact_collection`'s existing shared content-reference
   deduplication for shared artifacts. A header/container overhead extent
   cannot also be charged to a tensor. Total checkpoint bytes must equal
   candidate plus fixed plus explicitly excluded nonmodel file extents.
   This is separate from device residency.
7. For every supported transient class, sweep allocation/free events and
   compute the maximum **simultaneous** sum over its declared interval.
   Summing per-allocation maxima or subtracting independent peaks cannot
   derive a peak. Persistent bytes, activation bytes, scratch and KV may not
   reuse one physical extent in the same interval.

For a scalar adapter, the existing conservative measured composition is:

`fixed_resident + sum(candidate_resident) + fixed_activation + max(candidate_activation) + fixed_scratch + max(candidate_scratch) + fixed_KV`.

`evaluate_measured_assignment` additionally adds the caller's explicit
`slos.kv_bytes` and `slos.peak_scratch_bytes` reserves. Those reserves must
represent extra capacity beyond the measured terms; labeling an already
charged fixed allocation as a caller reserve does not establish disjoint
ownership.

The measured composition can exceed the measured instantaneous peak because independent maxima
need not coincide. That is disclosed composition conservatism, not permission
to duplicate physical ownership. Allocation reservation slack and supported
context/external backings require their own disjoint observed charge before
projection; nothing is filled with a residual from a whole-engine peak.

The current native row boundary needs qualification before that projection:
its scratch includes returned output, while another unit may consume that
same storage as input and while engine views can retain it across boundaries.
For each reference unit, the producer must reconcile exactly which raw
allocations its native scratch/input charge covers and prove that remaining
carried lifetimes fit the disjoint fixed terms. If that cannot be represented
without duplication or undercounting, introduce a versioned boundary/resource
model for both native and full-engine consumers. Do not silently reinterpret
existing v2 `activation_bytes` or subtract an output size from a peak; the
output need not be live at the peak.

An invariant fixed charge also needs evidence across the candidate scope.
An observation of one reference assignment cannot establish that shared
workspace, cache policy, launch paths or persistent buffers stay unchanged
under every alternative. The first fully bounded admission can cover one
complete assignment (one row per unit). A multi-option table additionally needs
an explicit, source-bound ownership/capacity invariance rule with qualified
substitution controls. Any option that changes shared/fixed ownership either
gets a separately measured assignment/context or requires a richer model.
CPU fixtures cannot establish this invariance for GLM.

## Recomputable timing partition

For each repeated full-engine sample and each prefill/decode phase, preserve
ordered native apply intervals and directly measured adjacent gaps using the
same CUDA-event chain. Recompute that every canonical unit executes exactly
as declared and that intervals plus gaps recompose the whole measured step
within only the documented event-representation rounding. Bind every GPU
launch to its CPU scope, correlation, stream and device. Unjoined streams,
overlapping candidate execution, dropped collection or an unobserved tail
refuse the sequential model. Shared experts, routing and collectives belong
where their actual boundaries put them, not where the planner expects them.

Compute the fixed sample as the sum of that sample's fixed gaps, then take
the median across complete repeated samples. Never subtract a sum of native
medians from an engine median. Preserve at least the existing three-sample
and positive warmup requirements; record every control/partition arm and its
actual generated tokens. Qualify profiler completeness and observer impact
before deriving proposal prices. Thresholds or allowed impact must come from
an explicit versioned measurement policy, not an agent-selected tolerance.

A fixed-gap median plus native-row medians remains an operator-sum proposal.
It does not equal a measured engine quantile, and the existing final served
TTFT/ITL and held-out quality gates remain independent.

## Exact prerequisites and implementation sequence

| Prerequisite | Existing owner to extend | Checkable acceptance |
| --- | --- | --- |
| Freeze target scope | Root's selected GLM artifact/configuration/workload | Export identity, canonical census, calibration IDs, actual worker topology and capacity policy all supplied; no new measurement runs before that freeze. |
| Complete physical and logical ownership | Tessera `full_engine_resources`, existing worker owners and CUDA-domain observers; Tessera #399 | Every supported lifetime and alias reconciles; no `unknown`, unresolved `shared`, escaped candidate allocation or unattributed API remains; allocator rounding/slack and context domains are explicit. |
| Set native/full-engine charge boundary | Existing native resource producer and PQ native consumers | Reference native input/output/scratch extents reconcile with full-engine lifetimes without overlap; counterexample controls exercise output retention and aliases. |
| Qualify cache/backing scope | Existing `full_engine_kv` and worker records | Capacity is recomputed from actual backing identities and selected limits; host/UMA aliasing is explicitly resolved or the memory scope is rejected. |
| Qualify timing observation | Existing timing recorder, profiler analysis and control arms | Complete stream/launch coverage, repeated same-workload event partitions and approved observer-impact policy. No invented times. |
| Relate all observation runs | Existing PQ runtime relation | Original manifests retained, exact production dependencies/config/device bindings checked, separately versioned support for any extra engine/control runs. |
| Freeze and implement producer schema | Tessera report assembly plus PQ pure artifact consumer | Closed fields and enums; independent raw recomputation; deterministic receipt generation; no arbitrary success flag accepted. |
| Integrate allocator admission | `admit_fixed_resources`, v2 loader and existing feasibility evaluation | Only recomputed resources set `producer_admitted`; full assignment and member coverage remain exact; unsupported topology/domain refuses. |

CPU regressions for the eventual implementation must first fail on missing or
incorrect admission: omitted unit/owner/extent; duplicate alias; candidate/fixed
overlap; reused pointer generation; live-free mismatch; missing parent segment;
unknown API; stale source/calibration/workload/runtime; altered cache capacity;
foreign rank/device; missing timing tail; overlapping streams; unsupported
boundary; nonfinite or boolean numeric fields; changed derived totals; and
assignment-dependent shared state. Tampering tests must update outer hashes
so the semantic consumer, not only the checksum reader, catches the defect.
A positive synthetic receipt proves the parser/recomputation contract only.
A positive real table additionally needs the qualified original measurements.

## Delivery boundary

This design resolves the implementation prerequisites for #420; it does not
resolve the missing producer observation or implement admission. #420 and
Tessera #399 remain open. The current refusal is preserved and will be checked
with the existing CPU refusal and architecture suites through PrismaBuild.
No new test merely restates this prose. Actual commands, source hashes and PB
terminal/CAS evidence accompany this change in the measurement record.
