# Selected-result pilot admission

PQ #2106 implements the CPU provenance and consumer portion of #1293. A
fan-out of more than one quantum requires producer counters, an explicit
`--pilot-selector ACTION_KEY PUBLISHED_UNIX ATTEMPT`, and an independently
reviewed `--pilot-source-contract PATH SHA256`. These arguments are repeatable.
Selectors must name exactly the supplied counter actions, once each. One
quantum can produce a pilot; `--force-unverified-pilot` remains an explicit,
persistently stamped override.

The source contract is a trusted review input, supplied separately from the
candidate completion and counters. Its exact field set is:

| Field | Meaning |
| --- | --- |
| `schema` | `prismaquant.joint_dispatch_pilot.source.v1` |
| `snapshot` | Exact `{id: "pbrun.checkout-snapshot", sha256, bytes}` PB input descriptor |
| `snapshot_selection` | Exact full authenticated `params.checkout_snapshot`: schema, commit, parent, subdirectory, refs and input |
| `implementation_sha256` | Reviewed relation between that snapshot and the complete durable PrismaQuant package digest |
| `launcher_argv` | `["python3", "-m", "tools.tessera_campaign_container"]` |
| `quantum_argv` | `["python3", "-m", "prismaquant.joint_cost_quantum"]` |
| `container_spec` | Exact expected inlined container specification, including image and environment |
| `outer_environment` | Exact expected PB request environment, excluding only PB's execution-specific container owner and marker |

A reviewer must establish the snapshot/package relation independently, from
the snapshot bytes and the intended code, before accepting the contract.
Neither a counter's implementation digest nor a candidate's descriptor can
create that relation. The contract digest supplied to the CLI binds that
review input. Contracts are deduplicated by the caller; zero or multiple
matching contracts refuse. A source or selection change needs a newly reviewed contract.
This is source qualification, independent of worker-generation deployment.

The existing pinned SDK client (exactly SDK5 since PQ #2152) reads the exact
selected generation/attempt's verified result and sealed request. A
coordinator using the scoped SDK interpreter must also configure the existing
sealed-root resolver to the reviewed SDK qualification tree
(`PRISMABUILD_READER_HELPER_ROOT`, or the
existing explicit helper-root API). Package installation alone is not that
production resolver configuration. Inside an admitted workload the worker's
protected helper root remains its actual deployed generation; this client
qualification does not override or upgrade that worker's reader context. The SDK's capture binder proves the
standard wrapper recipe; it returns wrapper argv, so PQ then inspects the
request's authenticated `params.command`. The supported command runs the
existing campaign-container adapter and the existing quantum entry. The
quantum's own parser validates the payload flags, with duplicate options
refused. The full outer environment, full container spec, source input and snapshot selection must
match the independent contract.

The producer uses the shared `stage_inputs.read_bound` owner once for the
quantum wire. Hashing, JSON parsing and completion publication consume that
same owned byte string. The typed completion contains its original base64
bytes, length and SHA-256, plus the committed counters reference and completed
units. The quantum control cap is 1 MiB, checked before JSON parsing and
completion encoding. The consumer checks the declared cap and encoded length
before base64 decoding, then verifies exact decoded length and the sealed
`--quantum-sha256`. It never stats or reopens the pilot's original record path.
Counter/result documents are capped at 8 MiB; decoded JSON overhead is
additional to these byte caps.

The shared quantum-record validator checks the authenticated record identity
and adjoint binding. Counter identity, quantum ID and complete integer units
must agree with the selected completion and record. PQ derives the actual
pilot binding from the reviewed implementation, sealed execution-plan digest,
spec replay regime, original record geometry and cotangent/handoff mode. It
compares the actual semantic invocation and binding with the proposed row,
then rederives the existing measured wait/rate/power gate. Input pathnames and
output namespaces remain provenance: equivalent geometry and byte-bound inputs
can use another namespace. Original quantum wire SHA is authenticated against
its own sealed invocation rather than required to equal a differently named
proposed output record.

Every row is admitted before the first submission. Missing/legacy/ambiguous
results, wrong selector, altered counters or quantum bytes, unsupported
wrappers, and resealed foreign source, entry, image, environment, plan, regime
or row shape refuse the whole fan-out. Dispatch state records the selected
identity, receipt, original record and independently bound source contract.
PB alone owns placement and retry. PQ adds no CAS reader, Git materializer,
cache, dispatcher or serving runtime.

CPU consumer smoke also runs the production `Gateway` through the production
root resolver, over an explicit private helper view of the installed SDK5
package, against a private real CAS/queue. Only the queue owner's default
root is redirected; result reading and capture binding use their unmodified
public implementations. A matching dry-run emits authenticated receipt stamps,
while foreign source, refs, wrapper and attempt selections refuse without
writing dispatch state or submitting fleet work. Controlled producer telemetry
and source-review fixtures remain CPU doubles, not genuine GPU pilot evidence.

## Finite genuine-pilot protocol for root review

The next measurement is one complete GLM-5.3-Flash layer-007 joint quantum in
the capture-batch-4 regime identified by #1291, on the existing GB10 worker
class. Keep the sealed calibration draw, sequence length, probes, candidate
roster, prepared inputs, adjoint slice and entire retained-window geometry of
the intended fan-out. A reduced toy workload cannot certify that row.

Before a GPU request is published, the review input must freeze the exact
original-source provider/control bindings, source snapshot descriptor and
package relation, immutable container image, plan, prepared record, adjoint
slice, original quantum wire and executable readset. Each input needs its
exact path, SHA-256 and byte count. The proposed command/spec and derived Stage
B resource policy must be filed with them. Current CPU original-material
controls do not supply GPU source/lifetime qualification. Until the owning
lane supplies that qualification and these input seals, this protocol has no
launchable request and requests no GPU approval.

The resource/stall contract is the existing row-derived one: the resource
policy's host/device/physical limits, bounded spill/cotangent reservations,
PB-assigned CPU affinity and native thread limits, and ordered head/load/compute
progress phases with their workload-derived allowances. PB stages the sealed
readset in its existing RAM/SSD tiers. No HDD fallback, custom sharding or
reservation inflation is introduced. Root reviews the exact derived numbers
with the input seals and issues resource GO before this one action runs.

Collect an in-process profiler, the producer's exposed-wait/rate/power data,
and raw Netdata series on both the GB10 host and dl380g10 over the same action
window. Power versus the measured device envelope is the occupancy evidence;
GB10 utilization percentage is not a saturation measure. Clock/power agreement
must be qualified before an energy or work-per-joule claim.

Durable acceptance requires successful terminal/attempt records and verified
CAS result, one unambiguous completion, exact original quantum/counter bindings,
complete resolved units, zero required residency misses, and measured wait
inside its rederived bound. Use its explicit selector and previously reviewed
source contract in a dry-run dispatch. Alter copies of the actual driver
counter bytes and reseal foreign source/invocation controls: each must refuse
before publication. An exact byte copy under another output namespace must
retain admission. Other layer/shape bindings still require their own pilots.

The #1291 historical before receipt remains historical. It lacks the new
completion/source contract and must refuse; it cannot be retro-stamped. The
parent's matched before/after driver-receipt acceptance requires independently
qualified executions with matched calibration and row geometry. If the former
behavior has no supported current-source configuration, root must select and
review a reproducible baseline before another measurement is proposed. This
child supplies CPU causal controls and this bounded protocol, and leaves that
measurement gate in #1293.
