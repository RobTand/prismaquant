# Produced-render destination and prewrite plans

Status: opt-in CPU control-plane foundation. Refs #870; closes only #1835.

## Contract

`ProducedRenderPublication.plan_render_prewrite(qname, fmt, *,
archive_max_bytes)` returns a frozen `ProducedRenderWritePlan`. It accepts no
Tensor and creates no directories, payloads or archives. The positive integer
reservation envelope is explicit and cannot exceed either sealed payload or
temporary durable maximum. It is not a derived or proven tensor archive bound.

The plan retains the complete qualified name and the existing cache writer's
registry-canonical format. The durable origin comes from the bound template's
`output_prefix`, not PB's instance, reader or movement metadata directories.
The layout is:

```
<origin>/renders-v1/<attempt component>/<render batch>/<legacy archive leaf>
```

The logical batch identity stays stable across retries. A versioned digest of
the bound owner, template, nonce and scope separates attempt directories. This
component is a PQ destination identifier, not a PB material namespace. Equal
legacy leaves for `layer.a`/`layer_a` or `layer/a`/`layer__a` therefore no longer
force equal planned destinations. The legacy leaf is unchanged; unsupported,
control-containing or overlong names refuse instead of being truncated.

The temporary plan uses the same leaf plus one `.tmp` extension. This preserves
the pathname convention for a future byte-preserving writer. It does not grant
concurrent serialization, overwrite, no-clobber or existing-byte adoption.

## Admission and absence

`require_render_prewrite(plan)` derives the plan again from the current
publication and compares every field before calling the existing shared
`require_prewrite`. A frozen dataclass is not authentication. PB still checks
the bound contract, current admitted owner and immutable prewrite accounting.
Payload and temporary ceilings are both reserved; both exact paths are listed.
Identical outstanding claims reuse the same immutable PB reservation; an
optional diagnostic duplicate flag is not the accounting proof. A changed envelope
for the same outstanding batch conflicts rather than accumulating credit.

Planning and revalidation refuse known occupied paths, symlinks, failed origin
containment and unreadable path metadata. These are snapshot checks, not a
filesystem concurrency mechanism. The eventual writer must establish exclusive
staging and enforce its actual byte envelope before any write.

Abort uses the inherited public API. PB proves all planned paths absent before
releasing outstanding accounting; present or unreadable paths, committed
batches and paid funding intents retain it. The planner never unlinks output or
fabricates a successful abort.

## Remaining integration

This slice adds no writer, PWC reader, mover, spool, restart adoption or
retirement integration. #870 remains open. Natural strict-LRU PWC loads already
record nested paths and compaction restores those paths; no compaction defect
is claimed by this plan.

For the later writer, retain pathname serialization and use a bounded post-write
SHA256 pass through the existing digest owner. Inline hashing is an optimization,
not an original #870 requirement. Declare that read IO and its resources. A
storage-aware archive ceiling still needs a proof for the supported Tensor,
Torch-version and configuration grammar; logical `numel` alone does not bound
CPU views' backing storage. No serializer or payload-digest implementation is
added here.

CPU fixtures use a real private PB queue, sealed request and claimed owner with
broker-shaped fixture control, not a running cgroup broker. They check pure
planning, immutable plans, aliases, physical collision separation, sealed
ceilings, forged/occupied/symlink/unstatable paths, real duplicate/conflict
accounting and absent/present-path abort. They do not qualify production
throughput, GPU residency, serving numerics or a shipped artifact. Production
integration still owes before/after in-process profiles and Netdata series,
existing resident-prefetch behavior and the applicable shipping gates.
