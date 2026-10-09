# Separately charged container compilation caches

CPU accounting slice: #1820. Lifetime and GPU acceptance: #1091 and PB #1360.

## Opt-in declaration

A new sealed container spec may declare both:

- `PRISMAQUANT_CONTAINER_CACHE_ROOT`: a canonical absolute host path on an
  explicit writable identity mount;
- `PRISMAQUANT_CONTAINER_CACHE_MAX_BYTES`: an ASCII positive integer byte
  ceiling chosen by the operator, not an inferred production default.

The existing `LOCAL_SCRATCH_KINDS` registry validates and forwards this pair
through `PRISMABUILD_LOCAL_SCRATCH_PAIRS`. PB rounds each declared ceiling to
GiB separately. For example, cotangent=1 byte, spill=1 GiB, and cache=1 GiB+1
byte declare 1+1+2=4 GiB, not the rounded aggregate. The control list belongs to
the outer admission request and is not forwarded into Docker.

The cache root must be disjoint from every other registered scratch root.
Equal, ancestor, and descendant roots refuse. A remapped, readonly, unmounted,
or nested mount refuses. Existing symlink ancestors and non-directory
ancestors refuse; validation neither creates nor deletes any path. Checks
are configuration/launch-time inspection, not a race-free filesystem sandbox.

## Routing

The explicitly charged cache root contains four defaults:

| Variable | Default |
|---|---|
| `HF_HOME` | `<cache-root>/hf` |
| `TRITON_CACHE_DIR` | `<cache-root>/triton` |
| `TORCHINDUCTOR_CACHE_DIR` | `<cache-root>/inductor` |
| `XDG_CACHE_HOME` | `<cache-root>/xdg` |

An explicit pin within that root is preserved. An escaped or overlay pin
refuses, even when `overlay_cache_reason` is supplied: that waiver does not
make an opt-in cache declaration account for bytes outside its root.

`PRISMAQUANT_TMPDIR` names a direct child of another charged root.
It selects that root and the child name for PB's versioned lifetime declaration.
PB supplies the actual host, action, generation, and attempt namespace.
The launcher uses only the registered path that the public SDK binds to its live claim.
It refuses absent, incomplete, cleaned, or foreign registration.

Legacy specs without the new pair retain their previous defaults and overlay
admission policy. No existing sealed spec is rewritten or silently selected
into this policy. The launcher, dry-run command builder and preamble use the
same registry-driven forwarding path.

## Limits and acceptance

The ROOT/MAX pair declares a reservation, not a filesystem quota.
SDK6 supplies the public lifetime builder, registration fields, and namespace binder.
The shared source pin and the reader pin select PB #1675 at `03a5451ac61bedcd455805f0fa2ab1f032163300`.
The dispatcher seals a versioned object with an ephemeral workspace and a persistent cache entry.
Legacy rows select no lifetime object.
The launcher refuses execution without selected support and complete attempt registration.

PB still owns directory creation, deletion, recovery, and the capacity barrier.
PrismaQuant adds no sweeper, private cleanup hook, or `finally` cleanup.
Source-bound fixtures use simulated broker and process endpoints.
They cannot establish live worker or launcher loss recovery.

CPU tests cover declaration/pricing, command forwarding, containment,
conflicting/partial/ASCII bounds, mount safety, symlink refusal and unchanged
legacy cases. PQ #2463 measured the workload-specific ceiling: two GPU rows
of a complete Stage B quantum peaked at 3596288 bytes each, and the
fixture `tests/fixtures/container_cache_ceiling_2463.py` derives the 1 GiB
ceiling from that peak with declared headroom. Receipts live in
`docs/measurements/container_cache_quantum_row1_2463.json`,
`container_cache_quantum_row2_2463.json` and
`container_cache_quantum_row1_netdata_2463.json`; the report is
`docs/measurements/container_cache_peak_2463.md`.
The recorded GPU results remain historical.
They predate the sampler lifecycle repair and this SDK6 integration.
They do not qualify cleanup, a new measurement window, or a performance change.
New measurement commands use the registered TMPDIR and retain receipts outside that temporary leaf.

## Sampler lifecycle requirements

The sampler confirms its first complete scan before the row can start.
The child appends that sample before it sends the startup confirmation.
A missing confirmation, failed initial scan, or dead child prevents row execution.

The parent requests the final scan after the row ends.
The child records the final scan and exits with code zero.
The parent checks the exit code, sample sequence, and both row boundaries.
An early exit, signal, failed final scan, or row exception invalidates the evidence.
An active sampler cannot report a complete row.
Both measurement commands return a nonzero status for an invalid receipt.

The control pipe has no shared process lock.
A killed child cannot hold the parent's stop request on an abandoned event lock.
The existing scan-error and sample-gap checks still apply.
The retained GPU receipts predate these lifecycle checks; this repair does not add fields to those immutable receipts.
The CPU repair evidence appears in the measurement report.

The IO engine starts the isolated scan process.
The sampler still owns its control pipe, samples, and child exit checks.
A refused process launch closes the control pipe and removes the sampler's temporary directory.
Both commands reuse one compilation probe and one runtime reader.
The probe uses the existing KDA qualification API.
Receipt publication reuses the existing profile writer and digest owners.
The Netdata collector and observer share one validator.
No frozen allowlist or duplication baseline expands.

This change adds no rendered-weight cache, activation cache, model arithmetic, wire format, or serving gate.
It moves only the PrismaBuild consumer contract and its connected fixture pin.
It changes no serving runtime pin, kernel, production recipe, or agent scheduler.
The opt-in cache-pair path requires a host interpreter with PrismaQuant and the selected SDK.
The small legacy host path stays unchanged.

## SDK6 qualification and external prerequisite

PB `8746986b175d79ace1ff549b18d70943f98bdda2d3fdb3f29bae9d7c6d5b2ba2` published the read-only source archive and installed SDK6.
The archive SHA-256 is `7d39c5234bc155e60196e1a72a904a35c17214b78f0913b46980ee7714f9bb94`.
The qualified x86 interpreter is `/home/rob/venvs/pq-pin-fca4c6ce0-pb03a5451a/bin/python`.
The installation retains Tessera `fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb`.
The published pin guard verifies both Git installs and their RECORD bytes before pytest.

PB `50eb3fb4dc55ecfead8451da3cc05fd2bcfd6d0b7a9cfd9389cb4a76a416d4ea` reproduces seven lifetime integration failures before the fix.
PB `6ee9dc43b4f630a963f12ffafba449163af4b46cc94b12e0fbfe3c69b7071b57` passes the first ten lifetime cases.
PB `ed30db4740f8107e8cf162137daddd2c42f6768a76f480f7919c6b4a2da32260` passes all 30 charge cases without skips.
PB `87842483fce5728841a258b595c8a1e97ed7056392a2e4a14fe989f0862c1a8f` passes all 16 process-isolation cases.
SDK6's relative pytest import requires a private package, not the former private standalone module.
The loader preserves the test bound and leaves canonical PB imports unbound.

PB `1d4071b6a337f6b1c351213acaaea4075c1a68daf8b4828a47f6e3757b878f01` exercises the actual Docker adapter.
It resolves SDK6, builds the versioned selection, and refuses an unregistered launch before Docker.
It also compiles five changed modules.
This smoke proves refusal, not successful cleanup.

PB `1d5e7fda251954612f3baaac685e6bd1a9ab42b6bf028defe6b18aa9bac1aabf` verifies the migrated measurement request.
The request carries the versioned lifetime object and does not override the registered TMPDIR.
The receipt path stays outside the ephemeral leaf.
The same action repeats the unregistered-launch refusal and compiles eight changed modules.

PB `f2a31c437c8a2fd12c8424762f26e078812b1baa17eb0da34b23cbdfae9fbd98` reaches the selected pool's filesystem guard.
Its four cleanup scenarios refuse with `scratch filesystem type not supported: tmpfs`.
The other 11 cases pass.
Two spool suites also refuse the `/tmp` filesystem in PB `ec7c740f59a03226c914f38530dd6225ff661fdcb28aab950789d495f748c9b3` and `fbf184367f20e0bee7d051cb4a234ed36f6cea20c838758ce397cd900fb7c12f`.
No filesystem guard changes or skips hide these failures.

Open PB #1689 owns the required disk-backed x86 scratch mount and measured offer.
The observed x86 offer has no `spool_gb`.
Resume the cleanup scenarios with that qualified mount as `pbtest --basetemp`.
Do not use the HDD pool or move CPU qualification to a Spark.
Real loss recovery and the new GPU measurement window remain unqualified.
