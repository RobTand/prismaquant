# Staged-read ownership, 2026-10-15

Provenance: 2026-10-14 PrismaQuant architecture refactor, base 138069b7993.
Grounded from integration 23972e18237230b9a3889322bdbfb46bf46982bf.

## Whole-entry transaction

The checkpoint shared-state/manifest reader now delegates its whole-entry
transaction to `staged_whole_file.read_staged_buffer(resolver, path, staged,
label=..., expected_bytes=...)`. It returns the same owned `bytearray`. Existing
`read_staged_entry` callers retain bytes or sealed memfd material. One private
transaction owns acquire, actual open, serving registration, independent size
comparison, bounded fill, extra-byte probe, descriptor identity fence, close,
exact-ref release and successful tier byte accounting. The checkpoint retains
manifest field admission, inactive-policy behavior, resolver refusals and the
independent digest check on the bytes its decoder consumes.

A generic opened-descriptor manager was rejected. It would leave the difficult
size/hash/fence/resource sequence with each caller. A universal bytes+receipt
reader was also rejected: range tensors retain descriptors across many reads;
activation archive readers retain the lease through metadata/CPU decode and
clear borrowed views before release; group leases outlive concurrent member
reads; the I/O engine separately fences the declared file and may borrow a group
pin. Their existing `LeaseWindow` ownership remains. Mount facts are a separate
already-integrated owner and are not the whole S1 change.

The source census found six acquire_entry_window calls in six modules and nine
serving registration sites in six modules. This unit removes the checkpoint's
actual copied transaction rather than forcing every lifetime into one API. No
cache, prefetch, numerical objective, format, stage, serving gate or default changes.

### Behavior and execution evidence

All execution used PrismaBuild immutable checkout capsules, native threads 1 and
one admitted CPU per action/shard, with explicit 4/6 GiB RAM demands. The qualified
CPU interpreter was `/home/rob/venvs/pq-pin-d2e98b3c-pb20261015-git/bin/python`.
The runtime dependency restricts placement to workers actually carrying that
absolute interpreter. PB assigned affinity remained intact.

- Baseline PB686a4ff1d5d7 + PB3c5bfbad0c89:167 passed, 0 skipped. The new 16-case
  real-SDK matrix covers both labels, bytearray identity, independent size
  divergence, descriptor size mutation, truncation, growth, changed metadata,
  read error and open refusal. It observes real descriptor closure and real
  pin removal before accounting; every failed read counts zero successful bytes.
- After PB91f4508b45f7 + PB7f9456b07d01:209 passed and 2 inherited guard failures,
  0 skipped. The failure names are solely the existing Tessera table/routed-tool
  ownership sites, not S1. Original-base PB2d775f1b859f reproduced exactly the
  same 2 guard failures and 5 passes. Root's existing e5158e3458/14ebc1ca31/
  e02ba9da27 fixes were applied without waivers. PB6ce4336d14db then passed 9,
  including the complete replacement duplication-file cohort.
- PBd402bb51af5a: 26 sealed original-material cases passed, 0 skipped, including
  kernel write/grow/shrink refusals, authority/digest/size mutation, failed
  material cleanup and retained sealed reads after pool mutation.
- Latest disjoint file outcomes total 239 passed, 0 skipped in 13 files. Earlier
  failed cohorts remain recorded. PBb9dbbe7af214 compiled the two touched
  modules and the new test using compile(...), without local compilation.

The paired CPU characterization executes the real checkpoint read 32 times over
identical 1 MiB bytes, SHA-256 fbbab289f7f94b25736c58be46a994c441fd02552cc6022352e3d86d2fab7c83.
Each arm records 32 successful opens, 32 SDK acquire/release/cover lookups and
zero remaining pins. PBca7bc3bbac65 captured the before whole+group fixtures;
PBd3f810121cb0 captured the after whole fixture. PB52eb0f58388a compared actual
outputs, types, event order, SDK counts, tier counters and cleanup exactly.
Both arms contain torch CPU traces, cProfile data/text and six actual Netdata
samples, three from each Spark. The initial DNS-only captures PB04ea7f839f97 and
PB0008bf82fa24 failed with no samples; these are failures, not zero load.
The rerun maps known SSH aliases to sparky 192.168.1.180 and sparklina 192.168.1.110,
recording both alias and actual endpoint in each sample.

Raw before/after directories and the manifest/comparison are under
`/mnt/shared/pq-refactor-20261014-native/s1-whole-{before,after}-r2` and
`s1-whole-pair-comparison.json`. The latter carries SHA256+size for each trace,
profile and series. Observed cProfile totals were 0.665441 s and 0.512043 s;
PB CPU masks differed, [8] and [12]. These short CPU read fixtures do not establish
a speed change, GPU work, cache throughput or a production bottleneck. Spark
Netdata is simultaneous context; the CPU work ran on dl380g10. No GPU/energy or
performance ranking is claimed.

Terminal records, scope settlement and actual CAS bytes were checked separately;
the rerunnable receipt inspector and source-census/migration tools are retained
in `/home/rob/tmp/pq-refactor-20261014/reports/`. Failed action logs are retained.
The exact read-bound, caller digest and sealed delivery obligations are independent.

## Independent PRs and findings

PR2611 growth-budget reservation and PR2591 opt-in page-locked fill are absent
from the grounded source and remain independent behavior changes with their
historical measurement records. They were not silently incorporated by S1.
PR2641 remains pending behavioral replacements: current seed determinism tests
lack an independent prior mapping vector, and census tests do not inject mutation
while reading or count the dense reader's bytes. Its raw hashlib/census AST locks
are not adopted. Existing actual activation digest, mutation, read amplification,
cleanup and calibration golden tests remain.

Off-task prose fix: `_acquire_bulk_window` incorrectly documented an entered
three-tuple and unresolved RAM covers. Separate commit 2c817fb691 describes the
actual unentered four-tuple and current RAM-first cover selection. Its executable
body is unchanged; source inspection is the verification.

## Group acquisition ownership

`staged_lease.enter_group_lease(entry_resolver, members, live_windows,
counters=None)` now owns the complete existing RAM/SSD group acquisition policy.
Its process-wide `EXACT_LEASE_COUNTERS` and `exact_lease_counters()` snapshot move
with it. The activation single-entry increment uses that same owner record. Both
production group callers migrate directly; one manual benchmark and three test
files migrate the snapshot import. The old cache helper, counter and snapshot
exports are deleted. The I/O engine has no private activation-cache dependency.

RAM-first grouping, availability-only fallback to SSD, partial-cover per-entry
refusal, entered-window cleanup and all counters keep their existing behavior.
Group windows still belong to the callers' live-window cleanup until every read
settles. Each member still opens through the SDK, independently verifies its
bytes and closes its descriptor; borrowing a group never transfers its release
obligation to a member. No parallel cache or lease manager is added.

The read-only source comparison against c72f96e22e verifies the group transaction
AST exactly after the public rename and removal of its same-owner import, and
verifies the counter definition and snapshot exactly. The caller bodies differ
only in import/name/documentation. SerializedEntryDigest, _read_exact_entry,
load_verified_activation_cache_entry, calibration_data_hash and _seed_from remain
exact executable declarations. These are review invariants, not substitute tests.
The rerunnable source comparison is `reports/s1_group_source_invariants.py` in the
external refactor directory.

A new real-SDK behavior check first reads engine members through one borrowed
group and observes that the pin remains live after each descriptor closes. It
releases that group, then reads the same tensors through the activation path and
observes two independent increments of the same owner snapshot. Real SDK counts
must be exactly two acquisitions, two cover lookups and two releases. Existing
PWC/activation tests continue to prove actual byte comparison, RAM/SSD grouping,
partial-cover fallback, single-entry rereads, bad-copy refusal and parallel
settlement. No source-parsing test replaces these checks, and no test was removed
in S1: the touched existing tests assert actual SDK/byte/lifetime behavior.

### Group verification and characterization

PBd59c3ee0fc54, PBd29b55771162 and PBa37ec130b191 passed 261 tests across
10 files. PB8774d2d87e40 passed the real produced-boundary group test; its
37 other cases were intentionally deselected, not skipped. This unit totals
262 passes, 0 skips. PB56fb8f5c4a75 compiled all eight touched Python files and
ran the manual benchmark's actual counter reader in a fresh interpreter,
obtaining the expected empty process record. Tests preserve the actual SDK
and cache behavior; no test is replaced by an import-identity assertion.

The initial group pbtest invocation refused a node ID as a discovery path,
before any action or test ran. The corrected invocation uses file inputs and
an explicit JSON -k selector for the bounded produced-boundary case. The failed
discovery log remains `reports/s1-group-after.log` in the external refactor
report directory. The admitted corrected outcomes are recorded separately.

PBca7bc3bbac65 captured the before group fixture, PB48166b76a9fe captured
after, and PB749530b36c54 compared their actual artifacts. Each arm runs eight
iterations through both the exact activation and I/O-engine group paths over
the same four deterministic tensors. All tensor digests, group results,
serving/release event sequences, tier counters and SDK counts match. Each
makes 16 acquisitions, 16 cover lookups and 16 releases, with no retained pin.
Each arm contains eight torch CPU record spans, cProfile binary/text and
10 actual Netdata samples, five per Spark, with no collection errors.

Artifacts and SHA-256/size manifest are under
`/mnt/shared/pq-refactor-20261014-native/s1-group-before-r2`,
`s1-group-after` and `s1-group-pair-comparison.json`. Observed cProfile totals
were 0.294493 s and 0.286523 s, with PB CPU masks [8] and [0]. The fixture
characterizes CPU read/lease work on dl380g10, with simultaneous Netdata on
both Sparks. The small sample, different masks and warm local fixture do
not support a speed, GPU, residency throughput or energy-ranking claim.

After the two S1 units, the live census has five acquire_entry_window sites
and eight serving registrations in five modules. Distinct range, archive,
wire-fallback and borrowed-group lifetimes remain intentional. A single
universal eager bytes reader is not claimed. The accepted common whole-entry
transaction and group acquisition policies each have one actual owner.
