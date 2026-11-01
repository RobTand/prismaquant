# DSv4 constructor and initialization repair

Provenance: 2026-10-14 PrismaQuant architecture refactor, base 138069b7993.
Repair base: e465245a58d085b968b9793b072a322dd96a80e9.

Registering per-expert DSv4 probes previously rebound the modeling module's
`DeepseekV4Experts` name. Sparse blocks used that name as their constructor,
and `_init_weights` also used it as the packed-type predicate. Genuine model
initialization therefore treated the per-expert ModuleList as packed weights
and raised `AttributeError: 'DeepseekV4Experts_PerExpert' object has no attribute
'gate_up_proj'`. The fresh isolated original grouped-Fisher case reproduced
this failure in PB509508c8ced4, independently of mixed-file test order.

Sparse blocks now own an `experts_type` constructor class attribute in both
the modular source and vendored generated copy. Registration selects the
existing per-expert class through that attribute. The packed class keeps its
type identity, so its initializer handles actual packed tensors and HF
initializes the adapter's Linear leaves normally. A second packed-type alias
while retaining the global replacement was rejected: it would preserve a
misleading type binding and make every future type-sensitive consumer know
about the alias. No initializer is disabled or guarded with hasattr.

The model-walk toy helper now explicitly enters `genuine_weight_initialization`,
as its real-CPU-init contract requires. The new test poisons Linear constructor
weights and biases with NaNs, constructs the actual toy, and requires every
Linear parameter to be finite, including routed expert leaves. It then poisons
the actual packed expert parameters and exercises their real initializer. A
skipped initializer cannot pass this test. There are no statistical thresholds
or timing bounds. Existing real Fisher-row and model-walk assertions remain.
No source-parsing test is added and no behavior test is removed.

Read-only source review confirms nine unchanged declarations, including both
initializer bodies, both sparse forward methods, both expert adapter classes
and the registration function's executable body. Constructor choice is separated
from type identity without changing forward math, parameter layout, routes,
runtime initialization permissions or checkpoint missing-state policy. This
repair changes cold construction dispatch; it makes no GPU or speed claim and
does not qualify serving, formats, native kernels or production artifacts.

All tests and compilation ran through PrismaBuild from immutable capsules,
using `/home/rob/venvs/pq-pin-d2e98b3c-pb20261015-git/bin/python`. Each action/shard
reserved one CPU, native threads 1, with 6 GiB RAM for tests or 2 GiB for
compilation. PB-assigned affinity was retained and placement followed the
qualified absolute-runtime dependency. No repository-local execution bypass,
other-repository edit or remote mutation occurred.

- Original isolated RED PB509508c8ced4: the actual grouped-Fisher case failed
  with the packed-type AttributeError. Exit 1, scope cleanup complete.
- Regression RED PBdf8759cadcba: the new poisoned-Linear test failed with the
  same AttributeError before production edits. One failed, 21 intentionally
  deselected. Exit 1, scope cleanup complete. Neither failed action issued a
  successful CAS result. Both canonical stdout/stderr logs were read and their
  byte counts/SHA-256 checked.
- GREEN PBd4cbb6023444: 46 passed. CAS result
  `6ed16bc9780c3151054f5210c7b7b391855469d65835dddfc71e1b28c41bf079`,
  22,477 bytes.
- GREEN PB3c7b7a76eeda: 49 passed. CAS result
  `1b2141d53b05d033a1939a6f49eaf11a2f645602f860bbfaf02ec6efa5d80c34`,
  24,504 bytes.
- Compile PB441d535404de: five touched Python files compiled. CAS result
  `63f039f0085d45c62f64c6c885f75dd5bc4fa89ef74631ba0475e8e7d83cdcd0`,
  11 bytes.

The final disjoint test cohorts contain 95 passes, zero skips across ten files:
model walk, grouped Linear Fisher, genuine initialization, pretrained buffer
initialization, package/runtime ownership, DSv4 profile and rename/layout,
architecture, documentation staleness and duplication guards. This includes
the original failing real Fisher row and independent no-init/context/thread/
checkpoint controls. Every GREEN action has terminal exit 0 and complete scope
cleanup; the actual CAS bytes were independently read and SHA-256/size checked.

External raw logs, capsule manifests, source review and rerunnable receipt
inspection are retained under `/home/rob/tmp/pq-refactor-20261014/reports/`
with `dsv4-*` names, including `dsv4-init-receipts.json` and
`inspect_dsv4_init_receipts.py`.
