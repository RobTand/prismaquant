# Shared AURA checkpoint journal verification

Provenance: 2026-10-14 PrismaQuant architecture refactor, base `138069b7993`.
This second J2 unit follows the shared probe constructor. It moves durable
checkpoint knowledge from `aura_cost` into `aura_checkpoint_journal`, and
Git and producer-package identity recipes into `aura_provenance`.

The journal owns the same three v1 schema spellings, identity construction,
manifest IO, archive-on-dev-drift rule, bounded unit pickle envelope,
state capture and restore, resume admission and publication. Its public
operations are `build_checkpoint_identity`, `prepare_journal`,
`capture_unit_state`, `restore_unit_state`, `encode_unit`, `decode_unit`,
`write_unit` and `load_unit`. Numerical passes supply their explicit inputs.
The existing `cost_stage_checkpoint` remains the atomic writer and unit-path
recipe. Both synchronous and asynchronous publication use the journal's
atomic-writer seam. No second store or compatibility API is retained in
`aura_cost`.

The moved owner has 774 lines; provenance has 78. The caller inventory covers
16 application files with 20 import sites, five tools with seven sites,
four experiment helpers with four sites, and 44 test files with 49 sites.
The remaining experiment and tool changes correct dynamic digest-pinning
hooks and document or profile the new owner. The only Tessera Joint AURA
source change is its producer-source import. No other repository changes.

The numerical reverse passes, incoming cotangent routes, residency guards,
replay arithmetic, async staging limits, publication finality and objective
contracts remain in their current owners. Full-run checkpoint rows still
require their exact probe identity; quantum rows retain their independent
D32 wall and run-seal comparisons. MTP retains its own objective. This unit
does not complete J2's proposed shared numerical price step or introduce a
GPU session. It changes no format, serving gate, default or stage graph.
No performance result is claimed.

## Behavior and removed tests

Existing fixtures execute real checkpoint writes, restored measurement
vectors, resume refusal, UTF8 bytes and refusal goldens, partial async
publication, interrupted writes and bitwise single-run versus quantum
measurement envelopes. Journal and provenance callers migrated together;
the old `aura_cost` API was deleted.

Two alias-identity assertions were removed:
`test_atomic_write_recipe_1573.test_aura_cost_uses_the_owner_recipe` and
`test_digest_profiles_1301.test_aura_round_trip_binding_is_the_owner`.
They asserted that imported aliases were the same function object. Actual
atomic-write failure/cleanup checks and fixed byte/refusal goldens remain.
`test_digest_profiles_1301_files.test_unit_path_site` now checks the single
existing path owner instead of checking the same recipe twice through an
alias. The historical golden table keeps its old diagnostic tags for the
same byte cases. The source-inventory `test_no_new_seals` remains: it is a
structural safety policy, and its owner and decode-scope inventory moved
with the code. No safety contract was removed or weakened.

## PrismaBuild evidence

The baseline ran 15 files in three portable CPU shards: 298 passed, three
skipped. The broad refactor run ran all 52 touched or architecture test files:
1,241 collected, 1,186 passed, 49 skipped and six failed. Three failures were
a missing module import in a migrated test; three were an affinity-dependent
worker-count test cohort deliberately asking for two or three CPUs under a
one-CPU reservation. The import was corrected. The cohort was rerun with
three admitted CPUs, retaining the guard against counts above affinity.
The narrowed reruns complete the same population without repeating the
successful unrelated checks.

The first after-run also found missed async publication and experiment
callers. They now use the new journal and the existing path owner. These
failed runs and their exact immutable source snapshots remain in the evidence.
No failed run is represented as green. Initial baseline discovery named the
absent `test_joint_checkpoint_publication.py`; PB refused before submission.
The file selection was corrected to the actual publication suite.

Final reruns passed all 31 state/publication tests and all 12 affinity tests,
with no additional skips. The complete distinct population therefore ends
at 1,192 passed and 49 explicitly skipped. Both rerun terminal records are
`executed`, exit zero, with actual pytest completion and CAS receipts:
`f8ce458514a110a248b0be65c0ea4aa36fd2c3f6149350b7004540d60a7dd543`
and `a5efbd99fe5977957175bddab0e93a3d4ef1c4bca9f3ad6f5d10f6791f84002e`.
Four successful broad shards also have verified terminal endings and CAS
receipts; the failed broad shard has exit one and no success receipt.
All 80 touched Python files compiled through PB with exit zero and a CAS
receipt (`a1f700a070d5e72356d6c39835a25ba6e89b0d71080d6c22057f0551d1b0cfe9`).
The corrected import and removal of an unused path-owner import compiled
again with exit zero and a CAS receipt
(`0d940b748f31fd98189636754e8ee915bcf0d8ce95ce134a072586cfe08f833a`).
Tests used `/home/rob/venvs/pq-pin-fca4c6ce0-pb027103d9/bin/python`, the qualified
Python 3.14.4 environment, one pytest worker and one native thread per shard,
6 GiB per shard, priority -10, portable placement, an action ceiling of
1,200 seconds (600 for targeted reruns), and 180 seconds per test. The broad
run reserved five CPUs and 30 GiB aggregate; the affinity cohort reserves
three CPUs because it queries that admitted capacity. Compile reserves one
CPU and 2 GiB with `--anywhere`. PB affinity is preserved.

All 49 broad skips are explicit: 39 require administrator-provisioned private
spill scratch (`PQ_STAGE_B_SPILL_TEST_ROOT`), six require CUDA (including the
pinned-host allocator), two require the historical sealed campaign package,
one imports a kernel defined with Triton, and one filesystem cannot perform
direct IO on an 8 KiB grid. Those qualification gaps remain unmeasured.
No GPU pass or full campaign reconstruction is claimed. Exact node IDs and
reasons are in `joint-j2-journal-skips.txt`.

Baseline actions: `bfccbddb68cb4a06935df9fcce6cdfbe8155dd32841c20efb8d59a645f27b4b8`,
`c866be7c92b8404096db6e3dce852d05030a914a9aaec729cd36ab5b062606c4`,
`3c8c77891c3e08995544e47915c89629dd1da72df741cc0c0c245f9f3f78cb1b`.
Broad refactor actions: `2c64090aece9cce6be1518d4d2713a4e1732f7e269677ac3ab2d3be2a4fc6f6f`,
`165649b2062d9bf9219fa210c448fa50bc4af5582b02c0403c364fff23caf008`,
`24d0a4f9df396632005599181b3d5edf645c472a721f1649179f6eb9df76dc7e`,
`c68405d3ec538a96e704fd785206512003603397266323ba9b42d0f84682d469`,
`ab9ac77d88453226ddeba5fb8dba1f6ccee2f0b9578c8764d3126e36af220f11`.

Commands, logs, per-test completion, caller inventory, skipped node IDs,
terminal records and CAS receipt inspection remain under
`/home/rob/tmp/pq-refactor-20261014/reports/` in `joint-j2-journal-*` and
`joint-journal-inventory.json`. `migrate_aura_journal.py` records the migration
recipe, including experiment callers; subsequent test cleanup is explicit
above. Source and Git inspection ran locally. All execution checks ran
through admitted PrismaBuild actions.
