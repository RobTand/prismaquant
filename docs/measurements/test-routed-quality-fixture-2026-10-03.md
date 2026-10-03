# GLM routed quality test fixture reuse (Refs #1929)

## Change and preserved semantics

The baseline profile identified repeated construction/validation of the same
864-member GLM quality input as the dominant setup cost. The ten existing
full-quality/rank-cut mutation cases now construct that genuine input once
per pytest module/process through a lazy module-scoped fixture. This follows
the existing lazy module baseline/private-consumer convention in

tests/test_tessera_campaign_resume.py. Each mutation deep-copies the entire
input/preflight/row object graph and uses the existing
_bind_glm_quality_fixture to write a fresh private ProductionWeightCache and
completion, recomputing their exact private-path digests. No file is hardlinked
or shared with a mutating consumer; no production validator, quality gate,
member geometry or error diagnostic is replaced or bypassed.

The positive full GLM world-vector test also performs the original private
single-rank truncation/refusal against the same genuinely emitted/admitted
world row. It no longer repeats _glm_cell/_routed_gate solely to run that
parse refusal, nor builds the unused small joined fixture. Its own
864-member/two-rank/slowest-median assertions and the exact
"exactly one record per rank" refusal remain. The reused row carries the
positive case's unequal FAST/SLOW rank samples rather than the old negative
case's equal FAST/FAST samples; the rank-count refusal is unchanged.

All ten mutation IDs and every distinct forged field/diagnostic remain.
Only test_a_glm_owner_is_refused_a_single_rank_vector_for_a_two_rank_world is
consolidated into test_a_glm_288_owner_prices_two_ranks_end_to_end.
Exact observed inventory: 62 passed before, 61 passed after, zero failures or
skips in either arm, no added nodes and precisely that one removed node.
Both logs retain 14 existing Torch JIT deprecation warnings. This is synthetic
CPU receipt/geometry/orchestration coverage, not a GPU/native-quality result.

## Frozen source and commands

Base: 736fd56aa6853c3ee06655eda6015b18bef763bd.
Test commit: b37e66ccf65fbd2cfb58593f919bbdc9f58205f7.
Only test source changed: tests/test_native_receipt_table_routed.py.
Before SHA-256: 79ac2a9eb22092f95aad41c61bd52b3d010ddb5b2d4d5771690297ff7d230196.
After SHA-256: 058aa319168641e4e5759b1a6e0ef626ad488e5f5c39afe369161e86d193a0b8.

Published pbtest.py, explicit --tag dl380g10, --shards 1,
--workers-per-shard 1 --threads-per-shard 1 --cpus-per-shard 2 --mem-gb 4,
--profile sample --timeout-s 900 --wait-s 1800 --priority -10,
tests/test_native_receipt_table_routed.py. PB owns partition/placement;
no manual test shards or Spark compute were submitted. Both allocations use
preferred cores [0,1], GPU visibility disabled. Both pin preflights verify
Python 3.14.4, Torch 2.11.0+cpu, Transformers 5.16.1,
PB SDK95a59051d48cda82eea7927f31870c6c862d7174 and
Tessera b40c93cb73745097e57a1ba4cf5b9eee166c759a.

Before action fc637ba0b82780420bd5896e35939b4e4a2f2edee5834d5340a72f017316273e.
Receipt 833ce7eaf670fcd8c83432c1b359e99fa7202208b7737dca4c6b91843a64f176.
Snapshot 9ed5af3ccbcd377283c763614b87ae06a7d37885, parent exact base;
bundle e2651c32430c7bf75cea79d0b6f8e517defa6cba86461112c97a4c4b0d95212f.
After action 5df41324506df4afc157421b66abb70ed11b2ebc68416f94b4060e0dce6d2f7f.
Receipt 9bae7aa3cf4d8ffdb346658f99a69c6fc1b56a40bf16fd60abd6d9b97d772fe2.
Snapshot 4b778d1f4092d13d942480809c3f381947096794, parent exact test commit;
bundle b903a3e961c17a21291e6581cebfe640084adf5cdee9d31db9c896db009f25c6.

## Profile and resource delta

| Observed measurement | Before | After |
|---|---:|---:|
| Pytest wall | 343.06s | 241.64s |
| PB contained wall | 350.606s | 249.086s |
| Cgroup CPU | 418.061s | 296.691s |
| Cgroup memory peak | 615985152 bytes | 614076416 bytes |
| py-spy100Hz samples | 35079 | 24504 |
| _glm_cell inclusive sampled time | 171.56s | 77.36s |
| Ten quality mutation calls inclusive sampled time | 154.12s | 65.72s |
| Shared template preparation sampled time | absent | 10.90s |
| Private fixture setup sampled time | absent | 12.15s |

Observed module pytest wall falls 101.42s (29.56%). The mutation tests' setup
moves into fixture frames; their call-frame reduction is not the total saving.
The native gate still runs separately for every mutation. Inclusive samples
overlap and must not be added. Both instruments are py-spy0.4.2 at100Hz:

- Before CAS53498267e45f41d026bfd161c077ef4c136e3987e3f42e7d0498cab688cceaa3.
- After CAS251aa6cfed291bd10154a44141097c55123091988348a9636dc0607bfe73e09f.

The before profiler itself returns1 but leaves the recorded usable profile
and independent successful action ending; PB explicitly records that backend
status as ignored. The after profiler returns0. Action success is grounded in
pytest outcomes and action exit0, not profiler exit status or a submit ack.

## DL and both-Spark telemetry; limitations

Two-second average CPU buckets, busy excluding idle/iowait:

| Host | Before busy | After busy |
|---|---:|---:|
| dl380g10 | 16.612% | 20.082% |
| sparky | 3.351% of observed samples | 8.579% |
| sparklina | 1.381% | 1.441% |

Before window1790997352..1790997703 (176 buckets); after
1790998020..1790998270 (125 buckets). The before Spark reboot creates16
flagged/empty Sparky buckets (160 missing dimension values); that host average
is not full-window coverage. DL and sparklina have zero flagged/empty buckets;
all three after series have zero flagged/empty buckets. PB retains its own
DL averages16.702% /20.351%, PSI-some avg10 maxima1.57 /2.23 and explicit
missing-pqteld-CSV diagnostics. Other DL load is higher in the after arm.

These are sequential, single, profiled file observations on a non-isolated
box, with an authorized reboot of the submitting Spark between arms. Timing
is therefore confounded, not a guaranteed isolated speedup. The source-level
work reduction and sampled setup reduction are independently observable.
There is no suite-wall, line-coverage-equality, GPU, energy or throughput-goal
qualification. #1929 stays OPEN. The integration owner alone qualifies one
compatible composed union; focused file savings cannot close that acceptance.

The before coordinator's SSH wrapper exits255 during the reboot, but the
healthy admitted DL row is not cancelled. Its terminal rc0, CAS receipt,
actual log, profile, inventory and telemetry are recovered after mount/SSH/PB
health restoration. No completed baseline is rerun. Raw before/after evidence
and all six raw Netdata series are preserved in
/home/rob/tmp/pq1929-routed-evidence.json. No background test action remains.


## Successor: remaining positive, rank and CLI consumers (Refs #1929)

A separate successor branch starts at exact frozen PR2130 head
818f8617a7d8ac515a127ef0888c6ae4ac7b9bb1; it does not edit that parent packet.
The remaining consumers now request the existing glm_quality_cell fixture:
unbound-quality refusal, whole-container quality identity, both rank windows,
positive admitted GLM world and the two allocator CLI controls. The CLI helper
requires that private cell explicitly. Its budgets, genuine serialized extents,
FAST/SLOW rank samples, report/capture identities, precedence and all assertions
are unchanged. Rank1 mutations and CLI wire/report edits can touch only their
function-scoped deep copies/private completion and PWC files. The existing ten
mutation consumers and their fixture implementation are unchanged.

Test commit:74c45b71c55585d7f6c654e2902afcebd2e26cf5.
Test file SHA-256:12ff75b5d8b7800459997ea5090c45d758f527e26a17452f027ab755da604db9.
Before test SHA-256 is the PR2130 after hash stated above. No pytest node is
added, removed, renamed or skipped in this successor. Both targeted arms run
exactly the seven changed nodes and intentionally deselect54 other nodes,
including the previously accepted ten mutation cases. This is not a new61-node
qualification. The selected population, in source order:

- test_a_rank_local_panel_refuses_an_unbound_quality_preparation
- test_the_frozen_joint_names_the_container_render_not_the_rank_cut
- test_the_rank_render_proof_moves_its_window_with_the_rank[0] and [1]
- test_a_glm_288_owner_prices_two_ranks_end_to_end
- test_the_glm_cost_model_reaches_the_cli_and_expands_to_its_864_members
- test_the_cli_refuses_a_forged_or_changed_rank_report

Published pbtest uses the earlier CPU2/mem4/native1/one-worker DL contract,
600-second deadline and the same interpreter/packages as the original section.
A -k OR-expression of the six test names seals precisely these seven nodes;
PB alone partitions/places the file. Both action and profiler returns are0;
both reconciliations record7passed,0skipped,54deselected, zero missing/duplicate
collection or outcome problems and14 existing upstream warnings.

| Successor observed measurement | Before | After |
|---|---:|---:|
| Pytest wall |135.17s|82.57s|
| PB contained wall |142.804s|89.995s|
| Cgroup CPU |163.847s|101.572s|
| Cgroup memory peak |570261504 bytes|575143936 bytes|
| Observed write bytes |41554392|39150040|
| Observed read bytes |32768|32768|
| _glm_cell inclusive sampled time |61.66s|10.09s|
| _glm_cli_fixture inclusive sampled time |43.71s|26.12s|
| Private fixture setup inclusive sampled time |absent|10.74s|
| py-spy100Hz samples |13641|8467|

The observed seven-case wall decreases52.60s (38.91%). This is one sequential
pair, not a guaranteed speedup or a suite-wall result. Preferred assigned cores
change from[1,2] to[2,3]; background DL load also changes. The source now calls
the expensive builder once per module/process instead of separately in all
seven selected consumers. Each still runs its real validation/emitter/admission
or allocator control. Inclusive sampled times overlap and must not be added.

Before action:eb9528c7fb05ee5e27cb2ce237b366a70adabde3b9b2fffd39b380748ae6253f.
Receipt:23f417e10c14bad4bd2f3e735dc9a0c4015174ec27c9342caae9c187a9d4060e.
Snapshot:d2c7b02cb92f8d0818e8cc5dbc2728bad1b58443; parent exact818f8617.
Input:755274cecba3f402c8e3d5bc73eab911671918eb6c5659d78aaa400af036c128.
Profile:ed6af6de9016a4772685c00e74a77fc8fe059ddcc38f03551f697b5869c708f9.
After action:8dd7642112944686717b3c00b297531a9286ac60cbf9b4e7f9aca4f133d49b6b.
Receipt:8ec423f4a97457e96249ac41e2aab266d4702d787572c33cf8b90934d1b20ad5.
Snapshot:bcd43880315d80d775c34b91c13575896c6c0396; parent exact74c45b71.
Input:79557fb3a613ebc26c856be80c7e98f31c6efe7c9d30567f6859d33ef84dfcf4.
Profile:9b82cd8c9e639f2e3e5b0c565714c33a8e01f819ae668d5c353fee0115213d9a.
A transient before-receipt lookup said absent; the actual emitted receipt and
subsequent complete read-only receipt lookup bind the checksum above. The
original absent observation remains in the raw evidence rather than being
silently relabeled. No work was rerun to resolve that lookup.

Two-second average host CPU busy excludes idle/iowait:

| Successor host | Before busy | After busy |
|---|---:|---:|
| dl380g10 |11.803%|9.851%|
| sparky |1.946%|2.436%|
| sparklina |1.493%|1.459%|

Windows are1791054574..1791054718 and1791054832..1791054923. All six series
have zero flagged/empty buckets. PB's own DL averages are11.764%/9.822% and
retain explicit missing-pqteld diagnostics. Raw receipts, inventories,
resource records and Netdata are/home/rob/tmp/pq1929-routed-positive-evidence.json.
No GPU/numeric acceptance, full-suite or line-coverage equality is claimed.
The frozen full1032-file94ce cohort remains untouched and integration-owned.

