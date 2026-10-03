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
