# Stage B packed firing metadata: CPU prerequisite for PQ #1086

The opt-in constructor `StageBReplaySpill(packed_records=True)` retains six
uint64 fields per firing record, with names bound to the window roster and
complete layouts interned. The legacy tuple interface still serves every
record consumer. The existing conservative spill part geometry bounds each
window's record count, including empty operands. Default list storage remains.

This separates window/replay ownership from storage before changing storage.
It adds no loader, cache, pool, persistence, arithmetic or serving contract.
Missing/false preserves the prior path. This is a CPU research prerequisite;
the original issue remains open for representative proxy and production
acceptance.

The synthetic CPU corpus has 867 targets and 512 invocations per target:
443,904 retained probe-0 gradient records. Four probes plus probe-0 inputs
give a conservative 2,219,520-part geometry. The corpus reconstructs fresh
shape/stride tuples on every invocation and varies rows and address residues.
It does not load GLM weights, reproduce routed firing membership, or allocate
the other read-plan structures. Each arm runs in a fresh child process.

| Instrument | List | Packed |
| --- | ---: | ---: |
| tracemalloc retained table bytes | 167,046,105 | 21,676,081 |
| tracemalloc peak table bytes | 167,046,186 | 21,676,162 |
| Whole child process maximum RSS bytes | 1,007,267,840 | 573,399,040 |
| Interned layouts | N/A | 128 |

Every ordered record field, including shape, stride and address residue,
hashes to `05b2a84b4f1412b5c44ead2f0c2ba323a1d470c5652bc091ab75a5a4fbad1aa1`
in both arms. This is the metadata order oracle, not the historical proxy's
arithmetic digest. Whole-process RSS includes imports, profiler bookkeeping
and hashing. The reported reduction applies to this table corpus; it does
not establish full-layer peak memory, CPU/GPU speed, work per joule, reader
wait or a production default.

PrismaBuild action
`e90f2f765bc655a91d35d825dba0a4daefb627e8e9d30d727a35e4d3748363be`
completed on dl380g10, CPU2/4 GiB, CUDA disabled, exit 0, unambiguous terminal,
CAS receipt `f7b989b756ef615c0386bbcac2de0992d7717d704c49d6bfabeefa91f9a90709`.
No timing isolation was requested and no throughput comparison is claimed.
The source-bound report and artifacts are retained at
`/mnt/shared/tessera-measurements/pq-spill-1086-20261002-memory01/`:

- `list.pstats`: `177e2884a7d1923cdcc9220ca76b8ad2be2f20cc034b972f03fff0458791b41e`
- `packed.pstats`: `f66612417e64169253755888844ad59f44f843cea0cb4a8e38dc6b9a5565e926`
- `netdata.jsonl`: `c84e432401a4989c27e1460fbb055a96feb92c9fadac8e898670e2642b26e583`

Both-Spark Netdata contains 30 complete samples and no sampler errors. The
CPU worker's enclosing action resource profile is retained separately. GPU
energy/power agreement remains outside this CPU claim.

Reproduce through PB using the current pinned project interpreter:

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd CHECKOUT --cpus 2 --demand mem_gb=4 --tag x86 --priority -10 \
  --timeout-s 540 --wait-s 900 \
  --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 -- \
  /home/rob/venvs/pq-pb95a59051-tessera-b40c93cb/bin/python \
  -m experiments.stageb_spill_record_memory --out FRESH_SHARED_OUTPUT
```

Validation used Python 3.14.4, torch 2.11 CPU, Transformers 5.16.1, with
PB95a59051 and Tessera b40c93cb install provenance guarded by pbtest.
Baseline `8811ad5f` plus the new tests failed four packed-replay cases at the
unsupported constructor policy, PB action
`ea1326de74fe12231890525740abc7ba8d0a622e3a1eaa112578b47130668326`.
The earlier infrastructure `pb-cpu` attempt failed collection for missing
compressed_tensors and supplies no causal RED evidence.

Seven candidate shards completed with 90 passed, 23 skipped, zero failures,
and 113 collected/ran outcomes. All 21 new packed-policy, live-operand,
probe-order/layout drift and corruption cases passed without skips, as did
all 15 existing integrity cases. Legacy full-row tests account for the 23
skips: CPU-worker statx direct-I/O grid qualification or CUDA-only prerequisites.
The live CPU packed tests model a 4 KiB reported grid while retaining real
local O_DIRECT reads/writes; they do not qualify the worker's missing statx.

Exact actions and summaries are in
`/home/rob/tmp/astra-resume-20261002/pq_spill/green-expanded.json`:

- Packed records: `5be0d0e7b4345dcf9e8deeb1fd97a280efb52216ba42d50ca2d6006f4543e10d`
- Integrity: `ce1a393723af27dbc18e3ca0f406b91dc0a5efc5520e2166c246dd0c6729d1e0`
- Legacy spill: `8ac9c3826712ad8b361953af73277b347d471bf9c0db9978e668322970221f29`
- IO-site freeze: `64ce6d5ed9e4cda65028ab8cb1bd3cff47a5aa14f5cb2e793c88915f1c9d3f23`
- Architecture: `fb3a60177ae660d3bbcf5c7c4338032891f9989838b39371c513d8f41c390553`
- Docs staleness: `866328a94a93ca4767ea4c6f8cc4cae77cf9f8558602596d51e9b41b74f47986`
- Duplication baseline: `f6be96655b0dd67f8a9807ae54ee1dc0781ddc423bdab68cee8ad1977496fe1f`

Three touched Python modules compile through PB action
`f0cd3ab70311af22960e7648f0ab622225e0f18a01414ae7bc78ff6610b23102`,
exit 0, CAS receipt `a80b3e6164ee3cc1517e8677fdb63c8db3af57bbea479863cdedbe1182ec267b`.
Every action's terminal and CAS metadata was read independently with the
published pbmcp `pb_action`, complete and without ambiguity. Published input,
compiled module and produced artifact identities remain in the lane ledger.
