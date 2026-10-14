# Quantum record ownership verification

The 2026-10-14 structure refactor starts at `138069b7993`. The new
`prismaquant.joint_campaign_ledger` owns quantum record layout, canonical
identity, campaign comparison, shared record-set rules, and record resealing.
It preserves v1 bytes and the existing dev-mode distinction between run
provenance and data integrity. No format, default, stage graph, serving gate,
GPU kernel, or numerical accumulation order changes.

The runtime still authenticates one owned read of the original record bytes.
The join still checks receipt identity, independent Stage A proof and payload
coverage. The dispatcher still waits for Stage A proof and verifies submitted
slice files. Forward recovery retains its historical correction rule.

## PrismaBuild results

The baseline ran 460 tests with no failures or skips. The final refactor ran
537 tests with no failures or skips, including additional caller suites and
the new ledger matrix. Both ran as three CPU shards on dl380g10, using
`/home/rob/venvs/pq-pin-fca4c6ce0-pb027103d9/bin/python`. PrismaBuild verified
the installed PB `027103d9a8417e06c7f13356e58779a313cd7088` and Tessera
`fca4c6ce0e16c41d94a1a3c4cfc21c4548dec6bb` dependencies before pytest.

Every shard reserved one CPU and 6 GiB, with one pytest worker and one native
thread. Placement was portable, priority was `-10`, the action ceiling was
1,200 seconds and the per-test bound was 180 seconds. Each observed terminal
was `executed`, exit 0, with a CAS receipt and the actual pytest completion
record. The targeted compile action also executed with exit 0 and a CAS
receipt. It reserved one CPU and 2 GiB. The client default pinned that compile
action to the submitting host; its log recorded that no other live worker
qualified. Future portable direct submissions use `--anywhere` explicitly.

| Run | Action keys |
|---|---|
| Baseline | `eb545514b656ebafb3ec949a7075ce39b04b9b461ba2ca4b3b93cf6fe5db5630`, `6ae65e7c4cc0e363461f9cccd3a87891f66247ee92d7a1f0644a0ff1b0f671b6`, `5492011f44c4251c7471b08c2a134496776321f3788955c5e567b697609c665c` |
| Refactor | `8cd35ace2f121790dd46127e74fce61a64831771cbde5bc486036f1bb5c37e82`, `36ccd1d23f74be11a33cb4d8306a6e9cc1a31b713f59a6059e2450adc182b14a`, `c52640271ffeca7926cbb445d2a44f8a649a3daa0d30d52a63f89dab2bd09762` |
| Compile | `884ee9e8e6ae02a8d9ca23df76bc5c775501572da7e46cea5388a01e2ad890e5` |

The commands and complete reports are retained under
`/home/rob/tmp/pq-refactor-20261014/reports/`: `joint-j1-baseline.log`,
`joint-j1-baseline.json`, `joint-j1-baseline-receipts.json`,
`joint-j1-after2.log`, `joint-j1-after2.json`,
`joint-j1-after2-receipts.json`, and `joint-j1-compile.log`.
CAS action receipts are under
`/mnt/shared/prismabuild-fleet/cas/actions/v3/<first-two-key-characters>/<key>.json`.

The selected suites cover record production, runtime, join, dispatch,
forward recovery, launch wiring, metadata generations, band binding,
boundary and executable readsets, catalog extensions, dev seals, the seal
inventory, architecture and documentation checks, roster goldens, hash
routing, and Stage A head intake. The new ledger matrix covers body tampering,
certified and dev campaign comparisons, bound and unbound records, mixed
generations, and immutable replacements. A fixed UTF-8 SHA-256 expectation
pins the identity recipe independently of another wrapper using the same
encoder. The existing runtime fixture still compares real single-run and
quantum measurement envelopes bitwise.

The first refactor run found three diagnostic-text assertion failures and
obsolete structural seal counts. Refusal behavior remained intact. The final
run corrected those expectations and retained their safety cases.

## Test dispositions

- Removed `test_canonical_helpers_delegate_to_the_shared_implementation`.
  Comparing wrappers over the same encoder repeated the implementation.
  The fixed identity golden and NaN refusal now check observable behavior.
- Removed `test_the_joiner_is_not_colocated`. Attribute existence asserted
  module arrangement. Actual producer-to-join behavior remains covered by
  `test_producer_records_join_synthetic_runtime_outputs`.
- Replaced `test_check_quantum_for_campaign_real_plan` with the ledger's
  campaign and binding matrix. The new cases run without historical campaign
  files and cover both certified and dev modes.
- Retained the structural `test_no_new_seals` policy check. Its relocated
  inventory reflects the shared owner. It detects new run refusals that a
  finite behavior suite cannot enumerate.
- Retained all independent reader tampering, no-write, receipt custody,
  coverage, gapped-allocation, and publication-readiness tests. No timing
  ceiling or source-parsing behavior test was introduced.

No GPU run or performance experiment was needed for this ownership-only
change. No speed, residency, power, KL, or bpp improvement is claimed.
