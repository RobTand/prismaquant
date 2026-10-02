# Tessera Forest Byte Tests: Avoid Quality Refits

Issue #2098, a bounded test-speed slice of #1929. B31 shard 14 attributed
481.43 of 490.41 pytest seconds to `tests/test_tessera_forest_bytes.py`.
That ranking selected this file for diagnosis; it is not the timing baseline
for the comparison below, and shard 10 was B31's longest shard at 567.61 s.

## Change And Coverage

The existing routine small-shape and WINDOW exporter tests now pass Tessera's
`scale_refit=0`. They retain a real trellis search, the default wire recipe,
plane serialization and round-trip verification. The default four searches
and quality refits change reconstructed values, but not the plane extents
these tests assert. Every shape, rung, parametrized case and assertion remains.
The eight large-shape cases retain their existing slow opt-in and default
refits. No production code, serving gate, calibration or numerical default moves.

An incidental stale comment about refusing partial superblocks was corrected
in its own commit: the same module already covers their pricing under #1849.

## Workload And Results

Both file runs executed through PB on `dl380g10`, CPU-only, with two admitted
CPUs, 4 GiB memory and `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`.
The interpreter was
`/home/rob/venvs/pq-pb059953bc-tessera-b40c93cb/bin/python`: Python 3.14.4,
torch 2.11.0+cpu, Transformers 5.16.1 and Tessera's installed Git provenance
`b40c93cb73745097e57a1ba4cf5b9eee166c759a`. Py-spy 0.4.2 sampled both at 100 Hz.
Native CPU affinity was assigned by PB. Neither action requested a GPU.

| Measurement | Before | After |
| --- | ---: | ---: |
| Pytest wall, seconds | 247.79 | 114.61 |
| Window exporter calls, seconds, summed pytest durations | 158.30 | 41.06 |
| `viterbi_window` inclusive sampled seconds | 200.29 | 61.98 |
| `encode_linear` inclusive sampled seconds | 211.57 | 70.36 |
| Action cgroup CPU seconds | 288.95 | 136.71 |
| Peak action cgroup memory, MiB | 1008.7 | 1003.1 |
| PB Netdata host CPU busy, mean percent | 4.68 | 11.21 |
| PB Netdata host CPU busy, peak percent | 23.62 | 44.40 |
| Tests passed / skipped | 41 / 8 | 41 / 8 |

The observed file wall delta is 133.18 s (53.75%). These are single sequential
file observations with different background load, not an isolated repeated
timing experiment or a full-suite speed claim. The causal count comparison
below establishes the removed searches independently. Inclusive profile times
overlap across functions and must not be added together. All eight skips were
the existing `PRISMAQUANT_TESSERA_SLOW_ENCODE!=1` large-shape cases; they were
not newly executed in this slice. The runs reported 14 existing torch warnings.
CPU-only receipts lack a pqteld GPU CSV; no GPU or energy claim is made.

The retained Netdata series cover the executing host and both Sparks. For the
same before/after windows, historical `system.cpu` queries measured mean busy
percent of 4.39 / 15.52 on sparky and 1.88 / 2.43 on sparklina. Raw JSON is in
`/home/rob/tmp/sol-test-speed-20261002/{before,after}-{dl380g10,sparky,sparklina}-cpu.json`.
The PB contemporaneous dl380 summaries above are authoritative for its action
windows; later history queries can differ slightly at the boundary samples.

## Causal Checks

PB `8cb92bf47307` ran a caller guard against the original file and failed all
15 routine exporter cases because they omitted the zero-refit argument.
PB `a7134ab4dfd7` then checked six 8x256 real encodes at the existing grid/rung
pairs. Parsed plane counts, chunk lengths, role facts, schedules and exact
bytes agreed between default and zero-refit arms, and actual searches were:

| Grid / rung | Default | Zero Refit |
| --- | ---: | ---: |
| E2M1x2 / 895 | 8 | 2 |
| E2M1x2 / 896 | 4 | 1 |
| E2M1 / 511 | 8 | 2 |
| E2M1 / 512 | 4 | 1 |
| E4M3 / 256 | 4 | 1 |
| BF16 / 256 | 4 | 1 |

Mixed-rate schedules need one search per distinct rate per pass. All six
blobs differed in values; this comparison proves layout equality, not byte
identity or reconstruction-quality equality. The probe also zeroed the
accountant's forest term: both existing K2/R896 and K1/R511 exporter tests
raised AssertionError, preserving sensitivity to the original omission.
One representative updated exporter case passed under the caller guard.
`py_compile` for the touched test module passed inside this admitted action.

The first probe (`a69d358b1524`) failed its call-count assertion because the
first export also runs Tessera's existing encoder identity self-check.
The same-worker diagnostic (`9771f2da916d`) confirmed those extra calls. The
successful probe explicitly warms `encoder_fixture_id()` before counting;
its identity is `03bbc5b1c56d55e1d7f5f0d1baa1107e462d5bad18a412d0c232c78d04c95519`.
Failed and diagnostic records are retained as bounded evidence, not passes
for the final causal check. The probe source is retained outside the repo at
`/home/rob/tmp/sol-test-speed-20261002/causal-probe.py`.

## Reproduction And Receipt Bindings

Both profiled file actions use this command, changing only the checkout source:

```bash
python3 /mnt/shared/prismabuild-fleet/repo/tools/pbrun.py \
  --cwd /home/rob/tmp/sol-test-speed-20261002/wt --anywhere \
  --cpus 2 --demand mem_gb=4 --priority -10 --timeout-s 1500 \
  --profile sample --env OMP_NUM_THREADS=1 --env MKL_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 -- \
  /home/rob/venvs/pq-pb059953bc-tessera-b40c93cb/bin/python \
  -m pytest tests/test_tessera_forest_bytes.py -q -p no:cacheprovider --durations=25
```

- Before action: `4efb6ac064b0116f9d251cc8a4a1574b2f01d42e6b4d33185236e26974d31168`,
  parent `693a38f3ae3467ac3234cd9a7d756e41b69c3242`, exit 0.
- After action: `3ecb51b7b36030e3157223e559302f4488979cc1c408afea4e01ace8b74d755d`,
  parent `bf7372a66356bc17be94cac0b1bb9d50fb7c9fab`, exit 0.
- Before profile SHA-256: `2d7421c921fd97e28c8069615e0b2ad9520c1962fa59cbeb378354eaf991574c`.
- After profile SHA-256: `a3874efffd6fa4ee0a9ad560f718a9a72adf22df577aa9333092cb3f11016987`.
- Before local-result claim: `bd8a0abc1263ba74562ca3235f8fbd434394ec5161260708e4cb0c9491953a87`.
- After local-result claim: `e746409c5d3a2010a969ac5ca9511d04f710d6adcc68fb2903888aa64a3cb2d8`.
- Final causal action: `a7134ab4dfd7762f0f3e476a57b1109f8ddbe141815473f8913c1109c10683e0`,
  exit 0, local-result claim `0f56ae6b52bf7394046a177f9004a7078a235f6ceb778ce15bf038ce2ca21e07`.

Profile blobs live under `/mnt/shared/prismabuild-fleet/cas/blobs/<first-two>/<sha256>`;
receipts under `cas/actions/v3/<first-two>/<action-key>.json`. PB log views retain
the test durations and causal output. Hash-payload verification passed for all
three successful local-result claims, and profile content hashes were checked.
PB's claim verifier does not independently verify worker attestation; that
requires its full action-manifest verifier. The measured after test source hash
is `568047cf8efcecf52c2a75418c634930ac9da65051598c6904943ab5eeffca24`;
the subsequent documentation commit does not change that test source.
