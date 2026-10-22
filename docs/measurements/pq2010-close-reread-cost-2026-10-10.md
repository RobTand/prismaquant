# PQ #2010 close reread cost profile, 2026-10-10

## Method

The profile times one record, read, and close capture over 4 64 MiB
safetensors shards (256 MiB). It runs each arm 3 times warm and keeps the
minimum. The off arm replaces the close reread with a no-op. The on arm runs
the then-shipped close reread. Both arms meet hot page cache. This does not
represent a bounded capture whose traversal has already released source pages.
The box is dl380g10, a shared CPU worker. The interpreter is the pinned
`pq-pin-fca4c6ce0-pb027103d9` Python 3.14 build. Scratch is on `/tmp`.

## Result

| run | action | loadavg | admit 64 MiB (s) | reread 64 MiB (s) | capture off (s) | capture on (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `0164f4965636ccb7a71b4011bc69cb00c576b48234b32da42dd5e76d162cadba` | 31.2 | 0.223 | 0.208 | 0.833 | 1.678 |
| 2 (measurement) | `c82a93da2854d52ee14d5fc6c397ed777c622f00770fac3d8c61c2bb1235250c` | 24.0 | 0.243 | 0.243 | 0.898 | 1.902 |

The reread costs one sequential pass per file. It matches the admission
hash speed (0.24 s per 64 MiB, about 260 MB/s on a loaded box). It runs on
the CPU caller thread at close, off the GPU hot path. It keeps the owner
resource guard.

## Why the fixture ratio does not transfer

The fixture has no model forward. Hashing dominates its wall, so one extra
read about doubles it. A real capture spends its wall in the forward. The
monolith CPU capture file (`test_capture_single_pass_source_1896.py`) takes
79 s for 28 tests with 2 full captures on the same box class
(action `dc7648deb9bf2c7caa50ee81be45925a766d8ae05f188fca4fee4e6b1493ed95`).
Its roster is tens of MiB. Its close reread is a fraction of a second. The
share is estimated below 1 percent.

## Full-scale estimate

The added cost includes one sequential read/hash of the fresh roster. The
hot-cache microbenchmark above observed about 260 MB/s; it is not a lower
bound on throughput when released pages must be fetched from storage. The
original below-10-percent estimate is unvalidated, not a measured capture
result. Follow-up #2606 owns a real before/after capture profile, with the
actual page-release policy and storage path, and prices a ZFS snapshot read
path. If overhead exceeds 10 percent, the CEO cost rule requires that
snapshot-path follow-up; this microbenchmark cannot decide the threshold.

## Limits

The box was shared (loadavg 24 to 31). The two arms ran back to back, which
does not establish equal load. No Netdata series is attached: this is a CPU
sequential-read micro-cost, not a box-load claim. Page-cache-hot bytes are
not guaranteed: bounded captures deliberately release them. The repaired
close pass honors that policy incrementally and advises every held file at
cleanup, including files previously released and exceptional exits. Its
successful verification now precedes durable quantum/monolithic completion.
No new real-capture or GPU timing is claimed here.

## Completion/page-policy repair verification (CPU, not a cost profile)

The focused regressions exercise actual quantum completion and monolithic
`_run_streamed_calibration`: successful close precedes publication, while a
same-signature byte mutation leaves no new fragment/manifest for reuse or
join. The monolithic fixture isolates collection from unrelated serving-menu
validation; it does not bypass the production automatic-source refusal.
Selected and recording owners both release close-reread blocks and the final
tail, including previously released files, resource-guard refusal and digest
mismatch. Admission read counts and recorded-digest join reuse remain covered.

Baseline command, before the repair:

```sh
python -m pytest tests/test_capture_single_pass_source_1896.py tests/test_selected_source_authentication.py tests/test_capture_layer_chain.py tests/test_automatic_capture_source_qualification.py tests/test_docs_staleness.py tests/test_architecture_doc.py -q --basetemp="$TMPDIR/pytest"
```

Observed: **124 passed, 2 failed, 2 CUDA skips**. Both failures are the CPU
`test_a_streamed_capture_reads_each_source_file_once` policies, before their
capture assertions: the installed Tessera lacks `_evidence_smoke_record`.
This reproduces the review's dependency incompatibility, not a page-policy
regression. No serving gate, test skip, or dependency pin was changed to
conceal it. Their claimed end-to-end pass remains unverified in this arm.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"83e7eb6cf518289713dfafc030569634adc59259c5b266f31035e0f63f5434d4","size":16124}},"operation":"ffc4ae8ff59b492fd34450e0c66eaa08ee3210df9d837771f12543a8ea2ba426","receipt":"36efac858c6419443022307a8aafda38af8b4a22625d9e7a540dcaad2d039931"}]

Repaired targeted suite, excluding only those two known-incompatible CPU
cases (CUDA cases still discover and skip normally):

```sh
python -m pytest tests/test_capture_source_close_2010.py tests/test_capture_completion_2010.py tests/test_capture_single_pass_source_1896.py tests/test_selected_source_authentication.py tests/test_capture_layer_chain.py tests/test_automatic_capture_source_qualification.py tests/test_tessera_calibration_cache.py tests/test_docs_staleness.py tests/test_architecture_doc.py -q -k 'not (test_a_streamed_capture_reads_each_source_file_once and not cuda)' --basetemp="$TMPDIR/pytest"
```

Observed: **202 passed, 2 CUDA skips, 2 deselected** on the supplied x86,
Python 3.14 / CPU Torch arm. This is correctness evidence, not GPU
qualification or a real-capture before/after cost measurement. #2606 already
tracks the latter; no duplicate measurement issue is needed.

Final focused/documentation check after the documentation edits:
`python -m pytest tests/test_capture_source_close_2010.py tests/test_capture_completion_2010.py tests/test_docs_staleness.py tests/test_architecture_doc.py -q --basetemp="$TMPDIR/pytest"`
— **29 passed**.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"8f446e016fd118c5d1058a88faf5ebf91fff956d8fc757f1d68e29c60355f9bc","size":822}},"operation":"56d5db25ec568f915cf697d0e7c370318478c3f7a45d2e2dddb9ece6944259b2","receipt":"936680ea89debbed3d7d6bb8de7b1e66295b38436c080288a263f1da275274de"}]

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"399a8a1e0b79cb76490df4e0ae376f74fd089ed41dcd5d2432f7cf7bfa41e129","size":1306}},"operation":"d90e92f31c65f037ea01c4bc7efef94e2383c4369dd1c4d79b648a120de3fe31","receipt":"166cd01de93d6f70da88073a1182b7376c0601c20ccb991e1d7274e8e123f622"}]
