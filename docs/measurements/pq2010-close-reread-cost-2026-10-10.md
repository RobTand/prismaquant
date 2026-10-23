# PQ #2010 close reread cost profile, 2026-10-10

## Method

The profile times one record, read, and close capture over 4 64 MiB
safetensors shards (256 MiB). It runs each arm 3 times warm and keeps the
minimum. The off arm replaces the close reread with a no-op. The on arm runs
the shipped close reread. Both arms use hot page cache in this fixture;
this is not an assumption that holds for bounded production traversal.
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
read about doubles it. A real capture also spends wall time in the forward.
The historical monolith CPU test file (`test_capture_single_pass_source_1896.py`)
reported 79 s for 28 tests with two captures on the same box class
(action `dc7648deb9bf2c7caa50ee81be45925a766d8ae05f188fca4fee4e6b1493ed95`).
That includes other tests and setup, not just capture wall, so it cannot
establish a capture overhead percentage. The repair profile below times
actual capture calls directly, but still uses only a tiny synthetic model.

## Full-scale estimate and unresolved cost condition

The added cost equals the roster bytes divided by the effective sequential
read-and-hash rate. The historical hot-cache microbenchmark measured about
260 MB/s under load, not a portable floor. Bounded traversal can already
have evicted shard pages: its terminal pass must reread storage and release
pages incrementally again. It also gives final eviction advice for every
held file, including previously released shards, on success or refusal.

The original below-10-percent estimate is not a measured production-capture
result. No representative original-source before/after measurement backs
that share yet. Existing follow-up #2606 owns that profile and the ZFS
snapshot read path; the CEO's greater-than-10-percent cost decision remains
pending that measurement. A snapshot would make the source immutable
without the second read. The integrity reread stays enabled meanwhile.

## Limits

The historical box was shared (loadavg 24 to 31). Back-to-back arms reduce
but do not cancel shared-load differences. No Netdata series is attached:
this is a CPU sequential-read micro-cost, not a box-load claim. Test-suite
wall time is not capture wall time, and neither proves the production cost
percentage. No full-scale or original-provider GPU qualification is claimed.

## Repair GPU validation plan (recorded before execution)

Decision: do both existing CUDA recording policies still pass after moving
verification before publication, and what is the paired close-pass cost in
an actual, but tiny synthetic-model, bounded GPU recording traversal?
CPU checks cannot establish CUDA transfer/reader lifetimes or GPU capture
wall time. The smallest case is the existing three-layer GLM fixture with
one 257-token sample, both CUDA policy tests, then one warm pair and three
alternating measured off/on pairs using `shared-inputs-bounded-v1`.
Only the ephemeral off arm disables `_verify_held_content`; production
policy is unchanged. Synchronize CUDA around wall timing and time the CPU
terminal checks separately. Expected GPU allocation: at most four minutes,
not a sweep. This measures fixture cost, not the unresolved full-source
10-percent condition or a qualified immutable source provider.

## Repair results

The two CUDA policy cases passed on GB10 (Python 3.14.4, Torch
2.11.0+cu130, Transformers 5.17.0), with no skips. The combined first action
then failed its profile setup before any timed capture: it omitted the
bounded policy's required startup settings. This was not a passing profile.
The corrected profile supplied `PRISMAQUANT_RELEASE_SOURCE_PAGES=1` and
`MIMALLOC_PURGE_DELAY=0` before Python startup; no environment guard changed.

[Test evidence {"artifacts":{"stderr":{"digest":"aa4b5ce70cd3aa8f97c6b65ffa46acd1d81587ac789da266535995a46e799af0","size":2541},"stdout":{"digest":"dfaf68e8c0d4eaba3429e898f994cd07dfb7eb1ca2418cc5a23c10efa12fc45a","size":3011}},"operation":"01403488fc6deadcbe07791b55ec6c8177882c7b60cd0bcca8643ec1582075a9","receipt":"b87e6c64eda1c96e87c4b6ce1d71d628d6271228fa1fc7cf83429803a42d4dcf"}]

The corrected command used `_glm_source` from
`tests/test_capture_single_pass_source_1896.py`: installed public producer
CLI, one test-local serving-menu rung, and the existing controlled legacy
recording-gate stub. Actual model traversal, expert projection, CPU source
authentication, capture writer and manifest publication remained real.
Temporary inputs/outputs stayed under the run's local `$TMPDIR`.

One warm off/on pair was excluded, then three measured pairs alternated
on/off order. Each `campaign.main` capture was bracketed by
`torch.cuda.synchronize()` and `time.perf_counter()`. Each arm used a fresh
capture/cache directory with the same census, tokens, roster, resource guard
and page-release policy. Only the off arm replaced `_verify_held_content`
with a no-op. The on arm timed the unmodified method and traced hash calls.

| measured pair | off wall (s) | on wall (s) | on terminal checks (s) |
| --- | ---: | ---: | ---: |
| 1 | 2.032202 | 2.043633 | 0.010619 |
| 2 | 2.019933 | 2.036849 | 0.012365 |
| 3 | 2.025714 | 2.020424 | 0.010972 |
| median | 2.025714 | 2.036849 | 0.010972 |

The roster was **5,972,264 bytes**. Median capture wall increased **0.55%**;
median terminal checks were **0.54%** of on-arm capture wall. Individual
wall times overlap, so the delta is not a precise overhead estimate even
for this fixture. The on arm added eight hash calls (8 to 16), all eight
with page release enabled (release-enabled calls 3 to 11). This exercises
the bounded advice path, not a verified cold-storage/cache-residency state.
All eight warm/measured captures produced manifests. This tiny synthetic
capture is **not** a representative full-source profile or evidence that
the production 10-percent condition holds. Follow-up #2606 remains needed.

[Test evidence {"artifacts":{"stderr":{"digest":"94ba9e1a65b333abc821d39b2d111bc7fd054eec2de2f6fb6ebf3ee185f98bdf","size":591},"stdout":{"digest":"34fe1445569afe0996660693ac2334860089dc82b5fe67d4f2db42de205ba7c5","size":92098}},"operation":"5b0b159095a936bd4081e2d5d0f92a1786d6273756487b92da3fc19cb9075aee","receipt":"ba1fb7e8c812b24f598dabc00cc6cbd62b1262de7dd2f892f26e409a6bb84926"}]

### Integrity regression validation

The default CPU action ran the following suite and then compiled all three
changed production modules: **346 passed, 2 CUDA skips**, no deselections;
compilation passed. The default action landed on aarch64. Explicit `arm="x86"`
was unavailable (no command ran); x86 remains the pipeline's check arm.

```sh
python -m pytest tests/test_capture_completion_2010.py tests/test_capture_source_close_2010.py tests/test_capture_single_pass_source_1896.py tests/test_selected_source_authentication.py tests/test_selected_source_authentication_ab.py tests/test_selected_source_validation_charge.py tests/test_selected_snapshot_scope.py tests/test_tessera_calibration_cache.py tests/test_capture_layer_chain.py tests/test_tessera_source_digest_adoption.py tests/test_automatic_capture_source_qualification.py tests/test_streaming_source_pages.py tests/test_collector_source_release.py tests/test_glm_capture_compatibility_cli.py tests/test_duplication_baseline.py tests/test_io_site_freeze.py tests/test_io_site_freeze_aliases.py tests/test_docs_staleness.py tests/test_architecture_doc.py -q --basetemp="$TMPDIR/pytest"
PYTHONPYCACHEPREFIX="$TMPDIR/pycache" python -m compileall -q prismaquant/tessera_calibration_cache.py prismaquant/tessera_campaign.py prismaquant/capture_layer_chain.py
```

Both formerly failing CPU capture policies run without the unrelated
serving-menu dependency or a private producer interpreter. They still check
the real descriptor-backed admission/close hashes, complete source receipt
and artifact identity. Automatic original-provider admission remains refused.
The new completion regressions freeze only metadata to reproduce
same-signature mutation: refusal leaves no new complete quantum, reusable
fragment, successor/join result or monolithic manifest. Page-policy tests
cover recording and selected owners, resource-guard/digest/stat refusal,
previous release, deferred reader/tensor lifetimes, and idempotent cleanup.

[Test evidence {"artifacts":{"stderr":{"digest":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855","size":0},"stdout":{"digest":"43079d408ad285557d505c589fdd837e79acedc27a3aa94ec168e90004ecad3d","size":1512}},"operation":"1fd46ba8e5a82b2296bc084a1b011248113e9bde9459336b2673a9ca8a5bdada","receipt":"b6c8c88a89c516b9b02161a658cf29889ac7ee30ecb91319d40a1ef5c22260fc"}]
