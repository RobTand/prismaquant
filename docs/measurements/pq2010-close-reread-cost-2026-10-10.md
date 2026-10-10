# PQ #2010 close reread cost profile, 2026-10-10

## Method

The profile times one record, read, and close capture over 4 64 MiB
safetensors shards (256 MiB). It runs each arm 3 times warm and keeps the
minimum. The off arm replaces the close reread with a no-op. The on arm runs
the shipped close reread. Both arms meet hot page cache, as in production.
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
share is below 1 percent.

## Full-scale bound

The added cost equals the roster bytes divided by the sequential read rate.
The measured floor is 260 MB/s under heavy load. Production captures run
minutes of GPU forward first. The close reread stays below 10 percent of
capture wall with wide margin. No ZFS snapshot read path is needed.

## Limits

The box was shared (loadavg 24 to 31). The two arms ran back to back, so
load cancels in their comparison. No Netdata series is attached: this is a
CPU sequential-read micro-cost, not a box-load claim. The estimate assumes
page-cache-hot bytes, which holds because the capture just read them.
