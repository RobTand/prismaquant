# Adaptive rate-curve CPU query cost

Date: 2026-09-11. Synthetic CPU measurement; no model-quality or campaign
speedup claim.

`experiments/adaptive_query_benchmark.py` compares the new
`AdaptiveAnchoredCurve` with the existing `TesseraRateSurface`. Both log2
arms use identical measured anchor coordinates and values, and query every
integer rate from 832 through 1088. The predictions match bit-for-bit across
each query roster. A separate value-space arm has no paired prior baseline.

The benchmark uses a deterministic synthetic decreasing curve, seven
interleaved timing samples of 719,600 queries per arm, and separate cProfile
runs. Construction and adaptive acquisition are outside query timing. It ran
on dl380g10 with Python 3.12.11, one admitted performance core (CPU 0), and
native threads bounded to one.

| Anchors | Existing log2, median µs/query | Adaptive log2, median µs/query | Adaptive value, median µs/query |
|---:|---:|---:|---:|
| 2 | 1.335 | 2.069 | 1.904 |
| 5 | 1.393 | 2.038 | 1.882 |
| 17 | 1.847 | 2.320 | 2.183 |
| 65 | 3.320 | 3.087 | 2.980 |

The adaptive implementation costs more per query at small anchor counts and
slightly less at 65. Its profile includes sorting the measured coordinates
and bisecting the interval; the existing implementation searches anchors
linearly. This measurement does not justify a general speedup claim.
Adaptive initialization takes about 0.2 ms. Acquiring all 65 synthetic
anchors takes 88.9 ms median in log2 mode, excluding initialization; the
oracle here is a cheap Python function, not a GPU quantization measurement.

The whole PB action took 144.899 s wall and 143.902 CPU-s, with a 375,484,416
byte peak cgroup memory footprint. The PB Netdata window contains 147 host
CPU samples, averaging 3.766% busy and peaking at 5.156%; one core was
reserved for measurement isolation. Netdata CPU and RAM histories from both
Sparks cover the same window. This CPU-only measurement has no GPU power or
work-per-joule claim; the absent dl380g10 pqteld series is explicitly recorded.

Evidence root:
`/mnt/shared/tessera-measurements/glm-canonical-census-20260908/sparse-rate-20260911`.

- PB measurement action:
  `880a798987598a153fd8bbc4b98a082db8385fdd1617125224a7f1dbd9c838cd`, exit 0.
- Checked receipt: `adaptive-fix-benchmark-verified-01.json`.
- Actual report: `adaptive-query-benchmark-01/report.json`, SHA256
  `94165001d7c805376514da777b16e73a33e1808fbc0f76be202427ebc2924d25`.
- Twelve binary profiles and twelve readable pstats reports:
  `adaptive-query-benchmark-01/profiles`.
- Spark host series: `telemetry/adaptive-query-benchmark-01.json`.
- Artifact hashes: `adaptive-terminal-benchmark-artifacts-verified-01.json`.

Reproduce through PrismaBuild from dl380g10 with `--measurement`, one CPU,
2 GiB, the recorded thread limits and the known CPU environment:
`/home/rob/venvs/pq-cpu312/bin/python experiments/adaptive_query_benchmark.py
--output /new/output/path`. The output path must be new. The earlier short
preliminary timing artifact is superseded and is not a reported sample.
