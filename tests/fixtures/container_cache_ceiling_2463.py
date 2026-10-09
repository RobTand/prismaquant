"""Measured container-cache ceiling for the Stage B compile row (PQ #2463).

Part of PQ #1091. The ROOT/MAX pair declares an operator-chosen reservation.
This fixture binds two measured GPU peaks to the derived ceiling, so the
ceiling derives from evidence, not from a guess.

Workload scope: ``tools.measure_container_cache_row`` in the known-good
campaign container (image ``sha256:c0e5...``, content
``d0256efb...``; torch 2.13.0+cu130, triton 3.7.1, transformers 5.16.1, GB10
compute 12.1). The row compiles the served NVFP4 RTN activation quantiser
(``torch.compile``) and probes the six KDA capture kernels (``triton.jit``
launches) under ``PRISMAQUANT_CONTAINER_CACHE_ROOT``, with
``PRISMAQUANT_TMPDIR`` in a separate charged workspace. Initial cache state
for the measured rows: a fresh empty root (row 2 proves the cold-compile
peak; row 1 ran first on its own fresh root). Two rows ran the same
workload; both peaks fit the ceiling.

Derivation: ``C = G * ceil((P + H) / G)`` with ``G = 1073741824``.
``P = 3596288`` is the maximum accepted peak. ``H = 2605056`` is four times
the largest single-interval growth seen (651264 bytes in 250 ms): one more
unseen jump past the final sample (the peak sits at the last sample), one
concurrent-compiler burst, and scan-gap margin. ``C = 1073741824`` (1 GiB).
The PB reservation charges ``ceil(C / G) = 1`` GiB for the cache plus each
separate scratch reservation. ``measurement_digest`` is the SHA-256 of the
canonical row-1 receipt bytes; ``repeat_digest`` is the row-2 receipt.

Scope limit: this ceiling covers this workload, runtime, concurrency and
initial cache state only. It proves no quota enforcement, cleanup or crash
recovery (PQ #1091 and PB #1360 own those).
"""
from __future__ import annotations

FIXTURE = {
    "schema": "prismaquant.container_cache_ceiling.v1",
    "issue": "prismaquant#2463",
    "measurement_receipt": "container_cache_peak_row1_2463.json",
    "repeat_receipt": "container_cache_peak_row2_2463.json",
    "measurement_digest":
        "4acc1cfe81cba1a0c4c9dc84101ffe84ad1c278461da60643390c074c4d001b0",
    "repeat_digest":
        "fee88bd132453221dde9911e607aef30f79eccf51fe7c8d0ae89a211108051e7",
    "peak_allocated_bytes": 3596288,
    "repeat_peak_allocated_bytes": 3596288,
    "headroom_bytes": 2605056,
    "headroom_basis": ("4x the largest single 250 ms interval growth "
                       "(651264 B): one unseen jump past the final sample, "
                       "one concurrent compile burst, scan-gap margin"),
    "gib_bytes": 1073741824,
    "ceiling_bytes": 1073741824,
    "formula": "C = G * ceil((P + H) / G)",
    "pb_cache_gib": 1,
    "workload": "tools.measure_container_cache_row",
    "runtime": {"torch": "2.13.0+cu130", "triton": "3.7.1",
                "transformers": "5.16.1", "device": "NVIDIA GB10",
                "capability": [12, 1]},
    "container": {"image": "sha256:c0e532d28a78b3bf425bbbc0d862e2840ba624249162aedfadd09748a6c68c37",
                  "content_sha256":
                      "d0256efb83294e879ca33dd2d3131e861221c415ac5b024c2415e51c5467f026"},
}
