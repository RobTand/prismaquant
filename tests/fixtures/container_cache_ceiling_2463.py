"""Measured container-cache ceiling for one Stage B quantum (PQ #2463).

Part of PQ #1091. The ROOT/MAX pair declares an operator-chosen reservation.
This fixture binds a measured peak to the derived ceiling, so the ceiling
derives from evidence, not from a guess.

Workload scope: a synthetic temp-file workload that grows then shrinks, run
on CPU through ``container_cache_peak.measure_around``. It proves the
instrument: the recorded peak exceeds the final size. It does not prove a
production ceiling. A GPU run on a real Stage A/B row must replace the
measurement fields and re-derive the ceiling before any production claim.

Derivation: ``C = G * ceil((P + H) / G)`` with ``G = 1073741824``. The PB
reservation charges ``ceil(C / G)`` GiB for the cache plus each separate
scratch reservation. Field ``measurement_digest`` is the SHA-256 of the
canonical receipt bytes the peak came from.
"""
from __future__ import annotations

FIXTURE = {
    "schema": "prismaquant.container_cache_ceiling.v1",
    "issue": "prismaquant#2463",
    "measurement_digest": "REPLACE_WITH_GPU_RECEIPT_SHA256",
    "peak_allocated_bytes": 1048576,
    "headroom_bytes": 1074790400,
    "gib_bytes": 1073741824,
    "ceiling_bytes": 2147483648,
    "formula": "C = G * ceil((P + H) / G)",
    "pb_cache_gib": 2,
}
