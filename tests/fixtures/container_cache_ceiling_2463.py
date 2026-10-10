"""Measured container-cache ceiling for the complete quantum row (PQ #2463).

Part of PQ #1091. The ROOT/MAX pair declares an operator-chosen reservation.
This fixture binds two measured GPU peaks to the derived ceiling, so the
ceiling derives from evidence, not from a guess.

Workload scope: ``tools.measure_container_cache_quantum_row`` in the
known-good campaign container (image ``sha256:c0e5...``, content
``d0256efb...``; torch 2.13.0+cu130, triton 3.7.1, transformers 5.16.1, GB10
compute 12.1). The row runs a complete Stage B layer quantum: the Stage A
adjoint capture core on a tiny two-layer bf16 fixture model, then one
layer quantum core for that capture's slice (one unit, three formats,
two probes), then the served NVFP4 RTN activation quantiser compile
(``torch.compile``) and the six KDA capture kernels (``triton.jit``
launches). Caches route under ``PRISMAQUANT_CONTAINER_CACHE_ROOT`` (hf,
triton, inductor, xdg). ``PRISMAQUANT_TMPDIR`` sits in a separate charged
cotangent workspace. Each row starts from a cleared empty cache root;
both rows grew from ~4 KiB to the peak over 27 gap-free 250 ms samples.

Derivation: ``C = G * ceil((P + H) / G)`` with ``G = 1073741824``.
``P = 3596288`` is the maximum accepted peak. ``H = 4341760`` is four
times the largest single-interval growth seen (1085440 bytes in 250 ms):
one more unseen jump past the final sample (the peak sits at the last
sample), one concurrent-compile burst, and scan-gap margin.
``P + H = 7938048``, so ``C = 1073741824`` (1 GiB). The PB reservation
charges ``ceil(C / G) = 1`` GiB for the cache plus each separate scratch
reservation. ``measurement_digest`` is the SHA-256 of the canonical row-1
receipt bytes; ``repeat_digest`` is the row-2 receipt.

Scope limit: this ceiling covers this workload, runtime, concurrency and
initial cache state only. It proves no quota enforcement, cleanup or crash
recovery (PQ #1091 and PB #1360 own those).
"""
from __future__ import annotations

FIXTURE = {
    "schema": "prismaquant.container_cache_ceiling.v1",
    "issue": "prismaquant#2463",
    "measurement_receipt": "container_cache_quantum_row1_2463.json",
    "repeat_receipt": "container_cache_quantum_row2_2463.json",
    "measurement_digest":
        "074ba9750361c0288f463b3fecb66b26365431023181942ef104e78a14f7134c",
    "repeat_digest":
        "a52a2c5e341020f0876ec6bd81691a7fdb173502aa66fcdd143484adc5e08c61",
    "measurement_action":
        "628ceda204fb2e1b5e126c66926f9da49d8da3945006d540d058ba0880bac078",
    "repeat_action":
        "02da0e85bb7472fc6f2ba0a5df2940fd046449475cbc5977585b8dc72cc94a4e",
    "peak_allocated_bytes": 3596288,
    "repeat_peak_allocated_bytes": 3596288,
    "headroom_bytes": 4341760,
    "headroom_basis": ("4x the largest single 250 ms interval growth "
                       "(1085440 B): one unseen jump past the final sample, "
                       "one concurrent compile burst, scan-gap margin"),
    "gib_bytes": 1073741824,
    "ceiling_bytes": 1073741824,
    "formula": "C = G * ceil((P + H) / G)",
    "pb_cache_gib": 1,
    "workload": "tools.measure_container_cache_quantum_row",
    "runtime": {"torch": "2.13.0+cu130", "triton": "3.7.1",
                "transformers": "5.16.1", "device": "NVIDIA GB10",
                "capability": [12, 1]},
    "container": {"image": "sha256:c0e532d28a78b3bf425bbbc0d862e2840ba624249162aedfadd09748a6c68c37",
                  "content_sha256":
                      "d0256efb83294e879ca33dd2d3131e861221c415ac5b024c2415e51c5467f026"},
}
