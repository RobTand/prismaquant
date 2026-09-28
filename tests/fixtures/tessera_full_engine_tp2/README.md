# TP2 producer-path protocol fixtures

Synthetic CPU inputs, produced by Tessera's real assembler/report path, **not GPU
captures, placement certificates, runtime-cell evidence or fixed-resource prices**.
No imported Tessera serving code is needed to parse them.

- Producer commit: `e3f75d880097832b02d6eb586517e472b34a7963` (committed source only).
- Generator: `tests/fixtures/generate_full_engine_tp2_protocol.py --output tp2-protocol`.
- PrismaBuild action: `a5b56cd19c708a12f55cc79344fb94db74d82cbf371c7403f251e335151fbdf4`.
- Environment: `pq-pb461728e4-tessera-09d6559d`, 2 CPUs, 2 GiB, native threads 1.
- `report-rank0.json` SHA-256: `f6b2dd4ee6d3f50d0306161b2159c1561e842ce52db9b4d255597d73fdac6555`.
- `report-rank1.json` SHA-256: `71652eaa295aa6cf597ff2eefa6c46642ca2d62cb35188c199b3f698ac433d3c`.
- `timing-observation.json` SHA-256: `04920907cabb05247f6b0c80f13be4dfb1a008e4c0f05e91d12000dfef540312`.

The original 51-file bundle is preserved byte-for-byte, including producer-side
absolute paths. Those paths document the generating sandbox; they are not
portable execution inputs. `manifest.json` and the raw resource/timing subtrees
retain the production-path derivation context. The tests consume the report
bytes through locally sealed references and never claim the fixture paths are
live runs. Missing ownership/observer/fixed-resource qualifications remain
refusals; v3's whole off-step peak is observation-only and never substitutes for
a candidate-specific placement bound.
