#!/usr/bin/env python3
"""Write the Tessera #509 traces this repo's route gate is tested against.

These fixtures are written by TESSERA'S OWN telemetry, never hand-authored
here.  The command below runs ``tessera.serving.telemetry._RouteTrace`` from
the producer tree at the commit named in ``PROVENANCE.md``, flushes it, and
leaves the JSON it wrote in this directory.  That is what keeps the consumer
honest about the schema it actually has to read: the header stamps
(``identity_version``, ``rank``, ``world_size``, ``rank_source``,
``rank_conflict``, ``platform``) and the per-entry identity (``module_names``,
``unnamed_modules``, ``dispatches_without_prefix``) are the producer's, so a
change there shows up as a fixture that no longer parses rather than as a
synthetic document that quietly drifts from it.

    /home/rob/venvs/pq-cpu312-tessera-4c384e60/bin/python \
      tests/fixtures/tessera_route_trace_509/generate.py \
      --producer-src /home/rob/tmp/tessera-509-route-trace-identity/src \
      --producer-commit e72d581 --out-dir tests/fixtures/tessera_route_trace_509

Two serves are emitted, each on two ranks at two token counts.  In
``routes-rank*.json`` every module rides the activation contract its price
names.  In ``routes-rank*-swapped.json`` two dense modules of different
families -- layer 0's BF16 MLP and layer 1's NVFP4 shared-expert down
projection -- ride each other's contract, which leaves every count in the
histogram identical and moves only the names.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

#: ``(target, policy family, structure, activation contract, N:K, symbol,
#: decoder)`` -- the five modules ``tessera_route_trace_m44e1/config.json``
#: prices, on the routes its TP2 eager serve dispatched.
MODULES = (
    ("model.language_model.layers.0.mlp.down_proj", "TESSERA_BF16", "dense",
     "bf16_unquantized", "N4096:K6144", "torch.mm", "torch_window"),
    ("model.language_model.layers.0.mlp.gate_up_proj", "TESSERA_FP8", "dense",
     "fp8_per_token_dynamic", "N8192:K6144", "torch.mm", "torch_window"),
    ("model.language_model.layers.1.mlp.experts", "TESSERA_NVFP4", "routed_moe",
     "e2m1_group16_ue4m3_static", "N2048:K512",
     "vllm.fused_moe.modular_kernel:FLASHINFER_CUTLASS", "native_span2"),
    ("model.language_model.layers.1.mlp.shared_experts.down_proj",
     "TESSERA_NVFP4", "dense", "e2m1_group16_ue4m3_static", "N4096:K1024",
     "torch.mm", "torch_window"),
    ("model.language_model.layers.1.mlp.shared_experts.gate_up_proj",
     "TESSERA_NVFP4", "dense", "e2m1_group16_ue4m3_static", "N8192:K1024",
     "torch.mm", "torch_window"),
)

#: The two modules the swapped serve moves between contracts.  Dense and of
#: different families, so the swap is invisible to a per-contract count.
SWAPPED = (
    "model.language_model.layers.0.mlp.down_proj",
    "model.language_model.layers.1.mlp.shared_experts.down_proj",
)

TOKEN_COUNTS = (1, 178)
WORLD_SIZE = 2


class _Layer:
    """A stand-in for a vLLM Linear: the trace reads only ``prefix``."""

    def __init__(self, prefix):
        self.prefix = str(prefix)


def _swapped_modules():
    """``MODULES`` with the two swapped pairs riding each other's contract."""
    by_target = {module[0]: module for module in MODULES}
    first, second = (by_target[target] for target in SWAPPED)
    served = dict(by_target)
    for target, other in zip(SWAPPED, (second, first)):
        own = by_target[target]
        served[target] = (own[0], other[1], own[2], other[3],
                          own[4], own[5], own[6])
    return [served[module[0]] for module in MODULES]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-src", required=True,
                        help="the producer tree's src/ (tessera_route_trace_identity)")
    parser.add_argument("--producer-commit", required=True,
                        help="the producer commit these fixtures were written at")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--platform", default="sm_121")
    args = parser.parse_args(argv)

    producer = Path(args.producer_src).resolve()
    if not (producer / "tessera" / "serving" / "telemetry.py").is_file():
        raise SystemExit(f"{producer} does not hold tessera/serving/telemetry.py")
    sys.path.insert(0, str(producer))
    from tessera.serving import telemetry  # noqa: PLC0415 -- the producer's own module

    if telemetry.IDENTITY_VERSION != 1:
        raise SystemExit(
            f"the producer writes identity_version {telemetry.IDENTITY_VERSION}; "
            "these fixtures and the consumer that reads them are written for 1")

    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for variant, modules in (("routes", list(MODULES)),
                             ("routes-swapped", _swapped_modules())):
        for rank in range(WORLD_SIZE):
            path = out / f"{variant}-rank{rank}.json"
            # The process facts the producer reads: this rank of this world,
            # and the platform token its route records carry.  Both are the
            # producer's own sources -- nothing is written into the file by
            # this script, it only says what the process would have observed.
            telemetry._process_rank = lambda rank=rank: (
                rank, WORLD_SIZE, "torch.distributed")
            telemetry._PLATFORM = args.platform
            trace = telemetry.start_route_trace(path)
            try:
                for token_count in TOKEN_COUNTS:
                    for target, family, structure, contract, shape, symbol, decoder in modules:
                        telemetry.emit_route(
                            _Layer(target),
                            kind=telemetry_trace_kind(structure),
                            policy=f"{family}:resident", symbol=symbol,
                            shape=f"M{token_count}:{shape}", contract=contract,
                            decoder=decoder)
                trace.flush()
            finally:
                telemetry.stop_route_trace()
            payload = json.loads(path.read_text())
            for entry in payload["entries"]:
                assert entry["modules"] == (
                    len(entry["module_names"]) + entry["unnamed_modules"])
            print(f"wrote {path.relative_to(out.parent.parent.parent)}: "
                  f"{len(payload['entries'])} entries, "
                  f"rank={payload['rank']}/{payload['world_size']} "
                  f"@ {payload['platform']}")
    return 0


def telemetry_trace_kind(structure: str) -> str:
    """The trace's ``kind`` for a priced structure (PrismaQuant's mapping)."""
    return {"dense": "dense", "routed_moe": "moe"}[structure]


if __name__ == "__main__":
    raise SystemExit(main())
