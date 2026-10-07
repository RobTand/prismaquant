"""What every Tessera admission gate answers under the tracked v2 pin (#1561).

A helper, not a test module. ``snapshot()`` records, as canonical JSON, the
answers PrismaQuant's Tessera gates give today under the tracked serving pin:

- the tracked pin's fields and whether the live pin gate admits;
- every packaged eligibility cell as parsed, and the unit route each cell's
  scope resolves to at every rung, residency and execution mode it names;
- the development contract's reviewed answer and its admitted cells;
- the route-trace gate's verdicts on the committed producer traces, with and
  without the ``serving_source_sha256`` header key Tessera v41 added.

``tests/fixtures/tessera_serving_identity_v2_snapshot.json`` is this output
taken on ``origin/main`` before #1561 changed any source (run through
PrismaBuild; the action key is in that PR). ``test_tessera_serving_code_identity``
requires the output after the change to equal it byte for byte, which is the
"a v2 pin behaves exactly as before" claim made as a before/after comparison
rather than as an argument.  A pin move re-takes it through PrismaBuild and
explains the diff: PQ #1274 (Tessera 38e96012, contract v42) moved only the
pin's commit, contract digest and extension rows, the reviewed answer's two
new extensions and four routed cells' launches, and the four q256 1024 routed
units' recorded launches.
At PQ #2262 the snapshot is re-taken on public 2dbac191/v56: v11 rule
coverage, four extensions and 22 eager cells move the expected answer.
At PQ #2426, PB da8fbb6d706a generates the snapshot from fca4c6ce0/v60.
Only the pin commit and contract digest change. The admission answer remains unchanged.
This is an admission snapshot, never compiled/model/performance qualification.

Run as ``python -m tests.tessera_serving_identity_snapshot`` to print it.
"""
from __future__ import annotations

import copy
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRACE_509 = ROOT / "tests" / "fixtures" / "tessera_route_trace_509"
TRACE_M44E1 = ROOT / "tests" / "fixtures" / "tessera_route_trace_m44e1"
TRACE_509_FILES = ("routes-rank0.json", "routes-rank1.json")
TRACE_M44E1_FILES = ("routes-rank0-sparky.json", "routes-rank1-sparklina.json")

#: The ``formats[]`` grids the trace gate's unit tests stand in with.
_GRIDS = {"TESSERA_E2M1_K2": "E2M1x2", "TESSERA_E4M3_K1": "E4M3",
          "TESSERA_BF16_K1": "BF16"}

#: A digest no tree hashes to, stamped into a copy of each trace header to
#: show what a v2 pin does with a header key it does not read.
FOREIGN_DIGEST = "ab" * 32


def _pin_answer() -> dict:
    from prismaquant import tessera_serving_runtime_pin as pin_module

    pin = pin_module.load_tessera_serving_runtime_pin()
    answer = {
        "schema": pin.schema,
        "repository": pin.repository,
        "commit": pin.commit,
        "version": pin.version,
        "version_is_release": pin.version_is_release,
        "contract_sha256": pin.contract_sha256,
        "runtime_contract_schema": pin.runtime_contract_schema,
        "plugin_entry_point": pin.plugin_entry_point,
        "serving_residency_env": pin.serving_residency_env,
        "native_extension_rows": pin.native_extension_rows(),
    }
    try:
        pin_module.require_pinned_tessera_runtime()
        answer["live_gate"] = "admits"
    except pin_module.TesseraServingRuntimePinError as exc:
        answer["live_gate"] = f"refuses: {exc}"
    return answer


def _unit_routes(table) -> list[dict]:
    from prismaquant import lane_eligibility as lane

    routes = []
    for cell in table.cells:
        rungs = cell.rungs_q256 if cell.is_trellis else cell.rungs
        for rung in rungs:
            facts = lane.UnitStructuralFacts(
                qname=f"{cell.id}@{rung}",
                format_name=f"{cell.family}@{rung}",
                payload_family=cell.family,
                k=None if cell.is_trellis else rung,
                n_sub=None,
                structure=cell.structure,
                role_split=False,
                in_features=4096,
                out_features=4096,
                rate_q256=rung if cell.is_trellis else None,
            )
            for residency in cell.residency_modes or ("resident",):
                for mode in cell.execution_modes or ("eager",):
                    route = lane.resolve_unit_route(
                        facts, table, platform=cell.platform,
                        residency=residency,
                        runtime_image=cell.runtime_image or None,
                        execution_mode=mode if cell.runtime_image else None)
                    routes.append({
                        "cell": cell.id, "rung": rung, "residency": residency,
                        "execution_mode": mode, "route": route.as_dict(),
                    })
    return routes


def _dev_answer() -> dict:
    from prismaquant import lane_eligibility as lane
    from prismaquant import tessera_runtime_contract as trc

    previous = os.environ.get(trc.TESSERA_DEV_PIN_ENV)
    os.environ[trc.TESSERA_DEV_PIN_ENV] = trc.TESSERA_DEV_PIN_COMMIT
    try:
        contract = trc.load_tessera_contract()
    finally:
        if previous is None:
            os.environ.pop(trc.TESSERA_DEV_PIN_ENV, None)
        else:
            os.environ[trc.TESSERA_DEV_PIN_ENV] = previous
    admitted = []
    for cell in contract.cells:
        for rate in sorted(cell.rungs_q256):
            for residency in cell.residency_modes:
                for mode in cell.execution_modes:
                    context = lane.ServingContext(
                        platform=cell.platform, structure=cell.structure,
                        residency=residency, runtime_image=cell.runtime_image,
                        execution_mode=mode)
                    admitted.append({
                        "cell": cell.cell_id, "rate_q256": rate,
                        "residency": residency, "execution_mode": mode,
                        "native_cells": [
                            c.cell_id for c in contract.native_cells(
                                cell.family, rate, serving_context=context)],
                    })
    return {"answer": trc.contract_answer(contract), "native_cells": admitted}


def _read(directory: pathlib.Path, names) -> list:
    return [(f"rank{index}", json.loads((directory / name).read_text()))
            for index, name in enumerate(names)]


def _stamped(traces) -> list:
    out = []
    for label, payload in traces:
        payload = copy.deepcopy(payload)
        payload["serving_source_sha256"] = FOREIGN_DIGEST
        out.append((label, payload))
    return out


def _trace_verdicts() -> dict:
    from prismaquant import tessera_route_trace_gate as gate
    from prismaquant.lane_spec import load_lane_spec

    spec = load_lane_spec("tessera")
    stand_in = (
        {platform: dict(entry) for platform, entry in
         spec.served_activation_quantization.executes_by_platform.items()},
        {family: {"family": family, "grid": grid}
         for family, grid in _GRIDS.items()},
    )
    real = gate.load_trace_contract()
    config = json.loads((TRACE_M44E1 / "config.json").read_text())
    cases = {
        "exact_509": _read(TRACE_509, TRACE_509_FILES),
        "exact_509_stamped": _stamped(_read(TRACE_509, TRACE_509_FILES)),
        "legacy_m44e1": _read(TRACE_M44E1, TRACE_M44E1_FILES),
        "legacy_m44e1_stamped": _stamped(_read(TRACE_M44E1, TRACE_M44E1_FILES)),
    }
    verdicts = {}
    for contract_name, (executes, formats) in (("stand_in", stand_in),
                                               ("packaged", real)):
        for name, traces in cases.items():
            verdicts[f"{contract_name}/{name}"] = gate.compare_route_traces(
                traces, expected_ranks=2, config=config, platform="sm_121",
                executes_by_platform=executes, formats=formats)
    return verdicts


def snapshot() -> dict:
    from prismaquant import tessera_render

    table, _formats = tessera_render._pinned_serving_table()
    return {
        "pin": _pin_answer(),
        "cells": [cell.as_dict() for cell in table.cells],
        "unit_routes": _unit_routes(table),
        "development_contract": _dev_answer(),
        "route_trace_verdicts": _trace_verdicts(),
    }


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, indent=1) + "\n"


if __name__ == "__main__":
    sys.stdout.write(canonical(snapshot()))
