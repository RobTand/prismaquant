"""Pin-update gate on Tessera v12 per-launch scope proof (PQ #2600).

Tessera v12 added per-launch scope the v11 reader could neither parse nor
preserve. The reader now parses it and the pin stays at v11, but no gate
protected future pins. This module is that gate.

A pin update to ``prismaquant/tessera_runtime/tessera_serving_runtime_pin.json``
is refused unless the four named v12 scope fixture classes -- census, derived,
absent and malformed -- all pass against the reader. The checks reuse the
fixture in ``tests/test_tessera_lane_v12.py``, not a new copy:

* ``census``: the fixture is the authoritative v66 pair and the reader keeps
  each launch scope on the record.
* ``derived``: the route cell keeps the derived launch coverage.
* ``absent``: a launch without the scope key keeps the scope of its cell,
  and a v11-shaped answer keeps the v11 row shape (18 entries, the
  image entry at index 13, no entry carries launch scopes).
* ``malformed``: a malformed launch scope is refused, and a v12 key under
  v11 is refused.

``check_pin_update`` is the wired entry point: it reads the tracked pin
file (a malformed pin is refused by its own reader) and then runs the
four classes and refuses when any is missing or fails. ``run_v12_scope_proof``
keeps the failure reason per class, so the refusal names the cause.
``check_producer_schema_pr`` is the producer-side rule: a Tessera schema PR
is refused unless it links consumer compatibility work and names fixture
results for all four classes. The check is mechanical string matching, not
a heuristic.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from . import lane_eligibility as lane
from . import tessera_runtime_contract as runtime
from . import tessera_serving_runtime_pin as pin_module

__all__ = [
    "CLASS_DECODER",
    "HISTORICAL",
    "HISTORICAL_RUNGS",
    "ProducerSchemaPRError",
    "REQUIRED_SCOPE_CLASSES",
    "ScopeClassProof",
    "TesseraPinScopeGateError",
    "V12_FIXTURE",
    "check_pin_update",
    "check_producer_schema_pr",
    "require_v12_scope_proof",
    "run_v12_scope_proof",
]

#: The four named v12 scope fixture classes. A pin update passes only when
#: every one of them passes against the reader.
REQUIRED_SCOPE_CLASSES = ("census", "derived", "absent", "malformed")

V12_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "fixtures"
    / "tessera_lane_v12_v66_routed.json"
)

CLASS_DECODER = "native_routed_window_classes_e4m3mma"
HISTORICAL = (
    "native_window_moe_compact",
    "native_routed_fused_window",
    "native_routed_fused_window_e4m3mma",
)
HISTORICAL_RUNGS = (832, 864, 896, 928, 944, 960, 1024, 1088)

_CELL_ID = "tessera_e4m3_k1_routed_moe_sm121_decode_resident"


class TesseraPinScopeGateError(ValueError):
    """A pin update without a passing v12 scope proof."""


class ProducerSchemaPRError(ValueError):
    """A producer schema PR without consumer work or fixture results."""


@dataclass(frozen=True)
class ScopeClassProof:
    """One fixture class outcome: pass flag plus failure reason."""

    passed: bool
    reason: str = ""


def _fail(message: str) -> None:
    """Refuse the fixture class with an explicit reason (never ``assert``)."""
    raise TesseraPinScopeGateError(message)


def _payload() -> dict:
    return json.loads(V12_FIXTURE.read_bytes())


def _table(payload=None):
    if payload is None:
        payload = _payload()
    return lane._parse_table(
        payload["lane_eligibility"],
        payload["formats"],
        "fixture",
        "fixture",
        "fixture",
        native_extensions=payload["native_extensions"],
    )


def _parsed(payload=None):
    if payload is None:
        payload = _payload()
    return runtime._parse(payload, commit="fixture", sha="fixture", path="fixture")


def _check_census() -> None:
    payload = _payload()
    if payload["lane_eligibility"]["schema"] != "tessera.lane-eligibility.v12":
        _fail("census: lane schema is not v12")
    if payload["contract_version"] != 66:
        _fail("census: contract version is not 66")
    cells = payload["lane_eligibility"]["cells"]
    if {c["id"] for c in cells} != {
        "tessera_e4m3_k1_routed_moe_sm121_decode_resident",
        "tessera_e4m3_k1_routed_moe_sm121_batch_resident",
    }:
        _fail("census: fixture cell pair is not the v66 pair")
    for cell in cells:
        if cell["rungs_q256"] != [768] + list(HISTORICAL_RUNGS):
            _fail(f"census: cell rung census moved for {cell['id']}")
        scopes = {e["decoder"]: e.get("rungs_q256") for e in cell["executes"]}
        if scopes[CLASS_DECODER] != [768]:
            _fail("census: class decoder scope is not [768]")
        for decoder in HISTORICAL:
            if scopes[decoder] != list(HISTORICAL_RUNGS):
                _fail(f"census: historical scope moved for {decoder}")
    table = _table(payload)
    if table.schema != "tessera.lane-eligibility.v12":
        _fail("census: parsed table schema is not v12")
    for cell in table.cells:
        scopes = dict(zip([d for _, d in cell.executes], cell.launch_rungs_q256))
        if scopes[CLASS_DECODER] != (768,):
            _fail("census: reader drops the class decoder scope")
        for decoder in HISTORICAL:
            if scopes[decoder] != HISTORICAL_RUNGS:
                _fail(f"census: reader drops a historical scope for {decoder}")


def _check_derived() -> None:
    parsed = _parsed()
    table = _table()
    route = next(c for c in parsed.cells if c.cell_id == _CELL_ID)
    cell = next(c for c in table.cells if c.id == route.cell_id)
    if tuple(route.launch_covered_rungs_q256) != tuple(cell.launch_covered_rungs_q256):
        _fail("derived: route cell loses the derived launch coverage")
    index = next(
        i for i, (_, decoder) in enumerate(route.executes) if decoder in HISTORICAL
    )
    if 800 not in route.launch_covered_rungs_q256[index]:
        _fail("derived: covered rungs miss the derived rate 800")
    if not route.launch_covers_rate(index, 800):
        _fail("derived: launch_covers_rate misses the derived rate 800")
    if not route.launch_covers_rate(index, 832):
        _fail("derived: launch_covers_rate misses the census rate 832")
    class_index = next(
        i for i, (_, decoder) in enumerate(route.executes)
        if decoder == CLASS_DECODER
    )
    if route.launch_covers_rate(class_index, 800):
        _fail("derived: class launch wrongly covers rate 800")


def _check_absent() -> None:
    payload = _payload()
    cell = next(
        c for c in payload["lane_eligibility"]["cells"] if c["id"] == _CELL_ID
    )
    del cell["executes"][0]["rungs_q256"]
    table = _table(payload)
    parsed_cell = next(c for c in table.cells if c.id == cell["id"])
    if parsed_cell.launch_rungs_q256[0] is not None:
        _fail("absent: launch without scope key keeps no cell scope")
    if not parsed_cell.launch_covers_rate(0, 768):
        _fail("absent: scopeless launch misses its cell rate 768")
    if not parsed_cell.launch_covers_rate(0, 832):
        _fail("absent: scopeless launch misses its cell rate 832")
    v11 = _payload()
    v11["lane_eligibility"]["schema"] = "tessera.lane-eligibility.v11"
    for v11_cell in v11["lane_eligibility"]["cells"]:
        for launch in v11_cell["executes"]:
            launch.pop("rungs_q256", None)
    before = runtime.contract_answer(_parsed())
    before_row = next(r for r in before["cells"] if r[0] == _CELL_ID)
    if not any(
        isinstance(entry, dict) and "launch_scopes" in entry
        for entry in before_row
    ):
        _fail("absent: v12 answer row carries no launch scopes")
    answer = runtime.contract_answer(
        runtime._parse(v11, commit="fixture", sha="fixture", path="fixture")
    )
    if runtime._answer_drift(before, before) != []:
        _fail("absent: answer drift baseline is not empty")
    row = next(r for r in answer["cells"] if r[0] == _CELL_ID)
    if len(row) != 18:
        _fail(f"absent: v11 answer row shape moved: len {len(row)}")
    if not (
        isinstance(row[13], dict)
        and set(row[13]) >= {"image", "execution_modes"}
    ):
        _fail("absent: v11 answer row entry 13 is not the image entry")
    if any(
        isinstance(entry, dict) and "launch_scopes" in entry for entry in row
    ):
        _fail("absent: v11 answer row carries launch scopes")


def _check_malformed() -> None:
    bad: list[dict] = [
        {"rungs_q256": []},
        {"rungs_q256": [832, 768]},
        {"rungs_q256": [832, 832]},
        {"rungs_q256": [True]},
        {"rungs_q256": [[832]]},
        {"rungs_q256": [640]},
        {"rungs_q256": "832"},
    ]
    for mutation in bad:
        payload = _payload()
        cell = next(
            c for c in payload["lane_eligibility"]["cells"] if c["id"] == _CELL_ID
        )
        cell["executes"][0].update(copy.deepcopy(mutation))
        try:
            _table(payload)
        except lane.LaneEligibilityError:
            continue
        _fail(f"malformed scope admitted: {mutation!r}")
    payload = _payload()
    payload["lane_eligibility"]["schema"] = "tessera.lane-eligibility.v11"
    try:
        _table(payload)
    except lane.LaneEligibilityError:
        return
    _fail("v12 key admitted under v11")


_CHECKS = {
    "census": _check_census,
    "derived": _check_derived,
    "absent": _check_absent,
    "malformed": _check_malformed,
}


def run_v12_scope_proof() -> dict[str, ScopeClassProof]:
    """Run the four named v12 scope fixture classes against the reader.

    Returns one outcome per class. A class that raises records its
    failure reason instead of propagating, so the gate refusal names
    the cause.
    """
    results: dict[str, ScopeClassProof] = {}
    for name in REQUIRED_SCOPE_CLASSES:
        try:
            _CHECKS[name]()
        except Exception as exc:
            results[name] = ScopeClassProof(
                passed=False, reason=f"{type(exc).__name__}: {exc}"
            )
        else:
            results[name] = ScopeClassProof(passed=True)
    return results


def require_v12_scope_proof(results: Mapping[str, ScopeClassProof]) -> None:
    """Refuse a pin update unless all four scope classes pass.

    A missing class refuses exactly as a failing class does: a dropped
    fixture is a dropped proof. The refusal carries each failure reason.
    """
    missing = [name for name in REQUIRED_SCOPE_CLASSES if name not in results]
    if missing:
        raise TesseraPinScopeGateError(
            "pin update refused: scope proof names no result for "
            + ", ".join(missing)
        )
    failing = [name for name in REQUIRED_SCOPE_CLASSES if not results[name].passed]
    if failing:
        detail = "; ".join(
            f"{name} ({results[name].reason})" for name in failing
        )
        raise TesseraPinScopeGateError(
            "pin update refused: scope fixture class fails: " + detail
        )


def check_pin_update(pin_path: Path | str | None = None):
    """Refuse a pin update unless the pin reads and the scope proof passes.

    Reads the pin file at ``pin_path`` (the tracked
    ``tessera_serving_runtime_pin.json`` when omitted), so a malformed
    pin is refused by its own reader, then runs the four named v12
    scope fixture classes against the reader. Returns the loaded pin
    when the update is admitted.
    """
    pin = pin_module.load_tessera_serving_runtime_pin(pin_path)
    require_v12_scope_proof(run_v12_scope_proof())
    return pin


def check_producer_schema_pr(body: str, linked: Sequence[str]) -> None:
    """Refuse a producer schema PR without consumer work or fixture results.

    ``linked`` holds the linked consumer compatibility issues or PRs. The
    PR ``body`` must name fixture results for all four scope classes. Both
    checks are literal string matches.
    """
    if not linked:
        raise ProducerSchemaPRError(
            "producer schema PR refused: no linked consumer compatibility work"
        )
    lowered = body.lower()
    if "fixture" not in lowered:
        raise ProducerSchemaPRError(
            "producer schema PR refused: body names no fixture results"
        )
    unnamed = [name for name in REQUIRED_SCOPE_CLASSES if name not in lowered]
    if unnamed:
        raise ProducerSchemaPRError(
            "producer schema PR refused: body names no fixture results for "
            + ", ".join(unnamed)
        )
