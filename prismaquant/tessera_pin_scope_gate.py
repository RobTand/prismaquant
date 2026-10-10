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
  and a v11-shaped answer keeps the v11 row shape.
* ``malformed``: a malformed launch scope is refused, and a v12 key under
  v11 is refused.

``require_v12_scope_proof`` refuses when any named class is missing or
fails, so a dropped fixture class refuses exactly as a broken reader does.
``check_producer_schema_pr`` is the producer-side rule: a Tessera schema PR
is refused unless it links consumer compatibility work and names fixture
results for all four classes. The check is mechanical string matching, not
a heuristic.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Mapping, Sequence

from . import lane_eligibility as lane
from . import tessera_runtime_contract as runtime

__all__ = [
    "CLASS_DECODER",
    "HISTORICAL",
    "HISTORICAL_RUNGS",
    "ProducerSchemaPRError",
    "REQUIRED_SCOPE_CLASSES",
    "TesseraPinScopeGateError",
    "V12_FIXTURE",
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
    assert payload["lane_eligibility"]["schema"] == "tessera.lane-eligibility.v12"
    assert payload["contract_version"] == 66
    cells = payload["lane_eligibility"]["cells"]
    assert {c["id"] for c in cells} == {
        "tessera_e4m3_k1_routed_moe_sm121_decode_resident",
        "tessera_e4m3_k1_routed_moe_sm121_batch_resident",
    }
    for cell in cells:
        assert cell["rungs_q256"] == [768] + list(HISTORICAL_RUNGS)
        scopes = {e["decoder"]: e.get("rungs_q256") for e in cell["executes"]}
        assert scopes[CLASS_DECODER] == [768]
        for decoder in HISTORICAL:
            assert scopes[decoder] == list(HISTORICAL_RUNGS)
    table = _table(payload)
    assert table.schema == "tessera.lane-eligibility.v12"
    for cell in table.cells:
        scopes = dict(zip([d for _, d in cell.executes], cell.launch_rungs_q256))
        assert scopes[CLASS_DECODER] == (768,)
        for decoder in HISTORICAL:
            assert scopes[decoder] == HISTORICAL_RUNGS


def _check_derived() -> None:
    parsed = _parsed()
    table = _table()
    route = next(c for c in parsed.cells if c.cell_id == _CELL_ID)
    cell = next(c for c in table.cells if c.id == route.cell_id)
    assert tuple(route.launch_covered_rungs_q256) == tuple(
        cell.launch_covered_rungs_q256
    )
    index = next(
        i for i, (_, decoder) in enumerate(route.executes) if decoder in HISTORICAL
    )
    assert 800 in route.launch_covered_rungs_q256[index]
    assert route.launch_covers_rate(index, 800)
    assert route.launch_covers_rate(index, 832)
    class_index = next(
        i for i, (_, decoder) in enumerate(route.executes)
        if decoder == CLASS_DECODER
    )
    assert not route.launch_covers_rate(class_index, 800)


def _check_absent() -> None:
    payload = _payload()
    cell = next(
        c for c in payload["lane_eligibility"]["cells"] if c["id"] == _CELL_ID
    )
    del cell["executes"][0]["rungs_q256"]
    table = _table(payload)
    parsed_cell = next(c for c in table.cells if c.id == cell["id"])
    assert parsed_cell.launch_rungs_q256[0] is None
    assert parsed_cell.launch_covers_rate(0, 768)
    assert parsed_cell.launch_covers_rate(0, 832)
    v11 = _payload()
    v11["lane_eligibility"]["schema"] = "tessera.lane-eligibility.v11"
    for v11_cell in v11["lane_eligibility"]["cells"]:
        for launch in v11_cell["executes"]:
            launch.pop("rungs_q256", None)
    before = runtime.contract_answer(_parsed())
    answer = runtime.contract_answer(
        runtime._parse(v11, commit="fixture", sha="fixture", path="fixture")
    )
    assert runtime._answer_drift(before, before) == []
    row = next(r for r in answer["cells"] if r[0] == _CELL_ID)
    assert "launch_scopes" not in row


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
        raise AssertionError(f"malformed scope admitted: {mutation!r}")
    payload = _payload()
    payload["lane_eligibility"]["schema"] = "tessera.lane-eligibility.v11"
    try:
        _table(payload)
    except lane.LaneEligibilityError:
        return
    raise AssertionError("v12 key admitted under v11")


_CHECKS = {
    "census": _check_census,
    "derived": _check_derived,
    "absent": _check_absent,
    "malformed": _check_malformed,
}


def run_v12_scope_proof() -> dict[str, bool]:
    """Run the four named v12 scope fixture classes against the reader.

    Returns one pass flag per class. A class that raises records ``False``
    instead of propagating, so the gate refusal names the class.
    """
    results: dict[str, bool] = {}
    for name in REQUIRED_SCOPE_CLASSES:
        try:
            _CHECKS[name]()
        except Exception:
            results[name] = False
        else:
            results[name] = True
    return results


def require_v12_scope_proof(results: Mapping[str, object]) -> None:
    """Refuse a pin update unless all four scope classes pass.

    A missing class refuses exactly as a failing class does: a dropped
    fixture is a dropped proof.
    """
    missing = [name for name in REQUIRED_SCOPE_CLASSES if name not in results]
    if missing:
        raise TesseraPinScopeGateError(
            "pin update refused: scope proof names no result for "
            + ", ".join(missing)
        )
    failing = [name for name in REQUIRED_SCOPE_CLASSES if not results[name]]
    if failing:
        raise TesseraPinScopeGateError(
            "pin update refused: scope fixture class fails: " + ", ".join(failing)
        )


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
