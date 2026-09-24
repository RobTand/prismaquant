"""A Stage B row's spill ceiling is the sealed need, not a spec literal.

PB action ``6f4f058751e6`` (layer 44 of GLM-5.3-Flash, regime (d), sparklina,
2026-09-24) refused 60 s into its head phase:

    Stage B spill needs 199051640832 bytes for this layer (196494753792 of
    payload and slot padding for 554880 parts at 4096-byte direct-I/O
    alignment) but its ceiling is 196494753792

The ceiling was the spec's hand-set ``PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES``
(183 GiB), and pbrun charged the same literal to the box's ``spool_gb``. The
payload alone reached it; the direct-I/O slot padding that #1060 added to the
file's reservation was never in it. The need is known before dispatch: it is
``spill_geometry``'s payload plus ``max_parts`` slots of padding on the
spill's direct-I/O grid, the arithmetic ``StageBSpillScratch`` reserves.

The record builder seals that need beside the read plan, and the dispatcher
sets the row's ceiling from it. PrismaBuild's ``spool_gb`` charge is that
ceiling rounded up to whole GiB, so the demand and the ceiling are one
number, and both cover the need.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _entry in (ROOT, ROOT / "tests", ROOT / "tools"):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from prismaquant import joint_layer_quanta as jl  # noqa: E402
from test_stageb_prepared_render_inputs import _dispatch_prepared  # noqa: E402

GIB = 1 << 30
SPILL_ROOT = "/home/rob/pb-scratch/stage-b-spill"
SPILL_ENV = ("PRISMAQUANT_STAGE_B_SPILL_ROOT", "PRISMAQUANT_STAGE_B_SPILL_MAX_BYTES")

# The refused layer-44 quantum (PB 6f4f058751e6): 867 targets, 4 probes and
# 128 capture groups (512 samples of one row, captured four at a time).
PAYLOAD = 196_494_753_792
PARTS = (4 + 1) * 867 * 128
BLOCK = 4096
ADDRESS_ALIGNMENT = 512
# What the file reserves: the payload, then each part's slot padding (fewer
# than 512 bytes of residue ahead of it, and the tail to the next grid
# boundary), rounded up to the grid.
NEED = PAYLOAD + PARTS * (ADDRESS_ALIGNMENT + BLOCK)
NEED += -NEED % BLOCK


def _spill_spec(ceiling: int) -> dict:
    return {"container": {"image": "sha256:" + "0" * 64,
                          "mounts": [{"source": SPILL_ROOT, "target": SPILL_ROOT,
                                      "readonly": False}]},
            "env": {SPILL_ENV[0]: SPILL_ROOT, SPILL_ENV[1]: str(ceiling)}}


def _reseal(record: dict) -> dict:
    body = {key: value for key, value in record.items() if key != "identity_sha256"}
    record["identity_sha256"] = jl.canonical_sha256(
        body, where=f"quantum record {record['quantum_id']}")
    return record


def _sealed_bound() -> dict:
    """The executable readset's spill block for the refused shape."""
    return {"schema": "prismaquant.stage_b_spill_bound.v1",
            "block": BLOCK, "capture_batch": 4, "element_dtype": "bfloat16",
            "geometry": {"n_probes": 4, "total_bytes": PAYLOAD, "max_parts": PARTS},
            "reservation_bytes": NEED}


def _row_ceiling(tmp_path, monkeypatch):
    """Dispatch the spill row under the 183 GiB spec; return the sealed ceilings."""
    (dispatch, bound, _receipt, _manifest, records, receipt_path,
     out) = _dispatch_prepared(tmp_path, monkeypatch, replay_mode="spill")
    dispatch.SPEC_PATH.write_text(json.dumps(_spill_spec(183 * GIB)))
    record = json.loads(json.dumps(bound))
    record["executable_readset"]["spill_bound"] = _sealed_bound()
    path = records / "layer-002.json"
    path.write_text(json.dumps(_reseal(record)))
    argv = dispatch.quantum_argv(record, record_path=path, output_root=out)
    outer = argv[:argv.index("--")]
    sealed = dict(outer[i + 1].split("=", 1) for i, word in enumerate(outer[:-1])
                  if word == "--env" and "=" in outer[i + 1])
    inner = json.loads(argv[argv.index("--spec") + 1])["env"]
    return int(sealed[SPILL_ENV[1]]), int(inner[SPILL_ENV[1]])


def test_the_refused_shape_needed_more_than_its_ceiling():
    """The arithmetic of the refusal message, reproduced."""
    assert PARTS == 554_880
    assert NEED == 199_051_640_832
    assert PAYLOAD == 183 * GIB < NEED


def test_the_row_ceiling_and_pb_demand_cover_the_sealed_need(tmp_path, monkeypatch):
    outer, inner = _row_ceiling(tmp_path, monkeypatch)
    # The container reads the same ceiling pbrun charges.
    assert inner == outer
    # PB #911 charges the pair's ceiling rounded up to whole GiB.
    demand_gib = math.ceil(outer / GIB)
    assert outer >= NEED, (
        f"the row seals a {outer}-byte spill ceiling for a {NEED}-byte need")
    assert demand_gib * GIB >= NEED
