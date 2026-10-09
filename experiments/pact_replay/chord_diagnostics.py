"""Layer-summed chord checks on actual whole-bit anchors and historical renders.

Historical R832 and R1088 are validation only, never serving candidates.
No new fractional render or price is produced. Missing whole-bit neighbours
are reported as unmeasured, never extrapolated or replaced by 3.6x lore.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path


def reduce(rows):
    by_unit = {}
    for row in rows:
        if row.get("family") != "T8" or row.get("passthrough"):
            continue
        q = row.get("q256")
        if type(q) is not int:
            raise ValueError("Measured diagnostic rates need integer q256")
        key = (row["qname"],q)
        existing = by_unit.setdefault(key,{"layer":row["layer"],"samples":set(),
            "E_W_sum":0.0,"E_A_sum":0.0,"E_WA_sum":0.0})
        sid = row["sample_id"]
        if sid in existing["samples"]:
            raise ValueError("Duplicate unit/rate/sample diagnostic")
        existing["samples"].add(sid)
        for field in ("E_W_sum","E_A_sum","E_WA_sum"):
            existing[field] += row[field]
    for measured in by_unit.values():
        expected = set(range(384, 512 if rows[0]["input_contract"] == "raw_512" else 448))
        if measured["samples"] != expected:
            raise ValueError("Partial cohort cannot validate a full-population chord")
    checks = {}
    for (name,q),measured in by_unit.items():
        if q not in (832,1088):
            continue
        lower = (q//256)*256
        upper = lower+256
        key = (measured["layer"],q)
        result = checks.setdefault(key,{"layer":key[0],"q256":q,"lower_q256":lower,
            "upper_q256":upper,"measured_units":0,"missing_anchor_units":[],
            "measured":{k:0.0 for k in ("E_W_sum","E_A_sum","E_WA_sum")},
            "chord":{k:0.0 for k in ("E_W_sum","E_A_sum","E_WA_sum")}})
        lo,hi = by_unit.get((name,lower)),by_unit.get((name,upper))
        if lo is None or hi is None:
            result["missing_anchor_units"].append(name)
            continue
        if lo["samples"] != measured["samples"] or hi["samples"] != measured["samples"]:
            raise ValueError("Fractional diagnostic and whole-bit anchor cohorts differ")
        fraction = (q-lower)/256
        result["measured_units"] += 1
        for field in result["measured"]:
            result["measured"][field] += measured[field]
            result["chord"][field] += lo[field]+fraction*(hi[field]-lo[field])
    for result in checks.values():
        if result["missing_anchor_units"]:
            result["status"] = "unmeasured whole-bit anchors"
            result["passed"] = None
        else:
            errors = {}
            for field,value in result["measured"].items():
                chord = result["chord"][field]
                errors[field] = abs(value-chord)/chord if chord else (0.0 if value == 0 else None)
            result["relative_error"] = errors
            result["passed"] = all(v is not None and v <= .02 for v in errors.values())
            result["status"] = "measured diagnostic within 2%" if result["passed"] else "chord miss; stop/report without adjusting"
    return {"schema":"pact.existing_render_chord_checks.v1","checks":list(checks.values()),
            "candidate_admission":False,"fractional_encodes":0,"fitting_or_rescaling":False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--energies",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    with args.energies.open() as stream:
        rows=[json.loads(line) for line in stream if line.strip()]
    if any(row.get("dry_run_cpu") for row in rows):
        raise ValueError("CPU retained rows cannot validate full population chord checks")
    if len({row["input_contract"] for row in rows}) != 1 or len({row["input_token_sha256"] for row in rows}) != 1:
        raise ValueError("Chord input contracts or actual token distributions differ")
    result=reduce(rows)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
