"""Whole-bit quality panel and current whole/half menu planning, never encoding.

No speed is borrowed from the old per-unit timing sums. Every candidate
needs a current measured D41 family/cell/rate row. Half-bit quality requires
both own-family whole-bit anchors; historical R832/R1088 renders are only
local/chord diagnostics. Missing encodeability and native evidence stays
missing. T4 q896 is diagnostic regardless of missing T4 menu admission.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def build(rows, backing=None):
    by_layer = {}
    for row in rows:
        by_layer.setdefault(row["layer"],[]).append(row)
    panels, coverage = [],[]
    for layer,layer_rows in sorted(by_layer.items()):
        routed = [r for r in layer_rows if r["kind"] == "routed"]
        experts = sorted({r["expert"] for r in routed})[:2]
        for row in sorted(layer_rows,key=lambda r:r["qname"]):
            if row["kind"] == "routed" and row["expert"] not in experts:
                continue
            whole16 = list(range(256,2049 if row["kind"] == "routed" else 3585,256))
            half16 = [q+128 for q in whole16[:-1]]
            options = row.get("formats",{})
            actual_rungs = sorted({int(fmt.rsplit("_R",1)[1]) for fmt in options
                                   if "E4M3" in fmt and "_R" in fmt})
            panels.append({"qname":row["qname"],"layer":layer,"kind":row["kind"],"role":row["role"],
                           "expert":row["expert"],"whole_bit_panel_requests":{
                               "T8":[512,768,1024,1280],"T16":whole16,
                               "T4":[768,1024]},
                           "existing_primary_anchors":{"T8":1024,"T4":896},
                           "own_family_only":True,
                           "unbacked_T8_quality_endpoints":[q for q in (512,1280) if q not in actual_rungs],
                           "encodeability_status":"R512/R1280 are quality-only requirements, not known encodeable; inspect renderer domain on admitted CPU before any render",
                           "T16_encode_status":"not requested; parent release only after prospective PASS"})
            coverage.append({"qname":row["qname"],"requested_D41_menu":{
                "T8":{"min_q256":640,"max_q256":1152},"T16":{"min_q256":whole16[0],"max_q256":whole16[-1]},"T4":"Every measured body/rate row in the current D41 table"},
                "menu_admission":"requires current D41 measured family/cell/rate row; this plan admits nothing",
                "quality_unbacked":"Every fractional rung needs both actual measured whole-bit anchors",
                "timing_rows": [r for r in (backing or []) if r.get("qname") == row["qname"]]})
    return {"schema":"pact.panel_pricing_plan.v2","panel":panels,"candidate_coverage":coverage,
        "panel_expert_policy":"Use two actual experts per routed layer and role. Include every nonrouted unit.",
        "quality_rule":"Every allowable fractional rung uses its whole-bit energy chord. The plan admits no unmeasured time row.",
        "T4":"q896 canonical A4 energy diagnostic; not admitted until measured native speed is supplied",
        "T16":"Cover the full measured range: routed R256..R2048 and nonrouted R256..R3584. Keep unbacked rates unpriced.",
        "outside_fused_range":"generic fallback is not a performant candidate",
        "cross_family_ratios":"own measured family anchors required; no level/tilt transfer",
        "historical_R832_R1088":"local/chord validation only; sum by layer and flag >2% chord miss, never fit it away",
        "half_speed":"actual D41 family/cell/rate measured cost, never an old per-unit sum or a neutral/free activation assumption",
        "new_fractional_encodes":0,"T16_encodes":0,"allowability_admitted":False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--measured-backing",type=Path)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    rows=json.loads(args.manifest.read_text())["rows"]
    backing=json.loads(args.measured_backing.read_text())["rows"] if args.measured_backing else None
    result=build(rows,backing)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
