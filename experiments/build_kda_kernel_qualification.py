"""Build ``kernels/kda_chunk_qualification.json`` from numerics runs (PQ #1199).

The qualification binds the identity fields that admission compares
(``glm_kda_capture_kernel.QUALIFIED_FIELDS``) to the numerics that qualify
them. It takes one ``experiments/kda_kernel_numerics.py`` run at the
production shape, which must pass every cell, and identity runs on other
boxes, whose identity must equal it. Nothing is measured here: every number
comes from the runs' ``numerics.json``, and the runs' PrismaBuild keys are
recorded as given.

Standard library only, so it runs anywhere:

    python3 experiments/build_kda_kernel_qualification.py \\
        --qualify RUN_DIR=PB_KEY@HOST --identity RUN_DIR=PB_KEY@HOST [...] \\
        [--rejected RUN_DIR=PB_KEY@HOST] --out prismaquant/kernels/kda_chunk_qualification.json

``HOST`` is the box the PrismaBuild record says ran the action.

``--rejected`` records a run of an earlier source that missed the bound, so
the file keeps the history of what did not qualify next to what did.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KERNEL_SOURCE = ROOT / "prismaquant/kernels/kda_chunk.py"
HARNESS = ROOT / "experiments/kda_kernel_numerics.py"
NUMERICS_SCHEMA = "prismaquant.kda_kernel_numerics.v1"
SCHEMA = "prismaquant.kda_capture_kernel_qualification.v1"
#: ``glm_kda_capture_kernel.QUALIFIED_FIELDS``; that module needs torch.
QUALIFIED_FIELDS = ("name", "source_sha256", "compiled", "probe", "runtime")
PRODUCTION_SHAPE = {"batch": 4, "seqlen": 512, "heads": 64, "head_dim": 128}
PATHS = ("fp32", "bf16")
TENSORS = ("o", "dq", "dk", "dv", "dg", "dbeta")
CRITERION = (
    "For every case, path (fp32, bf16) and tensor (o, dq, dk, dv, dg, dbeta), against a "
    "float64 run of the executed fallback source: the kernel's largest element error "
    "exceeds the FP32 fallback's by at most 2u max|ref|, and its relative Frobenius error "
    "exceeds the fallback's by at most 2u, u the unit roundoff of the path's dtype "
    "(the two implementations' final roundings); no non-finite value where the fallback "
    "has none; and two kernel runs are bit-equal.")


class QualificationRefused(ValueError):
    """A run that cannot qualify the kernel."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(spec: str):
    directory, _, rest = spec.partition("=")
    key, _, host = rest.partition("@")
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key) or not host:
        raise QualificationRefused(f"{spec!r}: expected RUN_DIR=<64-hex PrismaBuild key>@HOST")
    path = Path(directory) / "numerics.json"
    numerics = json.loads(path.read_bytes())
    if numerics.get("schema") != NUMERICS_SCHEMA:
        raise QualificationRefused(f"{path}: not a {NUMERICS_SCHEMA} record")
    return numerics, {"pb_action": key, "host": host, "artifact": str(path),
                      "artifact_receipt_sha256": _sha256(path)}


def _cells(numerics):
    for case, value in numerics["cases"].items():
        for path in PATHS:
            if path not in value:
                continue
            for tensor in TENSORS:
                yield case, path, tensor, value[path]


def summarize(numerics) -> dict:
    """Counts and worst excesses over every cell, read from the run's verdicts."""
    compare = {metric: {"lower": 0, "equal": 0, "higher": 0} for metric in ("max_abs", "rel_fro")}
    worst = {}
    failures, cells = [], 0
    for case, path, tensor, block in _cells(numerics):
        cells += 1
        verdict = block["verdict"][tensor]
        if not (verdict["pass"] and verdict["finite_where_fallback_finite"]
                and block["kernel_repeat_bit_equal"][tensor]):
            failures.append([case, path, tensor])
        for metric in compare:
            kernel, fallback = block["kernel"][tensor][metric], block["fallback"][tensor][metric]
            compare[metric]["lower" if kernel < fallback else
                            "equal" if kernel == fallback else "higher"] += 1
            excess = verdict[metric]["excess_in_u"]
            if metric not in worst or excess > worst[metric]["excess_in_u"]:
                worst[metric] = {"excess_in_u": excess, "bound_in_u": 2.0,
                                 "cell": [case, path, tensor]}
    return {"cells": cells, "failures": failures, "kernel_vs_fallback": compare,
            "worst_excess": worst}


def qualify(numerics) -> dict:
    """The numerics block of a qualifying run; refuses a run that cannot qualify."""
    if numerics["shape"] != PRODUCTION_SHAPE:
        raise QualificationRefused(f"shape {numerics['shape']} is not the production shape")
    if numerics["harness_sha256"] != _sha256(HARNESS):
        raise QualificationRefused("the run's harness is not this tree's harness")
    missing = [case for case, value in numerics["cases"].items()
               if any(path not in value for path in PATHS)]
    if missing:
        raise QualificationRefused(f"cases without both paths: {missing}")
    summary = summarize(numerics)
    run_summary = numerics["summary"]
    if (summary["failures"] or not run_summary["pass"] or run_summary["failures"]
            or run_summary["repeat_unequal"]):
        raise QualificationRefused(f"the run does not pass: {summary['failures']}")
    return {"shape": numerics["shape"], "cases": list(numerics["cases"]), "paths": list(PATHS),
            "tensors": list(TENSORS), "matmul": numerics["matmul"],
            "oracle": numerics["oracle"], **summary}


def build(qualifying, identities, rejected=()) -> dict:
    numerics, receipt = qualifying
    identity = {key: numerics["qualification_candidate"][key] for key in QUALIFIED_FIELDS}
    if identity["source_sha256"] != _sha256(KERNEL_SOURCE):
        raise QualificationRefused("the run's kernel source is not this tree's kda_chunk.py")
    block = qualify(numerics)
    evidence = [{**receipt, "role": "numerics at the production shape",
                 "gate": CRITERION, "shape": numerics["shape"]}]
    for other, other_receipt in identities:
        candidate = {key: other["qualification_candidate"][key] for key in QUALIFIED_FIELDS}
        if candidate != identity:
            changed = sorted(key for key in QUALIFIED_FIELDS if candidate[key] != identity[key])
            raise QualificationRefused(f"{other_receipt['artifact']}: identity differs in {changed}")
        if not other["summary"]["pass"]:
            raise QualificationRefused(f"{other_receipt['artifact']}: the run does not pass")
        evidence.append({**other_receipt, "role": "identity, equal",
                         "shape": other["shape"], "cases": list(other["cases"])})
    history = []
    for other, other_receipt in rejected:
        summary = summarize(other)
        if not summary["failures"]:
            raise QualificationRefused(f"{other_receipt['artifact']}: passes; not a rejected run")
        history.append({**other_receipt, "source_sha256": other["kernel"]["source_sha256"],
                        "shape": other["shape"], **summary})
    return {"schema": SCHEMA, "status": "qualified", "identity": identity,
            "implements": "glm_kda_causal_exp_v1",
            "harness": {"path": str(HARNESS.relative_to(ROOT)), "sha256": _sha256(HARNESS)},
            "criterion": CRITERION, "numerics": block, "evidence": evidence,
            "rejected_attempts": history}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--qualify", required=True)
    parser.add_argument("--identity", action="append", default=[])
    parser.add_argument("--rejected", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    value = build(_run(args.qualify), [_run(spec) for spec in args.identity],
                  [_run(spec) for spec in args.rejected])
    Path(args.out).write_text(json.dumps(value, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"out": args.out, "cells": value["numerics"]["cells"],
                      "worst_excess": value["numerics"]["worst_excess"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
