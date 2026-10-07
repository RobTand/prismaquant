"""Quality measurements and explicit criteria share one receipt contract."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from .schemas import strict_json_loads
from .digests import DIRECT_UTF8_INDENT2_STRICT, bytes_sha256hex


SCHEMA = "prismaquant.quality_stage/1"
STAGES = {"g3_v2": ("prismaquant.g3_v2/1", "offline_decoded_kl"),
          "task_suite": ("prismaquant.task_suite/1", "task_metrics")}


def finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def evaluate_criteria(metrics, criteria):
    """Missing criteria permit measurement, but never give a gate pass."""
    if not isinstance(metrics, dict):
        raise ValueError("metrics must be an object")
    for name, value in metrics.items():
        finite_number(value, name)
    if criteria is None or criteria == []:
        return {"status": "not_evaluated", "criteria": [], "failures": []}
    if not isinstance(criteria, list):
        raise ValueError("criteria must be an array")
    failures, seen = [], set()
    for row in criteria:
        if not isinstance(row, dict) or set(row) != {"metric", "op", "threshold"}:
            raise ValueError("each criterion requires metric, op and threshold")
        name, op = row["metric"], row["op"]
        if not isinstance(name, str) or not name or op not in ("le", "ge"):
            raise ValueError("criterion metric or operator is invalid")
        if (name, op) in seen:
            raise ValueError("duplicate criterion")
        seen.add((name, op))
        threshold = finite_number(row["threshold"], "criterion threshold")
        if name not in metrics:
            raise ValueError(f"criterion metric is missing: {name}")
        value = finite_number(metrics[name], name)
        if not (value <= threshold if op == "le" else value >= threshold):
            failures.append({"metric": name, "value": value, "op": op, "threshold": threshold})
    return {"status": "failed" if failures else "passed", "criteria": criteria, "failures": failures}


def verify_result(result, config, *, config_sha256):
    """Replay the decision. Never trust a producer's pass flag alone."""
    if not isinstance(result, dict) or result.get("schema") != SCHEMA:
        raise ValueError("unsupported quality result schema")
    stage = result.get("stage")
    if stage not in STAGES:
        raise ValueError("unknown quality stage")
    schema, kind = STAGES[stage]
    binding = result.get("configuration", {})
    if (config.get("schema") != schema or binding.get("schema") != schema
            or binding.get("sha256") != config_sha256):
        raise ValueError("quality configuration binding differs")
    measurement = result.get("measurement", {})
    if measurement.get("metric_kind") != kind:
        raise ValueError("quality metric kind differs")
    status = measurement.get("status")
    if status not in ("succeeded", "failed", "preflight"):
        raise ValueError("unknown measurement status")
    if status == "succeeded":
        if measurement.get("error") is not None or not measurement.get("metrics"):
            raise ValueError("successful measurement has an error or no metrics")
        expected = evaluate_criteria(measurement["metrics"], config.get("criteria"))
    else:
        expected = {"status": "not_evaluated", "criteria": [], "failures": []}
    if result.get("gate") != expected:
        raise ValueError("quality gate does not match the configured decision")
    for field, cls in (("identity", dict), ("population", dict), ("artifacts", list), ("limitations", list)):
        if not isinstance(result.get(field), cls):
            raise ValueError(f"quality result lacks {field}")
    return expected


def read_binding(binding, label):
    from .stage_inputs import read_bound
    if not isinstance(binding, dict) or not {"path", "sha256"} <= binding.keys():
        raise ValueError(f"{label} requires path and sha256")
    return read_bound(binding, label)


def artifact(path):
    from .digests import file_sha256hex
    return {"path": str(Path(path).resolve()), "sha256": file_sha256hex(path)}


def write_result(path, result):
    from .cost_stage_checkpoint import publish_new_bytes
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not publish_new_bytes(path, DIRECT_UTF8_INDENT2_STRICT.encoded(result) + b"\n"):
        raise ValueError(f"quality output already exists: {path}")


def cli(stage, measure, preflight, argv=None):
    parser = argparse.ArgumentParser(description=f"Run the {stage} quality stage.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    schema, kind = STAGES[stage]
    result = {"schema": SCHEMA, "stage": stage,
        "configuration": {"schema": schema, "path": str(args.config.resolve()), "sha256": None},
        "measurement": {"status": "failed", "metric_kind": kind, "metrics": {}, "error": None},
        "gate": {"status": "not_evaluated", "criteria": [], "failures": []},
        "identity": {}, "population": {}, "artifacts": [], "limitations": []}
    try:
        raw = args.config.read_bytes()
        result["configuration"]["sha256"] = bytes_sha256hex(raw)
        config = strict_json_loads(raw, duplicate=lambda key: ValueError("duplicate configuration key: "+key),
                                  constant=lambda value: ValueError("nonfinite configuration value: "+value))
        if not isinstance(config, dict) or config.get("schema") != schema:
            raise ValueError(f"configuration requires schema {schema}")
        facts = preflight(config) if args.preflight else measure(config, args.output)
        result.update(facts)
        result["measurement"] = {"status": "preflight" if args.preflight else "succeeded",
                                 "metric_kind": kind, "metrics": facts.get("metrics", {}), "error": None}
        result.pop("metrics", None)
        if not args.preflight:
            result["gate"] = evaluate_criteria(result["measurement"]["metrics"], config.get("criteria"))
        verify_result(result, config, config_sha256=result["configuration"]["sha256"])
    except Exception as error:
        result["measurement"] = {"status": "failed", "metric_kind": kind, "metrics": {},
                                 "error": f"{type(error).__name__}: {error}"}
        result["gate"] = {"status": "not_evaluated", "criteria": [], "failures": []}
    write_result(args.output, result)
    print(DIRECT_UTF8_INDENT2_STRICT.text(result))
    return int(result["measurement"]["status"] == "failed" or result["gate"]["status"] == "failed")
