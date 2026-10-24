"""Independent original expressions frozen from 8bc0b5af (see ledger_plans.json).

No production or digest-owner imports: these are the pre-routing recipes.
"""
import json


def export_build(report):
    return (json.dumps(report["build"], indent=2, sort_keys=True) + "\n").encode("utf-8")


def export_target(report):
    return json.dumps(report["serving_target"], sort_keys=True)


def materialization_file(value):
    return (json.dumps(value, indent=2, sort_keys=True,
                       allow_nan=False) + '\n').encode()


def plan_sidecar(provenance):
    return json.dumps(provenance, indent=2)
