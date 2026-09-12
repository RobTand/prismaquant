#!/usr/bin/env python3
"""Seal existing sparse-rate study outputs for a later allocation replay.

This is a post-run integrity record.  It binds bytes already written by a
study; it does not establish that a historical final run was frozen in time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SEAL_SCHEMA = "prismaquant.sparse_rate_study_evidence_seal.v1"
DATASET_SCHEMA = "prismaquant.sparse_rate_dataset.v1"
STUDY_SCHEMA = "prismaquant.sparse_rate_model_study.v1"
CURRENCY = "output_mse_under_route_activation_contract"


class StudySealError(ValueError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise StudySealError(message)


def load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise StudySealError(f"invalid JSON {path}") from exc
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def roster(study: Path, report: dict) -> list[dict]:
    models = report.get("models")
    require(isinstance(models, list) and models, "study report has no model roster")
    result, names = [], set()
    for row in models:
        require(isinstance(row, dict), "study report model is invalid")
        name, anchors = row.get("model"), row.get("anchors")
        require(isinstance(name, str) and name and name not in names,
                "study report model roster is duplicate or invalid")
        require(isinstance(anchors, list) and anchors and len(set(anchors)) == len(anchors)
                and all(type(rate) is int and rate > 0 for rate in anchors),
                f"study report model {name} has invalid anchors")
        npz = study / f"{name}.npz"
        require(npz.is_file() and not npz.is_symlink(), f"study model artifact missing: {name}")
        with np.load(npz, allow_pickle=False) as archive:
            require(set(archive.files) == {"predictions", "truth"}, f"invalid study model {name}")
        names.add(name)
        result.append({"name": name, "anchors": anchors, "prediction_npz_sha256": sha(npz)})
    require({path.stem for path in study.glob("*.npz")} == names,
            "study prediction artifacts do not match the report roster")
    return result


def verify_declared_anchor_predictions(dataset_npz: Path, study: Path, models: list[dict], *, stage: str) -> None:
    with np.load(dataset_npz, allow_pickle=False) as archive:
        rates, values = archive["rates"], archive["values"]
        selected = ((archive["families"] == "TESSERA_E4M3_K1")
                    & (archive["roles"] == "expert"))
        final = np.asarray(archive["layers"], dtype=np.int64) % 5 == 3
        selected &= final if stage == "final" else ~final
    rate_columns = {int(rate): index for index, rate in enumerate(rates)}
    for model in models:
        with np.load(study / f"{model['name']}.npz", allow_pickle=False) as archive:
            predictions = archive["predictions"]
        require(predictions.shape == values.shape, f"study model shape differs from dataset: {model['name']}")
        for anchor in model["anchors"]:
            require(anchor in rate_columns, f"declared anchor absent from dataset: {anchor}")
            expected, actual = (values[selected, rate_columns[anchor]],
                                predictions[selected, rate_columns[anchor]])
            finite = np.isfinite(expected)
            require(np.array_equal(actual[finite], expected[finite])
                    and not np.any(np.isfinite(actual[~finite])),
                    f"declared anchor prediction differs from dataset: {model['name']}/{anchor}")


def create_seal(dataset: str | Path, study: str | Path, out: str | Path, *,
                development_study: str | Path | None = None, choice: str | Path | None = None) -> dict:
    dataset, study, out = Path(dataset), Path(study), Path(out)
    require(not out.exists(), "seal output already exists")
    manifest_path, dataset_npz = dataset / "manifest.json", dataset / "sparse_rate_dataset.npz"
    plan_path, report_path = study / "plan.json", study / "report.json"
    manifest, plan, report = load_json(manifest_path), load_json(plan_path), load_json(report_path)
    require(manifest.get("schema") == DATASET_SCHEMA and manifest.get("currency") == CURRENCY,
            "dataset manifest currency or schema is unsupported")
    require(manifest.get("npz_sha256") == sha(dataset_npz), "dataset NPZ differs from manifest")
    require(plan.get("schema") == STUDY_SCHEMA and report.get("schema") == STUDY_SCHEMA
            and plan.get("stage") in {"development", "final"} and report.get("stage") == plan["stage"]
            and plan.get("currency") == CURRENCY and report.get("currency") == CURRENCY
            and report.get("identity") == plan.get("identity"), "study plan/report contract is unsupported")
    require(all(key in report and report[key] == value for key, value in plan.items()),
            "study report does not mirror its plan")
    identity = plan.get("identity", {})
    require(identity.get("dataset_npz_sha256") == sha(dataset_npz)
            and identity.get("dataset_manifest_sha256") == sha(manifest_path),
            "study does not bind this dataset")
    models = roster(study, report)
    verify_declared_anchor_predictions(dataset_npz, study, models, stage=plan["stage"])
    payload = {"schema": SEAL_SCHEMA, "post_run_sealing": True,
               "chronology": "post-run evidence seal; does not prove prior freeze chronology",
               "currency": CURRENCY,
               "dataset": {"npz_sha256": sha(dataset_npz), "manifest_sha256": sha(manifest_path)},
               "study": {"stage": plan["stage"], "plan_sha256": sha(plan_path),
                         "report_sha256": sha(report_path), "models": models}}
    if plan["stage"] == "final":
        require(development_study is not None and choice is not None,
                "final sealing requires --development-study and --choice")
        development_study, choice = Path(development_study), Path(choice)
        choice_value = load_json(choice)
        development_plan, development_report = development_study / "plan.json", development_study / "report.json"
        development_plan_value, development_report_value = load_json(development_plan), load_json(development_report)
        require(choice_value.get("schema") == STUDY_SCHEMA and choice_value.get("identity") == plan.get("identity")
                and development_plan_value.get("schema") == STUDY_SCHEMA
                and development_report_value.get("schema") == STUDY_SCHEMA
                and development_plan_value.get("stage") == development_report_value.get("stage") == "development"
                and development_plan_value.get("identity") == development_report_value.get("identity") == plan.get("identity")
                and all(key in development_report_value and development_report_value[key] == value
                        for key, value in development_plan_value.items())
                and choice_value.get("development_report_sha256") == sha(development_report)
                and plan.get("choice_sha256") == sha(choice),
                "final choice does not bind the supplied development report")
        expected = ["log_chord", choice_value.get("two_anchor"), choice_value.get("one_anchor")]
        require([row["name"] for row in payload["study"]["models"]] == list(dict.fromkeys(expected)),
                "final model roster does not match frozen choice")
        payload["final_dependency"] = {"choice_sha256": sha(choice),
                                        "development_report_sha256": sha(development_report)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True); parser.add_argument("--study", required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--development-study")
    parser.add_argument("--choice")
    args = parser.parse_args()
    create_seal(args.dataset, args.study, args.out, development_study=args.development_study, choice=args.choice)


if __name__ == "__main__":
    main()
