"""JSON-backed registry for shipped and candidate PrismaQuant artifacts."""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from .pipeline import MetricGateSpec
from .digests import DIRECT_ASCII_LAX


DEFAULT_REGISTRY_PATH = Path("/home/rob/dq-runs/prismaquant-artifact-registry.json")
COMPARE_EPSILON = 0.005


@dataclass
class ArtifactRecord:
    record_id: str
    model_path: str
    artifact_path: str | None
    layer_config_sha: str
    layer_config_path: str | None
    target_bpp: float
    achieved_bpp: float
    format_histogram: dict
    ppl_wikitext: float | None
    ppl_mmlu_acc: float | None
    end_kl: float | None
    eval_meta: dict
    created_at: str
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping) -> "ArtifactRecord":
        names = {f.name for f in fields(cls)}
        data = {name: payload.get(name) for name in names if name in payload}
        if "notes" not in data:
            data["notes"] = ""
        missing = [f.name for f in fields(cls) if f.name not in data]
        missing = [name for name in missing if name != "notes"]
        if missing:
            raise ValueError(f"registry record missing fields: {', '.join(missing)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)


def new_record_id() -> str:
    return uuid.uuid4().hex[:12]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


canonical_layer_config_json = DIRECT_ASCII_LAX.text


layer_config_sha256 = DIRECT_ASCII_LAX.sha256


def load_layer_config(layer_config: Mapping | str | Path) -> dict:
    if isinstance(layer_config, Mapping):
        return dict(layer_config)
    path = Path(layer_config)
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: layer_config is not a JSON object")
    return payload


def resolve_layer_config_sha(layer_config: Mapping | str | Path) -> str:
    if isinstance(layer_config, Mapping):
        return layer_config_sha256(layer_config)

    text = str(layer_config).strip()
    if len(text) == 64 and all(c in "0123456789abcdefABCDEF" for c in text):
        return text.lower()

    path = Path(text)
    if path.exists():
        return layer_config_sha256(load_layer_config(path))

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FileNotFoundError(
            f"layer_config must be a dict, existing JSON path, sha256, or JSON text: {text}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("layer_config JSON text is not an object")
    return layer_config_sha256(payload)


class ArtifactRegistry:
    def __init__(self, path: str | Path = DEFAULT_REGISTRY_PATH):
        self.path = Path(path)
        self._records = self._load()

    def add(self, record: ArtifactRecord) -> None:
        if any(existing.record_id == record.record_id for existing in self._records):
            raise ValueError(f"duplicate artifact record_id: {record.record_id}")
        self._records.append(record)
        self._write()

    def find_by_layer_config(self, layer_config: dict | str) -> ArtifactRecord | None:
        layer_sha = resolve_layer_config_sha(layer_config)
        for record in self._records:
            if record.layer_config_sha == layer_sha:
                return record
        return None

    def find_by_model(self, model_path: str) -> list[ArtifactRecord]:
        return [record for record in self._records if record.model_path == model_path]

    def all(self) -> list[ArtifactRecord]:
        return list(self._records)

    def compare(self, candidate_id: str, baseline_id: str) -> dict:
        candidate = self._record_by_id(candidate_id)
        baseline = self._record_by_id(baseline_id)

        metrics = {
            "end_kl": _compare_metric(
                "end_kl",
                candidate.end_kl,
                baseline.end_kl,
                MetricGateSpec(
                    name="artifact.end_kl_improves",
                    metric="end_kl",
                    direction="lower_is_better",
                    mode="all",
                ),
            ),
            "ppl_wikitext": _compare_metric(
                "ppl_wikitext",
                candidate.ppl_wikitext,
                baseline.ppl_wikitext,
                MetricGateSpec(
                    name="artifact.ppl_wikitext_preserved",
                    metric="ppl_wikitext",
                    direction="lower_is_better",
                    mode="all",
                    require_improvement=False,
                    max_relative_regression=COMPARE_EPSILON,
                ),
            ),
            "ppl_mmlu_acc": _compare_metric(
                "ppl_mmlu_acc",
                candidate.ppl_mmlu_acc,
                baseline.ppl_mmlu_acc,
                MetricGateSpec(
                    name="artifact.ppl_mmlu_acc_preserved",
                    metric="ppl_mmlu_acc",
                    direction="higher_is_better",
                    mode="all",
                    require_improvement=False,
                    max_relative_regression=COMPARE_EPSILON,
                ),
            ),
        }
        reasons = [
            f"{name} failed"
            for name, result in metrics.items()
            if not result["passed"]
        ]
        return {
            "candidate_id": candidate_id,
            "baseline_id": baseline_id,
            "epsilon": COMPARE_EPSILON,
            "metrics": metrics,
            "pass": not reasons,
            "reasons": reasons,
        }

    def _record_by_id(self, record_id: str) -> ArtifactRecord:
        for record in self._records:
            if record.record_id == record_id:
                return record
        raise KeyError(f"artifact record not found: {record_id}")

    def _load(self) -> list[ArtifactRecord]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            raw_records = payload.get("records", [])
        elif isinstance(payload, list):
            raw_records = payload
        else:
            raise ValueError(f"{self.path}: registry JSON must be a list or object")
        return [ArtifactRecord.from_dict(record) for record in raw_records]

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": [record.to_dict() for record in self._records]}
        tmp = self.path.with_name(
            f".{self.path.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        finally:
            if tmp.exists():
                tmp.unlink()


def _missing_metric(candidate: float | None, baseline: float | None) -> dict:
    return {
        "candidate": candidate,
        "baseline": baseline,
        "delta": None,
        "relative_delta": None,
        "passed": False,
    }


def _compare_metric(
    metric: str,
    candidate: float | None,
    baseline: float | None,
    gate: MetricGateSpec,
) -> dict:
    evaluation = gate.evaluate(
        baseline={metric: baseline},
        candidate={metric: candidate},
    )
    decision = evaluation.decisions[0] if evaluation.decisions else None
    if decision is None or candidate is None or baseline is None:
        return _missing_metric(candidate, baseline)
    delta = float(candidate) - float(baseline)
    rel = delta / abs(float(baseline)) if baseline else delta
    return {
        "candidate": float(candidate),
        "baseline": float(baseline),
        "delta": delta,
        "relative_delta": rel,
        "passed": bool(decision.accepted),
    }
