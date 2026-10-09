"""Standalone CPU verifier for the task-consumable serving runtime witness.

The verifier joins the producer witness (``serving_runtime_witness``) to
 explicit consumer expectations and returns one machine-readable verdict.
 It runs on CPU, imports no serving runtime, starts no rank, performs no
 model inference, and reads no endpoint state. A pass proves the recorded
 launch, never the current endpoint.

Refusals (nonzero status, ``"verdict": "refuse"``):

- absent, malformed, or duplicate-key witness bytes;
- missing endpoint, alias, attempt, rank, artifact, or tokenizer facts;
- alias-only evidence (no rank byte rows) or size-only evidence (no digests);
- incomplete rank coverage or ranks outside the expected set;
- any join mismatch: endpoint, alias, attempt, artifact digest, tokenizer
  digest, or a rank file digest that differs from the expected bytes.

Usage::

    python -m prismaquant.serving_runtime_verifier --witness witness.json \\
        --endpoint http://127.0.0.1:8000 --alias model \\
        --attempt launch-1 --ranks 0,1 --expected expected.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from prismaquant.serving_runtime_witness import (
    WITNESS_SCHEMA,
    witness_problems,
)

#: Machine-readable verdicts this CLI returns.
VERDICT_PASS = "pass"
VERDICT_REFUSE = "refuse"


def _duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate witness key: {key!r}")
        result[key] = value
    return result


def read_witness(path: str | Path, *, where: str = "witness") -> dict[str, Any]:
    """Parse ``path`` as strict JSON with duplicate-key refusal."""
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_duplicate_key,
                            parse_constant=lambda name: (_ for _ in ()).throw(
                                ValueError(f"{where}: nonfinite constant {name}")))
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{where}: witness must be a JSON object")
    return dict(value)


def read_expected(path: str | Path) -> dict[str, Any]:
    """Parse the consumer expectation document."""
    raw = Path(path).read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_duplicate_key,
                            parse_constant=lambda name: (_ for _ in ()).throw(
                                ValueError(f"expected: nonfinite constant {name}")))
    except ValueError as exc:
        raise ValueError(f"expected: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError("expected: expectation must be a JSON object")
    return dict(value)


def _parse_ranks(text: str) -> list[int]:
    try:
        ranks = sorted({int(part) for part in text.split(",") if part.strip() != ""})
    except ValueError as exc:
        raise ValueError(f"--ranks must be comma integers: {text!r}") from exc
    if not ranks or any(rank < 0 for rank in ranks):
        raise ValueError(f"--ranks must name one nonnegative rank at least: {text!r}")
    return ranks


def _refuse(reason: str) -> dict[str, Any]:
    return {"schema": WITNESS_SCHEMA, "verdict": VERDICT_REFUSE, "reason": reason}


def verify(witness: Mapping[str, Any], expected: Mapping[str, Any]) -> dict[str, Any]:
    """Join ``witness`` to ``expected``. Never touches a rank or endpoint."""
    problems = witness_problems(witness)
    if problems:
        return _refuse("; ".join(problems))
    failures: list[str] = []
    endpoint = witness["endpoint"]
    if endpoint.get("base_url") != expected.get("endpoint"):
        failures.append("endpoint differs from the observed endpoint")
    if witness.get("served_alias") != expected.get("served_alias"):
        failures.append("served alias differs from the served alias")
    attempt = witness["launch_attempt"]
    if attempt.get("attempt_id") != expected.get("attempt_id"):
        failures.append("launch attempt differs from the recorded attempt")
    wanted = expected.get("ranks")
    if not isinstance(wanted, list) or not wanted or any(type(r) is not int for r in wanted):
        return _refuse("expected ranks must be a nonempty integer list")
    seen = sorted(row["rank"] for row in witness["ranks"])
    if seen != sorted(wanted):
        failures.append(f"actual rank set {seen} differs from expected {sorted(wanted)}")
    for row in witness["ranks"]:
        if row.get("coverage") != "complete":
            failures.append(f"rank {row.get('rank')}: coverage is not complete")
    artifact_files = expected.get("artifact_files")
    if not isinstance(artifact_files, Mapping) or not artifact_files:
        return _refuse("expected artifact_files must be a nonempty path map")
    for row in witness["ranks"]:
        for entry in row["files"]:
            want = artifact_files.get(entry["path"])
            if want is None:
                failures.append(f"rank {row['rank']}: file {entry['path']!r} is not expected")
            elif entry["sha256"] != want:
                failures.append(f"rank {row['rank']}: file {entry['path']!r} digest differs")
    artifact = witness["artifact"]
    for key in ("model_sha256", "inventory_sha256", "artifact_bytes"):
        if artifact.get(key) != (expected.get("artifact") or {}).get(key):
            failures.append(f"artifact {key} differs from the expected bytes")
    tokenizer = witness["tokenizer"]
    for name, row in ((expected.get("tokenizer_files") or {}).items()):
        seen_row = (tokenizer.get("source_files") or {}).get(name)
        if seen_row is None:
            failures.append(f"tokenizer file {name!r} is absent from server evidence")
        elif seen_row.get("sha256") != row:
            failures.append(f"tokenizer file {name!r} digest differs")
    if tokenizer.get("content_sha256") != expected.get("tokenizer_content_sha256"):
        failures.append("tokenizer content differs from the expected bytes")
    if failures:
        return _refuse("; ".join(failures))
    from prismaquant.serving_runtime_witness import canonical_witness_bytes
    digest = hashlib.sha256(canonical_witness_bytes(witness)).hexdigest()
    return {"schema": WITNESS_SCHEMA, "verdict": VERDICT_PASS,
            "witness_sha256": digest,
            "ranks": sorted(wanted),
            "note": "Recorded launch only; not current endpoint state."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--witness", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--alias", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--ranks", required=True)
    parser.add_argument("--expected", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    try:
        witness = read_witness(args.witness)
        expected_doc = read_expected(args.expected)
        expected = {"endpoint": args.endpoint, "served_alias": args.alias,
                    "attempt_id": args.attempt, "ranks": _parse_ranks(args.ranks),
                    **{k: v for k, v in expected_doc.items()
                       if k not in ("endpoint", "served_alias", "attempt_id", "ranks")}}
        verdict = verify(witness, expected)
    except ValueError as exc:
        verdict = _refuse(str(exc))
    text = json.dumps(verdict, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if verdict["verdict"] == VERDICT_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
