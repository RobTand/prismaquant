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
    tokenizer_content_sha256,
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


def verify(witness: Mapping[str, Any], expected: Mapping[str, Any],
           *, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Join ``witness`` to ``expected``. Never touches a rank or endpoint.

    Recorded labels (endpoint, alias, attempt, expected rank list) go
    through the D32 seal: dev mode stamps and continues, certified mode
    refuses. Missing evidence, byte integrity, coverage, tokenizer joins,
    and cross-rank agreement refuse in both modes.
    """
    from prismaquant.dev_mode import seal_check

    problems = witness_problems(witness)
    if problems:
        return _refuse("; ".join(problems))
    failures: list[str] = []
    endpoint = witness["endpoint"]
    if not seal_check("served endpoint", expected.get("endpoint"),
                      endpoint.get("base_url"), where="served runtime witness",
                      refusal=ValueError("endpoint differs from the observed endpoint"),
                      environ=environ):
        failures.append("endpoint differs from the observed endpoint (dev stamp)")
    if not seal_check("served alias", expected.get("served_alias"),
                      witness.get("served_alias"), where="served runtime witness",
                      refusal=ValueError("served alias differs from the served alias"),
                      environ=environ):
        failures.append("served alias differs from the served alias (dev stamp)")
    attempt = witness["launch_attempt"]
    if not seal_check("launch attempt", expected.get("attempt_id"),
                      attempt.get("attempt_id"), where="served runtime witness",
                      refusal=ValueError("launch attempt differs from the recorded attempt"),
                      environ=environ):
        failures.append("launch attempt differs from the recorded attempt (dev stamp)")
    wanted = expected.get("ranks")
    if not isinstance(wanted, list) or not wanted or any(type(r) is not int for r in wanted):
        return _refuse("expected ranks must be a nonempty integer list")
    seen = sorted(row["rank"] for row in witness["ranks"])
    if not seal_check("served rank set", sorted(wanted), seen,
                      where="served runtime witness",
                      refusal=ValueError(
                          f"actual rank set {seen} differs from expected {sorted(wanted)}"),
                      environ=environ):
        failures.append(f"actual rank set {seen} differs from expected "
                        f"{sorted(wanted)} (dev stamp)")
    artifact_files = expected.get("artifact_files")
    if not isinstance(artifact_files, Mapping) or not artifact_files:
        return _refuse("expected artifact_files must be a nonempty path map")
    for row in witness["ranks"]:
        if row.get("coverage") != "complete":
            failures.append(f"rank {row.get('rank')}: coverage is not complete")
    rosters = [sorted(entry["path"] for entry in row["files"])
               for row in witness["ranks"]]
    if any(roster != rosters[0] for roster in rosters[1:]):
        failures.append("rank rows cover different file rosters")
    want_paths = sorted(artifact_files)
    if rosters[0] != want_paths:
        missing = sorted(set(want_paths) - set(rosters[0]))
        extra = sorted(set(rosters[0]) - set(want_paths))
        if missing:
            failures.append(f"rank evidence omits expected files: {missing}")
        if extra:
            failures.append(f"rank evidence holds unexpected files: {extra}")
    by_path: dict[str, dict[str, Any]] = {}
    for row in witness["ranks"]:
        for entry in row["files"]:
            prior = by_path.setdefault(entry["path"], dict(entry))
            if entry["sha256"] != prior["sha256"]:
                failures.append(f"file {entry['path']!r} digest differs across ranks")
            if entry["bytes"] != prior["bytes"]:
                failures.append(f"file {entry['path']!r} byte count differs across ranks")
    for path, want in sorted(artifact_files.items()):
        seen_entry = by_path.get(path)
        if seen_entry is None:
            continue
        want_digest = want.get("sha256") if isinstance(want, Mapping) else want
        if seen_entry["sha256"] != want_digest:
            failures.append(f"file {path!r} digest differs")
        if isinstance(want, Mapping) and seen_entry["bytes"] != want.get("bytes"):
            failures.append(f"file {path!r} byte count differs from the expected bytes")
    artifact = witness["artifact"]
    row_bytes = sum(entry["bytes"] for entry in witness["ranks"][0]["files"])
    if artifact.get("artifact_bytes") != row_bytes:
        failures.append("artifact byte total differs from the rank file rows")
    for key in ("model_sha256", "inventory_sha256", "artifact_bytes"):
        if artifact.get(key) != (expected.get("artifact") or {}).get(key):
            failures.append(f"artifact {key} differs from the expected bytes")
    tokenizer = witness["tokenizer"]
    source_files = tokenizer.get("source_files") or {}
    try:
        rebuilt = tokenizer_content_sha256(source_files)
    except (KeyError, TypeError, AttributeError):
        return _refuse("tokenizer source map cannot rebuild its content digest")
    if tokenizer.get("content_sha256") != rebuilt:
        failures.append("tokenizer content digest differs from its source files")
    want_tokens = expected.get("tokenizer_files")
    if not isinstance(want_tokens, Mapping) or not want_tokens:
        return _refuse("expected tokenizer_files must be a nonempty name map")
    for name in sorted(set(source_files) - set(want_tokens)):
        failures.append(f"tokenizer file {name!r} is absent from expected evidence")
    for name, want in sorted(want_tokens.items()):
        seen_row = source_files.get(name)
        if seen_row is None:
            failures.append(f"tokenizer file {name!r} is absent from server evidence")
        else:
            want_digest = want.get("sha256") if isinstance(want, Mapping) else want
            if seen_row.get("sha256") != want_digest:
                failures.append(f"tokenizer file {name!r} digest differs")
            if isinstance(want, Mapping) and seen_row.get("bytes") != want.get("bytes"):
                failures.append(f"tokenizer file {name!r} byte count differs")
    if tokenizer.get("content_sha256") != expected.get("tokenizer_content_sha256"):
        failures.append("tokenizer content differs from the expected bytes")
    want_effective = expected.get("tokenizer_effective")
    if not isinstance(want_effective, Mapping):
        return _refuse("expected tokenizer_effective must be an object")
    if dict(tokenizer.get("effective") or {}) != dict(want_effective):
        failures.append("tokenizer effective settings differ from the expected settings")
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
