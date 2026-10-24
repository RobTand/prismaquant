"""PR #2470 consumer binding gate, reconciled against the public producer CLI.

Only the configured standalone verifier establishes recorded launch evidence.
No serving runtime imports, inference, endpoint requests or local-only passes.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SERVED_BACKEND = "served"
HF_BACKEND = "hf"

def _reject_constant(value: str) -> Any:
    """Refuse NaN and Infinity tokens: the witness must be strict JSON."""
    raise ValueError(f"non-strict JSON constant {value}")


def _duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def read_witness(path: str | Path, *, where: str = "witness") -> dict[str, Any]:
    """Parse ``path`` as strict JSON with duplicate-key refusal."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"),
                           object_pairs_hook=_duplicate_key,
                           parse_constant=_reject_constant)
    except ValueError as exc:
        raise ValueError(f"{where} is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{where} is not a JSON object")
    return dict(value)


def read_expected(path: str | Path) -> dict[str, Any]:
    """Parse the consumer expectation document."""
    return read_witness(path, where="expected facts")


def _is_nonempty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def served_binding_problems(config: Mapping[str, Any]) -> list[str]:
    """Name each missing served-binding field. Empty means complete."""
    backend = config.get("backend") if isinstance(config, Mapping) else None
    if not isinstance(backend, Mapping) or backend.get("name") != SERVED_BACKEND:
        return ["served binding requires backend.name == 'served'"]
    problems: list[str] = []
    binding = backend.get("serving_runtime")
    if not isinstance(binding, Mapping):
        return ["served binding requires backend.serving_runtime"]
    if not _is_nonempty_str(binding.get("witness")):
        problems.append("served binding requires serving_runtime.witness")
    if not _is_nonempty_str(binding.get("endpoint")):
        problems.append("served binding requires serving_runtime.endpoint")
    if not _is_nonempty_str(binding.get("served_alias")):
        problems.append("served binding requires serving_runtime.served_alias")
    if not _is_nonempty_str(binding.get("attempt_id")):
        problems.append("served binding requires serving_runtime.attempt_id")
    ranks = binding.get("ranks")
    if (not isinstance(ranks, list) or not ranks
            or any(type(r) is not int or r < 0 for r in ranks)
            or ranks != list(range(len(ranks)))):
        problems.append(
            "served binding requires serving_runtime.ranks as a rank list")
    if not _is_nonempty_str(binding.get("expected")):
        problems.append("served binding requires serving_runtime.expected")
    for key in ("verifier", "served_dir"):
        if not _is_nonempty_str(binding.get(key)):
            problems.append(f"served binding requires serving_runtime.{key}")
    if "tokenizer_dir" in binding and not _is_nonempty_str(binding["tokenizer_dir"]):
        problems.append("served binding requires a non-empty serving_runtime.tokenizer_dir")
    return problems


def bind_served_task(config: Mapping[str, Any]) -> dict[str, Any]:
    """Require public producer authority before returning recorded launch evidence.

    The expectation document is independent input, never synthesized from the
    receipt. Local checks concern parsing and configuration consistency only;
    all witness semantics, loaded byte checks and tokenizer normalization belong
    to the public verifier, invoked exactly once.
    """
    from .served_task_public_verifier import run_public_verifier

    problems = served_binding_problems(config)
    if problems:
        raise ValueError("; ".join(problems))
    runtime = config["backend"]["serving_runtime"]
    try:
        witness = read_witness(runtime["witness"], where="served witness")
        expected = read_expected(runtime["expected"])
    except OSError as exc:
        raise ValueError(f"served witness/expected input unavailable: {exc}") from exc
    for key in ("endpoint", "served_alias", "attempt_id", "ranks"):
        # JSON equality, including numeric types, must not make bool a rank.
        if json.dumps(expected.get(key), sort_keys=True) != json.dumps(runtime[key], sort_keys=True):
            raise ValueError(f"served binding {key} differs from its expectation file")
    public = run_public_verifier(
        cli=runtime["verifier"], witness=runtime["witness"], expected=expected,
        served_dir=runtime["served_dir"], tokenizer_dir=runtime.get("tokenizer_dir"))
    return {"witness_fingerprint": witness["fingerprint"],
            **{key: expected[key] for key in ("endpoint", "served_alias", "attempt_id", "ranks")},
            "verified_facts": {key: expected[key] for key in ("artifacts", "tokenizer")},
            "proof_scope": public["proof_scope"], "verdict": "pass",
            "public_verifier": public,
            "note": "Recorded launch evidence only; not current endpoint availability."}


def is_served_config(config: Mapping[str, Any]) -> bool:
    """Whether ``config`` names the served backend."""
    backend = config.get("backend") if isinstance(config, Mapping) else None
    return isinstance(backend, Mapping) and backend.get("name") == SERVED_BACKEND


