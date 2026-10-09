"""Consumer adapter: bind a serving-backed task config to a producer witness.

The D50 serving task adapter consumes the kernels producer witness and the
 standalone public verifier only. It starts no rank, imports no serving
 runtime, and adds no identity seal. It refuses alias-only or size-only
 evidence through the verifier, never through its own predicate.

A served task config names ``backend.name == "served"`` and carries the
 producer witness path plus the explicit expected join facts (endpoint,
 alias, attempt, ranks). The adapter verifies the witness with
 ``serving_runtime_verifier`` and records the machine-readable verdict in
 the task result identity. HF configs (``backend.name == "hf"``) pass
 through unchanged; this adapter changes no HF behavior.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from prismaquant.serving_runtime_verifier import read_expected, read_witness, verify

#: Backend name of a serving-backed task config.
SERVED_BACKEND = "served"

#: HF backend name. The adapter passes HF configs through unchanged.
HF_BACKEND = "hf"


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
    if not isinstance(ranks, list) or not ranks or any(type(r) is not int or r < 0 for r in ranks):
        problems.append("served binding requires serving_runtime.ranks as a rank list")
    if not _is_nonempty_str(binding.get("expected")):
        problems.append("served binding requires serving_runtime.expected")
    return problems


def bind_served_task(config: Mapping[str, Any], *,
                     environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Verify the producer witness and return the verdict record.

    Reads the witness and expectation files the config names, joins them
    with the explicit expected endpoint, alias, attempt, and ranks, and
    refuses on any verifier refusal. Starts no rank, imports no serving
    runtime, and seals no identity beyond the D32 recorded-label check:
    the config-to-file label comparison stamps in dev mode and refuses
    in certified mode. Byte integrity still refuses in both modes.
    """
    from prismaquant.dev_mode import seal_check

    problems = served_binding_problems(config)
    if problems:
        raise ValueError("; ".join(problems))
    binding = config["backend"]["serving_runtime"]
    witness = read_witness(binding["witness"], where="served witness")
    expected_doc = read_expected(binding["expected"])
    for key in ("endpoint", "served_alias", "attempt_id"):
        seal_check(f"served binding {key}", expected_doc.get(key), binding[key],
                   where="served task config",
                   refusal=ValueError(
                       f"served binding {key} differs from its expectation file"),
                   environ=environ)
    seal_check("served binding ranks", sorted(expected_doc.get("ranks") or []),
               sorted(binding["ranks"]), where="served task config",
               refusal=ValueError(
                   "served binding ranks differ from their expectation file"),
               environ=environ)
    expected = {"endpoint": binding["endpoint"], "served_alias": binding["served_alias"],
                "attempt_id": binding["attempt_id"], "ranks": sorted(binding["ranks"]),
                **{k: v for k, v in expected_doc.items()
                   if k not in ("endpoint", "served_alias", "attempt_id", "ranks")}}
    verdict = verify(witness, expected, environ=environ)
    if verdict.get("verdict") != "pass":
        raise ValueError(f"served runtime witness refused: {verdict.get('reason')}")
    return {"witness_sha256": verdict["witness_sha256"],
            "endpoint": binding["endpoint"], "served_alias": binding["served_alias"],
            "attempt_id": binding["attempt_id"], "ranks": sorted(binding["ranks"]),
            "verdict": "pass"}


def is_served_config(config: Mapping[str, Any]) -> bool:
    """Whether ``config`` names the served backend."""
    backend = config.get("backend") if isinstance(config, Mapping) else None
    return isinstance(backend, Mapping) and backend.get("name") == SERVED_BACKEND
