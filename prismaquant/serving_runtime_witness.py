"""Task-consumable serving runtime witness: one versioned public contract.

The kernels serving producer (``eng-serve-measure``) publishes this witness
 beside a live serve. The D50 serving task adapter consumes it. The producer
 owns production facts. The adapter owns task facts. This module owns the
 shared shape both sides read, so neither side invents its own spelling.

Schema ``prismaquant.serving_runtime_witness/1`` joins six facts to one
 launch attempt:

- ``endpoint``: the observed OpenAI base URL (scheme, host, port).
- ``served_alias``: the served model name the ``/v1/models`` endpoint lists.
- ``launch_attempt``: the producer launch identity (attempt id, image,
  argv, session fingerprint when the producer has one).
- ``ranks``: the actual rank set. Each rank carries byte evidence from
  its own load: covered files with per-file SHA-256, explicit coverage,
  and any representation change (transcode, shard, layout) in plain text.
- ``artifact``: the loaded artifact byte evidence (model SHA, inventory
  SHA, byte count).
- ``tokenizer``: the server tokenizer evidence (source file SHA-256 map,
  content SHA, effective settings the server uses).

An alias, a size, an input manifest, a lease, a launch argv, or a
 publication receipt alone never proves loaded bytes. The standalone
 verifier (``prismaquant.serving_runtime_verifier``) refuses each of them.
 A passed offline check proves the recorded launch, never current state.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any

#: The one witness schema this tree publishes and consumes.
WITNESS_SCHEMA = "prismaquant.serving_runtime_witness/1"

#: Witness schema versions this tree reads. Only version 1 exists.
READABLE_SCHEMAS = (WITNESS_SCHEMA,)


def _is_sha256hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def _is_nonempty_str(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def tokenizer_content_sha256(source_files: Mapping[str, Any]) -> str:
    """Recompute the tokenizer content digest from its source map.

    The collector hashes this value from the observed source files. The
    verifier recomputes it from the witnessed map. A supplied digest that
    does not equal this recomputation refuses.
    """
    import hashlib

    payload = {"files": {name: {"bytes": row["bytes"], "sha256": row["sha256"]}
                         for name, row in sorted(source_files.items())}}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _check_rank(rank: object, where: str) -> list[str]:
    """Name each structural defect of one rank row. Empty means valid."""
    problems: list[str] = []
    if not isinstance(rank, Mapping):
        return [f"{where}: rank must be an object"]
    if type(rank.get("rank")) is not int or rank["rank"] < 0:
        problems.append(f"{where}: rank index must be a nonnegative integer")
    files = rank.get("files")
    if not isinstance(files, list) or not files:
        problems.append(f"{where}: rank byte evidence needs a nonempty files list")
    else:
        seen: list[str] = []
        for index, row in enumerate(files):
            site = f"{where}.files[{index}]"
            if not isinstance(row, Mapping):
                problems.append(f"{site}: file row must be an object")
                continue
            if not _is_nonempty_str(row.get("path")):
                problems.append(f"{site}: file path must be a nonempty string")
            elif row["path"] in seen:
                problems.append(f"{site}: file path {row['path']!r} repeats a row")
            else:
                seen.append(row["path"])
            if not _is_sha256hex(row.get("sha256")):
                problems.append(f"{site}: file sha256 must be 64 lowercase hex")
            size = row.get("bytes")
            if type(size) is not int or size < 0:
                problems.append(f"{site}: file bytes must be a nonnegative integer")
    coverage = rank.get("coverage")
    if coverage not in ("complete", "partial"):
        problems.append(f"{where}: coverage must be complete or partial")
    changes = rank.get("representation_changes")
    if not isinstance(changes, list) or not all(isinstance(c, str) and c for c in changes):
        problems.append(f"{where}: representation_changes must be a string list")
    return problems


def witness_problems(witness: object) -> list[str]:
    """Name each structural defect of ``witness``. Empty means valid."""
    if not isinstance(witness, Mapping):
        return ["witness: witness must be a JSON object"]
    problems: list[str] = []
    if witness.get("schema") not in READABLE_SCHEMAS:
        problems.append(f"witness: schema must be one of {list(READABLE_SCHEMAS)}")
    endpoint = witness.get("endpoint")
    if not isinstance(endpoint, Mapping) or not _is_nonempty_str(endpoint.get("base_url")):
        problems.append("witness: endpoint.base_url must be a nonempty string")
    if not _is_nonempty_str(witness.get("served_alias")):
        problems.append("witness: served_alias must be a nonempty string")
    attempt = witness.get("launch_attempt")
    if not isinstance(attempt, Mapping) or not _is_nonempty_str(attempt.get("attempt_id")):
        problems.append("witness: launch_attempt.attempt_id must be a nonempty string")
    ranks = witness.get("ranks")
    if not isinstance(ranks, list) or not ranks:
        problems.append("witness: ranks must be a nonempty rank list")
    else:
        for index, rank in enumerate(ranks):
            problems.extend(_check_rank(rank, f"witness.ranks[{index}]"))
        seen_ranks = [rank["rank"] for rank in ranks
                      if isinstance(rank, Mapping) and type(rank.get("rank")) is int]
        if len(set(seen_ranks)) != len(seen_ranks):
            problems.append("witness: ranks must name each rank once")
    artifact = witness.get("artifact")
    if not isinstance(artifact, Mapping):
        problems.append("witness: artifact byte evidence must be an object")
    else:
        if not _is_sha256hex(artifact.get("model_sha256")):
            problems.append("witness: artifact.model_sha256 must be 64 lowercase hex")
        if not _is_sha256hex(artifact.get("inventory_sha256")):
            problems.append("witness: artifact.inventory_sha256 must be 64 lowercase hex")
        if type(artifact.get("artifact_bytes")) is not int or artifact["artifact_bytes"] < 0:
            problems.append("witness: artifact.artifact_bytes must be a nonnegative integer")
    tokenizer = witness.get("tokenizer")
    if not isinstance(tokenizer, Mapping):
        problems.append("witness: tokenizer evidence must be an object")
    else:
        files = tokenizer.get("source_files")
        if not isinstance(files, Mapping) or not files:
            problems.append("witness: tokenizer.source_files must be a nonempty name map")
        else:
            for name, row in files.items():
                if not isinstance(row, Mapping) or not _is_sha256hex(row.get("sha256")):
                    problems.append(f"witness: tokenizer.source_files[{name!r}] needs a file sha256")
                elif type(row.get("bytes")) is not int or row["bytes"] < 0:
                    problems.append(f"witness: tokenizer.source_files[{name!r}] needs file bytes")
        if not _is_sha256hex(tokenizer.get("content_sha256")):
            problems.append("witness: tokenizer.content_sha256 must be 64 lowercase hex")
        if not isinstance(tokenizer.get("effective"), Mapping):
            problems.append("witness: tokenizer.effective must be an object")
    return problems


def canonical_witness_bytes(witness: Mapping[str, Any]) -> bytes:
    """The canonical UTF-8 bytes of ``witness`` (sorted keys, strict).

    The stored ``witness_sha256`` is metadata about the joined facts, never
    one of them: canonical bytes exclude it so the digest a collector prints
    before the write is the digest a verifier recomputes after the read.
    """
    payload = {k: v for k, v in witness.items() if k != "witness_sha256"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def witness_sha256(witness: Mapping[str, Any]) -> str:
    """The SHA-256 of the canonical witness bytes."""
    import hashlib

    return hashlib.sha256(canonical_witness_bytes(witness)).hexdigest()
