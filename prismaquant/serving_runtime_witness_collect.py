"""Collect a task-consumable serving runtime witness from a live serve.

The kernels serving producer runs this collector beside the serving ranks.
 It reads the observed endpoint, served alias, launch attempt, actual rank
 set, loaded artifact bytes, and server tokenizer bytes, then writes one
 versioned JSON witness (``serving_runtime_witness``). Each rank supplies
 byte evidence from its own load: covered files with SHA-256, explicit
 coverage, and representation changes in plain text.

The collector starts no rank and performs no inference. It queries the
 live ``/v1/models`` endpoint for the served alias, reads mounted artifact
 and tokenizer bytes, and joins them to the producer attempt the operator
 names. Size-only or alias-only evidence refuses: every witnessed file
 carries both bytes and a digest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from prismaquant.serving_runtime_witness import (
    WITNESS_SCHEMA,
    canonical_witness_bytes,
    witness_problems,
    witness_sha256,
)

#: Tokenizer source files the collector hashes when present beside the server
#: tokenizer directory. The witness records the observed set; this tuple only
#: bounds the scan so unrelated files cannot enter server evidence.
TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def query_served_alias(base_url: str, *, timeout: float = 30.0) -> str:
    """The one model id the live ``/v1/models`` endpoint lists, or a refusal."""
    url = base_url.rstrip("/") + "/v1/models"
    request = urllib.request.Request(url, method="GET",
                                     headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(1024 * 1024 + 1)
    except Exception as exc:
        raise ValueError(f"cannot reach {url}: {exc}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except ValueError as exc:
        raise ValueError(f"{url} returned non-JSON model bytes") from exc
    rows = payload.get("data") if isinstance(payload, Mapping) else None
    names = sorted({row.get("id") for row in rows
                    if isinstance(row, Mapping) and isinstance(row.get("id"), str)})
    if len(names) != 1:
        raise ValueError(f"{url} must list exactly one served alias, saw {names}")
    return names[0]


def rank_byte_evidence(rank: int, artifact_dir: Path, *,
                       representation_changes: tuple[str, ...] = ()) -> dict[str, Any]:
    """Byte evidence for the files rank ``rank`` loaded from ``artifact_dir``."""
    root = artifact_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"served artifact is not a directory: {root}")
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"served artifact contains symlink {path}")
        if path.is_file():
            files.append({"path": path.relative_to(root).as_posix(),
                          "sha256": _sha256_file(path),
                          "bytes": int(path.stat().st_size)})
    if not files:
        raise ValueError(f"served artifact holds no files: {root}")
    return {"rank": rank, "files": files, "coverage": "complete",
            "representation_changes": list(representation_changes)}


def tokenizer_evidence(tokenizer_dir: Path, *,
                       effective: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Server tokenizer evidence: source bytes plus effective settings."""
    root = tokenizer_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"server tokenizer is not a directory: {root}")
    files = {}
    for name in TOKENIZER_FILENAMES:
        path = root / name
        if path.is_file():
            files[name] = {"sha256": _sha256_file(path),
                           "bytes": int(path.stat().st_size)}
    if not files:
        raise ValueError(f"server tokenizer holds no tokenizer files: {root}")
    digest = hashlib.sha256(json.dumps(
        files, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()
    return {"source_files": files, "content_sha256": digest,
            "effective": dict(effective or {})}


def collect_witness(*, endpoint: str, served_alias: str, attempt_id: str,
                    image: str | None, launch_argv: list[str],
                    ranks: list[dict[str, Any]], artifact: Mapping[str, Any],
                    tokenizer: Mapping[str, Any]) -> dict[str, Any]:
    """Join producer observations to one launch attempt. Refuse gaps."""
    witness = {"schema": WITNESS_SCHEMA,
               "endpoint": {"base_url": endpoint},
               "served_alias": served_alias,
               "launch_attempt": {"attempt_id": attempt_id, "image": image,
                                  "argv": list(launch_argv)},
               "ranks": list(ranks),
               "artifact": dict(artifact),
               "tokenizer": dict(tokenizer)}
    problems = witness_problems(witness)
    if problems:
        raise ValueError("; ".join(problems))
    witness["witness_sha256"] = witness_sha256(witness)
    return witness


def write_witness(path: str | Path, witness: Mapping[str, Any]) -> Path:
    """Write ``witness`` without its self-digest. Refuse an existing destination."""
    out = Path(path)
    if out.exists():
        raise ValueError(f"witness output already exists: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    stored = dict(witness)
    stored.pop("witness_sha256", None)
    out.write_bytes(canonical_witness_bytes(stored) + b"\n")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--tokenizer-dir", default=None)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--launch-argv", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--ranks", required=True)
    parser.add_argument("--representation-changes", default="")
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--artifact-bytes", type=int, required=True)
    parser.add_argument("--tokenizer-setting", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        ranks = sorted({int(part) for part in args.ranks.split(",") if part.strip()})
    except ValueError as exc:
        raise SystemExit(f"collect: --ranks must be comma integers: {exc}")
    if not ranks or any(rank < 0 for rank in ranks):
        raise SystemExit("collect: --ranks must name one nonnegative rank at least")
    changes = tuple(c for c in args.representation_changes.split(",") if c.strip())
    artifact_dir = Path(args.artifact_dir)
    tokenizer_dir = Path(args.tokenizer_dir or args.artifact_dir)
    served_alias = query_served_alias(args.endpoint)
    rank_rows = [rank_byte_evidence(rank, artifact_dir, representation_changes=changes)
                 for rank in ranks]
    effective = dict(setting.split("=", 1) for setting in args.tokenizer_setting
                     if "=" in setting)
    witness = collect_witness(
        endpoint=args.endpoint, served_alias=served_alias, attempt_id=args.attempt,
        image=args.image, launch_argv=list(args.launch_argv), ranks=rank_rows,
        artifact={"model_sha256": args.model_sha256,
                  "inventory_sha256": args.inventory_sha256,
                  "artifact_bytes": args.artifact_bytes},
        tokenizer=tokenizer_evidence(tokenizer_dir, effective=effective))
    write_witness(args.out, witness)
    print(json.dumps({"witness": str(args.out), "witness_sha256": witness["witness_sha256"],
                      "served_alias": served_alias, "ranks": ranks}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
