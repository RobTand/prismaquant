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
    tokenizer_content_sha256,
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


def query_served_alias(base_url: str, *, timeout: float = 30.0,
                       expected_served_model: str | None = None) -> str:
    """The one model id the live ``/v1/models`` endpoint lists, or a refusal.

    Reuses the existing server binding: the response must carry one model
    card with the exact identity fields. ``expected_served_model`` names
    the alias the launch declares; the live reply must list exactly it.
    """
    import sys
    import urllib.request

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tools.serve_fingerprint import (  # noqa: E402
        _models_endpoint_url,
        models_endpoint_binding_from_bytes,
        models_endpoint_binding_identity,
        query_models_endpoint_binding,
    )

    if expected_served_model is not None:
        bound = query_models_endpoint_binding(
            base_url, expected_served_model=expected_served_model,
            timeout=timeout)
        return str(models_endpoint_binding_identity(bound)["model"]["id"])
    request_url = _models_endpoint_url(base_url)
    request = urllib.request.Request(
        request_url, method="GET",
        headers={"Accept": "application/json", "Accept-Encoding": "identity"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status < 200 or response.status >= 300:
                raise ValueError(
                    f"GET {request_url} returned HTTP {response.status}")
            raw = response.read(16 * 1024 * 1024 + 1)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ValueError(f"GET {request_url} failed: {exc}") from exc
    try:
        probe = json.loads(raw.decode("utf-8", "strict"))
        candidate = probe["data"][0]["id"]
    except (UnicodeError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"{request_url} returned no served alias") from exc
    if not isinstance(candidate, str) or not candidate:
        raise ValueError(f"{request_url} returned no served alias")
    bound = models_endpoint_binding_from_bytes(
        raw, request_url=request_url, expected_served_model=candidate)
    return str(models_endpoint_binding_identity(bound)["model"]["id"])


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


def rank_byte_evidence_from_roots(
        rank_roots: list[Path],
        *, representation_changes: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    """Byte evidence from each rank's own load root, in rank order.

    Each entry names one rank's observed load root: the local directory
    that rank's loader read. Rank ``i`` reads ``rank_roots[i]``. Every
    rank must cover the same file roster with the same digests and byte
    counts, or the join refuses. The same root twice refuses: two ranks
    that read one directory are not two rank observations.
    """
    if not rank_roots:
        raise ValueError("rank observation needs one load root at least")
    resolved = [root.resolve(strict=True) for root in rank_roots]
    if len({str(root) for root in resolved}) != len(resolved):
        raise ValueError("rank load roots must differ per rank")
    rows = [rank_byte_evidence(rank, root,
                               representation_changes=representation_changes)
            for rank, root in enumerate(resolved)]
    first = [(entry["path"], entry["sha256"], entry["bytes"])
             for entry in rows[0]["files"]]
    for row in rows[1:]:
        other = [(entry["path"], entry["sha256"], entry["bytes"])
                 for entry in row["files"]]
        if other != first:
            raise ValueError(
                f"rank {row['rank']} loaded bytes differ from rank 0")
    return rows


def tokenizer_evidence(tokenizer_dir: Path, *,
                       effective: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Server tokenizer evidence: source bytes plus effective settings.

    Hashes the tokenizer source files beside the server tokenizer
    directory at observation time. The content digest follows the one
    witness rule, so the verifier recomputes it from the source map.
    Effective settings name the server behavior the operator attests;
    the collector requires every named setting and refuses an empty map.
    """
    root = tokenizer_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"server tokenizer is not a directory: {root}")
    files = {}
    for name in TOKENIZER_FILENAMES:
        path = root / name
        if path.is_file() and not path.is_symlink():
            files[name] = {"sha256": _sha256_file(path),
                           "bytes": int(path.stat().st_size)}
    if not files:
        raise ValueError(f"server tokenizer holds no tokenizer files: {root}")
    settings = dict(effective or {})
    if not settings:
        raise ValueError("server tokenizer needs its effective settings")
    return {"source_files": files,
            "content_sha256": tokenizer_content_sha256(files),
            "effective": settings}


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
    parser.add_argument("--rank-dirs", default=None)
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
    parser.add_argument("--expected-alias", default=None)
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
    served_alias = query_served_alias(
        args.endpoint, expected_served_model=args.expected_alias)
    if args.rank_dirs is None:
        rank_rows = [rank_byte_evidence(rank, artifact_dir,
                                        representation_changes=changes)
                     for rank in ranks]
    else:
        mapping: dict[int, str] = {}
        for pair in args.rank_dirs.split(","):
            rank_text, _, directory = pair.partition("=")
            try:
                mapping[int(rank_text)] = directory
            except ValueError as exc:
                raise SystemExit(f"collect: --rank-dirs names no rank: {pair!r}") from exc
        if sorted(mapping) != ranks or any(not directory for directory in mapping.values()):
            raise SystemExit("collect: --rank-dirs must cover every requested rank")
        rank_rows = rank_byte_evidence_from_roots(
            [Path(mapping[rank]) for rank in ranks],
            representation_changes=changes)
        rank_rows = [dict(row, rank=rank) for row, rank in zip(rank_rows, ranks)]
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
