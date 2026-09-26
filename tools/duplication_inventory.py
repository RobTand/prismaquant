"""Find logically duplicated functionality by structure and by name (PQ #1295).

Clone-finding audits compare text, and duplicated functionality rarely
repeats text: renamed helpers and re-typed loops look different line by
line. This scanner compares structure and names instead:

- **Near-duplicates.** Each function's body becomes a sequence of AST node
  kinds, keeping attribute names and called names and dropping local
  identifiers, literals and docstrings. The sequence is cut into 6-token
  shingles and hashed with blake2b, so the result does not depend on the
  process's hash seed. Pairs of functions in different files whose Jaccard
  similarity is at least ``THRESHOLD`` are reported. MinHash with seeded
  salts proposes candidates, and each candidate's Jaccard is then computed
  exactly.
- **Same-name helpers.** A module-level function name defined in more than
  one module.

``tests/test_duplication_baseline.py`` holds the live result against
``tests/fixtures/duplication_baseline.json``, which only shrinks.
``--write-baseline`` rewrites the baseline after a consolidation removes
entries.

A pair that must stay two implementations is recorded in the baseline's
``must_differ`` list with its reason (PQ #1302): for example, an independent
reference oracle, or a sealed module whose bytes are bound into receipts.
``--write-baseline`` keeps those records for pairs that still exist.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("prismaquant", "tools")
EXCLUDED_PARTS = {"vendored", "archive", "__pycache__"}
BASELINE = ROOT / "tests" / "fixtures" / "duplication_baseline.json"
THRESHOLD = 0.9
MIN_LINES = 6
MIN_SHINGLES = 20
SHINGLE = 6
_K, _BANDS = 64, 16
# CLI entry points are expected once per script, not duplication.
ENTRY_POINTS = frozenset({"main", "_main", "parse_args", "_parse_args",
                          "build_parser", "_build_parser", "cli"})


def _files(root: Path):
    for top in SCANNED:
        for path in sorted((root / top).rglob("*.py")):
            if EXCLUDED_PARTS.isdisjoint(path.relative_to(root).parts):
                yield path


def _tokens(node: ast.AST) -> list[str]:
    out = []
    for sub in ast.walk(node):
        kind = type(sub).__name__
        if isinstance(sub, ast.Attribute):
            kind += "." + sub.attr
        elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
            kind += ":" + sub.func.id
        out.append(kind)
    return out


def _shingles(tokens: list[str]) -> frozenset[int]:
    return frozenset(
        int.from_bytes(hashlib.blake2b(
            "\x1f".join(tokens[i:i + SHINGLE]).encode(), digest_size=8).digest(), "big")
        for i in range(max(0, len(tokens) - SHINGLE + 1)))


def scan(root: Path = ROOT) -> dict:
    fingerprints = []
    names: dict[str, set[str]] = defaultdict(set)
    for path in _files(root):
        rel = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for top in tree.body:
            if isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)) and not top.name.startswith("__") and top.name not in ENTRY_POINTS:
                names[top.name].add(rel)

        def walk(node, scope):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    qual = scope + [child.name]
                    if not isinstance(child, ast.ClassDef) and child.end_lineno - child.lineno >= MIN_LINES:
                        body = child.body
                        if (body and isinstance(body[0], ast.Expr)
                                and isinstance(body[0].value, ast.Constant)
                                and isinstance(body[0].value.value, str)):
                            body = body[1:]
                        shingles = _shingles([t for stmt in body for t in _tokens(stmt)])
                        if len(shingles) >= MIN_SHINGLES:
                            fingerprints.append((f"{rel}::{'.'.join(qual)}", shingles))
                    walk(child, qual)
                else:
                    walk(child, scope)

        walk(tree, [])
    rng = random.Random(7)
    salts = [rng.getrandbits(61) for _ in range(_K)]
    rows = _K // _BANDS
    buckets = defaultdict(list)
    for index, (_, shingles) in enumerate(fingerprints):
        signature = [min(((h ^ s) * 0x9E3779B97F4A7C15) & ((1 << 61) - 1) for h in shingles)
                     for s in salts]
        for band in range(_BANDS):
            buckets[(band, tuple(signature[band * rows:(band + 1) * rows]))].append(index)
    pairs = set()
    for members in buckets.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = fingerprints[members[i]], fingerprints[members[j]]
                if a[0].split("::")[0] == b[0].split("::")[0]:
                    continue
                if len(a[1] & b[1]) / len(a[1] | b[1]) >= THRESHOLD:
                    pairs.add(tuple(sorted((a[0], b[0]))))
    return {
        "near_duplicates": sorted([list(p) for p in pairs]),
        "same_name_helpers": {n: sorted(m) for n, m in sorted(names.items()) if len(m) > 1},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()
    live = scan()
    old = json.loads(BASELINE.read_text(encoding="utf-8")) if BASELINE.exists() else {}
    pairs = {tuple(p) for p in live["near_duplicates"]}
    live["must_differ"] = [row for row in old.get("must_differ", [])
                           if tuple(row["pair"]) in pairs]
    if args.write_baseline:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(json.dumps(live, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    groups = live["same_name_helpers"]
    print(f"near-duplicate pairs >= {THRESHOLD}: {len(live['near_duplicates'])}; "
          f"same-name helper groups: {len(groups)} "
          f"({sum(len(v) for v in groups.values())} definitions); "
          f"must-differ pairs: {len(live['must_differ'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
