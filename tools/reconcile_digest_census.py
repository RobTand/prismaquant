"""Compare current digest observations with refresh_2540 without claiming completion."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.digests import DIRECT_UTF8_INDENT2_STRICT, bytes_sha256hex
from tools import duplication_inventory as inventory


ROOT = Path(__file__).resolve().parents[1]
CENSUS = Path("docs/audits/digest_site_census_pq1301_2026-10-04.json")


def _expression_key(row: dict) -> tuple[str, str, str, str]:
    expression = ast.dump(ast.parse(row["expression"], mode="eval"),
                          include_attributes=False)
    return row["family"], row["file"], row["scope"], expression


def observe_calls(root: Path) -> tuple[list[dict], dict[str, str]]:
    """Use the inventory's file, scope and alias rules; retain individual calls."""
    calls = []
    sources = {}
    for path in inventory._files(root):
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        sources[relative] = bytes_sha256hex(data)
        source = data.decode("utf-8")
        tree = ast.parse(source, filename=relative)
        for call, scope, aliases in inventory._iter_scoped_calls(tree):
            callee = inventory._resolved_callee(call.func, aliases)
            if callee is None:
                continue
            if callee.startswith("hashlib."):
                family = "hashlib"
            elif callee in ("json.dumps", "json.JSONEncoder") and any(
                    keyword.arg == "sort_keys"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True for keyword in call.keywords):
                family = "sorted-json"
            elif callee in ("json.dumps", "json.dump", "json.JSONEncoder"):
                family = "supplemental-json"
            else:
                continue
            site = None if family == "supplemental-json" else f"{family}:{relative}::{scope}"
            calls.append({
                "id": f"{family}:{relative}::{scope}@{call.lineno}:{call.col_offset}",
                "site": site, "family": family, "file": relative, "scope": scope,
                "line": call.lineno, "column": call.col_offset, "callee": callee,
                "expression": ast.get_source_segment(source, call),
            })
    projected = sorted({row["site"] for row in calls if row["site"] is not None})
    if projected != inventory.digest_sites(root):
        raise ValueError("primitive scope projection differs from digest_sites()")
    return sorted(calls, key=lambda row: row["id"]), sources


def reconcile(root: Path, refresh: dict) -> dict:
    """Keep historical candidates separate from current acceptance decisions."""
    historical = refresh["calls"] + refresh["supplemental_calls"]
    old_by_expression = defaultdict(list)
    ids = set()
    for row in historical:
        if row["id"] in ids:
            raise ValueError(f"duplicate historical id: {row['id']}")
        ids.add(row["id"])
        old_by_expression[_expression_key(row)].append(row)
    owners = {}
    for row in refresh["module_owners"]:
        if row["file"] in owners:
            raise ValueError(f"duplicate module owner: {row['file']}")
        owners[row["file"]] = row["epic"]

    calls, sources = observe_calls(root)
    current_counts = Counter(_expression_key(row) for row in calls)
    observed_ids = set()
    for row in calls:
        key = _expression_key(row)
        candidates = old_by_expression[key]
        row["historical_ids"] = sorted(old["id"] for old in candidates)
        observed_ids.update(row["historical_ids"])
        row["epic"] = owners.get(row["file"], "prismaquant#1301")
        row["review_required"] = True
        if not candidates:
            row["match"] = "new-or-changed"
        elif len(candidates) != 1 or current_counts[key] != 1:
            row["match"] = "ambiguous"
        else:
            old_source = refresh.get("source_blobs", {}).get(row["file"], {})
            unchanged = (old_source.get("sha256") == sources[row["file"]]
                         and old_source.get("git_blob") == candidates[0]["source_blob"])
            row["match"] = "source-identical" if unchanged else "expression-only"

    primitive = [row for row in calls if row["site"] is not None]
    supplemental = [row for row in calls if row["site"] is None]
    sites = sorted({row["site"] for row in primitive})
    ratcheted = [site for site in sites
                if site.split(":", 1)[1].split("::", 1)[0] not in inventory.DIGEST_OWNERS]
    return {
        "schema": "prismaquant.digest-reconciliation.v1",
        "acceptance": "observations-only",
        "historical_source_commit": refresh["source_commit"],
        "historical_refresh_sha256": DIRECT_UTF8_INDENT2_STRICT.sha256(refresh),
        "scanner_sha256": bytes_sha256hex(Path(inventory.__file__).read_bytes()),
        "source_sha256": sources,
        "primitive_sites": sites,
        "ratcheted_sites": ratcheted,
        "calls": primitive,
        "supplemental_calls": supplemental,
        "not_observed_historical_ids": sorted(ids - observed_ids),
        "summary": {
            "primitive_calls": len(primitive), "primitive_scopes": len(sites),
            "ratcheted_scopes": len(ratcheted), "supplemental_calls": len(supplemental),
            "historical_calls": len(historical),
            "not_observed_historical_calls": len(ids - observed_ids),
            "matches": dict(sorted(Counter(row["match"] for row in calls).items())),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--census", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--output", type=Path)
    action.add_argument("--check", type=Path)
    args = parser.parse_args(argv)
    census = args.census or args.root / CENSUS
    if args.output and args.output.resolve() == census.resolve():
        parser.error("output must not replace the historical census")
    document = json.loads(census.read_text(encoding="utf-8"))
    report = reconcile(args.root, document["refresh_2540"])
    if args.check:
        saved = json.loads(args.check.read_text(encoding="utf-8"))
        if saved != report:
            print("digest reconciliation differs; regenerate and review observations", file=sys.stderr)
            return 1
    else:
        args.output.write_text(DIRECT_UTF8_INDENT2_STRICT.text(report) + "\n", encoding="utf-8")
    print(DIRECT_UTF8_INDENT2_STRICT.text(report["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
