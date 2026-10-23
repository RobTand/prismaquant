#!/usr/bin/env python3
"""Report import and CLI reachability for proposed module removals."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (
    "layer_state_cache", "weight_session", "artifact_collection",
    "artifact_collection_legacy", "proposal_validation", "quality_prefill_screen",
    "tessera_full_domain_ledger", "quality_prefill_coverage", "fisher_col_weights",
    "per_row_pricing", "production_cache_stripes", "speed_quality_frontier",
    "shard_layout", "shard_commit_progress", "index_entropy",
    "tessera_reduced_schedule_repair", "sensitivity_coverage", "stage_a_local_spool",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modules", nargs="*", default=CANDIDATES)
    args = parser.parse_args()
    imports = {name: set() for name in args.modules}
    references = {name: set() for name in args.modules}
    cli = {}
    for directory in ("prismaquant", "tools", "experiments", "tests"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            if any(part in {"archive", "vendored", "__pycache__", "_vendored"} for part in path.parts):
                continue
            if path.resolve() == Path(__file__).resolve():
                continue
            source = path.read_text()
            relevant = [name for name in args.modules if name in source or path.stem == name]
            if not relevant:
                continue
            tree = ast.parse(source, filename=str(path))
            relative = str(path.relative_to(ROOT))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    targets = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    targets = [module, *(module + "." + alias.name for alias in node.names)]
                else:
                    targets = []
                for name in relevant:
                    if any(target == name or target.endswith("." + name) for target in targets):
                        imports[name].add(relative)
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    for name in relevant:
                        if "prismaquant." + name in node.value:
                            references[name].add(relative)
            if directory == "prismaquant" and path.stem in imports:
                cli[path.stem] = any(
                    isinstance(node, ast.If) and "__name__" in ast.unparse(node.test)
                    and "__main__" in ast.unparse(node.test) for node in tree.body
                )
    rows = []
    for name in args.modules:
        path = ROOT / "prismaquant" / (name + ".py")
        rows.append({"module": name, "exists": path.is_file(),
                     "lines": len(path.read_text().splitlines()) if path.is_file() else 0,
                     "cli": cli.get(name, False), "imports": sorted(imports[name]),
                     "string_references": sorted(references[name])})
    print(json.dumps({"schema": "prismaquant.refactor_reachability.v1", "modules": rows}, indent=2))


if __name__ == "__main__":
    main()
