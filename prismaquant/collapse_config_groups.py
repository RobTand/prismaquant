"""Rewrite an exported compressed-tensors `config.json` so per-expert
MoE Linear regex targets are collapsed from 1-per-expert into compact
per-(layer, projection) regexes.

Why: without this, vLLM's `find_matched_target` walks tens of thousands
of regex entries per Linear init, thrashing Python's re-compile cache
(bounded at ~512 distinct patterns) and stalling engine startup for
tens of minutes on large MoE artifacts. Collapsing shrinks each
format's target list to a few hundred entries and startup completes
in seconds.

Idempotent: running again on an already-collapsed config is a no-op.

Usage:
    python3 -m prismaquant.collapse_config_groups /path/to/exported_model

This is both the retroactive fix for previously-uploaded HF artifacts
and the runtime formatter used by the exporter itself — the
`_build_target_list` implementation lives in export_native_compressed
and is shared."""
from __future__ import annotations
import json
import sys
from pathlib import Path


def collapse(config_path: str | Path) -> dict:
    """Rewrite `config.json` in place. Returns a before/after summary."""
    from .export_native_compressed import _build_target_list, _explicit_regex

    p = Path(config_path)
    if p.is_dir():
        p = p / "config.json"
    c = json.loads(p.read_text())
    qc = c.get("quantization_config")
    if not qc:
        return {"error": "no quantization_config"}
    groups = qc.get("config_groups", {})
    summary: dict = {"before": {}, "after": {}, "total_before": 0,
                     "total_after": 0}
    for gname, g in groups.items():
        tgts = g.get("targets", [])
        summary["before"][gname] = len(tgts)
        summary["total_before"] += len(tgts)
        # Extract the un-anchored vLLM names (strip re:^...$)
        unpacked: list[str] = []
        passthrough: list[str] = []
        for t in tgts:
            if t.startswith("re:^") and t.endswith("$"):
                body = t[len("re:^"):-1]
                # If the body is a simple un-dotted name (no metachars
                # except [.] and $), reverse to the plain dotted form.
                if not any(ch in body for ch in "+?*|()"):
                    plain = body.replace("[.]", ".")
                    unpacked.append(plain)
                else:
                    # Already a real regex (e.g. catch-all). Keep as-is.
                    passthrough.append(t)
            else:
                passthrough.append(t)
        new_tgts = passthrough + _build_target_list(unpacked)
        g["targets"] = new_tgts
        summary["after"][gname] = len(new_tgts)
        summary["total_after"] += len(new_tgts)
    p.write_text(json.dumps(c, indent=2))
    return summary


def main():
    if len(sys.argv) != 2:
        print("usage: collapse_config_groups.py <model_dir_or_config.json>")
        sys.exit(2)
    summary = collapse(sys.argv[1])
    for gname, before in summary.get("before", {}).items():
        after = summary["after"][gname]
        if before != after:
            print(f"  {gname}: {before} -> {after} targets")
    print(f"TOTAL: {summary.get('total_before')} -> "
          f"{summary.get('total_after')}")


if __name__ == "__main__":
    main()
