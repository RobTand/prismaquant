#!/usr/bin/env python3
"""Build one row-0055 measurement arm from the census row that priced it.

Reads the planned GLM census row ``row-0055`` (manifest.rest-c92826fa4), keeps
its container image, mounts, environment and campaign argv, and changes only
what an arm must own:

* ``--out``, ``--cache-dir`` and ``--checkpoint`` move into ``--out``, so the
  census workspace is read and never written;
* the PrismaQuant tree the container pins moves to ``--source``, a ``git
  archive`` of the commit under measurement, mounted read-only exactly the way
  the census pinned its own tree;
* ``PRISMAQUANT_TMPDIR`` moves into ``--out``;
* the arm's own flags (after ``--``) are appended to the campaign argv.

Writes ``spec.json`` (the container spec) and ``argv.json`` (the campaign argv)
into ``--out``. ``entry.py`` reads both.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MANIFEST = Path(
    "/mnt/shared/tessera-measurements/glm-canonical-census-20260908/"
    "activation-runtime-allocation-20260911/extension-e2m1-01/workspace/"
    "manifest.rest-c92826fa4.dm.json")
ROW = "row-0055"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("extra", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra = list(args.extra)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    source = args.source.resolve()
    if not (source / "prismaquant" / "__init__.py").is_file():
        raise SystemExit(f"{source} holds no prismaquant package")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = json.loads(MANIFEST.read_text())
    matches = [row for row in rows
               if any(str(value).endswith(f"/units/{ROW}.json") for value in row["argv"])]
    if len(matches) != 1:
        raise SystemExit(f"expected one {ROW} in {MANIFEST}, found {len(matches)}")
    row = matches[0]
    command = list(row["argv"])
    spec = json.loads(command[command.index("--spec") + 1])
    last_module = len(command) - 1 - command[::-1].index("-m")
    if command[last_module + 1] != "prismaquant.tessera_campaign":
        raise SystemExit("row does not run prismaquant.tessera_campaign")
    campaign = command[last_module + 2:]

    locations = {"--out": out / "cost.pkl", "--cache-dir": out / "cache",
                 "--checkpoint": out / "cost.anchors.json"}
    for flag, value in locations.items():
        campaign[campaign.index(flag) + 1] = str(value)

    pinned = [mount for mount in spec["container"]["mounts"]
              if "/pq-source-" in mount["source"]]
    if len(pinned) != 1:
        raise SystemExit("row spec does not pin exactly one PrismaQuant tree")
    old = pinned[0]["source"]
    pinned[0].update(source=str(source), target=str(source), readonly=True)
    env = spec["env"]
    entries = env["PYTHONPATH"].split(":")
    if entries[0] != old:
        raise SystemExit("row PYTHONPATH does not name its pinned tree first")
    env["PYTHONPATH"] = ":".join([str(source), *entries[1:]])
    env["PRISMAQUANT_TMPDIR"] = str(out / "staging")
    (out / "staging").mkdir(exist_ok=True)

    campaign += extra
    (out / "spec.json").write_text(json.dumps(spec, sort_keys=True) + "\n")
    (out / "argv.json").write_text(json.dumps(campaign, indent=1) + "\n")
    (out / "arm.json").write_text(json.dumps({
        "row": ROW, "manifest": str(MANIFEST), "source": str(source),
        "replaced_source": old, "extra": extra,
        "demand": row["demand"], "tags": row["tags"]}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
