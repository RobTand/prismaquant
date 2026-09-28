#!/usr/bin/env python3
"""Write the GLM-5.3 route-trace fixtures for PQ #1490 from the measured U4 BAL serves.

Every trace here is what Tessera wrote on one rank, TRIMMED: the header fields
the gate does not read (``note``, ``pid``, ``started_utc``, ``flushed_utc``,
``flushes``) are dropped, and every entry and identity header field is kept.

* ``measured/mtp-r6-rank*.json`` -- the MTP k=1 serve on Tessera f18f08b5
  (with #680), run ``u4-BAL-20260928T0540Z-2c-r6-2c``. Every module is named.
* ``measured/mtp-r5-rank*.json`` -- the same serve before #680 (r5). The 29
  NVFP4 routed stacks are unnamed in every entry.
* ``measured/tr3-rank*.json`` -- the non-speculative TR3 serve, also pre-#680.
* ``named/tr3-rank*.json`` -- SYNTHETIC. The TR3 traces with the 29 NVFP4
  routed stacks named: the BAL ``config.json``'s priced targets for that
  entry's family and kind, mapped through the ``glm5_next`` profile
  (``served_module_name``). ``modules`` and ``launches`` keep their measured
  values and the two unnamed counts go to zero. It exists because no
  post-#680 non-speculative serve has been traced; r6 showed this naming rule
  gives exactly the names the serve writes (see ``SOURCE.md``).

``config.json`` is the BAL export's, trimmed to what the gate and the profile
read: ``model_type``, ``architectures``, the ``text_config`` layer counts, and
the whole ``quantization_config``.

    python tests/fixtures/tessera_route_trace_1490/generate.py \\
      --config /mnt/shared/tessera-runs/moe/glm53-pact-balanced-20260928/body-mtp/exported/config.json \\
      --mtp-r6-rank0 .../BAL-2c-r6/run/2c-evidence/route-trace-rank0.json \\
      --mtp-r6-rank1 .../BAL-2c-r6/run/2c-evidence/route-trace-rank1.json \\
      --mtp-r5-rank0 .../BAL-2c-r5/run/2c-evidence/route-trace-rank0.json \\
      --mtp-r5-rank1 .../BAL-2c-r5/run/2c-evidence/route-trace-rank1.json \\
      --tr3-rank0 .../u4-BAL-20260928T0540Z/head/route/tr3-rank0.json \\
      --tr3-rank1 .../BAL/run/route/tr3-rank1.json \\
      --out-dir tests/fixtures/tessera_route_trace_1490

Run from the repository root; it imports ``prismaquant.model_profiles``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

#: Header fields the gate never reads; everything else in the header is kept.
DROPPED_HEADER_FIELDS = ("note", "pid", "started_utc", "flushed_utc", "flushes")

#: The ``text_config`` fields the profile reads for the MTP draft range.
TEXT_CONFIG_FIELDS = ("model_type", "num_hidden_layers", "num_nextn_predict_layers")

#: The trace's ``kind`` for a priced ``scheme.structure``.
KIND = {"dense": "dense", "routed_moe": "moe"}


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def trim_config(config: dict) -> dict:
    text = config["text_config"]
    return {
        "architectures": config["architectures"],
        "model_type": config["model_type"],
        "text_config": {field: text[field] for field in TEXT_CONFIG_FIELDS},
        "quantization_config": config["quantization_config"],
    }


def trim_trace(trace: dict) -> dict:
    return {key: value for key, value in trace.items() if key not in DROPPED_HEADER_FIELDS}


def served_owners_by_key(config: dict) -> dict[tuple[str, str], list[str]]:
    """``{(family, kind): sorted served names}`` for every priced target."""
    from prismaquant.model_profiles.registry import profile_from_config

    profile = profile_from_config(config)
    if profile.name != "glm5_next":
        raise SystemExit(f"config resolves the {profile.name} profile, not glm5_next")
    out: dict[tuple[str, str], list[str]] = {}
    for group in config["quantization_config"]["config_groups"].values():
        scheme = group["scheme"]
        key = (scheme["family"], KIND[scheme.get("structure", "dense")])
        for target in group["targets"]:
            out.setdefault(key, []).append(profile.served_module_name(target, config))
    return {key: sorted(names) for key, names in out.items()}


def name_unnamed(trace: dict, owners: dict[tuple[str, str], list[str]]) -> tuple[dict, int]:
    """Fill each unnamed entry with the priced modules its M has not named yet."""
    named = json.loads(json.dumps(trace))
    filled = 0
    by_m: dict[str, set[str]] = {}
    for entry in named["entries"]:
        m = entry["shape"].split(":")[0]
        by_m.setdefault(m, set()).update(entry["module_names"])
    for entry in named["entries"]:
        if not entry["unnamed_modules"]:
            continue
        if entry["module_names"]:
            raise SystemExit(f"{entry['shape']}: an entry naming some modules and not others")
        m = entry["shape"].split(":")[0]
        family = entry["policy"].split(":")[0]
        candidates = [name for name in owners.get((family, entry["kind"]), ())
                      if name not in by_m[m]]
        if len(candidates) != entry["unnamed_modules"] or len(candidates) != entry["modules"]:
            raise SystemExit(
                f"{m} {family}/{entry['kind']}: {entry['unnamed_modules']} unnamed "
                f"module(s) but {len(candidates)} priced module(s) left unnamed at this M")
        entry["module_names"] = candidates
        entry["unnamed_modules"] = 0
        entry["dispatches_without_prefix"] = 0
        by_m[m].update(candidates)
        filled += len(candidates)
    return named, filled


#: ``(serve, fill in synthetic names)``: only TR3 has no post-#680 trace.
SERVES = (("mtp-r6", False), ("mtp-r5", False), ("tr3", True))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    for serve, _named in SERVES:
        for rank in (0, 1):
            parser.add_argument(f"--{serve}-rank{rank}", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    sys.path.insert(0, str(Path.cwd()))

    out = Path(args.out_dir)
    config = trim_config(json.loads(Path(args.config).read_text()))
    _dump(out / "config.json", config)
    owners = served_owners_by_key(config)
    sources = {}
    for serve, fill in SERVES:
        for rank in (0, 1):
            path = Path(getattr(args, f"{serve.replace('-', '_')}_rank{rank}"))
            raw = path.read_bytes()
            trace = json.loads(raw)
            if trace.get("rank") != rank:
                raise SystemExit(f"{path}: header rank {trace.get('rank')!r}, not {rank}")
            name = f"{serve}-rank{rank}.json"
            measured = trim_trace(trace)
            _dump(out / "measured" / name, measured)
            row = {"source": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                   "entries": len(trace["entries"]),
                   "unnamed_modules": sum(e["unnamed_modules"] for e in trace["entries"])}
            if fill:
                named, filled = name_unnamed(measured, owners)
                _dump(out / "named" / name, named)
                row["synthetic_names_filled"] = filled
            sources[name] = row
            print(f"{name}: {row}")
    _dump(out / "sources.json", {
        "config_source": args.config,
        "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
        "dropped_header_fields": list(DROPPED_HEADER_FIELDS),
        "runs": {
            "mtp-r6": "u4-BAL-20260928T0540Z-2c-r6-2c (Tessera f18f08b5, census receipt VALID)",
            "mtp-r5": "u4-BAL-20260928T0540Z-2c-r5-2c (pre-#680; its census REFUSED)",
            "tr3": "u4-BAL-20260928T0540Z TR3 phase (pre-#680)",
        },
        "traces": sources,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
