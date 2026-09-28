#!/usr/bin/env python3
"""Write the small A8 route-trace fixture for PQ #1620 from the measured U4 A8 serve.

The A8 serve (uniform Tessera-8, body E4M3, Tessera v39) traced 132 named
modules on every rank. That is too many to read in a test, so this cuts the
real traces down and keeps the real key shapes:

* Each entry keeps its first ``KEEP`` module names in the trace's own (sorted)
  order and nothing else about the entry changes, except ``modules`` (the
  number kept) and ``launches`` (the number kept times the entry's measured
  launches per module, which is constant per entry: 1 at M1, M2 and M2049,
  25 at M2048). An entry that already names ``KEEP`` or fewer modules is kept
  whole.
* The header drops the fields the gate never reads (``note``, ``pid``,
  ``started_utc``, ``flushed_utc``, ``flushes``); every other header field
  and every entry field is the serve's.
* ``config.json`` is the A8 export's ``config.json`` trimmed to what the gate
  and the profile read: ``architectures``, ``model_type``, the ``text_config``
  layer counts, and ``quantization_config`` with its ``config_groups`` cut to
  the groups whose target is served by a kept module, and its ``ignore`` list
  dropped (the gate does not read it). The one MTP draft-layer group is
  dropped too: the A8 serve is not speculative, so the draft layer is never
  dispatched and the full price refuses on it by design (PQ #1490 covers
  that); this fixture prices what the serve dispatches so its baseline agrees.

The fixture cuts the TR3 serve, ranks 0 and 1.

    python tests/fixtures/tessera_route_trace_1620/generate.py \\
      --config /mnt/shared/tessera-runs/moe/glm53-pact-uniform-arms-20260927/a8/body-mtp-v39/exported-r2/config.json \\
      --tr3-rank0 .../u4-A8-20260928T1809Z/head/route/tr3-rank0.json \\
      --tr3-rank1 .../u4-A8-20260928T1809Z/route/tr3-rank1.json \\
      --out-dir tests/fixtures/tessera_route_trace_1620

Run from the repository root; it imports ``prismaquant.model_profiles``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

#: Modules kept per entry.
KEEP = 4

#: Header fields the gate never reads; everything else in the header is kept.
DROPPED_HEADER_FIELDS = ("note", "pid", "started_utc", "flushed_utc", "flushes")

#: The ``text_config`` fields the profile reads for the MTP draft range.
TEXT_CONFIG_FIELDS = ("model_type", "num_hidden_layers", "num_nextn_predict_layers")


def _dump(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cut_entry(entry: dict) -> dict:
    names = entry["module_names"]
    if entry["unnamed_modules"] or entry["dispatches_without_prefix"]:
        raise SystemExit(f"{entry['shape']}: an unnamed dispatch; this cut reads named traces")
    if entry["modules"] != len(names):
        raise SystemExit(f"{entry['shape']}: modules != len(module_names)")
    if len(names) <= KEEP:
        return json.loads(json.dumps(entry))
    if entry["launches"] % entry["modules"]:
        raise SystemExit(f"{entry['shape']}: launches is not a whole number per module")
    per_module = entry["launches"] // entry["modules"]
    cut = dict(entry)
    cut["module_names"] = list(names[:KEEP])
    cut["modules"] = KEEP
    cut["launches"] = KEEP * per_module
    return cut


def cut_trace(trace: dict) -> dict:
    out = {key: value for key, value in trace.items() if key not in DROPPED_HEADER_FIELDS}
    out["entries"] = [cut_entry(entry) for entry in trace["entries"]]
    return out


def trim_config(config: dict, kept_served: set[str]) -> tuple[dict, int]:
    """The trimmed config, and how many priced groups the cut dropped."""
    from prismaquant.model_profiles.registry import profile_from_config

    profile = profile_from_config(config)
    if profile.name != "glm5_next":
        raise SystemExit(f"config resolves the {profile.name} profile, not glm5_next")
    quantization = dict(config["quantization_config"])
    quantization.pop("ignore", None)
    groups, dropped = {}, 0
    for name, group in quantization["config_groups"].items():
        served = {profile.served_module_name(target, config) for target in group["targets"]}
        if served <= kept_served:
            groups[name] = group
        else:
            dropped += 1
    quantization["config_groups"] = groups
    text = config["text_config"]
    return {
        "architectures": config["architectures"],
        "model_type": config["model_type"],
        "text_config": {field: text[field] for field in TEXT_CONFIG_FIELDS},
        "quantization_config": quantization,
    }, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tr3-rank0", type=Path, required=True)
    parser.add_argument("--tr3-rank1", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path.cwd()))

    sources = {"config_source": str(args.config), "config_sha256": _sha256(args.config),
               "keep_per_entry": KEEP, "dropped_header_fields": list(DROPPED_HEADER_FIELDS),
               "traces": {}}
    kept: set[str] = set()
    for rank, path in enumerate((args.tr3_rank0, args.tr3_rank1)):
        cut = cut_trace(json.loads(path.read_text()))
        for entry in cut["entries"]:
            kept.update(entry["module_names"])
        name = f"tr3-rank{rank}.json"
        _dump(args.out_dir / name, cut)
        sources["traces"][name] = {
            "source": str(path), "sha256": _sha256(path),
            "entries": len(cut["entries"]),
            "modules_per_entry": [entry["modules"] for entry in cut["entries"]],
        }
    config, dropped = trim_config(json.loads(args.config.read_text()), kept)
    sources["config_groups_kept"] = len(config["quantization_config"]["config_groups"])
    sources["config_groups_dropped"] = dropped
    _dump(args.out_dir / "config.json", config)
    _dump(args.out_dir / "sources.json", sources)
    print(json.dumps({key: sources[key] for key in
                      ("config_groups_kept", "config_groups_dropped")}))


if __name__ == "__main__":
    main()
