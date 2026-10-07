"""Read real trial inputs and existing profiles; make no quality claim.

Research-only intake for eng-pq-fine-grained. Run through PrismaBuild on x86.
Table flags are reported, not substituted for Tessera's canonical admission.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import socket


def own_file(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest()}


def inspect_table(path: Path) -> dict:
    table = json.loads(path.read_bytes())
    rows = table["rungs"]
    rungs = [row["rung"] for row in rows]
    if len(rungs) != len(set(rungs)):
        raise ValueError("duplicate rungs")
    flag_eligible = [row["rung"] for row in rows
                     if row["measurement_status"] == "measured"
                     and row["supported"] is True
                     and not row["excluded"] and not row["anomaly_flags"]]
    return {**own_file(path), "table_version": table["table_version"],
            "format": table["format"], "kernel_build": table["kernel_build"],
            "scope": table["scope"],
            "status_counts": dict(Counter(row["measurement_status"] for row in rows)),
            "flag_eligible_rungs": flag_eligible,
            "baseline_1024_has_eligible_flags": 1024 in flag_eligible,
            "eligible_below_1024": [r for r in flag_eligible if r < 1024],
            "eligible_above_1024": [r for r in flag_eligible if r > 1024],
            "admission_claim": False,
            "note": "Flags only. Use canonical validate_table/admit_rung before allocating."}


def inspect_profile(root: Path) -> dict:
    details = root / "details.csv"
    metrics = {}
    with details.open(newline="") as stream:
        for row in csv.DictReader(stream):
            kernel_id = row["ID"]
            kernel = metrics.setdefault(kernel_id, {
                "name": row["Kernel Name"], "grid": row["Grid Size"],
                "block": row["Block Size"], "metrics": []})
            name = row["Metric Name"]
            if any(part in name.lower() for part in (
                    "duration", "stall", "barrier", "register", "spill",
                    "shared memory", "long scoreboard", "short scoreboard")):
                kernel["metrics"].append({"name": name, "unit": row["Metric Unit"],
                                           "value": row["Metric Value"]})
    sources = []
    for number in range(1, 5):
        path = root / f"source-{number}.csv"
        with path.open(newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            name = header[1]
            columns = next(reader)
            samples = Counter()
            instructions = Counter()
            for fields in reader:
                row = dict(zip(columns, fields))
                if not row.get("Address", "").startswith("0x"):
                    continue  # SASS only; do not double-count source-line totals.
                match = re.match(r"\s*(?:@!?U?P(?:\d+|T)\s+)?([A-Z][A-Z0-9]*(?:\.[A-Z0-9]+)*)", row["Source"])
                if not match:
                    raise ValueError(f"cannot parse SASS: {row['Source']}")
                opcode = match.group(1).split(".")[0]
                samples[opcode] += int(float(row["# Samples"].replace(",", "")))
                instructions[opcode] += int(float(row["Instructions Executed"].replace(",", "")))
        sources.append({**own_file(path), "kernel": name,
                        "sass_samples_by_opcode": dict(samples),
                        "sass_instructions_by_opcode": dict(instructions),
                        "samples_are_not_wall_time": True})
    return {"details": own_file(details), "kernels": metrics, "source": sources}


def inspect_inputs(source: Path, draw: Path) -> dict:
    import torch
    from safetensors import safe_open

    torch.set_num_threads(1)
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_bytes())["weight_map"]
    inspected = []
    for layer in (3, 22, 44):
        name = f"model.language_model.layers.{layer}.mlp.experts.0.up_proj.weight"
        with safe_open(str(source / index[name]), framework="pt", device="cpu") as handle:
            view = handle.get_slice(name)
            sample = view[:2, :2]
            inspected.append({"name": name, "file": index[name], "shape": view.get_shape(),
                              "dtype": str(sample.dtype), "sample": sample.float().tolist()})
    name = "model.language_model.layers.0.mlp.up_proj.weight"
    with safe_open(str(source / index[name]), framework="pt", device="cpu") as handle:
        view = handle.get_slice(name)
        sample = view[:2, :2]
        inspected.append({"name": name, "file": index[name], "shape": view.get_shape(),
                          "dtype": str(sample.dtype), "sample": sample.float().tolist()})
    with safe_open(str(draw), framework="pt", device="cpu") as handle:
        token_tensors = []
        for key in handle.keys():
            view = handle.get_slice(key)
            sample = view[:1, :8]
            token_tensors.append({"key": key, "shape": view.get_shape(),
                                  "dtype": str(sample.dtype), "sample": sample.tolist()})
    return {"model_index": own_file(index_path), "weight_slices": inspected,
            "draw": own_file(draw), "token_tensors": token_tensors,
            "torch": torch.__version__, "cuda_visible": torch.cuda.is_available(),
            "captures_created": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--draw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {"schema": "prismaquant.block_feasibility_intake.v1",
              "host": socket.gethostname(), "python": platform.python_version(),
              "table": inspect_table(args.table),
              "profile": inspect_profile(args.profile_root),
              "inputs": inspect_inputs(args.source, args.draw),
              "quality_measured": False, "gpu_d38_approval": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "table_version": result["table"]["table_version"],
                      "rung_status_counts": result["table"]["status_counts"],
                      "flag_eligible_rungs": result["table"]["flag_eligible_rungs"],
                      "profile_kernels": len(result["profile"]["kernels"]),
                      "weight_shapes": [r["shape"] for r in result["inputs"]["weight_slices"]],
                      "quality_measured": False}), flush=True)


if __name__ == "__main__":
    main()
