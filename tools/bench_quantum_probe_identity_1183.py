"""PQ #1183: a quantum's per-row probe-identity work, main's calls against the fix.

Loads one real GLM-5.3 layer-044 unit checkpoint (v7, PB 2063ab925db1) and
replays, for UNITS copies of that unit's 7 rows, the calls a quantum makes:

* the operator digest (``_record_joint_operator``),
* ``make_joint_aura_entry`` (``commit_streamed_units``),
* ``validate_joint_aura_entry`` on the finished row (the payload's final loop).

Arm ``main`` passes the plain probe identity everywhere, as main does. Arm
``fix`` does what the fixed quantum does: it builds ``validated_probe_identity``
once, builds and checks every row through it, stores the ordinary dict in
the row, and hashes the ordinary dict once at the end. Both arms also take
the provenance digest once. Both run in one process on the same bytes;
cProfile gives each arm's top functions. The two arms' rows must pickle to
identical bytes.
"""
import cProfile
import io
import json
import pickle
import pstats
import sys
import time
from pathlib import Path

from prismaquant.aura_cost import _aura_unit_checkpoint_path
from prismaquant.joint_aura import (
    identity_sha256, make_joint_aura_entry, validate_joint_aura_entry,
    validated_probe_identity)

CHECKPOINTS = Path("/mnt/shared/tessera-measurements/glm-campaign-takeover-20260913/"
                   "r13-stageb-20260923/a4/overlay/layer-quanta/layer-044/checkpoints")
QNAME = "model.language_model.layers.44.mlp.experts.0.down_proj"
UNITS = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def load_rows():
    envelope = pickle.loads(_aura_unit_checkpoint_path(CHECKPOINTS, QNAME).read_bytes())
    state = pickle.loads(envelope["payload"])
    return state["joint_aura_rows"]


def arm_main(rows, plain):
    made = []
    for _ in range(UNITS):
        for fmt, row in rows.items():
            operator = dict(row["joint_operator_identity"])
            operator["probe_identity_sha256"] = identity_sha256(plain)
            made.append(make_joint_aura_entry(
                operator_identity=operator, probe_identity=plain,
                signed_components=row["signed_components_per_probe"]))
    identity_sha256(plain)  # the provenance digest
    for row in made:
        assert validate_joint_aura_entry(row)
    return made


def arm_fix(rows, plain):
    probe = validated_probe_identity(plain)
    made = []
    for _ in range(UNITS):
        for fmt, row in rows.items():
            operator = dict(row["joint_operator_identity"])
            operator["probe_identity_sha256"] = identity_sha256(probe)
            entry = make_joint_aura_entry(
                operator_identity=operator, probe_identity=probe,
                signed_components=row["signed_components_per_probe"])
            entry["probe_identity"] = plain
            made.append(entry)
    assert identity_sha256(plain) == identity_sha256(probe)  # provenance + end check
    for row in made:
        checked = {**row, "probe_identity": probe} if row["probe_identity"] is plain else row
        assert validate_joint_aura_entry(checked)
    return made


def profiled(label, rows, plain, arm):
    profile = cProfile.Profile()
    started = time.perf_counter()
    profile.enable()
    made = arm(rows, plain)
    profile.disable()
    wall = time.perf_counter() - started
    text = io.StringIO()
    pstats.Stats(profile, stream=text).sort_stats("cumulative").print_stats(12)
    return {"arm": label, "rows": len(made), "wall_s": round(wall, 3),
            "per_row_ms": round(1000 * wall / len(made), 3),
            "top": text.getvalue().splitlines()[:40]}, made


def row_bytes(rows):
    return pickle.dumps(rows, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    rows = load_rows()
    plain = next(iter(rows.values()))["probe_identity"]
    size = len(json.dumps(plain, sort_keys=True, separators=(",", ":")))
    before, made_before = profiled("main", rows, plain, arm_main)
    after, made_after = profiled("fix", rows, plain, arm_fix)
    same = row_bytes(made_before) == row_bytes(made_after)
    print(json.dumps({"qname": QNAME, "formats": len(rows), "units": UNITS,
                      "probe_identity_json_bytes": size,
                      "rows_pickle_byte_identical": same,
                      "main": {k: v for k, v in before.items() if k != "top"},
                      "fix": {k: v for k, v in after.items() if k != "top"},
                      "speedup": round(before["wall_s"] / max(after["wall_s"], 1e-9), 1)},
                     indent=1))
    for result in (before, after):
        print(f"--- cProfile, arm {result['arm']} (cumulative) ---")
        print("\n".join(result["top"]))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
