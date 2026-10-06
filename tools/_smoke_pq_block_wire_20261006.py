"""Throwaway detached CPU smoke for tools/pq_block_reference_wire.py.

Runs on the x86 CPU tier under the qualified b40c93cb interpreter with real
A8S export bytes.  This file is NOT a permanent test suite: it is deleted from
the worktree after the run; the sealed action snapshot keeps its copy.

Checks, on two distinct real expert units of one shape (never a duplicated
parent standing in for a mixed blob):
  1. body_costs formula and all-zeros selection total
  2. fixed_bytes selection-independence and accounting identities
  3. single-parent round trip == stock_dequant exactly (all row boundaries)
  4. mixed-parent round trip == blockwise stitch of the two stock decodes
  5. grouped packer byte-equivalence with tessera.wire.pack_body per fragment
  6. boundary states == replay_window over the parent's own prefix
  7. overflow refusal; explicit all-zero padding charged; decode still exact
  8. refusals: truncation, bit flip, misdeclared geometry, bad rate, dirty pad
  9. alternative row sizes 256/512 and block_cols 4: accounting + round trip
 10. blob written to /tmp scratch and decoded from the file bytes
"""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from tessera.decode import replay_window
from tessera.fused import parse_fused
from tessera.manifest import BodyKind, ScalePlaneKind
from tessera.stock import materialize_stock, stock_dequant
from tessera.unit_artifact import parse_unit_artifact
from tessera.wire import pack_body

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pq_block_reference_wire as W

STARTED = time.monotonic()
RESULTS: dict = {}
SCRATCH = Path("/tmp") / f"pq-block-wire-smoke-{time.strftime('%Y%m%dT%H%M%S')}"
SCRATCH.mkdir(parents=True, exist_ok=True)


def check(name, fn):
    started = time.monotonic()
    fn()
    RESULTS[name] = {"pass": True, "seconds": round(time.monotonic() - started, 3)}
    print(f"PASS {name} ({RESULTS[name]['seconds']}s)", flush=True)


# ---------------------------------------------------------------- environment
def env():
    import tessera
    assert "pq-pbdc4803-tessera-b40c93cb" in sys.executable, sys.executable
    RESULTS["env"] = {
        "python": sys.executable,
        "tessera": tessera.__file__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "tmpdir_scratch": str(SCRATCH),
        "tmp_free_gib": round(shutil.disk_usage("/tmp").free / 2**30, 1),
    }
    print(json.dumps(RESULTS["env"]), flush=True)


# ------------------------------------------------------------------- parents
EXPORT = Path(
    "/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported"
)
KEYS = [
    "model.language_model.layers.3.mlp.experts.0.up_proj.wire",
    "model.language_model.layers.3.mlp.experts.1.up_proj.wire",
]


def load_parent(key):
    index = json.loads((EXPORT / "model.safetensors.index.json").read_text())
    shard = EXPORT / index["weight_map"][key]
    with safe_open(str(shard), framework="pt", device="cpu") as handle:
        stream = handle.get_tensor(key)
    if stream.dtype != torch.uint8 or stream.ndim != 1:
        raise SystemExit(f"{key}: baseline wire must be one byte stream")
    blob = stream.numpy().tobytes()
    role = key.removesuffix(".wire").rsplit(".", 1)[1]
    members = [m for m in parse_fused(blob) if m.name == role]
    if len(members) != 1:
        raise SystemExit(f"{key}: expected one {role} member")
    member = members[0]
    parsed = parse_unit_artifact(member.blob, device="cpu")
    unit = parsed.unit
    if parsed.grid.name != "E4M3" or parsed.body is not BodyKind.WINDOW \
            or unit.scale_plane is not ScalePlaneKind.CHANNEL or unit.span != 1:
        raise SystemExit(f"{key}: not an untransformed E4M3 WINDOW span1 CHANNEL unit")
    return parsed, {
        "key": key,
        "shard": index["weight_map"][key],
        "member_sha256": hashlib.sha256(member.blob).hexdigest(),
        "rows": int(member.rows),
        "rates": sorted(set(int(r) for r in unit.rates)),
        "window_bits": int(unit.window_bits),
    }


def parents():
    global P0, P1, REF0, REF1, ROWS, COLS, PARENTS, BR, BC, NRB, NCB, MIX
    P0, meta0 = load_parent(KEYS[0])
    P1, meta1 = load_parent(KEYS[1])
    RESULTS["parents"] = [meta0, meta1]
    print(json.dumps(RESULTS["parents"]), flush=True)
    if (P0.unit.body_bits.shape) != (P1.unit.body_bits.shape):
        raise SystemExit("the two real units disagree in shape")
    ROWS, COLS = P0.unit.body_bits.shape
    PARENTS = [P0, P1]
    REF0 = stock_dequant(materialize_stock(P0.unit, P0.forests, P0.code))
    REF1 = stock_dequant(materialize_stock(P1.unit, P1.forests, P1.code))
    if REF0.shape != REF1.shape or REF0.shape != (ROWS, COLS):
        raise SystemExit("stock decodes disagree in shape")
    BR, BC = 128, 2
    NRB, NCB = ROWS // BR, COLS // BC
    MIX = ((np.add.outer(np.arange(NRB), np.arange(NCB))) % 2).astype(np.int64)


def body_costs_check():
    costs = W.body_costs(PARENTS, BR, BC)
    if costs.dtype != np.int64 or costs.shape != (NRB * NCB, 2):
        raise SystemExit(f"body_costs shape/dtype wrong: {costs.shape} {costs.dtype}")
    rng = np.random.default_rng(2329)
    sample = rng.choice(costs.shape[0], size=48, replace=False)
    for b in sample:
        i, j = divmod(int(b), NCB)
        for pi, parsed in enumerate(PARENTS):
            rates = np.asarray([int(r) for r in parsed.unit.rates])
            expected = (BR * int(rates[j * BC:(j + 1) * BC].sum()) + 7) // 8
            if int(costs[b, pi]) != expected:
                raise SystemExit(f"body_costs[{b},{pi}] {costs[b, pi]} != {expected}")
    RESULTS["body_costs_sample_max"] = int(costs.max())
    global BLOB0, BD0, BLOBM, BDM
    BLOB0, BD0 = W.pack_projection(PARENTS, np.zeros((NRB, NCB), dtype=np.int64), BR, BC)
    if BD0["body_bytes"] != int(costs[:, 0].sum()):
        raise SystemExit("all-zeros selection body total disagrees with body_costs")


# --------------------------------------------------------- 2 fixed accounting
def fixed_check():
    fixed = W.fixed_bytes(PARENTS, BR, BC)
    if fixed != BD0["fixed_bytes"] or BD0["total_bytes"] != fixed + BD0["body_bytes"]:
        raise SystemExit("fixed/body/total accounting disagrees")
    if not BD0["accounting_consistent"] or len(BLOB0) != BD0["total_bytes"]:
        raise SystemExit("breakdown accounting not consistent with the blob")
    global BLOBM, BDM
    BLOBM, BDM = W.pack_projection(PARENTS, MIX, BR, BC)
    if BDM["fixed_bytes"] != fixed:
        raise SystemExit("fixed_bytes depends on the selection")
    sel_flat = MIX.reshape(-1)
    costs = W.body_costs(PARENTS, BR, BC)
    if BDM["body_bytes"] != int(costs[np.arange(costs.shape[0]), sel_flat].sum()):
        raise SystemExit("mixed body total disagrees with per-block body_costs")
    RESULTS["sizes"] = {
        "single_total": BD0["total_bytes"], "single_body": BD0["body_bytes"],
        "fixed": fixed, "state_bytes": BD0["state_bytes"],
        "tag_bytes": BD0["tag_bytes"], "meta_bytes": BD0["meta_bytes"]["total"],
        "distinct_luts": BD0["meta_bytes"]["distinct_luts"],
        "mixed_body": BDM["body_bytes"], "stock_fp8_bytes": ROWS * COLS + ROWS * 4,
    }
    print(json.dumps(RESULTS["sizes"]), flush=True)


# --------------------------------------------- 3 single-parent exact round trip
def single_round_trip():
    decoded = W.decode_projection(BLOB0)
    if decoded.dtype != torch.float32 or tuple(decoded.shape) != (ROWS, COLS):
        raise SystemExit("decode returned the wrong dtype/shape")
    if not torch.equal(decoded, REF0):
        bad = decoded != REF0
        delta = (decoded - REF0).abs()
        block_bad = bad.reshape(NRB, BR, NCB, BC)
        row_block_bad = block_bad.any(dim=(1, 3))
        col_block_bad = block_bad.any(dim=(0, 2))
        rows_bad = bad.any(dim=1)
        idx = bad.nonzero()
        r, c = (int(x) for x in idx[0])
        i, j = r // BR, c // BC
        local_rows_bad = rows_bad.reshape(NRB, BR)
        # stored state vs full-unit replay for the first wrong block
        entry = {
            "body": P0.unit.body_bits.numpy(),
            "rates": np.asarray([int(x) for x in P0.unit.rates]),
            "window_bits": int(P0.unit.window_bits),
            "cols": COLS,
        }
        rate = int(P0.unit.rates[c])
        which = [cc for cc in range(COLS) if int(P0.unit.rates[cc]) == rate]
        full = replay_window(P0.unit.body_bits[:][:, which].long(),
                             int(P0.unit.window_bits), rate)
        stored = W._boundary_states(entry, BR, NRB)
        entering_full = full[i * BR - 1] if i > 0 else None
        stored_i = stored[i - 1][which] if i > 0 else None
        states_agree = (
            "row-block-0 (no state)"
            if i == 0
            else str(bool(torch.equal(entering_full, torch.from_numpy(stored_i))))
        )
        raise SystemExit(
            "single-parent decode != stock_dequant: "
            f"max |delta| {delta.max().item():.10g}, "
            f"wrong positions {int(bad.sum())}/{bad.numel()}, "
            f"first wrong (row {r}, col {c}) -> block (i {i}, j {j}, local row {r % BR}); "
            f"row blocks with errors: {torch.nonzero(row_block_bad).flatten().tolist()}; "
            f"col blocks with errors: {torch.nonzero(col_block_bad).flatten()[:8].tolist()} "
            f"(of {int(col_block_bad.sum())}); "
            f"per-row-block bad rows (first 4 blocks): "
            f"{[int(local_rows_bad[b].sum()) for b in range(min(4, NRB))]}; "
            f"stored state equals full-replay prefix state: {states_agree}"
        )
    RESULTS["single_round_trip"] = "torch.equal against stock_dequant"


# ----------------------------------------------- 4 mixed-parent exact stitch
def mixed_round_trip():
    per_element = torch.from_numpy(
        MIX.repeat(BR, axis=0).repeat(BC, axis=1)
    )
    expected = torch.where(per_element == 0, REF0, REF1)
    decoded = W.decode_projection(BLOBM)
    if not torch.equal(decoded, expected):
        raise SystemExit("mixed-parent decode != blockwise stitch of the two stock decodes")
    RESULTS["mixed_round_trip"] = "torch.equal against the blockwise stitch"


# --------------------------------- 5 grouped packer == wire.pack_body per byte
def packer_equivalence():
    fields = W._HEADER_STRUCT.unpack_from(BLOBM, 0)
    named = dict(zip(W._HEADER_FIELDS, fields[6:]))
    meta_start = W.HEADER_BYTES + named["alphabet_bytes"]
    meta = W._meta_plane(
        BLOBM, meta_start, named["meta_bytes"], named["window_bits"],
        named["num_parents"], named["rows"], named["cols"],
    )
    geometry = {
        "rows": named["rows"], "cols": named["cols"],
        "block_rows": named["block_rows"], "block_cols": named["block_cols"],
        "num_row_blocks": named["num_row_blocks"],
        "num_col_blocks": named["num_col_blocks"],
        "num_parents": named["num_parents"],
    }
    body_start = meta_start + named["meta_bytes"] + named["tag_bytes"] + named["state_bytes"]
    offsets = W._fragment_offsets(meta, MIX, geometry)
    rng = np.random.default_rng(23291)
    blocks = list(rng.choice(NRB * NCB, size=32, replace=False))
    blocks += [i * NCB for i in range(NRB)] + [i * NCB + (NCB - 1) for i in range(NRB)]
    compared = 0
    for b in blocks:
        i, j = divmod(int(b), NCB)
        pi = int(MIX[i, j])
        unit = PARENTS[pi].unit
        rates = tuple(int(r) for r in unit.rates[j * BC:(j + 1) * BC])
        fragment = unit.body_bits[i * BR:(i + 1) * BR, j * BC:(j + 1) * BC].contiguous()
        expected = pack_body(fragment, rates)
        start = body_start + int(offsets[b])
        got = BLOBM[start:start + int(offsets[b + 1] - offsets[b])]
        if got != expected:
            raise SystemExit(f"grouped packer bytes != pack_body at block {b}")
        compared += 1
    RESULTS["packer_equivalence_fragments"] = compared


# ------------------------------------------ 6 boundary states == replay_window
def boundary_states():
    checked = 0
    for i in (1, 2, 8, NRB - 1):
        if not 1 <= i <= NRB - 1:
            continue
        r0 = i * BR
        for pi, parsed in enumerate(PARENTS):
            unit = parsed.unit
            entry = {
                "body": unit.body_bits.numpy(),
                "rates": np.asarray([int(r) for r in unit.rates]),
                "window_bits": int(unit.window_bits),
                "cols": COLS,
            }
            want = W._boundary_states(entry, BR, NRB)[i - 1]
            columns = np.arange(0, COLS, 97)
            for rate in sorted(set(int(r) for r in unit.rates)):
                which = columns[np.asarray([int(unit.rates[c]) for c in columns]) == rate]
                if which.size == 0:
                    continue
                bits = unit.body_bits[:r0][:, which].long()
                got = replay_window(bits, int(unit.window_bits), rate)[-1]
                if not torch.equal(got, torch.from_numpy(want[which])):
                    raise SystemExit(
                        f"boundary state disagrees with replay_window at row {r0}"
                    )
                checked += int(which.size)
    RESULTS["boundary_state_columns_checked"] = checked


# --------------------------------------------------- 7 overflow and padding
def overflow_and_padding():
    global PADBLOB, PADBD
    try:
        W.pack_projection(PARENTS, np.zeros((NRB, NCB), dtype=np.int64), BR, BC,
                          target_bytes=BD0["total_bytes"] - 1)
    except W.BlockWireFormatError as exc:
        RESULTS["overflow_message"] = str(exc)
    else:
        raise SystemExit("overflow was not rejected")
    target = BD0["total_bytes"] + 137
    PADBLOB, PADBD = W.pack_projection(
        PARENTS, np.zeros((NRB, NCB), dtype=np.int64), BR, BC, target_bytes=target
    )
    if PADBD["pad_bytes"] != 137 or len(PADBLOB) != target:
        raise SystemExit("padding not charged exactly to target_bytes")
    pad_region = PADBLOB[target - 32 - 137:target - 32]
    if any(pad_region):
        raise SystemExit("padding bytes are not all zero")
    decoded = W.decode_projection(PADBLOB)
    if not torch.equal(decoded, REF0):
        raise SystemExit("decode of padded blob != stock_dequant")
    RESULTS["padding_check"] = {"pad_bytes": 137, "decode": "torch.equal"}


# ------------------------------------------------------- 8 corruption refusals
def refusals():
    fields = W._HEADER_STRUCT.unpack_from(BLOB0, 0)
    named = dict(zip(W._HEADER_FIELDS, fields[6:]))
    meta_start = W.HEADER_BYTES + named["alphabet_bytes"]
    field_base = W._HEADER_STRUCT.size - 4 * len(W._HEADER_FIELDS)

    def reseal(modified: bytearray) -> bytes:
        modified[-32:] = hashlib.sha256(bytes(modified[:-32])).digest()
        return bytes(modified)

    cases = {}
    cases["truncated"] = BLOB0[:-1]
    body_start = meta_start + named["meta_bytes"] + named["tag_bytes"] + named["state_bytes"]
    flipped = bytearray(BLOB0)
    flipped[body_start + named["body_bytes"] // 2] ^= 0x01
    cases["bitflip_checksum"] = bytes(flipped)
    misdeclared = bytearray(BLOB0)
    struct.pack_into(
        "<I", misdeclared, field_base + 4 * W._HEADER_FIELDS.index("block_cols"), 4
    )
    cases["misdeclared_block_cols"] = reseal(misdeclared)
    bad_rate = bytearray(BLOB0)
    rate_offset = (meta_start + 2 + BD0["meta_bytes"]["lut_table_bytes"]
                   + BD0["meta_bytes"]["lut_index_bytes"]
                   + BD0["meta_bytes"]["scale_global_bytes"]
                   + BD0["meta_bytes"]["scale_rows_bytes"] + 3)
    bad_rate[rate_offset] = 15  # above the units' window_bits of 14
    cases["rate_above_window_bits"] = reseal(bad_rate)
    dirty_pad = bytearray(PADBLOB)
    dirty_pad[len(dirty_pad) - 32 - 1] = 0x01  # last pad byte, then reseal
    cases["dirty_padding"] = reseal(dirty_pad)
    for name, blob in cases.items():
        try:
            W.decode_projection(blob)
        except W.BlockWireFormatError as exc:
            RESULTS.setdefault("refusals", {})[name] = str(exc)[:120]
        else:
            raise SystemExit(f"decode accepted a corrupt blob: {name}")


# --------------------------------- 9 alternative geometries stay configurable
def alternative_geometries():
    for abr, abc in ((256, 2), (512, 2), (128, 4)):
        anrb, ancb = ROWS // abr, COLS // abc
        sel = ((np.add.outer(np.arange(anrb), np.arange(ancb))) % 2).astype(np.int64)
        blob, bd = W.pack_projection(PARENTS, sel, abr, abc)
        if bd["total_bytes"] != W.fixed_bytes(PARENTS, abr, abc) + bd["body_bytes"]:
            raise SystemExit(f"accounting broke at {abr}x{abc}")
        costs = W.body_costs(PARENTS, abr, abc)
        if bd["body_bytes"] != int(costs[np.arange(costs.shape[0]), sel.reshape(-1)].sum()):
            raise SystemExit(f"body total broke at {abr}x{abc}")
        if abc == 2:
            per_element = torch.from_numpy(
                sel.repeat(abr, axis=0).repeat(abc, axis=1)
            )
            expected = torch.where(per_element == 0, REF0, REF1)
            if not torch.equal(W.decode_projection(blob), expected):
                raise SystemExit(f"round trip broke at {abr}x{abc}")
        RESULTS.setdefault("alternative_geometries", {})[f"{abr}x{abc}"] = bd["total_bytes"]


# ------------------------------------------------- 10 scratch bytes round trip
def scratch_bytes():
    blob_path = SCRATCH / "blob_single_parent.bin"
    blob_path.write_bytes(BLOB0)
    (SCRATCH / "breakdown_single.json").write_text(json.dumps(BD0, indent=1))
    (SCRATCH / "breakdown_mixed.json").write_text(json.dumps(BDM, indent=1))
    (SCRATCH / "blob_mixed.bin").write_bytes(BLOBM)
    from_disk = blob_path.read_bytes()
    if hashlib.sha256(from_disk).hexdigest() != hashlib.sha256(BLOB0).hexdigest():
        raise SystemExit("scratch file bytes differ from the packed blob")
    if not torch.equal(W.decode_projection(from_disk), REF0):
        raise SystemExit("decode from file bytes != stock_dequant")
    RESULTS["scratch"] = {
        "dir": str(SCRATCH),
        "blob_single_sha256": hashlib.sha256(BLOB0).hexdigest(),
        "blob_mixed_sha256": hashlib.sha256(BLOBM).hexdigest(),
    }


def main():
    check("env", env)
    check("parents_from_a8s_export", parents)
    check("body_costs_formula", body_costs_check)
    check("fixed_accounting", fixed_check)
    check("single_parent_round_trip_exact", single_round_trip)
    check("mixed_parent_round_trip_exact", mixed_round_trip)
    check("packer_matches_pack_body", packer_equivalence)
    check("boundary_states_match_replay", boundary_states)
    check("overflow_and_padding", overflow_and_padding)
    check("corruption_refusals", refusals)
    check("alternative_geometries", alternative_geometries)
    check("scratch_bytes_round_trip", scratch_bytes)
    RESULTS["elapsed_seconds"] = round(time.monotonic() - STARTED, 1)
    print("RESULT " + json.dumps(RESULTS), flush=True)
    print("SMOKE-PASS pq-block-reference-wire", flush=True)


if __name__ == "__main__":
    main()
