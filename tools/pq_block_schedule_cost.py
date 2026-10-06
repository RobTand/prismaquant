"""Actual block-schedule run and metadata accounting for the #2329 trial.

One summary over three real artifacts: parsed parent units (the existing
tessera public core), an actual selection grid at one ``(block_rows,
block_cols)`` granularity, and an actual ``pq_block_reference_wire
`` ``pack_projection`` breakdown.  Everything under ``schedule``,
``stored_bytes`` and ``classes`` is arithmetic on those artifacts only.

Everything under ``conditional`` is a LINKED MODEL, not a measurement: the
constants come from ``tools/pq_block_decode_estimate.py`` (b7e62b6
routed_fused.py framing: BN=128, BK=32, BDESC_INTS=12, run_pair 4x int32,
512-row tile per packed column; prior fixed-cost report 0.25 ms / 21 setup
launches).  No native reader is qualified, no runtime time or occupancy is
modeled, stored 512-row metadata is never reported as issued 128-row traffic,
and nothing here claims anything about an actual final schedule before that
schedule's own files exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

import pq_block_reference_wire as wire
from tessera.fused import parse_fused
from tessera.unit_artifact import parse_unit_artifact

SCHEMA = "prismaquant.block_schedule_cost.v1"
PARENT_BANK_SCHEMA = "prismaquant.block_trial_parent_bank.v1"
#: The real driver's breakdown wrapper (tools/pq_block_trial.py emission).
WRAPPER_SCHEMA = "prismaquant.block_trial_breakdown.v1"

#: Current native kernel framing (see the linked estimate source; declared
#: geometry, not a measured property of any reader).
ISSUE_TILE_ROWS = 128
ISSUE_TILE_COLS = 32
STORAGE_TILE_ROWS = 512
STORAGE_TILE_COLS = 32

_ESTIMATE_SOURCE = "tools/pq_block_decode_estimate.py"
_CONDITIONAL_PROVENANCE = (
    "linked constants from " + _ESTIMATE_SOURCE + "; b7e62b6 routed_fused.py "
    "framing BN=128, BK=32, BDESC_INTS=12, run_pair 4x int32, 512-row tile per "
    "packed column; conditional models, not measured bytes")

# Linked constants (never re-derived here): the uniform baseline carries one
# 48-byte descriptor (12 int32) per 32 columns; a hypothetical per-run wire
# costs 16 bytes (4 int32 run_pair).
RUN_METADATA_MODEL = {
    "existing_descriptor_bytes_per_32_columns": 48,
    "existing_descriptor_ints": 12,
    "hypothetical_wire_run_bytes": 16,
    "hypothetical_run_pair_int32s": 4,
}

# Linked constants: retained uniform dynamic-SMEM bases, the per-class
# projected kernel LUT and the published dynamic capacity.
SMEM_MODEL = {
    "kernel_lut_bytes": 1 << 14,
    "published_dynamic_cap_bytes": 101_376,
    "retained_uniform_bases": {
        "gate_up": {"BM64": 53_456, "BM128": 57_552},
        "down": {"BM64": 36_880, "BM128": 40_976},
    },
    "gate_up_projections": 2,
    "down_projections": 1,
}

# Linked constants: prior fixed-cost report proxy, transferable assumption
# only; never a measured launch latency or an upper bound.
LAUNCH_PROXY_MODEL = {
    "setup_and_gap_ms": 0.25,
    "setup_launches": 21,
    "split_class_extra_launches_per_class": 2,
}

MITIGATION_CONDITIONS = [
    "decode-once is at LOAD resident dense/shared eager M>=256, not routed "
    "per-chunk",
    "the same prepared instruction stream holds only under same width/LUT "
    "conditions",
    "class splitting adds actual launches/reductions/traffic; the profile "
    "setup-gap proxy 0.25 ms / 21 launches is a transferable assumption only",
    "a shared channel permutation needs all consumers and gate/up/down "
    "consistency; 2D schedules may not factor",
    "warp-specialization requires measured decode class costs; the current "
    "wait 59-77% is SASS samples, not wall fraction",
]

READER_RISKS = [
    "existing all-R4 single-run piece-major eligibility is a risk "
    "observation, not proof of a new supported reader",
    "existing single-run A-prefetch eligibility is a risk observation, not "
    "proof of a new supported reader",
    "every reference mixed-parent blob requires a new reader; none exists in "
    "this accounting",
]


# ---------------------------------------------------------------------------
# small JSON helpers
# ---------------------------------------------------------------------------

def _hist(values) -> dict:
    counts: dict = {}
    for value in values:
        key = int(value)
        counts[key] = counts.get(key, 0) + 1
    return {str(key): counts[key] for key in sorted(counts)}


def _stats(values):
    values = [int(v) for v in values]
    if not values:
        return None
    return {"min": min(values), "max": max(values),
            "mean": sum(values) / len(values), "n": len(values)}


def _file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


# ---------------------------------------------------------------------------
# issue-tile framing over a selection grid
# ---------------------------------------------------------------------------

def _issue_frame(geometry: dict) -> dict:
    """Tile the projection into ISSUE_TILE_ROWS x ISSUE_TILE_COLS tiles."""
    rows, cols = geometry["rows"], geometry["cols"]
    br, bc = geometry["block_rows"], geometry["block_cols"]
    _require(rows % ISSUE_TILE_ROWS == 0 and cols % ISSUE_TILE_COLS == 0,
             f"projection {rows}x{cols} does not tile "
             f"{ISSUE_TILE_ROWS}x{ISSUE_TILE_COLS} issue tiles")
    _require(bc <= ISSUE_TILE_COLS and ISSUE_TILE_COLS % bc == 0,
             f"block_cols {bc} must divide the {ISSUE_TILE_COLS} issue columns")
    if br <= ISSUE_TILE_ROWS:
        _require(ISSUE_TILE_ROWS % br == 0,
                 f"block_rows {br} must divide the {ISSUE_TILE_ROWS} issue rows")
        span = ISSUE_TILE_ROWS // br
        block_row_of = lambda tr: tr * span  # noqa: E731
    else:
        _require(br % ISSUE_TILE_ROWS == 0,
                 f"block_rows {br} must be a whole number of "
                 f"{ISSUE_TILE_ROWS}-row issue tiles")
        span = 1
        block_row_of = lambda tr: (tr * ISSUE_TILE_ROWS) // br  # noqa: E731
    return {"issue_tile_rows": rows // ISSUE_TILE_ROWS,
            "issue_tile_cols": cols // ISSUE_TILE_COLS,
            "slots": ISSUE_TILE_COLS // bc, "span": span,
            "block_row_of": block_row_of,
            "storage_tile_rows_divide": rows % STORAGE_TILE_ROWS == 0,
            "storage_tiles_rows": rows // STORAGE_TILE_ROWS,
            "storage_tiles_cols": cols // STORAGE_TILE_COLS}


def _tile_slot_tags(tags: np.ndarray, geometry: dict, frame: dict) -> np.ndarray:
    """Source-order tags of every issue tile as ``[tir, tic, span*slots]``."""
    ncb = geometry["num_col_blocks"]
    slots = frame["slots"]
    _require(ncb == frame["issue_tile_cols"] * slots,
             "column blocks do not group into whole issue tiles")
    tir, tic = frame["issue_tile_rows"], frame["issue_tile_cols"]
    flat = np.empty((tir, tic, frame["span"] * slots), dtype=np.int64)
    for tr in range(tir):
        i0 = frame["block_row_of"](tr)
        for k in range(frame["span"]):
            flat[tr, :, k * slots:(k + 1) * slots] = \
                tags[i0 + k].reshape(tic, slots)
    return flat


def _per_tile_distinct(tile_tensor: np.ndarray) -> np.ndarray:
    """Distinct-value count per tile of a ``[tir, tic, S]`` tag tensor."""
    return np.array([[len(np.unique(tile)) for tile in row] for row in tile_tensor])


# ---------------------------------------------------------------------------
# conditional (linked) models applied to the actual schedule
# ---------------------------------------------------------------------------

def _conditional_run_layout(runs, layout_label: str, num_weights: int) -> dict:
    """Apply the linked 16-byte-per-run model to one run-count array."""
    run_bytes = RUN_METADATA_MODEL["hypothetical_wire_run_bytes"]
    runs = [int(r) for r in runs]
    per_tile = [r * run_bytes for r in runs]
    extra = [(r - 1) * run_bytes for r in runs]
    return {
        "layout": layout_label,
        "hypothetical_total_bytes": sum(per_tile),
        "hypothetical_per_tile_bytes_hist": _hist(per_tile),
        "hypothetical_extra_bytes_vs_single_run_hist": _hist(extra),
        "hypothetical_extra_bytes_stats": _stats(extra),
        "hypothetical_extra_total_bytes": sum(extra),
        "hypothetical_extra_bpp_if_charged_per_issue_tile": (
            sum(extra) * 8 / num_weights),
    }


def _conditional_run_metadata(runs_per_tile, distinct_per_tile,
                              num_weights: int) -> dict:
    """Linked 16-byte-per-run model over BOTH already-computed run arrays.

    The existing uniform format carries one 48-byte descriptor per 32 issue
    columns.  These are model bytes, never measured traffic, and both layouts
    come from the same computed arrays, only the order differs.
    """
    tiles = len(runs_per_tile)
    return {
        "constants": dict(RUN_METADATA_MODEL),
        "not_measured_bytes": True,
        "applied_to": "actual per-issue-tile run counts, both layouts",
        "layouts": {
            "source_order": _conditional_run_layout(
                runs_per_tile, "actual source-order runs per issue tile",
                num_weights),
            "format_sorted": _conditional_run_layout(
                distinct_per_tile,
                "hypothetical same-parent-contiguous runs per issue tile",
                num_weights),
        },
        "existing_uniform_total_bytes_if_per_issue_tile": (
            RUN_METADATA_MODEL["existing_descriptor_bytes_per_32_columns"]
            * tiles),
    }


def _smem_mode(kind_bases: dict, projections: int, classes: int,
               cap: int, lut: int) -> dict:
    out = {}
    for bm, base in kind_bases.items():
        added = (classes - 1) * projections * lut
        out[bm] = {"retained_base_bytes": base, "added_lut_bytes": added,
                   "modeled_dynamic_smem_bytes": base + added,
                   "published_dynamic_cap_bytes": cap,
                   "fits_cap_in_model": base + added <= cap}
    return out


def _conditional_smem(classes: int, active_tables: int) -> dict:
    """Linked SMEM capacity deltas against the retained bases.

    ``classes`` is the ACTUAL distinct-parent count of the selection.  The
    ``same_class_count_on_both_gate_up_projections`` mode reproduces the old
    estimate assumption (one DISTINCT 16-KiB LUT staged per class per
    projection, same class count on gate and up, so 2 projections).  An
    ``up_only_allocation`` stages the extra LUT on the ONE scheduled gate_up
    projection and leaves down unchanged.
    """
    cap = SMEM_MODEL["published_dynamic_cap_bytes"]
    lut = SMEM_MODEL["kernel_lut_bytes"]
    bases = SMEM_MODEL["retained_uniform_bases"]
    return {
        "provenance": _CONDITIONAL_PROVENANCE,
        "capacity_changes_only": True,
        "occupancy_or_time_modeled": False,
        "assumption": "the historical estimate stages one DISTINCT 16-KiB "
                      "LUT per class per projection and assumes the SAME "
                      "class count on both gate_up projections (2); an "
                      "up-only allocation stages the extra LUT on the ONE "
                      "scheduled gate_up projection and does not change down",
        "class_count_used": {"source": "actual selection distinct parents",
                             "count": classes},
        "same_class_count_on_both_gate_up_projections": {
            "note": "historical estimate assumption, 2 projections",
            "gate_up": _smem_mode(bases["gate_up"],
                                  SMEM_MODEL["gate_up_projections"], classes,
                                  cap, lut),
            "down": _smem_mode(bases["down"], SMEM_MODEL["down_projections"],
                               classes, cap, lut)},
        "up_only_allocation": {
            "note": "one gate_up projection staged; down unchanged",
            "gate_up": _smem_mode(bases["gate_up"], 1, classes, cap, lut),
            "down": {bm: {"retained_base_bytes": base, "added_lut_bytes": 0,
                          "modeled_dynamic_smem_bytes": base,
                          "unchanged_by_up_only_allocation": True}
                     for bm, base in bases["down"].items()},
        },
        "distinct_window_table_alternative": {
            "note": "actual distinct window_codes byte tables among selected "
                    "parents could stage fewer LUTs than classes; reported "
                    "for completeness, not claimed as a saving",
            "distinct_active_window_tables": active_tables,
            "gate_up_modeled": _smem_mode(bases["gate_up"],
                                          SMEM_MODEL["gate_up_projections"],
                                          active_tables, cap, lut),
        },
    }


# ---------------------------------------------------------------------------
# stored vs issued accounting
# ---------------------------------------------------------------------------

def _stored_bytes(breakdown: dict, num_weights: int) -> dict:
    """Exact stored-plane bytes and bits-per-stored-weight from the breakdown."""
    meta = breakdown["meta_bytes"]
    planes = {
        "header_bytes": breakdown["header_bytes"],
        "alphabet_bytes": breakdown["alphabet_bytes"],
        "meta_bytes": meta["total"],
        "tag_bytes": breakdown["tag_bytes"],
        "state_bytes": breakdown["state_bytes"],
        "body_bytes": breakdown["body_bytes"],
        "pad_bytes": breakdown["pad_bytes"],
        "checksum_bytes": breakdown["checksum_bytes"],
    }
    meta_parts = {key: value for key, value in meta.items()
                  if key != "distinct_luts"}
    return {
        "frames": "whole-projection stored blob packed at the wire's own "
                  f"{breakdown['geometry']['block_rows']}x"
                  f"{breakdown['geometry']['block_cols']} blocks; the "
                  f"{STORAGE_TILE_ROWS}x{STORAGE_TILE_COLS} frame is the "
                  "native comparison framing, not this blob's physical "
                  "tiling",
        "planes_bytes": planes,
        "planes_bits_per_stored_weight": {
            name: value * 8 / num_weights for name, value in planes.items()},
        "meta_plane_bytes": meta_parts,
        "meta_plane_bits_per_stored_weight": {
            key: value * 8 / num_weights for key, value in meta_parts.items()},
        "distinct_luts": int(meta["distinct_luts"]),
        "fixed_bytes": breakdown["fixed_bytes"],
        "content_bytes": breakdown["content_bytes"],
        "total_bytes": breakdown["total_bytes"],
        "total_bits_per_stored_weight": breakdown["total_bytes"] * 8 / num_weights,
        "checksum_sha256": breakdown["checksum"],
        "accounting_consistent": breakdown["accounting_consistent"],
    }


def _issued_per_issue_tile(flat_tags: np.ndarray, tile_tables: np.ndarray,
                           geometry: dict, frame: dict, breakdown: dict,
                           body_block: dict) -> dict:
    """Diagnostic slices one 128x32 issue tile would touch.

    Slices are exact arithmetic on the stored planes; they are NOT a reader
    and never include storage descriptors automatically.
    """
    slots = frame["slots"] * frame["span"]
    window_bits = int(breakdown["window_bits"])
    tag_bits = int(breakdown["tag_bits"])
    br = geometry["block_rows"]
    tir = frame["issue_tile_rows"]
    classes_hist = _hist(_per_tile_distinct(flat_tags).reshape(-1))
    tables_hist = _hist(_per_tile_distinct(tile_tables).reshape(-1))
    # Stored incoming-state entries exist only where a tile's top row is a
    # storage block-row boundary; original row 0 is known zero and stored
    # nowhere.
    top_rows = np.arange(tir) * ISSUE_TILE_ROWS
    stored = (top_rows % br == 0) & (top_rows > 0)
    at_row_zero = int((top_rows == 0).sum())
    tic = frame["issue_tile_cols"]
    return {
        "label": "diagnostic slices of the stored planes for one 128x32 issue "
                 "tile; not a reader, not automatic traffic",
        "tag_slots_per_tile": slots,
        "tag_bits_per_tile": slots * tag_bits,
        "state_bits_per_tile_when_stored": slots * geometry["block_cols"]
        * window_bits,
        "state_entry_bytes_when_stored": breakdown["state_entry_bytes"],
        "tiles_with_stored_state_entry": int(stored.sum()) * tic,
        "tiles_at_row_zero": at_row_zero * tic,
        "tiles_without_stored_state_entry": int(tir * tic - int(stored.sum())
                                                * tic - at_row_zero * tic),
        "state_boundary_rule": f"stored only where the tile top row is a "
                               f"{br}-row block boundary; tiles strictly inside "
                               "a block row have no stored entry and would "
                               "need the running state (diagnostic)",
        "body_bytes_per_issue_tile": body_block,
        "body_bytes_are_exact_logical_fragments":
            body_block["model"] == "whole_fragments",
        "body_traffic_measured": False,
        "row_scale_bytes_per_tile_formula": "slice of the global per-parent "
                                            "stored plane: 128 rows * 2 bytes "
                                            "* distinct classes in tile",
        "row_scale_bytes_by_distinct_class_count": {
            k: int(k) * ISSUE_TILE_ROWS * 2 for k in classes_hist},
        "window_table_bytes_by_active_table_count": {
            k: int(k) * (1 << window_bits) for k in tables_hist},
        "window_table_working_set_note": "kernel-resident working set per "
                                         "tile; the stored plane carries the "
                                         "same tables once per parent",
        "distinct_classes_per_tile_hist": classes_hist,
        "active_window_tables_per_tile_hist": tables_hist,
    }


# ---------------------------------------------------------------------------
# breakdown binding
# ---------------------------------------------------------------------------

def _verify_breakdown(breakdown: dict, geometry: dict, sizes: dict,
                      window_bits: int, tags: np.ndarray,
                      costs: np.ndarray) -> None:
    """Bind the loaded breakdown to THIS parents/selection/geometry re-plan.

    Sizes and accounting ONLY: the packed blob itself is not available on
    this path, so no checksum or content binding is performed.  A reported
    checksum is echoed as provenance and never verified here.
    """
    _require(isinstance(breakdown, dict), "breakdown must be a JSON object")
    mismatches = []

    def check(name, want, got):
        if want != got:
            mismatches.append(f"{name}: breakdown {got!r} != replan {want!r}")

    check("format", wire.FORMAT_NAME, breakdown.get("format"))
    check("format_version", wire.FORMAT_VERSION, breakdown.get("format_version"))
    check("geometry", geometry, breakdown.get("geometry"))
    check("num_parents", geometry["num_parents"], breakdown.get("num_parents"))
    check("window_bits", window_bits, breakdown.get("window_bits"))
    check("tag_bits", int(sizes["tag_bits"]), breakdown.get("tag_bits"))
    check("state_bits", window_bits, breakdown.get("state_bits"))
    check("header_bytes", wire.HEADER_BYTES, breakdown.get("header_bytes"))
    check("alphabet_bytes", int(sizes["alphabet_bytes"]),
          breakdown.get("alphabet_bytes"))
    check("meta_bytes", dict(sizes["meta"]), breakdown.get("meta_bytes"))
    check("tag_bytes", int(sizes["tag_bytes"]), breakdown.get("tag_bytes"))
    check("tag_count", geometry["num_blocks"], breakdown.get("tag_count"))
    check("state_entries", int(sizes["state_entries"]),
          breakdown.get("state_entries"))
    check("state_entry_bytes", int(sizes["state_entry_bytes"]),
          breakdown.get("state_entry_bytes"))
    check("state_bytes", int(sizes["state_bytes"]), breakdown.get("state_bytes"))
    check("fixed_bytes", int(sizes["fixed_bytes"]), breakdown.get("fixed_bytes"))
    selected_body = int(costs[np.arange(costs.shape[0]), tags.reshape(-1)].sum())
    check("body_bytes", selected_body, breakdown.get("body_bytes"))
    check("checksum_bytes", wire.CHECKSUM_BYTES, breakdown.get("checksum_bytes"))
    content = int(sizes["fixed_bytes"]) + selected_body
    check("content_bytes", content, breakdown.get("content_bytes"))
    check("total_bytes", content + int(breakdown.get("pad_bytes", 0)),
          breakdown.get("total_bytes"))
    if breakdown.get("accounting_consistent") is not True:
        mismatches.append("accounting_consistent is not true")
    if mismatches:
        raise ValueError("breakdown does not match the re-planned wire: "
                         + "; ".join(mismatches))


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def summarize_schedule(parents, selection, block_rows: int, block_cols: int,
                       breakdown: dict) -> dict:
    """Summarize one actual block schedule into a JSON dict.

    ``parents``: parsed tessera units (existing public core output); bank
    order is source order and defines tags ``0..num_parents-1``.
    ``selection``: integer ``[num_row_blocks, num_col_blocks]`` tag grid.
    ``breakdown``: the actual ``pack_projection`` breakdown for the same
    parents/selection/geometry, verified against a re-plan rather than
    trusted blind.  Conditional numbers are labeled linked models.
    """
    prepared, geometry, sizes = wire._plan(parents, block_rows, block_cols)
    tags = wire._validate_selection(selection, geometry)
    rows, cols = geometry["rows"], geometry["cols"]
    num_weights = rows * cols
    num_parents = geometry["num_parents"]
    nrb, ncb = geometry["num_row_blocks"], geometry["num_col_blocks"]
    window_bits = int(prepared[0]["window_bits"])

    costs = wire.body_costs(parents, block_rows, block_cols)
    _verify_breakdown(breakdown, geometry, sizes, window_bits, tags, costs)

    frame = _issue_frame(geometry)
    flat_tags = _tile_slot_tags(tags, geometry, frame)
    tir, tic = frame["issue_tile_rows"], frame["issue_tile_cols"]
    runs_per_tile = 1 + (flat_tags[..., 1:] != flat_tags[..., :-1]).sum(axis=2)
    distinct_per_tile = _per_tile_distinct(flat_tags)
    table_ids = np.asarray(sizes["table_indices"], dtype=np.int64)
    tile_tables = table_ids[flat_tags]
    distinct_tables_per_tile = _per_tile_distinct(tile_tables)

    # issued body bytes per tile from the wire's own cost model; when a block
    # row is taller than an issue tile, its fragment streams once per block
    # and the per-tile share is a labeled proportional diagnostic.
    per_block = costs[np.arange(nrb * ncb), tags.reshape(-1)].reshape(nrb, ncb)
    grouped = per_block.reshape(nrb, tic, frame["slots"]).sum(axis=2)
    if frame["span"] == 1:
        i_idx = (np.arange(tir) * ISSUE_TILE_ROWS) // geometry["block_rows"]
    else:
        i_idx = np.repeat(np.arange(nrb), frame["span"])[:tir]
    enclosing_bytes = grouped[i_idx]
    if geometry["block_rows"] <= ISSUE_TILE_ROWS:
        tile_body = np.stack(
            [grouped[tr * frame["span"]:(tr + 1) * frame["span"]].sum(axis=0)
             for tr in range(tir)])
        _require(int(tile_body.sum()) == int(breakdown["body_bytes"]),
                 "issued body slices do not sum to the breakdown body_bytes")
        body_slice_model = "whole_fragments"
        body_slice_note = ("each issue tile covers whole blocks; its body "
                           "bytes are the exact wire cost of those fragments")
        enclosing_bytes = None
    else:
        share = ISSUE_TILE_ROWS / geometry["block_rows"]
        tile_body = enclosing_bytes * share
        _require(abs(float(tile_body.sum()) - int(breakdown["body_bytes"]))
                 < 1e-6,
                 "proportional body slices diverge from the breakdown "
                 "body_bytes")
        body_slice_model = "proportional_fragment_slices"
        body_slice_note = ("the stored fragment streams once per "
                           f"{geometry['block_rows']}-row block; a 128-row "
                           "issue tile reads a proportional share and "
                           "byte-exact slicing needs the reader's bit layout")

    whole_scan = flat_tags.reshape(-1)
    whole_switches = int((whole_scan[1:] != whole_scan[:-1]).sum())
    if body_slice_model == "whole_fragments":
        body_block = {"model": body_slice_model, "note": body_slice_note,
                      "hist": _hist(tile_body.reshape(-1)),
                      "stats": _stats(tile_body.reshape(-1)),
                      "total_bytes": int(tile_body.sum())}
    else:
        flat = tile_body.reshape(-1)
        body_block = {"model": body_slice_model, "note": body_slice_note,
                      "stats": {"min": round(float(flat.min()), 3),
                                "max": round(float(flat.max()), 3),
                                "mean": round(float(flat.mean()), 3),
                                "n": int(flat.size)},
                      "sum_matches_stored_body_bytes": True,
                      "enclosing_block_row_stored_body_bytes": {
                          "hist": _hist(enclosing_bytes.reshape(-1)),
                          "stats": _stats(enclosing_bytes.reshape(-1)),
                          "note": "whole stored fragment bytes of the block "
                                  "row each tile belongs to; a storage view, "
                                  "not issued per tile"}}
    selected = np.unique(tags)
    active_tables = sorted({int(sizes["table_indices"][int(p)]) for p in selected})
    widths = []
    for p, entry in enumerate(prepared):
        rates = entry["rates"]
        widths.append({
            "tag": p, "window_bits": int(entry["window_bits"]),
            "unit_id": str(parents[p].manifest.branch.unit_id),
            "root_q256": int(parents[p].manifest.branch.root_q256),
            "actual_integer_widths": {str(int(r)): int((rates == r).sum())
                                      for r in np.unique(rates)},
            "rates_tuple": [int(r) for r in rates]})
    equal_width_groups: dict = {}
    for p, entry in enumerate(widths):
        key = f"window_bits={entry['window_bits']};rates={entry['rates_tuple']}"
        equal_width_groups.setdefault(key, []).append(p)

    result = {
        "schema": SCHEMA,
        "provisional": True,
        "scope": "actual schedule run and byte accounting over parsed parents, "
                 "an actual selection grid and an actual reference-wire "
                 "breakdown; conditional numbers are labeled linked models",
        "geometry": {
            "rows": rows, "cols": cols,
            "grid": str(prepared[0]["grid"].name),
            "block_rows": geometry["block_rows"],
            "block_cols": geometry["block_cols"],
            "num_row_blocks": nrb, "num_col_blocks": ncb,
            "num_blocks": geometry["num_blocks"], "num_parents": num_parents,
            "window_bits": window_bits,
            "tag_bits": int(sizes["tag_bits"]),
            "issue_tile": {"rows": ISSUE_TILE_ROWS, "cols": ISSUE_TILE_COLS,
                           "count_rows": tir, "count_cols": tic,
                           "total": tir * tic,
                           "block_col_slots": frame["slots"],
                           "block_rows_spanned": frame["span"]},
            "native_comparison_frame": {
                "rows": STORAGE_TILE_ROWS, "cols": STORAGE_TILE_COLS,
                "count_rows": frame["storage_tiles_rows"],
                "count_cols": frame["storage_tiles_cols"],
                "rows_divide": frame["storage_tile_rows_divide"],
                "note": "the native 512x32 comparison framing only; the "
                        "reference wire packs at its own chosen block size"},
        },
        "schedule": {
            "source_order_runs_per_issue_tile": {
                "hist": _hist(runs_per_tile.reshape(-1)),
                "stats": _stats(runs_per_tile.reshape(-1)),
                "total_runs": int(runs_per_tile.sum())},
            "source_order_run_switches_per_issue_tile": {
                "hist": _hist(runs_per_tile.reshape(-1) - 1),
                "stats": _stats(runs_per_tile.reshape(-1) - 1),
                "total_switches": int((runs_per_tile - 1).sum())},
            "format_sorted_runs_per_issue_tile": {
                "note": "hypothetical same-parent-contiguous layout: runs "
                        "equal the tile's distinct parent count",
                "hist": _hist(distinct_per_tile.reshape(-1)),
                "stats": _stats(distinct_per_tile.reshape(-1)),
                "total_runs": int(distinct_per_tile.sum())},
            "format_sorted_run_switches_per_issue_tile": {
                "hist": _hist(distinct_per_tile.reshape(-1) - 1),
                "total_switches": int((distinct_per_tile - 1).sum())},
            "source_order_scan_over_issue_tiles": {
                "slots": int(flat_tags.size),
                "runs": whole_switches + 1,
                "run_switches": whole_switches,
                "note": "one scan in source order across all issue tiles, "
                        "counting a switch at each tile boundary too"},
            "issued_body_bytes_per_issue_tile": {
                **body_block,
                "source": "pq_block_reference_wire.body_costs on the actual "
                          "parents and selection"},
            "active_window_tables_per_issue_tile": {
                "hist": _hist(distinct_tables_per_tile.reshape(-1)),
                "stats": _stats(distinct_tables_per_tile.reshape(-1))},
        },
        "classes": {
            "distinct_parents_selected": len(selected),
            "selected_tags": [int(p) for p in selected],
            "selection_histogram": {str(p): int((tags == p).sum())
                                    for p in range(num_parents)},
            "class_counts_per_issue_tile_hist":
                _hist(distinct_per_tile.reshape(-1)),
            "parents": widths,
            "equal_width_groups": {
                key: {"tags": group,
                      "note": "equal actual integer widths do not merge "
                              "classes: tags select tables, row scales and "
                              "rate vectors, so equal-width parents stay "
                              "separate"}
                for key, group in equal_width_groups.items()},
            "window_tables": {
                "stored_distinct_tables": int(sizes["meta"]["distinct_luts"]),
                "active_distinct_tables": len(active_tables),
                "active_table_ids": active_tables,
                "bytes_per_table": 1 << window_bits,
                "dedup_key": "actual window_codes bytes"},
        },
        "stored_bytes": _stored_bytes(breakdown, num_weights),
        "replan_binding": {
            "scope": "sizes_and_accounting_only",
            "blob_checksum_or_content_verified": False,
            "note": "the CLI re-plans the wire from the parsed parents and "
                    "compares every size and accounting field; the packed "
                    "blob is not on this path, so its checksum is echoed as "
                    "reported provenance, never verified",
        },
        "stored_vs_issued": {
            "stored_planes_are_whole_projection": True,
            "storage_metadata_issued_automatically": False,
            "note": "row scales, rate vectors and window tables are global "
                    "stored planes of the packed blob (one plane per parent "
                    "over the whole projection), not per-512-row planes, and "
                    "are never counted as issued 128-row traffic here; the "
                    "48-byte uniform descriptor is the EXISTING format's "
                    "per-issue cost and appears only in the conditional model",
        },
        "issued_per_issue_tile": _issued_per_issue_tile(
            flat_tags, tile_tables, geometry, frame, breakdown, body_block),
        "conditional": {
            "provenance": _CONDITIONAL_PROVENANCE,
            "run_metadata": _conditional_run_metadata(
                runs_per_tile.reshape(-1), distinct_per_tile.reshape(-1),
                num_weights),
            "smem_lut_capacity": _conditional_smem(
                len(selected), len(active_tables)),
            "split_launch_proxy": {
                "constants": dict(LAUNCH_PROXY_MODEL),
                "per_launch_proxy_ms": LAUNCH_PROXY_MODEL["setup_and_gap_ms"]
                / LAUNCH_PROXY_MODEL["setup_launches"],
                "transferable_assumption_only": True,
                "total_time_measured": False,
                "source": _ESTIMATE_SOURCE},
        },
        "mitigation_conditions": list(MITIGATION_CONDITIONS),
        "reader_risks": list(READER_RISKS),
        "claims": {
            "native_reader_qualified": False,
            "runtime_speed_or_occupancy_measured": False,
            "allocation_quality_qualified": False,
            "variable_width_qualified": False,
            "final_schedule_reported": False,
            "final_schedule_note": "no claim about an actual final schedule "
                                   "until its files exist",
        },
    }
    return result


# ---------------------------------------------------------------------------
# artifact loaders (existing public core only)
# ---------------------------------------------------------------------------

def load_selection(path) -> np.ndarray:
    array = np.load(Path(path), allow_pickle=False)
    _require(isinstance(array, np.ndarray) and array.ndim in (1, 2),
             f"selection {path} must be flat row-major or a 2-D grid")
    _require(array.dtype != bool and np.issubdtype(array.dtype, np.integer),
             f"selection {path} must hold integer parent tags")
    return array.astype(np.int64, copy=False)


def load_breakdown(path) -> dict:
    return json.loads(Path(path).read_bytes())


def _order_by_candidate(parents, provenance, order):
    """Reorder parsed parents AND provenance by an explicit candidate order.

    The order names rungs in TAG order (control first in the real driver's
    emission).  Refuses missing, duplicate or unmatched entries; never infers
    an order.  Each reordered parent must decode to the rung it is named by.
    """
    rungs = [entry.get("rung") for entry in provenance]
    _require(all(isinstance(r, int) for r in rungs),
             "candidate ordering needs integer rungs on every bank entry")
    _require(isinstance(order, list) and bool(order),
             "candidate_order must be a non-empty list of rungs")
    _require(len(set(order)) == len(order),
             f"candidate_order carries duplicates: {order}")
    _require(sorted(int(r) for r in order) == sorted(int(r) for r in rungs),
             f"candidate_order {sorted(int(r) for r in order)} does not match "
             f"the bank's rung multiset {sorted(int(r) for r in rungs)}; "
             "refusing an unmatched order")
    index_of = {int(r): i for i, r in enumerate(rungs)}
    ordered_parents, ordered_provenance = [], []
    for tag, rung in enumerate(order):
        i = index_of[int(rung)]
        parsed = parents[i]
        actual = int(parsed.manifest.branch.root_q256)
        _require(actual == int(rung),
                 f"candidate_order names rung {rung} but the bank entry at "
                 f"position {i} decodes root_q256 {actual}")
        entry = dict(provenance[i])
        entry["tag"] = tag
        entry["bank_position"] = i
        ordered_parents.append(parsed)
        ordered_provenance.append(entry)
    return ordered_parents, ordered_provenance


def load_schedule_inputs(parent_bank, selection, breakdown, export_root=None,
                         packed_blob=None):
    """Load one schedule's real artifacts for the CLI.

    Accepts the raw ``pack_projection`` breakdown JSON (bank order is tag
    order) or the real driver's wrapper (``prismaquant.block_trial_breakdown
    .v1`` with ``wire`` + ``candidate_order`` + a producer stamp over the
    selection bytes).  A flat row-major selection is reshaped from the wire's
    exact geometry.  Returns everything ``summarize_schedule`` needs plus the
    handoff provenance.
    """
    breakdown_doc = load_breakdown(breakdown)
    _require(isinstance(breakdown_doc, dict), "breakdown must be a JSON object")
    parents, provenance = load_parent_bank(parent_bank, export_root)
    handoff = {"replan_binding": "sizes_and_accounting_only_no_blob_checksum"}
    if breakdown_doc.get("schema") == WRAPPER_SCHEMA:
        wire = breakdown_doc.get("wire")
        _require(isinstance(wire, dict) and isinstance(wire.get("geometry"), dict),
                 "wrapper carries no wire breakdown with geometry")
        order = breakdown_doc.get("candidate_order")
        files = breakdown_doc.get("files") or {}
        stamp = files.get("selection")
        _require(isinstance(stamp, dict) and "sha256" in stamp,
                 "wrapper carries no producer stamp over selection bytes; "
                 "refusing to diagnose unverified selection")
        actual = _file_sha256(selection)
        _require(actual == stamp["sha256"],
                 f"selection bytes fail their producer stamp: file "
                 f"{actual} != wrapper {stamp['sha256']}")
        handoff["selection_producer_stamp_verified"] = True
        handoff["selection_stamp_sha256"] = stamp["sha256"]
        handoff["selection_stamp_order_note"] = stamp.get("order")
        if packed_blob is not None and "packed_blob" in files:
            blob_stamp = files["packed_blob"].get("sha256")
            _require(_file_sha256(packed_blob) == blob_stamp,
                     "packed blob bytes fail the wrapper producer stamp")
            handoff["packed_blob_stamp_verified"] = True
        parents, provenance = _order_by_candidate(parents, provenance, order)
        handoff["form"] = "wrapper"
        handoff["candidate_order"] = [int(r) for r in order]
    else:
        wire = breakdown_doc
        handoff["form"] = "raw"
        handoff["candidate_order"] = None
        handoff["selection_producer_stamp_verified"] = None
        handoff["selection_stamp_order_note"] = None
    geometry = wire.get("geometry") or {}
    _require(isinstance(geometry.get("block_rows"), int)
             and isinstance(geometry.get("block_cols"), int),
             "breakdown carries no block geometry; pass an actual "
             "pack_projection breakdown or wrapper")
    nrb, ncb = geometry["num_row_blocks"], geometry["num_col_blocks"]
    selection_array = load_selection(selection)
    if selection_array.ndim == 1:
        _require(selection_array.size == nrb * ncb,
                 f"flat selection holds {selection_array.size} tags, the wire "
                 f"geometry needs exactly {nrb}*{ncb} = {nrb * ncb}")
        selection_array = selection_array.reshape(nrb, ncb)
        handoff["selection_flat_reshaped_row_major"] = True
    else:
        _require(selection_array.ndim == 2,
                 f"selection must be 1-D flat or 2-D, got {selection_array.ndim}-D")
        handoff["selection_flat_reshaped_row_major"] = False
    handoff["bank_tag_order_rungs"] = [entry.get("rung")
                                       for entry in provenance]
    handoff["selection_shape"] = [int(v) for v in selection_array.shape]
    return parents, provenance, selection_array, wire, handoff


def load_parent_bank(bank_path, export_root=None):
    """Parse a block-trial parent bank via the existing public core.

    Two entry forms, both ending in ``parse_unit_artifact``:
    ``path`` entries name a standalone unit blob file;
    ``shard``+``tensor`` entries name a fused export tensor whose members are
    split by ``parse_fused`` and matched by role (the qname's last component).
    Bank order is source order and defines the tags.
    """
    bank_path = Path(bank_path)
    bank = json.loads(bank_path.read_bytes())
    _require(bank.get("schema") == PARENT_BANK_SCHEMA,
             f"parent bank schema must be {PARENT_BANK_SCHEMA}, got "
             f"{bank.get('schema')!r}")
    entries = bank.get("parents")
    _require(isinstance(entries, list) and bool(entries),
             "parent bank carries no parents list")
    roots = dict(bank.get("roots") or {})
    if isinstance(bank.get("a8s_export"), str):
        roots.setdefault("a8s_export", bank["a8s_export"])
    bank_dir = bank_path.resolve().parent
    parents, provenance = [], []
    for index, entry in enumerate(entries):
        qname = entry.get("qname") or entry.get("name") or bank.get("qname")
        _require(isinstance(qname, str) and bool(qname),
                 f"parent {index} carries no qname")
        provenance.append({"tag": index, "qname": qname,
                           "rung": entry.get("rung")})
        if "path" in entry:
            path = Path(entry["path"])
            if not path.is_absolute():
                path = bank_dir / path
            blob = path.read_bytes()
            if "sha256" in entry:
                _require(hashlib.sha256(blob).hexdigest() == entry["sha256"],
                         f"parent {index} sha256 mismatch for {path}")
            if "bytes" in entry:
                _require(len(blob) == int(entry["bytes"]),
                         f"parent {index} byte count mismatch for {path}")
            parsed = parse_unit_artifact(blob, device="cpu")
            provenance[-1].update({"form": "unit_file", "path": str(path),
                                   "bytes": len(blob),
                                   "sha256": hashlib.sha256(blob).hexdigest()})
        else:
            _require("tensor" in entry,
                     f"parent {index} needs a 'path' or a 'tensor' (fused "
                     "export) form; the safetensors index resolves the shard")
            root = export_root
            if root is None and entry.get("root") is not None:
                root = roots.get(entry["root"])
            if root is None:
                root = roots.get("a8s_export")
            _require(root is not None,
                     f"parent {index} names a fused export but no export root "
                     "is available (bank roots or --export-root)")
            root = Path(root)
            index_map = json.loads(
                (root / "model.safetensors.index.json").read_bytes())["weight_map"]
            tensor_key = entry["tensor"]
            shard_path = root / index_map[tensor_key]
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                stream = handle.get_tensor(tensor_key)
            _require(stream.dtype == torch.uint8 and stream.ndim == 1,
                     f"{tensor_key}: fused wire must be one uint8 byte stream")
            blob = stream.numpy().tobytes()
            role = entry.get("role") or qname.rsplit(".", 1)[1]
            matching = [m for m in parse_fused(blob) if m.name == role]
            _require(len(matching) == 1,
                     f"{tensor_key}: expected one {role} member, found "
                     f"{[m.name for m in parse_fused(blob)]}")
            member = matching[0]
            if "member_sha256" in entry:
                _require(hashlib.sha256(member.blob).hexdigest()
                         == entry["member_sha256"],
                         f"parent {index} member sha256 mismatch")
            parsed = parse_unit_artifact(member.blob, device="cpu")
            provenance[-1].update({
                "form": "fused_export", "root": str(root),
                "shard": index_map[tensor_key], "tensor": tensor_key,
                "member": role, "member_bytes": len(member.blob),
                "member_sha256": hashlib.sha256(member.blob).hexdigest()})
        provenance[-1]["unit_id"] = str(parsed.manifest.branch.unit_id)
        provenance[-1]["root_q256"] = int(parsed.manifest.branch.root_q256)
        parents.append(parsed)
    return parents, provenance


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-bank", type=Path, required=True,
                        help="parent bank JSON "
                             "(prismaquant.block_trial_parent_bank.v1)")
    parser.add_argument("--selection", type=Path, required=True,
                        help="selection .npy, integer flat row-major "
                             "b=i*ncb+j or 2-D [num_row_blocks, num_col_blocks]")
    parser.add_argument("--breakdown", type=Path, required=True,
                        help="actual pack_projection breakdown JSON, raw or "
                             "the driver wrapper (wire + candidate_order)")
    parser.add_argument("--output", type=Path, required=True,
                        help="where to write the schedule-cost summary JSON")
    parser.add_argument("--export-root", type=Path, default=None,
                        help="fused export root for shard+tensor bank entries")
    parser.add_argument("--packed-blob", type=Path, default=None,
                        help="optional packed blob to verify against the "
                             "wrapper's producer stamp")
    args = parser.parse_args()

    parents, parent_provenance, selection, wire, handoff = load_schedule_inputs(
        args.parent_bank, args.selection, args.breakdown, args.export_root,
        args.packed_blob)
    geometry = wire["geometry"]
    summary = summarize_schedule(parents, selection,
                                 geometry["block_rows"],
                                 geometry["block_cols"], wire)
    summary["handoff"] = handoff
    summary["provenance"] = {
        "parent_bank": {"path": str(args.parent_bank),
                        "sha256": _file_sha256(args.parent_bank)},
        "selection": {"path": str(args.selection),
                      "sha256": _file_sha256(args.selection),
                      "shape": handoff["selection_shape"],
                      "dtype": "int64"},
        "breakdown": {"path": str(args.breakdown),
                      "sha256": _file_sha256(args.breakdown),
                      "form": handoff["form"]},
        "parents": parent_provenance,
        "sources": {"wire": "tools/pq_block_reference_wire.py",
                    "conditional_models": _ESTIMATE_SOURCE},
    }
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"schema": SCHEMA, "output": str(args.output),
                      "sha256": _file_sha256(args.output),
                      "handoff_form": handoff["form"],
                      "candidate_order": handoff["candidate_order"],
                      "issue_tiles": summary["geometry"]["issue_tile"]["total"],
                      "distinct_parents_selected":
                          summary["classes"]["distinct_parents_selected"],
                      "total_bytes": summary["stored_bytes"]["total_bytes"]}),
          flush=True)


if __name__ == "__main__":
    main()
