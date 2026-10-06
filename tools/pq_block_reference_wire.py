"""Packed block reference wire for the per-block feasibility trial (#2329).

Research-only format, written for the eng-pq-fine-grained workstream.  This is
NOT a serving container and imports no serving runtime: it packs selected
row/column fragments of existing whole-unit WINDOW encodings into one blob with
exact byte accounting, and decodes the blob back to fp32 weights through the
existing tessera wire primitives only (``wire.pack_body``-equivalent streams,
``decode.replay_window``).

A block (row block ``i``, column block ``j``) selects a fragment from ONE
parent; the codec never re-fits or re-encodes anything.  The incoming L-bit
window state of every fragment is preserved: it is derived from the parent's
own body bits above the fragment and stored on the wire, except at original
row 0 where the whole-unit state is known zero.  Decoding a fragment therefore
reproduces the parent's own weights at those positions bit for bit
(``stock_dequant(materialize_stock(unit, forests, code))``), at any row
boundary.

Byte accounting contract (what the solver in the parent lane consumes):

* ``body_costs(parents, block_rows, block_cols)[b, p]`` is the ACTUAL packed
  length of block ``b``'s body fragment when parent ``p`` is selected:
  ``ceil(block_rows * sum(rates_p[fragment columns]) / 8)`` -- the parent's
  real per-column rates, never a nominal mean.  Block order is row-major over
  the tiling: ``b = i * num_col_blocks + j``.
* ``fixed_bytes(parents, block_rows, block_cols)`` is the exact
  selection-independent overhead: fixed header, grid alphabet plane, per-parent
  metadata (native LUT tables deduplicated by identical bytes, fp16 row scales,
  fp32 global scales, per-column rate vectors -- shared once per parent), the
  selection tag plane and the incoming-state plane.  The tag plane's and the
  state plane's SIZES depend only on the geometry and the parent count, never
  on which parent each block selects, so a solver under ``target_bytes`` has
  body budget ``target_bytes - fixed_bytes(...)`` and charges each selected
  block its ``body_costs`` entry.
* ``pack_projection`` refuses overflow against ``target_bytes``.  Any padding
  is explicit: it only exists when ``target_bytes`` is given, is all zero, and
  is charged in the breakdown.  There is no discarded-byte accounting and no
  hidden whole weight: the blob carries tables, scales, rates, tags, states,
  fragment bodies and nothing else.

Format (v1, little-endian scalars, MSB-first bit planes)::

    header   94 bytes, magic b"PQBLKWR1", fixed field order (see _HEADER)
    alphabet 256 bytes, the grid's native E4M3FN byte map, indexed by grid code
    meta     deduplicated LUT tables + per-parent LUT index (u16), fp32
             scale_global, fp16 scale_rows, u8 per-column rates
    tags     num_row_blocks*num_col_blocks tags at tag_bits each, MSB-first
    states   (num_row_blocks-1)*num_col_blocks entries, each block_cols fields
             of window_bits bits (the incoming state, column-major over blocks
             with i >= 1, then column blocks), MSB-first
    bodies   num_blocks fragments in row-major block order, each the
             pack_body-format stream of the fragment at the parent's own rates
    pad      explicit all-zero padding to target_bytes (only when given)
    checksum 32 bytes, SHA-256 of everything before it

Every bit plane is canonical: slack after the last content bit must be zero.
``decode_projection`` verifies the exact total length, the checksum, the
declared geometry, the tag range and the padding BEFORE expanding any plane,
so truncated, corrupt or misdeclared blobs are refused without unbounded
allocation.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from tessera.alphabet import require_hardware_byte_grid
from tessera.decode import replay_window, require_untransformed
from tessera.manifest import BodyKind, ScalePlaneKind
from tessera.wire import pack_uniform, unpack_uniform

FORMAT_NAME = "pq-block-reference-wire"
FORMAT_VERSION = 1
MAGIC = b"PQBLKWR1"
GRID_E4M3FN = 1
CHECKSUM_BYTES = 32

#: 8s magic, u16 header_bytes, u8 version, u8 tag_bits, u8 flags, u8 grid_id,
#: then twenty u32 fields (see ``_HEADER_FIELDS`` for the order).
_HEADER_STRUCT = struct.Struct("<8sHBBBB" + "I" * 20)
HEADER_BYTES = _HEADER_STRUCT.size  # 94
_HEADER_FIELDS = (
    "block_rows", "block_cols", "num_row_blocks", "num_col_blocks",
    "num_parents", "rows", "cols", "window_bits", "span", "state_bits",
    "alphabet_bytes", "meta_bytes", "tag_bytes", "state_bytes",
    "state_entries", "body_bytes", "pad_bytes", "num_distinct_luts",
    "reserved0", "reserved1",
)

# Allocation guards on the read path, applied before any plane is expanded.
_MAX_WINDOW_BITS = 16  # one deduplicated table is 2^L bytes
_MAX_TAG_BITS = 32


class BlockWireFormatError(ValueError):
    """The blob, the parents or the geometry violate the research format."""


def _fail(message: str) -> None:
    raise BlockWireFormatError(message)


# ---------------------------------------------------------------------------
# Parent validation and shared planning
# ---------------------------------------------------------------------------

def _validate_parents(parents: Sequence[Any]) -> List[Dict[str, Any]]:
    """Check the family/shape/span/untransformed contract of every parent.

    Every parent must be a whole ``tessera.unit_artifact.ParsedUnit``: an
    untransformed WINDOW body over a scalar 256-code hardware byte grid, a
    CHANNEL scale plane, span 1, zero initial state, and the full-projection
    shape every other parent shares.  Returns plain numpy views of the fields
    the wire carries.
    """
    if not parents:
        _fail("pack needs at least one parent")
    prepared: List[Dict[str, Any]] = []
    ref_grid = None
    for index, parsed in enumerate(parents):
        unit = getattr(parsed, "unit", None)
        grid = getattr(parsed, "grid", None)
        if unit is None or grid is None:
            _fail(f"parent {index} is not a tessera ParsedUnit (needs .unit and .grid)")
        if getattr(unit, "body", BodyKind.TCQ) is not BodyKind.WINDOW:
            _fail(
                f"parent {index} body is {BodyKind(getattr(unit, 'body', BodyKind.TCQ)).name}, "
                "the reference wire packs WINDOW bodies only"
            )
        if getattr(unit, "scale_plane", ScalePlaneKind.S6B) is not ScalePlaneKind.CHANNEL:
            _fail(f"parent {index} scale plane is not CHANNEL")
        span = int(getattr(unit, "span", 1))
        if span != 1:
            _fail(f"parent {index} span is {span}, the reference wire packs span 1 only")
        require_untransformed(unit, f"{FORMAT_NAME} parent {index}")
        if getattr(unit, "initial_state", None) is not None:
            _fail(
                f"parent {index} carries a nonzero initial state; the wire packs "
                "whole projections whose row-0 state is zero"
            )
        require_hardware_byte_grid(grid, purpose=f"{FORMAT_NAME} parent {index}")
        window_bits = int(getattr(unit, "window_bits", 0))
        if not 1 <= window_bits <= _MAX_WINDOW_BITS:
            _fail(f"parent {index} window_bits {window_bits} outside 1..{_MAX_WINDOW_BITS}")
        table = getattr(unit, "window_codes", None)
        if table is None or table.numel() != (1 << window_bits):
            _fail(
                f"parent {index} window table holds "
                f"{None if table is None else table.numel()} entries, "
                f"window_bits {window_bits} needs {1 << window_bits}"
            )
        if int(table.max()) >= int(grid.size):
            _fail(f"parent {index} window table names a code outside its {grid.size}-code grid")
        rates = tuple(int(r) for r in unit.rates)
        rows, cols = (int(s) for s in unit.body_bits.shape)
        if len(rates) != cols:
            _fail(f"parent {index} carries {len(rates)} rates for {cols} columns")
        bad = [r for r in rates if not 0 <= r <= window_bits]
        if bad:
            _fail(f"parent {index} has rates {sorted(set(bad))} outside 0..{window_bits}")
        scale_rows = getattr(unit, "scale_rows", None)
        if scale_rows is None or scale_rows.numel() != rows:
            _fail(f"parent {index} CHANNEL plane needs {rows} fp16 row scales")
        if scale_rows.dtype != torch.float16:
            _fail(f"parent {index} row scales are {scale_rows.dtype}, expected float16")
        if unit.body_bits.dtype != torch.uint8:
            _fail(f"parent {index} body bits are {unit.body_bits.dtype}, expected uint8")
        scale_global = float(unit.scale_global)
        if not np.isfinite(scale_global) or float(np.float32(scale_global)) != scale_global:
            _fail(
                f"parent {index} scale_global {scale_global!r} is not finite fp32-exact; "
                "the wire refuses a global that would not decode to the parent's own weights"
            )
        native = np.asarray(grid.native, dtype=np.uint8)
        if ref_grid is None:
            ref_grid = (str(grid.name), native.tobytes())
        elif (str(grid.name), native.tobytes()) != ref_grid:
            _fail(
                f"parent {index} grid {grid.name} disagrees with parent 0's grid "
                f"{ref_grid[0]}; one blob carries one alphabet"
            )
        prepared.append({
            "unit": unit,
            "grid": grid,
            "rows": rows,
            "cols": cols,
            "window_bits": window_bits,
            "rates": np.asarray(rates, dtype=np.int64),
            "body": unit.body_bits.detach().cpu().contiguous().numpy(),
            "table": table.detach().cpu().numpy().astype(np.uint8, copy=False).tobytes(),
            "scale_rows": scale_rows.detach().cpu().contiguous().numpy(),
            "scale_global": scale_global,
        })
    first = prepared[0]
    for index, entry in enumerate(prepared[1:], start=1):
        if (entry["rows"], entry["cols"]) != (first["rows"], first["cols"]):
            _fail(
                f"parent {index} shape {entry['rows']}x{entry['cols']} disagrees with "
                f"parent 0's {first['rows']}x{first['cols']}"
            )
        if entry["window_bits"] != first["window_bits"]:
            _fail(f"parent {index} window_bits disagrees with parent 0's")
    return prepared


def _validate_geometry(
    parents: Sequence[Any], block_rows: int, block_cols: int
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    block_rows = int(block_rows)
    block_cols = int(block_cols)
    if block_rows % 8 or block_rows < 8:
        _fail(f"block_rows {block_rows} must be a positive multiple of 8")
    if block_cols < 1:
        _fail(f"block_cols {block_cols} must be positive")
    prepared = _validate_parents(parents)
    rows, cols = prepared[0]["rows"], prepared[0]["cols"]
    if rows % block_rows or cols % block_cols:
        _fail(
            f"{rows}x{cols} does not tile {block_rows}x{block_cols}; every dimension "
            "must be a whole number of blocks"
        )
    geometry = {
        "rows": rows,
        "cols": cols,
        "block_rows": block_rows,
        "block_cols": block_cols,
        "num_row_blocks": rows // block_rows,
        "num_col_blocks": cols // block_cols,
        "num_parents": len(prepared),
    }
    geometry["num_blocks"] = geometry["num_row_blocks"] * geometry["num_col_blocks"]
    return prepared, geometry


def _tag_bits(num_parents: int) -> int:
    return max(1, (num_parents - 1).bit_length())


def _dedup_tables(prepared: List[Dict[str, Any]]) -> Tuple[List[bytes], List[int]]:
    """Deduplicate the parents' native LUTs by identical bytes."""
    order: List[bytes] = []
    index_of: Dict[bytes, int] = {}
    indices: List[int] = []
    for entry in prepared:
        table = entry["table"]
        if table not in index_of:
            index_of[table] = len(order)
            order.append(table)
        indices.append(index_of[table])
    return order, indices


def _plan(
    parents: Sequence[Any], block_rows: int, block_cols: int
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, Any]]:
    """Every selection-independent size, once, for all three entry points."""
    prepared, geometry = _validate_geometry(parents, block_rows, block_cols)
    tables, table_indices = _dedup_tables(prepared)
    rows, cols = geometry["rows"], geometry["cols"]
    num_parents = geometry["num_parents"]
    nrb, ncb = geometry["num_row_blocks"], geometry["num_col_blocks"]
    window_bits = prepared[0]["window_bits"]

    lut_table_bytes = sum(2 + len(t) for t in tables)
    meta = {
        "distinct_luts": len(tables),
        "lut_table_bytes": lut_table_bytes,
        "lut_index_bytes": 2 * num_parents,
        "scale_global_bytes": 4 * num_parents,
        "scale_rows_bytes": 2 * rows * num_parents,
        "rate_bytes": cols * num_parents,
    }
    meta["total"] = sum(v for k, v in meta.items() if k != "distinct_luts")
    tag_bits = _tag_bits(num_parents)
    tag_bytes = (geometry["num_blocks"] * tag_bits + 7) // 8
    state_entry_bytes = (block_cols * window_bits + 7) // 8
    state_entries = (nrb - 1) * ncb
    state_bytes = state_entry_bytes * state_entries
    alphabet_bytes = 256
    fixed = (
        HEADER_BYTES + alphabet_bytes + meta["total"] + tag_bytes
        + state_bytes + CHECKSUM_BYTES
    )
    sizes = {
        "tables": tables,
        "table_indices": table_indices,
        "meta": meta,
        "tag_bits": tag_bits,
        "tag_bytes": tag_bytes,
        "state_entry_bytes": state_entry_bytes,
        "state_entries": state_entries,
        "state_bytes": state_bytes,
        "alphabet_bytes": alphabet_bytes,
        "fixed_bytes": fixed,
    }
    return prepared, geometry, sizes


# ---------------------------------------------------------------------------
# Body and state streams (batched, bit-exact with tessera.wire)
# ---------------------------------------------------------------------------

def _expand_bits(values: np.ndarray, width: int) -> np.ndarray:
    """MSB-first bit expansion, batched on a trailing axis."""
    if values.size and (int(values.min()) < 0 or int(values.max()) >= (1 << width)):
        _fail(
            f"value out of range for a {width}-bit field: "
            f"[{int(values.min())}, {int(values.max())}]"
        )
    shifts = np.arange(width - 1, -1, -1, dtype=np.int64)
    return ((values.astype(np.int64)[..., None] >> shifts) & 1).astype(np.uint8)


def _collapse_bits(bits: np.ndarray, width: int) -> np.ndarray:
    """Inverse of :func:`_expand_bits` on the same trailing axis."""
    shifts = np.arange(width - 1, -1, -1, dtype=np.int64)
    return (bits.astype(np.int64) << shifts).sum(axis=-1)


def _pack_group(values: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Pack one (parent, column-block) group of fragments, one row per block.

    ``values`` is ``[n, block_rows, block_cols]`` body-bit values in original
    row-major order; ``widths`` the parent's rates for the fragment's columns.
    Byte for byte this is ``wire.pack_body(fragment, rates, span=1)``, batched
    over the group: column-major stream, each position's field width most
    significant first, zero padding to the byte.
    """
    n, block_rows, block_cols = values.shape
    streams: List[np.ndarray] = []
    total = 0
    for k in range(block_cols):
        width = int(widths[k])
        column = values[:, :, k]
        if width == 0:
            if column.size and bool((column != 0).any()):
                _fail(
                    "nonzero body bits in a zero-rate column: the fragment stores "
                    "no bits there, so the bytes could not be canonical"
                )
            continue
        bits = _expand_bits(column, width).reshape(n, -1)
        streams.append(bits)
        total += block_rows * width
    if total == 0:
        return np.zeros((n, 0), dtype=np.uint8)
    stream = np.concatenate(streams, axis=1)
    pad = (-total) % 8
    if pad:
        stream = np.concatenate([stream, np.zeros((n, pad), dtype=np.uint8)], axis=1)
    return np.packbits(stream, axis=1, bitorder="big")


def _unpack_group(packed: np.ndarray, widths: np.ndarray, block_rows: int) -> np.ndarray:
    """Inverse of :func:`_pack_group`: body-bit values ``[n, block_rows, bc]``."""
    n = packed.shape[0]
    block_cols = len(widths)
    total = sum(block_rows * int(w) for w in widths)
    bits = np.unpackbits(packed, axis=1, bitorder="big")
    if bits.shape[1] < total:
        _fail(f"body fragment holds {bits.shape[1]} bits, the rates need {total}")
    if total < bits.shape[1] and bool(bits[:, total:].any()):
        _fail("body fragment: non-zero pad bits after the last content bit")
    bits = bits[:, :total]
    out = np.zeros((n, block_rows, block_cols), dtype=np.int64)
    cursor = 0
    for k in range(block_cols):
        width = int(widths[k])
        if width == 0:
            continue
        take = block_rows * width
        out[:, :, k] = _collapse_bits(bits[:, cursor:cursor + take].reshape(n, block_rows, width), width)
        cursor += take
    return out


def _boundary_states(entry: Dict[str, Any], block_rows: int, num_row_blocks: int) -> np.ndarray:
    """Incoming L-bit state at every interior row boundary, from the closed form.

    ``state`` after row ``t`` is the last ``window_bits`` bits of the column's
    stream, so the state entering row block ``i`` (row ``i*block_rows``) is the
    OR of the last ``ceil(L/rate)`` body-bit rows above it, shifted by their
    tap distance -- the same recursion ``decode.replay_window`` unrolls, taken
    at every boundary in one batched expression.
    """
    bits = entry["body"].astype(np.int64)
    rates = entry["rates"]
    window_bits = entry["window_bits"]
    bounds = np.arange(1, num_row_blocks, dtype=np.int64) * block_rows
    out = np.zeros((num_row_blocks - 1, entry["cols"]), dtype=np.int64)
    mask = (1 << window_bits) - 1
    for rate in sorted(set(int(r) for r in rates)):
        columns = np.flatnonzero(rates == rate)
        if rate == 0:
            continue  # a zero-rate column stores no bits; its state stays zero
        taps = -(-window_bits // rate)
        offsets = np.arange(taps, dtype=np.int64)
        rows = np.clip(bounds[:, None, None] - 1 - offsets[None, :, None], 0, None)
        gathered = bits[rows, columns[None, None, :]]
        gathered = np.where(
            (bounds[:, None, None] - 1 - offsets[None, :, None]) >= 0, gathered, 0
        )
        out[:, columns] = ((gathered << (offsets[None, :, None] * rate)).sum(axis=1) & mask)
    return out


# ---------------------------------------------------------------------------
# Public accounting API
# ---------------------------------------------------------------------------

def body_costs(parents: Sequence[Any], block_rows: int, block_cols: int) -> np.ndarray:
    """Actual packed body length of every (block, parent) choice.

    Returns an int64 array ``[num_blocks, num_parents]`` in row-major block
    order (``b = i * num_col_blocks + j``): ``ceil(block_rows * sum(rates_p of
    the fragment's columns) / 8)`` -- the exact ``pack_body`` stream length for
    that fragment at that parent's real per-column rates.  A fragment keeps its
    parent's global column rates; nothing here pretends a fragment carries the
    parent's nominal mean bits.
    """
    prepared, geometry, _ = _plan(parents, block_rows, block_cols)
    block_rows_, block_cols_ = geometry["block_rows"], geometry["block_cols"]
    ncb = geometry["num_col_blocks"]
    costs = np.zeros((ncb, geometry["num_parents"]), dtype=np.int64)
    for p_index, entry in enumerate(prepared):
        running = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(entry["rates"])])
        sums = running[block_cols_::block_cols_] - running[:-block_cols_:block_cols_]
        costs[:, p_index] = (block_rows_ * sums + 7) // 8
    return np.tile(costs, (geometry["num_row_blocks"], 1))


def fixed_bytes(parents: Sequence[Any], block_rows: int, block_cols: int) -> int:
    """Exact selection-independent overhead of a packed blob, in bytes.

    Header + grid alphabet + per-parent metadata (deduplicated native LUTs,
    fp16 row scales, fp32 globals, rate vectors) + selection tag plane +
    incoming-state plane + checksum.  The tag and state planes' sizes follow
    from the geometry and the parent count alone, so this number does not
    depend on which parent any block selects.  A solver holding
    ``target_bytes`` has body budget ``target_bytes - fixed_bytes(...)`` and
    charges each selected block its ``body_costs`` entry; padding to
    ``target_bytes`` is the only other bytes the blob can carry.
    """
    _, _, sizes = _plan(parents, block_rows, block_cols)
    return int(sizes["fixed_bytes"])


def _validate_selection(selection: Any, geometry: Dict[str, int]) -> np.ndarray:
    array = np.asarray(selection)
    if array.dtype == bool or not np.issubdtype(array.dtype, np.integer):
        _fail(
            f"selection must hold integer parent tags, got dtype {array.dtype}"
        )
    expected = (geometry["num_row_blocks"], geometry["num_col_blocks"])
    if array.shape != expected:
        _fail(f"selection shape {array.shape} != {expected}")
    tags = array.astype(np.int64, copy=False)
    if tags.size and (int(tags.min()) < 0 or int(tags.max()) >= geometry["num_parents"]):
        _fail(
            f"selection names parent {int(tags.max())} outside 0..{geometry['num_parents'] - 1}"
        )
    return tags


def pack_projection(
    parents: Sequence[Any],
    selection: Any,
    block_rows: int,
    block_cols: int,
    target_bytes: int | None = None,
) -> Tuple[bytes, Dict[str, Any]]:
    """Pack the selected fragments into one research wire blob.

    ``selection[i, j]`` names the parent whose fragment fills block ``(i, j)``.
    Returns ``(blob, breakdown)``.  Overflow against ``target_bytes`` is
    refused; padding to ``target_bytes`` is explicit, all zero and charged in
    the breakdown.  The blob carries no whole-parent weight and no covert
    slack: every counted byte is in the breakdown.
    """
    prepared, geometry, sizes = _plan(parents, block_rows, block_cols)
    tags = _validate_selection(selection, geometry)
    num_parents = geometry["num_parents"]
    tag_bits = sizes["tag_bits"]
    window_bits = prepared[0]["window_bits"]
    block_rows_, block_cols_ = geometry["block_rows"], geometry["block_cols"]
    nrb, ncb = geometry["num_row_blocks"], geometry["num_col_blocks"]

    # -- body plane: one batched group per (parent, column block) ------------
    fragments: Dict[Tuple[int, int], bytes] = {}
    per_parent = [0] * num_parents
    max_fragment = 0
    local_rows = np.arange(block_rows_, dtype=np.int64)
    for j in range(ncb):
        column = tags[:, j]
        for p_index in np.unique(column):
            p_index = int(p_index)
            rows_of = np.flatnonzero(column == p_index)
            entry = prepared[p_index]
            widths = entry["rates"][j * block_cols_:(j + 1) * block_cols_]
            columns = entry["body"][:, j * block_cols_:(j + 1) * block_cols_]
            values = columns[rows_of[:, None] * block_rows_ + local_rows[None, :], :]
            packed = _pack_group(values, widths)
            length = packed.shape[1]
            max_fragment = max(max_fragment, length)
            for local, i in enumerate(rows_of):
                fragment = packed[local].tobytes()
                fragments[(int(i), j)] = fragment
                per_parent[p_index] += len(fragment)
    body_blob = b"".join(
        fragments[(i, j)] for i in range(nrb) for j in range(ncb)
    )
    body_bytes = len(body_blob)

    # -- incoming-state plane: one entry per interior (row block, col block) --
    # entry e = (i-1)*num_col_blocks + j carries the selected parent's state
    # entering original row i*block_rows; original row 0 is known zero and
    # stores nothing.
    states = np.zeros((max(nrb - 1, 0) * ncb, block_cols_), dtype=np.int64)
    states_grid = states.reshape(nrb - 1, ncb, block_cols_) if nrb > 1 else None
    for p_index, entry in enumerate(prepared):
        boundaries = _boundary_states(entry, block_rows_, nrb)
        if boundaries.size == 0:
            continue
        bnd = boundaries.reshape(nrb - 1, ncb, block_cols_)
        ii, jj = np.nonzero(tags[1:] == p_index)
        if ii.size:
            states_grid[ii, jj] = bnd[ii, jj]
    state_packed = _pack_group(
        states.reshape(states.shape[0], 1, block_cols_),
        np.full(block_cols_, window_bits, dtype=np.int64),
    )
    state_blob = state_packed.tobytes() if state_packed.size else b""

    # -- tag plane -----------------------------------------------------------
    tag_blob = pack_uniform(
        torch.from_numpy(tags.reshape(-1)), tag_bits
    ) if tags.size else b""

    # -- meta plane ----------------------------------------------------------
    tables = sizes["tables"]
    table_indices = sizes["table_indices"]
    meta = bytearray()
    meta += struct.pack("<H", len(tables))
    for table in tables:
        meta += struct.pack("<H", len(table)) + table
    meta += b"".join(struct.pack("<H", i) for i in table_indices)
    meta += b"".join(struct.pack("<f", e["scale_global"]) for e in prepared)
    for entry in prepared:
        meta += entry["scale_rows"].astype("<f2").tobytes()
    for entry in prepared:
        meta += entry["rates"].astype(np.uint8).tobytes()
    meta_blob = bytes(meta)

    grid = prepared[0]["grid"]
    alphabet_blob = np.asarray(grid.native, dtype=np.uint8).tobytes()

    content = sizes["fixed_bytes"] + body_bytes
    if target_bytes is not None:
        target = int(target_bytes)
        if target < content:
            _fail(
                f"overflow: packed content needs {content} bytes, target_bytes "
                f"holds {target}; nothing is truncated or silently dropped"
            )
        pad_bytes = target - content
    else:
        pad_bytes = 0
    pad_blob = b"\x00" * pad_bytes
    if pad_bytes and any(pad_blob):
        _fail("padding must be all zero")

    header = _HEADER_STRUCT.pack(
        MAGIC, HEADER_BYTES, FORMAT_VERSION, tag_bits, 0, GRID_E4M3FN,
        block_rows_, block_cols_, nrb, ncb, num_parents,
        geometry["rows"], geometry["cols"], window_bits, 1, window_bits,
        len(alphabet_blob), len(meta_blob), len(tag_blob), len(state_blob),
        sizes["state_entries"], body_bytes, pad_bytes, len(tables), 0, 0,
    )
    blob = bytearray()
    blob += header
    blob += alphabet_blob
    blob += meta_blob
    blob += tag_blob
    blob += state_blob
    blob += body_blob
    blob += pad_blob
    digest = hashlib.sha256(bytes(blob)).digest()
    blob += digest
    total = len(blob)
    if total != content + pad_bytes + CHECKSUM_BYTES:
        _fail("internal accounting error: assembled length disagrees with the plan")

    breakdown = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "grid": str(grid.name),
        "grid_id": GRID_E4M3FN,
        "geometry": dict(geometry),
        "num_parents": num_parents,
        "window_bits": window_bits,
        "span": 1,
        "tag_bits": tag_bits,
        "state_bits": window_bits,
        "header_bytes": HEADER_BYTES,
        "alphabet_bytes": len(alphabet_blob),
        "meta_bytes": dict(sizes["meta"]),
        "tag_bytes": len(tag_blob),
        "tag_count": int(tags.size),
        "state_entries": sizes["state_entries"],
        "state_entry_bytes": sizes["state_entry_bytes"],
        "state_bytes": len(state_blob),
        "body_bytes": body_bytes,
        "body_fragment_bytes": {
            "per_parent_total": per_parent,
            "max_fragment": max_fragment,
        },
        "pad_bytes": pad_bytes,
        "checksum_bytes": CHECKSUM_BYTES,
        "checksum": digest.hex(),
        "fixed_bytes": int(sizes["fixed_bytes"]),
        "content_bytes": content,
        "total_bytes": total,
        "target_bytes": None if target_bytes is None else int(target_bytes),
        "accounting_consistent": total == (
            HEADER_BYTES + len(alphabet_blob) + len(meta_blob) + len(tag_blob)
            + len(state_blob) + body_bytes + pad_bytes + CHECKSUM_BYTES
        ),
    }
    return bytes(blob), breakdown


# ---------------------------------------------------------------------------
# Reference decode
# ---------------------------------------------------------------------------

def _meta_plane(data: bytes, start: int, length: int, window_bits: int,
                num_parents: int, rows: int, cols: int) -> List[Dict[str, Any]]:
    """Parse the meta plane with bounds checks at every step."""
    table_bytes = 1 << window_bits
    cursor = start
    end = start + length
    if cursor + 2 > end:
        _fail("meta plane truncated before the LUT count")
    (distinct,) = struct.unpack_from("<H", data, cursor)
    cursor += 2
    if distinct < 1 or distinct > num_parents:
        _fail(f"meta declares {distinct} distinct LUTs for {num_parents} parents")
    tables: List[bytes] = []
    for _ in range(distinct):
        if cursor + 2 > end:
            _fail("meta plane truncated inside the LUT table list")
        (declared,) = struct.unpack_from("<H", data, cursor)
        cursor += 2
        if declared != table_bytes:
            _fail(
                f"meta declares a {declared}-byte LUT, window_bits {window_bits} "
                f"needs {table_bytes}"
            )
        if cursor + declared > end:
            _fail("meta plane truncated inside a LUT table")
        tables.append(data[cursor:cursor + declared])
        cursor += declared
    parents: List[Dict[str, Any]] = []
    if cursor + 2 * num_parents > end:
        _fail("meta plane truncated inside the LUT index")
    indices = struct.unpack_from("<" + "H" * num_parents, data, cursor)
    cursor += 2 * num_parents
    for index in indices:
        if index >= distinct:
            _fail(f"LUT index {index} outside the {distinct} stored tables")
    if cursor + 4 * num_parents > end:
        _fail("meta plane truncated inside the global scales")
    globals_ = struct.unpack_from("<" + "f" * num_parents, data, cursor)
    cursor += 4 * num_parents
    if cursor + 2 * rows * num_parents > end:
        _fail("meta plane truncated inside the row scales")
    scales = np.frombuffer(data, dtype="<f2", count=rows * num_parents, offset=cursor)
    cursor += 2 * rows * num_parents
    if cursor + cols * num_parents > end:
        _fail("meta plane truncated inside the rate vectors")
    for p_index in range(num_parents):
        rates = np.frombuffer(data, dtype=np.uint8, count=cols, offset=cursor + p_index * cols)
        if rates.size and int(rates.max()) > window_bits:
            _fail(f"parent {p_index} declares rate {int(rates.max())} above window_bits")
        parents.append({
            "table": tables[indices[p_index]],
            "scale_global": globals_[p_index],
            "scale_rows": scales[p_index * rows:(p_index + 1) * rows].astype(np.float16),
            "rates": rates.astype(np.int64),
        })
        if not np.isfinite(parents[p_index]["scale_global"]):
            _fail(f"parent {p_index} carries a non-finite global scale")
    cursor += cols * num_parents
    if cursor != end:
        _fail(f"meta plane has {end - cursor} trailing bytes")
    return parents


def _fragment_offsets(
    parents: List[Dict[str, Any]], tags: np.ndarray, geometry: Dict[str, int]
) -> np.ndarray:
    """Byte offset of every fragment, from the tags and the parents' rates."""
    block_cols = geometry["block_cols"]
    ncb = geometry["num_col_blocks"]
    block_rows = geometry["block_rows"]
    cumsums = [
        np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(p["rates"])])
        for p in parents
    ]
    flat = tags.reshape(-1)
    j_index = np.arange(flat.size, dtype=np.int64) % ncb
    starts = j_index * block_cols
    sums = np.stack(
        [cumsums[p][starts + block_cols] - cumsums[p][starts] for p in range(len(parents))]
    )
    bits = block_rows * sums[flat, np.arange(flat.size, dtype=np.int64)]
    return np.concatenate(
        [np.zeros(1, dtype=np.int64), np.cumsum((bits + 7) // 8)]
    )


def decode_projection(blob: bytes, device: str = "cpu") -> torch.Tensor:
    """Decode a packed blob to fp32 weights, from the blob alone.

    Verifies the exact total length, the geometry declarations, the trailing
    SHA-256 checksum, the canonical padding and the tag range BEFORE any plane
    is expanded, then reconstructs each fragment with the existing tessera
    primitives (the ``unpack_body`` stream layout and ``replay_window``) and
    returns ``torch.float32 [rows, cols]`` on ``device``.  For an untransformed
    E4M3/CHANNEL parent the fragment equals ``stock_dequant`` exactly, at any
    row boundary.
    """
    if isinstance(blob, (bytearray, memoryview)):
        data = bytes(blob)
    elif isinstance(blob, bytes):
        data = blob
    else:
        _fail(f"decode needs bytes, got {type(blob).__name__}")
    if len(data) < HEADER_BYTES + CHECKSUM_BYTES:
        _fail(f"blob is {len(data)} bytes, shorter than header + checksum")
    fields = _HEADER_STRUCT.unpack_from(data, 0)
    (magic, header_bytes, version, tag_bits, flags, grid_id) = fields[:6]
    named = dict(zip(_HEADER_FIELDS, fields[6:]))
    if magic != MAGIC:
        _fail("bad magic: this is not a pq-block-reference-wire blob")
    if version != FORMAT_VERSION or header_bytes != HEADER_BYTES:
        _fail(f"unsupported format version {version} (header {header_bytes} bytes)")
    if flags != 0 or named["reserved0"] != 0 or named["reserved1"] != 0:
        _fail("reserved header fields must be zero")
    if grid_id != GRID_E4M3FN or named["alphabet_bytes"] != 256:
        _fail(f"unknown grid id {grid_id} or alphabet of {named['alphabet_bytes']} bytes")
    if tag_bits < 1 or tag_bits > _MAX_TAG_BITS:
        _fail(f"tag_bits {tag_bits} outside 1..{_MAX_TAG_BITS}")

    block_rows = int(named["block_rows"])
    block_cols = int(named["block_cols"])
    nrb = int(named["num_row_blocks"])
    ncb = int(named["num_col_blocks"])
    num_parents = int(named["num_parents"])
    rows = int(named["rows"])
    cols = int(named["cols"])
    window_bits = int(named["window_bits"])
    if block_rows % 8 or block_rows < 8 or block_cols < 1:
        _fail(f"bad block geometry {block_rows}x{block_cols}")
    if min(nrb, ncb, num_parents, rows, cols) < 1:
        _fail("header declares an empty projection")
    if nrb * block_rows != rows or ncb * block_cols != cols:
        _fail(
            f"header geometry does not tile: {nrb}x{block_rows} rows and "
            f"{ncb}x{block_cols} columns against {rows}x{cols}"
        )
    if int(named["span"]) != 1 or int(named["state_bits"]) != window_bits:
        _fail("the format packs span-1 window bodies with state_bits == window_bits")
    if not 1 <= window_bits <= _MAX_WINDOW_BITS:
        _fail(f"window_bits {window_bits} outside 1..{_MAX_WINDOW_BITS}")
    if num_parents > 1 and tag_bits != _tag_bits(num_parents):
        _fail(f"tag_bits {tag_bits} disagrees with {num_parents} parents")

    alphabet_bytes = int(named["alphabet_bytes"])
    meta_bytes = int(named["meta_bytes"])
    tag_bytes = int(named["tag_bytes"])
    state_bytes = int(named["state_bytes"])
    state_entries = int(named["state_entries"])
    body_bytes = int(named["body_bytes"])
    pad_bytes = int(named["pad_bytes"])
    needed = (
        HEADER_BYTES + alphabet_bytes + meta_bytes + tag_bytes + state_bytes
        + body_bytes + pad_bytes + CHECKSUM_BYTES
    )
    if needed != len(data):
        _fail(
            f"declared planes need {needed} bytes, the blob holds {len(data)}; "
            "refusing a truncated or misdeclared blob before any allocation"
        )
    if hashlib.sha256(data[:-CHECKSUM_BYTES]).digest() != data[-CHECKSUM_BYTES:]:
        _fail("checksum mismatch: the blob is corrupt")

    cursor = HEADER_BYTES + alphabet_bytes
    parents = _meta_plane(data, cursor, meta_bytes, window_bits, num_parents, rows, cols)
    cursor += meta_bytes

    pad_start = cursor + tag_bytes + state_bytes + body_bytes
    if pad_bytes and any(data[pad_start:pad_start + pad_bytes]):
        _fail("padding bytes are not zero")

    num_blocks = nrb * ncb
    tags = unpack_uniform(
        bytes(data[cursor:cursor + tag_bytes]), num_blocks, tag_bits
    ).numpy()
    if tags.size and (int(tags.min()) < 0 or int(tags.max()) >= num_parents):
        _fail(
            f"selection tag {int(tags.max())} outside 0..{num_parents - 1}"
        )
    tags = tags.reshape(nrb, ncb)
    cursor += tag_bytes

    entry_bytes = (block_cols * window_bits + 7) // 8
    if state_entries != (nrb - 1) * ncb or state_bytes != entry_bytes * state_entries:
        _fail("declared state plane disagrees with the geometry")
    state_values = np.zeros((state_entries, block_cols), dtype=np.int64)
    if state_entries:
        state_values = _unpack_group(
            np.frombuffer(data, dtype=np.uint8, count=state_bytes,
                          offset=cursor).reshape(state_entries, entry_bytes),
            np.full(block_cols, window_bits, dtype=np.int64),
            1,
        ).reshape(state_entries, block_cols)
    state_grid = state_values.reshape(nrb - 1, ncb, block_cols) if nrb > 1 else None
    cursor += state_bytes

    geometry = {
        "rows": rows,
        "cols": cols,
        "block_rows": block_rows,
        "block_cols": block_cols,
        "num_row_blocks": nrb,
        "num_col_blocks": ncb,
        "num_parents": num_parents,
    }
    offsets = _fragment_offsets(parents, tags, geometry)
    if int(offsets[-1]) != body_bytes:
        _fail(
            f"fragment lengths sum to {int(offsets[-1])} bytes, the body plane "
            f"declares {body_bytes}"
        )
    body_region = data[cursor:cursor + body_bytes]

    device = torch.device(device)
    out = torch.zeros(rows, cols, dtype=torch.float32, device=device)
    alphabet = torch.from_numpy(
        np.frombuffer(data, dtype=np.uint8, count=256, offset=HEADER_BYTES).copy()
    ).to(device)
    alphabet = alphabet.view(torch.float8_e4m3fn).float()
    if bool(torch.isnan(alphabet).any()):
        _fail("the alphabet plane holds an E4M3FN NaN byte")

    widths_all = [p["rates"] for p in parents]
    luts = [
        torch.from_numpy(np.frombuffer(p["table"], dtype=np.uint8).copy()).to(device)
        for p in parents
    ]
    scales = [
        torch.from_numpy(p["scale_rows"].astype(np.float16)).to(device).float()
        * float(np.float32(p["scale_global"]))
        for p in parents
    ]
    local_rows = np.arange(block_rows, dtype=np.int64)
    for j in range(ncb):
        c0 = j * block_cols
        for p_index in np.unique(tags[:, j]):
            p_index = int(p_index)
            rates = widths_all[p_index][c0:c0 + block_cols]
            rows_of = np.flatnonzero(tags[:, j] == p_index)
            flat_blocks = rows_of * ncb + j
            lengths = offsets[flat_blocks + 1] - offsets[flat_blocks]
            if len(set(int(x) for x in lengths)) != 1:
                _fail("fragments of one (parent, column block) group disagree in length")
            span_bytes = int(lengths[0])
            packed = np.frombuffer(
                body_region, dtype=np.uint8, count=span_bytes * rows_of.size,
                offset=int(offsets[flat_blocks[0]]),
            ).reshape(rows_of.size, span_bytes)
            values = _unpack_group(packed, rates, block_rows)
            for k in range(block_cols):
                rate = int(rates[k])
                column = c0 + k
                if rate == 0:
                    states = torch.zeros(
                        block_rows, rows_of.size, dtype=torch.int64, device=device
                    )
                else:
                    bits = torch.from_numpy(
                        np.ascontiguousarray(values[:, :, k].T)
                    ).to(device)
                    # zero at original row 0, the stored parent state at every
                    # interior boundary, per block and per column
                    initial = np.zeros(rows_of.size, dtype=np.int64)
                    interior = rows_of > 0
                    if interior.any():
                        initial[interior] = state_grid[rows_of[interior] - 1, j, k]
                    states = replay_window(
                        bits, window_bits, rate,
                        torch.from_numpy(initial).to(device),
                    ).T
                codes = luts[p_index][states].long()
                global_rows = torch.from_numpy(
                    rows_of[:, None] * block_rows + local_rows[None, :]
                ).to(device)
                weights = alphabet[codes] * scales[p_index][global_rows]
                out[global_rows, column] = weights
    return out


__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "HEADER_BYTES",
    "MAGIC",
    "BlockWireFormatError",
    "body_costs",
    "fixed_bytes",
    "pack_projection",
    "decode_projection",
]
