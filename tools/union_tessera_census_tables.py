#!/usr/bin/env python3
"""Union merged Tessera census tables that priced disjoint rungs of one census.

A census extension (a new rate band, a new family) is a second campaign over
the same sealed census: the same weights, Hessians, calibration draw, encoder
and projection, pricing (unit, format) cells the first campaign did not.  Each
campaign merges into its own table directory::

    <table>/cost.pkl                         allocator cost table
    <table>/cost.anchors.json                checkpoint manifest (the seal)
    <table>/cost.anchors.json.parts/units/   per-unit journal envelopes
    <table>/cache/wire/                      measured wire blobs
    <table>/cache/hessian_capture.references.json
    <table>/cache/input_scales.safetensors

This tool writes one directory of the same shape that every downstream reader
accepts as if one campaign had priced every cell::

    python tools/union_tessera_census_tables.py \\
        --table r1024=<dirA> --table e2m1=<dirB> --out <outdir> [--dry-run] [--resume]

Nothing is re-encoded, re-priced or interpolated.  Every check below runs
before the first byte is written; ``--dry-run`` runs all of them and writes
nothing.

What must be equal (refused by name otherwise)
----------------------------------------------
* Per unit, every sealed input except the menu: ``weight``, ``hessian``,
  ``scoring_rows``, ``input_global_scale`` (``identity.units[q]``).
* Every top-level identity key except ``units``, ``settings`` and
  ``family_restriction``: ``encoder_source_sha256``,
  ``prismaquant_source_sha256``, ``calibration``, ``campaign_schema``,
  ``currency``, ``encoder_recipe``, ``expert_projection``,
  ``input_global_scale_policy``, ``serving_scope``, ...
* Cost-table provenance: every key without a merge rule below, including
  ``campaign_scope``, ``tessera_expert_projection``, ``cost_mode``,
  ``model``/``nsamples``/``seqlen``/``layer_stride``, ``calibration_cache``;
  every ``hessian`` field except the capture path and digest (so
  ``hessian.calibration_identity`` and ``reference_binding``);
  ``activation_static_scales`` except its path, with ``units`` equal on every
  shared unit.
* A (unit, format) cell priced by two tables: the rows, after the capture
  digest restamp below, ignoring only wall-clock fields (``encode_seconds``,
  ``encode_seconds_accounting``, an anchor's ``seconds``) and the anchor
  batch width they were divided by (``encoding_batch_size``) -- a campaign that
  re-measured a cell the other already priced agrees when the price and the
  wire bytes agree, and refuses otherwise.  A per-family side table (``surfaces``,
  ``leave_one_anchor_out``, ``anchor_counts``) keyed by the same family in two
  tables: the entry.  Two tables pricing *different* rungs of *one* family
  therefore refuse: their rate surfaces were fitted to different anchors and
  a union would need a refit, which this tool does not do.
* A journal anchor or wire receipt for one (unit, format) in two tables.
* A wire blob name referenced by two tables: the receipt, then the files
  (``os.path.samefile`` first; size and SHA-256 only when they are different
  inodes).  No other blob byte is read.
* ``cache/input_scales.safetensors``: SHA-256.

The Hessian reference handoff is not byte-identical across tables
---------------------------------------------------------------------
``cache/hessian_capture.references.json`` commits to the Hessians of the
units its campaign priced, and its ``capture_sha256`` is computed over that
unit set (``tessera.hessian_capture.capture_sha256_from_units``).  Two tables
over different unit sets therefore carry different files and digests even
when every shared Hessian is identical, and every cost row carries its own
table's digest in ``hessian_identity.capture_sha256``.  The allocator refuses
a table holding two digests (``tessera_menu.assert_uniform_hessian_identity``)
unless it can open both reference files through the table's own hash-bound
inputs and finds equal per-unit content (RobTand/prismaquant#1270); a union
has no such chain to follow.  So instead of asserting the files
identical, the union requires equal ``schema``, ``canonical_capture``,
``provenance``, ``counts``, ``load_policy`` and ``census.sha256``; equal
commitments on every shared unit; each table's rows and provenance to carry
its own descriptor's digest; and one table's descriptor to cover every unit
of the union.  That descriptor is hardlinked as the union's handoff, and every
row of every table is restamped to its digest -- the same restamp
``dispatch_tessera_campaign.merge_payloads`` applies when rows merge.  The
union then passes ``assert_uniform_hessian_identity`` before anything is
written.  One file serves every table only because they bind one canonical
capture and one census, so that is asserted explicitly: each table's rows
and provenance carry a single ``reference_binding`` whose
``canonical_capture_sha256`` and ``census_sha256`` equal its handoff's
``canonical_capture.sha256`` and ``census.sha256``, and the bindings are
equal across tables.  A union no single descriptor covers is refused (building a new
descriptor needs the reference writer's commitments, not a JSON edit).

Identity shape (the sealed ``cost.anchors.json``)
-------------------------------------------------
* ``units[q]`` = the shared inputs, with ``menu`` = sorted union of the
  tables' menus.
* ``settings`` = only the settings keys every table binds with one value.
  The keys that differ (the rate band; ``family_restriction`` when one table
  was restricted) move to ``union.tables[<name>].settings``, which holds each
  table's complete settings.  Readers of ``identity.settings``:
  ``tools/reseal_campaign_identity.py:375`` and ``:639`` (pin
  classification, ``.get`` with a mapping guard -- a union is not a reseal
  row); ``tessera_campaign.py:1927``/``:1981`` build it rather than read it;
  ``tessera_campaign.py:2816`` (seed adoption) reads only ``currency``,
  ``calibration``, ``input_global_scale_policy`` and per-unit
  ``scoring_rows``/``input_global_scale``, all equal here.
* ``family_restriction`` is omitted at top level unless every table binds the
  same one; each table's lives under ``union.tables[<name>]``.  Its readers
  are the dispatcher's row merge (``dispatch_tessera_campaign.py:2323``) and
  the campaign's identity builder -- a restriction written for one table
  would misdescribe the other's cells.
* ``union`` = ``{schema: prismaquant.tessera_census_union.v1, tables:
  {<name>: {identity_sha256, settings[, family_restriction]}}}``, inside the
  identity so the seal covers it.
* ``identity_sha256`` is the checkpoint canonical digest
  (``cost_stage_checkpoint.canonical_json_sha256``), computed by streaming
  the same bytes; ``tessera_census_cache.seal_roster``
  (``tessera_census_cache.py:98``) recomputes it and the tests prove equality.
* ``identity_migration`` (beside the identity, outside the seal) = the tables'
  reseal records merged by ``cost_stage_checkpoint.merge_identity_migrations``,
  the rule the dispatcher's row merge applies: one record per proof bundle and
  pin pair.  ``provenance.identity_migration`` in ``cost.pkl`` follows the
  same rule, and each table's own records stay under
  ``provenance.union.tables``.  Every other key beside the identity must be
  equal across tables.

The manifest is written in the layout the campaign and the dispatcher write
(``json.dumps(indent=2, sort_keys=True, ensure_ascii=False)``,
``dispatch_tessera_campaign.py:2346``), byte for byte, but streamed.

Journal envelopes
-----------------
For each union unit the tables' envelopes are loaded through
``cost_stage_checkpoint._load_unit`` against each table's own seal; anchors
are concatenated in ``--table`` order (a duplicate format must be an equal
anchor and is kept once), ``wire_records`` are unioned, and the state is
written with ``write_unit`` under the union digest, so
``load_selected_wire_records`` (``tessera_census_cache.py:219``) accepts it.
A routed unit's cost-table receipts must equal its journal receipts (the
check ``tessera_census_cache.py:312`` makes at build time, made here first).

Wires and ``os.link``
---------------------
Every receipt's blob is hardlinked into ``<out>/cache/wire``;
``_verify_dense_blob`` (``tessera_census_cache.py:237``) and
``verify_expert_wire_record`` refuse a symlink or a file whose resolved
parent is not the one ``wire_dir``.  A failed ``os.link`` (``EXDEV``,
``EPERM``, a missing blob) refuses the union.  Nothing is ever copied: the
dispatcher's ``cmd_merge`` falls back to a copy, and 729 GB of wires copied
onto the census pool is the failure this tool must not have.

Cost table (``cost.pkl``)
-------------------------
* ``costs``: union of rows, capture digest restamped.
* ``formats``: sorted union.  ``tessera_expert_wires``,
  ``leave_one_anchor_out``, ``anchor_counts``: unioned per unit per key.
  ``non_interpolable``: de-duplicated union in ``canonical_refusals`` order.
* ``menu_sizes[q] = len(identity.units[q].menu)`` of the union.  This is
  what the campaign writes (``tessera_campaign.py:6737`` over the same
  ``menus`` whose names it seals at ``:2018``); it is *not* ``len(costs[q])``
  (a GLM dense unit has 5635 legal rungs and 515 priced ones).  Each table's
  ``menu_sizes`` is checked against its own identity first.
* ``provenance``: the shared fields, and

  - ``wire_dir`` / ``cache_dir``: the union's (``tessera_census_cache.py:271``
    requires ``wire_dir`` to be the resolved path; the allocation block needs
    one, ``tessera_expert_projection.py:685``);
  - ``hessian.capture_path`` / ``capture_sha256``: the union's handoff
    (``tessera_materialization.py:362`` reads the path);
  - ``activation_static_scales``: ``units`` unioned, ``path`` = the union's
    (read by ``tessera_materialization.py:263``/``:437``,
    ``tessera_joint_allocation.py:29``, the allocator's priced scales);
  - ``surfaces``, ``unservable``: unioned per unit per key;
    ``anchor_groups``: unioned (``tessera_anchored_surface.py:169``);
  - ``unit_selection``: groups unioned by key and checked against
    ``campaign_scope.anchor_groups`` (a sampled or v2 selection is refused);
    ``selected`` is recomputed as "some scope group is unpriced", so a table
    whose plan excluded groups (``selected`` True) and a whole-scope table
    (False) union to a whole-scope False when together they cover the scope;
    a table's ``coverage`` block is kept only per table;
    ``unit_selection_sample``, ``no_admitted_rung``,
    ``tessera_serving_scope`` unioned as ``merge_payloads`` does;
  - ``stopped_early`` or, ``wall_seconds`` sum, ``rounds_run`` max;
  - ``population``: rebuilt with ``campaign_population_block`` over the equal
    ``campaign_scope`` and the union's costs and menu sizes, so
    ``population.priced`` equals the nonempty cost rows
    (``tessera_expert_projection.py:642``);
  - ``campaign_fanout``: rows keyed ``<table>/<row>``.  Its only reader,
    ``tessera_joint_aura.py:420``, compares the rows to one dispatch plan and
    will refuse a union: a union is a census, not one fan-out;
  - ``rate_band``: ``None`` when the tables differ, the value the campaign
    writes for an unbanded run (``tessera_campaign.py:6539``); its only
    readers are the campaign and the dispatcher's row merge
    (``dispatch_tessera_campaign.py:114``);
  - ``family_restriction``, ``seed_checkpoint``,
    ``selected_source_preparation``: kept when equal, otherwise only per
    table;
  - ``union``: schema, the handoff table, and per table its root, identity
    digest, the SHA-256 of its ``cost.pkl``, ``cost.anchors.json`` and
    handoff, and its provenance: every field the union does not carry
    unchanged or merge at top level in full, and the others by digest
    (their content is at top level; ``tessera_expert_projection`` alone is
    19 MB on GLM).

  ``leave_one_anchor_out`` keeps its per-family shape for the allocator's
  band guard (``allocator.py:3173`` -> ``allocator_candidates.py:1278``).

Resume
------
An existing ``--out`` is refused unless ``--resume``.  With ``--resume`` the
plan is recomputed from the source tables and every existing output must be
the planned one: an envelope must load under the union seal with the planned
state, a wire must be the same inode or the same bytes, and ``cost.pkl`` and
``cost.anchors.json`` must be byte-identical.  Anything else is refused;
nothing existing is overwritten.  The manifest is written last.

Memory
------
The two cost tables are unpickled whole (GLM table A: 448 MB on disk,
2.3 GB resident).  A checkpoint manifest (7.2 GB) is streamed one unit at a
time and every unit's menu list is interned, so the 205 M menu strings of a
36,423-unit census become a handful of shared lists; the identity digest and
the output manifest reuse one encoding per distinct menu.  Merged envelope
states are held as pickles (~20 KB each) until written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prismaquant.cost_stage_checkpoint import (  # noqa: E402
    MANIFEST_SCHEMA, _load_unit, merge_identity_migrations, unique_temp_suffix, unit_path,
    write_unit,
)
from prismaquant.tessera_expert_projection import EXPERT_WIRES_KEY, POPULATION_KEY  # noqa: E402
from prismaquant.tessera_joint_aura import STAGE  # noqa: E402

UNION_SCHEMA = "prismaquant.tessera_census_union.v1"
COST_NAME = "cost.pkl"
MANIFEST_NAME = "cost.anchors.json"
PARTS_NAME = MANIFEST_NAME + ".parts"
REFERENCES_NAME = "hessian_capture.references.json"
SCALES_NAME = "input_scales.safetensors"

#: Cost-table keys with a merge rule; any other key must be equal.
COST_KEYS = {"schema", "currency", "costs", "formats", "leave_one_anchor_out",
             "non_interpolable", "menu_sizes", "anchor_counts", "provenance",
             EXPERT_WIRES_KEY}
#: Provenance keys the union merges; recorded per table by digest.
MERGED_PROVENANCE = {
    "wire_dir", "cache_dir", "hessian", "activation_static_scales", "surfaces",
    "anchor_groups", "unservable", "unit_selection", "unit_selection_sample",
    "no_admitted_rung", "tessera_serving_scope", "stopped_early", "wall_seconds",
    "rounds_run", POPULATION_KEY, "campaign_fanout",
}
#: Provenance keys that legitimately differ per table: kept at top level when
#: equal, otherwise only under ``union.tables`` (``rate_band`` becomes None).
PER_TABLE_PROVENANCE = {"rate_band", "family_restriction", "seed_checkpoint",
                        "selected_source_preparation", "coverage"}
#: Identity keys that legitimately differ per table.
PER_TABLE_IDENTITY = ("settings", "family_restriction")
HESSIAN_REFERENCE_EQUAL = ("schema", "canonical_capture", "provenance", "counts", "load_policy")

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_READ_BLOCK = 64 << 20


class UnionRefused(RuntimeError):
    """The tables do not union into one census table."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True, default=repr)


def _same(a: Any, b: Any) -> bool:
    """Equality that also holds for identical records carrying NaN."""
    return a is b or a == b or _json_key(a) == _json_key(b)


#: Wall-clock fields of a measured cell: how long the measurement took, not
#: what it measured.  Two campaigns that priced the same cell to the same
#: value (and the same wire bytes, checked separately) agree even though
#: their clocks do not; every other field must be equal.
#: ``encoding_batch_size`` is the anchor batch width the timing was divided
#: by (``tessera_campaign.py:1778-1780`` stamps it beside the two timing
#: fields; no allocator or ``tessera_*`` reader reads it), so a cell priced at
#: batch 8 in one table and batch 32 in the other is still one price.
TIMING_FIELDS = frozenset({"encode_seconds", "encode_seconds_accounting", "seconds",
                           "encoding_batch_size"})


def _same_price(a: Any, b: Any) -> bool:
    """:func:`_same` on measured cells, ignoring :data:`TIMING_FIELDS`."""
    if _same(a, b):
        return True
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        return False
    return _same({k: v for k, v in a.items() if k not in TIMING_FIELDS},
                 {k: v for k, v in b.items() if k not in TIMING_FIELDS})


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_key(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(_READ_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def _equal(label: str, values: Mapping[str, Any]) -> Any:
    items = list(values.items())
    first_name, first = items[0]
    for name, value in items[1:]:
        if not _same(first, value):
            raise UnionRefused(f"{label}: tables {first_name} and {name} differ")
    return first


def _merge_keyed(label: str, per_table: Mapping[str, Any]) -> dict:
    """``{key: value}`` maps unioned, a shared key requiring an equal value."""
    merged: dict = {}
    owner: dict = {}
    for table, mapping in per_table.items():
        if mapping is None:
            continue
        if not isinstance(mapping, Mapping):
            raise UnionRefused(f"{label} of table {table} is not a mapping")
        for key, value in mapping.items():
            if key in merged:
                if not _same(merged[key], value):
                    raise UnionRefused(
                        f"{label}[{key}]: tables {owner[key]} and {table} carry different values")
                continue
            merged[key] = value
            owner[key] = table
    return merged


def _merge_nested(label: str, per_table: Mapping[str, Any], same=_same) -> dict:
    """``{unit: {key: value}}`` maps unioned per unit per key."""
    merged: dict = {}
    owner: dict = {}
    for table, mapping in per_table.items():
        if mapping is None:
            continue
        if not isinstance(mapping, Mapping):
            raise UnionRefused(f"{label} of table {table} is not a mapping")
        for unit, inner in mapping.items():
            if not isinstance(inner, Mapping):
                raise UnionRefused(f"{label}[{unit}] of table {table} is not a mapping")
            held = merged.setdefault(unit, {})
            owners = owner.setdefault(unit, {})
            for key, value in inner.items():
                if key in held:
                    if not same(held[key], value):
                        raise UnionRefused(
                            f"{label}[{unit}][{key}]: tables {owners[key]} and {table} "
                            "carry different values")
                    continue
                held[key] = value
                owners[key] = table
    return dict(sorted(merged.items()))


# ---------------------------------------------------------------------------
# The checkpoint manifest: streamed read, canonical digest, streamed write
# ---------------------------------------------------------------------------
_CANON = json.JSONEncoder(sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode


def _indent_dump(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)


class _Memo:
    """One encoding per list object (the interned menus), keyed by identity."""

    def __init__(self):
        self._held: dict[int, tuple[Any, bytes]] = {}

    def get(self, value: Any, encode: Callable[[Any], bytes]) -> bytes:
        held = self._held.get(id(value))
        if held is None or held[0] is not value:
            held = (value, encode(value))
            self._held[id(value)] = held
        return held[1]


class MenuInterner:
    """Share one list object among units with the same menu."""

    def __init__(self):
        self._lists: dict[tuple, list] = {}

    def __call__(self, menu: Any) -> Any:
        if not isinstance(menu, list):
            return menu
        return self._lists.setdefault(tuple(menu), menu)


def identity_sha256(identity: Mapping[str, Any], memo: _Memo | None = None) -> str:
    """``cost_stage_checkpoint.canonical_json_sha256`` of a JSON-shaped identity, streamed."""
    memo = memo or _Memo()
    digest = hashlib.sha256()
    digest.update(b"{")
    for index, key in enumerate(sorted(identity)):
        value = identity[key]
        head = ("," if index else "") + _CANON(key) + ":"
        if key != "units" or not isinstance(value, Mapping):
            digest.update((head + _CANON(value)).encode("utf-8"))
            continue
        digest.update((head + "{").encode("utf-8"))
        for position, name in enumerate(sorted(value)):
            digest.update((("," if position else "") + _CANON(name) + ":").encode("utf-8"))
            unit = value[name]
            if not isinstance(unit, Mapping):
                digest.update(_CANON(unit).encode("utf-8"))
                continue
            digest.update(b"{")
            for slot, field in enumerate(sorted(unit)):
                digest.update((("," if slot else "") + _CANON(field) + ":").encode("utf-8"))
                item = unit[field]
                if field == "menu" and isinstance(item, list):
                    digest.update(memo.get(item, lambda v: _CANON(v).encode("utf-8")))
                else:
                    digest.update(_CANON(item).encode("utf-8"))
            digest.update(b"}")
        digest.update(b"}")
    digest.update(b"}")
    return digest.hexdigest()


def manifest_chunks(manifest: Mapping[str, Any], memo: _Memo | None = None) -> Iterator[bytes]:
    """The bytes of ``json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False)``."""
    memo = memo or _Memo()

    def walk(value, level, depth):
        pad = "  " * level
        if isinstance(value, Mapping) and value and depth < 4:
            yield b"{"
            for index, key in enumerate(sorted(value)):
                yield (("," if index else "") + "\n" + pad + "  "
                       + json.dumps(key, ensure_ascii=False) + ": ").encode("utf-8")
                yield from walk(value[key], level + 1, depth + 1)
            yield ("\n" + pad + "}").encode("utf-8")
            return
        if isinstance(value, list) and depth == 4:
            yield memo.get(value, lambda v: _indent_dump(v).replace("\n", "\n" + pad).encode("utf-8"))
            return
        yield _indent_dump(value).replace("\n", "\n" + pad).encode("utf-8")

    yield from walk(manifest, 0, 0)


_HEADER = b'{\n  "identity": {'
_UNITS_OPEN = b'\n    "units": {\n'
_UNIT_CLOSE = b'\n      }'


def read_manifest(path: Path, intern: MenuInterner) -> tuple[dict, str, int]:
    """``(manifest, file_sha256, bytes)``, one unit at a time, menus interned.

    The layout is the campaign's indent-2 dump: a unit closes at the only line
    of exactly six spaces and ``}``.  Any other layout is read with one
    ``json.loads`` (and menus interned afterwards); either way the caller
    recomputes the stored identity digest, which is the proof of the parse.
    """
    path = Path(path)
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        def more() -> bytes:
            nonlocal total
            block = handle.read(_READ_BLOCK)
            digest.update(block)
            total += len(block)
            return block

        buf = more()
        while len(buf) < len(_HEADER):
            block = more()
            if not block:
                break
            buf += block
        start = buf.find(_UNITS_OPEN)
        while start < 0 and buf.startswith(_HEADER):
            block = more()
            if not block:
                break
            scan = max(0, len(buf) - len(_UNITS_OPEN))
            buf += block
            start = buf.find(_UNITS_OPEN, scan)
        if start < 0 or not buf.startswith(_HEADER):
            while True:
                block = more()
                if not block:
                    break
                buf += block
            manifest = json.loads(buf)
            del buf
            units = (manifest.get("identity") or {}).get("units")
            if isinstance(units, dict):
                for unit in units.values():
                    if isinstance(unit, dict) and "menu" in unit:
                        unit["menu"] = intern(unit["menu"])
            return manifest, digest.hexdigest(), total
        header = json.loads(buf[:start].rstrip(b",") + b"}}")
        pos = start + len(_UNITS_OPEN)
        units: dict[str, Any] = {}
        while True:
            end = buf.find(_UNIT_CLOSE, pos)
            while end < 0 or len(buf) < end + len(_UNIT_CLOSE) + 2:
                block = more()
                if not block:
                    if end >= 0:
                        break
                    raise UnionRefused(f"{path}: checkpoint manifest ends inside its unit roster")
                buf = buf[pos:] + block
                pos = 0
                end = buf.find(_UNIT_CLOSE, pos)
            stop = end + len(_UNIT_CLOSE)
            one = json.loads(b"{" + buf[pos:stop] + b"}")
            if len(one) != 1:
                raise UnionRefused(f"{path}: unrecognised checkpoint unit layout near byte {total}")
            (name, unit), = one.items()
            if name in units:
                raise UnionRefused(f"{path}: checkpoint identity repeats unit {name}")
            if isinstance(unit, dict) and "menu" in unit:
                unit["menu"] = intern(unit["menu"])
            units[name] = unit
            if buf[stop:stop + 2] == b",\n":
                pos = stop + 2
                continue
            pos = stop
            break
        tail = buf[pos:]
        del buf
        rest = [tail]
        while True:
            block = more()
            if not block:
                break
            rest.append(block)
        trailer = json.loads(b'{"identity": {"units": {' + b"".join(rest))
    identity = header["identity"]
    for key, value in trailer["identity"].items():
        if key != "units":
            identity[key] = value
    identity["units"] = units
    manifest = {key: value for key, value in trailer.items() if key != "identity"}
    manifest["identity"] = identity
    return manifest, digest.hexdigest(), total


def _write_stream_atomic(path: Path, chunks) -> str:
    """Publish streamed bytes durably; returns their SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + unique_temp_suffix())
    digest = hashlib.sha256()
    try:
        with temporary.open("wb", buffering=_READ_BLOCK) as handle:
            for chunk in chunks:
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Source tables
# ---------------------------------------------------------------------------
class Table:
    def __init__(self, name: str, root: Path):
        self.name = name
        self.root = Path(root).resolve()
        self.cost_path = self.root / COST_NAME
        self.manifest_path = self.root / MANIFEST_NAME
        self.parts = self.root / PARTS_NAME
        self.cache = self.root / "cache"
        self.wire_dir = self.cache / "wire"
        self.references_path = self.cache / REFERENCES_NAME
        self.scales_path = self.cache / SCALES_NAME

    def load(self, intern: MenuInterner, memo: _Memo) -> "Table":
        for path in (self.cost_path, self.manifest_path, self.references_path, self.scales_path):
            if not path.is_file():
                raise UnionRefused(f"table {self.name}: {path} is missing")
        for path in (self.parts, self.wire_dir):
            if not path.is_dir():
                raise UnionRefused(f"table {self.name}: {path} is missing")
        raw = self.cost_path.read_bytes()
        self.cost_sha256 = hashlib.sha256(raw).hexdigest()
        self.cost_bytes = len(raw)
        self.cost = pickle.loads(raw)
        del raw
        if not isinstance(self.cost, dict) or not isinstance(self.cost.get("provenance"), dict):
            raise UnionRefused(f"table {self.name}: cost.pkl is not a campaign cost table")
        manifest, self.manifest_sha256, self.manifest_bytes = read_manifest(self.manifest_path, intern)
        if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("stage") != STAGE:
            raise UnionRefused(f"table {self.name}: not a Tessera campaign checkpoint manifest")
        identity = manifest.get("identity")
        if not isinstance(identity, dict) or not isinstance(identity.get("units"), dict):
            raise UnionRefused(f"table {self.name}: checkpoint manifest has no unit identity")
        actual = identity_sha256(identity, memo)
        if actual != manifest.get("identity_sha256"):
            raise UnionRefused(
                f"table {self.name}: checkpoint identity digest {actual} differs from its "
                f"stored seal {manifest.get('identity_sha256')}")
        listed = [entry.get("qname") for entry in manifest.get("units") or []]
        if len(set(listed)) != len(listed) or set(listed) != set(identity["units"]):
            raise UnionRefused(f"table {self.name}: journal list differs from its sealed unit roster")
        for entry in manifest["units"]:
            if entry.get("file") != unit_path(Path("."), entry["qname"]).as_posix():
                raise UnionRefused(f"table {self.name}: {entry['qname']} journal file is not its checkpoint path")
        self.identity = identity
        self.identity_sha256 = actual
        self.manifest_extra = {key: value for key, value in manifest.items()
                               if key not in {"schema", "stage", "identity", "identity_sha256", "units"}}
        raw = self.references_path.read_bytes()
        self.references_sha256 = hashlib.sha256(raw).hexdigest()
        self.references = json.loads(raw)
        del raw
        self.scales_sha256 = _sha256_file(self.scales_path)
        with os.scandir(self.wire_dir) as entries:
            self.wire_names = {entry.name for entry in entries}
        return self


def _parse_tables(specs: list[str]) -> list[tuple[str, Path]]:
    tables = []
    for spec in specs:
        name, sep, root = spec.partition("=")
        if not sep or not _NAME.match(name) or not root:
            raise UnionRefused(f"--table {spec!r}: expected NAME=DIR")
        if name in {existing for existing, _ in tables}:
            raise UnionRefused(f"--table {name}: table names must be unique")
        tables.append((name, Path(root)))
    if len(tables) < 2:
        raise UnionRefused("a union needs at least two --table arguments")
    return tables


# ---------------------------------------------------------------------------
# The plan: every check, no writes
# ---------------------------------------------------------------------------
def _union_identity(tables: list[Table], intern: MenuInterner) -> tuple[dict, dict]:
    for table in tables:
        if "union" in table.identity:
            raise UnionRefused(f"table {table.name} is already a union; union its sources instead")
    identity: dict[str, Any] = {}
    keys = set().union(*(t.identity for t in tables)) - {"units", *PER_TABLE_IDENTITY}
    for key in sorted(keys):
        missing = [t.name for t in tables if key not in t.identity]
        if missing:
            raise UnionRefused(f"identity.{key} is not bound by table(s) {', '.join(missing)}")
        identity[key] = _equal(f"identity.{key}", {t.name: t.identity[key] for t in tables})
    union_tables: dict[str, dict] = {}
    for table in tables:
        entry = {"identity_sha256": table.identity_sha256}
        for key in PER_TABLE_IDENTITY:
            if key in table.identity:
                entry[key] = table.identity[key]
        union_tables[table.name] = entry
    settings = {t.name: t.identity.get("settings") for t in tables}
    if any("settings" in t.identity for t in tables):
        if all(isinstance(value, Mapping) for value in settings.values()):
            shared = set.intersection(*(set(value) for value in settings.values()))
            identity["settings"] = {
                key: settings[tables[0].name][key] for key in sorted(shared)
                if all(_same(settings[tables[0].name][key], value[key]) for value in settings.values())}
        else:
            identity["settings"] = _equal("identity.settings", settings)
    restrictions = {t.name: t.identity.get("family_restriction") for t in tables}
    if (all("family_restriction" in t.identity for t in tables)
            and all(_same(restrictions[tables[0].name], v) for v in restrictions.values())):
        identity["family_restriction"] = restrictions[tables[0].name]
    identity["union"] = {"schema": UNION_SCHEMA, "tables": union_tables}

    names = sorted(set().union(*(t.identity["units"] for t in tables)))
    merged_menus: dict[tuple, list] = {}
    # Menus are interned, so one list object stands for every unit that seals
    # the same menu: validate each object once, not once per unit.  (Measured
    # on the A-twice dry-run of the 36,423-unit r1024 table, whose menus hold
    # 5,635 names: the per-unit check was 410 M isinstance calls, 129.5 s of
    # a 378 s plan; PrismaQuant #623.)  The held reference keeps the id live.
    checked_menus: dict[int, list] = {}
    units: dict[str, dict] = {}
    for name in names:
        bound = [(t.name, t.identity["units"][name]) for t in tables if name in t.identity["units"]]
        first_table, first = bound[0]
        if not isinstance(first, Mapping):
            raise UnionRefused(f"{name}: table {first_table} seals no unit inputs")
        for table, unit in bound[1:]:
            for key in set(first) | set(unit):
                if key == "menu":
                    continue
                if key not in first or key not in unit or not _same(first[key], unit[key]):
                    raise UnionRefused(
                        f"{name}: tables {first_table} and {table} bind different {key}")
        menus = [unit.get("menu") for _, unit in bound]
        for menu in menus:
            if checked_menus.get(id(menu)) is menu:
                continue
            if not isinstance(menu, list) or not all(isinstance(f, str) for f in menu):
                raise UnionRefused(f"{name}: a sealed menu is not a list of format names")
            checked_menus[id(menu)] = menu
        key = tuple(id(menu) for menu in menus)
        menu = merged_menus.get(key)
        if menu is None:
            menu = merged_menus[key] = intern(sorted(set().union(*menus)))
        units[name] = {**first, "menu": menu}
    identity["units"] = units
    return identity, union_tables


def _union_hessian_references(tables: list[Table], union_units: Mapping[str, Any]) -> Table:
    descriptors = {t.name: t.references for t in tables}
    for table, descriptor in descriptors.items():
        if not isinstance(descriptor, Mapping) or not isinstance(descriptor.get("hessians"), Mapping):
            raise UnionRefused(f"table {table}: Hessian reference handoff has no commitments")
    for field in HESSIAN_REFERENCE_EQUAL:
        _equal(f"hessian references.{field}", {n: d.get(field) for n, d in descriptors.items()})
    _equal("hessian references.census.sha256",
           {n: (d.get("census") or {}).get("sha256") for n, d in descriptors.items()})
    commitments = _merge_keyed("hessian references.hessians",
                               {n: d["hessians"] for n, d in descriptors.items()})
    bindings: dict[str, Any] = {}
    for table in tables:
        own = table.references.get("capture_sha256")
        hessian = table.cost["provenance"].get("hessian") or {}
        if hessian.get("capture_sha256") != own:
            raise UnionRefused(
                f"table {table.name}: provenance.hessian.capture_sha256 is not its handoff's digest")
        seen = {_json_key(hessian.get("reference_binding"))}
        for unit, rows in table.cost.get("costs", {}).items():
            for fmt, row in (rows or {}).items():
                identity = row.get("hessian_identity") if isinstance(row, Mapping) else None
                if not isinstance(identity, Mapping):
                    continue
                if identity.get("capture_sha256") != own:
                    raise UnionRefused(
                        f"table {table.name}: ({unit}, {fmt}) carries capture digest "
                        f"{identity.get('capture_sha256')}, not its handoff's {own}")
                seen.add(_json_key(identity.get("reference_binding")))
        if len(seen) != 1:
            raise UnionRefused(f"table {table.name}: rows bind more than one Hessian reference_binding")
        binding = hessian.get("reference_binding")
        # The one-file handoff is sound only because every table binds the
        # same canonical capture and the same census; say so explicitly.
        if binding is not None and (
                not isinstance(binding, Mapping)
                or binding.get("canonical_capture_sha256")
                != (table.references.get("canonical_capture") or {}).get("sha256")
                or binding.get("census_sha256") != (table.references.get("census") or {}).get("sha256")):
            raise UnionRefused(
                f"table {table.name}: reference_binding does not name its handoff's canonical "
                "capture and census")
        bindings[table.name] = binding
    _equal("provenance.hessian.reference_binding (canonical_capture_sha256, census_sha256)", bindings)
    for unit, sealed in union_units.items():
        hessian = sealed.get("hessian")
        if hessian is not None and unit in commitments and not _same(commitments[unit], hessian):
            raise UnionRefused(f"{unit}: sealed Hessian differs from the reference commitment")
    covering = [t for t in tables if set(t.references["hessians"]) == set(commitments)]
    if not covering:
        raise UnionRefused(
            "no table's Hessian reference handoff commits every unit of the union; "
            "a new handoff would need the reference writer, not a union")
    return covering[0]


def _restamp(row: Any, capture_sha256: str) -> Any:
    if not isinstance(row, Mapping):
        return row
    identity = row.get("hessian_identity")
    if not isinstance(identity, Mapping) or identity.get("capture_sha256") == capture_sha256:
        return row
    return {**row, "hessian_identity": {**identity, "capture_sha256": capture_sha256}}


def _union_provenance(tables: list[Table], *, costs, menu_sizes, capture_path: Path,
                      capture_sha256: str, wire_dir: Path, cache_dir: Path,
                      scales_path: Path) -> dict:
    from prismaquant.tessera_campaign import ExpertPopulation, campaign_population_block

    provs = {t.name: t.cost["provenance"] for t in tables}
    for name, prov in provs.items():
        if prov.get("research_exact_member_scope") is not None:
            raise UnionRefused(f"table {name}: a research exact-member scope is not a census")
    provenance: dict[str, Any] = {}
    keys = set().union(*provs.values())
    for key in sorted(keys - MERGED_PROVENANCE - PER_TABLE_PROVENANCE
                      - {"research_exact_member_scope", "identity_migration"}):
        missing = [n for n, p in provs.items() if key not in p]
        if missing:
            raise UnionRefused(f"provenance.{key} is not carried by table(s) {', '.join(missing)}")
        provenance[key] = _equal(f"provenance.{key}", {n: p[key] for n, p in provs.items()})
    for key in sorted(keys & PER_TABLE_PROVENANCE):
        values = {n: p.get(key) for n, p in provs.items()}
        if all(key in p for p in provs.values()) and all(
                _same(values[tables[0].name], v) for v in values.values()):
            provenance[key] = values[tables[0].name]
        elif key == "rate_band":
            provenance[key] = None
    # Tables resealed under one proof carry records that differ in clock,
    # strata and tool commit; each table's own stays under ``union.tables``.
    migration = merge_identity_migrations(
        {n: p.get("identity_migration") for n, p in provs.items()}, error=UnionRefused)
    if migration is not None:
        provenance["identity_migration"] = migration

    hessians = {n: p.get("hessian") for n, p in provs.items()}
    if not all(isinstance(h, Mapping) for h in hessians.values()):
        raise UnionRefused("provenance.hessian is missing from a table")
    shared_hessian = {}
    for field in sorted(set().union(*hessians.values()) - {"capture_path", "capture_sha256"}):
        shared_hessian[field] = _equal(f"provenance.hessian.{field}",
                                       {n: h.get(field) for n, h in hessians.items()})
    provenance["hessian"] = {**shared_hessian, "capture_path": str(capture_path),
                             "capture_sha256": capture_sha256}

    scales = {n: p.get("activation_static_scales") for n, p in provs.items()}
    if all(s is None for s in scales.values()):
        pass
    elif not all(isinstance(s, Mapping) for s in scales.values()):
        raise UnionRefused("provenance.activation_static_scales is missing from a table")
    else:
        block = {}
        for field in sorted(set().union(*scales.values()) - {"path", "units"}):
            block[field] = _equal(f"provenance.activation_static_scales.{field}",
                                  {n: s.get(field) for n, s in scales.items()})
        block["path"] = str(scales_path)
        block["units"] = dict(sorted(_merge_keyed(
            "provenance.activation_static_scales.units",
            {n: s.get("units") for n, s in scales.items()}).items()))
        provenance["activation_static_scales"] = block

    provenance["wire_dir"] = str(wire_dir)
    provenance["cache_dir"] = str(cache_dir)
    provenance["surfaces"] = _merge_nested("provenance.surfaces",
                                           {n: p.get("surfaces") for n, p in provs.items()},
                                           same=_same_price)
    provenance["unservable"] = _merge_nested("provenance.unservable",
                                             {n: p.get("unservable") for n, p in provs.items()})
    provenance["anchor_groups"] = dict(sorted(_merge_keyed(
        "provenance.anchor_groups", {n: p.get("anchor_groups") for n, p in provs.items()}).items()))

    selections = {n: p.get("unit_selection") for n, p in provs.items()}
    if not all(isinstance(s, Mapping) for s in selections.values()):
        raise UnionRefused("provenance.unit_selection is missing from a table")
    schema = _equal("provenance.unit_selection.schema", {n: s.get("schema") for n, s in selections.items()})
    if schema != "prismaquant.tessera_campaign_units.v1":
        raise UnionRefused(f"provenance.unit_selection {schema} is sampled; only v1 selections union")
    for name, selection in selections.items():
        for entry in selection.get("groups") or []:
            if "sampled" in entry or "stack_samples" in entry:
                raise UnionRefused(f"table {name}: sampled selection group {entry.get('key')}")
    groups = _merge_keyed("provenance.unit_selection.groups",
                          {n: {e["key"]: e for e in s.get("groups") or []}
                           for n, s in selections.items()})
    scope_groups = (provenance.get("campaign_scope") or {}).get("anchor_groups")
    if not isinstance(scope_groups, Mapping):
        raise UnionRefused("provenance.campaign_scope.anchor_groups is missing")
    if set(groups) - set(scope_groups):
        raise UnionRefused(
            f"provenance.unit_selection groups outside the campaign scope: {sorted(set(groups) - set(scope_groups))[:4]}")
    for key, entry in groups.items():
        if sorted(entry.get("members") or ()) != sorted(scope_groups[key]):
            raise UnionRefused(f"provenance.unit_selection group {key} differs from the campaign scope")
    # ``selected`` is False exactly when the selection is the whole scope
    # (dispatch_tessera_campaign.merge_payloads stamps False for a whole-scope
    # merge).  A table whose plan excluded groups carries True; the union is
    # whole-scope again once the tables together price every scope group.
    provenance["unit_selection"] = {
        "schema": schema,
        "selected": bool(set(scope_groups) - set(groups)),
        "groups": [groups[key] for key in sorted(groups)],
    }
    if any("unit_selection_sample" in p for p in provs.values()):
        samples = {n: p.get("unit_selection_sample") or {} for n, p in provs.items()}
        provenance["unit_selection_sample"] = {
            "audit_units": sorted({u for s in samples.values() for u in s.get("audit_units", [])}),
            "inclusion_probability": dict(sorted(_merge_keyed(
                "provenance.unit_selection_sample.inclusion_probability",
                {n: s.get("inclusion_probability") for n, s in samples.items()}).items())),
        }
    if any("no_admitted_rung" in p for p in provs.values()):
        provenance["no_admitted_rung"] = sorted(
            {u for p in provs.values() for u in p.get("no_admitted_rung") or []})
    serving = {n: p.get("tessera_serving_scope") for n, p in provs.items() if p.get("tessera_serving_scope")}
    if serving:
        provenance["tessera_serving_scope"] = {
            "target": _equal("provenance.tessera_serving_scope.target",
                             {n: s["target"] for n, s in serving.items()}),
            "by_unit": dict(sorted(_merge_keyed(
                "provenance.tessera_serving_scope.by_unit",
                {n: s["by_unit"] for n, s in serving.items()}).items())),
        }
    provenance["stopped_early"] = any(bool(p.get("stopped_early")) for p in provs.values())
    provenance["wall_seconds"] = float(sum(float(p.get("wall_seconds") or 0.0) for p in provs.values()))
    provenance["rounds_run"] = max(int(p.get("rounds_run") or 0) for p in provs.values())
    fanouts = {n: p.get("campaign_fanout") for n, p in provs.items() if p.get("campaign_fanout")}
    if fanouts:
        provenance["campaign_fanout"] = {
            "schema": _equal("provenance.campaign_fanout.schema",
                             {n: f.get("schema") for n, f in fanouts.items()}),
            "rows": {f"{n}/{row}": groups_ for n, f in fanouts.items()
                     for row, groups_ in sorted((f.get("rows") or {}).items())},
            "seed_checkpoints": [{"table": n, **entry} for n, f in fanouts.items()
                                 for entry in f.get("seed_checkpoints") or []],
        }

    scope = provenance.get("campaign_scope")
    if not isinstance(scope, Mapping):
        raise UnionRefused("provenance.campaign_scope is missing; the population cannot be rebuilt")
    population = ExpertPopulation(
        members=(),
        declared={stack: {name: tuple(shape) for name, shape in units.items()}
                  for stack, units in scope["declared_stacks"].items()},
        packed_in_scope={name: tuple(shape) for name, shape in scope["packed_in_scope"].items()},
        omitted_outside_layer_stride={
            name: tuple(shape) for name, shape in scope["packed_outside_layer_stride"].items()},
    )
    try:
        provenance[POPULATION_KEY] = campaign_population_block(
            dense_targets=scope["dense_targets"], expert_targets=scope["expert_targets"],
            dense_all=scope["dense_all"], pinned=scope["pinned"], population=population,
            layer_stride=int(provenance["layer_stride"]), costs=costs, menus=menu_sizes,
            stack_samples=None, profile=None)
    except (KeyError, ValueError, RuntimeError) as exc:
        raise UnionRefused(f"population rebuild over the union refused: {exc}") from exc
    return dict(sorted(provenance.items()))


def _union_cost(tables: list[Table], union_units: Mapping[str, dict], *, capture_sha256: str,
                paths: Mapping[str, Path]) -> tuple[dict, dict]:
    from prismaquant.tessera_campaign import SCHEMA, canonical_refusals

    for table in tables:
        cost = table.cost
        if cost.get("schema") != SCHEMA:
            raise UnionRefused(f"table {table.name}: cost table is not {SCHEMA}")
        if not isinstance(cost.get("costs"), Mapping):
            raise UnionRefused(f"table {table.name}: cost table has no rows")
        if set(cost["costs"]) != set(table.identity["units"]):
            raise UnionRefused(f"table {table.name}: cost rows and sealed units name different units")
        for unit, size in (cost.get("menu_sizes") or {}).items():
            sealed = table.identity["units"].get(unit)
            if sealed is None or size != len(sealed.get("menu") or ()):
                raise UnionRefused(
                    f"table {table.name}: menu_sizes[{unit}] is not its sealed menu's length")
    for key in sorted(set().union(*(t.cost for t in tables)) - COST_KEYS):
        missing = [t.name for t in tables if key not in t.cost]
        if missing:
            raise UnionRefused(f"cost table key {key} is not carried by table(s) {', '.join(missing)}")
        _equal(f"cost table {key}", {t.name: t.cost[key] for t in tables})
    extra = {key: tables[0].cost[key]
             for key in set().union(*(t.cost for t in tables)) - COST_KEYS}
    currency = _equal("currency", {t.name: t.cost.get("currency") for t in tables})

    costs: dict[str, dict] = {}
    owners: dict[str, dict] = {}
    collisions = 0
    for table in tables:
        for unit, rows in table.cost["costs"].items():
            held = costs.setdefault(unit, {})
            owner = owners.setdefault(unit, {})
            for fmt, row in (rows or {}).items():
                stamped = _restamp(row, capture_sha256)
                if fmt in held:
                    if not _same_price(held[fmt], stamped):
                        raise UnionRefused(
                            f"({unit}, {fmt}) is priced by tables {owner[fmt]} and {table.name} "
                            "with different rows")
                    collisions += 1
                    continue
                held[fmt] = stamped
                owner[fmt] = table.name
    del owners
    costs = dict(sorted(costs.items()))
    menu_units = set().union(*(set(t.cost.get("menu_sizes") or {}) for t in tables))
    menu_sizes = {unit: len(union_units[unit]["menu"]) for unit in sorted(menu_units)}
    seen_refusals: dict[str, dict] = {}
    for table in tables:
        for entry in table.cost.get("non_interpolable") or []:
            seen_refusals.setdefault(_json_key(entry), entry)
    payload = {
        **extra,
        "schema": SCHEMA,
        "currency": currency,
        "costs": costs,
        "formats": sorted(set().union(*(set(t.cost.get("formats") or ()) for t in tables))),
        "leave_one_anchor_out": _merge_nested(
            "leave_one_anchor_out", {t.name: t.cost.get("leave_one_anchor_out") for t in tables}),
        "non_interpolable": canonical_refusals(list(seen_refusals.values())),
        "menu_sizes": menu_sizes,
        "anchor_counts": _merge_nested(
            "anchor_counts", {t.name: t.cost.get("anchor_counts") for t in tables}),
    }
    if any(EXPERT_WIRES_KEY in t.cost for t in tables):
        payload[EXPERT_WIRES_KEY] = _merge_nested(
            EXPERT_WIRES_KEY, {t.name: t.cost.get(EXPERT_WIRES_KEY) for t in tables})
    payload["provenance"] = _union_provenance(
        tables, costs=costs, menu_sizes=menu_sizes, capture_path=paths["references"],
        capture_sha256=capture_sha256, wire_dir=paths["wire"], cache_dir=paths["cache"],
        scales_path=paths["scales"])
    from prismaquant.tessera_menu import assert_uniform_hessian_identity
    try:
        assert_uniform_hessian_identity(costs)
    except Exception as exc:  # ValueError here, tessera GrammarError from the binding check
        raise UnionRefused(f"union cost rows do not share one Hessian identity: {exc}") from exc
    return payload, {"unit_format_collisions": collisions}


def _merge_unit_journal(name: str, tables: list[Table]) -> tuple[str, bytes, list, int]:
    anchors: list = []
    by_format: dict[str, Any] = {}
    records: dict[str, Any] = {}
    extra: dict[str, Any] = {}
    wires: list[tuple[str, str, Any, Any]] = []
    duplicates = 0
    for table in tables:
        try:
            state = _load_unit(unit_path(table.parts, name), stage=STAGE, qname=name,
                               identity_sha256=table.identity_sha256)
        except (OSError, RuntimeError) as exc:
            raise UnionRefused(f"table {table.name}: {name}: {exc}") from exc
        for key, value in state.items():
            if key in ("anchors", "wire_records"):
                continue
            if key in extra and not _same(extra[key], value):
                raise UnionRefused(f"{name}: journal state {key} differs between tables")
            extra.setdefault(key, value)
        for anchor in state.get("anchors") or []:
            key = (anchor.get("format_name") if isinstance(anchor, Mapping) else None) \
                or _json_key(anchor)
            if key in by_format:
                if not _same_price(by_format[key], anchor):
                    raise UnionRefused(f"({name}, {key}): journal anchors differ between tables")
                duplicates += 1
                continue
            by_format[key] = anchor
            anchors.append(anchor)
        own = state.get("wire_records") or {}
        if not isinstance(own, Mapping):
            raise UnionRefused(f"table {table.name}: {name}: wire_records is not a mapping")
        for fmt, record in own.items():
            if fmt in records:
                if not _same(records[fmt], record):
                    raise UnionRefused(f"({name}, {fmt}): journal wire receipts differ between tables")
            else:
                records[fmt] = record
            if not isinstance(record, Mapping) or not isinstance(record.get("file"), str) \
                    or Path(record["file"]).name != record["file"]:
                raise UnionRefused(f"({name}, {fmt}): table {table.name} receipt names no local wire")
            wires.append((table.name, record["file"], record.get("blob_bytes"), record.get("blob_sha256")))
        priced = (table.cost.get(EXPERT_WIRES_KEY) or {}).get(name) or {}
        for fmt, record in priced.items():
            if not _same(own.get(fmt), record):
                raise UnionRefused(
                    f"({name}, {fmt}): table {table.name} cost-table receipt differs from its journal")
    state = {"anchors": anchors, "wire_records": records, **extra}
    return name, pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL), wires, duplicates


class Plan:
    pass


def build_plan(specs: list[str], out: Path) -> Plan:
    intern = MenuInterner()
    memo = _Memo()
    tables = [Table(name, root).load(intern, memo) for name, root in _parse_tables(specs)]
    by_name = {t.name: t for t in tables}
    out = Path(out)
    out = out.parent.resolve() / out.name
    cache = out / "cache"
    paths = {"out": out, "cache": cache, "wire": cache / "wire", "parts": out / PARTS_NAME,
             "references": cache / REFERENCES_NAME, "scales": cache / SCALES_NAME,
             "cost": out / COST_NAME, "manifest": out / MANIFEST_NAME}

    identity, union_tables = _union_identity(tables, intern)
    reference_table = _union_hessian_references(tables, identity["units"])
    capture_sha256 = reference_table.references["capture_sha256"]
    scales_sha = _equal("cache/input_scales.safetensors sha256",
                        {t.name: t.scales_sha256 for t in tables})
    del scales_sha
    extras = {t.name: {key: value for key, value in t.manifest_extra.items()
                       if key != "identity_migration"} for t in tables}
    manifest_extra = _equal("checkpoint manifest extra keys", extras)
    migration = merge_identity_migrations(
        {t.name: t.manifest_extra.get("identity_migration") for t in tables}, error=UnionRefused)
    if migration is not None:
        manifest_extra["identity_migration"] = migration

    cost, cost_stats = _union_cost(tables, identity["units"], capture_sha256=capture_sha256, paths=paths)
    top = cost["provenance"]

    def carried(prov):
        # A field the union carries unchanged at top level, or merged into a
        # top-level field, is recorded by digest; everything else in full.
        return {key: _digest(value) for key, value in sorted(prov.items())
                if key in MERGED_PROVENANCE or (key in top and _same(top[key], value))}

    carried_keys = {t.name: carried(t.cost["provenance"]) for t in tables}

    cost["provenance"]["union"] = {
        "schema": UNION_SCHEMA,
        "hessian_reference_table": reference_table.name,
        "tables": {
            t.name: {
                "root": str(t.root),
                "identity_sha256": t.identity_sha256,
                "cost_pkl_sha256": t.cost_sha256,
                "cost_anchors_json_sha256": t.manifest_sha256,
                "cost_anchors_json_bytes": t.manifest_bytes,
                "hessian_references_sha256": t.references_sha256,
                "hessian_capture_sha256": t.references.get("capture_sha256"),
                "input_scales_sha256": t.scales_sha256,
                "provenance": {key: value for key, value in sorted(t.cost["provenance"].items())
                               if key not in carried_keys[t.name]},
                "carried_provenance_sha256": carried_keys[t.name],
            }
            for t in tables
        },
    }

    memo_out = _Memo()
    union_sha = identity_sha256(identity, memo)
    names = sorted(identity["units"])
    manifest = {"identity": identity, "identity_sha256": union_sha, "schema": MANIFEST_SCHEMA,
                "stage": STAGE, **manifest_extra,
                "units": [{"file": unit_path(Path("."), name).as_posix(), "qname": name}
                          for name in names]}

    states: dict[str, bytes] = {}
    wire_plan: dict[str, tuple[str, Any, Any]] = {}
    wire_collisions = {"identical_inode": 0, "identical_bytes": 0}
    anchor_duplicates = 0
    # One thread, on purpose: the per-unit load is ``pickle.loads`` plus
    # pure-Python checks under the GIL, so a thread pool here ran the whole
    # plan at 101-102% CPU, and the profile of the A-twice dry-run put the
    # plan's time in manifest reading (167 s) and the menu check above
    # (129.5 s, since fixed), not here (PrismaQuant #623).
    for name in names:
        _, state, wires, duplicates = _merge_unit_journal(
            name, [t for t in tables if name in t.identity["units"]])
        states[name] = state
        anchor_duplicates += duplicates
        for table, file, size, sha in wires:
            if file not in by_name[table].wire_names:
                raise UnionRefused(f"table {table}: receipt blob {file} is not in its wire directory")
            held = wire_plan.get(file)
            if held is None:
                wire_plan[file] = (table, size, sha)
                continue
            if held[0] == table:
                continue
            if (held[1], held[2]) != (size, sha):
                raise UnionRefused(
                    f"wire {file}: tables {held[0]} and {table} carry different receipts")
    shared_dirs = {(t.name, u.name) for t in tables for u in tables
                   if t.name != u.name and os.path.samefile(t.wire_dir, u.wire_dir)}
    for file, (table, size, sha) in list(wire_plan.items()):
        others = sorted({t.name for t in tables if t.name != table and file in t.wire_names})
        for other in others:
            first, second = by_name[table].wire_dir / file, by_name[other].wire_dir / file
            # A name both tables hold is a collision whether or not the second
            # table's journal references it: the link would pick one of them.
            # One directory listed twice is one inode per name, without a stat.
            if (table, other) in shared_dirs or os.path.samefile(first, second):
                wire_collisions["identical_inode"] += 1
                continue
            if first.stat().st_size != second.stat().st_size or _sha256_file(first) != _sha256_file(second):
                raise UnionRefused(f"wire {file}: tables {table} and {other} hold different bytes")
            wire_collisions["identical_bytes"] += 1

    plan = Plan()
    plan.tables = tables
    plan.paths = paths
    plan.identity_sha256 = union_sha
    plan.manifest = manifest
    plan.memo = memo_out
    plan.cost = cost
    plan.states = states
    plan.wire_plan = wire_plan
    plan.reference_table = reference_table
    plan.scales_table = tables[0]
    families: dict[str, int] = {}
    for rows in cost["costs"].values():
        for fmt, row in rows.items():
            family = (row.get("tessera_family") if isinstance(row, Mapping) else None) \
                or fmt.rsplit("_R", 1)[0]
            families[family] = families.get(family, 0) + 1
    plan.summary = {
        "schema": UNION_SCHEMA,
        "tables": {
            t.name: {
                "root": str(t.root), "identity_sha256": t.identity_sha256,
                "units": len(t.identity["units"]),
                "rungs": sum(len(rows or {}) for rows in t.cost["costs"].values()),
                "cost_pkl_sha256": t.cost_sha256, "cost_anchors_json_sha256": t.manifest_sha256,
                "cost_anchors_json_bytes": t.manifest_bytes,
                "hessian_capture_sha256": t.references.get("capture_sha256"),
                "wire_blobs_listed": len(t.wire_names),
            } for t in tables},
        "union": {
            "out": str(out), "identity_sha256": union_sha, "units": len(names),
            "rungs": sum(len(rows) for rows in cost["costs"].values()),
            "rungs_by_family": dict(sorted(families.items())),
            "menu_sizes": dict(sorted(_count(cost["menu_sizes"].values()).items())),
            "hessian_reference_table": reference_table.name,
            "hessian_capture_sha256": capture_sha256,
            "wire_blobs": len(wire_plan),
            "population_counts": cost["provenance"][POPULATION_KEY].get("counts"),
        },
        "collisions": {**cost_stats, "journal_anchor_duplicates": anchor_duplicates,
                       "wire_names": wire_collisions},
    }
    return plan


def _count(values) -> dict:
    counts: dict = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------
def _link(source: Path, target: Path, *, resume: bool) -> str:
    try:
        os.link(source, target)
        return "linked"
    except FileExistsError:
        if not resume:
            raise UnionRefused(f"{target} already exists")
        if target.is_symlink():
            raise UnionRefused(f"{target} is a symlink; the census readers refuse it")
        if os.path.samefile(source, target):
            return "kept"
        if (target.stat().st_size == source.stat().st_size
                and _sha256_file(target) == _sha256_file(source)):
            return "kept"
        raise UnionRefused(f"{target} exists with different bytes than {source}")
    except OSError as exc:
        raise UnionRefused(
            f"hardlink {source} -> {target} failed ({exc}); refusing to copy wires") from exc


def write_plan(plan: Plan, *, resume: bool) -> dict:
    paths = plan.paths
    by_name = {t.name: t for t in plan.tables}
    if (paths["out"].exists() or paths["out"].is_symlink()) and not resume:
        raise UnionRefused(f"--out {paths['out']} exists; pass --resume to continue into it")
    for directory in (paths["out"], paths["cache"], paths["wire"], paths["parts"] / "units"):
        if directory.is_symlink():
            raise UnionRefused(f"{directory} is a symlink")
        directory.mkdir(parents=True, exist_ok=True)
    counts = {"wires_linked": 0, "wires_kept": 0, "envelopes_written": 0, "envelopes_kept": 0}
    for file, (table, _size, _sha) in sorted(plan.wire_plan.items()):
        state = _link(by_name[table].wire_dir / file, paths["wire"] / file, resume=resume)
        counts["wires_" + state] += 1
    _link(plan.reference_table.references_path, paths["references"], resume=resume)
    _link(plan.scales_table.scales_path, paths["scales"], resume=resume)

    def envelope(item):
        name, state_bytes = item
        path = unit_path(paths["parts"], name)
        state = pickle.loads(state_bytes)
        if path.exists():
            if not resume:
                raise UnionRefused(f"{path} already exists")
            try:
                stored = _load_unit(path, stage=STAGE, qname=name, identity_sha256=plan.identity_sha256)
            except (OSError, RuntimeError) as exc:
                raise UnionRefused(f"existing union envelope for {name} does not verify: {exc}") from exc
            if not _same(stored, state):
                raise UnionRefused(f"existing union envelope for {name} holds a different state")
            return "kept"
        write_unit(paths["parts"], stage=STAGE, qname=name, identity_sha256=plan.identity_sha256,
                   state=state)
        return "written"

    # Serial as well.  Each envelope is an atomic write with its own fsync, so
    # threads could overlap the fsyncs; that has not been measured on the
    # raidz1 pool the union is written to, and the tool does not claim it
    # (recorded as an open measurement on PrismaQuant #623).
    for item in sorted(plan.states.items()):
        counts["envelopes_" + envelope(item)] += 1

    cost_bytes = pickle.dumps(plan.cost, protocol=pickle.HIGHEST_PROTOCOL)
    cost_sha = hashlib.sha256(cost_bytes).hexdigest()
    if paths["cost"].exists():
        if not resume or paths["cost"].read_bytes() != cost_bytes:
            raise UnionRefused(f"{paths['cost']} exists and is not the planned cost table")
    else:
        _write_stream_atomic(paths["cost"], [cost_bytes])
    del cost_bytes
    if paths["manifest"].exists():
        planned = hashlib.sha256()
        for chunk in manifest_chunks(plan.manifest, plan.memo):
            planned.update(chunk)
        manifest_sha = planned.hexdigest()
        if not resume or _sha256_file(paths["manifest"]) != manifest_sha:
            raise UnionRefused(f"{paths['manifest']} exists and is not the planned manifest")
    else:
        manifest_sha = _write_stream_atomic(paths["manifest"], manifest_chunks(plan.manifest, plan.memo))
    return {**counts, "cost_pkl_sha256": cost_sha, "cost_anchors_json_sha256": manifest_sha}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", action="append", required=True, metavar="NAME=DIR",
                        help="a merged census table directory (repeat; order sets anchor order)")
    parser.add_argument("--out", required=True, help="union table directory to create")
    parser.add_argument("--dry-run", action="store_true", help="run every check, print the plan, write nothing")
    parser.add_argument("--resume", action="store_true",
                        help="continue into an existing --out whose files verify against the plan")
    args = parser.parse_args(argv)
    out = Path(args.out)
    try:
        if (out.exists() or out.is_symlink()) and not args.resume and not args.dry_run:
            raise UnionRefused(f"--out {out} exists; pass --resume to continue into it")
        plan = build_plan(args.table, out)
        summary = dict(plan.summary)
        if args.dry_run:
            summary["dry_run"] = True
            summary["writes"] = 0
        else:
            summary["written"] = write_plan(plan, resume=args.resume)
    except UnionRefused as exc:
        print(f"union refused: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=1, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
