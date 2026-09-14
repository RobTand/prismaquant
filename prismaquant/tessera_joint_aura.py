"""Research joint AURA over exact, completed Tessera campaign anchors.

The campaign's scalar MSE/interpolation is evidence of which wires were made,
never a joint price. Original decoded renders enter ProductionWeightCache;
Tessera's existing source/H/settings receipt and decoder qualify them before
ordinary streamed joint AURA consumes the exact per-Linear candidate roster.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import functools
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import pickle
import os
import socket
import stat
import time
from types import SimpleNamespace

from .cost_stage_checkpoint import (
    MANIFEST_SCHEMA, _load_unit, atomic_write_bytes, canonical_json_sha256,
    prepare_journal, unit_path, write_unit,
)

SCHEMA = "prismaquant.tessera_joint_aura.plan.v1"
PREPARED_SCHEMA = "prismaquant.tessera_joint_aura.prepared.v3"
RENDER_ORIGIN_SCHEMA = "prismaquant.tessera_joint_aura.render_origin.v1"
# Closed vocabularies. ``render_origin`` says where the decoded PWC shard on
# disk came from; ``render_comparison`` says what the ``torch.equal`` leg of
# ``verify_anchor_render`` established for that rung. They are two different
# facts and a record that collapses them claims verification it never had.
RENDER_ORIGINS = ("encoded", "synthesized_from_wire")
RENDER_COMPARISONS = ("independent_render_vs_wire", "wire_round_trip_only")
# An encoded render is an independently produced tensor, so comparing it with
# the decoded wire is evidence about the encode. A synthesized render was
# written by decoding that same wire, so the comparison can only establish
# that the ``.pt`` still round-trips to the bytes it was written from.
RENDER_COMPARISON_BY_ORIGIN = {"encoded": "independent_render_vs_wire",
                               "synthesized_from_wire": "wire_round_trip_only"}
CAMPAIGN_SCHEMA = "prismaquant.tessera_campaign_cost.v1"
CURRENCY = "output_mse_under_route_activation_contract"
STAGE = "Tessera campaign"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _stat_signature(value):
    """The identity fence for a wire byte read, including its file type."""
    return (value.st_mode, value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


def _read_verified_wire_blob(cell):
    """Read one receipt-sized regular wire and bind the bytes actually read.

    This is deliberately an ephemeral one-cell buffer, rather than a cache.
    The caller may overlap its read with qualification of the preceding cell,
    but must pass these exact bytes to ``verify_cached_unit`` and the decoder.
    """
    record, wire = cell["record"], Path(cell["wire"])
    size = record.get("blob_bytes")
    _require(type(size) is int and size > 0,
             f"{wire}: wire receipt needs positive blob_bytes")
    before = wire.lstat()
    _require(stat.S_ISREG(before.st_mode), f"{wire}: wire must be a regular file, not a symlink")
    _same(before.st_size, size, f"{wire}: wire size differs from measured receipt")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(wire, flags)
    try:
        with os.fdopen(fd, "rb", closefd=True) as handle:
            fd = None
            _same(_stat_signature(os.fstat(handle.fileno())), _stat_signature(before),
                  f"{wire}: wire changed before its content read")
            blob = handle.read(size + 1)
            after_open = os.fstat(handle.fileno())
    finally:
        if fd is not None:
            os.close(fd)
    _require(len(blob) == size, f"{wire}: wire changed during its content read")
    _same(_stat_signature(after_open), _stat_signature(before),
          f"{wire}: wire changed during its content read")
    _same(_stat_signature(wire.lstat()), _stat_signature(before),
          f"{wire}: wire changed during its content read")
    digest = hashlib.sha256(blob).hexdigest()
    _same(digest, record.get("blob_sha256"), f"{wire}: wire checksum")
    return blob, digest


def _bound(record, label):
    _require(isinstance(record, dict) and set(record) == {"path", "sha256"},
             f"{label}: independently bound path/SHA256 required")
    path = Path(record["path"])
    _require(_sha(path) == record["sha256"], f"{label}: artifact checksum changed")
    return path


def _same(actual, expected, label):
    _require(actual == expected, f"{label}: identity mismatch")


def _json(path, value):
    atomic_write_bytes(Path(path), (json.dumps(value, indent=2, sort_keys=True,
                                              allow_nan=False) + "\n").encode())


@dataclass
class MeasuredAnchorInput:
    inputs: dict
    payload: dict
    manifest: dict
    census: dict
    campaign_plan: dict
    cells: dict
    formats_by_qname: dict
    # A read restricted to ``sorted(names)[lo:hi]``, or None for the whole
    # roster. Only the standalone synthesis stage produces a scoped read, and
    # ``execute`` refuses one: a partial roster is not the campaign's input.
    unit_scope: "tuple | None" = None
    # Where the synthesized shards were written when they were NOT written to
    # the campaign's own row caches (the measure mirror), or None.
    render_mirror_root: "str | None" = None
    # How many shards THIS read synthesized. Distinct from the per-origin
    # census, which counts what is on disk however it got there.
    synthesized_now: int = 0

    @property
    def total_render_bytes(self):
        return sum(2 * math.prod(self.census["unit_shapes"][name]) for name, _ in self.cells)

    def layer_render_bytes(self, layer_for_name):
        sizes = defaultdict(int)
        for name, _fmt in self.cells:
            sizes[layer_for_name(name)] += 2 * math.prod(self.census["unit_shapes"][name])
        return dict(sizes)


def _render_origin_marker_path(render):
    return Path(render).with_name(Path(render).name + ".render_origin.json")


def render_origin_census(origins):
    """Count each closed-vocabulary value, including the ones nobody used.

    A census that omits the zero keeps a reader from telling "no synthesized
    renders" apart from "this report does not say".
    """
    counts = {value: 0 for value in RENDER_ORIGINS}
    comparisons = {value: 0 for value in RENDER_COMPARISONS}
    for origin in origins:
        _require(origin in RENDER_ORIGINS, f"unknown render origin {origin!r}")
        counts[origin] += 1
        comparisons[RENDER_COMPARISON_BY_ORIGIN[origin]] += 1
    return {"render_origins": counts, "render_comparisons": comparisons}


def cell_render_census(cells):
    """The census of a cell mapping, from the field every cell must carry."""
    origins = []
    for pair, cell in sorted(cells.items()):
        _require(isinstance(cell, dict) and "render_origin" in cell,
                 f"{pair}: cell carries no render_origin")
        origins.append(cell["render_origin"])
    return render_origin_census(origins)


def _decode_wire(blob, *, reader, device="cpu"):
    """The one decode seam: the bound reader's, or the module-level decoder."""
    if reader is not None:
        return reader.read_unit_artifact(blob, device=device)
    from tessera.unit_artifact import read_unit_artifact

    return read_unit_artifact(blob, device=device)


def _render_mirror_path(render, mirror_root):
    """Where a measuring run publishes a shard instead of the row cache.

    The mirror keeps the render's absolute path under ``mirror_root`` so a
    cell's two copies stay comparable by name and a measuring run can never
    replace the campaign's own bytes.
    """
    render = Path(render)
    return Path(mirror_root) / render.resolve().relative_to(Path(render.root))


def _pb_commit(units, phase, unit=None):
    """Report cumulative durable units to PrismaBuild; a no-op elsewhere.

    Held byte for byte against the published submission skill's snippet
    (``skills/prismabuild/SKILL.md``, ``pb-progress-snippet``) so an action
    inside a container that cannot import PrismaBuild still reports. It is a
    no-op when the action was not admitted under the progress contract, so it
    is called unconditionally rather than by testing how we were launched.
    """
    path = os.environ.get("PRISMABUILD_ACTION_PROGRESS_PATH")
    token = os.environ.get("PRISMABUILD_ACTION_PROGRESS_TOKEN")
    if not path or not token:
        return False
    record = {"schema": "prismabuild.action_progress.v1", "token": token,
              "phase": phase, "units_completed": units, "unit": unit,
              "reported_unix": time.time()}
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, path)
    return True


def parse_unit_scope(spec, count=None):
    """Read ``lo:hi`` as a half-open slice of the census's ``sorted(names)``.

    The stage fans out over the one order every reader already walks, so a
    row's range can be cut from the census alone and two rows with disjoint
    ranges never contend for a cell.
    """
    if spec is None:
        return None
    _require(isinstance(spec, str) and spec.count(":") == 1,
             "unit scope must be spelled lo:hi over the sorted census roster")
    low, _, high = spec.partition(":")
    start = 0 if not low.strip() else int(low)
    stop = (count if not high.strip() else int(high))
    _require(stop is not None, "an open-ended unit scope needs the roster size")
    _require(type(start) is int and type(stop) is int and 0 <= start < stop,
             f"empty or reversed unit scope {spec!r}")
    return (start, stop)


def _synthesize_render_from_wire(render, *, wire, record, name, fmt, shape, reader, device="cpu"):
    """Write the missing decoded PWC shard from the verified wire blob.

    A rung this campaign adopted rather than encoded has its wire but no
    ``.pt``. Re-encoding it costs GPU-hours on the critical path; decoding
    the wire costs an I/O pass. What the decode cannot buy is evidence about
    the encode, so the marker below is written FIRST: a crash between the two
    writes leaves a marker with no shard, which the next load re-synthesizes,
    whereas the other order would leave a shard that reads as ``encoded``.

    ``device`` is the decoder's, not the shard's. Tessera's decoder is device
    parameterized and the same pure-torch reconstruction runs wherever it is
    pointed; the published bytes are the canonical CPU BF16 tensor either way
    (``_canonical_rendered_weight_tensor``). A caller that has already
    reserved a GPU decodes on it rather than leaving it at idle while one
    Python thread walks the wire (measured 0.064 s vs 0.92 s an expert cell).
    """
    import torch
    from .production_weight_cache import _store_rendered_weight_entry

    blob = Path(wire).read_bytes()
    _same(hashlib.sha256(blob).hexdigest(), record["blob_sha256"],
          f"{name}@{fmt}: wire checksum before synthesizing its render")
    try:
        decoded = _decode_wire(blob, reader=reader, device=device).to(torch.bfloat16)
    except Exception as exc:
        raise ValueError(f"{name}@{fmt}: original decoded PWC shard missing and "
                         f"its wire does not decode: {exc}") from exc
    _require(isinstance(decoded, torch.Tensor) and decoded.dtype == torch.bfloat16 and
             list(decoded.shape) == list(shape) and bool(torch.isfinite(decoded).all()),
             f"{name}@{fmt}: decoded wire is not the census BF16 render")
    marker = _render_origin_marker_path(render)
    Path(render).parent.mkdir(parents=True, exist_ok=True)
    _json(marker, {"schema": RENDER_ORIGIN_SCHEMA, "render_origin": "synthesized_from_wire",
                   "unit": name, "format_name": fmt, "wire_sha256": record["blob_sha256"],
                   "wire_file": Path(wire).name})
    _store_rendered_weight_entry(weights={}, cache_dir_path=Path(render).parent,
                                 qname=name, fmt=fmt, tensor=decoded,
                                 weight_dtype=torch.bfloat16, durable=True)
    _require(Path(render).is_file(), f"{name}@{fmt}: synthesized render was not published")
    return "synthesized_from_wire"


def _resolve_render_origin(render, *, wire, record, name, fmt, shape, reader, device="cpu"):
    """Name where this rung's decoded PWC shard came from, never guess it.

    The campaign journals fresh and resumed wires through one receipt grammar
    (``_checkpoint_wire_record``), so nothing in the record distinguishes an
    adopted rung from an encoded one. The marker beside the shard is the only
    place that fact can live, and its absence beside an existing shard is the
    campaign's own render.
    """
    render, marker = Path(render), _render_origin_marker_path(render)
    if render.is_file():
        if not marker.is_file():
            return "encoded"
        stamp = json.loads(marker.read_text())
        _same(stamp.get("schema"), RENDER_ORIGIN_SCHEMA, f"{name}@{fmt}: render origin schema")
        _same(stamp.get("render_origin"), "synthesized_from_wire",
              f"{name}@{fmt}: marked render origin")
        _same(stamp.get("unit"), name, f"{name}@{fmt}: marked render unit")
        _same(stamp.get("format_name"), fmt, f"{name}@{fmt}: marked render format")
        _same(stamp.get("wire_sha256"), record["blob_sha256"],
              f"{name}@{fmt}: synthesized render names another wire")
        return "synthesized_from_wire"
    _require(Path(wire).is_file(), f"{name}@{fmt}: original decoded PWC shard missing")
    return _synthesize_render_from_wire(render, wire=wire, record=record, name=name,
                                        fmt=fmt, shape=shape, reader=reader, device=device)


def load_measured_anchor_input(inputs, *, file_hash_workers=1, verify_payloads=True,
                               defer_render_hashes=False, reader=None,
                               synthesis_device="cpu", unit_scope=None,
                               render_mirror_root=None, log_every=100,
                               require_existing_renders=False):
    """Read a complete merged journal and select only its measured wire cells.

    The default hashes all payload files. Preparation may explicitly defer
    payload reads to the existing PWC loader and wire verifier; metadata and
    roster gates still run here. Tensor/source/encoder verification occurs in
    ``prepare_cache`` using actual source weights and the original capture.
    Interpolated menu rows are deliberately excluded rather than converted.

    A prepared COST run can defer only render hashes to the PWC's verified
    consumption, after binding its prepared SHA roster. Every wire is still
    hashed here; a missing render or incomplete roster still refuses.

    A rung this campaign adopted has its wire but no decoded PWC shard. The
    shard is synthesized from that wire here and every cell carries the
    resulting ``render_origin``, so no later report can read "verified" as
    "independently compared". ``reader`` is the same bound Tessera consumer
    ``verify_anchor_render`` uses; there is one decode seam, not two.

    ``synthesis_device`` is where that decode runs. It changes no published
    byte -- the shard is the canonical CPU BF16 tensor either way -- so a
    caller holding a GPU reservation passes it rather than spending the
    reservation on one CPU core.

    ``unit_scope`` restricts the per-cell walk to ``sorted(names)[lo:hi]``.
    Every roster, seal and fanout gate above still runs over the whole
    census; only the cells are cut, which is what makes the standalone
    synthesis stage fannable. The result carries the scope, and a scoped
    read is refused wherever the complete campaign input is required.

    ``render_mirror_root`` publishes synthesized shards under that root
    instead of the campaign's row caches, so a measuring run can compare its
    bytes against the campaign's without being able to replace them.

    ``require_existing_renders`` is for metadata-only export handoff readers:
    it refuses a missing prepared shard instead of synthesizing into the
    campaign workspace. No export manifest builder may repair frozen inputs.

    ``log_every`` prints a cumulative count and rate every N synthesized
    shards. Silence is the defect this phase was reported for: it ran for
    hours at 2.6 cells/s saying nothing. The default is chosen against that
    measured rate rather than rounded -- 100 shards is ~38 s there and ~16 s
    on the GPU, inside the two minutes a silent phase is a defect after.
    """
    from .production_weight_cache import _cache_weight_filename
    from tools.dispatch_tessera_campaign import _require_receipts

    _require(type(verify_payloads) is bool, "verify_payloads must be an explicit boolean")
    _require(type(defer_render_hashes) is bool, "defer_render_hashes must be boolean")
    _require(not defer_render_hashes or verify_payloads,
             "deferred render hashes require complete wire verification")
    _require(not defer_render_hashes or require_existing_renders,
             "deferred render hashes require existing prepared renders")
    _require(not defer_render_hashes or unit_scope is None,
             "deferred render hashes require the complete candidate roster")
    _require(type(file_hash_workers) is int and file_hash_workers > 0,
             "positive file_hash_workers required")
    _require(type(log_every) is int and log_every >= 0, "non-negative log_every required")
    _require(type(require_existing_renders) is bool, "require_existing_renders must be boolean")
    # Hashing only part of a roster does not verify that roster, so the two
    # options are refused together rather than quietly producing a record
    # that reads as a verified campaign input.
    _require(unit_scope is None or not verify_payloads,
             "a scoped read cannot also verify the complete campaign payload")
    paths = {key: _bound(inputs[key], key) for key in (
        "campaign_plan", "census", "campaign_receipts", "merged_cost", "merged_checkpoint")}
    census = json.loads(paths["census"].read_text())
    plan = json.loads(paths["campaign_plan"].read_text())
    _same(plan.get("schema"), "prismaquant.tessera_campaign_plan.v1", "campaign plan schema")
    _same(Path(plan["census"]).resolve(), paths["census"].resolve(), "campaign census path")
    _same(paths["campaign_receipts"].resolve(),
          (paths["campaign_plan"].parent / "receipts.json").resolve(), "campaign receipt path")
    rows = plan["rows"]
    _require(len({row["row_id"] for row in rows}) == len(rows), "duplicate campaign row")
    _require_receipts(paths["campaign_plan"].parent, len(rows))
    owners, groups = {}, {}
    for row in rows:
        for name in row["members"]:
            _require(name not in owners, f"duplicate campaign unit {name}")
            owners[name] = Path(row["dir"])
        for group in row["groups"]:
            _require(group not in groups, f"duplicate campaign group {group}")
            groups[group] = row["row_id"]
    names = set(census["unit_shapes"])
    _same(set(owners), names, "complete census roster")
    _same(set(groups), set(census["anchor_groups"]), "complete census groups")
    _same(len(names), inputs["required_source_units"], "declared full source unit count")
    _same(len(groups), inputs["required_campaign_groups"], "declared full campaign group count")
    for group, members in census["anchor_groups"].items():
        owner = next(row for row in rows if row["row_id"] == groups[group])
        _require(set(members) <= set(owner["members"]), f"campaign group membership changed: {group}")

    payload = pickle.loads(paths["merged_cost"].read_bytes())
    _same(payload.get("schema"), CAMPAIGN_SCHEMA, "campaign cost schema")
    _same(payload.get("currency"), CURRENCY, "campaign scalar currency")
    _same(set(payload["costs"]), names, "complete merged cost roster")
    provenance = payload["provenance"]
    _same(provenance.get("cost_mode"), "production-render-score", "campaign cost mode")
    _same(provenance.get("model"), census["model"], "campaign model")
    _same(plan["model"], census["model"], "planned model")
    _require(provenance.get("stopped_early") is False, "campaign stopped before completing anchors")
    _same(provenance.get("campaign_fanout", {}).get("rows"),
          {row["row_id"]: sorted(row["groups"]) for row in rows}, "complete merged fanout")

    manifest = json.loads(paths["merged_checkpoint"].read_text())
    _same(manifest.get("schema"), MANIFEST_SCHEMA, "campaign checkpoint schema")
    _same(manifest.get("stage"), STAGE, "campaign checkpoint stage")
    identity = manifest["identity"]
    seal = canonical_json_sha256(identity, where="joint anchor input")
    _same(seal, manifest.get("identity_sha256"), "campaign checkpoint seal")
    _same(identity.get("campaign_schema"), CAMPAIGN_SCHEMA, "checkpoint campaign schema")
    _same(identity.get("currency"), CURRENCY, "checkpoint scalar currency")
    _same(set(identity["units"]), names, "complete checkpoint identity roster")
    listed = [row["qname"] for row in manifest["units"]]
    _require(len(listed) == len(names) and set(listed) == names, "incomplete checkpoint unit roster")
    for key in ("prismaquant_source_sha256", "encoder_source_sha256"):
        value = identity.get(key)
        _require(isinstance(value, str) and len(value) == 64 and
                 all(c in "0123456789abcdef" for c in value), f"missing checkpoint {key}")
    parts = paths["merged_checkpoint"].with_name(paths["merged_checkpoint"].name + ".parts")
    for row in manifest["units"]:
        _same(parts / row["file"], unit_path(parts, row["qname"]), "canonical checkpoint unit path")

    cells, formats = {}, {}
    wire_dir = Path(provenance["wire_dir"])
    roster = sorted(names)
    if unit_scope is not None:
        low, high = unit_scope
        _require(type(low) is int and type(high) is int and 0 <= low < high <= len(roster),
                 f"unit scope {unit_scope} is outside the {len(roster)}-unit census roster")
        roster = roster[low:high]
    synthesized, started = 0, time.time()
    for name in roster:
        state = _load_unit(unit_path(parts, name), stage=STAGE, qname=name, identity_sha256=seal)
        _require(isinstance(state, dict) and set(state) - {"unservable"} == {"anchors", "wire_records"},
                 f"{name}: incomplete measured anchor journal")
        anchors = {anchor["format_name"]: anchor for anchor in state["anchors"]}
        _require(anchors and len(anchors) == len(state["anchors"]) and
                 set(anchors) == set(state["wire_records"]), f"{name}: anchor/receipt coverage differs")
        measured = {fmt for fmt, row in payload["costs"][name].items()
                    if row.get("output_mse_measured") is True}
        _same(set(anchors), measured, f"{name}: measured payload/journal coverage")
        unit = identity["units"][name]
        _same(unit["weight"]["shape"], census["unit_shapes"][name], f"{name}: census source shape")
        for fmt, anchor in sorted(anchors.items()):
            row = payload["costs"][name][fmt]
            _require(fmt in unit["menu"] and anchor["qname"] == name, f"{name}: anchor outside exact menu")
            _require(row.get("cost_source") == "tessera_campaign_measured" and
                     row.get("tessera_provenance") == "measured" and row.get("currency") == CURRENCY,
                     f"{name}@{fmt}: interpolated or foreign measured row")
            for target, source in (("output_mse", "dloss"), ("tessera_family", "family"),
                    ("tessera_body_rate_q256", "body_rate_q256"), ("activation_contract", "activation_contract"),
                    ("activation_quantized", "activation_quantized"), ("wire_bytes", "wire_bytes"),
                    ("input_global_scale", "input_global_scale")):
                _same(row.get(target), anchor.get(source), f"{name}@{fmt}: measured {target}")
            _require(type(anchor["dloss"]) in (int, float) and math.isfinite(anchor["dloss"])
                     and anchor["dloss"] >= 0, f"{name}@{fmt}: invalid measured value")
            _same(row["hessian_identity"].get("applied"), anchor["hessian_applied"], f"{name}: H applicability")
            for key in ("supplied", "capture_sha256", "text_sha256", "fit_ids_sha256", "fit_tokens"):
                _same(row["hessian_identity"].get(key), provenance["hessian"].get(key), f"{name}: measured H {key}")
            if anchor.get("input_global_scale") is not None:
                _same(anchor["input_global_scale"], unit.get("input_global_scale"), f"{name}: checkpoint scale")
                _same(anchor["input_global_scale"], provenance["activation_static_scales"]["units"].get(name),
                      f"{name}: merged static scale")
            record = state["wire_records"][fmt]
            recorded = record["identity"]
            _same(recorded.get("unit"), name, f"{name}: wire unit")
            _same(recorded.get("source"), unit["weight"], f"{name}: recorded source")
            _same(recorded.get("encoder_source_sha256"), identity["encoder_source_sha256"], f"{name}: encoder source")
            _same(recorded["recipe"].get("q256"), anchor["body_rate_q256"], f"{name}: wire rung")
            if anchor["hessian_applied"]:
                _same(recorded["calibration"]["hessian"], unit["hessian"], f"{name}: recorded H")
            else:
                _same(recorded.get("calibration"), None, f"{name}: unexpected recorded H")
            filename = record["file"]
            _require(isinstance(filename, str) and Path(filename).name == filename and
                     filename not in {".", ".."}, f"{name}: escaping wire filename")
            wire = wire_dir / filename
            _require(not wire.is_symlink() and wire.resolve().parent == wire_dir.resolve(), f"{name}: escaping wire path")
            _same(wire.stat().st_size, record["blob_bytes"], f"{name}: wire size")
            render = owners[name] / "cache" / _cache_weight_filename(name, fmt)
            target = (render if render_mirror_root is None
                      else _render_mirror_path(render, render_mirror_root))
            present = Path(target).is_file()
            if require_existing_renders and not present:
                raise ValueError(f"{name}@{fmt}: prepared render is missing; selected cache will not synthesize it")
            origin = _resolve_render_origin(target, wire=wire, record=record, name=name,
                                            fmt=fmt, shape=census["unit_shapes"][name],
                                            reader=reader, device=synthesis_device)
            cells[name, fmt] = {"anchor": anchor, "record": record, "wire": str(wire.resolve()),
                               "render": str(Path(target).resolve()), "render_origin": origin,
                               **({} if render_mirror_root is None
                                  else {"campaign_render": str(render.resolve())})}
            if not present:
                synthesized += 1
                # After the shard is durable, never on entering the loop.
                _pb_commit(synthesized, "synthesize", unit=f"{name}@{fmt}")
                if log_every and synthesized % log_every == 0:
                    elapsed = max(time.time() - started, 1e-9)
                    print(f"tessera_joint_aura: synthesized {synthesized} renders "
                          f"in {elapsed:.0f}s ({synthesized / elapsed:.2f} cells/s), "
                          f"at {name}@{fmt}", flush=True)
        formats[name] = (*sorted(anchors), "BF16")
    if log_every and synthesized:
        elapsed = max(time.time() - started, 1e-9)
        print(f"tessera_joint_aura: synthesized {synthesized} renders in {elapsed:.0f}s "
              f"({synthesized / elapsed:.2f} cells/s) on {synthesis_device}", flush=True)
    scoped = dict(unit_scope=unit_scope, synthesized_now=synthesized,
                  render_mirror_root=None if render_mirror_root is None else str(render_mirror_root))
    if not verify_payloads:
        return MeasuredAnchorInput(dict(inputs), payload, manifest, census, plan, cells,
                                   formats, **scoped)

    def verify_files(item):
        pair, cell = item
        wire, render = Path(cell["wire"]), Path(cell["render"])
        # Metadata is only a race detector around the actual content hash.
        # Every byte is still hashed; neither timestamps nor a previous run
        # authorize reuse. Existing per-consumption render checks remain below.
        def signature(path):
            stat = path.stat()
            return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns
        before = [signature(p) for p in (wire, render)]
        _same(_sha(wire), cell["record"]["blob_sha256"], f"{pair}: wire checksum")
        digest = None if defer_render_hashes else _sha(render)
        after = [signature(p) for p in (wire, render)]
        _same(after, before, f"{pair}: input files changed while hashing")
        return pair, digest

    if file_hash_workers == 1:
        verified_files = map(verify_files, cells.items())
        for pair, digest in verified_files:
            if digest is not None:
                cells[pair]["render_file_sha256"] = digest
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=file_hash_workers, thread_name_prefix="anchor-file-hash") as workers:
            for pair, digest in workers.map(verify_files, cells.items()):
                if digest is not None:
                    cells[pair]["render_file_sha256"] = digest
    return MeasuredAnchorInput(dict(inputs), payload, manifest, census, plan, cells,
                               formats, **scoped)


def calibrated_maxima(data, profile):
    """Reuse the producer's full-census fused scale policy; never invert G."""
    from . import tessera_campaign as tc
    from .nvfp4_activation_contract import unify_fused_sibling_max_abs

    positive = {name: float(value) for name, value in data.census["max_abs"].items()
                if float(value) > 0.0}
    maxima = unify_fused_sibling_max_abs(positive, profile=profile, tolerate_profile_errors=True)
    scales, policy = tc._static_input_scales(data.census["max_abs"], profile=profile)
    stamped = data.payload["provenance"]["activation_static_scales"]
    _same(policy, stamped["policy"], "campaign static scale policy")
    _same(scales, stamped["units"], "campaign fused static scales")
    return maxima, scales


def verify_anchor_render(cell, source_weight, rendered_weight, *, calibration_source,
                         projected_unit, static_scales, bound_unit=None, reader=None,
                         release_file_pages=False, wire_blob=None, wire_sha256=None):
    """Re-derive encoder inputs from actual source/H and compare decoded bytes.

    Two legs, and they do not establish the same thing. ``verify_cached_unit``
    checks the wire against an encoder identity re-derived from the streamed
    source weights and H; it is independent of anything the cache holds and it
    is what qualifies an adopted rung at all. The ``torch.equal`` leg compares
    the decoded wire with the render on disk: for an ``encoded`` rung that is
    an independent render/wire agreement, and for a ``synthesized_from_wire``
    rung the render was written by decoding that same wire, so it can only
    establish that the shard still round-trips -- a corruption check between
    the write and this read, not evidence about the encode. The returned
    record names both facts so a reader never has to infer which one it holds.
    """
    import torch
    from . import tessera_campaign as tc
    from .production_weight_cache import _cb_cache_tensor_identity

    anchor = tc.CampaignAnchor(**cell["anchor"])
    name, fmt = anchor.qname, anchor.format_name
    render_origin = cell.get("render_origin")
    _require(render_origin in RENDER_ORIGINS,
             f"{name}@{fmt}: cell carries no closed-vocabulary render_origin")
    render_comparison = RENDER_COMPARISON_BY_ORIGIN[render_origin]
    _require(source_weight.dtype == rendered_weight.dtype == torch.bfloat16 and
             source_weight.ndim == 2 and rendered_weight.shape == source_weight.shape,
             f"{name}@{fmt}: source/render BF16 shape differs")
    source_receipt = (None if bound_unit is None else bound_unit.source_receipt(source_weight))
    _require((bound_unit is not None or bool(torch.isfinite(source_weight).all())) and
             bool(torch.isfinite(rendered_weight).all()), f"{name}@{fmt}: source/render is nonfinite")
    expected = tc._checkpoint_anchor_identity(anchor,
        weights={name: source_weight}, menus={name: [SimpleNamespace(format_name=fmt)]},
        calibration_source=calibration_source, static_scales=static_scales,
        projected_units={} if projected_unit is None else {name: projected_unit},
        **({} if bound_unit is None else {"bound_unit": bound_unit}))
    wire_path = Path(cell["wire"])
    wire_stat = wire_path.stat() if release_file_pages else None
    if wire_blob is None:
        blob, actual_wire_sha256 = _read_verified_wire_blob(cell)
    else:
        _require(isinstance(wire_blob, bytes), f"{name}@{fmt}: wire reader returned non-bytes")
        _same(len(wire_blob), cell["record"].get("blob_bytes"),
              f"{name}@{fmt}: read-ahead wire size differs from receipt")
        blob = wire_blob
        actual_wire_sha256 = hashlib.sha256(blob).hexdigest()
        if wire_sha256 is not None:
            _same(wire_sha256, actual_wire_sha256,
                  f"{name}@{fmt}: read-ahead wire digest changed")
        _same(actual_wire_sha256, cell["record"].get("blob_sha256"),
              f"{name}@{fmt}: read-ahead wire checksum")
    verifier = tc._checkpoint_identity_api() if reader is None else reader
    verifier.verify_cached_unit(blob, cell["record"], expected)
    decoded = _decode_wire(blob, reader=reader,
                           device=str(rendered_weight.device)).to(torch.bfloat16)
    # Run on both origins. On a synthesized render it cannot fail as evidence
    # about the encode, and it is still live evidence that the shard on disk
    # decodes to the bytes it was written from.
    _require(torch.equal(decoded, rendered_weight),
             f"{name}@{fmt}: decoded wire differs from original PWC render"
             if render_origin == "encoded" else
             f"{name}@{fmt}: synthesized PWC render no longer decodes from its wire")
    del decoded
    if release_file_pages:
        from .perturbed_x_cache import release_activation_cache_file_pages
        release_activation_cache_file_pages(wire_path, expected_stat=wire_stat)
    return {"source_weight": (_cb_cache_tensor_identity(source_weight)
                              if source_receipt is None else source_receipt),
            "rendered_weight": _cb_cache_tensor_identity(rendered_weight),
            "encoding_identity_sha256": canonical_json_sha256(expected, where="joint anchor encoding"),
            "wire_sha256": actual_wire_sha256,
            "render_file_sha256": cell["render_file_sha256"],
            "render_origin": render_origin, "render_comparison": render_comparison}


def _live_targets(runner, names):
    from .aura_cost import _target_linears
    from .routed_experts import profile_declared_packed_expert_projections

    targets = _target_linears(runner.model, include_routed_experts=True, profile=runner.profile)
    packed = profile_declared_packed_expert_projections(runner.model, runner.profile)
    targets.update({member.qname: member for member in packed})
    _require(set(names) <= set(targets), "census units are absent from the actual streamed source")
    return {name: targets[name] for name in names}


def _prepare_file_read_bound(data, *, max_render_bytes):
    """Refuse an oversized later donor before any layer allocates read buffers."""
    maximum = max(Path(cell["render"]).stat().st_size for cell in data.cells.values())
    _require(0 < maximum <= max_render_bytes,
             "original render shard exceeds the declared PWC read buffer budget")
    return maximum


def _prepare_wire_read_bound(data):
    """Reserve the one bounded, receipt-sized wire buffer used by qualification."""
    sizes = [cell["record"].get("blob_bytes") for cell in data.cells.values()]
    _require(bool(sizes) and all(type(size) is int and size > 0 for size in sizes),
             "measured wire receipts need positive blob_bytes")
    return max(sizes)


QUALIFICATION_WINDOW_SCHEMA = "prismaquant.joint_anchor_qualification.v1"
QUALIFICATION_STAGE = "Tessera joint anchor qualification"
QUALIFICATION_CELLS_SCHEMA = "prismaquant.joint_qualification_cells.v1"


def normalize_qualification_window(config):
    if config is None:
        return None
    fields = {"schema", "max_capture_resident_bytes", "max_load_buffer_bytes",
              "workspace_reserve_bytes"}
    _require(isinstance(config, dict) and set(config) == fields and
             config.get("schema") == QUALIFICATION_WINDOW_SCHEMA,
             "joint anchor qualification requires a complete v1 window policy")
    for key in fields - {"schema"}:
        _require(type(config[key]) is int and config[key] > 0,
                 f"qualification window requires positive finite {key}")
    return dict(config)


def _qualification_capture_sizes(data, identity, policy):
    """Validate the whole roster before a first unit's X/H can be loaded."""
    sizes = {}
    for name in data.formats_by_qname:
        columns = data.census["unit_shapes"][name][1]
        rows = min(data.census["counts"][name], identity["max_act_rows"])
        _require(type(columns) is int and columns > 0 and type(rows) is int and rows >= 0,
                 f"{name}: invalid canonical capture geometry")
        sizes[name] = 4 * (columns * columns + rows * columns)
        _require(sizes[name] <= policy["max_capture_resident_bytes"],
                 f"{name}: canonical capture exceeds qualification budget")
    return sizes


def _qualification_file_sha(path):
    """Hash one held regular file; reject symlinks and path/descriptor drift."""
    path = Path(path)
    def signature(value):
        return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
                value.st_mtime_ns, value.st_ctime_ns)
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode), f'qualification input is not a regular file: {path}')
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError as error:
        raise RuntimeError(f'qualification input changed before its read: {path}') from error
    with os.fdopen(fd, 'rb') as handle:
        _same(signature(os.fstat(handle.fileno())), signature(before),
              f'qualification input changed before its read: {path}')
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        _same(signature(os.fstat(handle.fileno())), signature(before),
              f'qualification input changed while hashing: {path}')
    _same(signature(path.lstat()), signature(before),
          f'qualification input changed while hashing: {path}')
    return digest


def _qualification_replay(data, manifest, completed):
    """A journal envelope alone does not authenticate files that remain live."""
    from .perturbed_x_cache import activation_cache_filename

    # The immutable manifest seals the full roster. Only a unit whose
    # qualification is actually skipped needs its X/H bytes re-authenticated
    # here; unfinished units pass the usual verified capture loader later.
    root = Path(data.payload['provenance']['calibration_cache']['path']).parent
    for name in sorted(completed):
        entry = manifest['entries'][name]
        expected = str(Path('inputs') / activation_cache_filename(name))
        _same(entry.get('path'), expected, f'{name}: canonical X/H entry')
        _same(_qualification_file_sha(root / expected), entry['sha256'],
              f'{name}: canonical X/H bytes changed')
    verified = {}
    for name, state in sorted(completed.items()):
        rows = state.get('verified_cells')
        _require(isinstance(rows, dict) and set(rows) == set(data.formats_by_qname[name]) - {'BF16'},
                 f'{name}: incomplete qualification journal cells')
        _require(isinstance(state.get('prefetch'), list), f'{name}: missing qualification prefetch')
        for fmt, record in rows.items():
            cell = data.cells[name, fmt]
            required = {'source_weight', 'rendered_weight', 'encoding_identity_sha256',
                        'wire_sha256', 'render_file_sha256', 'render_origin',
                        'render_comparison', 'activation'}
            _require(isinstance(record, dict) and set(record) == required and
                     isinstance(record['activation'], dict),
                     f'{name}@{fmt}: incomplete qualification receipt')
            _same(record.get('render_origin'), cell['render_origin'],
                  f'{name}@{fmt}: journal render origin')
            _same(record.get('render_comparison'), RENDER_COMPARISON_BY_ORIGIN[cell['render_origin']],
                  f'{name}@{fmt}: journal render comparison')
            _same(record.get('wire_sha256'), cell['record']['blob_sha256'],
                  f'{name}@{fmt}: journal wire receipt')
            _same(_qualification_file_sha(cell['wire']), record['wire_sha256'],
                  f'{name}@{fmt}: upstream wire bytes changed')
            _same(_qualification_file_sha(cell['render']), record.get('render_file_sha256'),
                  f'{name}@{fmt}: upstream render bytes changed')
            cell['render_file_sha256'] = record['render_file_sha256']
            verified[name, fmt] = record
    return verified


def _qualification_cells_sha256(cells):
    """Seal exact cell fields without a second, full-roster JSON allocation.

    The parent campaign checkpoint, cost and census are independently bound.
    Origin markers and resolved render/wire paths can still differ, so hash
    their complete sorted cell records here. Length framing keeps adjacent
    variable-sized JSON rows unambiguous; one row is the largest live buffer.
    """
    digest = hashlib.sha256(QUALIFICATION_CELLS_SCHEMA.encode() + b"\n")
    for name, fmt in sorted(cells):
        cell = cells[name, fmt]
        row = json.dumps((name, fmt, {
            'anchor': cell['anchor'], 'record': cell['record'],
            'render': cell['render'], 'wire': cell['wire'],
            'render_origin': cell['render_origin'],
        }), sort_keys=True, separators=(',', ':'), ensure_ascii=False,
            allow_nan=False).encode('utf-8')
        digest.update(len(row).to_bytes(8, 'big'))
        digest.update(row)
    return digest.hexdigest()


def prepare_cache(runner, data, *, capture, max_render_bytes, reader=None, file_load_workers=4,
                  qualification_window=None, capture_load_policy=None,
                  source_capture_compatibility=None, source_authentication=None,
                  qualification_guard=None, qualification_journal=None,
                  qualification_resume=False, qualification_identity=None):
    """Qualify original per-layer inputs and return the existing PWC object.

    Only the original calibration/PWC/source prefetch mechanisms own tensors.
    PWC's LRU records absolute donor paths so compact/release stays reversible
    even though the merged renders have more than one original directory.
    """
    import torch
    from contextlib import nullcontext
    from . import tessera_calibration_cache as cc, tessera_hessian as th, tessera_campaign as tc
    from .joint_aura import activation_identity, prefetch_joint_cache
    from .production_weight_cache import ProductionWeightCache
    from .routed_experts import PackedExpertProjection, refresh_packed_expert_projections
    from . import format_registry as fr

    _require(type(max_render_bytes) is int and max_render_bytes > 0, "positive PWC residency budget required")
    policy = normalize_qualification_window(qualification_window)
    from .perturbed_x_cache import normalize_verified_activation_load
    capture_load_policy = normalize_verified_activation_load(capture_load_policy)
    if capture_load_policy is not None:
        _require(policy is not None, 'verified capture loading requires explicit qualification windows')
    guard = qualification_guard
    if policy is not None and str(runner.device).startswith('cuda'):
        import os
        from .autoscale import require_bounded_capture_environment
        from .memory_management import CaptureMemoryGuard
        require_bounded_capture_environment(os.environ)
        if guard is None:
            guard = CaptureMemoryGuard(runner.device)
        guard.check('before_joint_qualification_identity')
    capture_path = _bound(capture, "canonical capture")
    stamped_capture = data.payload["provenance"].get("calibration_cache")
    _same(capture, stamped_capture, "priced canonical capture")
    manifest = cc.require_capture_contract(capture_path, expected_sha256=capture["sha256"])
    from .glm_capture_compatibility import require_capture_compatibility
    require_capture_compatibility(source_capture_compatibility, capture=capture, model=runner.model)
    recorded = manifest["identity"]
    # This verifies recorded canonical capture provenance plus current source
    # bytes/runtime. It does not pretend a from-config streaming skeleton is an
    # ordinary from_pretrained model or manufacture an initialization witness.
    expected = cc.capture_identity(data.inputs["census"]["path"],
        calibration=data.payload["provenance"]["hessian"]["calibration_identity"],
        max_act_rows=recorded["max_act_rows"],
        model_load_contract=data.census["model_load_contract"],
        attention_implementation=data.census["attention_implementation"],
        **({'source_authentication': source_authentication}
           if source_authentication is not None else {}),
        **(dict(resource_check=None if guard is None else guard.check,
                release_read_pages=True) if policy is not None else {}))
    _same(expected, recorded, "current source/canonical capture")
    _same(data.manifest["identity"]["calibration"], recorded["calibration"], "journal/canonical draw")
    metadata_owner = (None if policy is None else cc.open_capture_metadata(
        capture_path, expected_identity=expected, expected_sha256=capture['sha256']))
    capture_sizes = (None if policy is None else
                     _qualification_capture_sizes(data, expected, policy))
    _require(qualification_journal is None or
             (policy is not None and isinstance(qualification_identity, dict)),
             'qualification journal requires explicit windows and run identity')
    journal = journal_sha = None
    completed = {}
    if qualification_journal is not None:
        identity = dict(qualification_identity, schema='prismaquant.joint_qualification_journal.v1',
            inputs=data.inputs, campaign_checkpoint_sha256=data.manifest['identity_sha256'],
            capture=capture, capture_identity=expected,
            cells_digest_schema=QUALIFICATION_CELLS_SCHEMA,
            cell_count=len(data.cells), cells_sha256=_qualification_cells_sha256(data.cells),
            qualification_window=policy, capture_load_policy=capture_load_policy,
            source_capture_compatibility=source_capture_compatibility,
            max_render_bytes=max_render_bytes)
        journal, journal_sha, completed = prepare_journal(
            qualification_journal, stage=QUALIFICATION_STAGE,
            resume=qualification_resume, identity=identity,
            qnames=sorted(data.formats_by_qname))
        if completed:
            replayed = _qualification_replay(data, manifest, completed)
        else:
            replayed = {}
    capture_load_execution = cc._load_execution(capture_load_policy, expected)
    if capture_load_policy is not None:
        cc.preflight_verified_capture_entries(capture_path.parent, manifest['entries'],
            names=sorted(data.formats_by_qname), policy=capture_load_policy,
            census=data.census, max_rows=expected['max_act_rows'])
    maxima, scales = calibrated_maxima(data, runner.profile)
    cache = ProductionWeightCache(
        weights={pair: cell["render"] for pair, cell in data.cells.items()},
        levers={"tessera_campaign": True}, activation_max_abs=maxima,
        metadata={"schema": PREPARED_SCHEMA, "inputs": data.inputs,
                  "reader_identity": None if reader is None else reader.identity})
    cache.enable_lru(max_render_bytes)
    max_file_bytes = _prepare_file_read_bound(data, max_render_bytes=(max_render_bytes
        if policy is None else min(max_render_bytes, policy["max_load_buffer_bytes"])))
    max_wire_read_bytes = _prepare_wire_read_bound(data)
    cache.enable_file_load_receipts(max_file_bytes=max_file_bytes)
    targets = _live_targets(runner, data.formats_by_qname)
    layers = defaultdict(list)
    for name in targets:
        layers[runner.layer_index_for_qname(name)].append(name)
    projected = {name: unit for units in (data.census.get("expert_projection") or {}).get("stacks", {}).values()
                 for name, unit in units.items()}
    renders = {name: tuple(fmt for fmt in fmts if fmt != "BF16")
               for name, fmts in data.formats_by_qname.items()}
    verified, telemetry = replayed if completed else {}, []
    committed_units = len(completed)
    for depth in range(min(runner.num_layers, runner.prefetch_lookahead + 1)):
        runner.context.schedule_prefetch(depth)
    for layer in range(runner.num_layers):
        names = sorted(layers.get(layer, ()))
        runner.context.install(layer, require_prefetched=runner.require_prefetched_residency)
        runner.context.schedule_prefetch(layer + runner.prefetch_lookahead)
        members = [targets[name] for name in names if isinstance(targets[name], PackedExpertProjection)]
        try:
            targets.update({member.qname: member for member in refresh_packed_expert_projections(members, runner.profile)})
            if not names:
                continue
            if policy is not None:
                runner.context.settle_prefetched_layers(range(layer + 1,
                    min(runner.num_layers, layer + 1 + runner.prefetch_lookahead)))
            capture_windows = [names] if policy is None else [(name,) for name in names]
            layer_stats = []
            for unit_names in capture_windows:
                if len(unit_names) == 1 and unit_names[0] in completed:
                    name = unit_names[0]
                    layer_stats.extend(completed[name]['prefetch'])
                    if capture_load_execution is not None:
                        partial = completed[name].get('capture_load_execution')
                        _require(isinstance(partial, dict), f'{name}: missing capture load execution')
                        cc.merge_load_execution(capture_load_execution, partial)
                    continue
                stats_start = len(layer_stats)
                acts = hessians = calibration_source = source_weight = None
                resident = rendered = bound_unit = None
                unit_load_execution = {}
                try:
                    if guard is not None:
                        guard.check('before_joint_qualification_unit:' + unit_names[0], reserve_bytes=
                            2 * capture_sizes[unit_names[0]] + max_render_bytes +
                            policy['max_load_buffer_bytes'] + 2 * max_wire_read_bytes +
                            policy['workspace_reserve_bytes'] +
                            (0 if capture_load_policy is None else 2 * capture_load_policy['max_buffer_bytes'] +
                             capture_load_policy['max_scratch_bytes']))
                    (acts, hessians, _counts, _maxima), _receipt = cc.prefetch_capture(capture_path,
                        expected_sha256=capture["sha256"],
                        **({"expected_identity": expected} if metadata_owner is None else
                           {"metadata_owner": metadata_owner}),
                        census=data.census, names=unit_names, device=runner.device,
                        **(dict(resource_check=None if guard is None else guard.check,
                                release_file_pages=True) if policy is not None else {}),
                        **(dict(verified_load_policy=capture_load_policy,
                                load_execution=unit_load_execution) if capture_load_policy is not None else {}))
                    if capture_load_execution is not None:
                        cc.merge_load_execution(capture_load_execution, unit_load_execution)
                    calibration_source = th.activation_source(hessians, expected["calibration"])
                    if policy is None:
                        layer_stats.append(prefetch_joint_cache(cache, unit_names, renders,
                            max_resident_bytes=max_render_bytes, max_workers=file_load_workers))
                    for name in unit_names:
                        source_weight = targets[name].weight.detach()
                        anchors = [tc.CampaignAnchor(**data.cells[name, fmt]["anchor"]) for fmt in renders[name]]
                        keys = tuple((name, fmt) for fmt in renders[name])
                        windows = ((keys,) if policy is None else cache.plan_resident_windows(keys,
                            max_resident_bytes=min(max_render_bytes, policy['max_load_buffer_bytes']),
                            max_workers=file_load_workers))
                        with tc.bind_checkpoint_unit_identity(anchors, source_weight=source_weight,
                                calibration_source=calibration_source, projected_unit=projected.get(name),
                                static_scales=scales) as bound_unit:
                            for window in windows:
                                owner = (nullcontext() if policy is None else cache.resident_window(window,
                                    max_resident_bytes=max_render_bytes, max_workers=file_load_workers,
                                    max_load_buffer_bytes=policy['max_load_buffer_bytes'], release_file_pages=True))
                                with owner as window_receipt:
                                    if window_receipt is not None:
                                        layer_stats.append(dict(unit=name, **window_receipt))
                                    # One ephemeral wire blob is read ahead while the GPU
                                    # verifies its predecessor.  Keep its worker inside the
                                    # resident window so an exception waits for/cancels it
                                    # before PWC releases the tensors it could overlap.
                                    from concurrent.futures import ThreadPoolExecutor
                                    wire_reader = ThreadPoolExecutor(max_workers=1,
                                                                     thread_name_prefix="joint-wire-read")
                                    pending = None
                                    try:
                                        for index, (_, fmt) in enumerate(window):
                                            cell = data.cells[name, fmt]
                                            if pending is None:
                                                pending = wire_reader.submit(_read_verified_wire_blob, cell)
                                            # Consume before borrowing the resident tensor: a
                                            # failed wire never leaves a background read after
                                            # this window's PWC entries have been released.
                                            blob, wire_sha256 = pending.result()
                                            pending = (None if index + 1 == len(window) else
                                                       wire_reader.submit(_read_verified_wire_blob,
                                                                          data.cells[name, window[index + 1][1]]))
                                            resident = (cache.get(name, fmt) if policy is None else cache.get_resident(name, fmt))
                                            receipt = cache.file_load_receipt((name, fmt), resident)
                                            if "render_file_sha256" in cell:
                                                _same(receipt["sha256"], cell["render_file_sha256"], f"{name}: original render file changed")
                                            cell["render_file_sha256"] = receipt["sha256"]
                                            rendered = resident.to(runner.device)
                                            record = verify_anchor_render(cell, source_weight, rendered,
                                                calibration_source=calibration_source,
                                                projected_unit=projected.get(name), static_scales=scales,
                                                bound_unit=bound_unit, reader=reader, wire_blob=blob,
                                                wire_sha256=wire_sha256,
                                                **({'release_file_pages': True} if policy is not None else {}))
                                            activation = activation_identity(fr.get_format(fmt), cache.activation_max_abs, name)
                                            _same(activation["input_global_scale"], cell["anchor"].get("input_global_scale"),
                                                  f"{name}@{fmt}: joint/campaign static scale")
                                            record["activation"] = activation
                                            verified[name, fmt] = record
                                            resident = rendered = blob = None
                                    finally:
                                        if pending is not None:
                                            pending.cancel()
                                        wire_reader.shutdown(wait=True, cancel_futures=True)
                                    if guard is not None:
                                        guard.check('after_joint_qualification_window:' + name)
                finally:
                    acts = hessians = calibration_source = source_weight = None
                    resident = rendered = bound_unit = None
                if guard is not None:
                    guard.check('after_joint_qualification_unit:' + unit_names[0])
                if journal is not None:
                    name = unit_names[0]
                    write_unit(journal, stage=QUALIFICATION_STAGE, qname=name,
                        identity_sha256=journal_sha,
                        state={'verified_cells': {fmt: verified[name, fmt] for fmt in renders[name]},
                               'prefetch': layer_stats[stats_start:],
                               **({'capture_load_execution': unit_load_execution}
                                  if capture_load_execution is not None else {})})
                    committed_units += 1
                    _pb_commit(committed_units, 'qualification', unit=name)
            stats = layer_stats[0] if policy is None else {'windows': layer_stats}
            telemetry.append({"layer": layer, **stats})
            print(json.dumps({"qualified_layer": layer, "qualified_cells": len(verified),
                              "total_cells": len(data.cells), "prefetch": stats}), flush=True)
        finally:
            cache.compact_for_pickle()
            runner.context.unload(layer)
            targets.update({member.qname: member for member in refresh_packed_expert_projections(members, runner.profile)})
    _same(set(verified), set(data.cells), "complete qualified wire/render roster")
    cache.disable_file_load_receipts()
    census_of_renders = cell_render_census(data.cells)
    for pair, record in verified.items():
        _same(record["render_origin"], data.cells[pair]["render_origin"],
              f"{pair}: qualified render origin")
    _same(render_origin_census(record["render_origin"] for record in verified.values()),
          census_of_renders, "qualified render origin census")
    cache.metadata.update({"verified_cells": verified, "prefetch": telemetry,
        **census_of_renders,
        **({'capture_load_execution': capture_load_execution} if capture_load_execution is not None else {}),
        **({"qualification_window": policy, "capture_resident_bytes": capture_sizes,
            "qualification_memory_guard": None if guard is None else guard.snapshot()}
           if policy is not None else {})})
    return cache


def _source_prefetch(config):
    prefetch = config.get("source_prefetch")
    fields = {"max_cache_slots", "prefetch_workers", "prefetch_lookahead",
              "cache_headroom_gb", "prefetch_min_available_gb",
              "require_prefetched_residency"}
    _require(isinstance(prefetch, dict) and set(prefetch) == fields,
             "explicit complete source_prefetch settings required")
    _require(prefetch["require_prefetched_residency"] is True,
             "source_prefetch must require prefetched residency")
    for name in ("max_cache_slots", "prefetch_workers", "prefetch_lookahead"):
        _require(type(prefetch[name]) is int and prefetch[name] > 0,
                 f"source_prefetch requires positive {name}")
    _require(prefetch["prefetch_lookahead"] < prefetch["max_cache_slots"],
             "source_prefetch lookahead must fit the declared cache slots")
    for name in ("cache_headroom_gb", "prefetch_min_available_gb"):
        _require(type(prefetch[name]) in (int, float) and
                 math.isfinite(prefetch[name]) and prefetch[name] > 0,
                 f"source_prefetch requires positive finite {name}")
    return dict(prefetch)


def _operator_window_policy(config):
    from .joint_statistics_replay import normalize_operator_windows
    policy = normalize_operator_windows(config['execution'].get('operator_windows'))
    if policy is not None:
        _require(config['execution'].get('boundary_storage') is not None,
                 'operator-window campaign requires explicit exact boundary storage')
        _require(policy['max_render_resident_bytes'] <= config['max_render_bytes'],
                 'operator-window PWC cap exceeds campaign render admission')
    return policy


def _admit_candidate_phase(command, config, data, layer_bytes):
    """Keep legacy whole-layer admission; explicit windows admit each donor."""
    policy = _operator_window_policy(config)
    if command == 'run' and policy is not None:
        _prepare_file_read_bound(data, max_render_bytes=min(
            policy['max_render_resident_bytes'], policy['max_load_buffer_bytes']))
    elif command != 'prepare' or config.get('qualification_window') is None:
        _require(max(layer_bytes.values()) <= config['max_render_bytes'],
                 'largest measured candidate layer exceeds explicit PWC budget')
    return policy


def _load_plan(path, digest, *, projection_runtime=True):
    path = _bound({"path": str(path), "sha256": digest}, "joint anchor plan")
    config = json.loads(path.read_text())
    _same(config.get("schema"), SCHEMA, "joint anchor plan schema")
    if config.get("source_identity_cache") is not None:
        _bound(config["source_identity_cache"], "source identity cache")
    _source_prefetch(config)
    execution = config["execution"]
    from .glm_source_derivative import normalize_source_derivative
    normalize_source_derivative(execution.get('source_derivative'))
    normalize_qualification_window(config.get("qualification_window"))
    from .perturbed_x_cache import normalize_verified_activation_load
    if normalize_verified_activation_load(config.get('capture_load_policy')) is not None:
        _require(config.get('qualification_window') is not None,
                 'verified capture loading requires explicit qualification windows')
    from .joint_projection_backend import normalize_projection_backend, require_qualified_environment
    selector = normalize_projection_backend(execution.get("projection_backend"))
    if projection_runtime and selector["name"] != "torch":
        # Step 3a loads this plan inside the campaign's own container spec, so
        # the identity read here is the executing image's. Every capture-free
        # axis is compared -- torch, cuda, machine, ATen headers, compiler and
        # the image the launcher stamped. The ``device`` block needs
        # ``torch.cuda.get_device_properties``, which a ``--cpu-only``
        # preflight container does not have, so it is compared only when CUDA
        # is present here and is otherwise refused by the first gate in
        # ``execute`` -- seconds into the pass, before any render is written.
        require_qualified_environment()
    from .cost_streaming import normalize_boundary_storage
    normalize_boundary_storage(execution.get("boundary_storage"))
    _operator_window_policy(config)
    _require(type(config.get("file_hash_workers", 1)) is int and config.get("file_hash_workers", 1) > 0,
             "positive file_hash_workers required")
    for name, minimum in (("n_calib_samples", 1), ("calib_seqlen", 1),
                          ("probe_microbatch", 1), ("n_probes", 2)):
        _require(type(execution.get(name)) is int and execution[name] >= minimum,
                 f"explicit positive {name} required")
    _require(type(execution.get("seed_base")) is int, "explicit probe seed required")
    _same(execution.get("token_scope"), "all", "full-draw joint token scope")
    _same(execution.get("temperature"), 1.0, "joint probe temperature")
    _same(execution.get("production_act_scales"), "0", "campaign optional activation clipping")
    _require(config.get("profile_tool") in {"cprofile", "py-spy"},
             "explicit supported full-duration profiler required")
    for name in ("max_render_bytes", "max_gpu_bytes"):
        _require(type(config.get(name)) is int and config[name] > 0, f"positive {name} required")
    _require(type(config.get("min_free_gib")) in (int, float) and config["min_free_gib"] >= 0,
             "nonnegative memory floor required")
    return config


def _io_counters():
    values = {}
    for line in Path("/proc/self/io").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value)
    return values


ACTIVATION_SCALE_ENV = "PRISMAQUANT_PROD_ACT_SCALES"


def _prepare_source_owner(config, data, *, resource_check=None):
    """Bind the complete capture before any streamed source tensor is read.

    The selected-source descriptor owner already authenticates a shard once,
    at its first payload read, and checks replacement for the whole lease.
    Preparation uses that same owner for its full roster: the late completion
    gate authenticates any source file no streamed layer needed.
    """
    from . import tessera_calibration_cache as cc

    capture = config["canonical_capture"]
    manifest = cc.require_capture_contract(capture["path"],
                                           expected_sha256=capture["sha256"])
    owner = cc.authenticate_selected_capture_source(
        config["inputs"]["census"]["path"], capture["path"],
        expected_sha256=capture["sha256"], model=config["model"],
        max_act_rows=manifest["identity"]["max_act_rows"],
        attention_implementation=data.census["attention_implementation"],
        resource_check=resource_check, release_read_pages=True)
    try:
        identity_cache = Path(config["output_root"]) / "prepare/source-identity.json"
        if identity_cache.is_file():
            adopted = owner.adopt_streamed_identity_cache(identity_cache)
            print(f"tessera_joint_aura: adopted {adopted} full source SHA proofs "
                  f"from {identity_cache}", flush=True)
        return owner
    except BaseException:
        owner.close()
        raise


def _seed_source_identity_cache(config, root):
    """Carry an explicitly bound old digest record into this pass's cache slot.

    A new output root otherwise makes ``build_streamed_model_identity`` hash
    the whole source again before it can compare the live model. This copies
    only its existing identity JSON, never a weight or a render, and refuses
    any pre-existing different local cache rather than mixing two proofs.
    """
    destination = Path(root) / "source-identity.json"
    binding = config.get("source_identity_cache")
    if binding is None:
        return destination
    source = _bound(binding, "source identity cache")
    if source.resolve() == destination.resolve():
        return destination
    if destination.exists():
        _same(_sha(destination), binding["sha256"],
              "existing output source identity cache")
    else:
        atomic_write_bytes(destination, source.read_bytes())
        _same(_sha(destination), binding["sha256"],
              "seeded source identity cache")
    return destination


def _preflight_run_prepared(prepared, *, plan_sha256, implementation_sha256,
                           reader_identity, projection_backend):
    """Refuse a stale small completion before hashing the live wire roster.

    Runtime/source/model and exact cell checks still run after input intake;
    this early gate checks only fields already independently known at startup.
    """
    _require(prepared is not None, "cost execution requires independently bound prepared inputs")
    completion = json.loads(_bound(prepared, "prepared anchors").read_text())
    _same(completion.get("schema"), PREPARED_SCHEMA,
          "prepared v3 schema required; legacy preparation requires fresh prepare and recompute")
    _same(completion.get("status"), "complete", "prepared completion")
    for key, value in (("plan_sha256", plan_sha256),
                       ("implementation_sha256", implementation_sha256),
                       ("reader_identity", reader_identity),
                       ("projection_backend", projection_backend)):
        _same(completion.get(key), value, f"prepared {key}")
    return completion


def _restores_activation_scale_env(function):
    """Scope ``execute``'s activation-scale write to the call that makes it.

    ``execute`` sets ``PRISMAQUANT_PROD_ACT_SCALES`` from the admitted plan so
    the render path it drives reads the campaign's value.  As a process entry
    point that is right; called in-process it leaves the value behind.  Every
    admitted plan carries ``"0"`` (``_load_plan``), and that is the input
    which turns the render scorer's activation clip OFF for everything that
    runs afterwards (``production_weight_cache.py``, in
    ``_local_forward_render_score``).  Plenty of code outside ``execute``
    reads the key -- the render scorer is exactly that code, which is why the
    leak bites -- but nothing needs THIS command's value to still be set after
    ``execute`` has returned.  So restoring it on the way out leaves the
    campaign byte-identical and leaves the process as it was found.
    """
    absent = object()

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        import os

        prior = os.environ.get(ACTIVATION_SCALE_ENV, absent)
        try:
            return function(*args, **kwargs)
        finally:
            if prior is absent:
                os.environ.pop(ACTIVATION_SCALE_ENV, None)
            else:
                os.environ[ACTIVATION_SCALE_ENV] = prior

    return wrapper


@_restores_activation_scale_env
def execute(command, config, *, plan_sha256, prepared=None, resume=False, source_transition=None):
    """Execute one admitted preparation or one dependent cost action."""
    if source_transition is not None:
        from .joint_aura_source_transition import load_transition
        _require(command == "run" and resume, "source transition requires run --resume")
        source_transition = load_transition(
            source_transition, config=config, plan_sha256=plan_sha256,
            prepared=prepared, checkpoint_dir=Path(config["output_root"]) / "checkpoints",
        )
    import cProfile
    import io
    import os
    import pstats
    import socket
    import torch
    from .aura_cost import compute_aura_cost_streamed, _aura_source_sha256
    from .calibration_data import load_calibration_input
    from .cost_streaming import build_streamed_causal_lm, build_streamed_model_identity
    from .joint_aura import source_execution_identity, validate_joint_aura_entry
    from .joint_projection_backend import executing_image, prewarm_projection_backend
    from .model_profiles import detect_profile
    from .production_weight_cache import ProductionWeightCache
    from .gpu_guard import require_cuda_hot_path
    from .tessera_reader import load_declared_reader

    require_cuda_hot_path("tessera_joint_aura", "cuda")
    os.environ[ACTIVATION_SCALE_ENV] = config["execution"]["production_act_scales"]
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    execution = config["execution"]
    root = Path(config["output_root"]) / command
    root.mkdir(parents=True, exist_ok=True)
    identity_cache_path = _seed_source_identity_cache(config, root)
    result = {"schema": "prismaquant.tessera_joint_aura.execution.v1", "command": command,
              "plan_sha256": plan_sha256, "env": {"host": socket.gethostname(),
                  "started_epoch": time.time(), "torch": str(torch.__version__),
                  "cuda": torch.version.cuda, "affinity": sorted(os.sched_getaffinity(0))},
              "phases": [], "passed": False}
    result["env"]["container_content_sha256"] = executing_image()
    profile_tool = config.get("profile_tool", "cprofile")
    profiler = cProfile.Profile() if profile_tool == "cprofile" else None
    result["profile_tool"] = profile_tool
    if profiler is None:
        session_path = Path(os.environ.get("PRISMAQUANT_SAMPLER_SESSION", ""))
        _require(session_path.is_file(), "sampling must run through the checked profiler launcher")
        session_bytes = session_path.read_bytes()
        session = json.loads(session_bytes)
        _same(session.get("schema"), "prismaquant.profiled_command_start.v1", "sampler session schema")
        _same(session.get("wrapper_pid"), os.getppid(), "actual sampler child parent")
        _same(session.get("command", [])[1:4],
              ["-m", "prismaquant.tessera_joint_aura", command], "observed joint command")
        result["sampling_session"] = {"path": str(session_path),
                                      "sha256": hashlib.sha256(session_bytes).hexdigest()}
    runner = source_authentication = qualification_guard = None
    completion_path = completion = output = payload = None
    started, before_io = time.time(), _io_counters()
    if profiler is not None:
        profiler.enable()
    try:
        file_hash_workers = config.get("file_hash_workers", 1)
        _require(type(file_hash_workers) is int and 0 < file_hash_workers <= len(os.sched_getaffinity(0)),
                 "file_hash_workers exceeds PB-assigned CPU affinity")
        # Every capture-free identity gate runs first: an unqualified runtime,
        # kernel source digest, build flag or binary sha256 is refused in
        # seconds rather than after hours of measured anchor input (#553).
        projection_backend = prewarm_projection_backend(execution.get("projection_backend"), device="cuda")
        result["projection_backend"] = projection_backend.identity
        # The reader is bound first of the input owners: synthesizing an
        # adopted rung's missing render decodes its wire, and that decode must
        # come from the same bound consumer the qualification leg uses, not a
        # second one.
        reader = load_declared_reader(config.get("reader"))
        reader_identity = None if reader is None else reader.identity
        implementation = (_aura_source_sha256() if source_transition is None
                          else source_transition.measurement_source_sha256)
        if command == "run":
            _preflight_run_prepared(prepared, plan_sha256=plan_sha256,
                implementation_sha256=implementation, reader_identity=reader_identity,
                projection_backend=projection_backend.identity)
        # The command holds a CUDA reservation (``require_cuda_hot_path``
        # above), so any shard it still has to synthesize decodes on that
        # device rather than on one CPU core beside an idle GPU. The standalone
        # ``synthesize`` stage normally leaves nothing to do here.
        data = load_measured_anchor_input(config["inputs"], reader=reader,
            synthesis_device="cuda",
            **({} if file_hash_workers == 1 else {"file_hash_workers": file_hash_workers}),
            **({"verify_payloads": False} if command == "prepare" else
               {"defer_render_hashes": True, "require_existing_renders": True}))
        _require(data.unit_scope is None and data.render_mirror_root is None,
                 "joint execution requires the complete campaign roster in its own caches")
        result["file_hash_workers"] = file_hash_workers
        result["reader_identity"] = reader_identity
        # Per-run, not per-origin: the census says what is on disk, this says
        # how much of it this run had to write.
        result["renders_synthesized_now"] = data.synthesized_now
        render_census = cell_render_census(data.cells)
        # Stated whether or not this command reaches a completion: a run that
        # dies still says how many of its renders were only ever round-tripped.
        result.update(render_census)
        _same(config["model"], data.census["model"], "requested source model")
        _same(data.census["attention_implementation"], "eager", "qualified source attention")
        ids, calibration = load_calibration_input(config["calibration_input"]["path"],
            expected_sha256=config["calibration_input"]["sha256"],
            n_samples=execution["n_calib_samples"], seqlen=execution["calib_seqlen"])
        original_draw = data.payload["provenance"]["hessian"]["calibration_identity"]
        for name in ("fit_ids_sha256", "text_sha256", "nsamples", "seqlen", "seed"):
            _same(calibration["provenance"].get(name), original_draw.get(name), f"original full draw {name}")
        result["calibration_input"] = calibration
        if command == "prepare":
            if config.get("qualification_window") is not None:
                from .autoscale import require_bounded_capture_environment
                from .memory_management import CaptureMemoryGuard
                require_bounded_capture_environment(os.environ)
                qualification_guard = CaptureMemoryGuard("cuda")
                qualification_guard.check("before_joint_source_authentication")
            source_authentication = _prepare_source_owner(
                config, data, resource_check=(None if qualification_guard is None
                                              else qualification_guard.check))
        source_prefetch = _source_prefetch(config)
        runner = build_streamed_causal_lm(config["model"], device=torch.device("cuda"),
            dtype=torch.bfloat16, offload_folder=str(root / "offload"),
            profile=detect_profile(config["model"]), attn_implementation="eager",
            **({'source_derivative': execution['source_derivative']} if execution.get('source_derivative') is not None else {}),
            **({'source_authentication': source_authentication}
               if source_authentication is not None else {}),
            **source_prefetch)
        from .glm_capture_compatibility import require_capture_compatibility
        require_capture_compatibility(config.get('source_capture_compatibility'),
                                      capture=config['canonical_capture'], model=runner.model)
        result["source_prefetch"] = source_prefetch
        source = build_streamed_model_identity(runner, config["model"],
                                               identity_cache_path=identity_cache_path)
        source_execution = source_execution_identity(runner.model)
        layer_bytes = data.layer_render_bytes(runner.layer_index_for_qname)
        operator_policy = _admit_candidate_phase(command, config, data, layer_bytes)
        if operator_policy is not None:
            _require(operator_policy['prefetch_workers'] <= len(os.sched_getaffinity(0)),
                     'operator-window prefetch workers exceed PB-assigned CPU affinity')
        result.update(source_model_identity=source, source_execution=source_execution,
                      units=len(data.formats_by_qname), measured_cells=len(data.cells),
                      layer_render_bytes=layer_bytes)
        if source_transition is not None:
            result["source_transition"] = source_transition.execution_provenance
        if command == "prepare":
            _require(prepared is None, "preparation does not consume a prepared cache")
            completion_path = root / "prepared.json"
            _require(not completion_path.exists(), "prepared completion already exists; use its bound record")
            cache = prepare_cache(runner, data, capture=config["canonical_capture"],
                                  max_render_bytes=config["max_render_bytes"], reader=reader,
                                  file_load_workers=file_hash_workers,
                                  qualification_window=config.get("qualification_window"),
                                  source_authentication=source_authentication,
                                  qualification_guard=qualification_guard,
                                  qualification_journal=(root / 'qualification' if
                                                         config.get('qualification_window') is not None else None),
                                  qualification_resume=resume,
                                  qualification_identity={
                                      'plan_sha256': plan_sha256,
                                      'source_model_identity': source,
                                      'source_execution': source_execution,
                                      'implementation_sha256': implementation,
                                      'calibration_input': calibration,
                                      'reader_identity': reader_identity,
                                      'projection_backend': projection_backend.identity},
                                  **({'capture_load_policy': config['capture_load_policy']}
                                     if config.get('capture_load_policy') is not None else {}),
                                  **({'source_capture_compatibility': config['source_capture_compatibility']}
                                     if config.get('source_capture_compatibility') is not None else {}))
            source_receipt = source_authentication.authenticate_complete_source()
            cache.metadata["source_authentication"] = source_receipt
            result["source_authentication"] = source_receipt
            cache.metadata.update(plan_sha256=plan_sha256, source_model_identity=source,
                                  source_execution=source_execution, implementation_sha256=implementation,
                                  projection_backend=projection_backend.identity)
            cache.compact_for_pickle()
            cache_path = root / "production.pkl"
            atomic_write_bytes(cache_path, pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL))
            completion = {"schema": PREPARED_SCHEMA, "status": "complete", "plan_sha256": plan_sha256,
                "implementation_sha256": implementation, "source_model_identity": source,
                "reader_identity": reader_identity, "projection_backend": projection_backend.identity,
                "source_execution": source_execution, "calibration_input": calibration,
                "production_cache": {"path": str(cache_path), "sha256": _sha(cache_path)},
                "formats_by_qname": data.formats_by_qname, "measured_cells": len(data.cells),
                **render_census}
        else:
            _require(prepared is not None, "cost execution requires independently bound prepared inputs")
            completion = json.loads(_bound(prepared, "prepared anchors").read_text())
            _same(completion.get("schema"), PREPARED_SCHEMA,
                  "prepared v3 schema required; legacy preparation requires fresh prepare and recompute")
            _same(completion.get("status"), "complete", "prepared completion")
            for key, value in (("plan_sha256", plan_sha256), ("implementation_sha256", implementation),
                               ("source_model_identity", source), ("source_execution", source_execution),
                               ("calibration_input", calibration), ("measured_cells", len(data.cells)),
                               ("reader_identity", reader_identity),
                               ("render_origins", render_census["render_origins"]),
                               ("render_comparisons", render_census["render_comparisons"]),
                               ("projection_backend", projection_backend.identity)):
                _same(completion.get(key), value, f"prepared {key}")
            _same(completion["formats_by_qname"], {n: list(v) for n, v in data.formats_by_qname.items()},
                  "prepared exact candidate roster")
            cache = pickle.loads(_bound(completion["production_cache"], "qualified PWC").read_bytes())
            _require(isinstance(cache, ProductionWeightCache), "prepared cache is not ProductionWeightCache")
            _same(cache.metadata["inputs"], data.inputs, "prepared source bindings")
            _same(cache.metadata.get("reader_identity"), reader_identity, "prepared reader identity")
            _same(cache.metadata.get("projection_backend"), projection_backend.identity, "prepared backend identity")
            _same(set(cache.metadata["verified_cells"]), set(data.cells), "prepared verified cell coverage")
            _same(cache.weights, {pair: cell["render"] for pair, cell in data.cells.items()}, "prepared original render paths")
            for key in ("render_origins", "render_comparisons"):
                _same(cache.metadata.get(key), render_census[key], f"prepared cache {key}")
            for pair, cell in data.cells.items():
                _same(cache.metadata["verified_cells"][pair]["render_origin"], cell["render_origin"],
                      f"{pair}: qualified render origin changed")
                _same(cache.metadata["verified_cells"][pair]["wire_sha256"], cell["record"]["blob_sha256"],
                      f"{pair}: qualified wire changed")
            expected_renders = {pair: cache.metadata["verified_cells"][pair]["render_file_sha256"]
                                for pair in data.cells}
            # The prepared receipt bound the original serialized shard on
            # PREPARE's necessary load. COST verifies it on its own necessary
            # PWC read, before a changed tensor can reach AURA.
            cache.require_file_load_sha256(expected_renders,
                max_file_bytes=_prepare_file_read_bound(data,
                    max_render_bytes=config["max_render_bytes"]))
            _live_targets(runner, data.formats_by_qname)
            formats = list(dict.fromkeys(fmt for values in data.formats_by_qname.values() for fmt in values))
            payload = compute_aura_cost_streamed(runner, ids.to(runner.device), formats,
                n_probes=execution["n_probes"], probe_microbatch=execution["probe_microbatch"],
                seed_base=execution["seed_base"], token_scope="all", temperature=1.0,
                production_cache=cache, require_production_cache=True, joint_activation=True,
                joint_projection_backend=projection_backend,
                boundary_storage=execution.get("boundary_storage"),
                **({"operator_windows": operator_policy} if operator_policy is not None else {}),
                **({"source_transition": source_transition} if source_transition is not None else {}),
                include_routed_experts=True, include_lm_head=False, dw_dtype="float32",
                min_free_gib=config["min_free_gib"], formats_by_qname=data.formats_by_qname,
                checkpoint_dir=Path(config["output_root"]) / "checkpoints", resume=resume,
                model_identity=source, profile=runner.profile,
                checkpoint_identity_extra={"tessera_joint_anchor_plan_sha256": plan_sha256,
                    "prepared_anchor_sha256": prepared["sha256"], "calibration_input": calibration,
                    "reader_identity": reader_identity})
            _same(set(payload["costs"]), set(data.formats_by_qname), "complete joint output roster")
            for name, rows in payload["costs"].items():
                _same(set(rows), set(data.formats_by_qname[name]), f"{name}: joint output candidates")
                for row in rows.values():
                    _require(validate_joint_aura_entry(row), f"{name}: invalid measured joint cost")
            payload["provenance"]["tessera_joint_anchors"] = {
                "plan_sha256": plan_sha256, "prepared": prepared, "inputs": data.inputs,
                "calibration_input": calibration, "measured_cells": len(data.cells),
                **render_census}
            output = root / "joint-cost.pkl"
        torch.cuda.synchronize()
        result["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved()
        _require(result["peak_gpu_bytes"] <= config["max_gpu_bytes"], "observed GPU allocation exceeds declared budget")
        # A completion is published only after the source residency owner has
        # shut down successfully as well as after the allocation gate passes.
        completed_runner, runner = runner, None
        completed_runner.shutdown()
        if source_authentication is not None:
            source_authentication.close()
            source_authentication = None
        if command == "prepare":
            _json(completion_path, completion)
            result["prepared"] = {"path": str(completion_path), "sha256": _sha(completion_path)}
        else:
            atomic_write_bytes(output, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
            result["cost"] = {"path": str(output), "sha256": _sha(output)}
        result["passed"] = True
    finally:
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(str(root / "profile.pstats"))
            text = io.StringIO()
            pstats.Stats(profiler, stream=text).sort_stats("cumulative").print_stats(100)
            (root / "profile.txt").write_text(text.getvalue())
        result["env"]["finished_epoch"] = time.time()
        result["phases"].append({"phase": command, "kind": "profile", "start_epoch": started,
                                 "end_epoch": result["env"]["finished_epoch"]})
        result["io_before"], result["io_after"] = before_io, _io_counters()
        _json(root / "results.json", result)
        try:
            if runner is not None:
                runner.shutdown()
        finally:
            if source_authentication is not None:
                source_authentication.close()
    return result


SYNTHESIS_SCHEMA = "prismaquant.tessera_joint_aura.render_synthesis.v1"


def synthesize_renders(config, *, plan_sha256, units=None, device="cpu", log_every=100,
                       mirror_root=None, authorized=False, receipt=None, compare=False):
    """Write the campaign's missing decoded PWC shards, as an independent quantum.

    The joint ``prepare`` pass has always synthesized an adopted rung's
    missing shard on the way past. Doing it there put an embarrassingly
    parallel, GPU-free, one-decode-per-wire pass on the critical path of a
    reservation it does not use: measured, 125,144 shards at 2.6 cells/s on
    one core while the reserved GB10 sat at 5 W of 140 W (#549).

    This is the same function, addressable on its own: a unit range, no
    model, no capture, no GPU required, and idempotent -- a cell whose shard
    exists is skipped, the origin marker is published before the shard, and
    staging names are unique per writer. PrismaBuild owns the fan-out; rows
    carry disjoint ``sorted(names)[lo:hi]`` ranges cut from the census, so no
    two rows ever address the same cell and a retried row re-reads rather
    than re-writes.

    ``mirror_root`` publishes into a mirror of the render paths instead of
    the campaign's row caches, and ``compare`` then byte-compares each
    mirrored shard against the campaign's own. That is a measurement, not a
    build: writing into the row caches needs ``authorized``.
    """
    from .tessera_reader import load_declared_reader

    _require(bool(mirror_root) or authorized,
             "publishing into the campaign row caches requires explicit authorization")
    _require(mirror_root is not None or not compare,
             "a byte comparison needs a mirror to compare against the campaign's shards")
    census = json.loads(_bound(config["inputs"]["census"], "census").read_text())
    scope = parse_unit_scope(units, len(census["unit_shapes"]))
    reader = load_declared_reader(config.get("reader"))
    started = time.time()
    data = load_measured_anchor_input(config["inputs"], reader=reader, verify_payloads=False,
                                      synthesis_device=device, unit_scope=scope,
                                      render_mirror_root=mirror_root, log_every=log_every)
    record = {"schema": SYNTHESIS_SCHEMA, "plan_sha256": plan_sha256, "units": units,
              "unit_scope": scope, "device": device, "mirror_root": data.render_mirror_root,
              "host": socket.gethostname(), "pid": os.getpid(),
              "reader_identity": None if reader is None else reader.identity,
              "decoder_source": _decoder_identity(reader),
              "units_read": len(data.formats_by_qname), "cells": len(data.cells),
              "renders_synthesized_now": data.synthesized_now,
              "seconds": time.time() - started, **cell_render_census(data.cells)}
    if compare:
        record["comparison"] = _compare_mirrored_renders(data)
    if receipt is not None:
        _json(receipt, record)
    return record


def _decoder_identity(reader):
    """Name the decoder that produced these bytes, never assume the pinned one.

    A shard is a pure function of (wire bytes, decoder source), so a receipt
    that does not name the decoder cannot say which source it is a function
    of -- and the installed Tessera is not always the pinned one.
    """
    if reader is not None:
        return {"kind": "bound_reader", "identity": reader.identity}
    import tessera

    return {"kind": "installed", "file": str(Path(tessera.__file__).resolve()),
            "version": str(getattr(tessera, "__version__", "unknown"))}


def _compare_mirrored_renders(data):
    """Byte-compare each mirrored shard with the campaign's own, where it has one."""
    counts = {"compared": 0, "byte_identical": 0, "differs": 0, "no_campaign_render": 0}
    differing = []
    for pair, cell in sorted(data.cells.items()):
        campaign = Path(cell["campaign_render"])
        if not campaign.is_file():
            counts["no_campaign_render"] += 1
            continue
        counts["compared"] += 1
        if _sha(campaign) == _sha(Path(cell["render"])):
            counts["byte_identical"] += 1
        else:
            counts["differs"] += 1
            differing.append("@".join(pair))
    return {**counts, "differing_cells": differing[:32]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "synthesize"))
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--prepared-sha256")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--source-transition", type=Path)
    parser.add_argument("--source-transition-sha256")
    parser.add_argument("--units", help="synthesize: lo:hi over the sorted census roster. "
                        "Rows carry disjoint ranges; submit each with "
                        "--progress synthesize=SECONDS, because the stage reports its "
                        "cumulative durable count under exactly that phase name and an "
                        "undeclared phase grants no continuation")
    parser.add_argument("--device", default="cpu", help="synthesize: where the wire decode runs")
    parser.add_argument("--log-every", type=int, default=100,
                        help="synthesize: log a cumulative count and rate every N shards")
    parser.add_argument("--mirror-root", type=Path,
                        help="synthesize: publish into this mirror instead of the row caches")
    parser.add_argument("--compare", action="store_true",
                        help="synthesize: byte-compare each mirrored shard with the campaign's")
    parser.add_argument("--i-am-authorized", action="store_true",
                        help="synthesize: publish into the campaign row caches")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    if bool(args.source_transition) != bool(args.source_transition_sha256):
        parser.error("--source-transition and --source-transition-sha256 are required together")
    if bool(args.prepared) != bool(args.prepared_sha256):
        parser.error("--prepared and --prepared-sha256 are required together")
    # ``synthesize`` constructs no lease and loads no backend: it decodes wires
    # and publishes the canonical CPU BF16 shard, whose bytes are measured
    # identical across x86/aarch64 and CPU/CUDA. It is the one command that
    # does not need the projection runtime, and refusing it here would refuse
    # the stage that exists to run off the qualified box.
    config = _load_plan(args.plan, args.plan_sha256,
                        projection_runtime=args.command != "synthesize")
    if args.command == "synthesize":
        record = synthesize_renders(config, plan_sha256=args.plan_sha256, units=args.units,
                                    device=args.device, log_every=args.log_every,
                                    mirror_root=args.mirror_root, compare=args.compare,
                                    authorized=args.i_am_authorized, receipt=args.receipt)
        print(json.dumps({key: record[key] for key in (
            "units_read", "cells", "renders_synthesized_now", "seconds", "render_origins")}))
        return 0
    result = execute(args.command, config, plan_sha256=args.plan_sha256,
        prepared=None if args.prepared is None else {"path": str(args.prepared), "sha256": args.prepared_sha256},
        resume=args.resume,
        **({"source_transition": {"path": str(args.source_transition),
                                  "sha256": args.source_transition_sha256}}
           if args.source_transition is not None else {}))
    print(json.dumps({key: result[key] for key in ("command", "passed", "units",
                                                   "measured_cells", "render_origins",
                                                   "render_comparisons")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
