"""Does a read manifest declare every source read its run makes? (PQ #1095)

A strictly staged run reads its source checkpoint through
``residency_shard_reader``, which serves a tensor only from one declared,
landed staged range that holds it whole. A tensor the manifest does not
declare is refused (``readset-not-staged``) at the moment it is read: for
the first Stage B quantum on lina, 54 s in, on the resident head, after the
head phase had already staged. This module finds every such read before
submission, without a GPU, in one pass.

The reads come from ``layer_streaming.streaming_source_plan``, the streaming
loader's own enumeration:

* the resident head, which ``_materialize`` loads once before any layer and
  which the ``head`` phase has to declare;
* each layer the run installs, in its install order, which that layer's
  source phase has to declare;
* each layer the run prefetches. A prefetch outside the install order is a
  read no phase stages; it is reported as its own gap.

The install order and the prefetch schedule are the runtime's own:
``joint_layer_quanta.quantum_source_layer_order`` and
``source_read_plan.chain_prefetch_window`` for a Stage B quantum. The
Stage A chain in ``joint_cost_stage_a`` schedules with the same two
windows (PQ #1100); :func:`stage_a_prefetch_targets` states that schedule
for its check.

A gap is a dict with ``kind``, ``phase``, ``layer`` and, for a span,
``path``, ``start``, ``end``, ``bytes`` and ``tensor``. Kinds:

``resident-head``
    A head tensor the ``head`` phase does not hold whole.
``layer-source``
    A tensor of an installed layer its source phase does not hold whole.
``prefetch-outside-walk``
    A layer the run prefetches but never installs; no phase stages it.
``head-selection-unsealed``
    The bound record carries no ``executable_readset.head_source``, so the
    quantum's loader has nothing to compare its head selection with.
``head-selection-differs``
    The sealed head tensors or layers prefix differ from the loader's.
``missing-phase``
    The manifest has no phase of the name the run stages from.
``unreadable``
    An input the check needs could not be read or did not verify.

This module needs no GPU. It imports ``layer_streaming`` (and torch) only
through :func:`load_source_plan`.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path

from .joint_layer_quanta import (
    executable_own_source_phase_name,
    executable_source_phase_name,
    quantum_source_layer_order,
)
from .source_read_plan import (
    chain_opening_window,
    chain_prefetch_window,
    roster_layers_prefix,
    uncovered_spans,
)

GAPS_SCHEMA = "prismaquant.readset_coverage.gaps.v1"
HEAD_PHASE = "head"

#: ``(model dir, layers prefix, layers) -> streaming_source_plan(...)``.
SourcePlanner = Callable[[str, str, Sequence[int]], Mapping]


def load_source_plan(model_dir: str, layers_prefix: str,
                     layers: Sequence[int]) -> dict:
    """The streaming loader's reads for ``layers``, read off the pool."""
    from .layer_streaming import streaming_source_plan

    return streaming_source_plan(model_dir, layers_prefix=layers_prefix,
                                 layers=sorted(set(layers)))


def phase_entries(manifest: Mapping) -> dict[str, list[dict]]:
    """``{phase name: [entry, ...]}`` of a v2 manifest's read plan."""
    entries = manifest["entries"]
    return {phase["name"]: [entries[index] for index in phase["entry_indices"]]
            for phase in manifest["read_plan"]["phases"]}


def _span_gaps(entries, spans, *, kind, phase, layer, names) -> list[dict]:
    return [{"kind": kind, "phase": phase, "layer": layer, "path": path,
             "start": start, "end": end, "bytes": end - start,
             "tensor": names.get((path, start, end))}
            for path, start, end in uncovered_spans(entries, spans)]


def source_read_gaps(manifest: Mapping, source_plan: Mapping, *,
                     order: Sequence[int],
                     prefetch_targets: Iterable[int],
                     phase_of: Callable[[int], str]) -> list[dict]:
    """Every source read of one walk that ``manifest`` does not declare.

    ``order`` is the walk's install order and ``prefetch_targets`` every
    layer it prefetches. ``phase_of(layer)`` names the phase that stages an
    installed layer's source. The resident head has to sit in ``head``.
    """
    phases = phase_entries(manifest)
    names = source_plan.get("span_tensors") or {}
    gaps: list[dict] = []
    if HEAD_PHASE not in phases:
        gaps.append({"kind": "missing-phase", "phase": HEAD_PHASE, "layer": None})
    gaps += _span_gaps(phases.get(HEAD_PHASE, []), source_plan["head_spans"],
                       kind="resident-head", phase=HEAD_PHASE, layer=None,
                       names=names)
    for layer in order:
        phase = phase_of(layer)
        if phase not in phases:
            gaps.append({"kind": "missing-phase", "phase": phase, "layer": layer})
        gaps += _span_gaps(phases.get(phase, []), source_plan["layer_spans"][layer],
                           kind="layer-source", phase=phase, layer=layer,
                           names=names)
    walk = set(order)
    for layer in sorted(set(prefetch_targets) - walk):
        spans = source_plan["layer_spans"].get(layer, [])
        gaps.append({"kind": "prefetch-outside-walk", "phase": None,
                     "layer": layer, "spans": len(spans),
                     "bytes": sum(end - start for _path, start, end in spans)})
    return gaps


def quantum_prefetch_targets(order: Sequence[int], lookahead: int) -> set[int]:
    """Every layer a Stage B quantum prefetches: ``_install_with_settlement``.

    It prefetches each layer of its walk before installing it and
    ``chain_prefetch_window`` after.
    """
    targets = set(order)
    for position in range(len(order)):
        targets.update(chain_prefetch_window(order, position, lookahead))
    return targets


def stage_a_prefetch_targets(order: Sequence[int], lookahead: int) -> set[int]:
    """Every layer the Stage A chain walk prefetches.

    ``joint_cost_stage_a`` prefetches ``chain_opening_window`` before its
    first install, then ``chain_prefetch_window`` after each install: the
    next ``lookahead`` layers of its own order, never one past its last layer
    (PQ #1100). Until then it read ``range(max(0, L - lookahead), L)``, so a
    walk that ended at layer 40 with lookahead 2 also read layers 38 and 39,
    which its manifest does not declare.
    """
    targets = set(chain_opening_window(order, lookahead))
    for position in range(len(order)):
        targets.update(chain_prefetch_window(order, position, lookahead))
    return targets


def _quantum_head_gaps(record: Mapping, source_plan: Mapping) -> list[dict]:
    sealed = (record.get("executable_readset") or {}).get("head_source")
    if sealed is None:
        return [{"kind": "head-selection-unsealed", "phase": HEAD_PHASE,
                 "layer": None}]
    if (sealed.get("tensors") != source_plan["head_tensors"]
            or sealed.get("layers_prefix") != source_plan["layers_prefix"]):
        return [{"kind": "head-selection-differs", "phase": HEAD_PHASE,
                 "layer": None, "sealed_tensors": len(sealed.get("tensors") or []),
                 "loader_tensors": len(source_plan["head_tensors"])}]
    return []


def quantum_record_gaps(record: Mapping, manifest: Mapping,
                        source_plan: Mapping, *, lookahead: int,
                        order: Sequence[int] | None = None) -> list[dict]:
    """The gaps of one bound Stage B quantum and its executable manifest.

    ``order`` defaults to the quantum's own walk, its chain then its own
    layer. A band-serial consumer, which takes its cotangent from a handoff
    instead of rebuilding the chain, installs only its own layer.
    """
    layer = record["layer"]
    if order is None:
        order = quantum_source_layer_order(
            record["adjoint"]["chain_layers"], layer)

    def phase_of(installed: int) -> str:
        return (executable_own_source_phase_name(installed) if installed == layer
                else executable_source_phase_name(installed))

    gaps = _quantum_head_gaps(record, source_plan)
    gaps += source_read_gaps(manifest, source_plan, order=order,
                             prefetch_targets=quantum_prefetch_targets(order, lookahead),
                             phase_of=phase_of)
    for gap in gaps:
        gap["quantum_id"] = record.get("quantum_id")
    return gaps


def _read_verified(path: str, sha256: str | None, *, where: str,
                   wire: bytes | None = None) -> bytes:
    raw = Path(path).read_bytes() if wire is None else bytes(wire)
    if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"{where} at {path} does not hash to its sealed digest")
    return raw


def _read_declared(path: str, sealed_sha256: str | None, *, where: str) -> bytes:
    """A campaign input read by its on-disk bytes; the record's digest is a seal.

    The record names the plan and the prepared completion it was cut from. A
    campaign that re-declares either one (a resource plan re-derived from a
    measurement) leaves the old digest in the record, so the comparison is a
    run seal (PQ #1147): dev mode prints both digests and the check reads the
    bytes on disk, as the dispatcher's own row does. Certified mode raises the
    same ``ValueError`` as before, so the gap it becomes is unchanged.
    """
    raw = Path(path).read_bytes()
    if sealed_sha256 is not None:
        from .dev_mode import seal_check

        seal_check(f"campaign {where}", sealed_sha256, hashlib.sha256(raw).hexdigest(),
                   where=f"readset coverage at {path}",
                   refusal=lambda: ValueError(
                       f"{where} at {path} does not hash to its sealed digest"))
    return raw


def load_manifest(path: str, sha256: str | None = None, *,
                  wire: bytes | None = None) -> dict:
    """A (gzipped) manifest, checked against ``sha256`` when given.

    ``wire`` is the manifest's bytes when the caller derived them and did not
    publish them (a dispatch dry run's band-serial readset, PQ #1200); they
    are checked the same way and ``path`` only names them.
    """
    raw = _read_verified(path, sha256, where="read manifest", wire=wire)
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def quantum_rows_gaps(rows: Sequence[Mapping], *,
                      planner: SourcePlanner = load_source_plan) -> list[dict]:
    """Every gap of a set of Stage B quantum rows, in one pass.

    Each row is ``{"record": bound record, "manifest_path", "manifest_sha256",
    "order": install order or None}``, plus ``"manifest_wire"`` for a
    manifest not yet published (:func:`load_manifest`). The rows share one
    campaign, so the source plan is read once for every layer any of them
    installs.
    """
    if not rows:
        return []
    gaps: list[dict] = []
    campaign = rows[0]["record"]["campaign"]
    try:
        plan = json.loads(_read_declared(
            campaign["plan_path"], campaign["plan_sha256"], where="plan"))
        prepared = json.loads(_read_declared(
            campaign["prepared_path"], campaign["prepared_sha256"],
            where="prepared completion"))
        layers_prefix = roster_layers_prefix(prepared.get("formats_by_qname") or {})
        lookahead = plan["source_prefetch"]["prefetch_lookahead"]
        if type(lookahead) is not int or lookahead < 1:
            raise ValueError(f"plan source_prefetch lookahead {lookahead!r}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return [{"kind": "unreadable", "phase": None, "layer": None,
                 "detail": f"campaign inputs: {exc}"}]
    loaded = []
    wanted: set[int] = set()
    for row in rows:
        record = row["record"]
        try:
            manifest = load_manifest(row["manifest_path"], row["manifest_sha256"],
                                     wire=row.get("manifest_wire"))
        except (OSError, ValueError) as exc:
            gaps.append({"kind": "unreadable", "phase": None, "layer": None,
                         "quantum_id": record.get("quantum_id"),
                         "detail": f"read manifest: {exc}"})
            continue
        order = row.get("order")
        if order is None:
            order = quantum_source_layer_order(
                record["adjoint"]["chain_layers"], record["layer"])
        loaded.append((record, manifest, tuple(order)))
        wanted.update(quantum_prefetch_targets(order, lookahead))
    if not loaded:
        return gaps
    try:
        source_plan = planner(plan["model"], layers_prefix, sorted(wanted))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return gaps + [{"kind": "unreadable", "phase": None, "layer": None,
                        "detail": f"source plan of {plan.get('model')!r}: {exc}"}]
    for record, manifest, order in loaded:
        gaps += quantum_record_gaps(record, manifest, source_plan,
                                    lookahead=lookahead, order=order)
    return gaps


def gaps_report(gaps: Sequence[Mapping], **context) -> dict:
    """The one JSON document a check prints."""
    by_kind: dict[str, int] = {}
    for gap in gaps:
        by_kind[gap["kind"]] = by_kind.get(gap["kind"], 0) + 1
    return {"schema": GAPS_SCHEMA, **context, "covered": not gaps,
            "gap_count": len(gaps), "by_kind": by_kind, "gaps": list(gaps)}
