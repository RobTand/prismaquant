"""Stage-1 research entry point: same-pass disjoint FIT/HELDOUT split-H capture.

One runnable experiment entry with four modes over ONE capture root:

* ``prep``     -- reads the saved actual calibration IDs, stamps the fixed
  384/128 split (role hashes + provenance, ``split-manifest.json``), and
  records the existing v2 chain prep (``prismaquant.capture_layer_chain``)
  through an explicit ``record_capture_source`` owner. No automatic
  admission call is made or patched.
* ``quantum``  -- builds the existing streamed runner through
  :func:`prismaquant.capture_layer_chain.authenticate_quantum_source` and
  drives ``_collect_activations`` with ``want_hessian=False, max_rows=0,
  shared_packed_inputs=True, row_consumer=moments.consume`` so both split
  roles accumulate in the SAME forward pass. Every source layer in the
  range forwards, even a layer with no selected unit. Both roles are
  persisted once per layer and verified (file bytes/sha, tensor geometry,
  counts, fit+heldout == census) before ``ChainQuantum.complete`` carries
  the role receipts in its verified unit map.
* ``join``     -- research metadata verification only: the existing ranges /
  owner / fragment checks, the witness merge against the census contract,
  the recorded source digest union, and the role-manifest union. It does
  NOT call the automatic ``chain.join``: no ordinary capture manifest is
  published and no provider qualification is claimed.
* ``preflight``-- small real-metadata reads with no GPU, then the actual
  tiny one-layer GLM CPU control end-to-end (census -> prep -> quantum ->
  join) through the existing toy helpers, demonstrating split roles, full
  counts and boundary forward.

Authorization: CEO decision dec-1006-042211-25de option 1 -- explicitly
owner-recorded (``record_capture_source``) Stage-1 RESEARCH capture only.
``require_automatic_capture_source_recording`` is neither called nor
monkeypatched, and no immutable-provider qualification is claimed.

Memory guard: startup and live ``MemAvailable`` must stay >= 2 GiB; the
guard aborts cleanly (moments closed, runner shut down, source owner
closed, GPU/CPU buffers released) before propagating.

Both the ``prismaquant`` package and split-moment helper are loaded from this
checkout, so PrismaBuild carries the complete runnable research source.
"""
from __future__ import annotations

import argparse
import hashlib
import json

import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismaquant import capture_layer_chain as chain  # noqa: E402
from prismaquant import tessera_campaign as campaign  # noqa: E402
from prismaquant import tessera_calibration_cache as store  # noqa: E402
from prismaquant import tessera_hessian as th  # noqa: E402
from prismaquant.cost_streaming import (  # noqa: E402
    build_streamed_causal_lm,
    check_boundary_storage,
)
from prismaquant.routed_experts import (  # noqa: E402
    declared_shared_capture_groups,
    refresh_packed_expert_projections,
)
from experiments.indomain_split_stats import HELDOUT, FIT, DisjointRowMoments  # noqa: E402


SPLIT_MANIFEST_SCHEMA = "prismaquant.research_split_manifest.v1"
LAYER_MANIFEST_SCHEMA = "prismaquant.research_split_layer_manifest.v1"
RESEARCH_JOIN_SCHEMA = "prismaquant.research_split_join.v1"
ROLE_RECEIPT_SCHEMA = "prismaquant.research_split_role_receipt.v1"
CENSUS_VIEW_SCHEMA = "prismaquant.research_census_source_view.v1"
DEFAULT_MEMFLOOR_BYTES = 2 * (1 << 30)


class ResearchRefused(RuntimeError):
    """The research split capture refuses by name."""


# -- memory guard ----------------------------------------------------------------

def _mem_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ResearchRefused("/proc/meminfo has no MemAvailable row")


def make_memory_guard(floor_bytes: int):
    def guard(label: str) -> None:
        available = _mem_available_bytes()
        if available < floor_bytes:
            raise ResearchRefused(
                f"{label}: MemAvailable is {available} bytes, below the "
                f"{floor_bytes}-byte research floor; clean abort")
    return guard


# -- hashing and layout ----------------------------------------------------------

def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value) -> str:
    from prismaquant.cost_stage_checkpoint import canonical_json_sha256
    return canonical_json_sha256(value, where="research stamp")


def research_dir(capture_root) -> Path:
    return Path(capture_root)


def split_manifest_path(capture_root) -> Path:
    return research_dir(capture_root) / "split-manifest.json"


def join_document_path(capture_root) -> Path:
    return research_dir(capture_root) / "join.json"


def census_view_path(capture_root) -> Path:
    return research_dir(capture_root) / "census-source-view.json"


def split_stamp(manifest: dict) -> str:
    """Canonical split body identity, excluding its own digest field."""
    return _canonical_sha256({key: value for key, value in manifest.items()
                              if key != "split_sha256"})


def campaign_args(args, census) -> SimpleNamespace:
    """The minimal argument namespace the existing campaign helpers read."""
    model = args.model if args.model is not None else str(census["model"])
    receipt = census.get("calibration_input")
    return SimpleNamespace(
        model=model, nsamples=int(census["nsamples"]), seqlen=int(census["seqlen"]),
        seed=int(census["seed"]), layer_stride=int(census["layer_stride"]),
        calibration_census=args.calibration_census,
        calibration_input_receipt=receipt,
        max_act_rows=int(args.max_act_rows),
        attention_implementation=args.attention_implementation,
        units=args.units, research_exact_member=None,
        allow_pinned=None, pinned_roster_only=False,
    )


def effective_census_path(args, capture_root) -> Path:
    """The census the whole chain reads: the original, or the snapshot view.

    A source snapshot root supplies a byte-identical model view under another
    path; the derived census view points the census's ``model`` at it while
    preserving the original source/content stamps verbatim (no new seal and
    no immutable-provider claim). Prep, quanta and join must all use the SAME
    view, because the traversal identity binds the census bytes it read.
    """
    path = Path(args.calibration_census)
    if args.source_snapshot_root is None:
        return path
    view = census_view_path(capture_root)
    if view.is_file():
        return view
    original = json.loads(path.read_text())
    document = dict(original)
    document["model"] = str(Path(args.source_snapshot_root).resolve())
    document["research_source_view"] = {
        "schema": CENSUS_VIEW_SCHEMA,
        "original_model": str(original["model"]),
        "original_census_path": str(path),
        "original_census_sha256": _sha256_file(path),
        "snapshot_root": document["model"],
    }
    view.parent.mkdir(parents=True, exist_ok=True)
    from prismaquant.cost_stage_checkpoint import atomic_write_bytes
    from prismaquant.digests import indent2_json_file_bytes
    atomic_write_bytes(view, indent2_json_file_bytes(document))
    return view


def load_census(args, capture_root):
    """Load and draw-check the census through the existing loader."""
    path = effective_census_path(args, capture_root)
    namespace = campaign_args(args, json.loads(path.read_text()))
    census = campaign.load_calibration_census(path, args=namespace)
    return path, census, namespace


def load_draw(args, census):
    """The saved actual calibration IDs, byte-checked against the census.

    Positive explicit comparisons: the tokens file's sha256 against the
    census receipt's artifact digest, the corpus text sha against the
    census's ``text_sha256``, the tensor's dtype/geometry against the
    receipt, and the sample geometry against the census draw.
    """
    receipt = census["calibration_input"]
    tokens_path = Path(args.calibration_tokens)
    corpus_path = Path(args.corpus_text)
    actual = _sha256_file(tokens_path)
    if actual != receipt["artifact_sha256"]:
        raise ResearchRefused(
            f"{tokens_path}: sha256 {actual} is not the census draw's "
            f"{receipt['artifact_sha256']}")
    corpus_bytes = corpus_path.read_bytes()
    if hashlib.sha256(corpus_bytes).hexdigest() != str(census["text_sha256"]):
        raise ResearchRefused(
            f"{corpus_path}: text sha is not the census draw's "
            f"{census['text_sha256']}")
    corpus_text = corpus_bytes.decode("utf-8")
    from safetensors.torch import load_file
    tensors = load_file(str(tokens_path))
    if len(tensors) != 1:
        raise ResearchRefused(f"{tokens_path}: expected exactly one id tensor")
    (name, ids), = tensors.items()
    if ids.ndim != 2:
        raise ResearchRefused(f"{tokens_path}: id tensor {name} is not 2-D")
    total, seqlen = (int(value) for value in ids.shape)
    nsamples = int(census["nsamples"])
    if total != nsamples or seqlen != int(census["seqlen"]):
        raise ResearchRefused(
            f"{tokens_path}: draw is [{total}, {seqlen}], the census draw is "
            f"[{nsamples}, {census['seqlen']}]")
    if str(ids.dtype) != str(receipt["dtype"]):
        raise ResearchRefused(
            f"{tokens_path}: dtype {ids.dtype} is not the receipt's "
            f"{receipt['dtype']}")
    return ids, [ids[index:index + 1].contiguous() for index in range(total)], corpus_text


def selected_units(args, census, namespace) -> list:
    """The parent's selected capture targets, through the existing authority."""
    selection = campaign.load_unit_selection(args.units)
    names = campaign.selected_capture_unit_names(
        selection, args=namespace, resolved=census["anchor_groups"])
    missing = sorted(set(names) - set(census["counts"]))
    if missing:
        raise ResearchRefused(f"the selection prices units outside the census: {missing[:8]}")
    return names


def require_split_geometry(args, tokens, census) -> None:
    if args.total_samples != len(tokens):
        raise ResearchRefused(
            f"--total-samples {args.total_samples} but the draw holds {len(tokens)} samples")
    if args.fit_stop <= 0 or args.total_samples <= args.fit_stop:
        raise ResearchRefused(
            f"the fixed split needs 0 < fit_stop < total_samples, got "
            f"{args.fit_stop}/{args.total_samples}")
    if int(tokens[0].shape[1]) != int(census["seqlen"]):
        raise ResearchRefused("a calibration sample is not the census's seqlen")


def _write_atomic_json(path: Path, document: dict) -> None:
    from prismaquant.cost_stage_checkpoint import atomic_write_bytes
    from prismaquant.digests import indent2_json_file_bytes
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(path, indent2_json_file_bytes(document))


# -- prep ------------------------------------------------------------------------

def build_split_manifest(args, census, ids) -> dict:
    """The fixed disjoint split's hashes and provenance (parent contract)."""
    seqlen = int(ids.shape[1])
    fit_ids, held_ids = ids[:args.fit_stop], ids[args.fit_stop:]
    fit_digest = th.token_ids_sha256([row for row in fit_ids])
    held_digest = th.token_ids_sha256([row for row in held_ids])
    provenance = (census.get("calibration_input") or {}).get("provenance") or {}
    draw_seed = int(provenance.get("seed", census["seed"]))
    return {
        "schema": SPLIT_MANIFEST_SCHEMA,
        "seed": draw_seed,
        "sample_count": int(ids.shape[0]),
        "tokens_per_sample": seqlen,
        "same_forward_pass": True,
        "fit_stop": int(args.fit_stop),
        "source_draw": {
            "path": str(Path(args.calibration_tokens)),
            "sha256": _sha256_file(args.calibration_tokens),
            "provenance": (census.get("calibration_input") or {}).get("provenance"),
            "shape": [int(value) for value in ids.shape],
            "dtype": str(ids.dtype),
        },
        "roles": {
            FIT: {
                "sample_range": [0, int(args.fit_stop)],
                "token_ids_sha256": fit_digest,
                "token_count": int(args.fit_stop) * seqlen,
                "hessian_role": "fit",
                "provenance": {
                    "model": str(census["model"]), "seed": draw_seed,
                    "nsamples": int(args.fit_stop), "seqlen": seqlen,
                    "fit_tokens": int(args.fit_stop) * seqlen,
                    "text_sha256": str(census["text_sha256"]),
                    "fit_ids_sha256": fit_digest,
                    "hessian_role": "fit",
                    "source": "fixed disjoint stage-1 research split of the census draw",
                },
            },
            HELDOUT: {
                "sample_range": [int(args.fit_stop), int(ids.shape[0])],
                "token_ids_sha256": held_digest,
                "token_count": int(ids.shape[0] - args.fit_stop) * seqlen,
                "hessian_role": "held-out",
                "provenance": {
                    "model": str(census["model"]), "seed": draw_seed,
                    "nsamples": int(ids.shape[0] - args.fit_stop), "seqlen": seqlen,
                    "heldout_tokens": int(ids.shape[0] - args.fit_stop) * seqlen,
                    "text_sha256": str(census["text_sha256"]),
                    "fit_ids_sha256": held_digest,
                    "fit_tokens": int(ids.shape[0] - args.fit_stop) * seqlen,
                    "hessian_role": "held-out",
                    "source": "fixed disjoint stage-1 research split of the census draw",
                },
            },
        },
    }


def mode_prep(args, guard) -> dict:
    """The prep row: fixed-split stamps plus the existing v2 chain prep."""
    guard("research prep startup")
    capture_root = Path(args.capture_root).resolve()
    census_path, census, namespace = load_census(args, capture_root)
    ids, tokens, corpus_text = load_draw(args, census)
    require_split_geometry(args, tokens, census)
    unit_names = selected_units(args, census, namespace)
    manifest = build_split_manifest(args, census, ids)
    manifest["split_sha256"] = split_stamp(manifest)
    manifest_path = split_manifest_path(capture_root)
    _write_atomic_json(manifest_path, manifest)
    print(json.dumps({"research_split_manifest": {
        "path": str(manifest_path), "split_sha256": manifest["split_sha256"],
        "fit_samples": int(args.fit_stop),
        "heldout_samples": int(len(tokens) - args.fit_stop),
        "units": len(unit_names)}}), flush=True)
    if args.capture_chain_ranges is None or args.boundary_storage is None:
        raise ResearchRefused("research prep needs --capture-chain-ranges and --boundary-storage")
    ranges = chain.parse_layer_ranges(args.capture_chain_ranges)
    storage_text = args.boundary_storage if args.boundary_storage.lstrip().startswith("{") \
        else Path(args.boundary_storage).read_text()
    boundary_storage = json.loads(storage_text)
    check_boundary_storage(boundary_storage)
    # Explicit owner, exactly like the campaign's prep bookend: no automatic
    # admission call is made, patched or bypassed.
    with store.record_capture_source(census_path, model=namespace.model) as source:
        record = chain.prepare(
            capture_root, census_path=census_path, ranges=ranges,
            n_batches=len(tokens), boundary_storage=boundary_storage,
            identity=lambda: campaign._streamed_capture_identity(
                namespace, census, tokens, corpus_text,
                attention_implementation=args.attention_implementation,
                source_authentication=source, unit_names=unit_names))
    print(json.dumps({"research_prep": record["path"],
                      "sha256": record["sha256"]}), flush=True)
    return record


# -- quantum ---------------------------------------------------------------------

class SampleIndexer:
    """The monotonic global single-sample index over the quantum's forwards.

    The calibration draw is exactly one sample per forward; any mixed or
    reordered batch is refused, never guessed into a role. The global index
    increases by one per forward across the whole range; each layer's first
    forward continues it at a sample-aligned offset, so within a layer the
    sample coordinate is ``global_index % total_samples`` and both readings
    agree.
    """

    def __init__(self, moments, *, total_samples, seqlen, guard):
        self.moments = moments
        self.total_samples = int(total_samples)
        self.seqlen = int(seqlen)
        self.guard = guard
        self.global_index = 0
        self.layer = None
        self.in_layer = 0

    def wrapper(self, layer, forward_batch):
        def forward(batch):
            if batch.ndim != 2 or int(batch.shape[0]) != 1:
                raise ResearchRefused(
                    f"layer {layer}: a research forward takes exactly one "
                    f"[1, {self.seqlen}] sample, got {tuple(batch.shape)}")
            if int(batch.shape[1]) != self.seqlen:
                raise ResearchRefused(
                    f"layer {layer}: a calibration sample is {self.seqlen} "
                    f"tokens, got {tuple(batch.shape)}")
            if self.layer != layer:
                if self.layer is not None and self.in_layer != self.total_samples:
                    raise ResearchRefused(
                        f"layer {self.layer} forwarded {self.in_layer} of "
                        f"{self.total_samples} samples")
                self.layer, self.in_layer = layer, 0
            sample = self.in_layer
            if sample != self.global_index % self.total_samples:
                raise ResearchRefused(
                    f"layer {layer}: non-monotonic sample index {sample} for "
                    f"global forward {self.global_index}")
            self.guard(f"research forward layer {layer} sample {sample}")
            self.moments.set_sample(sample)
            value = forward_batch(batch)
            self.in_layer += 1
            self.global_index += 1
            return value
        return forward


def _write_and_verify_role_file(directory: Path, qname: str, role: str, record: dict,
                                census_count: int, unit_shape) -> dict:
    """Write one role file and verify its own bytes, sha and tensor geometry."""
    path = directory / f"{qname}.{role}.pt"
    payload = {"name": qname, "role": role, "hessian": record["hessian"],
               "inputs": record["inputs"], "count": int(record["count"]),
               "max_abs": float(record["max_abs"]),
               "prefix_sample_ids": record["prefix_sample_ids"]}
    torch.save(payload, path)
    digest = _sha256_file(path)
    loaded = torch.load(path, weights_only=True)
    if loaded["name"] != qname or loaded["role"] != role:
        raise ResearchRefused(f"{path}: role file does not name its own unit/role")
    if loaded["hessian"].dtype != torch.float32 or loaded["hessian"].device.type != "cpu":
        raise ResearchRefused(f"{path}: the role Hessian is not a CPU float32 tensor")
    dimension = int(unit_shape[1])
    if list(loaded["hessian"].shape) != [dimension, dimension]:
        raise ResearchRefused(
            f"{path}: Hessian geometry {list(loaded['hessian'].shape)} is not "
            f"the census unit geometry [{dimension}, {dimension}]")
    if loaded["inputs"] is not None:
        if loaded["inputs"].dtype != torch.float32 or loaded["inputs"].device.type != "cpu":
            raise ResearchRefused(f"{path}: the role inputs are not CPU float32")
        if int(loaded["inputs"].shape[1]) != dimension:
            raise ResearchRefused(f"{path}: role inputs disagree with the unit geometry")
    if loaded["prefix_sample_ids"] is not None and loaded["prefix_sample_ids"].dtype != torch.int64:
        raise ResearchRefused(f"{path}: role prefix sample ids are not int64")
    if int(loaded["count"]) <= 0:
        raise ResearchRefused(f"{path}: role count is not positive")
    return {
        "schema": ROLE_RECEIPT_SCHEMA, "name": qname, "role": role,
        "file": path.name,
        "bytes": path.stat().st_size, "sha256": digest, "count": int(loaded["count"]),
        "hessian_shape": list(loaded["hessian"].shape),
        "inputs_shape": None if loaded["inputs"] is None else list(loaded["inputs"].shape),
        "prefix_sample_ids_shape": None if loaded["prefix_sample_ids"] is None
        else list(loaded["prefix_sample_ids"].shape),
        "max_abs": float(loaded["max_abs"]),
        "census_count": int(census_count),
    }


def _role_file_records(receipt: dict) -> dict:
    return {
        "file": receipt["file"], "bytes": int(receipt["bytes"]),
        "sha256": receipt["sha256"], "count": int(receipt["count"]),
        "hessian_shape": [int(v) for v in receipt["hessian_shape"]],
        "inputs_shape": None if receipt["inputs_shape"] is None
        else [int(v) for v in receipt["inputs_shape"]],
        "prefix_sample_ids_shape": None if receipt["prefix_sample_ids_shape"] is None
        else [int(v) for v in receipt["prefix_sample_ids_shape"]],
    }


def persist_split_records(capture_root, census, census_path, records, seen,
                          split_sha256, guard) -> dict:
    """Persist both roles once per layer; verify every file and every count."""
    counts = census["counts"]
    maxima = census["max_abs"]
    maxima_observed = {}
    by_layer = {}
    for role in (FIT, HELDOUT):
        if role not in records:
            raise ResearchRefused(f"the split moments returned no {role} role")
    for role_records in (records[FIT], records[HELDOUT]):
        for qname in role_records:
            match = chain.DOTTED_LAYER_QNAME.search(qname)
            if match is None:
                raise ResearchRefused(f"{qname}: a research unit names no decoder layer")
            by_layer.setdefault(int(match.group(1)), set()).add(qname)
    verified = {}
    layers_dir = research_dir(capture_root) / "layers"
    for layer in sorted(by_layer):
        directory = layers_dir / f"L{layer:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        layer_manifest = {
            "schema": LAYER_MANIFEST_SCHEMA, "layer": layer,
            "split_sha256": split_sha256,
            "source_root": str(census["model"]),
            "read_contract_stamp": {
                "model_load_contract_sha256":
                    _canonical_sha256(census["model_load_contract"]),
                "census_sha256": _sha256_file(census_path)},
            "full_counts": {}, "units": {}}
        for qname in sorted(by_layer[layer]):
            census_count = int(counts[qname])
            layer_manifest["full_counts"][qname] = census_count
            observed = int(seen.get(qname, 0))
            if observed != census_count:
                raise ResearchRefused(
                    f"{qname}: observed {observed} rows, the census counts {census_count}")
            split_total = 0
            units_entry = {}
            unit_maximum = float("-inf")
            role_records = {FIT: records[FIT].get(qname),
                            HELDOUT: records[HELDOUT].get(qname)}
            for role in (FIT, HELDOUT):
                record = role_records.get(role)
                if record is None:
                    raise ResearchRefused(f"{qname}: the split left no {role} record")
                receipt = _write_and_verify_role_file(
                    directory, qname, role, record, census_count,
                    census["unit_shapes"][qname])
                split_total += receipt["count"]
                units_entry[role] = _role_file_records(receipt)
                units_entry[role]["file"] = str((directory / receipt["file"]).relative_to(research_dir(capture_root)))
                unit_maximum = max(unit_maximum, float(record["max_abs"]))
            if split_total != census_count:
                raise ResearchRefused(
                    f"{qname}: fit + heldout rows {split_total} != census {census_count}; "
                    "the split dropped or duplicated rows")
            maxima_observed[qname] = unit_maximum
            verified[qname] = {
                "schema": ROLE_RECEIPT_SCHEMA, "name": qname, "layer": layer,
                FIT: units_entry[FIT], HELDOUT: units_entry[HELDOUT],
                "census_count": census_count, "observed_rows": observed,
                "max_abs": unit_maximum}
            layer_manifest["units"][qname] = units_entry
        guard(f"research layer manifest {layer}")
        _write_atomic_json(directory / "manifest.json", layer_manifest)
    campaign.census_max_abs(census, maxima_observed)
    return verified


def mode_quantum(args, guard) -> dict:
    """One layer-range quantum: same-pass split moments over the chain owner."""
    guard("research quantum startup")
    capture_root = Path(args.capture_root).resolve()
    census_path, census, namespace = load_census(args, capture_root)
    ids, tokens, corpus_text = load_draw(args, census)
    require_split_geometry(args, tokens, census)
    unit_names = selected_units(args, census, namespace)
    start, stop = chain.parse_layer_range(args.capture_layer_range)
    contract = census.get("model_load_contract") or {}
    num_layers = int(contract.get("num_layers", 0))
    if num_layers <= 0:
        raise ResearchRefused("the census names no source layer depth")

    moments = DisjointRowMoments(
        fit_stop=int(args.fit_stop), total_samples=int(args.total_samples),
        tokens_per_sample=int(census["seqlen"]),
        max_prefix_rows=int(args.max_prefix_rows), resource_check=guard)
    with ExitStack() as scope:
        scope.callback(moments.close)
        source = scope.enter_context(chain.authenticate_quantum_source(
            capture_root, census_path=census_path, model=namespace.model,
            resource_check=guard))
        quantum = chain.ChainQuantum(capture_root, (start, stop),
                                     num_layers=num_layers, source_authentication=source)
        identity = campaign._streamed_capture_identity(
            namespace, census, tokens, corpus_text,
            attention_implementation=args.attention_implementation,
            source_authentication=source, unit_names=unit_names)
        identity = quantum.require_identity(identity, n_batches=len(tokens))
        split_sha256 = split_stamp(
            json.loads(split_manifest_path(capture_root).read_text()))

        from prismaquant.model_profiles import detect_profile
        profile = detect_profile(namespace.model)
        cache_dir = Path(args.cache_dir) if args.cache_dir else capture_root / "quantum-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        runner = build_streamed_causal_lm(
            namespace.model, device=torch.device(args.device), dtype=torch.bfloat16,
            profile=profile, offload_folder=str(cache_dir / "source-offload"),
            max_cache_slots=int(args.streaming_cache_slots),
            prefetch_workers=args.streaming_prefetch_workers,
            cache_headroom_gb=float(args.streaming_cache_headroom_gb),
            prefetch_min_available_gb=float(args.streaming_cache_headroom_gb),
            prefetch_lookahead=int(args.streaming_cache_slots) - 1,
            require_prefetched_residency=True,
            attn_implementation=args.attention_implementation,
            source_authentication=source)
        scope.callback(runner.shutdown)
        if runner.num_layers != num_layers:
            raise ResearchRefused(
                f"the source has {runner.num_layers} layers; the census names {num_layers}")
        runner.context.begin_source_initialization_audit()
        population = campaign._require_campaign_population(
            runner.model, profile, namespace.layer_stride)
        modules = dict(runner.model.named_modules())
        weights = {name: modules[name].weight for name in unit_names if name in modules}
        weights.update({member.qname: member.weight for member in population.members
                        if member.qname in set(unit_names)})
        carried = None
        if population.declared:
            if not census.get("expert_projection"):
                raise ResearchRefused(
                    "the census carries no producer expert projection to reuse "
                    "(PrismaQuant #183); refusing to price a projected population")
            # The census already asked the producer for exactly this scope;
            # binding it here is the check. Per-layer live views are re-checked
            # inside the visitor, exactly as the campaign's capture does.
            carried, _units = campaign._project_expert_population(
                population, weights=weights, menus={}, model_path=namespace.model,
                cache_dir=cache_dir, measured=set(),
                projection=census["expert_projection"], resource_check=guard,
                source_authentication=source)
        del weights, modules

        names_by_layer = {}
        for name in unit_names:
            names_by_layer.setdefault(runner.layer_index_for_qname(name), []).append(name)
        shapes = {name: list(census["unit_shapes"][name]) for name in unit_names}
        indexer = SampleIndexer(moments, total_samples=args.total_samples,
                                seqlen=int(census["seqlen"]), guard=guard)
        seen, telemetry = {}, []

        def visit(layer, forward_batch):
            names = names_by_layer.get(layer, [])
            members = [member for member in population.members if member.qname in names]
            live = refresh_packed_expert_projections(members, profile)
            if live:
                campaign._checked_projected_units(
                    carried["stacks"], weights={m.qname: m.weight for m in live},
                    model_path=namespace.model, source=carried["producer"]["source"],
                    measured={m.qname for m in live}, resource_check=guard,
                    source_authentication=source)
            del live
            expected = declared_shared_capture_groups(
                {name: shapes[name] for name in names}, profile)
            checked = indexer.wrapper(layer, forward_batch)
            _acts, _hessians, rows, _amax = campaign._collect_activations(
                runner.model, names, tokens, 0, runner.device,
                want_hessian=False, profile=runner.profile, forward_batch=checked,
                shared_packed_inputs=True, expected_shared_input_groups=expected,
                resource_check=guard, row_consumer=moments.consume)
            seen.update(rows)
            telemetry.append({"layer": layer, "units": len(names)})
            if runner.device.type == "cuda":
                torch.cuda.empty_cache()

        try:
            with quantum.owner():
                runner.visit_layer_batches(tokens, visit, start=quantum.frontier(),
                                           stop_layer=quantum.stop,
                                           boundary_consumer=quantum.boundary_consumer())
                targets = [name for name in unit_names
                           if start <= runner.layer_index_for_qname(name) < stop]
                if set(seen) != set(targets) or any(v <= 0 for v in seen.values()):
                    raise ResearchRefused(
                        "the research quantum did not observe every selected unit of its layers")
                campaign.census_token_counts(census, seen)
                records = moments.finish()
                verified = persist_split_records(
                    capture_root, census, census_path, records, seen, split_sha256, guard)
                fragment = quantum.complete(
                    witness=runner.context.source_selected_initialization_witness(
                        range(quantum.start, quantum.stop)),
                    verified=verified)
        except BaseException:
            if runner.device.type == "cuda":
                torch.cuda.empty_cache()
            raise
    print(json.dumps({"research_quantum": {
        "layers": [quantum.start, quantum.stop], "fragment": str(fragment),
        "units": len(verified), "forwards": indexer.global_index,
        "telemetry": telemetry}}), flush=True)
    return {"fragment": str(fragment), "units": len(verified)}


# -- join ------------------------------------------------------------------------

def mode_join(args, guard) -> dict:
    """Research metadata verification: ranges, witness merge, role-manifest union."""
    guard("research join startup")
    capture_root = Path(args.capture_root).resolve()
    census_path, census, namespace = load_census(args, capture_root)
    prep = chain.read_prep(capture_root)
    ranges = chain.require_layer_tiling(prep["ranges"])
    identity = prep["identity"]
    for start, stop in ranges:
        chain.require_owner_complete(prep, start, stop)
    fragments = [chain.read_fragment(capture_root, prep, start, stop)
                 for start, stop in ranges]
    depths = {fragment["num_layers"] for fragment in fragments}
    if len(depths) != 1:
        raise ResearchRefused("the quanta ran over sources of different depth")
    chain.require_layer_tiling(ranges, num_layers=depths.pop())
    chain.require_source_fingerprints(prep["source_fingerprints"], prep["source_root"],
                                      where="research split join")
    from prismaquant.streaming_model import merge_selected_initialization_witnesses
    from prismaquant import validate_source_initialization_contract
    merged = merge_selected_initialization_witnesses(
        [fragment["witness"] for fragment in fragments])
    if merged != validate_source_initialization_contract(identity["model_load_contract"]):
        raise ResearchRefused(
            "the quanta's merged initialization witness differs from the census contract")
    recorded = chain.recorded_source_digests(prep, fragments)
    verified = {}
    for fragment in fragments:
        repeated = sorted(set(verified) & set(fragment["units"]))
        if repeated:
            raise ResearchRefused(f"two quanta verified one unit: {repeated[:8]}")
        verified.update(fragment["units"])
    if set(verified) != set(identity["units"]):
        missing = sorted(set(identity["units"]) - set(verified))
        raise ResearchRefused(f"the quanta verified no role receipt for {missing[:8]}")
    # Role manifest union: every layer manifest is read back, its split stamp
    # held uniform, and every role file re-hashed against its receipt. Counts
    # are held to the census; no provider qualification is claimed here.
    split_sha256 = split_stamp(json.loads(split_manifest_path(capture_root).read_text()))
    counts = census["counts"]
    covered = {}
    for manifest_path in sorted((research_dir(capture_root) / "layers").glob("L*/manifest.json")):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != LAYER_MANIFEST_SCHEMA:
            raise ResearchRefused(f"{manifest_path}: not a {LAYER_MANIFEST_SCHEMA} manifest")
        if manifest["split_sha256"] != split_sha256:
            raise ResearchRefused(f"{manifest_path}: stamps another split")
        for qname, roles in manifest["units"].items():
            if qname in covered:
                raise ResearchRefused(f"{qname}: two layer manifests record one unit")
            total = 0
            for role in (FIT, HELDOUT):
                receipt = roles[role]
                path = research_dir(capture_root) / receipt["file"]
                if _sha256_file(path) != receipt["sha256"]:
                    raise ResearchRefused(f"{path}: role file bytes differ from the receipt")
                if path.stat().st_size != receipt["bytes"]:
                    raise ResearchRefused(f"{path}: role file length differs from the receipt")
                total += int(receipt["count"])
            if total != int(counts[qname]):
                raise ResearchRefused(
                    f"{qname}: role union {total} != census {counts[qname]}")
            covered[qname] = manifest_path.parent.name
    if set(covered) != set(identity["units"]):
        missing = sorted(set(identity["units"]) - set(covered))
        raise ResearchRefused(f"the layer manifests cover no record for {missing[:8]}")
    from prismaquant.cost_stage_checkpoint import (
        atomic_write_bytes, canonical_json_bytes, canonical_json_sha256)
    body = {"schema": RESEARCH_JOIN_SCHEMA, "prep_sha256": prep["prep_sha256"],
            "census_sha256": _sha256_file(census_path),
            "ranges": [list(pair) for pair in ranges],
            "witness_merged_sha256": _canonical_sha256(merged),
            "recorded_source_files": len(recorded),
            "role_units": len(covered), "split_sha256": split_sha256,
            "unit_layers": covered,
            "note": "research metadata verification only; no capture manifest is published"}
    document = {**body, "join_sha256": canonical_json_sha256(
        body, where="research split join")}
    atomic_write_bytes(join_document_path(capture_root),
                       canonical_json_bytes(document, where="research split join") + b"\n")
    print(json.dumps({"research_join": {
        "path": str(join_document_path(capture_root)), "units": len(covered),
        "ranges": [list(pair) for pair in ranges]}}), flush=True)
    return document


# -- preflight -------------------------------------------------------------------

def _real_metadata_preflight(args, guard) -> dict:
    """Small real-metadata reads with no GPU and no payload forward."""
    path = Path(args.calibration_census)
    census = json.loads(path.read_text())
    namespace = campaign_args(args, census)
    checked = campaign.load_calibration_census(path, args=namespace)
    selection = campaign.load_unit_selection(args.units)
    names = campaign.selected_capture_unit_names(
        selection, args=namespace, resolved=checked["anchor_groups"])
    missing = sorted(set(names) - set(checked["counts"]))
    if missing:
        raise ResearchRefused(f"the selection prices units outside the census: {missing[:8]}")
    ids, tokens, _corpus_text = load_draw(args, checked)
    require_split_geometry(args, tokens, checked)
    guard("research preflight metadata")
    return {"census": {"model": str(checked["model"]),
                       "units": len(checked["counts"]),
                       "groups": len(checked["anchor_groups"])},
            "units": {"selected": len(names)},
            "draw": {"samples": int(ids.shape[0]), "seqlen": int(ids.shape[1]),
                     "fit_stop": int(args.fit_stop)}}


def verified_roles_carry_counts(fragment) -> bool:
    units = fragment.get("units") or {}
    return bool(units) and all(
        isinstance(record.get(FIT), dict) and isinstance(record.get(HELDOUT), dict)
        and int(record[FIT]["count"]) > 0 and int(record[HELDOUT]["count"]) > 0
        and int(record[FIT]["count"]) + int(record[HELDOUT]["count"])
        == int(record["census_count"])
        for record in units.values())


def _toy_control_preflight(directory: Path, guard) -> dict:
    """The actual tiny one-layer GLM CPU control: census, prep, quantum, join."""
    import pytest
    tests = ROOT / "tests"
    if str(tests) not in sys.path:
        sys.path.insert(0, str(tests))
    from test_glm5_next_streamed_forward_parity import (
        _build_model, _tiny_config, _torch_only_causal_conv1d)
    from test_glm_campaign_streaming import write_original_layout_checkpoint

    torch.manual_seed(20261001)
    config = _tiny_config()
    text = config.text_config
    text.hidden_size = 64
    text.intermediate_size = 128
    text.num_hidden_layers = 1
    text.layer_types = ["linear_attention"]
    text.mlp_layer_types = ["dense"]
    text.indexer_types = ["full"]
    text.first_k_dense_replace = 1
    config.vision_config.out_hidden_size = 64
    toy_config = type(config).from_dict(config.to_dict())
    model = _build_model(toy_config).to(torch.bfloat16)

    source = directory / "source"
    write_original_layout_checkpoint(model, source)
    ids = torch.randint(2, 120, (4, 8),
                        generator=torch.Generator().manual_seed(20261001))
    from safetensors.torch import save_file
    save_file({"input_ids": ids}, str(directory / "tokens.safetensors"))
    corpus = directory / "corpus.txt"
    corpus.write_text("tiny GLM frozen research preflight draw\n")
    tokens = [ids[index:index + 1].contiguous() for index in range(ids.shape[0])]

    # The toy draw replaces the wikitext fetch for this preflight only; the
    # census, prep, quantum and join below are the real code paths.
    original_tokens = campaign._calibration_tokens
    campaign._calibration_tokens = lambda *_args: (tokens, corpus.read_text())
    kernels = pytest.MonkeyPatch()
    try:
        _torch_only_causal_conv1d.__wrapped__(kernels)
        census_path = directory / "census.json"
        common = ["--model", str(source), "--out", str(directory / "unused.pkl"),
                  "--menu-mode", "research", "--nsamples", "4", "--seqlen", "8",
                  "--max-act-rows", "7", "--attention-implementation", "eager",
                  "--streaming", "--streaming-cache-headroom-gb", "0"]
        if campaign.main([*common, "--cache-dir", str(directory / "census-cache"),
                          "--census-out", str(census_path)]) != 0:
            raise ResearchRefused("the toy census run failed")
        census = json.loads(census_path.read_text())
        # The toy draw's own receipt, computed from the toy artifacts: the
        # census a wikitext run writes carries the receipt; a plain toy
        # census does not, so the control stamps the draw it actually used.
        if "calibration_input" not in census:
            hi = max(int(value) for value in census["counts"].values())
            identity = th.calibration_identity(
                corpus.read_text(), tokens, fit_tokens=hi,
                model=str(census["model"]), seed=int(census["seed"]),
                nsamples=int(census["nsamples"]), seqlen=int(census["seqlen"]),
                split_role="calibration", source="indomain split preflight toy draw")
            provenance = {key: value for key, value in identity.items()
                          if key not in ("calibration_input",)}
            census["calibration_input"] = {
                "artifact_sha256": _sha256_file(directory / "tokens.safetensors"),
                "calibration_sha256": identity["fit_ids_sha256"],
                "dtype": str(ids.dtype), "shape": [int(v) for v in ids.shape],
                "provenance": provenance}
            _write_atomic_json(census_path, census)
        selection = {"schema": "prismaquant.tessera_campaign_units.v1",
                     "model": str(source), "layer_stride": 1,
                     "groups": [{"key": key, "members": members}
                                for key, members in sorted(census["anchor_groups"].items())]}
        selection_path = directory / "units.json"
        selection_path.write_text(json.dumps(selection))
        root = directory / "capture"
        storage = {"schema": "prismaquant.aura.boundary_storage.v2",
                   "capture_order": "layer_major",
                   "directory": str(directory / "boundaries"),
                   "max_resident_bytes": 64 << 20, "max_auxiliary_bytes": 1 << 20,
                   "max_artifact_bytes": 1 << 30, "prefetch_batches": 1}

        def toy_args(**extra):
            return argparse.Namespace(
                capture_root=str(root), calibration_census=str(census_path),
                units=str(selection_path),
                calibration_tokens=str(directory / "tokens.safetensors"),
                corpus_text=str(corpus), model=str(source),
                source_snapshot_root=None, total_samples=4, fit_stop=2,
                max_prefix_rows=4, max_act_rows=7,
                attention_implementation="eager", device="cpu",
                cache_dir=str(directory / "quantum-cache"),
                streaming_cache_slots=2, streaming_prefetch_workers=1,
                streaming_cache_headroom_gb=0.0, **extra)

        mode_prep(toy_args(capture_chain_ranges="0:1",
                           boundary_storage=json.dumps(storage)), guard)
        mode_quantum(toy_args(capture_layer_range="0:1"), guard)
        document = mode_join(toy_args(), guard)
        prep = chain.read_prep(root)
        for start, stop in chain.require_layer_tiling(prep["ranges"]):
            chain.require_owner_complete(prep, start, stop)
            fragment = chain.read_fragment(root, prep, start, stop)
            if not verified_roles_carry_counts(fragment):
                raise ResearchRefused("the toy fragment carries no role counts")
        manifests = sorted((research_dir(root) / "layers").glob("L*/manifest.json"))
        if len(manifests) != 1:
            raise ResearchRefused("the toy split wrote one layer manifest, and one only")
        manifest = json.loads(manifests[0].read_text())
        if set(manifest["units"]) != set(census["counts"]):
            raise ResearchRefused("the toy split did not cover the toy census scope")
        for qname, roles in manifest["units"].items():
            total = sum(int(roles[role]["count"]) for role in (FIT, HELDOUT))
            if total != int(census["counts"][qname]):
                raise ResearchRefused(
                    f"{qname}: toy fit+heldout {total} != census {census['counts'][qname]}")
            if roles[FIT]["count"] == 0 or roles[HELDOUT]["count"] == 0:
                raise ResearchRefused(f"{qname}: a toy role observed no rows")
        print(json.dumps({"research_preflight_toy": {
            "units": len(manifest["units"]),
            "join": str(join_document_path(root)),
            "boundary_forward": True}}), flush=True)
        return {"schema": document["schema"], "units": len(manifest["units"])}
    finally:
        campaign._calibration_tokens = original_tokens
        kernels.undo()


def mode_preflight(args, guard) -> dict:
    """Real metadata reads with no GPU, then the tiny CPU control."""
    guard("research preflight startup")
    result = {"metadata": None, "toy": None}
    if all((args.calibration_census, args.units, args.calibration_tokens,
            args.corpus_text)):
        result["metadata"] = _real_metadata_preflight(args, guard)
    else:
        print(json.dumps({"research_preflight_metadata":
                              "skipped: not all real inputs were given"}), flush=True)
    with tempfile.TemporaryDirectory(prefix="indomain-split-preflight-") as tmp:
        result["toy"] = _toy_control_preflight(Path(tmp), guard)
    print(json.dumps({"research_preflight": result}), flush=True)
    return result


# -- CLI -------------------------------------------------------------------------

MODES = ("prep", "quantum", "join", "preflight")
CAPTURE_INPUTS = ("--calibration-census", "--units", "--calibration-tokens", "--corpus-text")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--capture-root")
    parser.add_argument("--calibration-census")
    parser.add_argument("--units")
    parser.add_argument("--calibration-tokens")
    parser.add_argument("--corpus-text")
    parser.add_argument("--model", default=None,
                        help="defaults to the census's source model")
    parser.add_argument("--source-snapshot-root", default=None,
                        help="optional byte-identical source view; not an added gate")
    parser.add_argument("--capture-chain-ranges", default=None)
    parser.add_argument("--boundary-storage", default=None,
                        help="inline JSON or a path to the boundary-storage config")
    parser.add_argument("--capture-layer-range", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--streaming-cache-slots", type=int, default=2)
    parser.add_argument("--streaming-prefetch-workers", default=1)
    parser.add_argument("--streaming-cache-headroom-gb", type=float, default=0.0)
    parser.add_argument("--fit-stop", type=int, default=384)
    parser.add_argument("--total-samples", type=int, default=512)
    parser.add_argument("--max-prefix-rows", dest="max_prefix_rows", type=int, default=512)
    parser.add_argument("--max-act-rows", type=int, default=1,
                        help="the identity's scoring prefix, not the research rows")
    parser.add_argument("--attention-implementation", default="eager")
    parser.add_argument("--memfloor-gib", type=float, default=2.0)
    args = parser.parse_args(argv)
    guard = make_memory_guard(int(args.memfloor_gib * (1 << 30)))
    if args.mode == "preflight":
        mode_preflight(args, guard)
        return 0
    if not args.capture_root:
        parser.error(f"--mode {args.mode} needs --capture-root")
    missing = [flag for flag, value in zip(CAPTURE_INPUTS, (
        args.calibration_census, args.units, args.calibration_tokens,
        args.corpus_text)) if not value]
    if missing:
        parser.error(f"--mode {args.mode} needs {', '.join(missing)}")
    if args.mode == "prep":
        mode_prep(args, guard)
    elif args.mode == "quantum":
        if not args.capture_layer_range:
            parser.error("--mode quantum needs --capture-layer-range")
        mode_quantum(args, guard)
    else:
        mode_join(args, guard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
